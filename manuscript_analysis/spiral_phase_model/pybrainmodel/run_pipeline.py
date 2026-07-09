#!/usr/bin/env python3
"""Phase field reconstruction from vortex singularities.

Equivalent to MATLAB singularity_based_Phase_model2.m.

Usage:
    python run_pipeline.py                           # default: matlab_frame100.mat
    python run_pipeline.py --input my_data.mat       # custom input
    python run_pipeline.py --iterations 500 --lr 0.5 # tune optimisation
    python run_pipeline.py --device cpu               # force CPU
"""
# %%
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path
from phase_utils import compute_phase_gradient
from singularity_tracking import detect_singularities
from measure_spiral_rotation import measure_spiral_rotation
from vortex_model_gpu import VortexPhaseModel, optimize_radial_influence
# %%
v_threshold = 0.9
iterations = 500
lr = 0.05


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
# ---- Load data ----
bundle_dir = Path(
    os.environ.get(
        "PSYCHEDELIC_SPIRAL_EXAMPLE_BUNDLE",
        "../../detect_results/LSD/LSD_PCB_S01_Atlas_dtseries_nii_LSDPCB01R",
    )
)
test_frames = [149, 100, 24]
phase_cube = np.load(bundle_dir / "phase_cube.npy")
phase_map = phase_cube[:, :, test_frames[0]]
H, W = phase_map.shape
print(f"Phase map shape: {H} x {W}")

# ---- Optional mask ----
mask = ~np.isnan(phase_map)

mask = mask[:H, :W]

# ---- Compute phase gradient ----
vPhaseX, vPhaseY = compute_phase_gradient(phase_map)
print(f"Detected {np.sum(np.isfinite(vPhaseX))} valid gradient points")

# ---- Detect vortex singularities ----
centroids_pos, centroids_neg = detect_singularities(
    vPhaseX, vPhaseY, phase_map, v_threshold=v_threshold)
n_pos = centroids_pos.shape[0]
n_neg = centroids_neg.shape[0]
print(f"Detected vortices: {n_pos} positive (ACW), {n_neg} negative (CW)")

if n_pos + n_neg == 0:
    print("No vortices detected. Exiting.")


# ---- Measure spiral rotation angles ----
omega_pos, omega_neg = measure_spiral_rotation(
    centroids_pos, centroids_neg, phase_map)
# print(f"Omega pos: {omega_pos}")
# print(f"Omega neg: {omega_neg}")

# ---- Build GPU model and optimise ----
model = VortexPhaseModel(
    height=H, width=W,
    centroids_pos=centroids_pos, centroids_neg=centroids_neg,
    omega_pos=omega_pos, omega_neg=omega_neg,
    mask=mask, device=device,
)
# plt.imshow(model.mask.cpu().numpy(), cmap='gray')
# plt.show()

print(f"\nOptimising sigma ({iterations} iterations, lr={lr}) ...")
sigma_pos_history, sigma_neg_history, cost_history = optimize_radial_influence(
    model, phase_map,
    iterations=iterations, lr=lr,
)
print(
    f"\nFinal R2 = {1 - cost_history[-1]:.6f}  (r = {np.sqrt(max(1 - cost_history[-1], 0)):.6f})")
best_cost_index = np.argmin(cost_history)
best_r2 = 1 - cost_history[best_cost_index]
best_sigma_pos = sigma_pos_history[best_cost_index]
best_sigma_neg = sigma_neg_history[best_cost_index]
print(
    f"Best R2 during optimisation: {1 - best_r2:.6f}  (r = {np.sqrt(max(best_r2, 0)):.6f})")
# ---- Generate final model phase ----
# with torch.no_grad():
#     Model_Phase = model().cpu().numpy()
# ---- Update model with best sigma values ----
with torch.no_grad():
    if model.n_pos > 0:
        # ensure correct dtype and device
        model.sigma_pos.data = torch.tensor(
            best_sigma_pos, dtype=torch.float32, device=device)

    if model.n_neg > 0:
        model.sigma_neg.data = torch.tensor(
            best_sigma_neg, dtype=torch.float32, device=device)

# 3. Generate final Model Phase (this will be the one corresponding to the best cost)
with torch.no_grad():
    Model_Phase = model().cpu().numpy()

# ---- Plot results ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Real phase map
ax = axes[0]
im = ax.imshow(phase_map, cmap='hsv', origin='upper', vmin=-np.pi, vmax=np.pi)
ax.scatter(centroids_pos[:, 0], centroids_pos[:, 1],
           edgecolors='white', facecolors='none', linewidths=2, s=80, label='Positive')
ax.scatter(centroids_neg[:, 0], centroids_neg[:, 1],
           edgecolors='black', facecolors='none', linewidths=2, s=80, label='Negative')
ax.set_title('Real Phase Map', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.invert_yaxis()
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Phase (rad)')
cbar.set_ticks([-np.pi + 0.1, 0, np.pi - 0.1])
cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])

# Model phase map
ax = axes[1]
im = ax.imshow(Model_Phase, cmap='hsv',
               origin='upper', vmin=-np.pi, vmax=np.pi)
ax.scatter(centroids_pos[:, 0], centroids_pos[:, 1],
           edgecolors='white', facecolors='none', linewidths=2, s=80)
ax.scatter(centroids_neg[:, 0], centroids_neg[:, 1],
           edgecolors='black', facecolors='none', linewidths=2, s=80)
ax.set_title(
    f'Model Phase (R2={best_r2:.4f}), R={np.sqrt(max(best_r2, 0)):.4f}', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.invert_yaxis()
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_aspect('equal')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Phase (rad)')
cbar.set_ticks([-np.pi + 0.1, 0, np.pi - 0.1])
cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])

# Cost convergence
ax = axes[2]
ax.plot(range(1, len(cost_history) + 1), cost_history, '-o', markersize=2)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Cost (1 - R2)', fontsize=14)
ax.set_title('Optimisation Convergence', fontsize=14)
ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('reconstruction_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: reconstruction_result.png")
# %%

# cost_history
# %%
