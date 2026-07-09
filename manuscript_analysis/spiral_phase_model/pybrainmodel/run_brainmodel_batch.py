# %%
from vortex_model_gpu import VortexPhaseModel
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from vortex_model_gpu import VortexPhaseModel, optimize_radial_influence
from singularity_tracking import detect_singularities
from measure_spiral_rotation import measure_spiral_rotation
from phase_utils import compute_phase_gradient
from pathlib import Path
from matplotlib import pyplot as plt
import os


class VortexReconstructor:
    def __init__(self, phase_cube, mask, v_threshold=1.0, iterations=350, lr=0.1, device='cuda'):
        """
        Initialize once and keep reusable tensors resident on the target device.
        """
        self.device = device
        self.phase_cube = phase_cube  # (H, W, T)
        self.H, self.W, self.T = phase_cube.shape
        self.mask_cpu = mask
        self.v_threshold = v_threshold
        self.iterations = iterations
        self.lr = lr

        # Precompute the GPU grid once.
        print("Pre-computing GPU grids...")
        col = torch.arange(self.W, dtype=torch.float32, device=self.device)
        row = torch.arange(self.H, dtype=torch.float32, device=self.device)
        X_full, Y_full = torch.meshgrid(col, row, indexing='xy')

        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        # Keep static grid data resident on the device.
        self.static_grid_data = {
            'mask': mask_t,
            'X_m': X_full[mask_t],
            'Y_m': Y_full[mask_t]
        }
        print(
            f"Grid ready. Masked pixels: {self.static_grid_data['X_m'].shape[0]}")

    def process_frame(self, frame_idx):
        """
        Process one frame with only the required per-frame computations.
        """
        # Fetch the current phase map.
        phase_map = self.phase_cube[:, :, frame_idx]

        # Singularity detection remains on CPU.
        # The detection step remains on CPU.
        vPhaseX, vPhaseY = compute_phase_gradient(phase_map)
        centroids_pos, centroids_neg = detect_singularities(
            vPhaseX, vPhaseY, phase_map, v_threshold=self.v_threshold)

        # Return an empty result if no vortices were detected.
        if len(centroids_pos) + len(centroids_neg) == 0:
            return {
                'frame': frame_idx,
                'sigma_pos': [], 'sigma_neg': [],
                'R2': 0.0, 'model_phase': None
            }

        omega_pos, omega_neg = measure_spiral_rotation(
            centroids_pos, centroids_neg, phase_map)

        # Initialize the model with the precomputed static grid.
        # This only allocates a few learned parameters.
        model = VortexPhaseModel(
            self.H, self.W,
            centroids_pos, centroids_neg,
            omega_pos, omega_neg,
            precomputed_grid=self.static_grid_data,
            device=self.device
        )

        # Run optimization.
        sigmas_pos, sigmas_neg, cost_hist = optimize_radial_influence(
            model, phase_map, iterations=self.iterations, lr=self.lr, verbose=False
        )

        return {
            'frame': frame_idx,
            'centroids_pos': centroids_pos,
            'centroids_neg': centroids_neg,
            'sigmas_pos': sigmas_pos,
            'sigmas_neg': sigmas_neg,
            'omega_pos': omega_pos,
            'omega_neg': omega_neg,
            'cost_history': cost_hist,
            # 'reconstructed_phase': reconstructed_phase
        }

    def run_batch(self, start_frame, end_frame):
        results = []
        for t in tqdm(range(start_frame, end_frame), desc="Processing Frames"):
            res = self.process_frame(t)
            results.append(res)
        return results


# Example run on one retained detection bundle.
# Prepare data.
bundle_dir = Path(
    os.environ.get(
        "PSYCHEDELIC_SPIRAL_EXAMPLE_BUNDLE",
        "../../detect_results/DMT/DMT_DMT_S01_Atlas_s0_dtseries_nii_DMTDMT01L",
    )
)
phase_cube = np.load(bundle_dir / "phase_cube.npy")
mask = ~np.isnan(phase_cube[:, :, 0])
v_threshold = 0.9
iterations = 400
lr = 0.05

# Initialize the pipeline.
pipeline = VortexReconstructor(
    phase_cube, mask, v_threshold=v_threshold, iterations=iterations, lr=lr, device='cuda')

# Run all frames.
results = pipeline.run_batch(0, phase_cube.shape[2])

# # Analyze results.
# average_correlation = np.mean(
#     np.sqrt(1 - np.array([res['cost_history'][-1] for res in results])))
# print(f"Average R across frames: {average_correlation:.4f}")
# print(np.sqrt(1 - np.array([np.min(res['cost_history']) for res in results])))
# vis_frame = 0

# best_R_idx = np.argmin(results[vis_frame]['cost_history'])


# def reconstruct_single_frame(omega_pos, omega_neg, centroids_pos, centroids_neg,
#                              saved_sigma_pos, saved_sigma_neg, mask, device='cuda'):
#     """
#     Reconstruct one phase field from saved sigma parameters.
#     """
#     # Prepare base data.
#     H, W = mask.shape

#     # Initialize model.
#     # Sigma defaults are assigned inside the model.
#     model = VortexPhaseModel(
#         height=H, width=W,
#         centroids_pos=centroids_pos, centroids_neg=centroids_neg,
#         omega_pos=omega_pos, omega_neg=omega_neg,
#         mask=mask, device=device
#     )

#     # Inject saved sigma parameters.
#     # Convert values to tensors on the correct device.
#     with torch.no_grad():
#         if model.n_pos > 0:
#             model.sigma_pos.data = torch.tensor(
#                 saved_sigma_pos, dtype=torch.float32, device=device
#             )

#         if model.n_neg > 0:
#             model.sigma_neg.data = torch.tensor(
#                 saved_sigma_neg, dtype=torch.float32, device=device
#             )

#     # Generate phase with a forward pass.
#     with torch.no_grad():
#         reconstructed_phase = model().cpu().numpy()

#     return reconstructed_phase


# # Run reconstruction.
# recon_map = reconstruct_single_frame(
#     omega_pos=results[vis_frame]['omega_pos'],
#     omega_neg=results[vis_frame]['omega_neg'],
#     centroids_pos=results[vis_frame]['centroids_pos'],
#     centroids_neg=results[vis_frame]['centroids_neg'],
#     saved_sigma_pos=results[vis_frame]['sigmas_pos'][best_R_idx],
#     saved_sigma_neg=results[vis_frame]['sigmas_neg'][best_R_idx],
#     mask=mask
# )

# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(phase_cube[:, :, vis_frame], cmap='hsv')
# ax.set_title(f'Original Phase (Frame {vis_frame})', fontsize=14)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.invert_yaxis()
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(recon_map, cmap='hsv')
# ax.scatter(results[vis_frame]['centroids_pos'][:, 0], results[vis_frame]['centroids_pos'][:, 1],
#            edgecolors='white', facecolors='none', linewidths=2, s=80)
# ax.scatter(results[vis_frame]['centroids_neg'][:, 0], results[vis_frame]['centroids_neg'][:, 1],
#            edgecolors='black', facecolors='none', linewidths=2, s=80)
# ax.set_title(
#     f'Model Phase (Frame {vis_frame}), R2={1-results[vis_frame]["cost_history"][best_R_idx]:.4f}, R={np.sqrt(max(1-results[vis_frame]["cost_history"][best_R_idx], 0)):.4f}', fontsize=14)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.invert_yaxis()
# plt.show()
# %%
