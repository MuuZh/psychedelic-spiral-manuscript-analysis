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
import pandas as pd
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



# Configuration
detect_results_root = Path(os.environ.get("PSYCHEDELIC_SPIRAL_DETECT_ROOT", "../../detect_results"))
output_root = Path(os.environ.get("PSYCHEDELIC_SPIRAL_MODEL_RESULTS_ROOT", "../../model_results"))
output_root.mkdir(parents=True, exist_ok=True)

v_threshold = 0.9
iterations = 400
lr = 0.05

# Get all drug directories
drug_dirs = [d for d in detect_results_root.iterdir() if d.is_dir()]

print(f"Found {len(drug_dirs)} drug types: {[d.name for d in drug_dirs]}")

# Process each drug
for drug_dir in tqdm(drug_dirs, desc="Processing drugs", position=0):
    drug_name = drug_dir.name
    print(f"\n{'='*60}")
    print(f"Processing drug: {drug_name}")
    print(f"{'='*60}")

    # Get all subject bundles for this drug
    bundle_dirs = sorted([d for d in drug_dir.iterdir() if d.is_dir()])
    print(f"Found {len(bundle_dirs)} bundles for {drug_name}")

    # Output file path
    output_file = output_root / f"{drug_name}_vortex_results.pkl"

    # Load existing data if file exists
    if output_file.exists():
        df_existing = pd.read_pickle(output_file)
        print(f"Loaded existing data: {len(df_existing)} rows")
    else:
        df_existing = pd.DataFrame()

    # Process each bundle
    for bundle_dir in tqdm(bundle_dirs, desc=f"Bundles ({drug_name})", position=1, leave=False):
        bundle_name = bundle_dir.name

        # Check if phase_cube.npy exists
        phase_cube_path = bundle_dir / "phase_cube.npy"
        if not phase_cube_path.exists():
            print(f"\nSkipping {bundle_name}: phase_cube.npy not found")
            continue

        # Parse bundle_name to extract metadata from the last part
        # Example: DMTDMT01L, LSDPCB04R, DMTDMT19L
        last_part = bundle_name.split('_')[-1]  # DMTDMT01L or LSDPCB04R

        # Hemisphere is always the last character
        hemisphere = last_part[-1]  # L or R

        # Remove hemisphere to get the rest: DMTDMT01, LSDPCB04
        code = last_part[:-1]

        # Extract drug (first 3 chars: DMT or LSD)
        drug = code[:3]

        # Extract group (next 3 chars: DMT, PCB, etc.)
        group = code[3:6]

        # Extract subject number (remaining digits)
        subject_num = int(code[6:])

        # Check if this bundle has already been processed
        if not df_existing.empty:
            already_processed = (
                (df_existing['group'] == group) &
                (df_existing['subject'] == subject_num) &
                (df_existing['hemisphere'] == hemisphere)
            ).any()

            if already_processed:
                print(f"\nSkipping {bundle_name}: already processed")
                continue

        # Load phase cube
        try:
            phase_cube = np.load(phase_cube_path)
            mask = ~np.isnan(phase_cube[:, :, 0])

            # Initialize reconstructor
            pipeline = VortexReconstructor(
                phase_cube, mask,
                v_threshold=v_threshold,
                iterations=iterations,
                lr=lr,
                device='cuda'
            )

            # Run batch processing
            results = pipeline.run_batch(0, phase_cube.shape[2])

            # Convert results to DataFrame rows
            bundle_rows = []
            for frame_result in results:
                row = {
                    'drug': drug,
                    'group': group,
                    'subject': subject_num,
                    'hemisphere': hemisphere,
                    'frame': frame_result['frame'],
                    'centroids_pos': frame_result.get('centroids_pos', []),
                    'centroids_neg': frame_result.get('centroids_neg', []),
                    'sigmas_pos': frame_result.get('sigmas_pos', []),
                    'sigmas_neg': frame_result.get('sigmas_neg', []),
                    'omega_pos': frame_result.get('omega_pos', []),
                    'omega_neg': frame_result.get('omega_neg', []),
                    'cost_history': frame_result.get('cost_history', []),
                    'n_vortex_pos': len(frame_result.get('centroids_pos', [])),
                    'n_vortex_neg': len(frame_result.get('centroids_neg', []))
                }
                bundle_rows.append(row)

            # Append to existing DataFrame and save immediately
            df_new = pd.DataFrame(bundle_rows)
            df_existing = pd.concat([df_existing, df_new], ignore_index=True)
            df_existing.to_pickle(output_file)

            print(f"\nSaved {bundle_name}: {len(bundle_rows)} frames (total: {len(df_existing)} rows)")

            # Clean up GPU memory
            del pipeline
            del phase_cube
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError processing {bundle_name}: {str(e)}")
            continue

    print(f"\nCompleted {drug_name}: {len(df_existing)} total rows saved to {output_file}")

print("\n" + "="*60)
print("All processing complete!")
print(f"Results saved to: {output_root.absolute()}")
print("="*60)
