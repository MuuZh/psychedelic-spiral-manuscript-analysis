# Dynamic Functional Regions Analysis

This package computes DFR boundaries from wrapped phase cubes using circular phase gradients:

```python
from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field
```

The boundary definition is `circular_gradient_magnitude > magnitude_threshold` inside the current Schaefer grid mask. The old `interaction` boundary outputs are not used.

## Run

From the project root:

```powershell
conda activate base
cd <toolbox_root>
python -m analysis.dfr_analysis.run_dfr --drug DMT --save-frame-qc
python -m analysis.dfr_analysis.run_dfr --drug LSD --save-frame-qc
```

Both drugs can also be run together:

```powershell
python -m analysis.dfr_analysis.run_dfr --drug both --save-frame-qc
```

Useful smoke-test option:

```powershell
python -m analysis.dfr_analysis.run_dfr --drug DMT --limit 2 --save-frame-qc
```

Progress bars are enabled by default for entry-level and frame-level work. Use
`--no-progress` for quiet batch logs.

The DFR summary for each phase cube is cached by phase-cube path, file metadata,
grid metadata, DFR parameters, and requested QC frames:

```powershell
python -m analysis.dfr_analysis.run_dfr --drug DMT --cache-dir analysis_outputs/dfr_analysis/cache
```

Use `--no-cache` to force recomputation. Reusing the same `--cache-dir` lets
separate output runs share the expensive circular-gradient and frame-labeling
work.

## Default Inputs

- DMT empirical: `analysis_outputs/phase_fc_batch_phase_corr_7networks`
- DMT reconstruction: `analysis_outputs/phase_fc_recon_7networks`
- LSD empirical: `analysis_outputs/phase_fc_batch_phase_corr_7networks_LSD`
- LSD reconstruction: `analysis_outputs/phase_fc_recon_7networks_LSD`

Each bundle is expected to provide `run_metadata.json` with a phase cube path, plus current `grid_labels.npy` and `parcel_metadata.csv` for Schaefer 400 / 7-network overlap.

## Outputs

Default output root:

```text
analysis_outputs/dfr_analysis
```

Main tables:

- `per_condition_subject_dfr.csv`
- `paired_dfr_stats.csv`
- `per_subject_dfr_drug_minus_pcb_delta.csv`
- `dfr_spiral_boundary_overlap.csv`
- `dfr_spiral_delta_correlations.csv`
- `dfr_network_boundary_overlap.csv`
- `dfr_network_boundary_distance.csv`
- `empirical_recon_dfr_delta_correlations.csv`
- `run_metadata.json`

Figures are written under `figures/`, with frame-level QC under `figures/frame_qc/`.

Statistical output columns include uncertainty wherever a metric is estimated
from repeated observations or subject pairs: `*_n`, `*_std`, `*_sem`,
`*_ci95_low`, and `*_ci95_high`. Correlation tables include Pearson/Spearman
coefficients with Fisher-transform 95% CIs, `r_squared` with CI bounds derived
from the Pearson interval, and regression `slope`, `slope_se`, and slope 95% CI.

Network overlap uses `--network-overlap-mode top-mean` by default. This ranks
the average DFR boundary occupancy map and selects the same number of pixels as
the mean per-frame DFR boundary count before computing precision, recall, Dice,
enrichment, and distance metrics. This avoids the invalid all-cortex behavior
from treating `avg_boundary > 0` as the subject-level DFR boundary. The alternate
mode is `--network-overlap-mode occupancy-threshold` with
`--network-dfr-occupancy-threshold`.

Spiral-center overlap is computed for empirical bundles with
`frame_index.parquet`, using weighted centroids when available. The output table
reports nearest DFR-boundary distance, k-pixel overlap fractions for k=0..3,
center counts per frame, and mean size/power when those columns are present.
