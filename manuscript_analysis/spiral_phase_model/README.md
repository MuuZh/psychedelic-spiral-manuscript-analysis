# Spiral Phase-Field Model

This folder contains the spiral phase-field model and reconstruction scripts used for the manuscript's model section.

`pybrainmodel/` contains the vortex/spiral model implementation and batch fitting scripts copied from the manuscript analysis workspace. The main fitting code is in `run_brainmodel_batch.py`, `run_brainmodel_batch2.py`, and `vortex_model_gpu.py`, with helper functions for phase gradients, singularity detection, and spiral rotation measurement.

`phase_recon_v1.py` reconstructs phase cubes from saved vortex fitting results such as `*_vortex_results.pkl`.

These scripts depend on derived detection bundles, phase cubes, and model-fitting result files that are not distributed in this repository. Generated model outputs, pickle fitting results, `.npy`/`.npz` arrays, and figures are intentionally excluded.

This is an archival code section for transparency. It is not a one-command reproduction package, and the copied scripts have not been revalidated end-to-end after release cleanup.
