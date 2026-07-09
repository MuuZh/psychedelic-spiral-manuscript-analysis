# Manuscript Code Map

This is a human-readable provenance summary, not a full execution log. It maps manuscript result families to the archived scripts in this repository and to the derived export families used by the figure/source-data workflow.

| Manuscript result family | Archived script location | Derived export family |
| --- | --- | --- |
| Spiral count, spiral size, pattern duration, PTDR, NGSC, and CAI-related all-metrics summaries | `manuscript_analysis/all_metrics/` | `02_exports/all_metrics_runner/` |
| Sigma sensitivity analyses | `manuscript_analysis/upstream_generation/run_detection_batch_sigma_window.py`, `manuscript_analysis/all_metrics/run_sigma_window_all_metrics.py` | Supplementary sigma-window sensitivity tables and retained all-metrics exports |
| Curl threshold and optical-flow sensitivity analyses | `manuscript_analysis/upstream_generation/run_detection_batch_curl_window.py`, `manuscript_analysis/upstream_generation/compare_curl1_optflow_window_overlap.py`, `manuscript_analysis/upstream_generation/spiral_optflow_focus/`, `manuscript_analysis/all_metrics/run_curl_window_all_metrics.py` | Supplementary curl-window and optical-flow comparison tables |
| GIPR spatial dispersion | `manuscript_analysis/added_analysis/GIPR/gipr.py` | `02_exports/gipr/` |
| Network-level spiral density and spiral-size effects | `manuscript_analysis/phase_fc_group/network_spiral_metrics.py` and related scripts | `02_exports/network_spiral_metrics/`, `02_exports/network_spiral_metrics_v2/` |
| Path entropy | `manuscript_analysis/path_entropy_paired.py` | `02_exports/path_entropy/` |
| MSD beta | `manuscript_analysis/upstream_generation/pattern_msd_beta.py`, `manuscript_analysis/pattern_msd_beta_group.py` | `02_exports/pattern_msd_beta/` |
| GCOR comparison summaries | `manuscript_analysis/upstream_generation/gcor_batch.py`, `manuscript_analysis/all_metrics/run_gcor_all_metrics.py`, `manuscript_analysis/phase_fc_group/pattern_gcor_correlations.py`, `manuscript_analysis/phase_fc_group/pattern_metric_gcor_correlations.py` | retained GCOR comparison exports |
| Model fitting / spiral phase-field construction | `manuscript_analysis/spiral_phase_model/pybrainmodel/run_brainmodel_batch.py`, `manuscript_analysis/spiral_phase_model/pybrainmodel/run_brainmodel_batch2.py`, `manuscript_analysis/spiral_phase_model/pybrainmodel/vortex_model_gpu.py` | model-fitting result files generated outside this release |
| Model-derived reconstructed phase cubes | `manuscript_analysis/spiral_phase_model/phase_recon_v1.py` | reconstructed phase-cube intermediates generated outside this release |
| Downstream spiral model validation | `manuscript_analysis/phase_fc_group/recon_parcel_fc_correlations.py`, `manuscript_analysis/phase_fc_group/phase_recon_edge_weighted_wbfc_delta.py`, `manuscript_analysis/dfr_analysis/run_dfr.py`, related model-validation export and figure scripts | `02_exports/phase_ngsc_recon_corr/`, `02_exports/phase_recon_wbfc/`, `02_exports/phase_ngsc_model/`, `02_exports/dfr/` |
| Empirical and reconstructed phase-FC intermediate generation | `manuscript_analysis/upstream_generation/phase_fc_batch.py`, `manuscript_analysis/upstream_generation/phase_fc_recon_batch.py`, `manuscript_analysis/upstream_generation/phase_fc_single_subject.py`, `manuscript_analysis/upstream_generation/phase_fc_build_atlas_metadata.py` | phase-FC intermediate outputs consumed by model-validation, wFC/bFC, DFR, and phase-reconstruction analyses |
| Coherent spatiotemporal domain / DFR analyses | `manuscript_analysis/dfr_analysis/run_dfr.py` | `02_exports/dfr/` |
| Cortical alignment index and angular-distribution analyses | `manuscript_analysis/phase_gradient_alignment.py`, `export_scripts/export_exp017_cai_polar.py` | `02_exports/cai_model_delta/`, `02_exports/cai_polar/` |
| Raw functional-connectivity comparison summaries | `manuscript_analysis/raw_fc/run_fc_hemisphere_pipeline.py`, `manuscript_analysis/raw_fc/analyze_fc_hemisphere_first.py`, related `manuscript_analysis/raw_fc/` scripts, `export_scripts/export_exp024_retained_raw_fc.py` | `02_exports/raw_fc_hemisphere_first/` |
| Final figure generation | `figure_scripts/scripts/` | manuscript figure outputs generated outside this repository |

The corresponding raw fMRI data, large derived data, model fitting outputs, reconstructed phase cubes, and final figure image files are not distributed in this repository. General-purpose preprocessing, phase extraction, vorticity-based spiral detection, tracking, and detection-bundle generation code is intentionally not duplicated here because it is covered by the separate Brain-Vortex-Detection-PY repository at `github.com/MuuZh/Brain-Vortex-Detection-PY`.

Static audit notes:

- Intermediate-generation scripts referenced by retained downstream code are represented for phase-FC, reconstructed phase-FC, MSD beta, GCOR, sigma-window sensitivity, curl-window sensitivity, curl-vs-optical-flow overlap, spiral optical-flow focus analysis, spiral phase-model fitting/reconstruction, and raw-FC hemisphere-first comparisons.
- `manuscript_analysis/upstream_generation/run_detection_batch_sigma_window.py` and `manuscript_analysis/upstream_generation/run_detection_batch_curl_window.py` retain imports from `run_full_detection_bundle.py`. That helper belongs to the broader Brain-Vortex-Detection-PY preprocessing/detection pipeline and is intentionally not copied into this manuscript-specific release.
- Raw-FC source scripts found in the local FC workspace are included under `manuscript_analysis/raw_fc/`; no requested raw-FC script remains unavailable in this release pass.
