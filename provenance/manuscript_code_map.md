# Manuscript Code Map

This is a human-readable provenance summary, not a full execution log. It maps manuscript result families to the archived scripts in this repository and to the derived export families used by the figure/source-data workflow.

| Manuscript result family | Archived script location | Derived export family |
| --- | --- | --- |
| Spiral count, spiral size, pattern duration, PTDR, NGSC, and CAI-related all-metrics summaries | `manuscript_analysis/all_metrics/` | `02_exports/all_metrics_runner/` |
| Optical-flow validation and sigma/curl sensitivity analyses | `manuscript_analysis/all_metrics/run_sigma_window_all_metrics.py`, `manuscript_analysis/all_metrics/run_curl_window_all_metrics.py` | Supplementary sensitivity tables and retained all-metrics exports |
| GIPR spatial dispersion | `manuscript_analysis/added_analysis/GIPR/gipr.py` | `02_exports/gipr/` |
| Network-level spiral density and spiral-size effects | `manuscript_analysis/phase_fc_group/network_spiral_metrics.py` and related scripts | `02_exports/network_spiral_metrics/`, `02_exports/network_spiral_metrics_v2/` |
| Path entropy | `manuscript_analysis/path_entropy_paired.py` | `02_exports/path_entropy/` |
| MSD beta | `manuscript_analysis/pattern_msd_beta_group.py` | `02_exports/pattern_msd_beta/` |
| Phase-field reconstruction and model validation | `manuscript_analysis/phase_fc_group/recon_parcel_fc_correlations.py`, `manuscript_analysis/phase_fc_group/phase_recon_edge_weighted_wbfc_delta.py` | `02_exports/phase_ngsc_recon_corr/`, `02_exports/phase_recon_wbfc/`, `02_exports/phase_ngsc_model/` |
| Coherent spatiotemporal domain / DFR analyses | `manuscript_analysis/dfr_analysis/run_dfr.py` | `02_exports/dfr/` |
| Cortical alignment index and angular-distribution analyses | `manuscript_analysis/phase_gradient_alignment.py`, `export_scripts/export_exp017_cai_polar.py` | `02_exports/cai_model_delta/`, `02_exports/cai_polar/` |
| Raw functional-connectivity comparison summaries | retained raw-FC export scripts | `02_exports/raw_fc_hemisphere_first/` |
| Final figure generation | `figure_scripts/scripts/` | manuscript figure outputs generated outside this repository |

The corresponding raw fMRI data, large derived data, and final figure image files are not distributed in this repository.
