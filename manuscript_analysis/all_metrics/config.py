from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # results_prefix: str = "lsd_vs_pcb_run"
    # group_drug: str = "LSD"
    # group_pcb: str = "PCB"
    # combined_dir: Path = Path(
    #     "<derived_data>/combined_outputs/LSD")
    # detect_results_dir: Path = Path(
    #     "<detect_results>/LSD")
    # analytic_dir: Path = Path(
    #     "<derived_data>/analytic_cubes/LSD")
    # parcellation_config: Path = Path("configs/defaults.yaml")
    # reference_gmap: Path = Path("testdata/interpolated_gmap.npy")
    # reference_gmap_left: Path = Path("testdata/interpolated_gmap_left.npy")
    # reference_gmap_right: Path = Path("testdata/interpolated_gmap_right.npy")
    # subjects_manifest: Optional[Path] = None
    # output_root: Path = Path("analysis_outputs") / "all_metrics"
    # reuse_cache: bool = True
    # save_plots: bool = True
    results_prefix: str = "dmt_vs_pcb_run"
    group_drug: str = "DMT"
    group_pcb: str = "PCB"
    combined_dir: Path = Path(
        "<derived_data>/combined_outputs/DMT")
    detect_results_dir: Path = Path(
        "<detect_results>/DMT")
    analytic_dir: Path = Path(
        "<derived_data>/analytic_cubes/DMT")
    parcellation_config: Path = Path("configs/defaults.yaml")
    reference_gmap: Path = Path("testdata/interpolated_gmap.npy")
    reference_gmap_left: Optional[Path] = None
    reference_gmap_right: Optional[Path] = None
    subjects_manifest: Optional[Path] = None
    output_root: Path = Path("analysis_outputs") / "all_metrics"
    reuse_cache: bool = True
    save_plots: bool = True
    tr_seconds: float = 2.0
    min_frames_for_angular_velocity: int = 3
    min_duration_for_msd: int = 10
    csvd_method: str = "phase_gradient"
