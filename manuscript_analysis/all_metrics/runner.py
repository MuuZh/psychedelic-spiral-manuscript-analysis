from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd

from .config import Config
from .utils import plot_abs_effect_size_bars, setup_logging, set_output_naming, write_table
from .loaders import load_bundles
from .angle_cos import run_weighted_mean_cos2_alignment
from .csvd import run_csvd
from .curl import run_curl_spatial
from .vortex import run_vortex_occupancy
from .ngsc import run_ngsc
from .patterns import run_pattern_stats
from .pattern_dynamics import run_pattern_dynamics
from .boundary import run_boundary_regions


def _run_stage(name: str, fn, *args) -> None:
    logging.info("Start stage: %s", name)
    fn(*args)
    logging.info("Finished stage: %s", name)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="All metrics runner (modular).")
    parser.add_argument("--prefix", dest="results_prefix", default=None)
    parser.add_argument("--drug-label", dest="group_drug", default=None)
    parser.add_argument("--pcb-label", dest="group_pcb", default=None)
    parser.add_argument("--combined-dir", type=Path, default=None)
    parser.add_argument("--detect-dir", type=Path, default=None)
    parser.add_argument("--reference-gmap", type=Path, default=None)
    parser.add_argument("--reference-gmap-left", type=Path, default=None)
    parser.add_argument("--reference-gmap-right", type=Path, default=None)
    parser.add_argument("--analytic-dir", type=Path, default=None)
    parser.add_argument("--parcellation-config", type=Path, default=None)
    parser.add_argument("--tr-seconds", type=float, default=None)
    parser.add_argument("--csvd-method", choices=["phase_gradient", "optical_flow"], default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.results_prefix:
        cfg.results_prefix = args.results_prefix
        cfg.group_drug = args.results_prefix
    if args.group_drug:
        cfg.group_drug = args.group_drug
    if args.group_pcb:
        cfg.group_pcb = args.group_pcb
    if args.combined_dir:
        cfg.combined_dir = args.combined_dir
    if args.detect_dir:
        cfg.detect_results_dir = args.detect_dir
    if args.reference_gmap:
        cfg.reference_gmap = args.reference_gmap
    if args.reference_gmap_left:
        cfg.reference_gmap_left = args.reference_gmap_left
    if args.reference_gmap_right:
        cfg.reference_gmap_right = args.reference_gmap_right
    if args.analytic_dir:
        cfg.analytic_dir = args.analytic_dir
    if args.parcellation_config:
        cfg.parcellation_config = args.parcellation_config
    if args.tr_seconds:
        cfg.tr_seconds = args.tr_seconds
    if args.csvd_method:
        cfg.csvd_method = args.csvd_method
    if args.no_cache:
        cfg.reuse_cache = False
    if args.no_plots:
        cfg.save_plots = False
    return cfg


def main() -> None:
    cfg = parse_args()
    out_root = cfg.output_root / cfg.results_prefix
    set_output_naming(cfg.results_prefix, datetime.now().strftime("%Y%m%d"))
    setup_logging(out_root)
    summary: List[Dict] = []

    bundles = load_bundles(cfg)
    if bundles.empty:
        logging.warning("No bundles found; exiting.")
        return

    logging.info("Loaded bundles: %d", len(bundles))
    _run_stage("boundary_regions", run_boundary_regions, cfg, bundles, summary)
    _run_stage("pattern_stats", run_pattern_stats, cfg, summary)
    _run_stage("pattern_dynamics", run_pattern_dynamics, cfg, summary)
    _run_stage("csvd", run_csvd, cfg, summary)
    _run_stage("weighted_mean_cos2_alignment", run_weighted_mean_cos2_alignment, cfg, bundles, summary)
    _run_stage("curl_spatial", run_curl_spatial, cfg, bundles, summary)
    _run_stage("vortex_occupancy", run_vortex_occupancy, cfg, bundles, summary)
    _run_stage("ngsc", run_ngsc, cfg, bundles, summary)

    summary_df = pd.DataFrame(summary)
    sum_dir = out_root / "summary"
    sum_dir.mkdir(parents=True, exist_ok=True)
    write_table(summary_df, sum_dir / "all_metrics_summary.csv")
    plot_abs_effect_size_bars(
        summary_df,
        sum_dir / "abs_cohens_dz_by_metric.png",
        title="Absolute Cohen's dz by metric",
        save=cfg.save_plots,
    )
    logging.info("Done. Summary rows: %d", len(summary_df))


if __name__ == "__main__":
    main()
