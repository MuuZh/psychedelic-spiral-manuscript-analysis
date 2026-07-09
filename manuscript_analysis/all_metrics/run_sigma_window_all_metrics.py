from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .angle_cos import run_angle_diff_abs_cos
from .boundary import run_boundary_regions
from .config import Config
from .csvd import run_csvd
from .curl import run_curl_spatial
from .ngsc import run_ngsc
from .patterns import run_pattern_stats
from .pattern_dynamics import run_pattern_dynamics
from .utils import plot_abs_effect_size_bars, setup_logging, write_table
from .vortex import run_vortex_occupancy
from .utils import set_output_naming

matplotlib.use("Agg")


GROUP_SUB_RE = re.compile(r"^[A-Za-z]+_(DMT|PCB)_S(\d+)", re.IGNORECASE)


def _run_stage(name: str, fn, *args) -> None:
    logging.info("Start stage: %s", name)
    fn(*args)
    logging.info("Finished stage: %s", name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all_metrics-style analysis for each sigma window batch and compare sigma segments.",
    )
    parser.add_argument(
        "--sigma-root",
        type=Path,
        default=Path("output/sigma_window_batch"),
        help="Root directory created by run_detection_batch_sigma_window.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs") / "all_metrics_sigma_window",
    )
    parser.add_argument("--results-prefix", type=str, default="dmt_sigma_window")
    parser.add_argument("--drug-label", type=str, default="DMT")
    parser.add_argument("--pcb-label", type=str, default="PCB")
    parser.add_argument("--reference-gmap", type=Path, default=Path("testdata/interpolated_gmap.npy"))
    parser.add_argument("--reference-gmap-left", type=Path, default=None)
    parser.add_argument("--reference-gmap-right", type=Path, default=None)
    parser.add_argument("--parcellation-config", type=Path, default=Path("configs/defaults.yaml"))
    parser.add_argument("--analytic-dir", type=Path, default=Path("analytic_cubes/DMT"))
    parser.add_argument("--csvd-method", choices=["phase_gradient", "optical_flow"], default="phase_gradient")
    parser.add_argument("--tr-seconds", type=float, default=2.0)
    parser.add_argument("--min-duration-for-msd", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _discover_sigmas(sigma_root: Path) -> List[str]:
    sigmas = set()
    for subject_dir in sigma_root.iterdir():
        if not subject_dir.is_dir():
            continue
        for sigma_dir in subject_dir.iterdir():
            if sigma_dir.is_dir() and sigma_dir.name.startswith("sigma_"):
                sigmas.add(sigma_dir.name)
    return sorted(sigmas)


def _parse_group_subid(subject_dir_name: str, cifti_file: Optional[str], drug: str, pcb: str) -> Tuple[Optional[str], Optional[str]]:
    m = GROUP_SUB_RE.search(subject_dir_name)
    if m:
        group_token, subid = m.groups()
    else:
        group_token, subid = None, None

    if (group_token is None or subid is None) and cifti_file:
        m2 = GROUP_SUB_RE.search(Path(cifti_file).stem.replace(".", "_"))
        if m2:
            group_token, subid = m2.groups()

    if group_token is None or subid is None:
        return None, None

    group_token = group_token.upper()
    if group_token == "DMT":
        group = drug
    elif group_token == "PCB":
        group = pcb
    else:
        return None, None
    return group, str(int(subid))


def _parse_hemi(bundle_name: str, metadata: Dict) -> Optional[str]:
    extra = metadata.get("extra_metadata", {})
    hemi = extra.get("hemisphere") or metadata.get("hemisphere")
    if hemi in {"left", "right"}:
        return hemi
    low = bundle_name.lower()
    if low.endswith("_left") or low.endswith("left"):
        return "left"
    if low.endswith("_right") or low.endswith("right"):
        return "right"
    return None


def _collect_bundle_rows(sigma_root: Path, sigma_tag: str, drug: str, pcb: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for subject_dir in sorted(sigma_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        sigma_dir = subject_dir / sigma_tag
        if not sigma_dir.exists():
            continue
        for bundle_dir in sorted(p for p in sigma_dir.iterdir() if p.is_dir()):
            meta_path = bundle_dir / "metadata.json"
            phase_cube = bundle_dir / "phase_cube.npy"
            if not (meta_path.exists() and phase_cube.exists()):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            group, subid = _parse_group_subid(
                subject_dir.name, meta.get("cifti_file"), drug=drug, pcb=pcb)
            hemi = _parse_hemi(bundle_dir.name, meta)
            if group is None or subid is None or hemi is None:
                continue
            rows.append(
                {
                    "group": group,
                    "subid": subid,
                    "hemisphere": hemi,
                    "bundle_dir": bundle_dir,
                    "phase_cube": phase_cube,
                    "vortex_coords": bundle_dir / "vortex_coords.json",
                    "vortex_occupancy": bundle_dir / "vortex_occupancy.npy",
                }
            )
    return pd.DataFrame(rows)


def _combine_for_sigma(bundle_df: pd.DataFrame, out_dir: Path) -> Path:
    combined_dir = out_dir / "_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    patterns_all: List[pd.DataFrame] = []
    frames_all: List[pd.DataFrame] = []
    coords_all: List[pd.DataFrame] = []
    coord_offset = 0

    for row in bundle_df.itertuples(index=False):
        bdir = Path(row.bundle_dir)
        patt_path = bdir / "patterns.parquet"
        frame_path = bdir / "frame_index.parquet"
        coord_path = bdir / "coords.feather"
        if not (patt_path.exists() and frame_path.exists() and coord_path.exists()):
            continue
        try:
            patt_df = pd.read_parquet(patt_path)
            frame_df = pd.read_parquet(frame_path)
            coords_df = pd.read_feather(coord_path)
        except Exception:
            continue

        patt_df["group"] = row.group
        patt_df["subid"] = row.subid
        patt_df["hemisphere"] = row.hemisphere
        patt_df["bundle_dir"] = str(bdir)
        patterns_all.append(patt_df)

        frame_df["coord_start"] = frame_df["coord_start"] + coord_offset
        frame_df["coord_end"] = frame_df["coord_end"] + coord_offset
        frame_df["group"] = row.group
        frame_df["subid"] = row.subid
        frame_df["hemisphere"] = row.hemisphere
        frame_df["bundle_dir"] = str(bdir)
        frames_all.append(frame_df)

        coords_all.append(coords_df)
        coord_offset += len(coords_df)

    if patterns_all:
        pd.concat(patterns_all, ignore_index=True).to_parquet(
            combined_dir / "combined_patterns.parquet", index=False
        )
    if frames_all:
        pd.concat(frames_all, ignore_index=True).to_parquet(
            combined_dir / "combined_frame_index.parquet", index=False
        )
    if coords_all:
        pd.concat(coords_all, ignore_index=True).to_feather(
            combined_dir / "combined_coords.feather"
        )
    return combined_dir


def _run_single_sigma(args: argparse.Namespace, sigma_tag: str, sigma_df: pd.DataFrame) -> Path:
    sigma_out = args.output_root / args.results_prefix / sigma_tag
    set_output_naming(args.results_prefix, datetime.now().strftime("%Y%m%d"))
    setup_logging(sigma_out)
    logging.info("Running sigma=%s with %d bundles", sigma_tag, len(sigma_df))

    combined_dir = _combine_for_sigma(sigma_df, sigma_out)

    cfg = Config()
    cfg.results_prefix = "."
    cfg.group_drug = args.drug_label
    cfg.group_pcb = args.pcb_label
    cfg.detect_results_dir = args.sigma_root
    cfg.combined_dir = combined_dir
    cfg.analytic_dir = args.analytic_dir
    cfg.parcellation_config = args.parcellation_config
    cfg.reference_gmap = args.reference_gmap
    cfg.reference_gmap_left = args.reference_gmap_left
    cfg.reference_gmap_right = args.reference_gmap_right
    cfg.output_root = sigma_out
    cfg.reuse_cache = not args.no_cache
    cfg.save_plots = not args.no_plots
    cfg.tr_seconds = args.tr_seconds
    cfg.min_duration_for_msd = args.min_duration_for_msd
    cfg.csvd_method = args.csvd_method

    summary: List[Dict] = []
    _run_stage("boundary_regions", run_boundary_regions, cfg, sigma_df, summary)
    _run_stage("pattern_stats", run_pattern_stats, cfg, summary)
    _run_stage("pattern_dynamics", run_pattern_dynamics, cfg, summary)
    _run_stage("csvd", run_csvd, cfg, summary)
    _run_stage("angle_diff_abs_cos", run_angle_diff_abs_cos, cfg, sigma_df, summary)
    _run_stage("curl_spatial", run_curl_spatial, cfg, sigma_df, summary)
    _run_stage("vortex_occupancy", run_vortex_occupancy, cfg, sigma_df, summary)
    _run_stage("ngsc", run_ngsc, cfg, sigma_df, summary)

    summary_df = pd.DataFrame(summary)
    write_table(summary_df, sigma_out / "summary" / "all_metrics_summary.csv")
    plot_abs_effect_size_bars(
        summary_df,
        sigma_out / "summary" / "abs_cohens_dz_by_metric.png",
        title=f"Absolute Cohen's dz by metric ({sigma_tag})",
        save=cfg.save_plots,
    )
    logging.info("Sigma %s done. Summary rows=%d", sigma_tag, len(summary_df))
    return sigma_out


def _build_group_mean_table(sigma_out_dirs: Dict[str, Path], drug: str, pcb: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for sigma_tag, sigma_out in sigma_out_dirs.items():
        for section_dir in sigma_out.iterdir():
            if not section_dir.is_dir() or section_dir.name.startswith("_"):
                continue
            gs = section_dir / "group_summary.csv"
            if not gs.exists():
                continue
            try:
                gdf = pd.read_csv(gs)
            except Exception:
                continue
            gdf = gdf[gdf["group"].isin({drug, pcb})].copy()
            if gdf.empty:
                continue
            for (hemi, metric), sub in gdf.groupby(["hemisphere", "metric"], dropna=False):
                drug_row = sub[sub["group"] == drug]
                pcb_row = sub[sub["group"] == pcb]
                if drug_row.empty or pcb_row.empty:
                    continue
                drug_mean = float(drug_row["mean"].iloc[0])
                pcb_mean = float(pcb_row["mean"].iloc[0])
                rows.append(
                    {
                        "sigma": sigma_tag,
                        "section": section_dir.name,
                        "hemisphere": hemi,
                        "metric": metric,
                        "mean_drug": drug_mean,
                        "mean_pcb": pcb_mean,
                        "mean_diff_drug_minus_pcb": drug_mean - pcb_mean,
                    }
                )
    return pd.DataFrame(rows)


def _plot_pvalue_heatmap(p_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if p_df.empty:
        return
    plot_df = p_df.copy()
    plot_df["metric_name"] = plot_df["section"] + ":" + plot_df["metric"]
    plot_df["neg_log10_p"] = -np.log10(plot_df["p"].clip(lower=1e-300))
    hm = plot_df.pivot_table(
        index="metric_name",
        columns="sigma",
        values="neg_log10_p",
        aggfunc="min",
    )
    if hm.empty:
        return
    fig_h = max(6, 0.32 * len(hm.index))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.heatmap(hm, cmap="mako", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Sigma segment")
    ax.set_ylabel("Metric")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _aggregate_sigma_comparison(root_out: Path, sigma_out_dirs: Dict[str, Path], drug: str, pcb: str) -> None:
    p_rows: List[pd.DataFrame] = []
    for sigma_tag, sigma_out in sigma_out_dirs.items():
        summary_csv = sigma_out / "summary" / "all_metrics_summary.csv"
        if not summary_csv.exists():
            continue
        sdf = pd.read_csv(summary_csv)
        if sdf.empty or "p" not in sdf.columns:
            continue
        sdf["sigma"] = sigma_tag
        p_rows.append(sdf)

    if not p_rows:
        return
    all_p = pd.concat(p_rows, ignore_index=True)
    all_p = all_p[pd.to_numeric(all_p["p"], errors="coerce").notna()].copy()
    write_table(all_p, root_out / "sigma_comparison_pvalues_long.csv")

    group_cols = ["section", "metric", "hemisphere", "comparison"]
    rank_rows: List[Dict] = []
    for keys, sub in all_p.groupby(group_cols, dropna=False):
        sub = sub.sort_values(by="p", ascending=True)
        best = sub.iloc[0]
        row = {
            "section": keys[0],
            "metric": keys[1],
            "hemisphere": keys[2],
            "comparison": keys[3],
            "best_sigma": best["sigma"],
            "best_p": best["p"],
            "best_t": best.get("t", math.nan),
        }
        rank_rows.append(row)
    ranking_df = pd.DataFrame(rank_rows).sort_values(by="best_p", ascending=True)
    write_table(ranking_df, root_out / "sigma_best_by_metric.csv")

    means_df = _build_group_mean_table(sigma_out_dirs, drug=drug, pcb=pcb)
    if not means_df.empty:
        write_table(means_df, root_out / "sigma_group_means_long.csv")

    for comparison in sorted(all_p["comparison"].dropna().unique()):
        for hemi in sorted(all_p["hemisphere"].dropna().unique()):
            sub = all_p[(all_p["comparison"] == comparison) & (all_p["hemisphere"] == hemi)]
            _plot_pvalue_heatmap(
                sub,
                root_out / "plots" / f"heatmap_neglog10p_{comparison}_{hemi}.png",
                title=f"-log10(p), {comparison}, {hemi}",
            )


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sigmas = _discover_sigmas(args.sigma_root)
    if not sigmas:
        raise SystemExit(f"No sigma_* folders found in: {args.sigma_root}")

    sigma_out_dirs: Dict[str, Path] = {}
    for sigma_tag in sigmas:
        sigma_df = _collect_bundle_rows(
            args.sigma_root, sigma_tag, drug=args.drug_label, pcb=args.pcb_label
        )
        if sigma_df.empty:
            logging.warning("Skip sigma=%s: no valid bundles parsed.", sigma_tag)
            continue
        sigma_out_dirs[sigma_tag] = _run_single_sigma(args, sigma_tag, sigma_df)

    if not sigma_out_dirs:
        raise SystemExit("No sigma results generated.")

    root_out = args.output_root / args.results_prefix
    _aggregate_sigma_comparison(
        root_out, sigma_out_dirs, drug=args.drug_label, pcb=args.pcb_label
    )
    print(f"Done. Outputs in: {root_out}")


if __name__ == "__main__":
    main()
