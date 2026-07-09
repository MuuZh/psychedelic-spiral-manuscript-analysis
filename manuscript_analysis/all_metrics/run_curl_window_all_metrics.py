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
from .patterns import run_pattern_stats
from .pattern_dynamics import run_pattern_dynamics
from .utils import plot_abs_effect_size_bars, setup_logging, set_output_naming, write_table
from .vortex import run_vortex_occupancy

matplotlib.use("Agg")


GROUP_SUB_RE = re.compile(r"^[A-Za-z]+_(DMT|PCB)_S(\d+)", re.IGNORECASE)


def _run_stage(name: str, fn, *args) -> None:
    logging.info("Start stage: %s", name)
    fn(*args)
    logging.info("Finished stage: %s", name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all_metrics-style analysis for each curl/method window batch and compare method tags.",
    )
    parser.add_argument(
        "--curl-root",
        type=Path,
        default=Path("output/curl_threshold_sweep"),
        help="Root directory created by run_detection_batch_curl_window.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs") / "all_metrics_curl_window",
    )
    parser.add_argument("--results-prefix", type=str, default="dmt_curl_window")
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


def _discover_tags(curl_root: Path) -> List[str]:
    tags = set()
    for subject_dir in curl_root.iterdir():
        if not subject_dir.is_dir():
            continue
        for tag_dir in subject_dir.iterdir():
            if not tag_dir.is_dir():
                continue
            name = tag_dir.name
            if name.startswith("curl_") or name == "optflow_focus":
                tags.add(name)
    return sorted(tags)


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


def _collect_bundle_rows(curl_root: Path, method_tag: str, drug: str, pcb: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for subject_dir in sorted(curl_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        tag_dir = subject_dir / method_tag
        if not tag_dir.exists():
            continue
        for bundle_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
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


def _combine_for_tag(bundle_df: pd.DataFrame, out_dir: Path) -> Path:
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


def _run_single_tag(args: argparse.Namespace, method_tag: str, tag_df: pd.DataFrame) -> Path:
    tag_out = args.output_root / args.results_prefix / method_tag
    set_output_naming(args.results_prefix, datetime.now().strftime("%Y%m%d"))
    setup_logging(tag_out)
    logging.info("Running tag=%s with %d bundles", method_tag, len(tag_df))

    combined_dir = _combine_for_tag(tag_df, tag_out)

    cfg = Config()
    cfg.results_prefix = "."
    cfg.group_drug = args.drug_label
    cfg.group_pcb = args.pcb_label
    cfg.detect_results_dir = args.curl_root
    cfg.combined_dir = combined_dir
    cfg.analytic_dir = args.analytic_dir
    cfg.parcellation_config = args.parcellation_config
    cfg.reference_gmap = args.reference_gmap
    cfg.reference_gmap_left = args.reference_gmap_left
    cfg.reference_gmap_right = args.reference_gmap_right
    cfg.output_root = tag_out
    cfg.reuse_cache = not args.no_cache
    cfg.save_plots = not args.no_plots
    cfg.tr_seconds = args.tr_seconds
    cfg.min_duration_for_msd = args.min_duration_for_msd
    cfg.csvd_method = args.csvd_method

    summary: List[Dict] = []
    _run_stage("boundary_regions", run_boundary_regions, cfg, tag_df, summary)
    _run_stage("pattern_stats", run_pattern_stats, cfg, summary)
    _run_stage("pattern_dynamics", run_pattern_dynamics, cfg, summary)
    _run_stage("csvd", run_csvd, cfg, summary)
    _run_stage("angle_diff_abs_cos", run_angle_diff_abs_cos, cfg, tag_df, summary)
    _run_stage("curl_spatial", run_curl_spatial, cfg, tag_df, summary)
    _run_stage("vortex_occupancy", run_vortex_occupancy, cfg, tag_df, summary)

    summary_df = pd.DataFrame(summary)
    write_table(summary_df, tag_out / "summary" / "all_metrics_summary.csv")
    plot_abs_effect_size_bars(
        summary_df,
        tag_out / "summary" / "abs_cohens_dz_by_metric.png",
        title=f"Absolute Cohen's dz by metric ({method_tag})",
        save=cfg.save_plots,
    )
    logging.info("Tag %s done. Summary rows=%d", method_tag, len(summary_df))
    return tag_out


def _build_group_mean_table(tag_out_dirs: Dict[str, Path], drug: str, pcb: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for method_tag, tag_out in tag_out_dirs.items():
        for section_dir in tag_out.iterdir():
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
                        "tag": method_tag,
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
        columns="tag",
        values="neg_log10_p",
        aggfunc="min",
    )
    if hm.empty:
        return
    fig_h = max(6, 0.32 * len(hm.index))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.heatmap(hm, cmap="mako", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Method tag")
    ax.set_ylabel("Metric")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _aggregate_tag_comparison(root_out: Path, tag_out_dirs: Dict[str, Path], drug: str, pcb: str) -> None:
    p_rows: List[pd.DataFrame] = []
    for method_tag, tag_out in tag_out_dirs.items():
        summary_csv = tag_out / "summary" / "all_metrics_summary.csv"
        if not summary_csv.exists():
            continue
        sdf = pd.read_csv(summary_csv)
        if sdf.empty or "p" not in sdf.columns:
            continue
        sdf["tag"] = method_tag
        p_rows.append(sdf)

    if not p_rows:
        return
    all_p = pd.concat(p_rows, ignore_index=True)
    all_p = all_p[pd.to_numeric(all_p["p"], errors="coerce").notna()].copy()
    write_table(all_p, root_out / "tag_comparison_pvalues_long.csv")

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
            "best_tag": best["tag"],
            "best_p": best["p"],
            "best_t": best.get("t", math.nan),
        }
        rank_rows.append(row)
    ranking_df = pd.DataFrame(rank_rows).sort_values(by="best_p", ascending=True)
    write_table(ranking_df, root_out / "tag_best_by_metric.csv")

    means_df = _build_group_mean_table(tag_out_dirs, drug=drug, pcb=pcb)
    if not means_df.empty:
        write_table(means_df, root_out / "tag_group_means_long.csv")

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

    tags = _discover_tags(args.curl_root)
    if not tags:
        raise SystemExit(f"No curl_* or optflow_focus folders found in: {args.curl_root}")

    tag_out_dirs: Dict[str, Path] = {}
    for method_tag in tags:
        tag_df = _collect_bundle_rows(
            args.curl_root, method_tag, drug=args.drug_label, pcb=args.pcb_label
        )
        if tag_df.empty:
            logging.warning("Skip tag=%s: no valid bundles parsed.", method_tag)
            continue
        tag_out_dirs[method_tag] = _run_single_tag(args, method_tag, tag_df)

    if not tag_out_dirs:
        raise SystemExit("No tag results generated.")

    root_out = args.output_root / args.results_prefix
    _aggregate_tag_comparison(
        root_out, tag_out_dirs, drug=args.drug_label, pcb=args.pcb_label
    )
    print(f"Done. Outputs in: {root_out}")


if __name__ == "__main__":
    main()
