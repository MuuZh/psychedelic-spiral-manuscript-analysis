from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.all_metrics.config import Config
from analysis.all_metrics.loaders import load_bundles
from analysis.all_metrics.utils import (
    build_group_summary_df,
    plot_paired_violin,
    save_fig,
    set_output_naming,
    setup_logging,
    write_table,
)


DEFAULT_DRUGS = ("DMT", "LSD")
METRICS = ["gipr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Generalized Inverse Participation Ratio (GIPR) from vortex occupancy maps."
    )
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=list(DEFAULT_DRUGS),
        help="Drug labels to analyze. Each label reads detect_results/<drug> by default.",
    )
    parser.add_argument("--pcb-label", default="PCB")
    parser.add_argument("--detect-root", type=Path, default=Path("detect_results"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs") / "added_analysis" / "GIPR",
    )
    parser.add_argument("--prefix-suffix", default="vs_pcb_run")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def build_occupancy_map(bundle_dir: Path) -> tuple[np.ndarray | None, int]:
    occ_path = bundle_dir / "vortex_occupancy.npy"
    if occ_path.exists():
        occ = np.load(occ_path)
        if occ.ndim == 2:
            return occ.astype(np.float64, copy=False), 0

    meta_path = bundle_dir / "metadata.json"
    fi_path = bundle_dir / "frame_index.parquet"
    coords_path = bundle_dir / "coords.feather"
    if not (meta_path.exists() and fi_path.exists() and coords_path.exists()):
        return None, 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    height = int(meta.get("grid_height", 0))
    width = int(meta.get("grid_width", 0))
    frame_count_meta = int(meta.get("frame_count", 0))
    frame_index = pd.read_parquet(fi_path, columns=["abs_time", "coord_start", "coord_end"])
    n_frames = int(frame_index["abs_time"].nunique())
    frame_count = frame_count_meta if frame_count_meta > 0 else n_frames
    if height <= 0 or width <= 0 or frame_count <= 0:
        return None, 0

    coords_df = pd.read_feather(coords_path, columns=["y", "x"])
    coords = coords_df.to_numpy()
    occupancy = np.zeros((height, width), dtype=np.int64)
    for _, frame_rows in frame_index.groupby("abs_time", sort=True):
        slices = []
        for _, row in frame_rows.iterrows():
            start = int(row["coord_start"])
            end = int(row["coord_end"])
            if end > start:
                slices.append(coords[start:end])
        if not slices:
            continue
        pts = slices[0] if len(slices) == 1 else np.concatenate(slices, axis=0)
        if pts.size == 0:
            continue
        uniq = np.unique(pts, axis=0)
        ys = uniq[:, 0].astype(int)
        xs = uniq[:, 1].astype(int)
        valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        occupancy[ys[valid], xs[valid]] += 1

    return occupancy.astype(np.float64) / float(frame_count), frame_count


def compute_gipr(occupancy: np.ndarray) -> tuple[float, np.ndarray]:
    rho = np.asarray(occupancy, dtype=np.float64)
    finite = np.isfinite(rho)
    total = float(np.nansum(rho[finite]))
    contribution = np.full_like(rho, np.nan, dtype=np.float64)
    if total <= 0.0:
        return math.nan, contribution
    contribution[finite] = (rho[finite] ** 2) / (total ** 2)
    return float(np.nansum(contribution)), contribution


def _append_mean_map(
    map_sums: Dict[tuple[str, str], np.ndarray],
    map_counts: Dict[tuple[str, str], int],
    key: tuple[str, str],
    value_map: np.ndarray,
) -> None:
    if key not in map_sums:
        map_sums[key] = np.zeros_like(value_map, dtype=np.float64)
        map_counts[key] = 0
    if map_sums[key].shape != value_map.shape:
        logging.warning("Skip mean map accumulation for %s due to shape mismatch.", key)
        return
    map_sums[key] += np.nan_to_num(value_map, nan=0.0)
    map_counts[key] += 1


def plot_group_heatmaps(
    out_dir: Path,
    cfg: Config,
    active_groups: Iterable[str],
    contribution_sums: Dict[tuple[str, str], np.ndarray],
    contribution_counts: Dict[tuple[str, str], int],
) -> None:
    for hemi in ["left", "right"]:
        for group in active_groups:
            key = (hemi, group)
            count = contribution_counts.get(key, 0)
            if count <= 0 or key not in contribution_sums:
                continue
            mean_map = contribution_sums[key] / count
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mean_map, cmap="magma")
            ax.invert_yaxis()
            ax.set_title(f"GIPR contribution ({group}, {hemi})")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            save_fig(fig, out_dir / f"heatmap_gipr_contribution_{group}_{hemi}.png", cfg.save_plots)


def plot_gipr_distributions(subj_df: pd.DataFrame, out_dir: Path, cfg: Config) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, hemi in zip(axes, ["left", "right"]):
        tidy = subj_df[subj_df["hemisphere"] == hemi]
        sns.violinplot(
            data=tidy,
            x="group",
            y="gipr",
            order=[cfg.group_pcb, cfg.group_drug],
            ax=ax,
            cut=0,
        )
        ax.set_title(f"GIPR ({hemi})")
    fig.suptitle("GIPR")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_dir / "violin_gipr.png", cfg.save_plots)

    for hemi in ["left", "right"]:
        tidy = subj_df[subj_df["hemisphere"] == hemi][["subid", "group", "gipr"]]
        plot_paired_violin(tidy, "gipr", hemi, "GIPR paired", out_dir / f"paired_gipr_{hemi}.png", cfg)


def run_one_drug(args: argparse.Namespace, drug: str) -> pd.DataFrame:
    cfg = Config()
    cfg.group_drug = drug
    cfg.group_pcb = args.pcb_label
    cfg.results_prefix = f"{drug.lower()}_{args.prefix_suffix}"
    cfg.detect_results_dir = args.detect_root / drug
    cfg.output_root = args.output_root
    cfg.reuse_cache = not args.no_cache
    cfg.save_plots = not args.no_plots

    out_dir = cfg.output_root / cfg.results_prefix
    set_output_naming(cfg.results_prefix, datetime.now().strftime("%Y%m%d"))
    setup_logging(out_dir)
    logging.info("Running GIPR for %s vs %s from %s", cfg.group_drug, cfg.group_pcb, cfg.detect_results_dir)

    bundles = load_bundles(cfg)
    target_groups = {cfg.group_pcb, cfg.group_drug}
    bundles = bundles[bundles["group"].isin(target_groups)].copy()
    if bundles.empty:
        logging.warning("No bundles found for %s.", drug)
        return pd.DataFrame()

    records: list[dict] = []
    contribution_sums: Dict[tuple[str, str], np.ndarray] = {}
    contribution_counts: Dict[tuple[str, str], int] = {}

    iterator = tqdm(
        bundles.iterrows(),
        total=len(bundles),
        desc=f"GIPR {drug}",
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for _, row in iterator:
        group = str(row["group"])
        subid = str(row["subid"])
        hemi = str(row["hemisphere"]).lower()
        bundle_dir = Path(row["bundle_dir"])
        base = {"drug_set": drug, "group": group, "subid": subid, "hemisphere": hemi}
        occupancy, _ = build_occupancy_map(bundle_dir)
        if occupancy is None or occupancy.size == 0:
            records.append({**base, "gipr": math.nan, "occupancy_total": math.nan, "occupancy_nonzero_fraction": math.nan})
            continue

        if cfg.reuse_cache and not (bundle_dir / "vortex_occupancy.npy").exists():
            np.save(bundle_dir / "vortex_occupancy.npy", occupancy)

        gipr, contribution = compute_gipr(occupancy)
        records.append(
            {
                **base,
                "gipr": gipr,
                "occupancy_total": float(np.nansum(occupancy)),
                "occupancy_nonzero_fraction": float(np.count_nonzero(occupancy) / occupancy.size),
            }
        )
        _append_mean_map(contribution_sums, contribution_counts, (hemi, group), contribution)

    subj_df = (
        pd.DataFrame(records)
        .groupby(["drug_set", "group", "subid", "hemisphere"], as_index=False)
        .mean(numeric_only=True)
    )
    write_table(subj_df, out_dir / "per_subject.csv")
    write_table(build_group_summary_df(subj_df, METRICS, cfg), out_dir / "group_summary.csv")
    plot_gipr_distributions(subj_df, out_dir, cfg)
    plot_group_heatmaps(out_dir, cfg, target_groups, contribution_sums, contribution_counts)
    logging.info("Finished GIPR for %s. Rows: %d", drug, len(subj_df))
    return subj_df


def main() -> None:
    args = parse_args()
    all_frames = []
    for drug in args.drugs:
        drug_clean = str(drug).strip().upper()
        if not drug_clean:
            continue
        result = run_one_drug(args, drug_clean)
        if not result.empty:
            all_frames.append(result)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out_dir = args.output_root / "combined"
        set_output_naming("gipr_combined", datetime.now().strftime("%Y%m%d"))
        write_table(combined, out_dir / "per_subject_all_drugs.csv")


if __name__ == "__main__":
    main()
