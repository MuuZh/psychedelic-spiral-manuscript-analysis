#!/usr/bin/env python
# %%
"""
Path entropy and occupancy statistics for paired DMT vs PCB subjects.

This standalone script follows the all_metrics runner style:
1) Compute multiple per-subject path metrics from frame-level centroids.
2) Save subject-level tables, paired deltas, group summaries, and plots.
3) Run paired and unpaired tests through analysis/all_metrics/utils.py.

Run cell-by-cell in an interactive window, or execute as:
    python analysis/path_entropy_paired.py
"""

# %%
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from all_metrics.utils import (
    build_group_summary_df,
    paired_t,
    plot_paired_violin as runner_plot_paired_violin,
    save_fig,
    unpaired_t,
    write_table,
)

sns.set_theme(style="whitegrid", context="talk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Path entropy and occupancy statistics for paired drug vs PCB subjects."
    )
    parser.add_argument("--drug-label", default="DMT", help="Drug group label, e.g. DMT or LSD.")
    parser.add_argument("--pcb-label", default="PCB", help="Placebo group label.")
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=None,
        help="Directory containing combined_frame_index.parquet. Defaults to combined_outputs/<drug-label>.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to analysis_outputs/path_entropy/<drug-label lower>_vs_<pcb-label lower>.",
    )
    parser.add_argument("--bin-size", type=float, default=2.0)
    parser.add_argument("--min-pattern-frames", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots.")
    parser.add_argument(
        "--no-inline-preview",
        action="store_true",
        help="Close figures after saving when running non-interactively.",
    )
    return parser.parse_args()


args = parse_args()

# Paths and labels
here = Path(__file__).resolve().parent
project_root = here.parent
combined_dir = args.combined_dir or (project_root / "combined_outputs" / args.drug_label)
if not (combined_dir / "combined_frame_index.parquet").exists():
    # Backward-compatible fallback for older merged outputs.
    combined_dir = project_root / "combined_outputs"
out_dir = args.out_dir or (
    project_root
    / "analysis_outputs"
    / "path_entropy"
    / f"{args.drug_label.lower()}_vs_{args.pcb_label.lower()}"
)
out_dir.mkdir(parents=True, exist_ok=True)

GROUP_PCB = args.pcb_label
GROUP_DRUG = args.drug_label
GROUP_ORDER = [GROUP_PCB, GROUP_DRUG]
PALETTE = {GROUP_PCB: "#4575b4", GROUP_DRUG: "#d73027"}

# Inline preview flag: set True in interactive runs to keep figs open/returned.
INLINE_PREVIEW = not args.no_inline_preview
SAVE_PLOTS = not args.no_plots

# Minimum frames required for a pattern to be included.
MIN_PATTERN_FRAMES = args.min_pattern_frames

# Bin size for discretizing centroids to grid cells.
BIN_SIZE = args.bin_size

cfg = SimpleNamespace(
    group_pcb=GROUP_PCB,
    group_drug=GROUP_DRUG,
    output_root=project_root / "analysis_outputs",
    results_prefix=f"path_entropy/{GROUP_DRUG.lower()}_vs_{GROUP_PCB.lower()}",
    save_plots=SAVE_PLOTS,
)

# %%
# Load frame-level centroids.
frame_path = combined_dir / "combined_frame_index.parquet"
if not frame_path.exists():
    raise FileNotFoundError(f"Missing frame index parquet: {frame_path}")

available_cols = pd.read_parquet(frame_path).columns.tolist()
preferred_cols = [
    "group",
    "subid",
    "hemisphere",
    "pattern_id",
    "centroid_x",
    "centroid_y",
    "weighted_centroid_x",
    "weighted_centroid_y",
]
frame_cols = [col for col in preferred_cols if col in available_cols]
missing_required = {"group", "subid", "hemisphere", "pattern_id", "centroid_x", "centroid_y"} - set(frame_cols)
if missing_required:
    raise ValueError(f"Missing required columns in {frame_path}: {sorted(missing_required)}")

frames = pd.read_parquet(frame_path, columns=frame_cols)
frames = frames[frames["group"].isin(GROUP_ORDER)].copy()
frames["subid"] = frames["subid"].astype(str)

if {"weighted_centroid_x", "weighted_centroid_y"}.issubset(frames.columns):
    frames["path_x"] = pd.to_numeric(frames["weighted_centroid_x"], errors="coerce")
    frames["path_y"] = pd.to_numeric(frames["weighted_centroid_y"], errors="coerce")
    missing_xy = frames["path_x"].isna() | frames["path_y"].isna()
    frames.loc[missing_xy, "path_x"] = pd.to_numeric(frames.loc[missing_xy, "centroid_x"], errors="coerce")
    frames.loc[missing_xy, "path_y"] = pd.to_numeric(frames.loc[missing_xy, "centroid_y"], errors="coerce")
else:
    frames["path_x"] = pd.to_numeric(frames["centroid_x"], errors="coerce")
    frames["path_y"] = pd.to_numeric(frames["centroid_y"], errors="coerce")

frames = frames.dropna(subset=["path_x", "path_y"])
if MIN_PATTERN_FRAMES > 1:
    counts = (
        frames.groupby(["group", "subid", "hemisphere", "pattern_id"], dropna=False)
        .size()
        .reset_index(name="frame_count")
    )
    keep_patterns = counts[counts["frame_count"] >= MIN_PATTERN_FRAMES][
        ["group", "subid", "hemisphere", "pattern_id"]
    ]
    before = len(frames)
    frames = frames.merge(
        keep_patterns,
        on=["group", "subid", "hemisphere", "pattern_id"],
        how="inner",
    )
    print(
        f"Filtered by MIN_PATTERN_FRAMES={MIN_PATTERN_FRAMES}: "
        f"kept {len(keep_patterns)} patterns; frames {len(frames)} (from {before})"
    )
print(f"Loaded frames: {len(frames)} rows from {frame_path}")


# %%
def compute_path_metrics(coords: np.ndarray, bin_size: float = 1.0) -> dict[str, float]:
    """Compute spatial occupancy metrics from XY centroid coordinates."""
    if coords.size == 0:
        return {
            "entropy": math.nan,
            "normalized_entropy": math.nan,
            "effective_cells": math.nan,
            "visited_cells": 0,
            "total_points": 0,
            "max_cell_probability": math.nan,
            "cell_count_cv": math.nan,
        }

    binned = np.rint(coords / bin_size).astype(int)
    _, counts = np.unique(binned, axis=0, return_counts=True)
    total = int(counts.sum())
    if total == 0:
        return {
            "entropy": math.nan,
            "normalized_entropy": math.nan,
            "effective_cells": math.nan,
            "visited_cells": 0,
            "total_points": 0,
            "max_cell_probability": math.nan,
            "cell_count_cv": math.nan,
        }

    probs = counts.astype(float) / float(total)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    visited_cells = int(len(probs))
    normalized_entropy = (
        float(entropy / np.log2(visited_cells)) if visited_cells > 1 else math.nan
    )
    effective_cells = float(2.0 ** entropy)
    count_mean = float(np.mean(counts)) if counts.size else math.nan
    count_std = float(np.std(counts, ddof=1)) if counts.size > 1 else 0.0

    return {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "effective_cells": effective_cells,
        "visited_cells": visited_cells,
        "total_points": total,
        "max_cell_probability": float(np.max(probs)),
        "cell_count_cv": float(count_std / count_mean) if count_mean else math.nan,
    }


def summarize_metrics(df: pd.DataFrame, by: list[str], bin_size: float = 1.0) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = df.groupby(by, dropna=False)
    for keys, chunk in tqdm(grouped, total=grouped.ngroups, desc="Computing path metrics"):
        coords = chunk[["path_x", "path_y"]].to_numpy(dtype=float)
        metrics = compute_path_metrics(coords, bin_size=bin_size)
        entry = dict(zip(by, keys)) if isinstance(keys, tuple) else {by[0]: keys}
        entry.update(metrics)
        records.append(entry)
    return pd.DataFrame.from_records(records)


def paired_delta_table(subject_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hemi in ["left", "right", "whole"]:
        hemi_df = subject_df[subject_df["hemisphere"] == hemi]
        if hemi_df.empty:
            continue
        for metric in metrics:
            pivot = hemi_df.pivot_table(index="subid", columns="group", values=metric, aggfunc="mean")
            if GROUP_PCB not in pivot.columns or GROUP_DRUG not in pivot.columns:
                continue
            pivot = pivot[[GROUP_PCB, GROUP_DRUG]].dropna()
            for subid, row in pivot.iterrows():
                rows.append(
                    {
                        "subid": str(subid),
                        "hemisphere": hemi,
                        "metric": metric,
                        GROUP_PCB: float(row[GROUP_PCB]),
                        GROUP_DRUG: float(row[GROUP_DRUG]),
                        "delta_drug_minus_pcb": float(row[GROUP_DRUG] - row[GROUP_PCB]),
                    }
                )
    return pd.DataFrame(rows)


def compact_test_summary(subject_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hemi in ["left", "right", "whole"]:
        hemi_df = subject_df[subject_df["hemisphere"] == hemi]
        if hemi_df.empty:
            continue
        for metric in metrics:
            drug = hemi_df[hemi_df["group"] == GROUP_DRUG].set_index("subid")[metric]
            pcb = hemi_df[hemi_df["group"] == GROUP_PCB].set_index("subid")[metric]
            paired_ids = drug.index.intersection(pcb.index)
            rows.append(
                {
                    "section": "path_entropy",
                    "metric": metric,
                    "hemisphere": hemi,
                    "comparison": "paired_drug_vs_pcb",
                    **paired_t(drug.loc[paired_ids], pcb.loc[paired_ids]),
                }
            )
            rows.append(
                {
                    "section": "path_entropy",
                    "metric": metric,
                    "hemisphere": hemi,
                    "comparison": "unpaired_drug_vs_pcb",
                    **unpaired_t(drug, pcb),
                }
            )
            if len(paired_ids) > 0:
                try:
                    wil = stats.wilcoxon(drug.loc[paired_ids], pcb.loc[paired_ids], nan_policy="omit")
                    rows.append(
                        {
                            "section": "path_entropy",
                            "metric": metric,
                            "hemisphere": hemi,
                            "comparison": "paired_wilcoxon_drug_vs_pcb",
                            "n": int(len(paired_ids)),
                            "stat": float(wil.statistic),
                            "p": float(wil.pvalue),
                        }
                    )
                except ValueError:
                    rows.append(
                        {
                            "section": "path_entropy",
                            "metric": metric,
                            "hemisphere": hemi,
                            "comparison": "paired_wilcoxon_drug_vs_pcb",
                            "n": int(len(paired_ids)),
                            "stat": math.nan,
                            "p": math.nan,
                        }
                    )
    return pd.DataFrame(rows)


def plot_metric_grid(subject_df: pd.DataFrame, metrics: list[str], hemi: str, save_path: Path) -> None:
    plot_df = subject_df[subject_df["hemisphere"] == hemi].copy()
    if plot_df.empty:
        return
    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        sns.violinplot(
            data=plot_df,
            x="group",
            y=metric,
            order=GROUP_ORDER,
            palette=PALETTE,
            cut=0,
            ax=ax,
        )
        ax.set_title(f"{metric} ({hemi})")
        ax.set_xlabel("")
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(f"Path entropy metrics ({hemi})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if INLINE_PREVIEW:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        save_fig(fig, save_path, SAVE_PLOTS)


# %% Compute per-subject metrics.
metric_cols = [
    "entropy",
    "normalized_entropy",
    "effective_cells",
    "visited_cells",
    "total_points",
    "max_cell_probability",
    "cell_count_cv",
]

entropy_hemi = summarize_metrics(
    frames,
    by=["group", "subid", "hemisphere"],
    bin_size=BIN_SIZE,
)
entropy_whole = summarize_metrics(
    frames,
    by=["group", "subid"],
    bin_size=BIN_SIZE,
)
entropy_whole["hemisphere"] = "whole"
subject_metrics = pd.concat([entropy_hemi, entropy_whole], ignore_index=True)

print(subject_metrics.head())

# %% Save tables and statistics.
write_table(entropy_hemi, out_dir / "per_subject_by_hemisphere.csv")
write_table(entropy_whole, out_dir / "per_subject_whole.csv")
write_table(subject_metrics, out_dir / "per_subject_with_whole.csv")

paired_deltas = paired_delta_table(subject_metrics, metric_cols)
write_table(paired_deltas, out_dir / "paired_deltas.csv")

group_summary = build_group_summary_df(entropy_hemi, metric_cols, cfg)
write_table(group_summary, out_dir / "group_summary.csv")

test_summary = compact_test_summary(subject_metrics, metric_cols)
write_table(test_summary, out_dir / "test_summary.csv")

print("Saved subject metrics:", out_dir / "per_subject_with_whole.csv")
print("Saved paired deltas:", out_dir / "paired_deltas.csv")
print("Saved group summary:", out_dir / "group_summary.csv")
print("Saved compact test summary:", out_dir / "test_summary.csv")

# %% Plots.
for hemi in ["left", "right", "whole"]:
    plot_metric_grid(subject_metrics, metric_cols, hemi, out_dir / f"violin_{hemi}.png")
    for metric in metric_cols:
        tidy = subject_metrics[subject_metrics["hemisphere"] == hemi][["subid", "group", metric]]
        runner_plot_paired_violin(
            tidy,
            metric,
            hemi,
            f"Path entropy {metric} paired",
            out_dir / f"paired_{metric}_{hemi}.png",
            cfg,
        )

print("Done. Output directory:", out_dir)
