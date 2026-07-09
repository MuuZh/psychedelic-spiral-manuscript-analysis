from __future__ import annotations

import math
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import Config
from .utils import build_group_summary_df, write_table, save_fig, paired_t, unpaired_t, plot_paired_violin


def run_pattern_stats(cfg: Config, summary: List[Dict]) -> None:
    patterns_path = cfg.combined_dir / "combined_patterns.parquet"
    frames_path = cfg.combined_dir / "combined_frame_index.parquet"
    if not patterns_path.exists():
        return

    patterns = pd.read_parquet(patterns_path)
    required = {"group", "subid", "hemisphere", "pattern_id", "mean_size", "duration", "mean_power"}
    missing = required - set(patterns.columns)
    if missing:
        return

    groups = {cfg.group_pcb, cfg.group_drug}
    patterns = patterns[patterns["group"].isin(groups)].copy()
    frame_counts = None
    if frames_path.exists():
        frame_index = pd.read_parquet(
            frames_path,
            columns=["group", "subid", "hemisphere", "bundle_dir", "abs_time"],
        )
        frame_index = frame_index[frame_index["group"].isin(groups)].copy()
        frame_counts = (
            frame_index.drop_duplicates(["group", "subid", "hemisphere", "bundle_dir", "abs_time"])
            .groupby(["group", "subid", "hemisphere"], as_index=False)
            .size()
            .rename(columns={"size": "frame_count_total"})
        )

    def aggregate(df: pd.DataFrame) -> pd.DataFrame:
        agg_df = (
            df.groupby(["group", "subid", "hemisphere"])
            .agg(
                pattern_count=("pattern_id", "nunique"),
                mean_size=("mean_size", "mean"),
                mean_duration=("duration", "mean"),
                mean_power=("mean_power", "mean"),
            )
            .reset_index()
        )
        if frame_counts is not None:
            agg_df = agg_df.merge(frame_counts, on=["group", "subid", "hemisphere"], how="left")
            agg_df["pattern_count_per_frame"] = agg_df["pattern_count"] / agg_df["frame_count_total"]
            agg_df["pattern_count_per_100_frames"] = agg_df["pattern_count_per_frame"] * 100.0
        else:
            agg_df["frame_count_total"] = pd.NA
            agg_df["pattern_count_per_frame"] = pd.NA
            agg_df["pattern_count_per_100_frames"] = pd.NA
        return agg_df

    agg_df = aggregate(patterns)
    metrics = [
        "pattern_count",
        "pattern_count_per_frame",
        "pattern_count_per_100_frames",
        "mean_size",
        "mean_duration",
        "mean_power",
    ]

    out_dir = cfg.output_root / cfg.results_prefix / "pattern_stats"
    write_table(agg_df, out_dir / "per_subject.csv")

    # paired per hemisphere
    for hemi in ["left", "right"]:
        hemi_df = agg_df[agg_df["hemisphere"] == hemi]
        pivot = hemi_df.pivot(index="subid", columns="group", values=metrics)
        pivot.columns = ["_".join(col) for col in pivot.columns]
        pivot = pivot.dropna()
        for metric in metrics:
            drug = pivot.get(f"{metric}_{cfg.group_drug}", pd.Series(dtype=float))
            pcb = pivot.get(f"{metric}_{cfg.group_pcb}", pd.Series(dtype=float))
            summary.append({"section": "pattern_stats", "metric": metric, "hemisphere": hemi, "comparison": "paired_drug_vs_pcb", **paired_t(drug, pcb)})

    # unpaired per hemisphere
    for hemi in ["left", "right"]:
        hemi_df = agg_df[agg_df["hemisphere"] == hemi]
        for metric in metrics:
            drug = hemi_df[hemi_df["group"] == cfg.group_drug][metric]
            pcb = hemi_df[hemi_df["group"] == cfg.group_pcb][metric]
            summary.append({"section": "pattern_stats", "metric": metric, "hemisphere": hemi, "comparison": "unpaired_drug_vs_pcb", **unpaired_t(drug, pcb)})

    # group summaries
    group_summary_df = build_group_summary_df(agg_df, metrics, cfg)
    write_table(group_summary_df, out_dir / "group_summary.csv")

    sns.set_theme(style="whitegrid", context="talk")
    for hemi in ["left", "right"]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, metric in zip(axes.flat, metrics):
            tidy = agg_df[agg_df["hemisphere"] == hemi][["subid", "group", metric]]
            sns.violinplot(data=tidy, x="group", y=metric, order=[cfg.group_pcb, cfg.group_drug], ax=ax, cut=0)
            ax.set_title(f"{metric} ({hemi})")
        fig.suptitle(f"Pattern metrics ({hemi})")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, out_dir / f"violin_{hemi}.png", cfg.save_plots)
        for metric in metrics:
            tidy = agg_df[agg_df["hemisphere"] == hemi][["subid", "group", metric]]
            plot_paired_violin(tidy, metric, hemi, f"Pattern {metric} paired", out_dir / f"paired_{metric}_{hemi}.png", cfg)
