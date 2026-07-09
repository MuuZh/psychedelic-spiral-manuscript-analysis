from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import Config
from .utils import (
    build_group_summary_df,
    paired_t,
    plot_paired_violin,
    save_fig,
    set_output_naming,
    setup_logging,
    unpaired_t,
    write_table,
)


SOURCE_LABELS = {
    "pattern_all": "Pattern all",
    "instantaneous_all": "Instantaneous all",
    "pattern_unfiltered": "Pattern all_frames_filtered=False",
    "instantaneous_compat_pass": "Instantaneous compatibility_pass=True",
}
METRIC_COLS = ["count_per_frame", "mean_size"]


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean count and mean size for four vortex sources: "
            "pattern-level, instantaneous, pattern-level retained by all_frames_filtered, "
            "and instantaneous retained by compatibility_pass. Count is normalized as count per frame."
        )
    )
    parser.add_argument("--prefix", dest="results_prefix", default=None)
    parser.add_argument("--drug-label", dest="group_drug", default=None)
    parser.add_argument("--pcb-label", dest="group_pcb", default=None)
    parser.add_argument("--combined-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.results_prefix:
        cfg.results_prefix = args.results_prefix
    if args.group_drug:
        cfg.group_drug = args.group_drug
    if args.group_pcb:
        cfg.group_pcb = args.group_pcb
    if args.combined_dir:
        cfg.combined_dir = args.combined_dir
    if args.output_root:
        cfg.output_root = args.output_root
    if args.no_plots:
        cfg.save_plots = False
    return cfg


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def _load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    patterns_path = cfg.combined_dir / "combined_patterns.parquet"
    frames_path = cfg.combined_dir / "combined_frame_index.parquet"
    if not patterns_path.exists():
        raise FileNotFoundError(f"Missing {patterns_path}")
    if not frames_path.exists():
        raise FileNotFoundError(f"Missing {frames_path}")

    patterns = pd.read_parquet(patterns_path)
    frames = pd.read_parquet(frames_path)
    groups = {cfg.group_pcb, cfg.group_drug}
    if "group" in patterns.columns:
        patterns = patterns[patterns["group"].isin(groups)].copy()
    if "group" in frames.columns:
        frames = frames[frames["group"].isin(groups)].copy()
    return patterns, frames


def _base_subject_hemi(patterns: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    keys = ["group", "subid", "hemisphere"]
    pieces = []
    for df in (patterns, frames):
        if set(keys).issubset(df.columns):
            pieces.append(df[keys].drop_duplicates())
    if not pieces:
        return pd.DataFrame(columns=keys)
    base = pd.concat(pieces, ignore_index=True).drop_duplicates()
    base["subid"] = base["subid"].astype(str)
    if set(keys + ["abs_time"]).issubset(frames.columns):
        frame_count_cols = keys + ["abs_time"]
        if "bundle_dir" in frames.columns:
            frame_count_cols = keys + ["bundle_dir", "abs_time"]
        frame_counts = (
            frames.drop_duplicates(frame_count_cols)
            .groupby(keys, as_index=False)
            .size()
            .rename(columns={"size": "frame_count_total"})
        )
        frame_counts["subid"] = frame_counts["subid"].astype(str)
        base = base.merge(frame_counts, on=keys, how="left")
    else:
        base["frame_count_total"] = pd.NA
    return base.sort_values(keys).reset_index(drop=True)


def _pattern_count(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in ["bundle_dir", "pattern_id"] if c in df.columns]
    if "pattern_id" not in cols:
        return pd.Series(dtype=float)
    return df.drop_duplicates(cols).groupby(["group", "subid", "hemisphere"]).size()


def _aggregate_patterns(
    df: pd.DataFrame,
    base: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    keys = ["group", "subid", "hemisphere"]
    out = base.copy()
    out["source"] = source
    out["source_label"] = SOURCE_LABELS[source]

    if df.empty:
        out["raw_count"] = 0
        out["count_per_frame"] = pd.NA
        out["mean_size"] = math.nan
        out = _add_count_per_frame(out)
        return out

    counts = _pattern_count(df).rename("raw_count").reset_index()
    sizes = (
        df.groupby(keys, as_index=False)["mean_size"]
        .mean()
        .rename(columns={"mean_size": "mean_size"})
    )
    out = out.merge(counts, on=keys, how="left").merge(sizes, on=keys, how="left")
    out["raw_count"] = out["raw_count"].fillna(0).astype(int)
    return _add_count_per_frame(out)


def _aggregate_frames(
    df: pd.DataFrame,
    base: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    keys = ["group", "subid", "hemisphere"]
    out = base.copy()
    out["source"] = source
    out["source_label"] = SOURCE_LABELS[source]

    if df.empty:
        out["raw_count"] = 0
        out["count_per_frame"] = pd.NA
        out["mean_size"] = math.nan
        out = _add_count_per_frame(out)
        return out

    agg = (
        df.groupby(keys, as_index=False)
        .agg(raw_count=("pattern_id", "size"), mean_size=("instantaneous_size", "mean"))
    )
    out = out.merge(agg, on=keys, how="left")
    out["raw_count"] = out["raw_count"].fillna(0).astype(int)
    return _add_count_per_frame(out)


def _add_count_per_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    frame_counts = pd.to_numeric(out["frame_count_total"], errors="coerce")
    raw_counts = pd.to_numeric(out["raw_count"], errors="coerce").fillna(0.0)
    out["count_per_frame"] = raw_counts / frame_counts.where(frame_counts > 0)
    return out


def _build_per_subject(patterns: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    required_patterns = {"group", "subid", "hemisphere", "pattern_id", "mean_size"}
    required_frames = {"group", "subid", "hemisphere", "pattern_id", "instantaneous_size"}
    missing_patterns = required_patterns - set(patterns.columns)
    missing_frames = required_frames - set(frames.columns)
    if missing_patterns:
        raise ValueError(f"combined_patterns.parquet missing columns: {sorted(missing_patterns)}")
    if missing_frames:
        raise ValueError(f"combined_frame_index.parquet missing columns: {sorted(missing_frames)}")

    base = _base_subject_hemi(patterns, frames)
    if base.empty:
        return pd.DataFrame()

    rows = [
        _aggregate_patterns(patterns, base, "pattern_all"),
        _aggregate_frames(frames, base, "instantaneous_all"),
    ]

    if "all_frames_filtered" in patterns.columns:
        retained_patterns = patterns[~_as_bool(patterns["all_frames_filtered"])].copy()
        rows.append(_aggregate_patterns(retained_patterns, base, "pattern_unfiltered"))
    else:
        logging.warning("Skipping pattern_unfiltered: all_frames_filtered column is missing")

    if "compatibility_pass" in frames.columns:
        retained_frames = frames[_as_bool(frames["compatibility_pass"])].copy()
        rows.append(_aggregate_frames(retained_frames, base, "instantaneous_compat_pass"))
    else:
        logging.warning("Skipping instantaneous_compat_pass: compatibility_pass column is missing")

    per_subject = pd.concat(rows, ignore_index=True)
    per_subject["subid"] = per_subject["subid"].astype(str)
    return per_subject


def _append_tests(per_subject: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[Dict] = []
    for source, source_df in per_subject.groupby("source", sort=False):
        for hemi in ["left", "right"]:
            hemi_df = source_df[source_df["hemisphere"] == hemi]
            if hemi_df.empty:
                continue
            pivot = hemi_df.pivot_table(index="subid", columns="group", values=METRIC_COLS, aggfunc="mean")
            if not pivot.empty:
                pivot.columns = ["_".join(col) for col in pivot.columns]
            for metric in METRIC_COLS:
                drug = pivot.get(f"{metric}_{cfg.group_drug}", pd.Series(dtype=float))
                pcb = pivot.get(f"{metric}_{cfg.group_pcb}", pd.Series(dtype=float))
                rows.append(
                    {
                        "section": "vortex_count_size_sources",
                        "source": source,
                        "source_label": SOURCE_LABELS.get(source, source),
                        "metric": metric,
                        "hemisphere": hemi,
                        "comparison": "paired_drug_vs_pcb",
                        **paired_t(drug, pcb),
                    }
                )

                drug_unpaired = hemi_df[hemi_df["group"] == cfg.group_drug][metric]
                pcb_unpaired = hemi_df[hemi_df["group"] == cfg.group_pcb][metric]
                rows.append(
                    {
                        "section": "vortex_count_size_sources",
                        "source": source,
                        "source_label": SOURCE_LABELS.get(source, source),
                        "metric": metric,
                        "hemisphere": hemi,
                        "comparison": "unpaired_drug_vs_pcb",
                        **unpaired_t(drug_unpaired, pcb_unpaired),
                    }
                )
    return pd.DataFrame(rows)


def _build_group_summary(per_subject: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    summaries = []
    for source, source_df in per_subject.groupby("source", sort=False):
        summary = build_group_summary_df(source_df, METRIC_COLS, cfg)
        if summary.empty:
            continue
        summary.insert(0, "source", source)
        summary.insert(1, "source_label", SOURCE_LABELS.get(source, source))
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def _plot(per_subject: pd.DataFrame, out_dir: Path, cfg: Config) -> None:
    if per_subject.empty:
        return
    sns.set_theme(style="whitegrid", context="talk")
    for source, source_df in per_subject.groupby("source", sort=False):
        source_dir = out_dir / source
        for hemi in ["left", "right"]:
            hemi_df = source_df[source_df["hemisphere"] == hemi]
            if hemi_df.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, metric in zip(axes, METRIC_COLS):
                sns.violinplot(
                    data=hemi_df,
                    x="group",
                    y=metric,
                    order=[cfg.group_pcb, cfg.group_drug],
                    ax=ax,
                    cut=0,
                )
                ax.set_title(f"{SOURCE_LABELS.get(source, source)} {metric} ({hemi})")
            fig.tight_layout()
            save_fig(fig, source_dir / f"violin_{hemi}.png", cfg.save_plots)
            for metric in METRIC_COLS:
                plot_paired_violin(
                    hemi_df[["subid", "group", metric]],
                    metric,
                    hemi,
                    f"{SOURCE_LABELS.get(source, source)} {metric} paired",
                    source_dir / f"paired_{metric}_{hemi}.png",
                    cfg,
                )


def main() -> None:
    cfg = _parse_args()
    out_root = cfg.output_root / cfg.results_prefix / "vortex_count_size_sources"
    set_output_naming(cfg.results_prefix, datetime.now().strftime("%Y%m%d"))
    setup_logging(out_root)
    logging.info("Loading combined inputs from %s", cfg.combined_dir)

    patterns, frames = _load_inputs(cfg)
    per_subject = _build_per_subject(patterns, frames)
    if per_subject.empty:
        logging.warning("No per-subject rows produced")
        return

    write_table(per_subject, out_root / "per_subject.csv")
    write_table(_build_group_summary(per_subject, cfg), out_root / "group_summary.csv")
    write_table(_append_tests(per_subject, cfg), out_root / "summary.csv")
    _plot(per_subject, out_root / "plots", cfg)
    logging.info("Done. Wrote %d per-subject/source rows to %s", len(per_subject), out_root)


if __name__ == "__main__":
    main()
