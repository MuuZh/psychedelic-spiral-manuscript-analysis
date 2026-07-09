#!/usr/bin/env python
"""
Correlate per-subject pattern metrics with GCOR.

This is a focused version of pattern_gcor_correlations.py. It only joins the
pattern table and GCOR table, then computes correlation statistics for each
metric across several grouping levels.

Expected normalized join keys:

    drug, condition, subid, hemisphere

The pattern table usually stores condition in ``group`` and has no explicit
``drug`` column, so ``--pattern-drug`` assigns that label before joining.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DEFAULT_PATTERN_CSV = Path("analysis_outputs/all_metrics/dmt-run/pattern_stats/per_subject.csv")
DEFAULT_GCOR_CSV = Path("analysis_outputs/gcor_batch_DMT/all_gcor_by_subject.csv")
DEFAULT_OUT_DIR = Path("analysis_outputs/phase_fc_group/pattern_metric_gcor_correlations")
DEFAULT_METRICS = [
    "pattern_count",
    "mean_size",
    "mean_duration",
    "mean_power",
    "pattern_count_per_frame",
    "pattern_count_per_100_frames",
]
KEY_COLUMNS = ["drug", "condition", "subid", "hemisphere"]
GROUPING_SPECS = [
    ("condition_hemisphere", ["drug", "condition", "hemisphere"]),
    ("condition", ["drug", "condition"]),
    ("hemisphere", ["drug", "hemisphere"]),
    ("all", ["drug"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate pattern metrics with GCOR.")
    parser.add_argument("--pattern-csv", default=DEFAULT_PATTERN_CSV, type=Path)
    parser.add_argument("--gcor-csv", default=DEFAULT_GCOR_CSV, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument(
        "--pattern-drug",
        default="DMT",
        help="Drug label to assign to the pattern table if it does not have a drug column.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Pattern metric columns to correlate with GCOR.",
    )
    parser.add_argument("--min-n", default=3, type=int, help="Minimum complete rows required per correlation.")
    parser.add_argument("--no-plots", action="store_true", help="Skip scatter/regression plots.")
    parser.add_argument("--show-rows", default=20, type=int, help="Merged rows printed to console.")
    return parser.parse_args()


def normalize_subid(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("S"):
        text = text[1:]
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def load_pattern(path: Path, pattern_drug: str, metrics: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "condition" not in df.columns and "group" in df.columns:
        df = df.rename(columns={"group": "condition"})
    if "drug" not in df.columns:
        df["drug"] = pattern_drug

    required = set(KEY_COLUMNS + metrics)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = df[KEY_COLUMNS + metrics].copy()
    out["drug"] = out["drug"].astype(str).str.strip()
    out["condition"] = out["condition"].astype(str).str.strip()
    out["subid"] = out["subid"].map(normalize_subid)
    out["hemisphere"] = out["hemisphere"].astype(str).str.strip().str.lower()
    for metric in metrics:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")

    return (
        out.groupby(KEY_COLUMNS, as_index=False, dropna=False)
        .mean(numeric_only=True)
        .sort_values(KEY_COLUMNS)
    )


def load_gcor(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subid" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "subid"})

    required = set(KEY_COLUMNS + ["gcor"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = df[KEY_COLUMNS + ["gcor"]].copy()
    out["drug"] = out["drug"].astype(str).str.strip()
    out["condition"] = out["condition"].astype(str).str.strip()
    out["subid"] = out["subid"].map(normalize_subid)
    out["hemisphere"] = out["hemisphere"].astype(str).str.strip().str.lower()
    out["gcor"] = pd.to_numeric(out["gcor"], errors="coerce")

    return (
        out.groupby(KEY_COLUMNS, as_index=False, dropna=False)
        .mean(numeric_only=True)
        .sort_values(KEY_COLUMNS)
    )


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(q, index=p_values.index)

    finite_idx = np.where(finite)[0]
    finite_p = p[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[finite_idx[order]] = adjusted
    return pd.Series(q, index=p_values.index)


def pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se)))


def correlation_row(values: pd.DataFrame, metric: str, min_n: int) -> dict[str, object]:
    sub = values[[metric, "gcor"]].dropna()
    n = int(len(sub))
    x = sub[metric].to_numpy(dtype=float)
    y = sub["gcor"].to_numpy(dtype=float)

    row: dict[str, object] = {
        "metric": metric,
        "n": n,
        "n_missing_metric": int(values[metric].isna().sum()),
        "n_missing_gcor": int(values["gcor"].isna().sum()),
        "metric_mean": float(np.mean(x)) if n else np.nan,
        "metric_sd": float(np.std(x, ddof=1)) if n > 1 else np.nan,
        "metric_median": float(np.median(x)) if n else np.nan,
        "metric_min": float(np.min(x)) if n else np.nan,
        "metric_max": float(np.max(x)) if n else np.nan,
        "gcor_mean": float(np.mean(y)) if n else np.nan,
        "gcor_sd": float(np.std(y, ddof=1)) if n > 1 else np.nan,
        "gcor_median": float(np.median(y)) if n else np.nan,
        "gcor_min": float(np.min(y)) if n else np.nan,
        "gcor_max": float(np.max(y)) if n else np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "pearson_ci_low": np.nan,
        "pearson_ci_high": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "kendall_tau": np.nan,
        "kendall_p": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
        "slope_se": np.nan,
        "intercept_se": np.nan,
        "regression_p": np.nan,
    }

    if n < min_n or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return row

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    kendall = stats.kendalltau(x, y)
    reg = stats.linregress(x, y)
    ci_low, ci_high = pearson_ci(float(pearson.statistic), n)

    row.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "pearson_ci_low": ci_low,
            "pearson_ci_high": ci_high,
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "kendall_tau": float(kendall.statistic),
            "kendall_p": float(kendall.pvalue),
            "slope": float(reg.slope),
            "intercept": float(reg.intercept),
            "r_squared": float(reg.rvalue**2),
            "slope_se": float(reg.stderr),
            "intercept_se": float(reg.intercept_stderr),
            "regression_p": float(reg.pvalue),
        }
    )
    return row


def correlation_table(merged: pd.DataFrame, metrics: list[str], min_n: int) -> pd.DataFrame:
    rows = []
    for grouping_name, columns in GROUPING_SPECS:
        for keys, group in merged.groupby(columns, sort=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(columns, keys))
            for metric in metrics:
                row = correlation_row(group, metric, min_n=min_n)
                row.update(base)
                row["grouping"] = grouping_name
                rows.append(row)

    out = pd.DataFrame(rows)
    for p_col in ["pearson_p", "spearman_p", "kendall_p", "regression_p"]:
        out[f"{p_col}_fdr_bh"] = out.groupby("grouping", group_keys=False)[p_col].apply(bh_fdr)

    front = ["grouping", "drug", "condition", "hemisphere", "metric", "n"]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest].sort_values(["grouping", "drug", "condition", "hemisphere", "metric"])


def summary_table(merged: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg_spec = {
        "n": ("gcor", "count"),
        "gcor_mean": ("gcor", "mean"),
        "gcor_sd": ("gcor", "std"),
        "gcor_min": ("gcor", "min"),
        "gcor_max": ("gcor", "max"),
    }
    for metric in metrics:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_sd"] = (metric, "std")
        agg_spec[f"{metric}_min"] = (metric, "min")
        agg_spec[f"{metric}_max"] = (metric, "max")

    return (
        merged.groupby(["drug", "condition", "hemisphere"], as_index=False, dropna=False)
        .agg(**agg_spec)
        .sort_values(["drug", "condition", "hemisphere"])
    )


def plot_scatter_panels(merged: pd.DataFrame, corr: pd.DataFrame, metrics: list[str], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_corr = corr[corr["grouping"] == "condition_hemisphere"].copy()

    sns.set_theme(style="whitegrid")
    for metric in metrics:
        g = sns.lmplot(
            data=merged,
            x=metric,
            y="gcor",
            row="condition",
            col="hemisphere",
            hue="condition",
            palette="Set2",
            height=3.4,
            aspect=1.15,
            scatter_kws={"s": 42, "alpha": 0.85},
            line_kws={"linewidth": 1.5},
            ci=95,
            facet_kws={"sharex": False, "sharey": True},
        )
        g.set_axis_labels(metric, "GCOR")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        for (condition, hemisphere), ax in g.axes_dict.items():
            hit = plot_corr[
                (plot_corr["condition"].astype(str) == condition)
                & (plot_corr["hemisphere"].astype(str) == hemisphere)
                & (plot_corr["metric"] == metric)
            ]
            if hit.empty:
                continue
            row = hit.iloc[0]
            ax.text(
                0.03,
                0.96,
                f"n={int(row['n'])}, r={row['pearson_r']:.3f}, p={row['pearson_p']:.3g}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
        g.figure.suptitle(f"{metric} vs GCOR", y=1.02)
        g.savefig(fig_dir / f"{metric}_vs_gcor_condition_hemisphere.png", dpi=240, bbox_inches="tight")
        plt.close(g.figure)


def write_report(out_dir: Path, merged: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    main = corr[corr["grouping"] == "condition_hemisphere"].copy()
    top = corr[np.isfinite(corr["pearson_r"])].copy()
    top["abs_pearson_r"] = top["pearson_r"].abs()
    top = top.sort_values(["abs_pearson_r", "n"], ascending=[False, False]).head(30)

    lines = [
        "# Pattern Metrics vs GCOR",
        "",
        "## Input Summary",
        "",
        f"- Merged rows: {len(merged)}",
        f"- Subjects: {merged['subid'].nunique()}",
        f"- Conditions: {', '.join(map(str, sorted(merged['condition'].dropna().unique())))}",
        f"- Hemispheres: {', '.join(map(str, sorted(merged['hemisphere'].dropna().unique())))}",
        "",
        "## Descriptive Statistics",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Main Correlations: Condition x Hemisphere",
        "",
        main[
            [
                "drug",
                "condition",
                "hemisphere",
                "metric",
                "n",
                "pearson_r",
                "pearson_p",
                "pearson_p_fdr_bh",
                "spearman_rho",
                "spearman_p",
                "kendall_tau",
                "kendall_p",
                "slope",
                "r_squared",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Strongest Pearson Correlations",
        "",
        top[
            [
                "grouping",
                "drug",
                "condition",
                "hemisphere",
                "metric",
                "n",
                "pearson_r",
                "pearson_p",
                "pearson_p_fdr_bh",
                "spearman_rho",
                "spearman_p",
                "slope",
                "r_squared",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Merged Data Preview",
        "",
        merged.head(30).to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pattern = load_pattern(args.pattern_csv, pattern_drug=args.pattern_drug, metrics=args.metrics)
    gcor = load_gcor(args.gcor_csv)
    merged = pattern.merge(gcor, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No rows matched between pattern metrics and GCOR after normalization.")

    summary = summary_table(merged, args.metrics)
    corr = correlation_table(merged, args.metrics, min_n=args.min_n)

    pattern.to_csv(args.out_dir / "pattern_normalized_by_subject.csv", index=False)
    gcor.to_csv(args.out_dir / "gcor_normalized_by_subject.csv", index=False)
    merged.to_csv(args.out_dir / "pattern_gcor_merged_by_subject.csv", index=False)
    summary.to_csv(args.out_dir / "descriptive_stats_by_condition_hemisphere.csv", index=False)
    corr.to_csv(args.out_dir / "metric_gcor_correlations.csv", index=False)

    if not args.no_plots:
        plot_scatter_panels(merged, corr, args.metrics, args.out_dir)
    write_report(args.out_dir, merged, summary, corr)

    metadata = {
        "pattern_csv": str(args.pattern_csv),
        "gcor_csv": str(args.gcor_csv),
        "out_dir": str(args.out_dir),
        "pattern_drug": args.pattern_drug,
        "metrics": args.metrics,
        "key_columns": KEY_COLUMNS,
        "groupings": {name: columns for name, columns in GROUPING_SPECS},
        "min_n": int(args.min_n),
        "n_pattern_rows": int(len(pattern)),
        "n_gcor_rows": int(len(gcor)),
        "n_merged_rows": int(len(merged)),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nMerged data")
    print(merged.head(args.show_rows).to_string(index=False))
    print("\nCondition x hemisphere correlations")
    main_corr = corr[corr["grouping"] == "condition_hemisphere"]
    print(
        main_corr[
            [
                "drug",
                "condition",
                "hemisphere",
                "metric",
                "n",
                "pearson_r",
                "pearson_p",
                "pearson_p_fdr_bh",
                "spearman_rho",
                "spearman_p",
                "slope",
                "r_squared",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote metric-GCOR analysis to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
