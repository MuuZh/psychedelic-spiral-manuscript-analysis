#!/usr/bin/env python
"""
Correlate network within-FC deltas with network-wise spiral metric deltas.

Inputs:
    - wb_pconn_within_between network_subject_deltas.csv files
    - network_spiral_metrics paired_deltas_long.csv

The analysis joins rows by subject, hemisphere, and network. For each
hemisphere x network x metric, it computes Pearson correlation between:

    x = within-network FC delta (Drug - PCB)
    y = spiral metric delta (Drug - PCB)

Figures include per-network regression panels with 95% CI bands and summary
heatmaps of correlation coefficients.
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


NETWORK_ORDER_7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_ORDER_17 = [
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicB",
    "LimbicA",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
]
METRIC_LABELS = {
    "spiral_count_per_frame": "Spiral count / frame",
    "spiral_count_per_network_px": "Spiral count / network px",
    "mean_spiral_size": "Mean spiral size",
    "mean_network_footprint_px": "Mean footprint in network (px)",
    "mean_network_footprint_fraction": "Mean footprint fraction",
    "mean_spiral_power": "Mean spiral power",
    "mean_expansion_radius": "Mean expansion radius",
    "mean_cos2_alignment": "Mean cos(2theta)",
    "weighted_mean_cos2_alignment": "Weighted mean cos(2theta)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate within-network FC deltas with network spiral metric deltas."
    )
    parser.add_argument(
        "--wb-root",
        default=Path("analysis_outputs/phase_fc_group/wb_pconn_within_between/DMT_DMT_minus_DMT_PCB"),
        type=Path,
        help="Comparison directory containing left/right/network_subject_deltas.csv.",
    )
    parser.add_argument(
        "--spiral-deltas",
        default=Path("analysis_outputs/phase_fc_group/network_spiral_metrics/paired_deltas_long.csv"),
        type=Path,
        help="paired_deltas_long.csv from network_spiral_metrics.py.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/network_spiral_fc_correlation"),
        type=Path,
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument("--min-subjects", default=5, type=int)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric subset. Defaults to all metrics present in the spiral delta table.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def infer_network_order(networks: pd.Series | list[str]) -> list[str]:
    values = [str(v) for v in networks if pd.notna(v)]
    available = set(values)
    for known in (NETWORK_ORDER_7, NETWORK_ORDER_17):
        ordered = [network for network in known if network in available]
        if ordered:
            extras = [network for network in values if network not in set(known)]
            return ordered + list(dict.fromkeys(extras))
    return list(dict.fromkeys(values))


def p_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def load_within_fc(wb_root: Path, hemispheres: list[str]) -> pd.DataFrame:
    frames = []
    for hemi in hemispheres:
        path = wb_root / hemi / "network_subject_deltas.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"network_a", "network_b", "type", "delta", "subid", "hemisphere"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        within = df[
            (df["type"].astype(str).str.lower() == "within")
            & (df["network_a"].astype(str) == df["network_b"].astype(str))
        ].copy()
        within = within.rename(columns={"network_a": "network", "delta": "fc_within_delta"})
        within["subid"] = within["subid"].astype(str)
        frames.append(within[["subid", "hemisphere", "network", "fc_within_delta"]])
    if not frames:
        raise RuntimeError(f"No within-network FC delta files found under {wb_root}")
    return pd.concat(frames, ignore_index=True)


def load_spiral_deltas(path: Path, hemispheres: list[str], metrics: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subid", "hemisphere", "network", "metric", "delta_drug_minus_pcb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[df["hemisphere"].isin(hemispheres)].copy()
    if metrics:
        df = df[df["metric"].isin(metrics)].copy()
    df["subid"] = df["subid"].astype(str)
    return df[["subid", "hemisphere", "network", "metric", "delta_drug_minus_pcb"]]


def build_joined(fc_df: pd.DataFrame, spiral_df: pd.DataFrame) -> pd.DataFrame:
    joined = spiral_df.merge(fc_df, on=["subid", "hemisphere", "network"], how="inner")
    joined = joined.rename(columns={"delta_drug_minus_pcb": "spiral_metric_delta"})
    joined["metric_label"] = joined["metric"].map(metric_label)
    return joined


def corr_rows(joined: pd.DataFrame, min_subjects: int) -> pd.DataFrame:
    rows = []
    for keys, sub in joined.groupby(["hemisphere", "network", "metric"], sort=False):
        hemi, network, metric = keys
        work = sub[["fc_within_delta", "spiral_metric_delta"]].apply(pd.to_numeric, errors="coerce").dropna()
        n = int(len(work))
        if n >= min_subjects and work["fc_within_delta"].std(ddof=1) > 0 and work["spiral_metric_delta"].std(ddof=1) > 0:
            r, p = stats.pearsonr(work["fc_within_delta"], work["spiral_metric_delta"])
            slope, intercept, _, _, slope_stderr = stats.linregress(
                work["fc_within_delta"], work["spiral_metric_delta"]
            )
            r = float(r)
            p = float(p)
            slope = float(slope)
            intercept = float(intercept)
            slope_stderr = float(slope_stderr)
        else:
            r = p = slope = intercept = slope_stderr = math.nan
        rows.append(
            {
                "hemisphere": hemi,
                "network": network,
                "metric": metric,
                "metric_label": metric_label(metric),
                "n": n,
                "pearson_r": r,
                "p": p,
                "slope": slope,
                "intercept": intercept,
                "slope_stderr": slope_stderr,
                "stars": p_to_stars(p),
            }
        )
    return pd.DataFrame(rows)


def save_per_network_plots(
    joined: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
    network_order: list[str],
    metrics: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for hemi in sorted(joined["hemisphere"].dropna().unique()):
        hemi_df = joined[joined["hemisphere"] == hemi]
        for network in network_order:
            net_df = hemi_df[hemi_df["network"] == network]
            if net_df.empty:
                continue
            ncols = 3
            nrows = int(math.ceil(len(metrics) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
            for ax, metric in zip(axes.flat, metrics):
                mdf = net_df[net_df["metric"] == metric].dropna(
                    subset=["fc_within_delta", "spiral_metric_delta"]
                )
                if len(mdf) >= 2:
                    sns.regplot(
                        data=mdf,
                        x="fc_within_delta",
                        y="spiral_metric_delta",
                        ax=ax,
                        ci=95,
                        scatter_kws={"s": 34, "alpha": 0.78, "edgecolor": "white", "linewidths": 0.4},
                        line_kws={"linewidth": 1.8, "color": "#222222"},
                        color="#4c72b0",
                    )
                stat = stats_df[
                    (stats_df["hemisphere"] == hemi)
                    & (stats_df["network"] == network)
                    & (stats_df["metric"] == metric)
                ]
                if not stat.empty:
                    row = stat.iloc[0]
                    text = f"r={row.pearson_r:.2f}, p={row.p:.3g}{row.stars}" if pd.notna(row.pearson_r) else "r=NA"
                    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=10)
                ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.55)
                ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.55)
                ax.set_title(metric_label(metric), pad=9)
                ax.set_xlabel("Within-network FC delta")
                ax.set_ylabel("Spiral metric delta")
            for ax in axes.flat[len(metrics) :]:
                ax.axis("off")
            fig.suptitle(f"Within-FC vs spiral deltas | {network} | {hemi}", y=1.015)
            fig.tight_layout()
            fig.savefig(out_dir / f"correlation_{hemi}_{network}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def save_summary_heatmaps(
    stats_df: pd.DataFrame,
    out_dir: Path,
    network_order: list[str],
    metrics: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for hemi in sorted(stats_df["hemisphere"].dropna().unique()):
        sub = stats_df[stats_df["hemisphere"] == hemi]
        mat = pd.DataFrame(index=[metric_label(m) for m in metrics], columns=network_order, dtype=float)
        annot = pd.DataFrame("", index=mat.index, columns=mat.columns)
        for row in sub.itertuples(index=False):
            if row.metric not in metrics or row.network not in network_order:
                continue
            label = metric_label(row.metric)
            mat.loc[label, row.network] = row.pearson_r
            if np.isfinite(row.pearson_r):
                annot.loc[label, row.network] = f"{row.pearson_r:.2f}{row.stars}"
        plt.figure(figsize=(10.5, max(5.0, 0.55 * len(metrics))))
        sns.heatmap(
            mat,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            annot=annot,
            fmt="",
            cbar_kws={"label": "Pearson r"},
        )
        plt.title(f"Within-FC delta vs spiral metric delta correlations ({hemi})")
        plt.tight_layout()
        plt.savefig(out_dir / f"correlation_summary_heatmap_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_summary_pointplot(
    stats_df: pd.DataFrame,
    out_dir: Path,
    network_order: list[str],
    metrics: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for hemi in sorted(stats_df["hemisphere"].dropna().unique()):
        sub = stats_df[(stats_df["hemisphere"] == hemi) & (stats_df["metric"].isin(metrics))].copy()
        if sub.empty:
            continue
        sub["network"] = pd.Categorical(sub["network"], categories=network_order, ordered=True)
        sub["metric_label"] = sub["metric"].map(metric_label)
        fig, ax = plt.subplots(figsize=(12, 6.2))
        sns.lineplot(
            data=sub.sort_values(["metric", "network"]),
            x="network",
            y="pearson_r",
            hue="metric_label",
            marker="o",
            linewidth=1.4,
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel("")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Correlation summary by network ({hemi})", pad=12)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.82])
        fig.savefig(out_dir / f"correlation_summary_lines_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    hemispheres = ["left", "right"] if args.hemisphere == "both" else [args.hemisphere]

    fc_df = load_within_fc(args.wb_root, hemispheres)
    spiral_df = load_spiral_deltas(args.spiral_deltas, hemispheres, args.metrics)
    joined = build_joined(fc_df, spiral_df)
    if joined.empty:
        raise RuntimeError("No rows after joining FC and spiral deltas.")

    network_order = infer_network_order(joined["network"])
    metrics = args.metrics or [m for m in METRIC_LABELS if m in set(joined["metric"])]
    metrics += [m for m in joined["metric"].drop_duplicates() if m not in metrics]
    stats_df = corr_rows(joined, min_subjects=args.min_subjects)

    joined.to_csv(args.out_dir / "fc_spiral_delta_joined.csv", index=False)
    stats_df.to_csv(args.out_dir / "fc_spiral_delta_correlations.csv", index=False)

    if not args.no_plots:
        fig_dir = args.out_dir / "figures"
        save_per_network_plots(joined, stats_df, fig_dir / "per_network", network_order, metrics)
        save_summary_heatmaps(stats_df, fig_dir, network_order, metrics)
        save_summary_pointplot(stats_df, fig_dir, network_order, metrics)

    metadata = {
        "wb_root": str(args.wb_root),
        "spiral_deltas": str(args.spiral_deltas),
        "hemispheres": hemispheres,
        "min_subjects": int(args.min_subjects),
        "network_order": network_order,
        "metrics": metrics,
        "n_joined_rows": int(len(joined)),
        "n_stats_rows": int(len(stats_df)),
        "x": "within-network FC delta (Drug - PCB)",
        "y": "network spiral metric delta (Drug - PCB)",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote FC/spiral correlation analysis to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
