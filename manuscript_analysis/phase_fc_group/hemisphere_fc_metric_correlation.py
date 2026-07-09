#!/usr/bin/env python
"""
Correlate hemisphere-level FC summaries with hemisphere-level pattern metrics.

For each drug comparison and hemisphere, network FC rows are collapsed to:

    - within-network FC mean
    - between-network FC mean

Both unweighted network-row means and edge-count-weighted means are saved. The
script correlates FC with selected all_metrics per-subject measures for:

    - delta: Drug - PCB
    - drug: drug condition value
    - placebo: PCB condition value
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DEFAULT_ATLAS = Path("analysis_outputs/phase_fc_batch/atlas_metadata/Schaefer2018_400Parcels_7Networks_parcels.csv")
DEFAULT_OUT_DIR = Path("analysis_outputs/phase_fc_group/hemisphere_fc_metric_correlation_v2")
PATTERN_STATS_METRICS = ["pattern_count", "mean_size", "mean_duration", "mean_power", "pattern_count_per_frame"]
ANGLE_METRICS = ["angle_cos_abs_mean"]
DYNAMICS_METRICS = ["phase_angular_velocity_abs_mean_rad_per_frame"]
METRIC_SOURCES = {
    "pattern_stats": PATTERN_STATS_METRICS,
    "angle_diff_abs_cos": ANGLE_METRICS,
    "pattern_dynamics": DYNAMICS_METRICS,
}
VALUE_KINDS = ["delta", "drug", "placebo"]
FC_TYPES = ["within", "between"]
AVERAGE_METHODS = ["unweighted", "weighted"]
PALETTE = {"DMT": "#4c72b0", "LSD": "#dd8452"}


@dataclass(frozen=True)
class ComparisonSpec:
    drug: str
    fc_root: Path
    metrics_root: Path
    drug_group: str
    placebo_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate hemisphere-level within/between FC summaries with all_metrics outputs."
    )
    parser.add_argument(
        "--dmt-fc-root",
        default=Path("analysis_outputs/phase_fc_group/wb_pconn_within_between"),
        type=Path,
        help="Root containing DMT_DMT_minus_DMT_PCB/left|right/network_subject_deltas.csv.",
    )
    parser.add_argument(
        "--lsd-fc-root",
        default=Path("analysis_outputs/phase_fc_group/wb_pconn_within_between_lsd"),
        type=Path,
        help="Root containing LSD_LSD_minus_LSD_PCB/left|right/network_subject_deltas.csv.",
    )
    parser.add_argument("--dmt-metrics-root", default=Path("analysis_outputs/all_metrics/dmt-run"), type=Path)
    parser.add_argument("--lsd-metrics-root", default=Path("analysis_outputs/all_metrics/lsd-run"), type=Path)
    parser.add_argument("--atlas-parcels", default=DEFAULT_ATLAS, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument("--min-n", default=5, type=int)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def normalize_subid(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("S"):
        text = text[1:]
    if text.isdigit():
        return f"{int(text):02d}"
    return text


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


def comparison_dir(root: Path) -> Path:
    dirs = sorted(path for path in root.iterdir() if path.is_dir() and "_minus_" in path.name)
    if len(dirs) != 1:
        raise RuntimeError(f"Expected exactly one comparison directory under {root}, found {len(dirs)}")
    return dirs[0]


def load_hemi_parcels(path: Path, hemisphere: str) -> pd.DataFrame:
    parcels = pd.read_csv(path)
    hemi_code = {"left": "LH", "right": "RH"}[hemisphere]
    out = parcels[parcels["hemi"].astype(str) == hemi_code].copy()
    if out.empty:
        raise RuntimeError(f"No {hemi_code} parcels found in {path}")
    return out


def fc_weights(atlas_parcels: Path, hemisphere: str) -> pd.DataFrame:
    parcels = load_hemi_parcels(atlas_parcels, hemisphere)
    counts = parcels.groupby("network", sort=False).size().to_dict()
    networks = list(counts)
    rows = []
    for i, net_a in enumerate(networks):
        n_a = int(counts[net_a])
        rows.append(
            {
                "network_a": net_a,
                "network_b": net_a,
                "type": "within",
                "edge_weight": n_a * (n_a - 1) / 2,
            }
        )
        for net_b in networks[i + 1 :]:
            rows.append(
                {
                    "network_a": net_a,
                    "network_b": net_b,
                    "type": "between",
                    "edge_weight": n_a * int(counts[net_b]),
                }
            )
    return pd.DataFrame(rows)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return math.nan
    return float(np.average(v[ok], weights=w[ok]))


def load_fc_summary(spec: ComparisonSpec, hemispheres: list[str], atlas_parcels: Path) -> pd.DataFrame:
    comp_dir = comparison_dir(spec.fc_root)
    rows = []
    for hemi in hemispheres:
        path = comp_dir / hemi / "network_subject_deltas.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"network_a", "network_b", "type", "drug_value", "placebo_value", "delta", "subid", "hemisphere"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        work = df.copy()
        work["subid"] = work["subid"].map(normalize_subid)
        work["hemisphere"] = work["hemisphere"].astype(str).str.lower()
        work["type"] = work["type"].astype(str).str.lower()
        weights = fc_weights(atlas_parcels, hemi)
        work = work.merge(weights, on=["network_a", "network_b", "type"], how="left")

        for (subid, hemi_name, fc_type), group in work.groupby(["subid", "hemisphere", "type"], sort=False):
            if fc_type not in FC_TYPES:
                continue
            for avg_method in AVERAGE_METHODS:
                if avg_method == "unweighted":
                    drug_value = float(pd.to_numeric(group["drug_value"], errors="coerce").mean())
                    placebo_value = float(pd.to_numeric(group["placebo_value"], errors="coerce").mean())
                    delta = float(pd.to_numeric(group["delta"], errors="coerce").mean())
                else:
                    drug_value = weighted_mean(group["drug_value"], group["edge_weight"])
                    placebo_value = weighted_mean(group["placebo_value"], group["edge_weight"])
                    delta = weighted_mean(group["delta"], group["edge_weight"])
                rows.append(
                    {
                        "drug": spec.drug,
                        "comparison": comp_dir.name,
                        "subid": subid,
                        "hemisphere": hemi_name,
                        "fc_type": fc_type,
                        "average_method": avg_method,
                        "fc_drug_value": drug_value,
                        "fc_placebo_value": placebo_value,
                        "fc_delta": delta,
                        "n_network_rows": int(len(group)),
                        "edge_weight_sum": float(pd.to_numeric(group["edge_weight"], errors="coerce").sum()),
                    }
                )
    if not rows:
        raise RuntimeError(f"No FC rows loaded from {spec.fc_root}")
    return pd.DataFrame(rows)


def load_metric_source(root: Path, source: str, metrics: list[str]) -> pd.DataFrame:
    path = root / source / "per_subject.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"group", "subid", "hemisphere", *metrics}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = df[["group", "subid", "hemisphere", *metrics]].copy()
    out["subid"] = out["subid"].map(normalize_subid)
    out["hemisphere"] = out["hemisphere"].astype(str).str.lower()
    out["group"] = out["group"].astype(str).str.strip()
    for metric in metrics:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out.melt(
        id_vars=["group", "subid", "hemisphere"],
        value_vars=metrics,
        var_name="metric",
        value_name="metric_value",
    ).assign(metric_source=source)


def load_metric_summary(spec: ComparisonSpec, hemispheres: list[str]) -> pd.DataFrame:
    frames = [
        load_metric_source(spec.metrics_root, source, metrics)
        for source, metrics in METRIC_SOURCES.items()
    ]
    long = pd.concat(frames, ignore_index=True)
    long = long[long["hemisphere"].isin(hemispheres)].copy()
    drug = long[long["group"] == spec.drug_group].rename(columns={"metric_value": "metric_drug_value"})
    pcb = long[long["group"] == spec.placebo_group].rename(columns={"metric_value": "metric_placebo_value"})
    merged = drug.merge(
        pcb[["subid", "hemisphere", "metric", "metric_source", "metric_placebo_value"]],
        on=["subid", "hemisphere", "metric", "metric_source"],
        how="inner",
    )
    merged["drug"] = spec.drug
    merged["metric_delta"] = merged["metric_drug_value"] - merged["metric_placebo_value"]
    return merged[
        [
            "drug",
            "subid",
            "hemisphere",
            "metric_source",
            "metric",
            "metric_drug_value",
            "metric_placebo_value",
            "metric_delta",
        ]
    ]


def build_joined(fc_df: pd.DataFrame, metric_df: pd.DataFrame) -> pd.DataFrame:
    return fc_df.merge(metric_df, on=["drug", "subid", "hemisphere"], how="inner")


def fisher_r_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return (math.nan, math.nan)
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)))


def corr_one(group: pd.DataFrame, x_col: str, y_col: str, min_n: int) -> dict[str, float]:
    work = group[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(work))
    out = {
        "n": n,
        "pearson_r": math.nan,
        "r_squared": math.nan,
        "p": math.nan,
        "r_ci95_low": math.nan,
        "r_ci95_high": math.nan,
        "slope": math.nan,
        "intercept": math.nan,
        "slope_stderr": math.nan,
        "slope_ci95_low": math.nan,
        "slope_ci95_high": math.nan,
    }
    if n < min_n or work[x_col].std(ddof=1) <= 0 or work[y_col].std(ddof=1) <= 0:
        return out

    x = work[x_col].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)
    r, p = stats.pearsonr(x, y)
    lr = stats.linregress(x, y)
    r_low, r_high = fisher_r_ci(float(r), n)
    if n > 2 and np.isfinite(lr.stderr):
        crit = stats.t.ppf(0.975, df=n - 2)
        slope_low = float(lr.slope - crit * lr.stderr)
        slope_high = float(lr.slope + crit * lr.stderr)
    else:
        slope_low = slope_high = math.nan
    out.update(
        {
            "pearson_r": float(r),
            "r_squared": float(r * r),
            "p": float(p),
            "r_ci95_low": r_low,
            "r_ci95_high": r_high,
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "slope_stderr": float(lr.stderr),
            "slope_ci95_low": slope_low,
            "slope_ci95_high": slope_high,
        }
    )
    return out


def correlation_table(joined: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    key_cols = ["drug", "hemisphere", "fc_type", "average_method", "metric_source", "metric"]
    for value_kind in VALUE_KINDS:
        x_col = f"fc_{'delta' if value_kind == 'delta' else value_kind + '_value'}"
        y_col = f"metric_{'delta' if value_kind == 'delta' else value_kind + '_value'}"
        for keys, group in joined.groupby(key_cols, sort=False):
            row = dict(zip(key_cols, keys))
            row["value_kind"] = value_kind
            row["x_col"] = x_col
            row["y_col"] = y_col
            row.update(corr_one(group, x_col, y_col, min_n=min_n))
            row["stars"] = p_to_stars(row["p"])
            rows.append(row)
    return pd.DataFrame(rows)


def metric_order() -> list[str]:
    out = []
    for metrics in METRIC_SOURCES.values():
        out.extend(metrics)
    return out


def plot_grid(
    joined: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_path: Path,
    hemisphere: str,
    fc_type: str,
    average_method: str,
    value_kind: str,
    drug_filter: str | None = None,
) -> None:
    metrics = metric_order()
    sub = joined[
        (joined["hemisphere"] == hemisphere)
        & (joined["fc_type"] == fc_type)
        & (joined["average_method"] == average_method)
    ].copy()
    if drug_filter is not None:
        sub = sub[sub["drug"] == drug_filter].copy()
    if sub.empty:
        return
    x_col = f"fc_{'delta' if value_kind == 'delta' else value_kind + '_value'}"
    y_col = f"metric_{'delta' if value_kind == 'delta' else value_kind + '_value'}"

    ncols = 3
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        mdf = sub[sub["metric"] == metric].copy()
        for drug in sorted(mdf["drug"].dropna().unique()):
            ddf = mdf[mdf["drug"] == drug].dropna(subset=[x_col, y_col])
            if len(ddf) >= 2:
                sns.regplot(
                    data=ddf,
                    x=x_col,
                    y=y_col,
                    ax=ax,
                    ci=95,
                    color=PALETTE.get(drug),
                    scatter_kws={"s": 32, "alpha": 0.72, "edgecolor": "white", "linewidths": 0.4},
                    line_kws={"linewidth": 1.8, "label": drug},
                )
        stat = stats_df[
            (stats_df["hemisphere"] == hemisphere)
            & (stats_df["fc_type"] == fc_type)
            & (stats_df["average_method"] == average_method)
            & (stats_df["value_kind"] == value_kind)
            & (stats_df["metric"] == metric)
        ]
        if drug_filter is not None:
            stat = stat[stat["drug"] == drug_filter]
        labels = []
        for row in stat.sort_values("drug").itertuples(index=False):
            if np.isfinite(row.pearson_r):
                labels.append(f"{row.drug}: r={row.pearson_r:.2f}, p={row.p:.3g}{row.stars}")
        if labels:
            ax.text(0.04, 0.96, "\n".join(labels), transform=ax.transAxes, ha="left", va="top", fontsize=9)
        x_vals = pd.to_numeric(mdf[x_col], errors="coerce")
        y_vals = pd.to_numeric(mdf[y_col], errors="coerce")
        if value_kind == "delta" or (x_vals.min(skipna=True) <= 0 <= x_vals.max(skipna=True)):
            ax.axvline(0, color="gray", linewidth=0.8, alpha=0.55)
        if value_kind == "delta" or (y_vals.min(skipna=True) <= 0 <= y_vals.max(skipna=True)):
            ax.axhline(0, color="gray", linewidth=0.8, alpha=0.55)
        ax.set_title(metric, pad=8)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    for ax in axes.flat[len(metrics) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    drug_title = f" | {drug_filter}" if drug_filter is not None else ""
    fig.suptitle(
        f"{value_kind}: {fc_type} FC vs metrics | {hemisphere} | {average_method}{drug_title}",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(stats_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = metric_order()
    for value_kind in VALUE_KINDS:
        for hemisphere in sorted(stats_df["hemisphere"].dropna().unique()):
            for fc_type in FC_TYPES:
                for avg_method in AVERAGE_METHODS:
                    sub = stats_df[
                        (stats_df["value_kind"] == value_kind)
                        & (stats_df["hemisphere"] == hemisphere)
                        & (stats_df["fc_type"] == fc_type)
                        & (stats_df["average_method"] == avg_method)
                    ]
                    if sub.empty:
                        continue
                    mat = pd.DataFrame(index=metrics, columns=sorted(sub["drug"].unique()), dtype=float)
                    annot = pd.DataFrame("", index=mat.index, columns=mat.columns)
                    for row in sub.itertuples(index=False):
                        mat.loc[row.metric, row.drug] = row.pearson_r
                        if np.isfinite(row.pearson_r):
                            annot.loc[row.metric, row.drug] = f"{row.pearson_r:.2f}{row.stars}"
                    fig, ax = plt.subplots(figsize=(7.8, max(5.4, 0.58 * len(metrics))))
                    sns.heatmap(
                        mat,
                        cmap="coolwarm",
                        center=0,
                        vmin=-1,
                        vmax=1,
                        annot=annot,
                        fmt="",
                        cbar_kws={"label": "Pearson r"},
                        ax=ax,
                    )
                    ax.set_title(f"{value_kind} | {hemisphere} | {fc_type} | {avg_method}", pad=10)
                    ax.tick_params(axis="y", labelsize=9)
                    ax.tick_params(axis="x", labelsize=10)
                    fig.subplots_adjust(left=0.48, right=0.96, top=0.91, bottom=0.12)
                    fig.savefig(
                        out_dir / f"heatmap_{value_kind}_{hemisphere}_{fc_type}_{avg_method}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(fig)


def save_plots(joined: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    scatter_dir = out_dir / "scatter_regression"
    for value_kind in VALUE_KINDS:
        for hemisphere in sorted(joined["hemisphere"].dropna().unique()):
            for fc_type in FC_TYPES:
                for avg_method in AVERAGE_METHODS:
                    if value_kind == "delta":
                        plot_grid(
                            joined,
                            stats_df,
                            scatter_dir / "delta" / f"{value_kind}_{hemisphere}_{fc_type}_{avg_method}.png",
                            hemisphere=hemisphere,
                            fc_type=fc_type,
                            average_method=avg_method,
                            value_kind=value_kind,
                        )
                    else:
                        for drug in sorted(joined["drug"].dropna().unique()):
                            plot_grid(
                                joined,
                                stats_df,
                                scatter_dir / value_kind / f"{drug}_{value_kind}_{hemisphere}_{fc_type}_{avg_method}.png",
                                hemisphere=hemisphere,
                                fc_type=fc_type,
                                average_method=avg_method,
                                value_kind=value_kind,
                                drug_filter=drug,
                            )
    plot_heatmaps(stats_df, out_dir / "heatmaps")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hemispheres = ["left", "right"] if args.hemisphere == "both" else [args.hemisphere]
    specs = [
        ComparisonSpec("DMT", args.dmt_fc_root, args.dmt_metrics_root, "DMT", "PCB"),
        ComparisonSpec("LSD", args.lsd_fc_root, args.lsd_metrics_root, "LSD", "PCB"),
    ]

    fc_frames = []
    metric_frames = []
    for spec in specs:
        fc_frames.append(load_fc_summary(spec, hemispheres, args.atlas_parcels))
        metric_frames.append(load_metric_summary(spec, hemispheres))
    fc_df = pd.concat(fc_frames, ignore_index=True)
    metric_df = pd.concat(metric_frames, ignore_index=True)
    joined = build_joined(fc_df, metric_df)
    if joined.empty:
        raise RuntimeError("No rows after joining FC summaries and metric summaries.")
    stats_df = correlation_table(joined, min_n=args.min_n)

    fc_df.to_csv(args.out_dir / "hemisphere_fc_summary.csv", index=False)
    metric_df.to_csv(args.out_dir / "hemisphere_metric_summary.csv", index=False)
    joined.to_csv(args.out_dir / "hemisphere_fc_metric_joined.csv", index=False)
    stats_df.to_csv(args.out_dir / "hemisphere_fc_metric_correlations.csv", index=False)

    if not args.no_plots:
        save_plots(joined, stats_df, args.out_dir / "figures")

    metadata = {
        "fc_roots": {spec.drug: str(spec.fc_root) for spec in specs},
        "metrics_roots": {spec.drug: str(spec.metrics_root) for spec in specs},
        "atlas_parcels": str(args.atlas_parcels),
        "hemispheres": hemispheres,
        "min_n": int(args.min_n),
        "metric_sources": METRIC_SOURCES,
        "fc_types": FC_TYPES,
        "average_methods": AVERAGE_METHODS,
        "value_kinds": VALUE_KINDS,
        "n_fc_rows": int(len(fc_df)),
        "n_metric_rows": int(len(metric_df)),
        "n_joined_rows": int(len(joined)),
        "n_stats_rows": int(len(stats_df)),
        "weighted_fc_note": "within weight=n*(n-1)/2; between weight=n_a*n_b, derived from atlas parcel counts.",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote hemisphere FC/metric correlation analysis to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
