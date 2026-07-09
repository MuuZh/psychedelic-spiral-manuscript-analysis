#!/usr/bin/env python
"""
Correlate per-subject pattern metrics with GCOR.

Both inputs are normalized to:

    drug, condition, subid, hemisphere

The pattern table stores condition in ``group`` and has no explicit ``drug``
column, so this script defaults to drug=DMT for that input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DEFAULT_PATTERN_CSV = Path(
    r"<analysis_outputs>\all_metrics\dmt-run\pattern_stats\per_subject.csv"
)
DEFAULT_GCOR_CSV = Path(
    r"<analysis_outputs>\gcor_batch_DMT\all_gcor_by_subject.csv"
)
DEFAULT_NETWORK_DELTA_ROOT = Path(
    r"<analysis_outputs>\phase_fc_group\wb_pconn_within_between\DMT_DMT_minus_DMT_PCB"
)
METRICS = ["mean_size", "mean_duration", "mean_power", "pattern_count_per_frame"]
KEY_COLUMNS = ["drug", "condition", "subid", "hemisphere"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate pattern metrics with GCOR.")
    parser.add_argument("--pattern-csv", default=DEFAULT_PATTERN_CSV, type=Path)
    parser.add_argument("--gcor-csv", default=DEFAULT_GCOR_CSV, type=Path)
    parser.add_argument(
        "--network-delta-root",
        default=DEFAULT_NETWORK_DELTA_ROOT,
        type=Path,
        help="Root containing left/right network_subject_deltas.csv from wb_pconn_within_between.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/pattern_gcor_correlations"),
        type=Path,
        help="Output directory for merged tables, correlation tables, and plots.",
    )
    parser.add_argument(
        "--pattern-drug",
        default="DMT",
        help="Drug label to assign to the pattern table because it only has a group column.",
    )
    parser.add_argument("--min-n", default=3, type=int, help="Minimum rows required per correlation.")
    parser.add_argument(
        "--show-rows",
        default=20,
        type=int,
        help="Number of merged rows to print to the console.",
    )
    return parser.parse_args()


def normalize_subid(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("S"):
        text = text[1:]
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def load_pattern(path: Path, pattern_drug: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"group", "subid", "hemisphere", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = df.rename(columns={"group": "condition"}).copy()
    out["drug"] = pattern_drug
    out["condition"] = out["condition"].astype(str).str.strip()
    out["subid"] = out["subid"].map(normalize_subid)
    out["hemisphere"] = out["hemisphere"].astype(str).str.strip().str.lower()
    for metric in METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")

    return (
        out[KEY_COLUMNS + METRICS]
        .groupby(KEY_COLUMNS, as_index=False, dropna=False)
        .mean(numeric_only=True)
        .sort_values(KEY_COLUMNS)
    )


def load_gcor(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"drug", "condition", "id", "hemisphere", "gcor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = df.rename(columns={"id": "subid"}).copy()
    out["drug"] = out["drug"].astype(str).str.strip()
    out["condition"] = out["condition"].astype(str).str.strip()
    out["subid"] = out["subid"].map(normalize_subid)
    out["hemisphere"] = out["hemisphere"].astype(str).str.strip().str.lower()
    out["gcor"] = pd.to_numeric(out["gcor"], errors="coerce")

    return (
        out[KEY_COLUMNS + ["gcor"]]
        .groupby(KEY_COLUMNS, as_index=False, dropna=False)
        .mean(numeric_only=True)
        .sort_values(KEY_COLUMNS)
    )


def pearson_row(values: pd.DataFrame, group_name: str, metric: str, min_n: int) -> dict[str, object]:
    sub = values[[metric, "gcor"]].dropna()
    n = int(len(sub))
    result: dict[str, object] = {
        "grouping": group_name,
        "metric": metric,
        "n": n,
        "metric_mean": float(sub[metric].mean()) if n else np.nan,
        "gcor_mean": float(sub["gcor"].mean()) if n else np.nan,
        "r": np.nan,
        "p": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
    }
    if n < min_n or sub[metric].nunique(dropna=True) < 2 or sub["gcor"].nunique(dropna=True) < 2:
        return result

    r, p = stats.pearsonr(sub[metric].to_numpy(dtype=float), sub["gcor"].to_numpy(dtype=float))
    lr = stats.linregress(sub[metric].to_numpy(dtype=float), sub["gcor"].to_numpy(dtype=float))
    result.update(
        {
            "r": float(r),
            "p": float(p),
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "r_squared": float(r**2),
        }
    )
    return result


def pearson_xy(values: pd.DataFrame, x_col: str, y_col: str, min_n: int) -> dict[str, object]:
    sub = values[[x_col, y_col]].dropna()
    n = int(len(sub))
    result: dict[str, object] = {
        "n": n,
        "x_mean": float(sub[x_col].mean()) if n else np.nan,
        "y_mean": float(sub[y_col].mean()) if n else np.nan,
        "r": np.nan,
        "p": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
    }
    if n < min_n or sub[x_col].nunique(dropna=True) < 2 or sub[y_col].nunique(dropna=True) < 2:
        return result

    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    r, p = stats.pearsonr(x, y)
    lr = stats.linregress(x, y)
    result.update(
        {
            "r": float(r),
            "p": float(p),
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "r_squared": float(r**2),
        }
    )
    return result


def correlation_tables(merged: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    grouping_specs = [
        ("condition_hemisphere", ["drug", "condition", "hemisphere"]),
        ("condition", ["drug", "condition"]),
        ("hemisphere", ["drug", "hemisphere"]),
        ("all", ["drug"]),
    ]
    for grouping_name, columns in grouping_specs:
        for keys, group in merged.groupby(columns, sort=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(columns, keys))
            for metric in METRICS:
                row = pearson_row(group, grouping_name, metric, min_n=min_n)
                row.update(base)
                rows.append(row)
    corr = pd.DataFrame(rows)
    front = ["grouping", "drug", "condition", "hemisphere", "metric", "n"]
    rest = [col for col in corr.columns if col not in front]
    return corr[front + rest].sort_values(["grouping", "drug", "condition", "hemisphere", "metric"])


def pattern_condition_deltas(
    pattern: pd.DataFrame,
    drug_condition: str = "DMT",
    placebo_condition: str = "PCB",
) -> pd.DataFrame:
    subset = pattern[pattern["condition"].isin([drug_condition, placebo_condition])].copy()
    wide = subset.pivot(index=["drug", "subid", "hemisphere"], columns="condition", values=METRICS)
    rows = []
    for metric in METRICS:
        if drug_condition not in wide[metric].columns or placebo_condition not in wide[metric].columns:
            continue
        delta = wide[metric][drug_condition] - wide[metric][placebo_condition]
        rows.append(delta.rename(f"{metric}_delta"))
    if not rows:
        raise RuntimeError(
            f"Could not compute pattern deltas for {drug_condition}-{placebo_condition}; "
            "check condition labels."
        )
    out = pd.concat(rows, axis=1).reset_index()
    out["comparison"] = f"{drug_condition}-{placebo_condition}"
    return out.dropna(subset=[f"{metric}_delta" for metric in METRICS], how="all").sort_values(
        ["drug", "subid", "hemisphere"]
    )


def load_network_deltas(root: Path) -> pd.DataFrame:
    frames = []
    for hemisphere in ["left", "right"]:
        path = root / hemisphere / "network_subject_deltas.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"network_a", "network_b", "type", "delta", "subid", "hemisphere", "comparison"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df = df.copy()
        df["subid"] = df["subid"].map(normalize_subid)
        df["hemisphere"] = df["hemisphere"].astype(str).str.strip().str.lower()
        df["network_a"] = df["network_a"].astype(str).str.strip()
        df["network_b"] = df["network_b"].astype(str).str.strip()
        df["type"] = df["type"].astype(str).str.strip()
        df["fc_delta"] = pd.to_numeric(df["delta"], errors="coerce")
        frames.append(
            df[
                [
                    "comparison",
                    "subid",
                    "hemisphere",
                    "network_a",
                    "network_b",
                    "type",
                    "drug_value",
                    "placebo_value",
                    "fc_delta",
                ]
            ]
        )
    if not frames:
        raise RuntimeError(f"No network_subject_deltas.csv files found under {root}")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["hemisphere", "subid", "type", "network_a", "network_b"]
    )


def network_delta_correlations(merged: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    grouping_specs = [
        ("network_pair_hemisphere", ["hemisphere", "network_a", "network_b", "type"]),
        ("network_pair_all_hemispheres", ["network_a", "network_b", "type"]),
        ("type_hemisphere_mean", ["hemisphere", "type"]),
        ("type_all_mean", ["type"]),
    ]
    type_mean = (
        merged.groupby(["subid", "hemisphere", "type"], as_index=False, dropna=False)
        .agg(
            fc_delta=("fc_delta", "mean"),
            **{f"{metric}_delta": (f"{metric}_delta", "first") for metric in METRICS},
        )
        .assign(network_a="ALL", network_b="ALL")
    )
    sources = {
        "network_pair_hemisphere": merged,
        "network_pair_all_hemispheres": merged,
        "type_hemisphere_mean": type_mean,
        "type_all_mean": type_mean,
    }
    for grouping_name, columns in grouping_specs:
        source = sources[grouping_name]
        for keys, group in source.groupby(columns, sort=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(columns, keys))
            for metric in METRICS:
                x_col = f"{metric}_delta"
                row = pearson_xy(group, x_col=x_col, y_col="fc_delta", min_n=min_n)
                row.update(base)
                row["grouping"] = grouping_name
                row["metric"] = metric
                rows.append(row)
    corr = pd.DataFrame(rows)
    front = ["grouping", "hemisphere", "network_a", "network_b", "type", "metric", "n"]
    rest = [col for col in corr.columns if col not in front]
    return corr[front + rest].sort_values(
        ["grouping", "hemisphere", "type", "network_a", "network_b", "metric"]
    )


def p_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=nan"
    if p_value < 1e-3:
        return f"p={p_value:.1e}"
    return f"p={p_value:.3f}"


def plot_condition_hemisphere(merged: pd.DataFrame, corr: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_corr = corr[corr["grouping"] == "condition_hemisphere"].copy()

    sns.set_theme(style="whitegrid")
    for metric in METRICS:
        g = sns.lmplot(
            data=merged,
            x=metric,
            y="gcor",
            row="condition",
            col="hemisphere",
            hue="condition",
            palette="Set2",
            height=3.6,
            aspect=1.15,
            scatter_kws={"s": 42, "alpha": 0.85},
            line_kws={"linewidth": 1.5},
            ci=95,
            facet_kws={"sharex": False, "sharey": True},
        )
        g.set_axis_labels(metric, "GCOR")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        for (cond, hemi), ax in g.axes_dict.items():
            hit = plot_corr[
                (plot_corr["condition"].astype(str) == cond)
                & (plot_corr["hemisphere"].astype(str) == hemi)
                & (plot_corr["metric"] == metric)
            ]
            if hit.empty:
                continue
            row = hit.iloc[0]
            label = f"n={int(row['n'])}, r={row['r']:.3f}, {p_label(float(row['p']))}"
            ax.text(0.03, 0.96, label, transform=ax.transAxes, va="top", ha="left", fontsize=9)
        g.figure.suptitle(f"{metric} vs GCOR", y=1.02)
        g.savefig(fig_dir / f"{metric}_vs_gcor_by_condition_hemisphere.png", dpi=240)
        plt.close(g.figure)


def plot_network_delta_correlations(
    merged: pd.DataFrame,
    corr: pd.DataFrame,
    out_dir: Path,
    top_n: int = 8,
) -> None:
    fig_dir = out_dir / "figures" / "pattern_delta_vs_network_delta"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pair_corr = corr[corr["grouping"] == "network_pair_hemisphere"].copy()
    finite = pair_corr[np.isfinite(pair_corr["r"])].copy()
    if finite.empty:
        return

    for metric in METRICS:
        metric_corr = finite[finite["metric"] == metric].copy()
        if metric_corr.empty:
            continue
        metric_corr["abs_r"] = metric_corr["r"].abs()
        selected = metric_corr.sort_values(["abs_r", "n"], ascending=[False, False]).head(top_n)
        selected = selected.copy()
        selected["edge_label"] = (
            selected["hemisphere"]
            + " "
            + selected["network_a"]
            + "-"
            + selected["network_b"]
            + " "
            + selected["type"]
        )
        selected = selected.sort_values("r")

        plt.figure(figsize=(8.5, max(4.5, 0.4 * len(selected))))
        ax = sns.barplot(data=selected, x="r", y="edge_label", hue="type", dodge=False, palette="Set2")
        ax.axvline(0, color="0.25", linewidth=1)
        ax.set_title(f"Top pattern-delta vs network-delta correlations: {metric}")
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("")
        ax.legend(title="FC type", loc="best")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}_top_network_delta_correlations.png", dpi=240)
        plt.close()

        for row in selected.itertuples(index=False):
            sub = merged[
                (merged["hemisphere"] == row.hemisphere)
                & (merged["network_a"] == row.network_a)
                & (merged["network_b"] == row.network_b)
                & (merged["type"] == row.type)
            ].copy()
            if sub.empty:
                continue
            x_col = f"{metric}_delta"
            plt.figure(figsize=(5.4, 4.2))
            ax = sns.regplot(
                data=sub,
                x=x_col,
                y="fc_delta",
                scatter_kws={"s": 45, "alpha": 0.85},
                line_kws={"linewidth": 1.5},
                ci=95,
            )
            ax.set_title(
                f"{row.hemisphere} {row.network_a}-{row.network_b} {row.type}\n"
                f"{metric} delta vs FC delta"
            )
            ax.set_xlabel(f"{metric} delta (DMT - PCB)")
            ax.set_ylabel("Network FC delta (DMT - PCB)")
            ax.text(
                0.03,
                0.96,
                f"n={int(row.n)}, r={row.r:.3f}, {p_label(float(row.p))}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
            plt.tight_layout()
            name = (
                f"{metric}_{row.hemisphere}_{row.network_a}_"
                f"{row.network_b}_{row.type}_scatter.png"
            )
            plt.savefig(fig_dir / name, dpi=240)
            plt.close()


def write_report(
    out_dir: Path,
    merged: pd.DataFrame,
    corr: pd.DataFrame,
    pattern_delta_network_merged: pd.DataFrame | None = None,
    network_corr: pd.DataFrame | None = None,
) -> None:
    main_corr = corr[corr["grouping"] == "condition_hemisphere"].copy()
    main_corr = main_corr.sort_values(["condition", "hemisphere", "metric"])
    summary = (
        merged.groupby(["drug", "condition", "hemisphere"], as_index=False)
        .agg(
            n=("gcor", "count"),
            gcor_mean=("gcor", "mean"),
            mean_size=("mean_size", "mean"),
            mean_duration=("mean_duration", "mean"),
            mean_power=("mean_power", "mean"),
            pattern_count_per_frame=("pattern_count_per_frame", "mean"),
        )
        .sort_values(["drug", "condition", "hemisphere"])
    )

    lines = [
        "# Pattern Metrics vs GCOR",
        "",
        "## Normalized Data Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Main Correlations: Condition x Hemisphere",
        "",
        main_corr[
            ["drug", "condition", "hemisphere", "metric", "n", "r", "p", "slope", "intercept", "r_squared"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Merged Data Preview",
        "",
        merged.head(30).to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    if pattern_delta_network_merged is not None and network_corr is not None:
        main_network_corr = network_corr[network_corr["grouping"] == "network_pair_hemisphere"].copy()
        main_network_corr = main_network_corr.sort_values(
            ["hemisphere", "type", "network_a", "network_b", "metric"]
        )
        top_network_corr = main_network_corr[np.isfinite(main_network_corr["r"])].copy()
        top_network_corr["abs_r"] = top_network_corr["r"].abs()
        top_network_corr = top_network_corr.sort_values(["abs_r", "n"], ascending=[False, False]).head(30)
        type_corr = network_corr[network_corr["grouping"].isin(["type_hemisphere_mean", "type_all_mean"])].copy()
        lines.extend(
            [
                "## Pattern Delta vs Network FC Delta",
                "",
                "Pattern deltas are DMT - PCB and are matched by subject and hemisphere to wb_pconn network deltas.",
                "",
                "### Top Network-Pair Correlations",
                "",
                top_network_corr[
                    [
                        "hemisphere",
                        "network_a",
                        "network_b",
                        "type",
                        "metric",
                        "n",
                        "r",
                        "p",
                        "slope",
                        "r_squared",
                    ]
                ].to_markdown(index=False, floatfmt=".4f"),
                "",
                "### Mean Within/Between Correlations",
                "",
                type_corr[
                    ["grouping", "hemisphere", "type", "metric", "n", "r", "p", "slope", "r_squared"]
                ].to_markdown(index=False, floatfmt=".4f"),
                "",
                "### Pattern Delta / Network Delta Preview",
                "",
                pattern_delta_network_merged.head(30).to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pattern = load_pattern(args.pattern_csv, pattern_drug=args.pattern_drug)
    gcor = load_gcor(args.gcor_csv)
    merged = pattern.merge(gcor, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No rows matched between pattern metrics and GCOR after normalization.")

    pattern.to_csv(args.out_dir / "pattern_normalized_by_subject.csv", index=False)
    gcor.to_csv(args.out_dir / "gcor_normalized_by_subject.csv", index=False)
    merged.to_csv(args.out_dir / "pattern_gcor_merged_by_subject.csv", index=False)

    corr = correlation_tables(merged, min_n=args.min_n)
    corr.to_csv(args.out_dir / "pattern_gcor_correlations.csv", index=False)

    pattern_deltas = pattern_condition_deltas(pattern, drug_condition="DMT", placebo_condition="PCB")
    pattern_deltas = pattern_deltas.rename(columns={"comparison": "pattern_comparison"})
    network_deltas = load_network_deltas(args.network_delta_root)
    network_deltas = network_deltas.rename(columns={"comparison": "fc_comparison"})
    pattern_network = network_deltas.merge(
        pattern_deltas,
        on=["subid", "hemisphere"],
        how="inner",
        validate="many_to_one",
    )
    if pattern_network.empty:
        raise RuntimeError("No rows matched between pattern deltas and network FC deltas.")
    pattern_deltas.to_csv(args.out_dir / "pattern_condition_deltas_DMT_minus_PCB.csv", index=False)
    network_deltas.to_csv(args.out_dir / "network_subject_deltas_left_right.csv", index=False)
    pattern_network.to_csv(
        args.out_dir / "pattern_delta_network_delta_merged_by_subject.csv",
        index=False,
    )
    network_corr = network_delta_correlations(pattern_network, min_n=args.min_n)
    network_corr.to_csv(args.out_dir / "pattern_delta_network_delta_correlations.csv", index=False)

    plot_condition_hemisphere(merged, corr, args.out_dir)
    plot_network_delta_correlations(pattern_network, network_corr, args.out_dir)
    write_report(
        args.out_dir,
        merged,
        corr,
        pattern_delta_network_merged=pattern_network,
        network_corr=network_corr,
    )

    metadata = {
        "pattern_csv": str(args.pattern_csv),
        "gcor_csv": str(args.gcor_csv),
        "network_delta_root": str(args.network_delta_root),
        "out_dir": str(args.out_dir),
        "pattern_drug": args.pattern_drug,
        "metrics": METRICS,
        "key_columns": KEY_COLUMNS,
        "n_pattern_rows": int(len(pattern)),
        "n_gcor_rows": int(len(gcor)),
        "n_merged_rows": int(len(merged)),
        "n_pattern_delta_rows": int(len(pattern_deltas)),
        "n_network_delta_rows": int(len(network_deltas)),
        "n_pattern_network_rows": int(len(pattern_network)),
        "min_n": int(args.min_n),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nMerged data")
    print(merged.head(args.show_rows).to_string(index=False))
    print("\nCondition x hemisphere correlations")
    main_corr = corr[corr["grouping"] == "condition_hemisphere"]
    print(
        main_corr[
            ["drug", "condition", "hemisphere", "metric", "n", "r", "p", "slope", "r_squared"]
        ].to_string(index=False)
    )
    print("\nPattern delta vs network FC delta: top network-pair correlations")
    pair_corr = network_corr[network_corr["grouping"] == "network_pair_hemisphere"].copy()
    pair_corr = pair_corr[np.isfinite(pair_corr["r"])].copy()
    pair_corr["abs_r"] = pair_corr["r"].abs()
    print(
        pair_corr.sort_values(["abs_r", "n"], ascending=[False, False])
        .head(20)[["hemisphere", "network_a", "network_b", "type", "metric", "n", "r", "p", "slope", "r_squared"]]
        .to_string(index=False)
    )
    print(f"\nWrote pattern-GCOR analysis to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
