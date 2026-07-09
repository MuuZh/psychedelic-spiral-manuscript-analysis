#!/usr/bin/env python
"""
Edge-weighted wFC/bFC drug-placebo delta analysis for original vs reconstructed phase-FC.

This script is intended for the "global integration / local disorganization"
analysis. It reads subject-level parcel Pearson-r FC matrices, transforms them
to Fisher z before averaging, and computes:

1. Network wFC: mean within-network parcel edges for each 7-network label.
2. Global wFC: edge-weighted mean of all within-network parcel edges.
3. Global bFC: edge-weighted mean of all between-network parcel edges.

For each metric it saves subject-level drug, placebo, and drug-placebo delta
values for original and reconstructed FC. It then reports:

- drug-placebo delta tests against zero for original and reconstructed values
- original-vs-reconstructed agreement for drug, placebo, and delta values

All primary values and errors are in Fisher-z FC units. Mean z values are also
back-transformed to r in selected summary columns for interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


CONDITION_RE = re.compile(r"^(?P<condition>[A-Za-z]+_[A-Za-z]+)_S(?P<subid>\d+)")
NETWORK_ORDER_7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
VALUE_KINDS = ["delta", "drug", "placebo"]


@dataclass(frozen=True)
class DatasetSpec:
    drug_label: str
    drug_condition: str
    placebo_condition: str
    original_root: Path
    recon_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge-weighted wFC/bFC delta analysis for original vs reconstructed phase-FC."
    )
    parser.add_argument("--dmt-original-root", default=Path("analysis_outputs/phase_fc_batch_phase_corr_7networks"), type=Path)
    parser.add_argument("--dmt-recon-root", default=Path("analysis_outputs/phase_fc_recon_7networks"), type=Path)
    parser.add_argument("--lsd-original-root", default=Path("analysis_outputs/phase_fc_batch_phase_corr_7networks_LSD"), type=Path)
    parser.add_argument("--lsd-recon-root", default=Path("analysis_outputs/phase_fc_recon_7networks_LSD"), type=Path)
    parser.add_argument("--fc-file", default="parcel_phase_corr.npy")
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/phase_recon_edge_weighted_wbfc_delta_v3"),
        type=Path,
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument("--min-n", default=5, type=int)
    parser.add_argument(
        "--clip-eps",
        default=1e-7,
        type=float,
        help="Clip Pearson r into [-1+eps, 1-eps] before arctanh.",
    )
    return parser.parse_args()


def normalize_subid(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("S"):
        text = text[1:]
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def infer_hemisphere_from_meta(meta_path: Path) -> str:
    meta = pd.read_csv(meta_path, usecols=["hemi"])
    hemi = str(meta["hemi"].iloc[0])
    if hemi == "LH":
        return "left"
    if hemi == "RH":
        return "right"
    raise ValueError(f"Unexpected hemi value {hemi!r} in {meta_path}")


def parse_entry(out_dir: Path, fc_file: str) -> dict[str, object] | None:
    match = CONDITION_RE.search(out_dir.name)
    if not match:
        return None
    fc_path = out_dir / fc_file
    meta_path = out_dir / "parcel_metadata.csv"
    if not fc_path.exists() or not meta_path.exists():
        return None
    return {
        "condition": match.group("condition"),
        "subid": normalize_subid(match.group("subid")),
        "hemisphere": infer_hemisphere_from_meta(meta_path),
        "out_dir": out_dir,
        "fc_path": fc_path,
        "meta_path": meta_path,
    }


def discover_entries(root: Path, fc_file: str, hemispheres: list[str]) -> pd.DataFrame:
    rows = []
    for out_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "atlas_metadata"):
        entry = parse_entry(out_dir, fc_file)
        if entry is not None and entry["hemisphere"] in hemispheres:
            rows.append(entry)
    if not rows:
        raise RuntimeError(f"No subject FC matrices found under {root}")
    return pd.DataFrame(rows)


def load_metadata(entries: pd.DataFrame, condition: str, subid: str, hemisphere: str) -> pd.DataFrame:
    row = entries[
        (entries["condition"] == condition)
        & (entries["subid"] == subid)
        & (entries["hemisphere"] == hemisphere)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one metadata row for {condition} {subid} {hemisphere}, found {len(row)}")
    return pd.read_csv(Path(row.iloc[0]["meta_path"]))


def load_matrix(entries: pd.DataFrame, condition: str, subid: str, hemisphere: str, clip_eps: float) -> np.ndarray:
    row = entries[
        (entries["condition"] == condition)
        & (entries["subid"] == subid)
        & (entries["hemisphere"] == hemisphere)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one FC matrix for {condition} {subid} {hemisphere}, found {len(row)}")
    mat = np.asarray(np.load(Path(row.iloc[0]["fc_path"])), dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, got {mat.shape}: {row.iloc[0]['fc_path']}")
    mat = np.clip(mat, -1.0 + clip_eps, 1.0 - clip_eps)
    z = np.arctanh(mat)
    np.fill_diagonal(z, np.nan)
    return z


def assert_metadata_match(left: pd.DataFrame, right: pd.DataFrame) -> None:
    cols = ["parcel_id", "parcel_name", "hemi", "network"]
    if not left[cols].reset_index(drop=True).equals(right[cols].reset_index(drop=True)):
        raise ValueError("Parcel metadata mismatch between original and reconstructed roots")


def infer_network_order(networks: pd.Series | list[str]) -> list[str]:
    values = [str(network) for network in networks if pd.notna(network)]
    available = set(values)
    ordered = [network for network in NETWORK_ORDER_7 if network in available]
    extras = [network for network in values if network not in set(NETWORK_ORDER_7)]
    return ordered + list(dict.fromkeys(extras)) if ordered else list(dict.fromkeys(values))


def paired_subjects(original: pd.DataFrame, recon: pd.DataFrame, drug: str, placebo: str, hemisphere: str) -> list[str]:
    subjects = None
    for entries, condition in [(original, drug), (original, placebo), (recon, drug), (recon, placebo)]:
        current = set(entries[(entries["condition"] == condition) & (entries["hemisphere"] == hemisphere)]["subid"])
        subjects = current if subjects is None else subjects & current
    return sorted(subjects or [])


def edge_masks(parcels: pd.DataFrame) -> dict[str, np.ndarray]:
    networks = parcels["network"].astype(str).to_numpy()
    n = len(networks)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    same = networks[:, None] == networks[None, :]
    return {
        "global_within": upper & same,
        "global_between": upper & ~same,
    }


def edge_mean(z_mat: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    values = z_mat[mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, 0
    return float(np.mean(finite)), int(finite.size)


def metric_values(z_mat: np.ndarray, parcels: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    network_values = parcels["network"].astype(str).to_numpy()
    for network in infer_network_order(parcels["network"]):
        idx = np.flatnonzero(network_values == network)
        if idx.size < 2:
            continue
        sub = z_mat[np.ix_(idx, idx)]
        values = sub[np.triu_indices_from(sub, k=1)]
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "metric": "wFC_network",
                "scope": "network",
                "network": network,
                "fc_type": "within",
                "value_z": float(np.mean(finite)) if finite.size else math.nan,
                "n_edges": int(finite.size),
            }
        )

    for metric, fc_type in [("global_within_edge_weighted", "within"), ("global_between_edge_weighted", "between")]:
        value, n_edges = edge_mean(z_mat, masks["global_within" if fc_type == "within" else "global_between"])
        rows.append(
            {
                "metric": metric,
                "scope": "global",
                "network": "ALL",
                "fc_type": fc_type,
                "value_z": value,
                "n_edges": n_edges,
            }
        )
    return pd.DataFrame(rows)


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n <= 1:
        return math.nan, math.nan
    sem = stats.sem(values, nan_policy="omit")
    if not np.isfinite(sem):
        return math.nan, math.nan
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=float(np.mean(values)), scale=float(sem))
    return float(lo), float(hi)


def r_ci95(r_value: float, n: int) -> tuple[float, float]:
    if not np.isfinite(r_value) or n <= 3 or abs(r_value) >= 1:
        return math.nan, math.nan
    z = np.arctanh(r_value)
    se = 1.0 / math.sqrt(n - 3)
    crit = stats.norm.ppf(0.975)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


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


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite].astype(float), y[finite].astype(float)


def delta_test_stats(values: np.ndarray, min_n: int) -> dict[str, object]:
    values = values[np.isfinite(values)].astype(float)
    n = int(values.size)
    lo, hi = mean_ci95(values)
    out: dict[str, object] = {
        "n": n,
        "mean_delta_z": float(np.mean(values)) if n else math.nan,
        "mean_delta_r": float(np.tanh(np.mean(values))) if n else math.nan,
        "sd_delta_z": float(np.std(values, ddof=1)) if n > 1 else math.nan,
        "sem_delta_z": float(stats.sem(values)) if n > 1 else math.nan,
        "ci95_delta_z_low": lo,
        "ci95_delta_z_high": hi,
        "median_delta_z": float(np.median(values)) if n else math.nan,
        "min_delta_z": float(np.min(values)) if n else math.nan,
        "max_delta_z": float(np.max(values)) if n else math.nan,
    }
    if n >= min_n and n > 1 and np.std(values, ddof=1) > 0:
        t_stat, p_two = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
        mean_value = float(np.mean(values))
        if mean_value > 0:
            p_greater = float(p_two / 2)
            p_less = float(1 - p_two / 2)
        elif mean_value < 0:
            p_less = float(p_two / 2)
            p_greater = float(1 - p_two / 2)
        else:
            p_less = 0.5
            p_greater = 0.5
        out.update(
            {
                "t_vs_zero": float(t_stat),
                "p_two_sided": float(p_two),
                "p_less_than_zero": p_less,
                "p_greater_than_zero": p_greater,
                "cohen_dz": mean_value / float(np.std(values, ddof=1)),
            }
        )
    else:
        out.update(
            {
                "t_vs_zero": math.nan,
                "p_two_sided": math.nan,
                "p_less_than_zero": math.nan,
                "p_greater_than_zero": math.nan,
                "cohen_dz": math.nan,
            }
        )
    return out


def agreement_stats(x: np.ndarray, y: np.ndarray, min_n: int) -> dict[str, object]:
    x, y = finite_pair(x, y)
    n = int(x.size)
    err = y - x
    abs_err = np.abs(err)
    sq_err = err**2
    err_lo, err_hi = mean_ci95(err)
    abs_err_lo, abs_err_hi = mean_ci95(abs_err)
    out: dict[str, object] = {
        "n": n,
        "original_mean_z": float(np.mean(x)) if n else math.nan,
        "original_mean_r": float(np.tanh(np.mean(x))) if n else math.nan,
        "original_sd_z": float(np.std(x, ddof=1)) if n > 1 else math.nan,
        "original_sem_z": float(stats.sem(x)) if n > 1 else math.nan,
        "recon_mean_z": float(np.mean(y)) if n else math.nan,
        "recon_mean_r": float(np.tanh(np.mean(y))) if n else math.nan,
        "recon_sd_z": float(np.std(y, ddof=1)) if n > 1 else math.nan,
        "recon_sem_z": float(stats.sem(y)) if n > 1 else math.nan,
        "bias_mean_error_z": float(np.mean(err)) if n else math.nan,
        "bias_mean_error_r": float(np.tanh(np.mean(err))) if n else math.nan,
        "bias_error_sd_z": float(np.std(err, ddof=1)) if n > 1 else math.nan,
        "bias_error_sem_z": float(stats.sem(err)) if n > 1 else math.nan,
        "bias_error_ci95_low_z": err_lo,
        "bias_error_ci95_high_z": err_hi,
        "mean_abs_error_z": float(np.mean(abs_err)) if n else math.nan,
        "mean_abs_error_sd_z": float(np.std(abs_err, ddof=1)) if n > 1 else math.nan,
        "mean_abs_error_sem_z": float(stats.sem(abs_err)) if n > 1 else math.nan,
        "mean_abs_error_ci95_low_z": abs_err_lo,
        "mean_abs_error_ci95_high_z": abs_err_hi,
        "median_abs_error_z": float(np.median(abs_err)) if n else math.nan,
        "max_abs_error_z": float(np.max(abs_err)) if n else math.nan,
        "mse_z": float(np.mean(sq_err)) if n else math.nan,
        "rmse_z": float(np.sqrt(np.mean(sq_err))) if n else math.nan,
        "mean_abs_original_z": float(np.mean(np.abs(x))) if n else math.nan,
    }
    denom = np.abs(x) + np.abs(y)
    smape = np.divide(2 * abs_err, denom, out=np.full_like(abs_err, np.nan), where=denom > 0)
    out["smape_z"] = float(np.nanmean(smape)) if n and np.isfinite(smape).any() else math.nan
    out["nrmse_by_original_sd_z"] = (
        out["rmse_z"] / out["original_sd_z"]
        if np.isfinite(out["rmse_z"]) and np.isfinite(out["original_sd_z"]) and out["original_sd_z"] > 0
        else math.nan
    )
    out["nrmse_by_original_abs_mean_z"] = (
        out["rmse_z"] / out["mean_abs_original_z"]
        if np.isfinite(out["rmse_z"]) and np.isfinite(out["mean_abs_original_z"]) and out["mean_abs_original_z"] > 0
        else math.nan
    )
    out["identity_r2_score_z"] = (
        1.0 - float(np.sum(sq_err)) / float(np.sum((x - np.mean(x)) ** 2))
        if n > 1 and np.sum((x - np.mean(x)) ** 2) > 0
        else math.nan
    )
    out["explained_variance_score_z"] = (
        1.0 - float(np.var(err, ddof=0)) / float(np.var(x, ddof=0))
        if n > 1 and np.var(x, ddof=0) > 0
        else math.nan
    )
    if n >= min_n and n > 1 and np.std(err, ddof=1) > 0:
        t_stat, t_p = stats.ttest_1samp(err, popmean=0.0, nan_policy="omit")
        out["error_t_vs_zero"] = float(t_stat)
        out["error_p_vs_zero"] = float(t_p)
        out["error_cohen_dz"] = float(np.mean(err) / np.std(err, ddof=1))
    else:
        out["error_t_vs_zero"] = math.nan
        out["error_p_vs_zero"] = math.nan
        out["error_cohen_dz"] = math.nan
    if n >= min_n and np.std(x, ddof=1) > 0 and np.std(y, ddof=1) > 0:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        slope, intercept, reg_r, reg_p, slope_stderr = stats.linregress(x, y)
        lo, hi = r_ci95(float(pearson_r), n)
        out.update(
            {
                "pearson_r": float(pearson_r),
                "pearson_r2": float(pearson_r**2),
                "pearson_p": float(pearson_p),
                "pearson_r_ci95_low": lo,
                "pearson_r_ci95_high": hi,
                "pearson_stars": p_to_stars(float(pearson_p)),
                "slope_z": float(slope),
                "intercept_z": float(intercept),
                "regression_r2": float(reg_r**2),
                "regression_p": float(reg_p),
                "slope_stderr_z": float(slope_stderr),
            }
        )
    else:
        out.update(
            {
                "pearson_r": math.nan,
                "pearson_r2": math.nan,
                "pearson_p": math.nan,
                "pearson_r_ci95_low": math.nan,
                "pearson_r_ci95_high": math.nan,
                "pearson_stars": "",
                "slope_z": math.nan,
                "intercept_z": math.nan,
                "regression_r2": math.nan,
                "regression_p": math.nan,
                "slope_stderr_z": math.nan,
            }
        )
    return out


def build_condition_table(values: pd.DataFrame) -> pd.DataFrame:
    keys = ["drug", "comparison", "hemisphere", "subid", "source", "metric", "scope", "network", "fc_type", "n_edges"]
    work = values.copy()
    work["condition_value_kind"] = work["condition_role"].map(
        {"drug": "condition_drug", "placebo": "condition_placebo"}
    )
    wide = work.pivot_table(index=keys, columns="condition_value_kind", values="value_z", aggfunc="first").reset_index()
    if "condition_drug" not in wide.columns or "condition_placebo" not in wide.columns:
        raise RuntimeError("Could not pivot drug/placebo values; check condition labels.")
    wide["delta"] = wide["condition_drug"] - wide["condition_placebo"]
    long = wide.melt(
        id_vars=keys,
        value_vars=["delta", "condition_drug", "condition_placebo"],
        var_name="value_kind",
        value_name="value_z",
    )
    long["value_kind"] = long["value_kind"].map(
        {"delta": "delta", "condition_drug": "drug", "condition_placebo": "placebo"}
    )
    long["value_r"] = np.tanh(long["value_z"])
    return long.sort_values(["drug", "hemisphere", "subid", "source", "metric", "network", "value_kind"])


def joined_original_recon(condition_values: pd.DataFrame) -> pd.DataFrame:
    keys = ["drug", "comparison", "hemisphere", "subid", "metric", "scope", "network", "fc_type", "value_kind"]
    original = condition_values[condition_values["source"] == "original"][keys + ["value_z", "value_r", "n_edges"]].rename(
        columns={"value_z": "original_value_z", "value_r": "original_value_r", "n_edges": "n_edges_original"}
    )
    recon = condition_values[condition_values["source"] == "recon"][keys + ["value_z", "value_r", "n_edges"]].rename(
        columns={"value_z": "recon_value_z", "value_r": "recon_value_r", "n_edges": "n_edges_recon"}
    )
    joined = original.merge(recon, on=keys, how="inner")
    joined["error_z"] = joined["recon_value_z"] - joined["original_value_z"]
    joined["abs_error_z"] = joined["error_z"].abs()
    return joined


def delta_tests(condition_values: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    subset = condition_values[condition_values["value_kind"] == "delta"]
    keys = ["drug", "comparison", "hemisphere", "source", "metric", "scope", "network", "fc_type"]
    for group_values, group in subset.groupby(keys, sort=False):
        row = dict(zip(keys, group_values))
        row.update(delta_test_stats(group["value_z"].to_numpy(dtype=float), min_n=min_n))
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_tables(joined: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    keys = ["drug", "comparison", "hemisphere", "metric", "scope", "network", "fc_type", "value_kind"]
    for group_values, group in joined.groupby(keys, sort=False):
        row = dict(zip(keys, group_values))
        row.update(
            agreement_stats(
                group["original_value_z"].to_numpy(dtype=float),
                group["recon_value_z"].to_numpy(dtype=float),
                min_n=min_n,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_delta_scatter(joined: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = joined[joined["value_kind"] == "delta"].copy()
    if plot_df.empty:
        return
    sns.set_theme(style="whitegrid")
    for drug in sorted(plot_df["drug"].unique()):
        for hemisphere in sorted(plot_df["hemisphere"].unique()):
            sub = plot_df[(plot_df["drug"] == drug) & (plot_df["hemisphere"] == hemisphere)]
            if sub.empty:
                continue
            metrics = ["global_between_edge_weighted", "global_within_edge_weighted", "wFC_network"]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
            for ax, metric in zip(axes[0], metrics):
                metric_df = sub[sub["metric"] == metric]
                if metric_df.empty:
                    ax.axis("off")
                    continue
                if metric == "wFC_network":
                    sns.scatterplot(
                        data=metric_df,
                        x="original_value_z",
                        y="recon_value_z",
                        hue="network",
                        s=38,
                        alpha=0.85,
                        ax=ax,
                    )
                    ax.legend(fontsize=7, title="", loc="best")
                else:
                    sns.regplot(
                        data=metric_df,
                        x="original_value_z",
                        y="recon_value_z",
                        ci=95,
                        scatter_kws={"s": 46, "alpha": 0.85},
                        line_kws={"color": "#222222", "linewidth": 1.4},
                        ax=ax,
                    )
                finite_vals = pd.concat([metric_df["original_value_z"], metric_df["recon_value_z"]]).replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if not finite_vals.empty:
                    lo = float(finite_vals.min())
                    hi = float(finite_vals.max())
                    pad = max((hi - lo) * 0.08, 1e-4)
                    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="gray", linewidth=0.9)
                    ax.set_xlim(lo - pad, hi + pad)
                    ax.set_ylim(lo - pad, hi + pad)
                stat = stats_df[
                    (stats_df["drug"] == drug)
                    & (stats_df["hemisphere"] == hemisphere)
                    & (stats_df["metric"] == metric)
                    & (stats_df["value_kind"] == "delta")
                ]
                if metric != "wFC_network":
                    stat = stat[stat["network"] == "ALL"]
                if not stat.empty and pd.notna(stat.iloc[0]["pearson_r"]):
                    row = stat.iloc[0]
                    ax.text(
                        0.04,
                        0.96,
                        f"r={row.pearson_r:.2f}\nR2={row.pearson_r2:.2f}\nMAE={row.mean_abs_error_z:.4f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                    )
                ax.set_title(metric.replace("_edge_weighted", "").replace("_", " "))
                ax.set_xlabel("Original delta (z)")
                ax.set_ylabel("Recon delta (z)")
            fig.suptitle(f"{drug} {hemisphere}: original vs recon drug-placebo delta", y=1.02, fontsize=14)
            fig.tight_layout()
            fig.savefig(out_dir / f"delta_scatter_{drug}_{hemisphere}.png", dpi=260, bbox_inches="tight")
            plt.close(fig)


def analyze_dataset(
    spec: DatasetSpec,
    hemispheres: list[str],
    fc_file: str,
    out_dir: Path,
    min_n: int,
    clip_eps: float,
) -> dict[str, object]:
    original_entries = discover_entries(spec.original_root, fc_file, hemispheres)
    recon_entries = discover_entries(spec.recon_root, fc_file, hemispheres)
    rows = []
    dataset_out = out_dir / spec.drug_label
    dataset_out.mkdir(parents=True, exist_ok=True)

    for hemisphere in hemispheres:
        subjects = paired_subjects(
            original_entries,
            recon_entries,
            spec.drug_condition,
            spec.placebo_condition,
            hemisphere,
        )
        if len(subjects) < min_n:
            print(f"Skipping {spec.drug_label} {hemisphere}: only {len(subjects)} complete subjects")
            continue
        parcels = load_metadata(original_entries, spec.drug_condition, subjects[0], hemisphere)
        recon_parcels = load_metadata(recon_entries, spec.drug_condition, subjects[0], hemisphere)
        assert_metadata_match(parcels, recon_parcels)
        masks = edge_masks(parcels)

        subject_rows = []
        for subid in subjects:
            mats = {
                ("original", "drug"): load_matrix(original_entries, spec.drug_condition, subid, hemisphere, clip_eps),
                ("original", "placebo"): load_matrix(original_entries, spec.placebo_condition, subid, hemisphere, clip_eps),
                ("recon", "drug"): load_matrix(recon_entries, spec.drug_condition, subid, hemisphere, clip_eps),
                ("recon", "placebo"): load_matrix(recon_entries, spec.placebo_condition, subid, hemisphere, clip_eps),
            }
            expected_shape = (len(parcels), len(parcels))
            if any(mat.shape != expected_shape for mat in mats.values()):
                raise ValueError(f"Matrix shape mismatch for {spec.drug_label} {subid} {hemisphere}")
            for (source, condition_role), mat in mats.items():
                metric_df = metric_values(mat, parcels, masks)
                metric_df["drug"] = spec.drug_label
                metric_df["comparison"] = f"{spec.drug_condition}-{spec.placebo_condition}"
                metric_df["hemisphere"] = hemisphere
                metric_df["subid"] = subid
                metric_df["source"] = source
                metric_df["condition_role"] = condition_role
                rows.append(metric_df)
            subject_rows.append({"drug": spec.drug_label, "hemisphere": hemisphere, "subid": subid})
        pd.DataFrame(subject_rows).to_csv(dataset_out / f"paired_subjects_{hemisphere}.csv", index=False)

    if not rows:
        raise RuntimeError(f"No rows generated for {spec.drug_label}")
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(dataset_out / "subject_condition_values_z.csv", index=False)
    return {
        "drug": spec.drug_label,
        "original_root": str(spec.original_root),
        "recon_root": str(spec.recon_root),
        "n_original_entries": int(len(original_entries)),
        "n_recon_entries": int(len(recon_entries)),
        "n_subject_condition_rows": int(len(raw)),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hemispheres = ["left", "right"] if args.hemisphere == "both" else [args.hemisphere]
    specs = [
        DatasetSpec("DMT", "DMT_DMT", "DMT_PCB", args.dmt_original_root, args.dmt_recon_root),
        DatasetSpec("LSD", "LSD_LSD", "LSD_PCB", args.lsd_original_root, args.lsd_recon_root),
    ]

    metadata = {
        "fc_file": args.fc_file,
        "hemispheres": hemispheres,
        "min_n": int(args.min_n),
        "clip_eps": float(args.clip_eps),
        "fisher_z_formula": "z = arctanh(r) = 0.5 * ln((1+r)/(1-r))",
        "primary_values": "All averaging, deltas, correlations, and error metrics are computed in Fisher-z FC units.",
        "global_metrics": {
            "global_within_edge_weighted": "Mean of all within-network parcel-pair edges.",
            "global_between_edge_weighted": "Mean of all between-network parcel-pair edges.",
        },
        "network_metrics": {
            "wFC_network": "Mean of within-network parcel-pair edges for each functional network.",
        },
        "datasets": [],
    }

    all_raw = []
    for spec in specs:
        dataset_meta = analyze_dataset(
            spec,
            hemispheres=hemispheres,
            fc_file=args.fc_file,
            out_dir=args.out_dir,
            min_n=args.min_n,
            clip_eps=args.clip_eps,
        )
        metadata["datasets"].append(dataset_meta)
        all_raw.append(pd.read_csv(args.out_dir / spec.drug_label / "subject_condition_values_z.csv"))

    raw_values = pd.concat(all_raw, ignore_index=True)
    condition_values = build_condition_table(raw_values)
    joined = joined_original_recon(condition_values)
    tests = delta_tests(condition_values, min_n=args.min_n)
    agreement = agreement_tables(joined, min_n=args.min_n)
    if agreement.empty:
        raise RuntimeError("No original/reconstructed rows matched; check join keys in subject_drug_placebo_delta_values_z.csv.")
    delta_agreement = agreement[agreement["value_kind"] == "delta"].copy()

    raw_values.to_csv(args.out_dir / "subject_condition_values_z.csv", index=False)
    condition_values.to_csv(args.out_dir / "subject_drug_placebo_delta_values_z.csv", index=False)
    joined.to_csv(args.out_dir / "original_recon_joined_values_z.csv", index=False)
    tests.to_csv(args.out_dir / "drug_placebo_delta_tests_z.csv", index=False)
    agreement.to_csv(args.out_dir / "original_recon_agreement_stats_z.csv", index=False)
    delta_agreement.to_csv(args.out_dir / "original_recon_delta_agreement_stats_z.csv", index=False)

    save_delta_scatter(joined, agreement, args.out_dir / "figures" / "delta_scatter")

    metadata.update(
        {
            "n_subject_condition_rows": int(len(raw_values)),
            "n_subject_value_rows": int(len(condition_values)),
            "n_joined_rows": int(len(joined)),
            "n_delta_test_rows": int(len(tests)),
            "n_agreement_rows": int(len(agreement)),
            "no_spearman": True,
        }
    )
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(f"Wrote edge-weighted wFC/bFC delta analysis to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
