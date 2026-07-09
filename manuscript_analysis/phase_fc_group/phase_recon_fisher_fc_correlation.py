#!/usr/bin/env python
"""
Fisher-z based original-vs-reconstructed phase-FC correlation analysis.

This script reads subject-level parcel Pearson-r matrices, applies Fisher's
z-transform before any averaging, and reports original/reconstructed agreement
at three levels:

1. Matrix level:
   - mean z matrices for drug, placebo, and delta
   - spatial pattern correlation across parcel-pair edges
2. Network level:
   - within/between network mean z per subject
   - subject-level original/reconstructed correlations per network pair
3. Global level:
   - grand mean FC per subject for all/within/between edges
   - subject-level original/reconstructed correlations
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
GLOBAL_SCOPES = ["global_within", "global_between"]


@dataclass(frozen=True)
class DatasetSpec:
    drug_label: str
    drug_condition: str
    placebo_condition: str
    original_root: Path
    recon_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fisher-z original/reconstructed phase-FC correlations from parcel matrices."
    )
    parser.add_argument("--dmt-original-root", default=Path("analysis_outputs/phase_fc_batch_phase_corr_7networks"), type=Path)
    parser.add_argument("--dmt-recon-root", default=Path("analysis_outputs/phase_fc_recon_7networks"), type=Path)
    parser.add_argument("--lsd-original-root", default=Path("analysis_outputs/phase_fc_batch_phase_corr_7networks_LSD"), type=Path)
    parser.add_argument("--lsd-recon-root", default=Path("analysis_outputs/phase_fc_recon_7networks_LSD"), type=Path)
    parser.add_argument("--fc-file", default="parcel_phase_corr.npy")
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/phase_recon_fisher_fc_correlation"),
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


def network_metric(z_mat: np.ndarray, parcels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    network_values = parcels["network"].astype(str).to_numpy()
    networks = infer_network_order(parcels["network"])
    for i, net_a in enumerate(networks):
        idx_a = np.flatnonzero(network_values == net_a)
        if idx_a.size >= 2:
            sub = z_mat[np.ix_(idx_a, idx_a)]
            rows.append(
                {
                    "network_a": net_a,
                    "network_b": net_a,
                    "type": "within",
                    "value_z": float(np.nanmean(sub[np.triu_indices_from(sub, k=1)])),
                }
            )
        for net_b in networks[i + 1 :]:
            idx_b = np.flatnonzero(network_values == net_b)
            sub = z_mat[np.ix_(idx_a, idx_b)]
            rows.append(
                {
                    "network_a": net_a,
                    "network_b": net_b,
                    "type": "between",
                    "value_z": float(np.nanmean(sub)),
                }
            )
    return pd.DataFrame(rows)


def global_metric_from_network(network_df: pd.DataFrame) -> pd.DataFrame:
    """Global within/between as unweighted means of network matrix cells."""
    within = network_df[network_df["type"] == "within"]["value_z"].to_numpy(dtype=float)
    between = network_df[network_df["type"] == "between"]["value_z"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {"scope": "global_within", "value_z": float(np.nanmean(within))},
            {"scope": "global_between", "value_z": float(np.nanmean(between))},
        ]
    )


def paired_subjects(original: pd.DataFrame, recon: pd.DataFrame, drug: str, placebo: str, hemisphere: str) -> list[str]:
    subjects = None
    for entries, condition in [(original, drug), (original, placebo), (recon, drug), (recon, placebo)]:
        current = set(entries[(entries["condition"] == condition) & (entries["hemisphere"] == hemisphere)]["subid"])
        subjects = current if subjects is None else subjects & current
    return sorted(subjects or [])


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


def r_ci95(r_value: float, n: int) -> tuple[float, float]:
    if not np.isfinite(r_value) or n <= 3 or abs(r_value) >= 1:
        return math.nan, math.nan
    z = np.arctanh(r_value)
    se = 1.0 / math.sqrt(n - 3)
    crit = stats.norm.ppf(0.975)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def agreement_stats(x: np.ndarray, y: np.ndarray, min_n: int) -> dict[str, object]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite].astype(float)
    y = y[finite].astype(float)
    n = int(x.size)
    err = y - x
    abs_err = np.abs(err)
    sq_err = err**2
    out: dict[str, object] = {
        "n": n,
        "original_mean_z": float(np.mean(x)) if n else math.nan,
        "original_sd_z": float(np.std(x, ddof=1)) if n > 1 else math.nan,
        "recon_mean_z": float(np.mean(y)) if n else math.nan,
        "recon_sd_z": float(np.std(y, ddof=1)) if n > 1 else math.nan,
        "original_mean_r": float(np.tanh(np.mean(x))) if n else math.nan,
        "recon_mean_r": float(np.tanh(np.mean(y))) if n else math.nan,
        "bias_mean_error_z": float(np.mean(err)) if n else math.nan,
        "mean_abs_error_z": float(np.mean(abs_err)) if n else math.nan,
        "median_abs_error_z": float(np.median(abs_err)) if n else math.nan,
        "max_abs_error_z": float(np.max(abs_err)) if n else math.nan,
        "mse_z": float(np.mean(sq_err)) if n else math.nan,
        "rmse_z": float(np.sqrt(np.mean(sq_err))) if n else math.nan,
        "error_sd_z": float(np.std(err, ddof=1)) if n > 1 else math.nan,
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
        spearman_r, spearman_p = stats.spearmanr(x, y)
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
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
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
                "spearman_r": math.nan,
                "spearman_p": math.nan,
                "slope_z": math.nan,
                "intercept_z": math.nan,
                "regression_r2": math.nan,
                "regression_p": math.nan,
                "slope_stderr_z": math.nan,
            }
        )
    return out


def matrix_long(mean_mats: dict[str, np.ndarray], parcels: pd.DataFrame) -> pd.DataFrame:
    ids = parcels["parcel_id"].to_numpy()
    names = parcels["parcel_name"].to_numpy()
    networks = parcels["network"].to_numpy()
    n = len(parcels)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            row = {
                "parcel_i": int(ids[i]),
                "parcel_j": int(ids[j]),
                "parcel_i_name": names[i],
                "parcel_j_name": names[j],
                "network_i": networks[i],
                "network_j": networks[j],
            }
            for name, mat in mean_mats.items():
                value_z = float(mat[i, j]) if np.isfinite(mat[i, j]) else math.nan
                row[f"{name}_z"] = value_z
                row[f"{name}_r"] = float(np.tanh(value_z)) if np.isfinite(value_z) else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def network_matrix_from_rows(stats_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()
    networks = infer_network_order(list(stats_df["network_a"]) + list(stats_df["network_b"]))
    mat = pd.DataFrame(np.nan, index=networks, columns=networks, dtype=float)
    for row in stats_df.itertuples(index=False):
        value = getattr(row, value_col)
        mat.loc[row.network_a, row.network_b] = value
        mat.loc[row.network_b, row.network_a] = value
    return mat


def save_matrix_heatmap(mat: np.ndarray, parcels: pd.DataFrame, out_path: Path, title: str, label: str) -> None:
    network_order = infer_network_order(parcels["network"])
    sort_key = parcels["network"].map({network: i for i, network in enumerate(network_order)}).fillna(999)
    order = np.lexsort((parcels["parcel_id"].to_numpy(), sort_key.to_numpy()))
    sorted_mat = mat[np.ix_(order, order)]
    sorted_parcels = parcels.iloc[order].reset_index(drop=True)
    vmax = np.nanmax(np.abs(sorted_mat)) if np.isfinite(sorted_mat).any() else 1.0
    vmax = max(vmax, 1e-6)

    plt.figure(figsize=(9, 8))
    sns.heatmap(sorted_mat, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, square=True, cbar_kws={"label": label})
    bounds = []
    labels = []
    start = 0
    for network, group in sorted_parcels.groupby("network", sort=False):
        end = start + len(group)
        bounds.append((start, end))
        labels.append(network)
        start = end
    centers = [(start + end) / 2 for start, end in bounds]
    for _, end in bounds[:-1]:
        plt.axhline(end, color="black", linewidth=0.5)
        plt.axvline(end, color="black", linewidth=0.5)
    plt.xticks(centers, labels, rotation=45, ha="right")
    plt.yticks(centers, labels, rotation=0)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=260)
    plt.close()


def save_network_stat_heatmap(mat: pd.DataFrame, out_path: Path, title: str, *, label: str, center: float | None = None) -> None:
    if mat.empty:
        return
    plt.figure(figsize=(7.2, 6.2))
    if center is None:
        sns.heatmap(mat, cmap="viridis", annot=True, fmt=".3f", cbar_kws={"label": label})
    else:
        sns.heatmap(mat, cmap="coolwarm", center=center, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar_kws={"label": label})
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=260)
    plt.close()


def save_global_scatter_plots(global_joined: pd.DataFrame, global_stats: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    value_kinds = [kind for kind in VALUE_KINDS if kind in set(global_joined["value_kind"])]
    columns = [("left", "global_within"), ("left", "global_between"), ("right", "global_within"), ("right", "global_between")]
    for drug in sorted(global_joined["drug"].dropna().unique()):
        fig, axes = plt.subplots(len(value_kinds), len(columns), figsize=(5.0 * len(columns), 4.2 * len(value_kinds)), squeeze=False)
        drug_df = global_joined[global_joined["drug"] == drug]
        for row_idx, value_kind in enumerate(value_kinds):
            for col_idx, (hemisphere, scope) in enumerate(columns):
                ax = axes[row_idx, col_idx]
                group = drug_df[
                    (drug_df["hemisphere"] == hemisphere)
                    & (drug_df["value_kind"] == value_kind)
                    & (drug_df["scope"] == scope)
                ]
                if group.empty:
                    ax.axis("off")
                    continue
                sns.regplot(
                    data=group,
                    x="original_value_z",
                    y="recon_value_z",
                    ci=95,
                    scatter_kws={"s": 38, "alpha": 0.82, "edgecolor": "white", "linewidths": 0.45},
                    line_kws={"color": "#222222", "linewidth": 1.4},
                    color="#4c72b0" if hemisphere == "left" else "#dd8452",
                    ax=ax,
                )
                finite_vals = pd.concat([group["original_value_z"], group["recon_value_z"]]).replace([np.inf, -np.inf], np.nan).dropna()
                if not finite_vals.empty:
                    lo = float(finite_vals.min())
                    hi = float(finite_vals.max())
                    pad = max((hi - lo) * 0.08, 1e-4)
                    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="gray", linewidth=0.9)
                    ax.set_xlim(lo - pad, hi + pad)
                    ax.set_ylim(lo - pad, hi + pad)
                stat = global_stats[
                    (global_stats["drug"] == drug)
                    & (global_stats["hemisphere"] == hemisphere)
                    & (global_stats["value_kind"] == value_kind)
                    & (global_stats["scope"] == scope)
                ]
                if not stat.empty and pd.notna(stat.iloc[0]["pearson_r"]):
                    stat_row = stat.iloc[0]
                    label = f"r={stat_row.pearson_r:.2f}\np={stat_row.pearson_p:.2g}\nMAE={stat_row.mean_abs_error_z:.4f}"
                    ax.text(0.04, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=9)
                ax.set_title(f"{hemisphere} | {scope.replace('global_', '')}")
                ax.set_xlabel("Original (z)" if row_idx == len(value_kinds) - 1 else "")
                ax.set_ylabel(f"{value_kind}\nRecon (z)" if col_idx == 0 else "")
        fig.suptitle(f"{drug}: inter-subject original vs reconstructed Global FC", y=1.01, fontsize=15)
        fig.tight_layout()
        fig.savefig(out_dir / f"global_scatter_combined_{drug}.png", dpi=280, bbox_inches="tight")
        plt.close(fig)


def save_combined_network_heatmap(
    r_mat: pd.DataFrame,
    mae_mat: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if r_mat.empty:
        return
    annot = pd.DataFrame("", index=r_mat.index, columns=r_mat.columns)
    for row_label in r_mat.index:
        for col_label in r_mat.columns:
            r_value = r_mat.loc[row_label, col_label]
            mae_value = mae_mat.loc[row_label, col_label] if row_label in mae_mat.index and col_label in mae_mat.columns else np.nan
            if np.isfinite(r_value):
                annot.loc[row_label, col_label] = f"r={r_value:.2f}\nMAE={mae_value:.3f}" if np.isfinite(mae_value) else f"r={r_value:.2f}"

    plt.figure(figsize=(8.4, 7.4))
    ax = sns.heatmap(
        r_mat,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot,
        fmt="",
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
        annot_kws={"fontsize": 8},
    )
    for idx in range(min(len(r_mat.index), len(r_mat.columns))):
        ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor="black", linewidth=2.0))
    ax.text(
        0.0,
        -0.12,
        "Diagonal cells are within-network; off-diagonal cells are between-network. Text shows Pearson r and MAE in Fisher-z units.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close()


def save_network_heatmaps(network_stats: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for keys, group in network_stats.groupby(["drug", "hemisphere", "value_kind"], sort=False):
        drug, hemisphere, value_kind = keys
        r_mat = network_matrix_from_rows(group, "pearson_r")
        mae_mat = network_matrix_from_rows(group, "mean_abs_error_z")
        save_combined_network_heatmap(
            r_mat,
            mae_mat,
            out_dir / f"network_r_and_mae_{drug}_{hemisphere}_{value_kind}.png",
            f"{drug} {hemisphere} {value_kind}: original vs recon network FC",
        )


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
    dataset_out = out_dir / spec.drug_label
    dataset_out.mkdir(parents=True, exist_ok=True)

    network_rows = []
    global_rows = []
    matrix_stats_rows = []

    for hemisphere in hemispheres:
        subject_rows = []
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
        hemi_out = dataset_out / hemisphere
        hemi_out.mkdir(parents=True, exist_ok=True)

        parcels = load_metadata(original_entries, spec.drug_condition, subjects[0], hemisphere)
        recon_parcels = load_metadata(recon_entries, spec.drug_condition, subjects[0], hemisphere)
        assert_metadata_match(parcels, recon_parcels)
        n_parcels = len(parcels)

        stacks: dict[str, list[np.ndarray]] = {f"original_{kind}": [] for kind in VALUE_KINDS}
        stacks.update({f"recon_{kind}": [] for kind in VALUE_KINDS})

        for subid in subjects:
            original_drug = load_matrix(original_entries, spec.drug_condition, subid, hemisphere, clip_eps)
            original_placebo = load_matrix(original_entries, spec.placebo_condition, subid, hemisphere, clip_eps)
            recon_drug = load_matrix(recon_entries, spec.drug_condition, subid, hemisphere, clip_eps)
            recon_placebo = load_matrix(recon_entries, spec.placebo_condition, subid, hemisphere, clip_eps)
            if any(mat.shape != (n_parcels, n_parcels) for mat in [original_drug, original_placebo, recon_drug, recon_placebo]):
                raise ValueError(f"Matrix shape mismatch for {spec.drug_label} {subid} {hemisphere}")

            matrices = {
                "original_drug": original_drug,
                "original_placebo": original_placebo,
                "original_delta": original_drug - original_placebo,
                "recon_drug": recon_drug,
                "recon_placebo": recon_placebo,
                "recon_delta": recon_drug - recon_placebo,
            }
            for key, mat in matrices.items():
                stacks[key].append(mat.astype(np.float32, copy=False))

            for source in ["original", "recon"]:
                for value_kind in VALUE_KINDS:
                    net = network_metric(matrices[f"{source}_{value_kind}"], parcels)
                    net["drug"] = spec.drug_label
                    net["comparison"] = f"{spec.drug_condition}-{spec.placebo_condition}"
                    net["hemisphere"] = hemisphere
                    net["subid"] = subid
                    net["source"] = source
                    net["value_kind"] = value_kind
                    network_rows.append(net)

                    glob = global_metric_from_network(net)
                    glob["drug"] = spec.drug_label
                    glob["comparison"] = f"{spec.drug_condition}-{spec.placebo_condition}"
                    glob["hemisphere"] = hemisphere
                    glob["subid"] = subid
                    glob["source"] = source
                    glob["value_kind"] = value_kind
                    global_rows.append(glob)

            subject_rows.append({"drug": spec.drug_label, "hemisphere": hemisphere, "subid": subid})

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            mean_mats = {name: np.nanmean(np.stack(values, axis=0), axis=0) for name, values in stacks.items()}
        for name, mat in mean_mats.items():
            np.save(hemi_out / f"{name}_mean_z.npy", mat)
            np.save(hemi_out / f"{name}_mean_r.npy", np.tanh(mat))
        matrix_long(mean_mats, parcels).to_csv(hemi_out / "matrix_level_mean_edges_z_and_r.csv", index=False)
        pd.DataFrame(subject_rows).to_csv(hemi_out / "paired_subjects.csv", index=False)

        for value_kind in VALUE_KINDS:
            save_matrix_heatmap(
                mean_mats[f"original_{value_kind}"],
                parcels,
                hemi_out / "figures" / f"matrix_original_{value_kind}_mean_z.png",
                f"{spec.drug_label} {hemisphere} original {value_kind} mean FC",
                "Mean FC (Fisher z)",
            )
            save_matrix_heatmap(
                mean_mats[f"recon_{value_kind}"],
                parcels,
                hemi_out / "figures" / f"matrix_recon_{value_kind}_mean_z.png",
                f"{spec.drug_label} {hemisphere} recon {value_kind} mean FC",
                "Mean FC (Fisher z)",
            )
            save_matrix_heatmap(
                mean_mats[f"recon_{value_kind}"] - mean_mats[f"original_{value_kind}"],
                parcels,
                hemi_out / "figures" / f"matrix_recon_minus_original_{value_kind}_z.png",
                f"{spec.drug_label} {hemisphere} recon - original {value_kind}",
                "Error (Fisher z)",
            )

        edge_mask = np.triu(np.ones((n_parcels, n_parcels), dtype=bool), k=1)
        for value_kind in VALUE_KINDS:
            stats_row = {
                "drug": spec.drug_label,
                "comparison": f"{spec.drug_condition}-{spec.placebo_condition}",
                "hemisphere": hemisphere,
                "value_kind": value_kind,
                "level": "matrix_spatial_pattern",
            }
            stats_row.update(
                agreement_stats(
                    mean_mats[f"original_{value_kind}"][edge_mask],
                    mean_mats[f"recon_{value_kind}"][edge_mask],
                    min_n=min_n,
                )
            )
            matrix_stats_rows.append(stats_row)

    network_df = pd.concat(network_rows, ignore_index=True)
    global_df = pd.concat(global_rows, ignore_index=True)
    network_df.to_csv(dataset_out / "network_subject_values_z.csv", index=False)
    global_df.to_csv(dataset_out / "global_subject_values_z.csv", index=False)
    pd.DataFrame(matrix_stats_rows).to_csv(dataset_out / "matrix_level_spatial_pattern_correlations.csv", index=False)

    return {
        "drug": spec.drug_label,
        "original_root": str(spec.original_root),
        "recon_root": str(spec.recon_root),
        "n_original_entries": int(len(original_entries)),
        "n_recon_entries": int(len(recon_entries)),
        "n_network_value_rows": int(len(network_df)),
        "n_global_value_rows": int(len(global_df)),
        "n_matrix_stats_rows": int(len(matrix_stats_rows)),
    }


def joined_network_stats(network_df: pd.DataFrame, min_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["drug", "comparison", "hemisphere", "subid", "value_kind", "type", "network_a", "network_b"]
    original = network_df[network_df["source"] == "original"][keys + ["value_z"]].rename(columns={"value_z": "original_value_z"})
    recon = network_df[network_df["source"] == "recon"][keys + ["value_z"]].rename(columns={"value_z": "recon_value_z"})
    joined = original.merge(recon, on=keys, how="inner")
    joined["error_z"] = joined["recon_value_z"] - joined["original_value_z"]
    rows = []
    group_keys = ["drug", "comparison", "hemisphere", "value_kind", "type", "network_a", "network_b"]
    for group_values, group in joined.groupby(group_keys, sort=False):
        row = dict(zip(group_keys, group_values))
        row.update(agreement_stats(group["original_value_z"].to_numpy(), group["recon_value_z"].to_numpy(), min_n=min_n))
        rows.append(row)
    return joined, pd.DataFrame(rows)


def joined_global_stats(global_df: pd.DataFrame, min_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["drug", "comparison", "hemisphere", "subid", "value_kind", "scope"]
    original = global_df[global_df["source"] == "original"][keys + ["value_z"]].rename(columns={"value_z": "original_value_z"})
    recon = global_df[global_df["source"] == "recon"][keys + ["value_z"]].rename(columns={"value_z": "recon_value_z"})
    joined = original.merge(recon, on=keys, how="inner")
    joined["error_z"] = joined["recon_value_z"] - joined["original_value_z"]
    rows = []
    group_keys = ["drug", "comparison", "hemisphere", "value_kind", "scope"]
    for group_values, group in joined.groupby(group_keys, sort=False):
        row = dict(zip(group_keys, group_values))
        row.update(agreement_stats(group["original_value_z"].to_numpy(), group["recon_value_z"].to_numpy(), min_n=min_n))
        rows.append(row)
    return joined, pd.DataFrame(rows)


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
        "note": "All averaging, deltas, correlations, and error metrics are computed in Fisher-z space. Mean z values are also back-transformed to r for display columns/files.",
        "datasets": [],
    }

    all_network = []
    all_global = []
    all_matrix_stats = []
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
        dataset_out = args.out_dir / spec.drug_label
        all_network.append(pd.read_csv(dataset_out / "network_subject_values_z.csv"))
        all_global.append(pd.read_csv(dataset_out / "global_subject_values_z.csv"))
        all_matrix_stats.append(pd.read_csv(dataset_out / "matrix_level_spatial_pattern_correlations.csv"))

    network_values = pd.concat(all_network, ignore_index=True)
    global_values = pd.concat(all_global, ignore_index=True)
    matrix_stats = pd.concat(all_matrix_stats, ignore_index=True)
    network_joined, network_stats = joined_network_stats(network_values, min_n=args.min_n)
    global_joined, global_stats = joined_global_stats(global_values, min_n=args.min_n)

    network_values.to_csv(args.out_dir / "network_subject_values_z.csv", index=False)
    global_values.to_csv(args.out_dir / "global_subject_values_z.csv", index=False)
    matrix_stats.to_csv(args.out_dir / "matrix_level_spatial_pattern_correlations.csv", index=False)
    network_joined.to_csv(args.out_dir / "network_original_recon_joined_z.csv", index=False)
    network_stats.to_csv(args.out_dir / "network_original_recon_correlations_z.csv", index=False)
    global_joined.to_csv(args.out_dir / "global_original_recon_joined_z.csv", index=False)
    global_stats.to_csv(args.out_dir / "global_original_recon_correlations_z.csv", index=False)

    save_global_scatter_plots(global_joined, global_stats, args.out_dir / "figures" / "global_scatter")
    save_network_heatmaps(network_stats, args.out_dir / "figures" / "network_heatmaps")

    metadata.update(
        {
            "n_network_joined_rows": int(len(network_joined)),
            "n_network_stats_rows": int(len(network_stats)),
            "n_global_joined_rows": int(len(global_joined)),
            "n_global_stats_rows": int(len(global_stats)),
            "n_matrix_stats_rows": int(len(matrix_stats)),
        }
    )
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote Fisher-z FC correlation analysis to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
