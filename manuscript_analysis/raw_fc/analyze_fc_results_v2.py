#!/usr/bin/env python3
"""Analyze Workbench pconn results and generate subject-level / group-level FC summaries.

Supports:
- independent or paired t-tests (paired by subject/pair ID)
- optional multiple-comparison handling: none or Benjamini-Hochberg FDR
- subject-level QC plots and group-level statistic plots
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.cifti2.cifti2_axes import ParcelsAxis
from scipy import stats
from tqdm import tqdm

LEFT_DEFAULT = r"(^L_|^LH_|^Left|^CORTEX_LEFT|^ctx-lh-|^lh_|_LH_|-lh-|\bLH\b|left)"
RIGHT_DEFAULT = r"(^R_|^RH_|^Right|^CORTEX_RIGHT|^ctx-rh-|^rh_|_RH_|-rh-|\bRH\b|right)"
DEFAULT_PAIR_PATTERNS = [
    r"(sub-[A-Za-z0-9]+)",
    r"(S\d+)",
]
SUMMARY_METRICS = [
    "mean_abs_fc",
    "mean_fc",
    "mean_pos_fc",
    "mean_neg_fc",
    "fc_dispersion_sd",
    "fc_dispersion_iqr_abs",
    "mean_strength",
    "sd_strength",
]


@dataclass(frozen=True)
class GroupSpec:
    label: str
    pconn_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Workbench pconn results.")
    parser.add_argument("--group-a-dir", required=True, type=Path)
    parser.add_argument("--group-b-dir", required=True, type=Path)
    parser.add_argument("--group-a-label", default="drug")
    parser.add_argument("--group-b-label", default="pcb")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--left-regex", default=LEFT_DEFAULT)
    parser.add_argument("--right-regex", default=RIGHT_DEFAULT)
    parser.add_argument(
        "--test-mode",
        choices=["independent", "paired"],
        default="independent",
        help="Use independent-samples test or pair subjects by pair ID and use paired t-test.",
    )
    parser.add_argument(
        "--pair-id-regex",
        default=None,
        help=(
            "Optional regex with one capture group used to extract a pairing ID from the pconn filename stem. "
            "Examples: '(S\\d+)' or '(sub-[^_]+)'. If omitted, common patterns such as sub-XX or S01 are tried."
        ),
    )
    parser.add_argument(
        "--multiple-comparison",
        choices=["none", "fdr"],
        default="none",
        help="How to handle multiple comparisons in summary/edgewise outputs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha threshold used for significant masks and summary tables.",
    )
    parser.add_argument(
        "--vortex-csv",
        type=Path,
        help=(
            "Optional subject-level vortex metrics CSV with columns: subject_id, group, hemisphere, "
            "and one or more metric columns."
        ),
    )
    parser.add_argument(
        "--save-subject-heatmaps",
        action="store_true",
        help="Save subject-level LL and RR heatmaps for quick QC.",
    )
    parser.add_argument(
        "--max-subject-heatmaps",
        type=int,
        default=12,
        help="Maximum number of subjects per group to save as subject-level heatmaps.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("analyze_fc_results")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def collect_pconn_files(pconn_dir: Path) -> List[Path]:
    files = sorted(pconn_dir.glob("*.pconn.nii"))
    if not files:
        nested = pconn_dir / "pconn"
        files = sorted(nested.glob("*.pconn.nii"))
    return files


def strip_known_suffixes(name: str) -> str:
    suffixes = [".pconn.nii", ".ptseries.nii", ".dtseries.nii", ".dtseries.nii.nii"]
    result = name
    for suffix in suffixes:
        if result.endswith(suffix):
            return result[: -len(suffix)]
    return Path(name).stem


def infer_pair_id(subject_id: str, pair_id_regex: Optional[str]) -> str:
    patterns = [pair_id_regex] if pair_id_regex else DEFAULT_PAIR_PATTERNS
    for pattern in patterns:
        if not pattern:
            continue
        m = re.search(pattern, subject_id, flags=re.IGNORECASE)
        if m:
            if m.groups():
                return str(m.group(1))
            return str(m.group(0))
    return subject_id


def get_parcel_names(axis: ParcelsAxis) -> List[str]:
    names = getattr(axis, "name", None)
    if names is not None:
        return [str(x) for x in np.asarray(names)]
    return [str(item.name) for item in axis]


def hemisphere_indices(names: Sequence[str], left_regex: str, right_regex: str) -> Dict[str, np.ndarray]:
    left = np.array([bool(re.search(left_regex, n, re.IGNORECASE)) for n in names])
    right = np.array([bool(re.search(right_regex, n, re.IGNORECASE)) for n in names])

    if left.sum() == 0 and right.sum() == 0:
        left = np.array([("_LH_" in n) or n.startswith("LH_") or n.lower().startswith("lh_") for n in names])
        right = np.array([("_RH_" in n) or n.startswith("RH_") or n.lower().startswith("rh_") for n in names])

    overlap = left & right
    if overlap.any():
        raise ValueError(
            "Some parcels matched both left and right hemisphere regex. "
            "Please use more specific --left-regex / --right-regex."
        )
    if left.sum() == 0 or right.sum() == 0:
        sample = ", ".join(map(str, list(names[:8])))
        raise ValueError(
            "Could not split parcels into left/right hemispheres from parcel names. "
            "Pass --left-regex / --right-regex that match your atlas names. "
            f"Sample parcel names: {sample}"
        )
    return {"left": np.where(left)[0], "right": np.where(right)[0]}


def finite_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    tri = matrix[np.triu_indices(matrix.shape[0], k=1)]
    tri = tri[np.isfinite(tri)]
    return tri


def compute_metrics(values: np.ndarray, matrix: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {metric: math.nan for metric in SUMMARY_METRICS}

    abs_values = np.abs(values)
    pos_values = values[values > 0]
    neg_values = values[values < 0]

    abs_matrix = np.abs(matrix.copy())
    np.fill_diagonal(abs_matrix, np.nan)
    node_strength = np.nanmean(abs_matrix, axis=1)

    return {
        "mean_abs_fc": float(np.nanmean(abs_values)),
        "mean_fc": float(np.nanmean(values)),
        "mean_pos_fc": float(np.nanmean(pos_values)) if pos_values.size else math.nan,
        "mean_neg_fc": float(np.nanmean(neg_values)) if neg_values.size else math.nan,
        "fc_dispersion_sd": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
        "fc_dispersion_iqr_abs": float(np.nanpercentile(abs_values, 75) - np.nanpercentile(abs_values, 25)),
        "mean_strength": float(np.nanmean(node_strength)),
        "sd_strength": float(np.nanstd(node_strength, ddof=1)) if np.isfinite(node_strength).sum() > 1 else 0.0,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    pv = p[mask]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    unsorted = np.empty_like(q)
    unsorted[order] = q
    out[mask] = unsorted
    return out


def apply_multiple_comparison(p_values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return np.asarray(p_values, dtype=float)
    if method == "fdr":
        return bh_fdr(np.asarray(p_values, dtype=float))
    raise ValueError(f"Unsupported multiple-comparison method: {method}")


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled_num = (len(a) - 1) * va + (len(b) - 1) * vb
    pooled_den = len(a) + len(b) - 2
    if pooled_den <= 0:
        return math.nan
    pooled_sd = math.sqrt(pooled_num / pooled_den) if pooled_num >= 0 else math.nan
    if not np.isfinite(pooled_sd) or pooled_sd == 0:
        return 0.0
    d = (float(np.mean(a)) - float(np.mean(b))) / pooled_sd
    correction = 1.0 - (3.0 / (4.0 * (len(a) + len(b)) - 9.0)) if (len(a) + len(b)) > 2 else 1.0
    return float(d * correction)


def cohen_dz(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if len(d) < 2:
        return math.nan
    sd = np.std(d, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def save_matrix_plot(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(6, 5))
    display_matrix = np.array(matrix, dtype=float)
    plt.imshow(display_matrix, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Parcel")
    plt.ylabel("Parcel")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_subject_metric_plot(
    df: pd.DataFrame,
    out_path: Path,
    metric: str,
    hemi: str,
    group_a: str,
    group_b: str,
    test_mode: str,
) -> None:
    hemi_df = df[df["hemisphere"] == hemi].copy()
    groups = [
        hemi_df.loc[hemi_df["group"] == group_a, metric].dropna().to_numpy(dtype=float),
        hemi_df.loc[hemi_df["group"] == group_b, metric].dropna().to_numpy(dtype=float),
    ]

    plt.figure(figsize=(7.5, 4.8))
    plt.boxplot(groups, positions=[1, 2], widths=0.45)
    for idx, values in enumerate(groups, start=1):
        if values.size == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, num=values.size) if values.size > 1 else np.array([0.0])
        plt.scatter(np.full(values.size, idx) + jitter, values, alpha=0.75, s=24)

    if test_mode == "paired":
        a_df = hemi_df.loc[hemi_df["group"] == group_a, ["pair_id", metric]].dropna()
        b_df = hemi_df.loc[hemi_df["group"] == group_b, ["pair_id", metric]].dropna()
        merged = a_df.merge(b_df, on="pair_id", suffixes=("_a", "_b"))
        for _, row in merged.iterrows():
            plt.plot([1, 2], [row[f"{metric}_a"], row[f"{metric}_b"]], alpha=0.25)

    plt.xticks([1, 2], [group_a, group_b])
    plt.ylabel(metric)
    plt.title(f"{metric} ({hemi})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_metric_bar_summary(stats_df: pd.DataFrame, out_path: Path, hemi: str, stat_col: str, title: str) -> None:
    hemi_df = stats_df[stats_df["hemisphere"] == hemi].copy()
    if hemi_df.empty or stat_col not in hemi_df.columns:
        return
    plot_df = hemi_df.copy()
    if stat_col in ["effect_size_abs", "neg_log10_p"]:
        plot_df = plot_df.sort_values(stat_col, ascending=False)
    elif stat_col == "p_value":
        plot_df = plot_df.sort_values(stat_col, ascending=True)
    elif stat_col == "t_stat":
        plot_df["abs_t"] = plot_df["t_stat"].abs()
        plot_df = plot_df.sort_values("abs_t", ascending=False)
    plot_df = plot_df.head(10)

    plt.figure(figsize=(8.5, max(4.5, 0.55 * len(plot_df))))
    y = np.arange(len(plot_df))
    vals = plot_df[stat_col].to_numpy(dtype=float)
    plt.barh(y, vals)
    plt.yticks(y, plot_df["metric"].tolist())
    if stat_col == "t_stat":
        plt.axvline(0.0, linewidth=1)
        xlabel = "t statistic"
    elif stat_col == "effect_size":
        plt.axvline(0.0, linewidth=1)
        xlabel = "effect size"
    elif stat_col == "effect_size_abs":
        xlabel = "|effect size|"
    elif stat_col == "neg_log10_p":
        xlabel = "-log10(p)"
    else:
        xlabel = stat_col
    plt.xlabel(xlabel)
    plt.title(f"{title} ({hemi})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_stats_table_plot(stats_df: pd.DataFrame, out_path: Path, hemi: str, multiple_comparison: str, alpha: float) -> None:
    hemi_df = stats_df[stats_df["hemisphere"] == hemi].copy()
    if hemi_df.empty:
        return
    threshold_col = "corrected_p_value"
    hemi_df = hemi_df.sort_values([threshold_col, "effect_size_abs"], ascending=[True, False]).head(12)
    cols = ["metric", "group_a_mean", "group_b_mean", "delta_mean", "t_stat", "p_value", "corrected_p_value", "effect_size"]
    table_df = hemi_df[cols].copy().round(4)

    fig_h = max(3.5, 0.38 * (len(table_df) + 2))
    plt.figure(figsize=(12, fig_h))
    plt.axis("off")
    title = f"Summary statistics ({hemi}) | mc={multiple_comparison}, alpha={alpha:.3g}"
    plt.title(title, loc="left")
    table = plt.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def compute_edgewise_stats_independent(group_a_mats: Sequence[np.ndarray], group_b_mats: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stack_a = np.stack(group_a_mats, axis=0)
    stack_b = np.stack(group_b_mats, axis=0)
    t_map, p_map = stats.ttest_ind(stack_a, stack_b, axis=0, equal_var=False, nan_policy="omit")
    return np.asarray(t_map, dtype=float), np.asarray(p_map, dtype=float)


def compute_edgewise_stats_paired(mats_a: Dict[str, np.ndarray], mats_b: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    common = sorted(set(mats_a.keys()) & set(mats_b.keys()))
    if len(common) < 2:
        raise ValueError("Need at least 2 paired subjects to run paired edgewise t-test.")
    stack_a = np.stack([mats_a[k] for k in common], axis=0)
    stack_b = np.stack([mats_b[k] for k in common], axis=0)
    t_map, p_map = stats.ttest_rel(stack_a, stack_b, axis=0, nan_policy="omit")
    return np.asarray(t_map, dtype=float), np.asarray(p_map, dtype=float)


def load_subject_metrics_and_matrices(
    group: GroupSpec,
    left_regex: str,
    right_regex: str,
    pair_id_regex: Optional[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, List[np.ndarray]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, np.ndarray]]]:
    files = collect_pconn_files(group.pconn_dir)
    if not files:
        raise FileNotFoundError(f"No .pconn.nii files found in {group.pconn_dir}")

    rows: List[Dict[str, object]] = []
    matrices: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
    subjects_by_hemi: Dict[str, List[str]] = {"left": [], "right": []}
    parcels_by_hemi: Dict[str, List[str]] = {}
    matrices_by_pair: Dict[str, Dict[str, np.ndarray]] = {"left": {}, "right": {}}

    for file_path in tqdm(files, desc=f"Loading {group.label}", unit="file"):
        img = nib.load(str(file_path))
        data = np.asarray(img.dataobj, dtype=float)
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f"Expected square pconn matrix in {file_path}, got shape {data.shape}")

        axis = img.header.get_axis(0)
        if not isinstance(axis, ParcelsAxis):
            raise ValueError(f"Expected ParcelsAxis in {file_path}, got {type(axis)}")

        names = get_parcel_names(axis)
        hemi_map = hemisphere_indices(names, left_regex=left_regex, right_regex=right_regex)

        subject_id = strip_known_suffixes(file_path.name)
        pair_id = infer_pair_id(subject_id, pair_id_regex)

        for hemi in ["left", "right"]:
            idx = hemi_map[hemi]
            hemi_names = [names[i] for i in idx]
            hemi_matrix = np.asarray(data[np.ix_(idx, idx)], dtype=float)

            if hemi not in parcels_by_hemi:
                parcels_by_hemi[hemi] = hemi_names
            elif parcels_by_hemi[hemi] != hemi_names:
                raise ValueError(
                    f"Parcel ordering mismatch within group {group.label} for hemisphere {hemi}. File: {file_path}"
                )

            tri = finite_upper_triangle(hemi_matrix)
            metric_row = {
                "group": group.label,
                "subject_id": subject_id,
                "pair_id": pair_id,
                "hemisphere": hemi,
            }
            metric_row.update(compute_metrics(tri, hemi_matrix))
            rows.append(metric_row)

            matrices[hemi].append(hemi_matrix)
            subjects_by_hemi[hemi].append(subject_id)
            matrices_by_pair[hemi][pair_id] = hemi_matrix

    return pd.DataFrame(rows), matrices, subjects_by_hemi, parcels_by_hemi, matrices_by_pair


def compare_groups(
    df: pd.DataFrame,
    group_a_label: str,
    group_b_label: str,
    test_mode: str,
    multiple_comparison: str,
    alpha: float,
) -> pd.DataFrame:
    results: List[Dict[str, object]] = []

    for hemi in sorted(df["hemisphere"].unique()):
        hemi_df = df[df["hemisphere"] == hemi]
        df_a = hemi_df[hemi_df["group"] == group_a_label].copy()
        df_b = hemi_df[hemi_df["group"] == group_b_label].copy()

        for metric in SUMMARY_METRICS:
            if metric not in hemi_df.columns:
                continue

            if test_mode == "paired":
                pair_a = df_a[["pair_id", metric]].dropna()
                pair_b = df_b[["pair_id", metric]].dropna()
                merged = pair_a.merge(pair_b, on="pair_id", suffixes=("_a", "_b"))
                a = merged[f"{metric}_a"].to_numpy(dtype=float)
                b = merged[f"{metric}_b"].to_numpy(dtype=float)
                if len(a) < 2:
                    continue
                t_stat, p_value = stats.ttest_rel(a, b, nan_policy="omit")
                effect_size = cohen_dz(a, b)
                effect_name = "cohen_dz"
                n_a = n_b = int(len(a))
                delta_mean = float(np.mean(a - b))
            else:
                a = df_a[metric].dropna().to_numpy(dtype=float)
                b = df_b[metric].dropna().to_numpy(dtype=float)
                if len(a) == 0 or len(b) == 0:
                    continue
                t_stat, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                effect_size = hedges_g(a, b)
                effect_name = "hedges_g"
                n_a = int(len(a))
                n_b = int(len(b))
                delta_mean = float(np.mean(a) - np.mean(b))

            results.append(
                {
                    "hemisphere": hemi,
                    "metric": metric,
                    "test_mode": test_mode,
                    "effect_name": effect_name,
                    "group_a_mean": float(np.mean(a)),
                    "group_b_mean": float(np.mean(b)),
                    "group_a_std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                    "group_b_std": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
                    "delta_mean": delta_mean,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "effect_size": float(effect_size),
                    "n_group_a": n_a,
                    "n_group_b": n_b,
                }
            )

    out = pd.DataFrame(results)
    if out.empty:
        return out
    out["corrected_p_value"] = apply_multiple_comparison(out["p_value"].to_numpy(dtype=float), multiple_comparison)
    out["significant"] = out["corrected_p_value"] < alpha
    out["effect_size_abs"] = out["effect_size"].abs()
    out["neg_log10_p"] = -np.log10(np.clip(out["p_value"].to_numpy(dtype=float), 1e-300, None))
    out["mc_method"] = multiple_comparison
    return out


def maybe_compare_with_vortex(
    fc_stats: pd.DataFrame,
    vortex_csv: Optional[Path],
    logger: logging.Logger,
    test_mode: str,
    multiple_comparison: str,
    alpha: float,
) -> Optional[pd.DataFrame]:
    if vortex_csv is None:
        return None

    vortex = pd.read_csv(vortex_csv)
    required = {"subject_id", "group", "hemisphere"}
    missing = required - set(vortex.columns)
    if missing:
        raise ValueError(f"Vortex CSV is missing required columns: {sorted(missing)}")

    value_cols = [c for c in vortex.columns if c not in required]
    if not value_cols:
        raise ValueError("Vortex CSV must include at least one metric column.")

    rows: List[Dict[str, object]] = []
    for hemi in sorted(vortex["hemisphere"].unique()):
        hemi_df = vortex[vortex["hemisphere"] == hemi].copy()
        groups = list(hemi_df["group"].dropna().unique())
        if len(groups) != 2:
            logger.warning("Skipping hemisphere %s in vortex CSV because it does not contain exactly 2 groups.", hemi)
            continue
        g1, g2 = groups[0], groups[1]
        for metric in value_cols:
            if test_mode == "paired":
                a_df = hemi_df.loc[hemi_df["group"] == g1, ["subject_id", metric]].dropna()
                b_df = hemi_df.loc[hemi_df["group"] == g2, ["subject_id", metric]].dropna()
                merged = a_df.merge(b_df, on="subject_id", suffixes=("_a", "_b"))
                a = merged[f"{metric}_a"].to_numpy(dtype=float)
                b = merged[f"{metric}_b"].to_numpy(dtype=float)
                if len(a) < 2:
                    continue
                t_stat, p_value = stats.ttest_rel(a, b, nan_policy="omit")
                effect_size = cohen_dz(a, b)
                effect_name = "cohen_dz"
                n_a = n_b = int(len(a))
            else:
                a = hemi_df.loc[hemi_df["group"] == g1, metric].to_numpy(dtype=float)
                b = hemi_df.loc[hemi_df["group"] == g2, metric].to_numpy(dtype=float)
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if len(a) == 0 or len(b) == 0:
                    continue
                t_stat, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                effect_size = hedges_g(a, b)
                effect_name = "hedges_g"
                n_a = int(len(a))
                n_b = int(len(b))

            rows.append(
                {
                    "method": "vortex",
                    "hemisphere": hemi,
                    "metric": metric,
                    "group_a": g1,
                    "group_b": g2,
                    "test_mode": test_mode,
                    "effect_name": effect_name,
                    "p_value": float(p_value),
                    "corrected_p_value": float(p_value),
                    "t_stat": float(t_stat),
                    "effect_size": float(effect_size),
                    "n_group_a": n_a,
                    "n_group_b": n_b,
                }
            )

    fc_rows = pd.DataFrame()
    if not fc_stats.empty:
        fc_rows = fc_stats[["hemisphere", "metric", "test_mode", "effect_name", "p_value", "corrected_p_value", "t_stat", "effect_size", "n_group_a", "n_group_b"]].copy()
        fc_rows.insert(0, "method", "fc")
        fc_rows.insert(3, "group_a", pd.NA)
        fc_rows.insert(4, "group_b", pd.NA)

    vortex_rows = pd.DataFrame(rows)
    if fc_rows.empty and vortex_rows.empty:
        return None

    combined = pd.concat([fc_rows, vortex_rows], ignore_index=True, sort=False)
    combined["corrected_p_value"] = apply_multiple_comparison(combined["p_value"].to_numpy(dtype=float), multiple_comparison)
    combined["significant"] = combined["corrected_p_value"] < alpha
    combined["mc_method"] = multiple_comparison
    combined["neg_log10_p"] = -np.log10(np.clip(combined["p_value"].to_numpy(dtype=float), 1e-300, None))
    combined["effect_size_abs"] = combined["effect_size"].abs()
    return combined


def save_parcel_names(parcel_names: Dict[str, List[str]], out_dir: Path, prefix: str) -> None:
    ensure_dir(out_dir)
    for hemi, names in parcel_names.items():
        pd.DataFrame({"parcel_name": names}).to_csv(out_dir / f"{prefix}_{hemi}_parcel_names.csv", index=False)


def make_summary_plots(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
    group_a: str,
    group_b: str,
    test_mode: str,
    multiple_comparison: str,
    alpha: float,
) -> None:
    plot_dir = out_dir / "plots" / "summary_metrics"
    ensure_dir(plot_dir)

    for hemi in sorted(df["hemisphere"].unique()):
        for metric in SUMMARY_METRICS:
            if metric in df.columns:
                save_subject_metric_plot(
                    df,
                    plot_dir / f"{hemi}_{metric}.png",
                    metric,
                    hemi,
                    group_a,
                    group_b,
                    test_mode,
                )
        save_metric_bar_summary(stats_df, plot_dir / f"{hemi}_top_by_abs_effect.png", hemi, "effect_size_abs", "Top summary metrics by |effect size|")
        save_metric_bar_summary(stats_df, plot_dir / f"{hemi}_top_by_neglog10p.png", hemi, "neg_log10_p", "Top summary metrics by -log10(p)")
        save_metric_bar_summary(stats_df, plot_dir / f"{hemi}_top_by_tstat.png", hemi, "t_stat", "Top summary metrics by |t|")
        save_stats_table_plot(stats_df, plot_dir / f"{hemi}_stats_table.png", hemi, multiple_comparison, alpha)


def make_matrix_visuals(
    group_a_label: str,
    group_b_label: str,
    group_a_mats: Dict[str, List[np.ndarray]],
    group_b_mats: Dict[str, List[np.ndarray]],
    group_a_subjects: Dict[str, List[str]],
    group_b_subjects: Dict[str, List[str]],
    group_a_pair_mats: Dict[str, Dict[str, np.ndarray]],
    group_b_pair_mats: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    save_subject_heatmaps: bool,
    max_subject_heatmaps: int,
    test_mode: str,
    multiple_comparison: str,
    alpha: float,
) -> None:
    matrix_dir = out_dir / "plots" / "matrix_maps"
    ensure_dir(matrix_dir)

    for hemi in ["left", "right"]:
        if not group_a_mats[hemi] or not group_b_mats[hemi]:
            continue

        mean_a = np.nanmean(np.stack(group_a_mats[hemi], axis=0), axis=0)
        mean_b = np.nanmean(np.stack(group_b_mats[hemi], axis=0), axis=0)
        diff = mean_a - mean_b

        if test_mode == "paired":
            t_map, p_map = compute_edgewise_stats_paired(group_a_pair_mats[hemi], group_b_pair_mats[hemi])
            test_label = "paired t"
        else:
            t_map, p_map = compute_edgewise_stats_independent(group_a_mats[hemi], group_b_mats[hemi])
            test_label = "Welch t"

        corrected_map = apply_multiple_comparison(p_map, multiple_comparison)
        sig_mask = (corrected_map < alpha).astype(float)

        stacked = np.concatenate(
            [np.ravel(mean_a[np.isfinite(mean_a)]), np.ravel(mean_b[np.isfinite(mean_b)]), np.ravel(diff[np.isfinite(diff)])]
        )
        if stacked.size == 0:
            continue
        max_abs = float(np.nanmax(np.abs(stacked))) if np.isfinite(stacked).any() else None
        t_abs = float(np.nanmax(np.abs(t_map[np.isfinite(t_map)]))) if np.isfinite(t_map).any() else None

        save_matrix_plot(mean_a, matrix_dir / f"{hemi}_{group_a_label}_mean_matrix.png", f"{group_a_label} mean FC ({hemi})", vmin=-max_abs, vmax=max_abs)
        save_matrix_plot(mean_b, matrix_dir / f"{hemi}_{group_b_label}_mean_matrix.png", f"{group_b_label} mean FC ({hemi})", vmin=-max_abs, vmax=max_abs)
        save_matrix_plot(diff, matrix_dir / f"{hemi}_{group_a_label}_minus_{group_b_label}_mean_diff.png", f"Mean FC difference {group_a_label} - {group_b_label} ({hemi})", vmin=-max_abs, vmax=max_abs)
        save_matrix_plot(t_map, matrix_dir / f"{hemi}_edgewise_t_map.png", f"Edgewise {test_label} map ({hemi})", vmin=-t_abs, vmax=t_abs)
        save_matrix_plot(-np.log10(np.clip(p_map, 1e-300, None)), matrix_dir / f"{hemi}_edgewise_neglog10_p.png", f"Edgewise -log10(p) ({hemi})")
        save_matrix_plot(sig_mask, matrix_dir / f"{hemi}_edgewise_significant_mask.png", f"Edgewise {multiple_comparison}<alpha mask ({hemi})", vmin=0.0, vmax=1.0)

        pd.DataFrame(mean_a).to_csv(matrix_dir / f"{hemi}_{group_a_label}_mean_matrix.csv", index=False)
        pd.DataFrame(mean_b).to_csv(matrix_dir / f"{hemi}_{group_b_label}_mean_matrix.csv", index=False)
        pd.DataFrame(diff).to_csv(matrix_dir / f"{hemi}_{group_a_label}_minus_{group_b_label}_mean_diff.csv", index=False)
        pd.DataFrame(t_map).to_csv(matrix_dir / f"{hemi}_edgewise_t_map.csv", index=False)
        pd.DataFrame(p_map).to_csv(matrix_dir / f"{hemi}_edgewise_p_map.csv", index=False)
        pd.DataFrame(corrected_map).to_csv(matrix_dir / f"{hemi}_edgewise_corrected_p_map.csv", index=False)

        if save_subject_heatmaps:
            subj_dir = matrix_dir / "subject_examples" / hemi
            ensure_dir(subj_dir)
            for label, mats, subjects in [
                (group_a_label, group_a_mats[hemi], group_a_subjects[hemi]),
                (group_b_label, group_b_mats[hemi], group_b_subjects[hemi]),
            ]:
                count = min(max_subject_heatmaps, len(mats))
                for idx in range(count):
                    save_matrix_plot(
                        mats[idx],
                        subj_dir / f"{label}_{subjects[idx]}.png",
                        f"{label} :: {subjects[idx]} ({hemi})",
                        vmin=-max_abs,
                        vmax=max_abs,
                    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "logs" / "analyze_fc_results.log")

    logger.info("test_mode=%s | multiple_comparison=%s | alpha=%.4f", args.test_mode, args.multiple_comparison, args.alpha)
    if args.test_mode == "paired":
        logger.info("paired analysis enabled; pair_id_regex=%s", args.pair_id_regex or "<auto>")

    group_a = GroupSpec(label=args.group_a_label, pconn_dir=args.group_a_dir)
    group_b = GroupSpec(label=args.group_b_label, pconn_dir=args.group_b_dir)

    logger.info("Loading group A from %s", group_a.pconn_dir)
    df_a, mats_a, subjects_a, parcels_a, pair_mats_a = load_subject_metrics_and_matrices(
        group_a, args.left_regex, args.right_regex, args.pair_id_regex, logger
    )
    logger.info("Loading group B from %s", group_b.pconn_dir)
    df_b, mats_b, subjects_b, parcels_b, pair_mats_b = load_subject_metrics_and_matrices(
        group_b, args.left_regex, args.right_regex, args.pair_id_regex, logger
    )

    for hemi in ["left", "right"]:
        if hemi in parcels_a and hemi in parcels_b and parcels_a[hemi] != parcels_b[hemi]:
            raise ValueError(f"Parcel ordering mismatch between groups for hemisphere '{hemi}'.")

    if args.test_mode == "paired":
        common_ids = sorted(set(df_a["pair_id"]) & set(df_b["pair_id"]))
        logger.info("Found %d paired IDs shared by both groups. Example IDs: %s", len(common_ids), common_ids[:10])
        if len(common_ids) < 2:
            raise ValueError("Paired mode requested, but fewer than 2 shared pair IDs were found across groups.")

    parcel_dir = args.out_dir / "parcel_info"
    save_parcel_names(parcels_a, parcel_dir, f"{args.group_a_label}")
    save_parcel_names(parcels_b, parcel_dir, f"{args.group_b_label}")

    df = pd.concat([df_a, df_b], ignore_index=True)
    subject_csv = args.out_dir / "subject_level_fc_metrics.csv"
    df.to_csv(subject_csv, index=False)
    logger.info("Wrote subject-level metrics to %s", subject_csv)

    stats_df = compare_groups(df, args.group_a_label, args.group_b_label, args.test_mode, args.multiple_comparison, args.alpha)
    stats_csv = args.out_dir / "group_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Wrote group statistics to %s", stats_csv)

    method_df = maybe_compare_with_vortex(stats_df, args.vortex_csv, logger, args.test_mode, args.multiple_comparison, args.alpha)
    if method_df is not None:
        method_csv = args.out_dir / "method_comparison.csv"
        method_df.to_csv(method_csv, index=False)
        logger.info("Wrote method comparison table to %s", method_csv)

    make_summary_plots(df, stats_df, args.out_dir, args.group_a_label, args.group_b_label, args.test_mode, args.multiple_comparison, args.alpha)
    make_matrix_visuals(
        args.group_a_label,
        args.group_b_label,
        mats_a,
        mats_b,
        subjects_a,
        subjects_b,
        pair_mats_a,
        pair_mats_b,
        args.out_dir,
        args.save_subject_heatmaps,
        args.max_subject_heatmaps,
        args.test_mode,
        args.multiple_comparison,
        args.alpha,
    )

    logger.info("Saved plots under %s", args.out_dir / "plots")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
