#!/usr/bin/env python3
"""Analyze Workbench pconn results and generate subject-level / group-level FC summaries."""

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

LEFT_DEFAULT = r"^(L_|LH_|Left|CORTEX_LEFT|ctx-lh-|lh_)"
RIGHT_DEFAULT = r"^(R_|RH_|Right|CORTEX_RIGHT|ctx-rh-|rh_)"
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


@dataclass
class MetricResult:
    hemisphere: str
    metric: str
    group_a_mean: float
    group_b_mean: float
    group_a_std: float
    group_b_std: float
    delta_mean: float
    t_stat: float
    p_value: float
    hedges_g: float
    n_group_a: int
    n_group_b: int


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


def get_parcel_names(axis: ParcelsAxis) -> List[str]:
    names = getattr(axis, "name", None)
    if names is not None:
        return [str(x) for x in np.asarray(names)]
    return [str(item.name) for item in axis]


def hemisphere_indices(
    names: Sequence[str], left_regex: str, right_regex: str
) -> Dict[str, np.ndarray]:
    left = np.array([bool(re.search(left_regex, n, re.IGNORECASE)) for n in names])
    right = np.array([bool(re.search(right_regex, n, re.IGNORECASE)) for n in names])
    if left.sum() == 0 or right.sum() == 0:
        raise ValueError(
            "Could not split parcels into left/right hemispheres from parcel names. "
            "Pass --left-regex / --right-regex that match your atlas names."
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
        "fc_dispersion_iqr_abs": float(
            np.nanpercentile(abs_values, 75) - np.nanpercentile(abs_values, 25)
        ),
        "mean_strength": float(np.nanmean(node_strength)),
        "sd_strength": float(np.nanstd(node_strength, ddof=1))
        if np.isfinite(node_strength).sum() > 1
        else 0.0,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
) -> None:
    hemi_df = df[df["hemisphere"] == hemi].copy()
    group_order = [group_a, group_b]
    positions = [1, 2]
    groups = [
        hemi_df.loc[hemi_df["group"] == group_a, metric].dropna().to_numpy(dtype=float),
        hemi_df.loc[hemi_df["group"] == group_b, metric].dropna().to_numpy(dtype=float),
    ]

    plt.figure(figsize=(7, 4.5))
    plt.boxplot(groups, positions=positions, widths=0.45)
    for idx, values in enumerate(groups, start=1):
        if values.size == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, num=values.size) if values.size > 1 else np.array([0.0])
        plt.scatter(np.full(values.size, idx) + jitter, values, alpha=0.75, s=24)
    plt.xticks(positions, group_order)
    plt.ylabel(metric)
    plt.title(f"{metric} ({hemi})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_metric_bar_summary(
    stats_df: pd.DataFrame,
    out_path: Path,
    hemi: str,
    top_n: int = 8,
) -> None:
    hemi_df = stats_df[stats_df["hemisphere"] == hemi].copy()
    if hemi_df.empty:
        return
    hemi_df["abs_effect"] = hemi_df["hedges_g"].abs()
    hemi_df = hemi_df.sort_values(["q_value_bh", "abs_effect"], ascending=[True, False]).head(top_n)

    plt.figure(figsize=(8, max(4, 0.6 * len(hemi_df))))
    y = np.arange(len(hemi_df))
    plt.barh(y, hemi_df["hedges_g"].to_numpy(dtype=float))
    plt.yticks(y, hemi_df["metric"].tolist())
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Hedges' g (group A - group B)")
    plt.title(f"Top summary metrics by q / effect ({hemi})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def compute_edgewise_stats(group_a_mats: Sequence[np.ndarray], group_b_mats: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stack_a = np.stack(group_a_mats, axis=0)
    stack_b = np.stack(group_b_mats, axis=0)
    t_map, p_map = stats.ttest_ind(stack_a, stack_b, axis=0, equal_var=False, nan_policy="omit")
    return np.asarray(t_map, dtype=float), np.asarray(p_map, dtype=float)


def bh_fdr_matrix(p_matrix: np.ndarray) -> np.ndarray:
    mask = np.isfinite(p_matrix)
    flat = p_matrix[mask]
    q_matrix = np.full_like(p_matrix, np.nan, dtype=float)
    if flat.size == 0:
        return q_matrix
    q_flat = bh_fdr(flat)
    q_matrix[mask] = q_flat
    return q_matrix


def load_subject_metrics_and_matrices(
    group: GroupSpec,
    left_regex: str,
    right_regex: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, List[np.ndarray]], Dict[str, List[str]], Dict[str, List[str]]]:
    files = collect_pconn_files(group.pconn_dir)
    if not files:
        raise FileNotFoundError(f"No .pconn.nii files found in {group.pconn_dir}")

    rows: List[Dict[str, object]] = []
    hemi_matrices: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
    hemi_subjects: Dict[str, List[str]] = {"left": [], "right": []}
    hemi_parcel_names: Dict[str, List[str]] = {}

    for file_path in tqdm(files, desc=f"Loading {group.label}"):
        img = nib.load(str(file_path))
        axis0 = img.header.get_axis(0)
        if not isinstance(axis0, ParcelsAxis):
            raise TypeError(f"Expected ParcelsAxis in {file_path}, got {type(axis0)!r}")

        names = get_parcel_names(axis0)
        hemi_map = hemisphere_indices(names, left_regex=left_regex, right_regex=right_regex)
        matrix = np.asarray(img.dataobj, dtype=np.float64)
        subject_id = strip_known_suffixes(file_path.name)

        logger.info("Loaded %s (%s)", subject_id, file_path)

        for hemi_name, idx in hemi_map.items():
            submat = matrix[np.ix_(idx, idx)].copy()
            np.fill_diagonal(submat, np.nan)
            values = finite_upper_triangle(submat)
            metrics = compute_metrics(values, submat)
            row: Dict[str, object] = {
                "subject_id": subject_id,
                "group": group.label,
                "hemisphere": hemi_name,
                "n_parcels": int(len(idx)),
                "n_edges": int(values.size),
            }
            row.update(metrics)
            rows.append(row)
            hemi_matrices[hemi_name].append(submat)
            hemi_subjects[hemi_name].append(subject_id)
            current_names = [names[i] for i in idx]
            if hemi_name not in hemi_parcel_names:
                hemi_parcel_names[hemi_name] = current_names
            elif hemi_parcel_names[hemi_name] != current_names:
                raise ValueError(
                    f"Parcel ordering mismatch detected in hemisphere '{hemi_name}' for group {group.label}."
                )

    return pd.DataFrame(rows), hemi_matrices, hemi_subjects, hemi_parcel_names


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return math.nan
    nx, ny = len(x), len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return math.nan
    d = (np.mean(x) - np.mean(y)) / np.sqrt(pooled)
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    return float(correction * d)


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        prev = min(prev, value)
        adjusted[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def compare_groups(df: pd.DataFrame, group_a: str, group_b: str) -> pd.DataFrame:
    metrics = [
        c
        for c in df.columns
        if c not in {"subject_id", "group", "hemisphere", "n_parcels", "n_edges"}
    ]
    results: List[MetricResult] = []

    for hemi in sorted(df["hemisphere"].unique()):
        hemi_df = df[df["hemisphere"] == hemi]
        for metric in metrics:
            a = hemi_df.loc[hemi_df["group"] == group_a, metric].to_numpy(dtype=float)
            b = hemi_df.loc[hemi_df["group"] == group_b, metric].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if len(a) == 0 or len(b) == 0:
                continue
            t_stat, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            results.append(
                MetricResult(
                    hemisphere=hemi,
                    metric=metric,
                    group_a_mean=float(np.mean(a)),
                    group_b_mean=float(np.mean(b)),
                    group_a_std=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                    group_b_std=float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
                    delta_mean=float(np.mean(a) - np.mean(b)),
                    t_stat=float(t_stat),
                    p_value=float(p_value),
                    hedges_g=hedges_g(a, b),
                    n_group_a=int(len(a)),
                    n_group_b=int(len(b)),
                )
            )

    out = pd.DataFrame([r.__dict__ for r in results])
    if not out.empty:
        out["q_value_bh"] = bh_fdr(out["p_value"].to_numpy())
    return out


def maybe_compare_with_vortex(
    fc_stats: pd.DataFrame,
    vortex_csv: Optional[Path],
    logger: logging.Logger,
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
        hemi_df = vortex[vortex["hemisphere"] == hemi]
        groups = list(hemi_df["group"].unique())
        if len(groups) != 2:
            logger.warning(
                "Skipping hemisphere %s in vortex CSV because it does not contain exactly 2 groups.", hemi
            )
            continue
        g1, g2 = groups[0], groups[1]
        for metric in value_cols:
            a = hemi_df.loc[hemi_df["group"] == g1, metric].to_numpy(dtype=float)
            b = hemi_df.loc[hemi_df["group"] == g2, metric].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if len(a) == 0 or len(b) == 0:
                continue
            t_stat, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            rows.append(
                {
                    "method": "vortex",
                    "hemisphere": hemi,
                    "metric": metric,
                    "group_a": g1,
                    "group_b": g2,
                    "p_value": float(p_value),
                    "t_stat": float(t_stat),
                    "hedges_g": hedges_g(a, b),
                    "n_group_a": int(len(a)),
                    "n_group_b": int(len(b)),
                }
            )

    if fc_stats.empty and not rows:
        return None

    fc_rows = fc_stats[["hemisphere", "metric", "p_value", "t_stat", "hedges_g", "n_group_a", "n_group_b"]].copy()
    fc_rows.insert(0, "method", "fc")
    fc_rows.insert(3, "group_a", pd.NA)
    fc_rows.insert(4, "group_b", pd.NA)

    vortex_rows = pd.DataFrame(rows)
    combined = pd.concat([fc_rows, vortex_rows], ignore_index=True, sort=False)
    if not combined.empty:
        combined["q_value_bh"] = bh_fdr(combined["p_value"].to_numpy())
    return combined


def save_parcel_names(parcel_names: Dict[str, List[str]], out_dir: Path, prefix: str) -> None:
    ensure_dir(out_dir)
    for hemi, names in parcel_names.items():
        pd.DataFrame({"parcel_name": names}).to_csv(
            out_dir / f"{prefix}_{hemi}_parcel_names.csv", index=False
        )


def make_summary_plots(df: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path, group_a: str, group_b: str) -> None:
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
                )
        save_metric_bar_summary(stats_df, plot_dir / f"{hemi}_top_metrics_effects.png", hemi)


def make_matrix_visuals(
    group_a_label: str,
    group_b_label: str,
    group_a_mats: Dict[str, List[np.ndarray]],
    group_b_mats: Dict[str, List[np.ndarray]],
    group_a_subjects: Dict[str, List[str]],
    group_b_subjects: Dict[str, List[str]],
    out_dir: Path,
    save_subject_heatmaps: bool,
    max_subject_heatmaps: int,
) -> None:
    matrix_dir = out_dir / "plots" / "matrix_maps"
    ensure_dir(matrix_dir)

    for hemi in ["left", "right"]:
        if not group_a_mats[hemi] or not group_b_mats[hemi]:
            continue

        mean_a = np.nanmean(np.stack(group_a_mats[hemi], axis=0), axis=0)
        mean_b = np.nanmean(np.stack(group_b_mats[hemi], axis=0), axis=0)
        diff = mean_a - mean_b
        t_map, p_map = compute_edgewise_stats(group_a_mats[hemi], group_b_mats[hemi])
        q_map = bh_fdr_matrix(p_map)
        sig_mask = (q_map < 0.05).astype(float)

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
        save_matrix_plot(t_map, matrix_dir / f"{hemi}_edgewise_t_map.png", f"Edgewise Welch t map ({hemi})", vmin=-t_abs, vmax=t_abs)
        save_matrix_plot(-np.log10(np.clip(p_map, 1e-300, None)), matrix_dir / f"{hemi}_edgewise_neglog10_p.png", f"Edgewise -log10(p) ({hemi})")
        save_matrix_plot(sig_mask, matrix_dir / f"{hemi}_edgewise_q_lt_0p05_mask.png", f"Edgewise q<0.05 mask ({hemi})", vmin=0.0, vmax=1.0)

        pd.DataFrame(mean_a).to_csv(matrix_dir / f"{hemi}_{group_a_label}_mean_matrix.csv", index=False)
        pd.DataFrame(mean_b).to_csv(matrix_dir / f"{hemi}_{group_b_label}_mean_matrix.csv", index=False)
        pd.DataFrame(diff).to_csv(matrix_dir / f"{hemi}_{group_a_label}_minus_{group_b_label}_mean_diff.csv", index=False)
        pd.DataFrame(t_map).to_csv(matrix_dir / f"{hemi}_edgewise_t_map.csv", index=False)
        pd.DataFrame(p_map).to_csv(matrix_dir / f"{hemi}_edgewise_p_map.csv", index=False)
        pd.DataFrame(q_map).to_csv(matrix_dir / f"{hemi}_edgewise_q_map.csv", index=False)

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

    group_a = GroupSpec(label=args.group_a_label, pconn_dir=args.group_a_dir)
    group_b = GroupSpec(label=args.group_b_label, pconn_dir=args.group_b_dir)

    logger.info("Loading group A from %s", group_a.pconn_dir)
    df_a, mats_a, subjects_a, parcels_a = load_subject_metrics_and_matrices(
        group_a, args.left_regex, args.right_regex, logger
    )
    logger.info("Loading group B from %s", group_b.pconn_dir)
    df_b, mats_b, subjects_b, parcels_b = load_subject_metrics_and_matrices(
        group_b, args.left_regex, args.right_regex, logger
    )

    for hemi in ["left", "right"]:
        if hemi in parcels_a and hemi in parcels_b and parcels_a[hemi] != parcels_b[hemi]:
            raise ValueError(f"Parcel ordering mismatch between groups for hemisphere '{hemi}'.")

    parcel_dir = args.out_dir / "parcel_info"
    save_parcel_names(parcels_a, parcel_dir, f"{args.group_a_label}")
    save_parcel_names(parcels_b, parcel_dir, f"{args.group_b_label}")

    df = pd.concat([df_a, df_b], ignore_index=True)
    subject_csv = args.out_dir / "subject_level_fc_metrics.csv"
    df.to_csv(subject_csv, index=False)
    logger.info("Wrote subject-level metrics to %s", subject_csv)

    stats_df = compare_groups(df, args.group_a_label, args.group_b_label)
    stats_csv = args.out_dir / "group_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Wrote group statistics to %s", stats_csv)

    method_df = maybe_compare_with_vortex(stats_df, args.vortex_csv, logger)
    if method_df is not None:
        method_csv = args.out_dir / "method_comparison.csv"
        method_df.to_csv(method_csv, index=False)
        logger.info("Wrote method comparison table to %s", method_csv)

    make_summary_plots(df, stats_df, args.out_dir, args.group_a_label, args.group_b_label)
    make_matrix_visuals(
        args.group_a_label,
        args.group_b_label,
        mats_a,
        mats_b,
        subjects_a,
        subjects_b,
        args.out_dir,
        args.save_subject_heatmaps,
        args.max_subject_heatmaps,
    )
    logger.info("Saved plots under %s", args.out_dir / "plots")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
