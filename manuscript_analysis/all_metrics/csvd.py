from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .config import Config
from .utils import (
    build_group_summary_df,
    paired_t,
    plot_paired_violin,
    resolve_bundle_dir,
    save_fig,
    unpaired_t,
    write_table,
)

TOP_K = 10
DEMEAN = False
MAX_VOXELS = None
RANDOM_SEED = 7
DEFAULT_CSVD_METHOD = "phase_gradient"
SVD_MODES = ("complex_svd", "real_svd")

# Optical-flow parameters for the Charbonnier-penalized Horn-Schunck variant.
FLOW_ALPHA = 0.1
FLOW_BETA = 10.0
FLOW_MAX_ITER = 200
FLOW_TOL = 1.0e-4


def _collect_entries(cfg: Config) -> pd.DataFrame:
    path = cfg.combined_dir / "combined_frame_index.parquet"
    if not path.exists():
        return pd.DataFrame()

    cols = ["group", "subid", "hemisphere", "bundle_dir"]
    df = pd.read_parquet(path, columns=cols).drop_duplicates()

    rows: List[Dict[str, object]] = []
    for row in df.itertuples(index=False):
        row_series = pd.Series(
            {
                "group": row.group,
                "subid": row.subid,
                "hemisphere": row.hemisphere,
                "bundle_dir": row.bundle_dir,
            }
        )
        bundle_dir = resolve_bundle_dir(row_series, cfg)
        if bundle_dir is None:
            continue
        phase_path = bundle_dir / "phase_cube.npy"
        if not phase_path.exists():
            continue
        rows.append(
            {
                "group": row.group,
                "subid": str(row.subid),
                "hemisphere": row.hemisphere,
                "bundle_dir": bundle_dir,
                "phase_path": phase_path,
            }
        )
    return pd.DataFrame(rows)


def _phase_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _phase_derivative_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    deriv = np.full_like(arr, np.nan, dtype=np.float64)
    n = arr.shape[axis]
    if n < 2:
        return deriv

    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim

    dst[axis] = 0
    src_a = src.copy()
    src_b = src.copy()
    src_a[axis] = 1
    src_b[axis] = 0
    deriv[tuple(dst)] = _phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

    dst[axis] = n - 1
    src_a[axis] = n - 1
    src_b[axis] = n - 2
    deriv[tuple(dst)] = _phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

    if n >= 3:
        dst[axis] = 1
        src_a[axis] = 2
        src_b[axis] = 0
        deriv[tuple(dst)] = 0.5 * _phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

        dst[axis] = n - 2
        src_a[axis] = n - 1
        src_b[axis] = n - 3
        deriv[tuple(dst)] = 0.5 * _phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

    if n >= 5:
        dst[axis] = slice(2, n - 2)
        src_p1 = src.copy()
        src_m1 = src.copy()
        src_p2 = src.copy()
        src_m2 = src.copy()
        src_p1[axis] = slice(3, n - 1)
        src_m1[axis] = slice(1, n - 3)
        src_p2[axis] = slice(4, n)
        src_m2[axis] = slice(0, n - 4)
        term1 = _phase_diff(arr[tuple(src_p1)], arr[tuple(src_m1)])
        term2 = _phase_diff(arr[tuple(src_p2)], arr[tuple(src_m2)])
        deriv[tuple(dst)] = (8.0 * term1 - term2) / 12.0

    finite = np.isfinite(arr)
    if n == 2:
        valid = finite & np.roll(finite, -1, axis=axis)
    else:
        valid = finite.copy()
        valid &= np.roll(finite, -1, axis=axis) | np.roll(finite, 1, axis=axis)
    return np.where(valid, deriv, np.nan)


def _neighbor_average(field: np.ndarray) -> np.ndarray:
    avg = np.zeros_like(field, dtype=np.float64)
    count = np.zeros_like(field, dtype=np.float64)

    avg[1:, :] += field[:-1, :]
    count[1:, :] += 1.0
    avg[:-1, :] += field[1:, :]
    count[:-1, :] += 1.0
    avg[:, 1:] += field[:, :-1]
    count[:, 1:] += 1.0
    avg[:, :-1] += field[:, 1:]
    count[:, :-1] += 1.0

    return np.divide(avg, count, out=np.zeros_like(avg), where=count > 0)


def _compute_pairwise_velocity(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    u0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ex = 0.5 * (_phase_derivative_axis(phase_a, axis=1) + _phase_derivative_axis(phase_b, axis=1))
    ey = 0.5 * (_phase_derivative_axis(phase_a, axis=0) + _phase_derivative_axis(phase_b, axis=0))
    et = _phase_diff(phase_a, phase_b)

    valid = np.isfinite(phase_a) & np.isfinite(phase_b) & np.isfinite(ex) & np.isfinite(ey) & np.isfinite(et)
    if not np.any(valid):
        nan_frame = np.full_like(phase_a, np.nan, dtype=np.float64)
        return nan_frame, nan_frame.copy()

    u = np.zeros_like(phase_a, dtype=np.float64) if u0 is None else np.where(np.isfinite(u0), u0, 0.0)
    v = np.zeros_like(phase_a, dtype=np.float64) if v0 is None else np.where(np.isfinite(v0), v0, 0.0)

    alpha2 = FLOW_ALPHA * FLOW_ALPHA
    beta2 = FLOW_BETA * FLOW_BETA

    for _ in range(FLOW_MAX_ITER):
        u_avg = _neighbor_average(u)
        v_avg = _neighbor_average(v)

        residual = ex * u_avg + ey * v_avg + et
        weight = 1.0 / np.sqrt(residual * residual + beta2)
        denom = alpha2 + weight * (ex * ex + ey * ey)
        denom = np.where(denom > 0.0, denom, np.nan)
        delta = weight * residual / denom

        u_new = np.where(valid, u_avg - ex * delta, 0.0)
        v_new = np.where(valid, v_avg - ey * delta, 0.0)

        step = np.nanmax(np.abs(u_new - u) + np.abs(v_new - v))
        u, v = u_new, v_new
        if np.isfinite(step) and step < FLOW_TOL:
            break

    return np.where(valid, u, np.nan), np.where(valid, v, np.nan)


def _compute_complex_velocity_cube(phase_cube: np.ndarray) -> np.ndarray:
    if phase_cube.ndim != 3:
        raise ValueError(f"phase_cube must be 3D, got shape={phase_cube.shape}")
    if phase_cube.shape[2] < 2:
        raise ValueError(f"phase_cube needs at least 2 frames, got shape={phase_cube.shape}")

    rows, cols, frames = phase_cube.shape
    u_cube = np.full((rows, cols, frames - 1), np.nan, dtype=np.float64)
    v_cube = np.full((rows, cols, frames - 1), np.nan, dtype=np.float64)

    u_prev: np.ndarray | None = None
    v_prev: np.ndarray | None = None
    for t in range(frames - 1):
        u_frame, v_frame = _compute_pairwise_velocity(
            phase_cube[:, :, t],
            phase_cube[:, :, t + 1],
            u0=u_prev,
            v0=v_prev,
        )
        u_cube[:, :, t] = u_frame
        v_cube[:, :, t] = v_frame
        u_prev, v_prev = u_frame, v_frame

    return u_cube + 1j * v_cube


def _compute_phase_gradient_cube(phase_cube: np.ndarray) -> np.ndarray:
    if phase_cube.ndim != 3:
        raise ValueError(f"phase_cube must be 3D, got shape={phase_cube.shape}")
    gx = _phase_derivative_axis(phase_cube, axis=1)
    gy = _phase_derivative_axis(phase_cube, axis=0)
    return gx + 1j * gy


def _empty_metrics() -> Dict[str, float]:
    return {
        "top1_energy": math.nan,
        "top3_energy": math.nan,
        "top5_energy": math.nan,
        "energy_entropy": math.nan,
        "participation_ratio": math.nan,
        "k90": math.nan,
        "field_frames": math.nan,
        "n_valid_voxels": math.nan,
        "n_features": math.nan,
    }


def _prepare_svd_matrix(complex_field_cube: np.ndarray, svd_mode: str) -> tuple[np.ndarray | None, float, float]:
    rows, cols, timepoints = complex_field_cube.shape
    flat = complex_field_cube.reshape(rows * cols, timepoints)
    mask_flat = np.isfinite(flat).all(axis=1)
    if not mask_flat.any():
        return None, math.nan, math.nan

    data_complex = flat[mask_flat].T
    if DEMEAN:
        data_complex = data_complex - np.mean(data_complex, axis=0, keepdims=True)

    if MAX_VOXELS is not None and data_complex.shape[1] > MAX_VOXELS:
        rng = np.random.default_rng(RANDOM_SEED)
        keep = rng.choice(data_complex.shape[1], size=MAX_VOXELS, replace=False)
        data_complex = data_complex[:, keep]
        n_valid_voxels = float(keep.size)
    else:
        n_valid_voxels = float(data_complex.shape[1])

    if svd_mode == "complex_svd":
        data = data_complex
    elif svd_mode == "real_svd":
        data = np.concatenate([np.real(data_complex), np.imag(data_complex)], axis=1)
    else:
        raise ValueError(f"Unsupported svd mode: {svd_mode}")

    return data, n_valid_voxels, float(data.shape[1])


def _compute_svd_metrics(complex_field_cube: np.ndarray, svd_mode: str) -> tuple[Dict[str, float], np.ndarray | None]:
    try:
        data, n_valid_voxels, n_features = _prepare_svd_matrix(complex_field_cube, svd_mode)
    except Exception:
        return _empty_metrics(), None

    if data is None:
        return _empty_metrics(), None

    if data.shape[0] < 2 or data.shape[1] < 1:
        metrics = _empty_metrics()
        metrics["field_frames"] = float(data.shape[0])
        metrics["n_valid_voxels"] = n_valid_voxels
        metrics["n_features"] = n_features
        return metrics, None

    try:
        _, s, _ = np.linalg.svd(data, full_matrices=False)
    except np.linalg.LinAlgError:
        metrics = _empty_metrics()
        metrics["field_frames"] = float(data.shape[0])
        metrics["n_valid_voxels"] = n_valid_voxels
        metrics["n_features"] = n_features
        return metrics, None

    energy = s * s
    total = float(np.sum(energy))
    if total <= 0.0:
        metrics = _empty_metrics()
        metrics["field_frames"] = float(data.shape[0])
        metrics["n_valid_voxels"] = n_valid_voxels
        metrics["n_features"] = n_features
        return metrics, None

    frac = energy / total
    eps = 1.0e-12
    metrics = {
        "top1_energy": float(frac[0]) if frac.size >= 1 else math.nan,
        "top3_energy": float(np.sum(frac[: min(3, frac.size)])) if frac.size >= 1 else math.nan,
        "top5_energy": float(np.sum(frac[: min(5, frac.size)])) if frac.size >= 1 else math.nan,
        "energy_entropy": float(-np.sum(frac * np.log(frac + eps))),
        "participation_ratio": float((np.sum(frac) ** 2) / np.sum(frac ** 2 + eps)),
        "k90": float(int(np.searchsorted(np.cumsum(frac), 0.9) + 1)) if frac.size else math.nan,
        "field_frames": float(data.shape[0]),
        "n_valid_voxels": n_valid_voxels,
        "n_features": n_features,
    }

    energy_curve = frac[:TOP_K]
    if energy_curve.size < TOP_K:
        energy_curve = np.pad(energy_curve, (0, TOP_K - energy_curve.size), constant_values=np.nan)
    return metrics, energy_curve


def _compute_csvd_metrics(phase_cube_path: Path, method: str) -> tuple[str, np.ndarray, Dict[str, tuple[Dict[str, float], np.ndarray | None]]]:
    phase_cube = np.load(phase_cube_path)
    if phase_cube.ndim != 3:
        raise ValueError(f"phase_cube must be 3D, got shape={phase_cube.shape}")

    phase_cube = np.asarray(phase_cube, dtype=np.float64)
    if method == "phase_gradient":
        complex_field_cube = _compute_phase_gradient_cube(phase_cube)
    elif method == "optical_flow":
        complex_field_cube = _compute_complex_velocity_cube(phase_cube)
    else:
        raise ValueError(f"Unsupported csvd method: {method}")

    out: Dict[str, tuple[Dict[str, float], np.ndarray | None]] = {}
    for svd_mode in SVD_MODES:
        out[svd_mode] = _compute_svd_metrics(complex_field_cube, svd_mode)
    return method, complex_field_cube, out


def run_csvd(cfg: Config, summary: List[Dict]) -> None:
    entries = _collect_entries(cfg)
    if entries.empty:
        logging.warning("csvd: no phase_cube entries found in %s", cfg.combined_dir)
        return

    method = str(getattr(cfg, "csvd_method", DEFAULT_CSVD_METHOD) or DEFAULT_CSVD_METHOD).strip().lower()
    target_groups = {cfg.group_pcb, cfg.group_drug}
    present = set(entries["group"].unique())
    active_groups = present if len(present & target_groups) < 2 else target_groups
    df = entries[entries["group"].isin(active_groups)]
    if df.empty:
        return

    records: List[Dict] = []
    energy_curves: List[Dict] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="csvd cubes", file=sys.stdout, dynamic_ncols=True):
        try:
            field_method, _, result_by_mode = _compute_csvd_metrics(Path(row.phase_path), method=method)
        except Exception as exc:
            logging.warning("csvd failed for %s: %s", row.phase_path, exc)
            continue

        for svd_mode, (metrics, curve) in result_by_mode.items():
            metrics.update(
                {
                    "group": row.group,
                    "subid": str(row.subid),
                    "hemisphere": row.hemisphere,
                    "csvd_method": field_method,
                    "svd_mode": svd_mode,
                }
            )
            records.append(metrics)
            if curve is not None:
                energy_curves.append(
                    {
                        "group": row.group,
                        "hemisphere": row.hemisphere,
                        "curve": curve,
                        "csvd_method": field_method,
                        "svd_mode": svd_mode,
                    }
                )

    if not records:
        return

    records_df = pd.DataFrame(records)
    numeric_cols = records_df.select_dtypes(include=[np.number]).columns.tolist()
    subj_df = records_df.groupby(
        ["group", "subid", "hemisphere", "csvd_method", "svd_mode"],
        as_index=False,
    )[numeric_cols].mean()
    if subj_df.empty:
        return

    out_dir = cfg.output_root / cfg.results_prefix / "csvd"
    write_table(subj_df, out_dir / "per_subject.csv")

    metrics = [
        "top1_energy",
        "top3_energy",
        "top5_energy",
        "energy_entropy",
        "participation_ratio",
        "k90",
    ]
    group_summary_parts: List[pd.DataFrame] = []

    for svd_mode in SVD_MODES:
        mode_df = subj_df[subj_df["svd_mode"] == svd_mode].copy()
        if mode_df.empty:
            continue

        section_name = "csvd_complex" if svd_mode == "complex_svd" else "csvd_real"

        for hemi in ["left", "right"]:
            pivot = mode_df[mode_df["hemisphere"] == hemi].pivot(index="subid", columns="group", values=metrics)
            if pivot.empty:
                continue
            pivot.columns = ["_".join(col) for col in pivot.columns]
            for metric in metrics:
                drug = pivot.get(f"{metric}_{cfg.group_drug}", pd.Series(dtype=float))
                pcb = pivot.get(f"{metric}_{cfg.group_pcb}", pd.Series(dtype=float))
                summary.append(
                    {
                        "section": section_name,
                        "metric": metric,
                        "hemisphere": hemi,
                        "comparison": "paired_drug_vs_pcb",
                        "csvd_method": method,
                        **paired_t(drug, pcb),
                    }
                )

        for hemi in ["left", "right"]:
            hemi_df = mode_df[mode_df["hemisphere"] == hemi]
            for metric in metrics:
                drug = hemi_df[hemi_df["group"] == cfg.group_drug][metric]
                pcb = hemi_df[hemi_df["group"] == cfg.group_pcb][metric]
                summary.append(
                    {
                        "section": section_name,
                        "metric": metric,
                        "hemisphere": hemi,
                        "comparison": "unpaired_drug_vs_pcb",
                        "csvd_method": method,
                        **unpaired_t(drug, pcb),
                    }
                )

        group_summary_df = build_group_summary_df(mode_df, metrics, cfg)
        if not group_summary_df.empty:
            group_summary_df["csvd_method"] = method
            group_summary_df["svd_mode"] = svd_mode
            group_summary_parts.append(group_summary_df)
        write_table(group_summary_df, out_dir / f"group_summary_{svd_mode}.csv")

        sns.set_theme(style="whitegrid", context="talk")
        field_label = "phase-gradient fields" if method == "phase_gradient" else "optical-flow fields"
        svd_label = "Complex SVD" if svd_mode == "complex_svd" else "Real SVD"

        for hemi in ["left", "right"]:
            hemi_df = mode_df[mode_df["hemisphere"] == hemi]
            if hemi_df.empty:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(12, 9))
            for ax, metric in zip(axes.flat, metrics):
                sns.violinplot(
                    data=hemi_df,
                    x="group",
                    y=metric,
                    order=[cfg.group_pcb, cfg.group_drug],
                    cut=0,
                    ax=ax,
                )
                ax.set_title(f"{metric} ({hemi})")
            fig.suptitle(f"{svd_label} metrics from {field_label} ({hemi})")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            save_fig(fig, out_dir / f"violin_{svd_mode}_{hemi}.png", cfg.save_plots)

            for metric in metrics:
                tidy = hemi_df[["subid", "group", metric]]
                plot_paired_violin(
                    tidy,
                    metric,
                    hemi,
                    f"{svd_label} {metric} paired",
                    out_dir / f"paired_{svd_mode}_{metric}_{hemi}.png",
                    cfg,
                )

    if energy_curves:
        curves_df = pd.DataFrame(energy_curves)
        for svd_mode in SVD_MODES:
            mode_curves = curves_df[curves_df["svd_mode"] == svd_mode]
            if mode_curves.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for (grp, hemi), grp_rows in mode_curves.groupby(["group", "hemisphere"]):
                curves = np.vstack(grp_rows["curve"].to_list())
                mean_curve = np.nanmean(curves, axis=0)
                ax.plot(np.arange(1, len(mean_curve) + 1), mean_curve, label=f"{grp} ({hemi})")
            ax.set_xlabel("Mode index")
            ax.set_ylabel("Energy fraction")
            svd_label = "Complex SVD" if svd_mode == "complex_svd" else "Real SVD"
            field_label = "phase-gradient fields" if method == "phase_gradient" else "optical-flow fields"
            ax.set_title(f"Mean energy spectra: {svd_label} on {field_label}")
            ax.legend()
            fig.tight_layout()
            save_fig(fig, out_dir / f"energy_spectra_{svd_mode}.png", cfg.save_plots)

    if group_summary_parts:
        write_table(pd.concat(group_summary_parts, ignore_index=True), out_dir / "group_summary.csv")
