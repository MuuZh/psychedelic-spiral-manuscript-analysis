#!/usr/bin/env python
"""
Model-vs-empirical cSVD and CAI drug-effect agreement.

For each complete subject/hemisphere, this script reads empirical and
reconstructed phase cubes, computes:

1. cSVD metrics on the phase-gradient vector-field cube.
2. CAI as weighted_mean_cos2_alignment, matching phase_gradient_alignment.py:
   sum(ref_mag * phase_mag * cos(2 theta)) / sum(ref_mag * phase_mag).

It then compares empirical and model values for drug, placebo, and
Drug-Placebo delta. The primary manuscript-facing analysis is the
empirical-delta vs model-delta agreement table.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CONDITION_RE = re.compile(r"^(?P<condition>[A-Za-z]+_[A-Za-z]+)_S(?P<subid>\d+)")
HEMISPHERES = ("left", "right")
SVD_MODES = ("complex_svd", "real_svd")
CSVD_METRICS = (
    "top1_energy",
    "top3_energy",
    "top5_energy",
    "energy_entropy",
    "participation_ratio",
    "k90",
)
CAI_METRIC = "cai_weighted_mean_cos2_alignment"


@dataclass(frozen=True)
class DatasetSpec:
    drug_label: str
    drug_condition: str
    placebo_condition: str
    empirical_root: Path
    model_root: Path
    ref_left: Path
    ref_right: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cSVD/CAI model-vs-empirical Drug-PCB delta correlations."
    )
    parser.add_argument("--dmt-empirical-root", default=Path("analysis_outputs/phase_fc_batch_phase_corr_7networks"), type=Path)
    parser.add_argument("--dmt-model-root", default=Path("analysis_outputs/phase_fc_recon_7networks"), type=Path)
    parser.add_argument("--lsd-empirical-root", default=Path("analysis_outputs/phase_fc_batch_phase_corr_7networks_LSD"), type=Path)
    parser.add_argument("--lsd-model-root", default=Path("analysis_outputs/phase_fc_recon_7networks_LSD"), type=Path)
    parser.add_argument("--ref-root", default=Path("analysis_outputs/group_emb_to_grid/emb_001"), type=Path)
    parser.add_argument("--dmt-ref-left", type=Path, default=None)
    parser.add_argument("--dmt-ref-right", type=Path, default=None)
    parser.add_argument("--lsd-ref-left", type=Path, default=None)
    parser.add_argument("--lsd-ref-right", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/added_analysis/SVD-CAI-model-delta-corr"),
        type=Path,
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--edge-margin", type=int, default=2)
    parser.add_argument("--min-n", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional paired-subject limit per dataset/hemi for smoke tests.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def normalize_subid(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("S"):
        text = text[1:]
    return f"{int(text):02d}" if text.isdigit() else text


def resolve_existing_path(path_like: str | Path, base: Path | None = None) -> Path | None:
    path = Path(path_like)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, Path.cwd() / path])
        if base is not None:
            candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_hemisphere(out_dir: Path, metadata: dict[str, object]) -> str | None:
    hemi = str(metadata.get("hemisphere", "")).lower()
    if hemi in {"left", "right"}:
        return hemi
    meta_path = out_dir / "parcel_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, usecols=["hemi"])
        value = str(meta["hemi"].iloc[0])
        if value == "LH":
            return "left"
        if value == "RH":
            return "right"
    suffix = out_dir.name[-1:].upper()
    if suffix == "L":
        return "left"
    if suffix == "R":
        return "right"
    return None


def parse_entry(out_dir: Path, source: str) -> dict[str, object] | None:
    match = CONDITION_RE.search(out_dir.name)
    if match is None:
        return None
    metadata_path = out_dir / "run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    phase_path = resolve_existing_path(str(metadata.get("phase_cube", "")), base=metadata_path.parent)
    hemisphere = infer_hemisphere(out_dir, metadata)
    if phase_path is None or hemisphere is None:
        return None
    return {
        "source": source,
        "condition": match.group("condition"),
        "subid": normalize_subid(match.group("subid")),
        "hemisphere": hemisphere,
        "out_dir": out_dir,
        "phase_cube": phase_path,
        "run_metadata": metadata_path,
    }


def discover_entries(root: Path, source: str, hemispheres: Iterable[str]) -> pd.DataFrame:
    rows = []
    if not root.exists():
        raise FileNotFoundError(f"{source} root does not exist: {root}")
    for out_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "atlas_metadata"):
        entry = parse_entry(out_dir, source)
        if entry is not None and entry["hemisphere"] in hemispheres:
            rows.append(entry)
    if not rows:
        raise RuntimeError(f"No usable phase-cube entries found under {root}")
    return pd.DataFrame(rows)


def phase_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def phase_derivative_axis(arr: np.ndarray, axis: int) -> np.ndarray:
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
    deriv[tuple(dst)] = phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

    dst[axis] = n - 1
    src_a[axis] = n - 1
    src_b[axis] = n - 2
    deriv[tuple(dst)] = phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

    if n >= 3:
        dst[axis] = 1
        src_a[axis] = 2
        src_b[axis] = 0
        deriv[tuple(dst)] = 0.5 * phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

        dst[axis] = n - 2
        src_a[axis] = n - 1
        src_b[axis] = n - 3
        deriv[tuple(dst)] = 0.5 * phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

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
        term1 = phase_diff(arr[tuple(src_p1)], arr[tuple(src_m1)])
        term2 = phase_diff(arr[tuple(src_p2)], arr[tuple(src_m2)])
        deriv[tuple(dst)] = (8.0 * term1 - term2) / 12.0

    finite = np.isfinite(arr)
    if n == 2:
        valid = finite & np.roll(finite, -1, axis=axis)
    else:
        valid = finite.copy()
        valid &= np.roll(finite, -1, axis=axis) | np.roll(finite, 1, axis=axis)
    return np.where(valid, deriv, np.nan)


def phase_gradient_for_cai(phase: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    grad_x = np.full_like(phase, np.nan, dtype=np.float64)
    grad_y = np.full_like(phase, np.nan, dtype=np.float64)
    if phase.shape[1] >= 3:
        grad_x[:, 1:-1, :] = phase_diff(phase[:, 2:, :], phase[:, :-2, :]) / (2.0 * spacing)
    if phase.shape[0] >= 3:
        grad_y[1:-1, :, :] = phase_diff(phase[2:, :, :], phase[:-2, :, :]) / (2.0 * spacing)
    return grad_x, grad_y


def apply_border_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    out = mask.copy()
    if margin <= 0:
        return out
    out[:margin, :] = False
    out[-margin:, :] = False
    out[:, :margin] = False
    out[:, -margin:] = False
    return out


def align_2d(arr: np.ndarray, target_shape: tuple[int, int], name: str) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    if arr.shape[0] == target_shape[0] + 1 and arr.shape[1] == target_shape[1]:
        return arr[:-1, :]
    if arr.shape[0] + 1 == target_shape[0] and arr.shape[1] == target_shape[1]:
        out = np.full(target_shape, np.nan, dtype=float)
        out[:-1, :] = arr
        return out
    raise ValueError(f"Cannot align {name} shape {arr.shape} to phase shape {target_shape}")


def load_reference(path: Path, spacing: float, edge_margin: int) -> dict[str, np.ndarray]:
    gmap = np.asarray(np.load(path), dtype=np.float64)
    if gmap.ndim != 2:
        raise ValueError(f"Reference map must be 2D, got {gmap.shape}: {path}")
    gy, gx = np.gradient(gmap, spacing)
    mag = np.hypot(gx, gy)
    valid = np.isfinite(gmap) & np.isfinite(gx) & np.isfinite(gy) & (mag > 0)
    return {
        "path": str(path),
        "gx": gx,
        "gy": gy,
        "mag": mag,
        "valid": apply_border_mask(valid, edge_margin),
    }


def compute_csvd_metrics(phase_cube: np.ndarray) -> dict[str, float]:
    gx = phase_derivative_axis(phase_cube, axis=1)
    gy = phase_derivative_axis(phase_cube, axis=0)
    field = gx + 1j * gy
    rows, cols, frames = field.shape
    flat = field.reshape(rows * cols, frames)
    mask = np.isfinite(flat).all(axis=1)
    out: dict[str, float] = {}
    if not mask.any():
        for mode in SVD_MODES:
            for metric in CSVD_METRICS:
                out[f"csvd_{mode}_{metric}"] = math.nan
        return out

    data_complex = flat[mask].T
    for mode in SVD_MODES:
        if mode == "complex_svd":
            data = data_complex
        else:
            data = np.concatenate([np.real(data_complex), np.imag(data_complex)], axis=1)
        prefix = f"csvd_{mode}"
        out[f"{prefix}_field_frames"] = float(data.shape[0])
        out[f"{prefix}_n_valid_voxels"] = float(data_complex.shape[1])
        out[f"{prefix}_n_features"] = float(data.shape[1])
        if data.shape[0] < 2 or data.shape[1] < 1:
            for metric in CSVD_METRICS:
                out[f"{prefix}_{metric}"] = math.nan
            continue
        try:
            _, svals, _ = np.linalg.svd(data, full_matrices=False)
        except np.linalg.LinAlgError:
            for metric in CSVD_METRICS:
                out[f"{prefix}_{metric}"] = math.nan
            continue
        energy = svals * svals
        total = float(np.sum(energy))
        if total <= 0:
            for metric in CSVD_METRICS:
                out[f"{prefix}_{metric}"] = math.nan
            continue
        frac = energy / total
        eps = 1.0e-12
        out[f"{prefix}_top1_energy"] = float(frac[0]) if frac.size else math.nan
        out[f"{prefix}_top3_energy"] = float(np.sum(frac[: min(3, frac.size)])) if frac.size else math.nan
        out[f"{prefix}_top5_energy"] = float(np.sum(frac[: min(5, frac.size)])) if frac.size else math.nan
        out[f"{prefix}_energy_entropy"] = float(-np.sum(frac * np.log(frac + eps)))
        out[f"{prefix}_participation_ratio"] = float((np.sum(frac) ** 2) / np.sum(frac**2 + eps))
        out[f"{prefix}_k90"] = float(int(np.searchsorted(np.cumsum(frac), 0.9) + 1)) if frac.size else math.nan
    return out


def compute_weighted_cai(phase_cube: np.ndarray, ref: dict[str, np.ndarray], spacing: float) -> dict[str, float]:
    h, w, frames = phase_cube.shape
    ref_gx = align_2d(ref["gx"], (h, w), "ref_gx")
    ref_gy = align_2d(ref["gy"], (h, w), "ref_gy")
    ref_mag = align_2d(ref["mag"], (h, w), "ref_mag")
    ref_valid = align_2d(ref["valid"], (h, w), "ref_valid").astype(bool)

    grad_x, grad_y = phase_gradient_for_cai(phase_cube, spacing=spacing)
    phase_mag = np.hypot(grad_x, grad_y)
    phase_ux = np.divide(-grad_x, phase_mag, out=np.full_like(grad_x, np.nan), where=phase_mag > 0)
    phase_uy = np.divide(-grad_y, phase_mag, out=np.full_like(grad_y, np.nan), where=phase_mag > 0)

    weighted_sum = 0.0
    weight_sum = 0.0
    n_samples = 0
    frame_values = []
    for frame in range(frames):
        valid = (
            ref_valid
            & np.isfinite(phase_cube[:, :, frame])
            & np.isfinite(phase_ux[:, :, frame])
            & np.isfinite(phase_uy[:, :, frame])
            & np.isfinite(phase_mag[:, :, frame])
            & (phase_mag[:, :, frame] > 0)
        )
        if not np.any(valid):
            continue
        dot = ref_gx * phase_ux[:, :, frame] + ref_gy * phase_uy[:, :, frame]
        cos_theta = np.divide(
            dot,
            ref_mag,
            out=np.full((h, w), np.nan, dtype=np.float64),
            where=valid & (ref_mag > 0),
        )
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        cos2 = 2.0 * cos_theta * cos_theta - 1.0
        weight = ref_mag * phase_mag[:, :, frame]
        good = valid & np.isfinite(cos2) & np.isfinite(weight)
        if not np.any(good):
            continue
        wvals = weight[good]
        cvals = cos2[good]
        denom = float(np.sum(wvals))
        if denom <= 0:
            continue
        weighted_sum += float(np.sum(wvals * cvals))
        weight_sum += denom
        n_samples += int(cvals.size)
        frame_values.append(float(np.sum(wvals * cvals) / denom))

    return {
        CAI_METRIC: float(weighted_sum / weight_sum) if weight_sum > 0 else math.nan,
        "cai_n_samples": n_samples,
        "cai_n_frames": frames,
        "cai_frame_mean": float(np.nanmean(frame_values)) if frame_values else math.nan,
    }


def compute_phase_metrics(phase_path: Path, ref: dict[str, np.ndarray], spacing: float) -> dict[str, float]:
    phase_cube = np.asarray(np.load(phase_path, mmap_mode="r"), dtype=np.float64)
    if phase_cube.ndim != 3:
        raise ValueError(f"Expected 3D phase cube, got {phase_cube.shape}: {phase_path}")
    metrics = {
        "phase_height": int(phase_cube.shape[0]),
        "phase_width": int(phase_cube.shape[1]),
        "phase_frames": int(phase_cube.shape[2]),
    }
    metrics.update(compute_csvd_metrics(phase_cube))
    metrics.update(compute_weighted_cai(phase_cube, ref=ref, spacing=spacing))
    return metrics


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite].astype(float), y[finite].astype(float)


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return math.nan, math.nan
    sem = stats.sem(values)
    if not np.isfinite(sem):
        return math.nan, math.nan
    lo, hi = stats.t.interval(0.95, df=int(values.size) - 1, loc=float(np.mean(values)), scale=float(sem))
    return float(lo), float(hi)


def r_ci95(r_value: float, n: int) -> tuple[float, float]:
    if not np.isfinite(r_value) or n <= 3 or abs(r_value) >= 1:
        return math.nan, math.nan
    z = np.arctanh(r_value)
    se = 1.0 / math.sqrt(n - 3)
    crit = stats.norm.ppf(0.975)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def delta_test_stats(values: np.ndarray, min_n: int) -> dict[str, float]:
    values = values[np.isfinite(values)].astype(float)
    n = int(values.size)
    lo, hi = mean_ci95(values)
    out = {
        "n": n,
        "mean_delta": float(np.mean(values)) if n else math.nan,
        "sd_delta": float(np.std(values, ddof=1)) if n > 1 else math.nan,
        "sem_delta": float(stats.sem(values)) if n > 1 else math.nan,
        "ci95_delta_low": lo,
        "ci95_delta_high": hi,
        "median_delta": float(np.median(values)) if n else math.nan,
    }
    if n >= min_n and n > 1 and np.std(values, ddof=1) > 0:
        t_stat, p_val = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
        out.update(
            {
                "t_vs_zero": float(t_stat),
                "p_two_sided": float(p_val),
                "cohen_dz": float(np.mean(values) / np.std(values, ddof=1)),
            }
        )
    else:
        out.update({"t_vs_zero": math.nan, "p_two_sided": math.nan, "cohen_dz": math.nan})
    return out


def agreement_stats(empirical: np.ndarray, model: np.ndarray, min_n: int) -> dict[str, float]:
    x, y = finite_pair(empirical, model)
    n = int(x.size)
    err = y - x
    abs_err = np.abs(err)
    sq_err = err**2
    out = {
        "n": n,
        "empirical_mean": float(np.mean(x)) if n else math.nan,
        "model_mean": float(np.mean(y)) if n else math.nan,
        "bias_model_minus_empirical": float(np.mean(err)) if n else math.nan,
        "mean_abs_error": float(np.mean(abs_err)) if n else math.nan,
        "rmse": float(np.sqrt(np.mean(sq_err))) if n else math.nan,
        "capture_ratio_mean_model_over_empirical": (
            float(np.mean(y) / np.mean(x)) if n and np.isfinite(np.mean(x)) and abs(np.mean(x)) > 0 else math.nan
        ),
        "identity_r2_score": (
            1.0 - float(np.sum(sq_err)) / float(np.sum((x - np.mean(x)) ** 2))
            if n > 1 and np.sum((x - np.mean(x)) ** 2) > 0
            else math.nan
        ),
        "explained_variance_score": (
            1.0 - float(np.var(err, ddof=0)) / float(np.var(x, ddof=0))
            if n > 1 and np.var(x, ddof=0) > 0
            else math.nan
        ),
    }
    if n >= min_n and np.std(x, ddof=1) > 0 and np.std(y, ddof=1) > 0:
        r_val, p_val = stats.pearsonr(x, y)
        slope, intercept, reg_r, reg_p, slope_stderr = stats.linregress(x, y)
        lo, hi = r_ci95(float(r_val), n)
        out.update(
            {
                "pearson_r": float(r_val),
                "pearson_r2": float(r_val**2),
                "pearson_p": float(p_val),
                "pearson_r_ci95_low": lo,
                "pearson_r_ci95_high": hi,
                "slope": float(slope),
                "intercept": float(intercept),
                "regression_r2": float(reg_r**2),
                "regression_p": float(reg_p),
                "slope_stderr": float(slope_stderr),
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
                "slope": math.nan,
                "intercept": math.nan,
                "regression_r2": math.nan,
                "regression_p": math.nan,
                "slope_stderr": math.nan,
            }
        )
    return out


def metric_columns(df: pd.DataFrame) -> list[str]:
    csvd_allowed = {
        f"csvd_{mode}_{metric}"
        for mode in SVD_MODES
        for metric in CSVD_METRICS
    }
    return [col for col in df.columns if col in csvd_allowed or col == CAI_METRIC]


def build_condition_values(subject_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        work = subject_df[
            ["drug", "source", "condition_role", "condition", "subid", "hemisphere", metric]
        ].rename(columns={metric: "value"})
        work["metric"] = metric
        rows.append(work)
    return pd.concat(rows, ignore_index=True)


def build_drug_placebo_values(condition_values: pd.DataFrame) -> pd.DataFrame:
    keys = ["drug", "source", "subid", "hemisphere", "metric"]
    work = condition_values.copy()
    work["condition_value_kind"] = work["condition_role"].map(
        {"drug": "condition_drug", "placebo": "condition_placebo"}
    )
    wide = work.pivot_table(index=keys, columns="condition_value_kind", values="value", aggfunc="first").reset_index()
    if "condition_drug" not in wide.columns or "condition_placebo" not in wide.columns:
        raise RuntimeError("Could not pivot drug/placebo values.")
    wide["delta"] = wide["condition_drug"] - wide["condition_placebo"]
    long = wide.melt(
        id_vars=keys,
        value_vars=["condition_drug", "condition_placebo", "delta"],
        var_name="value_kind",
        value_name="value",
    )
    long["value_kind"] = long["value_kind"].map(
        {"condition_drug": "drug", "condition_placebo": "placebo", "delta": "delta"}
    )
    return long.sort_values(["drug", "hemisphere", "subid", "source", "metric", "value_kind"])


def join_empirical_model(values: pd.DataFrame) -> pd.DataFrame:
    keys = ["drug", "subid", "hemisphere", "metric", "value_kind"]
    empirical = values[values["source"] == "empirical"][keys + ["value"]].rename(columns={"value": "empirical_value"})
    model = values[values["source"] == "model"][keys + ["value"]].rename(columns={"value": "model_value"})
    joined = empirical.merge(model, on=keys, how="inner")
    joined["error_model_minus_empirical"] = joined["model_value"] - joined["empirical_value"]
    joined["abs_error"] = joined["error_model_minus_empirical"].abs()
    return joined


def delta_tests(values: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    subset = values[values["value_kind"] == "delta"]
    for keys, group in subset.groupby(["drug", "hemisphere", "source", "metric"], sort=False):
        row = dict(zip(["drug", "hemisphere", "source", "metric"], keys))
        row.update(delta_test_stats(group["value"].to_numpy(dtype=float), min_n=min_n))
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_table(joined: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    for keys, group in joined.groupby(["drug", "hemisphere", "metric", "value_kind"], sort=False):
        row = dict(zip(["drug", "hemisphere", "metric", "value_kind"], keys))
        row.update(
            agreement_stats(
                group["empirical_value"].to_numpy(dtype=float),
                group["model_value"].to_numpy(dtype=float),
                min_n=min_n,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_delta_scatter(joined: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path, save_plots: bool) -> None:
    if not save_plots:
        return
    plot_df = joined[joined["value_kind"] == "delta"].copy()
    if plot_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_metrics = [
        "csvd_complex_svd_top1_energy",
        "csvd_real_svd_top1_energy",
        CAI_METRIC,
    ]
    for drug in sorted(plot_df["drug"].unique()):
        for hemi in sorted(plot_df["hemisphere"].unique()):
            fig, axes = plt.subplots(1, len(primary_metrics), figsize=(5.0 * len(primary_metrics), 4.4), squeeze=False)
            for ax, metric in zip(axes[0], primary_metrics):
                sub = plot_df[(plot_df["drug"] == drug) & (plot_df["hemisphere"] == hemi) & (plot_df["metric"] == metric)]
                if sub.empty:
                    ax.axis("off")
                    continue
                x = sub["empirical_value"].to_numpy(dtype=float)
                y = sub["model_value"].to_numpy(dtype=float)
                finite_xy = np.isfinite(x) & np.isfinite(y)
                ax.scatter(x[finite_xy], y[finite_xy], s=46, alpha=0.85)
                if np.sum(finite_xy) >= 2 and np.std(x[finite_xy], ddof=1) > 0:
                    slope, intercept, _, _, _ = stats.linregress(x[finite_xy], y[finite_xy])
                    x_line = np.linspace(float(np.min(x[finite_xy])), float(np.max(x[finite_xy])), 100)
                    ax.plot(x_line, intercept + slope * x_line, color="#222222", linewidth=1.3)
                finite = pd.concat([sub["empirical_value"], sub["model_value"]]).replace([np.inf, -np.inf], np.nan).dropna()
                if not finite.empty:
                    lo = float(finite.min())
                    hi = float(finite.max())
                    pad = max((hi - lo) * 0.08, 1e-6)
                    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", linewidth=0.9)
                    ax.set_xlim(lo - pad, hi + pad)
                    ax.set_ylim(lo - pad, hi + pad)
                stat = stats_df[
                    (stats_df["drug"] == drug)
                    & (stats_df["hemisphere"] == hemi)
                    & (stats_df["metric"] == metric)
                    & (stats_df["value_kind"] == "delta")
                ]
                if not stat.empty and pd.notna(stat.iloc[0]["pearson_r"]):
                    row = stat.iloc[0]
                    ax.text(
                        0.04,
                        0.96,
                        f"r={row.pearson_r:.2f}\nR2={row.pearson_r2:.2f}\nEV={row.explained_variance_score:.2f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                    )
                ax.set_title(metric.replace("csvd_", "").replace("_", " "))
                ax.set_xlabel("Empirical delta")
                ax.set_ylabel("Model delta")
                ax.grid(True, alpha=0.3)
            fig.suptitle(f"{drug} {hemi}: model vs empirical Drug-PCB delta", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / f"delta_scatter_{drug}_{hemi}.png", dpi=260, bbox_inches="tight")
            plt.close(fig)


def build_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    ref_root = args.ref_root
    return [
        DatasetSpec(
            drug_label="DMT",
            drug_condition="DMT_DMT",
            placebo_condition="DMT_PCB",
            empirical_root=args.dmt_empirical_root,
            model_root=args.dmt_model_root,
            ref_left=args.dmt_ref_left or ref_root / "DMT_PCB_left_emb_001.npy",
            ref_right=args.dmt_ref_right or ref_root / "DMT_PCB_right_emb_001.npy",
        ),
        DatasetSpec(
            drug_label="LSD",
            drug_condition="LSD_LSD",
            placebo_condition="LSD_PCB",
            empirical_root=args.lsd_empirical_root,
            model_root=args.lsd_model_root,
            ref_left=args.lsd_ref_left or ref_root / "LSD_PCB_left_emb_001.npy",
            ref_right=args.lsd_ref_right or ref_root / "LSD_PCB_right_emb_001.npy",
        ),
    ]


def complete_subjects(entries: pd.DataFrame, spec: DatasetSpec, hemisphere: str) -> list[str]:
    subjects = None
    requirements = [
        ("empirical", spec.drug_condition),
        ("empirical", spec.placebo_condition),
        ("model", spec.drug_condition),
        ("model", spec.placebo_condition),
    ]
    for source, condition in requirements:
        current = set(
            entries[
                (entries["source"] == source)
                & (entries["condition"] == condition)
                & (entries["hemisphere"] == hemisphere)
            ]["subid"]
        )
        subjects = current if subjects is None else subjects & current
    return sorted(subjects or [])


def one_entry(entries: pd.DataFrame, source: str, condition: str, subid: str, hemisphere: str) -> pd.Series:
    rows = entries[
        (entries["source"] == source)
        & (entries["condition"] == condition)
        & (entries["subid"] == subid)
        & (entries["hemisphere"] == hemisphere)
    ]
    if rows.empty:
        raise RuntimeError(f"Missing entry for {source} {condition} S{subid} {hemisphere}")
    return rows.iloc[0]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hemispheres = ["left", "right"] if args.hemisphere == "both" else [args.hemisphere]
    specs = build_specs(args)

    records = []
    failures = []
    dataset_meta = []
    for spec in specs:
        print(f"Discovering {spec.drug_label} entries")
        empirical = discover_entries(spec.empirical_root, source="empirical", hemispheres=hemispheres)
        model = discover_entries(spec.model_root, source="model", hemispheres=hemispheres)
        entries = pd.concat([empirical, model], ignore_index=True)
        refs = {
            "left": load_reference(spec.ref_left, spacing=args.spacing, edge_margin=args.edge_margin),
            "right": load_reference(spec.ref_right, spacing=args.spacing, edge_margin=args.edge_margin),
        }
        dataset_meta.append(
            {
                "drug": spec.drug_label,
                "empirical_root": str(spec.empirical_root),
                "model_root": str(spec.model_root),
                "ref_left": str(spec.ref_left),
                "ref_right": str(spec.ref_right),
                "n_empirical_entries": int(len(empirical)),
                "n_model_entries": int(len(model)),
            }
        )
        for hemi in hemispheres:
            subjects = complete_subjects(entries, spec, hemi)
            if args.limit is not None:
                subjects = subjects[: args.limit]
            print(f"{spec.drug_label} {hemi}: complete subjects={len(subjects)}")
            for subid in subjects:
                for source, condition, role in [
                    ("empirical", spec.placebo_condition, "placebo"),
                    ("empirical", spec.drug_condition, "drug"),
                    ("model", spec.placebo_condition, "placebo"),
                    ("model", spec.drug_condition, "drug"),
                ]:
                    row = one_entry(entries, source, condition, subid, hemi)
                    try:
                        print(f"Processing {spec.drug_label} {hemi} S{subid} {source} {condition}")
                        metrics = compute_phase_metrics(Path(row["phase_cube"]), ref=refs[hemi], spacing=args.spacing)
                        records.append(
                            {
                                "drug": spec.drug_label,
                                "source": source,
                                "condition": condition,
                                "condition_role": role,
                                "subid": subid,
                                "hemisphere": hemi,
                                "phase_cube": str(row["phase_cube"]),
                                **metrics,
                            }
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "drug": spec.drug_label,
                                "source": source,
                                "condition": condition,
                                "condition_role": role,
                                "subid": subid,
                                "hemisphere": hemi,
                                "phase_cube": str(row.get("phase_cube", "")),
                                "error": str(exc),
                            }
                        )
                        print(f"Failed {spec.drug_label} {hemi} S{subid} {source} {condition}: {exc}", file=sys.stderr)

    if not records:
        raise RuntimeError("No metric records were generated.")

    subject_df = pd.DataFrame(records)
    subject_df.to_csv(args.out_dir / "subject_condition_metrics.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.out_dir / "failures.csv", index=False)

    metrics = metric_columns(subject_df)
    condition_values = build_condition_values(subject_df, metrics)
    drug_placebo_values = build_drug_placebo_values(condition_values)
    joined = join_empirical_model(drug_placebo_values)
    tests = delta_tests(drug_placebo_values, min_n=args.min_n)
    agreement = agreement_table(joined, min_n=args.min_n)
    delta_agreement = agreement[agreement["value_kind"] == "delta"].copy()
    manuscript_summary = delta_agreement[
        [
            "drug",
            "hemisphere",
            "metric",
            "n",
            "empirical_mean",
            "model_mean",
            "capture_ratio_mean_model_over_empirical",
            "pearson_r",
            "pearson_r2",
            "pearson_p",
            "identity_r2_score",
            "explained_variance_score",
            "mean_abs_error",
            "rmse",
        ]
    ].copy()

    condition_values.to_csv(args.out_dir / "subject_condition_values_long.csv", index=False)
    drug_placebo_values.to_csv(args.out_dir / "subject_drug_placebo_values_long.csv", index=False)
    joined.to_csv(args.out_dir / "empirical_model_joined_values.csv", index=False)
    tests.to_csv(args.out_dir / "drug_placebo_delta_tests.csv", index=False)
    agreement.to_csv(args.out_dir / "empirical_model_agreement_stats.csv", index=False)
    delta_agreement.to_csv(args.out_dir / "empirical_model_delta_agreement_stats.csv", index=False)
    manuscript_summary.to_csv(args.out_dir / "manuscript_effect_capture_summary.csv", index=False)
    save_delta_scatter(joined, agreement, args.out_dir / "figures" / "delta_scatter", save_plots=not args.no_plots)

    metadata = {
        "description": "cSVD and weighted CAI model-vs-empirical Drug-PCB delta agreement.",
        "datasets": dataset_meta,
        "hemispheres": hemispheres,
        "spacing": float(args.spacing),
        "edge_margin": int(args.edge_margin),
        "min_n": int(args.min_n),
        "limit": args.limit,
        "n_subject_condition_rows": int(len(subject_df)),
        "n_condition_value_rows": int(len(condition_values)),
        "n_joined_rows": int(len(joined)),
        "n_failures": int(len(failures)),
        "primary_delta_metrics": ["csvd_complex_svd_top1_energy", CAI_METRIC],
        "csvd_definition": "SVD of phase-gradient vector-field matrix; complex_svd uses gx+1j*gy, real_svd concatenates real and imaginary components.",
        "cai_definition": "weighted_mean_cos2_alignment = sum(ref_mag * phase_mag * cos(2 theta)) / sum(ref_mag * phase_mag), with phase vectors using -grad/|grad|.",
        "primary_model_agreement": "Empirical Drug-PCB delta is joined to model Drug-PCB delta by drug, hemisphere, subject, and metric.",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote cSVD/CAI model-delta analysis to: {args.out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        raise SystemExit(main())
