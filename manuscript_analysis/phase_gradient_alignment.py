#!/usr/bin/env python
"""
Compare phase-gradient directions with a reference cortical-gradient field.

For each phase_cube.npy bundle, this script computes the phase gradient using
circular central differences, converts it to the negated unit-vector convention
used by MatPhase (-grad / |grad|), compares it with the gradient of a supplied
reference cortical-gradient map, and saves pointwise angle/axial-alignment
arrays plus subject-level paired statistics.
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
import seaborn as sns
from scipy import stats

try:
    import pingouin as pg
except Exception:  # pragma: no cover - optional in base pyproject
    pg = None

try:
    from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field
except ModuleNotFoundError:  # allow running from a source checkout without installation
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))
    from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field


HEMISPHERES = ("left", "right")
DEFAULT_METRICS = ("circ_corr_z", "mean_cos2_alignment", "weighted_mean_cos2_alignment")
PALETTE = {"PCB": "#4575b4", "Drug": "#d73027"}
BUNDLE_RE = re.compile(
    r"(?P<condition>DMTDMT|DMTPCB|LSDLSD|LSDPCB)(?P<subid>\d+)(?P<hemi>[LR])$",
    re.IGNORECASE,
)
CONDITION_LABELS = {
    "DMTDMT": "DMT_DMT",
    "DMTPCB": "DMT_PCB",
    "LSDLSD": "LSD_LSD",
    "LSDPCB": "LSD_PCB",
}


@dataclass(frozen=True)
class BundleEntry:
    condition_code: str
    condition: str
    role: str
    subid: str
    hemisphere: str
    bundle_dir: Path
    phase_cube: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-gradient/cortical-gradient angle alignment analysis."
    )
    parser.add_argument(
        "--drug-root",
        required=True,
        action="append",
        type=Path,
        help=(
            "Root to search recursively for drug-condition phase_cube.npy bundles. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--pcb-root",
        required=True,
        action="append",
        type=Path,
        help=(
            "Root to search recursively for placebo-condition phase_cube.npy bundles. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--drug-condition",
        default=None,
        help="Optional condition label/code to keep for drug, e.g. DMT_DMT or DMTDMT.",
    )
    parser.add_argument(
        "--pcb-condition",
        default=None,
        help="Optional condition label/code to keep for placebo, e.g. DMT_PCB or DMTPCB.",
    )
    parser.add_argument("--ref-left", required=True, type=Path, help="Left reference gmap .npy.")
    parser.add_argument("--ref-right", required=True, type=Path, help="Right reference gmap .npy.")
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_gradient_alignment"),
        type=Path,
        help="Output directory.",
    )
    parser.add_argument("--spacing", default=1.0, type=float, help="Grid spacing for gradients.")
    parser.add_argument(
        "--edge-margin",
        default=2,
        type=int,
        help="Mask this many pixels at every border after gradient computation.",
    )
    parser.add_argument(
        "--save-pointwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save angle/cos/weight cubes. Default: true.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute pointwise cubes when they already exist.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Optional limit on the number of paired subjects per hemisphere.",
    )
    parser.add_argument(
        "--min-pairs",
        default=3,
        type=int,
        help="Minimum paired subjects required for t-tests.",
    )
    parser.add_argument(
        "--polar-sample-fraction",
        default=0.10,
        type=float,
        help="Fraction of finite angle values sampled from each saved angle cube.",
    )
    parser.add_argument(
        "--polar-max-points-per-group",
        default=1_500_000,
        type=int,
        help="Maximum sampled angle values per group per polar plot.",
    )
    parser.add_argument("--polar-bins", default=90, type=int)
    parser.add_argument("--rng-seed", default=42, type=int)
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save PNG figures. Default: true.",
    )
    return parser.parse_args()


def canonical_condition(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def infer_bundle(path: Path, role: str) -> BundleEntry | None:
    match = BUNDLE_RE.search(path.name)
    if match is None:
        return None
    code = match.group("condition").upper()
    hemi = "left" if match.group("hemi").upper() == "L" else "right"
    cube = path / "phase_cube.npy"
    if not cube.exists():
        return None
    return BundleEntry(
        condition_code=code,
        condition=CONDITION_LABELS.get(code, code),
        role=role,
        subid=match.group("subid"),
        hemisphere=hemi,
        bundle_dir=path,
        phase_cube=cube,
    )


def discover_entries(
    roots: Iterable[Path],
    role: str,
    condition_filter: str | None,
) -> pd.DataFrame:
    rows = []
    for root in roots:
        for cube in sorted(root.rglob("phase_cube.npy")):
            entry = infer_bundle(cube.parent, role=role)
            if entry is None:
                continue
            if condition_filter and canonical_condition(entry.condition_code) != condition_filter:
                continue
            rows.append(entry.__dict__)
    if not rows:
        raise RuntimeError(f"No usable {role} phase_cube.npy bundles found.")
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["role", "subid", "hemisphere", "phase_cube"])


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
        pad = np.full(target_shape, np.nan, dtype=float)
        pad[:-1, :] = arr
        return pad
    raise ValueError(f"Cannot align {name} shape {arr.shape} to phase shape {target_shape}")


def load_reference(path: Path, spacing: float, edge_margin: int) -> dict[str, np.ndarray]:
    gmap = np.load(path).astype(np.float64)
    if gmap.ndim != 2:
        raise ValueError(f"Reference gmap must be 2D, got shape={gmap.shape}: {path}")
    gy, gx = np.gradient(gmap, spacing)
    mag = np.hypot(gx, gy)
    valid = np.isfinite(gmap) & np.isfinite(gx) & np.isfinite(gy) & (mag > 0)
    valid = apply_border_mask(valid, edge_margin)
    return {"gmap": gmap, "gx": gx, "gy": gy, "mag": mag, "valid": valid}


def signed_angle_and_cos2(
    ref_gx: np.ndarray,
    ref_gy: np.ndarray,
    phase_ux: np.ndarray,
    phase_uy: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dot = ref_gx * phase_ux + ref_gy * phase_uy
    ref_mag = np.hypot(ref_gx, ref_gy)
    cos_theta = np.divide(
        dot,
        ref_mag,
        out=np.full(ref_gx.shape, np.nan, dtype=np.float64),
        where=valid & (ref_mag > 0) & np.isfinite(phase_ux) & np.isfinite(phase_uy),
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    cross_z = ref_gx * phase_uy - ref_gy * phase_ux
    angle = np.where(cross_z < 0, -angle, angle)
    cos2_theta = np.cos(2.0 * angle)
    angle[~valid] = np.nan
    cos2_theta[~valid] = np.nan
    return angle, cos2_theta


def circ_mean(angles: np.ndarray) -> float:
    if angles.size == 0:
        return math.nan
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def circular_correlation_js(alpha: np.ndarray, beta: np.ndarray) -> float:
    mask = np.isfinite(alpha) & np.isfinite(beta)
    alpha = alpha[mask]
    beta = beta[mask]
    if alpha.size < 3:
        return math.nan
    alpha_bar = circ_mean(alpha)
    beta_bar = circ_mean(beta)
    sa = np.sin(alpha - alpha_bar)
    sb = np.sin(beta - beta_bar)
    denom = float(np.sqrt(np.sum(sa**2) * np.sum(sb**2)))
    if denom <= 0:
        return math.nan
    r = float(np.sum(sa * sb) / denom)
    return float(np.clip(r, -0.999999, 0.999999))


def finite_concat(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def compute_bundle(
    row: pd.Series,
    ref: dict[str, np.ndarray],
    out_dir: Path,
    spacing: float,
    save_pointwise: bool,
    overwrite: bool,
) -> dict[str, object]:
    cube_path = Path(row["phase_cube"])
    phase_cube = np.load(cube_path, mmap_mode="r")
    if phase_cube.ndim != 3:
        raise ValueError(f"Expected phase cube shape (H, W, T), got {phase_cube.shape}: {cube_path}")
    h, w, n_frames = phase_cube.shape

    ref_gx = align_2d(ref["gx"], (h, w), "ref_gx")
    ref_gy = align_2d(ref["gy"], (h, w), "ref_gy")
    ref_mag = align_2d(ref["mag"], (h, w), "ref_mag")
    ref_valid = align_2d(ref["valid"], (h, w), "ref_valid").astype(bool)
    ref_angle = np.arctan2(ref_gy, ref_gx)

    file_stem = f"{row['role']}_{row['condition']}_S{row['subid']}_{row['hemisphere']}"
    point_dir = out_dir / "pointwise"
    angle_path = point_dir / f"angle_diff_{file_stem}.npy"
    cos2_path = point_dir / f"cos2_alignment_{file_stem}.npy"
    weight_path = point_dir / f"weight_{file_stem}.npy"
    frame_path = point_dir / f"weighted_cos2_frame_mean_{file_stem}.npy"

    if (
        save_pointwise
        and not overwrite
        and angle_path.exists()
        and cos2_path.exists()
        and weight_path.exists()
        and frame_path.exists()
    ):
        angle_cube = np.load(angle_path, mmap_mode="r")
        cos2_cube = np.load(cos2_path, mmap_mode="r")
        weight_cube = np.load(weight_path, mmap_mode="r")
        finite = np.isfinite(angle_cube)
        angles = angle_cube[finite].astype(np.float64)
        cvals = cos2_cube[np.isfinite(cos2_cube)].astype(np.float64)
        wvals = weight_cube[np.isfinite(weight_cube) & np.isfinite(cos2_cube)].astype(np.float64)
        weighted_c = cos2_cube[np.isfinite(weight_cube) & np.isfinite(cos2_cube)].astype(np.float64)
        alpha_chunks = []
        beta_chunks = []
        for fidx in range(n_frames):
            m = np.isfinite(angle_cube[:, :, fidx]) & ref_valid
            if np.any(m):
                alpha_chunks.append(ref_angle[m])
                beta_chunks.append((ref_angle + angle_cube[:, :, fidx])[m])
    else:
        if save_pointwise:
            point_dir.mkdir(parents=True, exist_ok=True)
            angle_cube = np.lib.format.open_memmap(
                angle_path, mode="w+", dtype=np.float32, shape=(h, w, n_frames)
            )
            cos2_cube = np.lib.format.open_memmap(
                cos2_path, mode="w+", dtype=np.float32, shape=(h, w, n_frames)
            )
            weight_cube = np.lib.format.open_memmap(
                weight_path, mode="w+", dtype=np.float32, shape=(h, w, n_frames)
            )
        else:
            angle_cube = None
            cos2_cube = None
            weight_cube = None

        frame_weighted = np.full((n_frames,), np.nan, dtype=np.float32)
        cos2_chunks: list[np.ndarray] = []
        angle_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        weighted_cos2_chunks: list[np.ndarray] = []
        alpha_chunks: list[np.ndarray] = []
        beta_chunks: list[np.ndarray] = []

        grad_x, grad_y = compute_phase_gradient(
            np.asarray(phase_cube), spacing=spacing, show_progress=False
        )
        phase_ux, phase_uy, phase_mag = normalize_vector_field(grad_x, grad_y)

        for fidx in range(n_frames):
            p_mag = phase_mag[:, :, fidx]
            valid = (
                ref_valid
                & np.isfinite(phase_cube[:, :, fidx])
                & np.isfinite(phase_ux[:, :, fidx])
                & np.isfinite(phase_uy[:, :, fidx])
                & np.isfinite(p_mag)
                & (p_mag > 0)
            )
            angle, cos2_theta = signed_angle_and_cos2(
                ref_gx=ref_gx,
                ref_gy=ref_gy,
                phase_ux=phase_ux[:, :, fidx],
                phase_uy=phase_uy[:, :, fidx],
                valid=valid,
            )
            weight = ref_mag * p_mag
            weight[~valid] = np.nan

            finite = valid & np.isfinite(angle) & np.isfinite(cos2_theta) & np.isfinite(weight)
            if np.any(finite):
                angle_vals = angle[finite]
                cos2_vals = cos2_theta[finite]
                weight_vals = weight[finite]
                angle_chunks.append(angle_vals)
                cos2_chunks.append(cos2_vals)
                weight_chunks.append(weight_vals)
                weighted_cos2_chunks.append(cos2_vals)
                alpha_chunks.append(ref_angle[finite])
                beta_chunks.append(np.arctan2(phase_uy[:, :, fidx], phase_ux[:, :, fidx])[finite])
                denom = float(np.sum(weight_vals))
                if denom > 0:
                    frame_weighted[fidx] = float(np.sum(weight_vals * cos2_vals) / denom)

            if save_pointwise:
                angle_cube[:, :, fidx] = angle.astype(np.float32)
                cos2_cube[:, :, fidx] = cos2_theta.astype(np.float32)
                weight_cube[:, :, fidx] = weight.astype(np.float32)

        if save_pointwise:
            np.save(frame_path, frame_weighted)

        angles = finite_concat(angle_chunks)
        cvals = finite_concat(cos2_chunks)
        wvals = finite_concat(weight_chunks)
        weighted_c = finite_concat(weighted_cos2_chunks)

    alpha = finite_concat(alpha_chunks)
    beta = finite_concat(beta_chunks)
    circ_r = circular_correlation_js(alpha, beta)
    circ_corr_z = float(np.arctanh(circ_r)) if np.isfinite(circ_r) else math.nan

    weight_denom = float(np.sum(wvals)) if wvals.size else 0.0
    weighted_mean = (
        float(np.sum(wvals * weighted_c) / weight_denom) if weight_denom > 0 else math.nan
    )
    cbar = float(np.mean(cvals)) if cvals.size else math.nan

    return {
        "role": row["role"],
        "condition": row["condition"],
        "condition_code": row["condition_code"],
        "subid": str(row["subid"]),
        "hemisphere": row["hemisphere"],
        "bundle_dir": str(row["bundle_dir"]),
        "phase_cube": str(cube_path),
        "angle_cube": str(angle_path) if save_pointwise else "",
        "cos2_alignment_cube": str(cos2_path) if save_pointwise else "",
        "weight_cube": str(weight_path) if save_pointwise else "",
        "weighted_cos2_frame_mean": str(frame_path) if save_pointwise else "",
        "height": h,
        "width": w,
        "n_frames": n_frames,
        "n_samples": int(cvals.size),
        "circ_corr": circ_r,
        "circ_corr_z": circ_corr_z,
        "mean_cos2_alignment": cbar,
        "weighted_mean_cos2_alignment": weighted_mean,
        "mean_angle": float(circ_mean(angles)) if angles.size else math.nan,
    }


def paired_t_rows(subject_df: pd.DataFrame, metrics: Iterable[str], min_pairs: int) -> pd.DataFrame:
    rows = []

    def mean_std_sem(values: np.ndarray) -> tuple[float, float, float]:
        if values.size == 0:
            return math.nan, math.nan, math.nan
        mean = float(np.mean(values))
        if values.size < 2:
            return mean, math.nan, math.nan
        std = float(np.std(values, ddof=1))
        sem = float(std / math.sqrt(values.size))
        return mean, std, sem

    for hemi in HEMISPHERES:
        hemi_df = subject_df[subject_df["hemisphere"] == hemi]
        for metric in metrics:
            pivot = hemi_df.pivot_table(index="subid", columns="role", values=metric, aggfunc="mean")
            if not {"Drug", "PCB"}.issubset(pivot.columns):
                continue
            pivot = pivot[["PCB", "Drug"]].dropna()
            if len(pivot) < min_pairs:
                rows.append(
                    {
                        "metric": metric,
                        "hemisphere": hemi,
                        "n": int(len(pivot)),
                        "mean_pcb": math.nan,
                        "std_pcb": math.nan,
                        "sem_pcb": math.nan,
                        "mean_drug": math.nan,
                        "std_drug": math.nan,
                        "sem_drug": math.nan,
                        "mean_delta_drug_minus_pcb": math.nan,
                        "std_delta": math.nan,
                        "sem_delta": math.nan,
                        "t": math.nan,
                        "p": math.nan,
                        "cohen_dz": math.nan,
                    }
                )
                continue
            pcb = pivot["PCB"].to_numpy(dtype=float)
            drug = pivot["Drug"].to_numpy(dtype=float)
            delta = drug - pcb
            pcb_mean, pcb_std, pcb_sem = mean_std_sem(pcb)
            drug_mean, drug_std, drug_sem = mean_std_sem(drug)
            delta_mean, delta_std, delta_sem = mean_std_sem(delta)
            if len(pivot) >= max(min_pairs, 2):
                t_val, p_val = stats.ttest_rel(drug, pcb, nan_policy="omit")
                t_val = float(t_val)
                p_val = float(p_val)
            else:
                t_val = math.nan
                p_val = math.nan
            dz = float(delta_mean / delta_std) if delta_std and np.isfinite(delta_std) else math.nan
            rows.append(
                {
                    "metric": metric,
                    "hemisphere": hemi,
                    "n": int(len(pivot)),
                    "mean_pcb": pcb_mean,
                    "std_pcb": pcb_std,
                    "sem_pcb": pcb_sem,
                    "mean_drug": drug_mean,
                    "std_drug": drug_std,
                    "sem_drug": drug_sem,
                    "mean_delta_drug_minus_pcb": delta_mean,
                    "std_delta": delta_std,
                    "sem_delta": delta_sem,
                    "t": t_val,
                    "p": p_val,
                    "cohen_dz": dz,
                }
            )
    return pd.DataFrame(rows)


def rm_anova_rows(subject_df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    if pg is None:
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "test_type": "two_way_rm_anova",
                    "status": "skipped_missing_pingouin",
                }
                for metric in metrics
            ]
        )
    for metric in metrics:
        work = subject_df[["subid", "role", "hemisphere", metric]].copy()
        work = work.dropna(subset=[metric])
        expected = work["role"].nunique() * work["hemisphere"].nunique()
        counts = work.groupby("subid").apply(
            lambda x: x[["role", "hemisphere"]].drop_duplicates().shape[0],
            include_groups=False,
        )
        keep = counts[counts == expected].index
        work = work[work["subid"].isin(keep)]
        if work["subid"].nunique() < 2 or work["role"].nunique() < 2 or work["hemisphere"].nunique() < 2:
            rows.append(
                {
                    "metric": metric,
                    "test_type": "two_way_rm_anova",
                    "status": "skipped_insufficient_complete_cases",
                    "n": int(work["subid"].nunique()),
                }
            )
            continue
        work = work.groupby(["subid", "role", "hemisphere"], as_index=False)[metric].mean()
        try:
            aov = pg.rm_anova(
                data=work,
                dv=metric,
                within=["role", "hemisphere"],
                subject="subid",
                detailed=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "metric": metric,
                    "test_type": "two_way_rm_anova",
                    "status": f"failed: {exc}",
                    "n": int(work["subid"].nunique()),
                }
            )
            continue
        for _, effect in aov.iterrows():
            f_val = float(effect["F"]) if pd.notna(effect.get("F")) else math.nan
            ddof1 = float(effect["ddof1"]) if pd.notna(effect.get("ddof1")) else math.nan
            ddof2 = float(effect["ddof2"]) if pd.notna(effect.get("ddof2")) else math.nan
            eta_p2 = (
                float((f_val * ddof1) / ((f_val * ddof1) + ddof2))
                if np.isfinite(f_val) and np.isfinite(ddof1) and np.isfinite(ddof2)
                else math.nan
            )
            rows.append(
                {
                    "metric": metric,
                    "test_type": "two_way_rm_anova",
                    "status": "ok",
                    "effect": effect.get("Source", ""),
                    "n": int(work["subid"].nunique()),
                    "F": f_val,
                    "ddof1": ddof1,
                    "ddof2": ddof2,
                    "p": float(effect["p-unc"]) if pd.notna(effect.get("p-unc")) else math.nan,
                    "eta_p2": eta_p2,
                    "ng2": float(effect["ng2"]) if pd.notna(effect.get("ng2")) else math.nan,
                    "eps": float(effect["eps"]) if pd.notna(effect.get("eps")) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def save_paired_plots(subject_df: pd.DataFrame, metrics: Iterable[str], fig_dir: Path, save: bool) -> None:
    if not save:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    for hemi in HEMISPHERES:
        sub = subject_df[subject_df["hemisphere"] == hemi]
        for metric in metrics:
            pivot = sub.pivot_table(index="subid", columns="role", values=metric, aggfunc="mean")
            if not {"PCB", "Drug"}.issubset(pivot.columns):
                continue
            pivot = pivot[["PCB", "Drug"]].dropna()
            if pivot.empty:
                continue
            tidy = pivot.reset_index().melt(id_vars="subid", var_name="role", value_name=metric)
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            sns.violinplot(
                data=tidy,
                x="role",
                y=metric,
                order=["PCB", "Drug"],
                palette=PALETTE,
                inner="quartile",
                cut=0,
                ax=ax,
            )
            for _, row in pivot.iterrows():
                ax.plot([0, 1], [row["PCB"], row["Drug"]], color="gray", alpha=0.45, linewidth=0.9)
                ax.scatter(
                    [0, 1],
                    [row["PCB"], row["Drug"]],
                    color=[PALETTE["PCB"], PALETTE["Drug"]],
                    edgecolor="white",
                    linewidth=0.5,
                    s=30,
                    zorder=3,
                )
            ax.set_title(f"{metric} ({hemi})")
            ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(fig_dir / f"paired_{metric}_{hemi}.png", dpi=300)
            plt.close(fig)


def sample_angles(paths: Iterable[Path], fraction: float, max_points: int, rng: np.random.Generator) -> np.ndarray:
    samples = []
    for path in paths:
        if not path.exists():
            continue
        cube = np.load(path, mmap_mode="r")
        finite = cube[np.isfinite(cube)]
        if finite.size == 0:
            continue
        take = min(finite.size, max(1, int(math.ceil(finite.size * fraction))))
        idx = rng.choice(finite.size, size=take, replace=False)
        samples.append(finite[idx].astype(np.float32))
    if not samples:
        return np.empty((0,), dtype=np.float32)
    out = np.concatenate(samples)
    if out.size > max_points:
        idx = rng.choice(out.size, size=max_points, replace=False)
        out = out[idx]
    return out


def save_polar_plots(
    subject_df: pd.DataFrame,
    fig_dir: Path,
    fraction: float,
    max_points: int,
    bins: int,
    rng: np.random.Generator,
    save: bool,
) -> None:
    if not save:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    for hemi in HEMISPHERES:
        sub = subject_df[subject_df["hemisphere"] == hemi]
        pcb_paths = [Path(p) for p in sub[sub["role"] == "PCB"]["angle_cube"] if str(p)]
        drug_paths = [Path(p) for p in sub[sub["role"] == "Drug"]["angle_cube"] if str(p)]
        pcb_vals = sample_angles(pcb_paths, fraction, max_points, rng)
        drug_vals = sample_angles(drug_paths, fraction, max_points, rng)
        if pcb_vals.size == 0 or drug_vals.size == 0:
            continue
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection="polar")
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        ax1.hist(pcb_vals, bins=bins, color=PALETTE["PCB"], alpha=0.85)
        ax2.hist(drug_vals, bins=bins, color=PALETTE["Drug"], alpha=0.85)
        ax1.set_title(f"PCB ({hemi})")
        ax2.set_title(f"Drug ({hemi})")
        fig.suptitle(f"Signed angle difference distribution ({hemi})")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(fig_dir / f"polar_angle_diff_{hemi}.png", dpi=300)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drug_filter = canonical_condition(args.drug_condition)
    pcb_filter = canonical_condition(args.pcb_condition)
    drug_df = discover_entries(args.drug_root, role="Drug", condition_filter=drug_filter)
    pcb_df = discover_entries(args.pcb_root, role="PCB", condition_filter=pcb_filter)
    entries = pd.concat([pcb_df, drug_df], ignore_index=True)

    refs = {
        "left": load_reference(args.ref_left, spacing=args.spacing, edge_margin=args.edge_margin),
        "right": load_reference(args.ref_right, spacing=args.spacing, edge_margin=args.edge_margin),
    }

    records = []
    failures = []
    for hemi in HEMISPHERES:
        hemi_entries = entries[entries["hemisphere"] == hemi]
        pcb_subs = set(hemi_entries[hemi_entries["role"] == "PCB"]["subid"])
        drug_subs = set(hemi_entries[hemi_entries["role"] == "Drug"]["subid"])
        paired_subs = sorted(pcb_subs & drug_subs)
        if args.limit is not None:
            paired_subs = paired_subs[: args.limit]
        print(f"{hemi}: paired subjects={len(paired_subs)}")
        for subid in paired_subs:
            for role in ("PCB", "Drug"):
                candidates = hemi_entries[(hemi_entries["subid"] == subid) & (hemi_entries["role"] == role)]
                if candidates.empty:
                    continue
                row = candidates.iloc[0]
                try:
                    print(f"Processing {role} S{subid} {hemi}: {Path(row['bundle_dir']).name}")
                    records.append(
                        compute_bundle(
                            row=row,
                            ref=refs[hemi],
                            out_dir=args.out_dir,
                            spacing=args.spacing,
                            save_pointwise=args.save_pointwise,
                            overwrite=args.overwrite,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "role": role,
                            "subid": subid,
                            "hemisphere": hemi,
                            "bundle_dir": str(row.get("bundle_dir", "")),
                            "error": str(exc),
                        }
                    )
                    print(f"Failed {role} S{subid} {hemi}: {exc}", file=sys.stderr)

    subject_df = pd.DataFrame(records)
    subject_df.to_csv(args.out_dir / "subject_metrics.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.out_dir / "failures.csv", index=False)

    paired_stats = paired_t_rows(subject_df, DEFAULT_METRICS, min_pairs=args.min_pairs)
    paired_stats.to_csv(args.out_dir / "paired_ttests.csv", index=False)
    anova = rm_anova_rows(subject_df, DEFAULT_METRICS)
    anova.to_csv(args.out_dir / "two_way_rm_anova.csv", index=False)

    fig_dir = args.out_dir / "figures"
    save_paired_plots(subject_df, DEFAULT_METRICS, fig_dir=fig_dir, save=args.save_plots)
    save_polar_plots(
        subject_df,
        fig_dir=fig_dir,
        fraction=args.polar_sample_fraction,
        max_points=args.polar_max_points_per_group,
        bins=args.polar_bins,
        rng=np.random.default_rng(args.rng_seed),
        save=args.save_plots and args.save_pointwise,
    )

    metadata = {
        "drug_root": [str(p) for p in args.drug_root],
        "pcb_root": [str(p) for p in args.pcb_root],
        "drug_condition": args.drug_condition,
        "pcb_condition": args.pcb_condition,
        "ref_left": str(args.ref_left),
        "ref_right": str(args.ref_right),
        "spacing": args.spacing,
        "edge_margin": args.edge_margin,
        "save_pointwise": args.save_pointwise,
        "n_subject_rows": int(len(subject_df)),
        "n_failures": int(len(failures)),
        "metrics": list(DEFAULT_METRICS),
        "phase_direction_convention": "-grad / |grad|",
        "axial_alignment": "c_i = cos(2 * delta_theta_i); positive means aligned with the reference axis, negative means perpendicular",
        "circular_correlation": "Jammalamadaka-Sengupta circular-circular correlation, Fisher z=arctanh(r)",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote phase-gradient alignment analysis to: {args.out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        raise SystemExit(main())
