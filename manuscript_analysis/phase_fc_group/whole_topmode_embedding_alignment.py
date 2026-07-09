#!/usr/bin/env python
"""
Whole-hemisphere top-mode angle correlation with condition-matched embedding gradients.

For each subject x condition x hemisphere, this script extracts the whole-
hemisphere top mode from the phase-gradient vector-field cube and correlates its
angle field with the matching group embedding-gradient angle field:

    Drug topmode  vs <drug-condition> embedding gradient
    PCB topmode   vs <pcb-condition> embedding gradient

The main subject metrics are circular-circular correlation and axial
circular-circular correlation. Group inference is done over subject Fisher-z
values, not over pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    from network_topmode_alignment import (
        ROLE_PALETTE,
        align_2d,
        build_field_cube,
        extract_top_mode,
        load_reference,
    )
except ModuleNotFoundError:
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from network_topmode_alignment import (
        ROLE_PALETTE,
        align_2d,
        build_field_cube,
        extract_top_mode,
        load_reference,
    )


CORR_METRICS = {
    "circ_corr_z": "Circular correlation z",
    "axial_circ_corr_z": "Axial circular correlation z",
}
ANGLE_METRICS = {
    "mean_cos2_alignment": "Mean cos(2theta)",
    "weighted_mean_cos2_alignment": "Weighted mean cos(2theta)",
}
ALL_METRICS = {**CORR_METRICS, **ANGLE_METRICS}
RAW_R_COLUMNS = {
    "circ_corr_z": "circ_corr_r",
    "axial_circ_corr_z": "axial_circ_corr_r",
}
CONDITION_RE = re.compile(r"^(?P<condition>[A-Za-z]+_[A-Za-z]+)_S(?P<subid>\d+)")
SUFFIX_RE = re.compile(r"(?P<code>[A-Za-z]+)(?P<subid>\d+)(?P<hemi>[LR])$")


@dataclass(frozen=True)
class DetectEntry:
    contrast: str
    drug_condition: str
    pcb_condition: str
    condition: str
    role: str
    subid: str
    hemisphere: str
    detect_dir: Path
    phase_cube: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whole-hemisphere top-mode circular correlation with condition-matched embedding gradients."
    )
    parser.add_argument(
        "--detect-root",
        default=Path("detect_results"),
        type=Path,
        help="Root to scan recursively for phase_cube.npy, e.g. detect_results, detect_results/DMT, or detect_results/LSD.",
    )
    parser.add_argument("--drug-condition", default="DMT_DMT")
    parser.add_argument("--pcb-condition", default="DMT_PCB")
    parser.add_argument(
        "--contrast",
        action="append",
        default=None,
        help=(
            "Optional repeated contrast spec as label:drug_condition:pcb_condition, "
            "e.g. DMT:DMT_DMT:DMT_PCB --contrast LSD:LSD_LSD:LSD_PCB. "
            "If omitted, uses --drug-condition/--pcb-condition as one contrast."
        ),
    )
    parser.add_argument("--emb-root", default=Path("analysis_outputs/group_emb_to_grid/emb_001"), type=Path)
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/whole_topmode_embedding_alignment"),
        type=Path,
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument("--field-method", choices=["phase_gradient", "optical_flow"], default="phase_gradient")
    parser.add_argument("--svd-mode", choices=["real_svd", "complex_svd"], default="real_svd")
    parser.add_argument("--spacing", default=1.0, type=float)
    parser.add_argument("--edge-margin", default=2, type=int)
    parser.add_argument("--max-bundles", default=None, type=int)
    parser.add_argument("--plot-sample-points", default=2500, type=int)
    parser.add_argument("--quiver-step", default=6, type=int)
    parser.add_argument("--random-seed", default=20260507, type=int)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def circ_mean(angles: np.ndarray) -> float:
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return math.nan
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def circular_correlation_js(alpha: np.ndarray, beta: np.ndarray) -> tuple[float, float, int]:
    """Jammalamadaka-Sengupta circular-circular correlation with approximate pixel p."""
    mask = np.isfinite(alpha) & np.isfinite(beta)
    alpha = alpha[mask]
    beta = beta[mask]
    n = int(alpha.size)
    if n < 3:
        return math.nan, math.nan, n
    alpha_bar = circ_mean(alpha)
    beta_bar = circ_mean(beta)
    sa = np.sin(alpha - alpha_bar)
    sb = np.sin(beta - beta_bar)
    denom = float(np.sqrt(np.sum(sa * sa) * np.sum(sb * sb)))
    if denom <= 0:
        return math.nan, math.nan, n
    r = float(np.sum(sa * sb) / denom)
    r = float(np.clip(r, -0.999999, 0.999999))
    t_val = r * math.sqrt((n - 2) / max(1.0e-12, 1.0 - r * r))
    p_val = float(2.0 * stats.t.sf(abs(t_val), df=n - 2))
    return r, p_val, n


def fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return math.nan
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def canonical_condition(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def parse_contrasts(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.contrast:
        label = args.drug_condition.split("_", maxsplit=1)[0]
        return [{"label": label, "drug_condition": args.drug_condition, "pcb_condition": args.pcb_condition}]
    contrasts = []
    for spec in args.contrast:
        parts = spec.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid --contrast {spec!r}; expected label:drug_condition:pcb_condition")
        label, drug_condition, pcb_condition = parts
        contrasts.append({"label": label, "drug_condition": drug_condition, "pcb_condition": pcb_condition})
    labels = [item["label"] for item in contrasts]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate contrast labels are not allowed: {labels}")
    return contrasts


def parse_detect_dir_name(name: str, drug_condition: str, pcb_condition: str) -> tuple[str, str, str, str] | None:
    cond_match = CONDITION_RE.search(name)
    suffix_match = SUFFIX_RE.search(name)
    if cond_match is None or suffix_match is None:
        return None
    condition = cond_match.group("condition")
    subid = cond_match.group("subid")
    hemi = "left" if suffix_match.group("hemi").upper() == "L" else "right"
    canon = canonical_condition(condition)
    if canon == canonical_condition(drug_condition):
        role = "Drug"
    elif canon == canonical_condition(pcb_condition):
        role = "PCB"
    else:
        return None
    return condition, role, subid, hemi


def discover_detect_entries(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    hemispheres = {"left", "right"} if args.hemisphere == "both" else {args.hemisphere}
    contrasts = parse_contrasts(args)
    for cube_path in sorted(args.detect_root.rglob("phase_cube.npy")):
        detect_dir = cube_path.parent
        for contrast in contrasts:
            parsed = parse_detect_dir_name(
                detect_dir.name,
                contrast["drug_condition"],
                contrast["pcb_condition"],
            )
            if parsed is None:
                continue
            condition, role, subid, hemi = parsed
            if hemi not in hemispheres:
                continue
            rows.append(
                DetectEntry(
                    contrast=contrast["label"],
                    drug_condition=contrast["drug_condition"],
                    pcb_condition=contrast["pcb_condition"],
                    condition=condition,
                    role=role,
                    subid=subid,
                    hemisphere=hemi,
                    detect_dir=detect_dir,
                    phase_cube=cube_path,
                ).__dict__
            )
    df = pd.DataFrame(rows)
    if df.empty:
        contrast_text = ", ".join(f"{c['label']}:{c['drug_condition']}:{c['pcb_condition']}" for c in contrasts)
        raise RuntimeError(f"No phase_cube.npy entries for {contrast_text} under {args.detect_root}")
    df = df.drop_duplicates(subset=["contrast", "role", "subid", "hemisphere", "phase_cube"])
    df = df.sort_values(["contrast", "role", "subid", "hemisphere"]).reset_index(drop=True)
    if args.max_bundles is not None:
        df = df.head(args.max_bundles).copy()
    return df


def condition_ref_path(emb_root: Path, condition: str, hemisphere: str) -> Path:
    return emb_root / f"{condition}_{hemisphere}_emb_001.npy"


def load_condition_refs(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    refs = {}
    for contrast in parse_contrasts(args):
        for role, condition in [("Drug", contrast["drug_condition"]), ("PCB", contrast["pcb_condition"])]:
            for hemi in ["left", "right"]:
                path = condition_ref_path(args.emb_root, condition, hemi)
                if not path.exists():
                    raise FileNotFoundError(f"Missing {contrast['label']} {role} {hemi} embedding map: {path}")
                ref = load_reference(path, spacing=args.spacing, edge_margin=args.edge_margin)
                ref["path"] = str(path)
                ref["condition"] = condition
                ref["contrast"] = contrast["label"]
                refs[(contrast["label"], role, hemi)] = ref
    return refs


def plot_quiver_field(
    field_x: np.ndarray,
    field_y: np.ndarray,
    valid: np.ndarray,
    title: str,
    out_path: Path,
    step: int,
    color: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mag = np.hypot(field_x, field_y)
    good = valid & np.isfinite(field_x) & np.isfinite(field_y) & np.isfinite(mag) & (mag > 0)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    if np.any(good):
        ux = np.divide(field_x, mag, out=np.zeros_like(field_x, dtype=np.float64), where=good)
        uy = np.divide(field_y, mag, out=np.zeros_like(field_y, dtype=np.float64), where=good)
        yy, xx = np.mgrid[0 : field_x.shape[0], 0 : field_x.shape[1]]
        sample = np.zeros_like(good, dtype=bool)
        sample[::step, ::step] = good[::step, ::step]
        ax.quiver(
            xx[sample],
            yy[sample],
            ux[sample],
            uy[sample],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=0.28,
            width=0.0026,
        )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_quiver(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    mode_x: np.ndarray,
    mode_y: np.ndarray,
    valid: np.ndarray,
    title: str,
    out_path: Path,
    step: int,
    role: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_mag = np.hypot(ref_x, ref_y)
    mode_mag = np.hypot(mode_x, mode_y)
    good = (
        valid
        & np.isfinite(ref_x)
        & np.isfinite(ref_y)
        & np.isfinite(mode_x)
        & np.isfinite(mode_y)
        & (ref_mag > 0)
        & (mode_mag > 0)
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if np.any(good):
        yy, xx = np.mgrid[0 : ref_x.shape[0], 0 : ref_x.shape[1]]
        sample = np.zeros_like(good, dtype=bool)
        sample[::step, ::step] = good[::step, ::step]
        ref_ux = np.divide(ref_x, ref_mag, out=np.zeros_like(ref_x, dtype=np.float64), where=good)
        ref_uy = np.divide(ref_y, ref_mag, out=np.zeros_like(ref_y, dtype=np.float64), where=good)
        mode_ux = np.divide(mode_x, mode_mag, out=np.zeros_like(mode_x, dtype=np.float64), where=good)
        mode_uy = np.divide(mode_y, mode_mag, out=np.zeros_like(mode_y, dtype=np.float64), where=good)
        ax.quiver(
            xx[sample],
            yy[sample],
            ref_ux[sample],
            ref_uy[sample],
            color="#222222",
            alpha=0.72,
            angles="xy",
            scale_units="xy",
            scale=0.30,
            width=0.0023,
            label="Embedding gradient",
        )
        ax.quiver(
            xx[sample],
            yy[sample],
            mode_ux[sample],
            mode_uy[sample],
            color=ROLE_PALETTE[role],
            alpha=0.72,
            angles="xy",
            scale_units="xy",
            scale=0.30,
            width=0.0023,
            label="Top mode",
        )
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_reference_quiver_plots(
    refs: dict[tuple[str, str], dict[str, np.ndarray]],
    fig_dir: Path,
    step: int,
) -> None:
    seen = set()
    for (_contrast, role, hemi), ref in refs.items():
        key = (ref["condition"], hemi)
        if key in seen:
            continue
        seen.add(key)
        plot_quiver_field(
            np.asarray(ref["gx"], dtype=np.float64),
            np.asarray(ref["gy"], dtype=np.float64),
            np.asarray(ref["valid"], dtype=bool),
            title=f"{ref['condition']} embedding gradient ({hemi})",
            out_path=fig_dir / "embedding_gradient_quiver" / f"emb_gradient_{ref['condition']}_{hemi}.png",
            step=step,
            color=ROLE_PALETTE.get(role, "#222222"),
        )


def sample_for_plot(
    emb_angle: np.ndarray,
    mode_angle: np.ndarray,
    valid: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    yx = np.flatnonzero(valid & np.isfinite(emb_angle) & np.isfinite(mode_angle))
    if yx.size == 0:
        return pd.DataFrame(columns=["emb_angle", "topmode_angle", "emb_angle_axial", "topmode_angle_axial"])
    take = min(int(max_points), int(yx.size))
    if take < yx.size:
        yx = rng.choice(yx, size=take, replace=False)
    emb = emb_angle.reshape(-1)[yx]
    top = mode_angle.reshape(-1)[yx]
    return pd.DataFrame(
        {
            "emb_angle": emb,
            "topmode_angle": top,
            "angle_diff": np.angle(np.exp(1j * (top - emb))),
            "emb_angle_axial": np.angle(np.exp(1j * 2.0 * emb)),
            "topmode_angle_axial": np.angle(np.exp(1j * 2.0 * top)),
            "angle_diff_axial": np.angle(np.exp(1j * 2.0 * (top - emb))),
        }
    )


def summarize_one_bundle(
    entry: pd.Series,
    refs: dict[tuple[str, str], dict[str, np.ndarray]],
    field_method: str,
    svd_mode: str,
    plot_sample_points: int,
    rng: np.random.Generator,
    topmode_fig_dir: Path | None,
    quiver_step: int,
    overlay_fig_dir: Path | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    phase_cube = np.asarray(np.load(entry.phase_cube, mmap_mode="r"), dtype=np.float64)
    field_cube = build_field_cube(phase_cube, field_method)
    h, w, frames = field_cube.shape
    support = np.isfinite(field_cube).all(axis=2)

    ref = refs[(entry.contrast, entry.role, entry.hemisphere)]
    ref_gx = align_2d(ref["gx"], (h, w), "ref_gx")
    ref_gy = align_2d(ref["gy"], (h, w), "ref_gy")
    ref_valid = align_2d(ref["valid"], (h, w), "ref_valid").astype(bool)
    ref_mag = np.hypot(ref_gx, ref_gy)
    emb_angle = np.arctan2(ref_gy, ref_gx)

    mode_map, extra = extract_top_mode(field_cube, support, svd_mode)
    mode_x = np.real(mode_map)
    mode_y = np.imag(mode_map)
    mode_mag = np.hypot(mode_x, mode_y)
    mode_angle = np.arctan2(mode_y, mode_x)

    if topmode_fig_dir is not None:
        plot_quiver_field(
            mode_x,
            mode_y,
            support & np.isfinite(mode_x) & np.isfinite(mode_y) & (mode_mag > 0),
            title=f"{entry.contrast} {entry.role} S{entry.subid} topmode ({entry.hemisphere})",
            out_path=topmode_fig_dir / f"topmode_quiver_{entry.contrast}_{entry.role}_S{entry.subid}_{entry.hemisphere}.png",
            step=quiver_step,
            color=ROLE_PALETTE[entry.role],
        )

    valid = (
        support
        & ref_valid
        & np.isfinite(emb_angle)
        & np.isfinite(mode_angle)
        & np.isfinite(ref_mag)
        & np.isfinite(mode_mag)
        & (ref_mag > 0)
        & (mode_mag > 0)
    )
    alpha = emb_angle[valid]
    beta = mode_angle[valid]
    circ_r, circ_p, n_samples = circular_correlation_js(alpha, beta)
    axial_r, axial_p, _ = circular_correlation_js(2.0 * alpha, 2.0 * beta)
    dot = ref_gx * mode_x + ref_gy * mode_y
    cross = ref_gx * mode_y - ref_gy * mode_x
    signed_angle_diff = np.arctan2(cross, dot)
    cos2 = np.cos(2.0 * signed_angle_diff)
    weights = ref_mag * mode_mag
    good_cos = valid & np.isfinite(cos2) & np.isfinite(weights)
    cos2_vals = cos2[good_cos]
    weight_vals = weights[good_cos]
    weight_denom = float(np.sum(weight_vals)) if weight_vals.size else 0.0
    weighted_cos2 = float(np.sum(weight_vals * cos2_vals) / weight_denom) if weight_denom > 0 else math.nan
    angle_vals = signed_angle_diff[good_cos]

    if overlay_fig_dir is not None:
        plot_overlay_quiver(
            ref_gx,
            ref_gy,
            mode_x,
            mode_y,
            good_cos,
            title=f"{entry.contrast} {entry.role} S{entry.subid} topmode over {ref['condition']} emb ({entry.hemisphere})",
            out_path=overlay_fig_dir / f"overlay_quiver_{entry.contrast}_{entry.role}_S{entry.subid}_{entry.hemisphere}.png",
            step=quiver_step,
            role=entry.role,
        )

    plot_df = sample_for_plot(emb_angle, mode_angle, valid, plot_sample_points, rng)
    row = {
        "contrast": entry.contrast,
        "drug_condition": entry.drug_condition,
        "pcb_condition": entry.pcb_condition,
        "condition": entry.condition,
        "role": entry.role,
        "subid": str(entry.subid),
        "hemisphere": entry.hemisphere,
        "network": "whole",
        "network_index": -1,
        "emb_condition": ref["condition"],
        "emb_path": ref["path"],
        "phase_cube": str(entry.phase_cube),
        "field_method": field_method,
        "svd_mode": svd_mode,
        "height": int(h),
        "width": int(w),
        "field_frames": int(frames),
        "n_samples": int(n_samples),
        "whole_support_pixels": int(np.sum(support)),
        "top1_energy": float(extra.get("top1_energy", math.nan)),
        "n_mode_voxels": float(extra.get("n_mode_voxels", math.nan)),
        "circ_corr_r": circ_r,
        "circ_corr_z": fisher_z(circ_r),
        "circ_corr_pixel_p_approx": circ_p,
        "axial_circ_corr_r": axial_r,
        "axial_circ_corr_z": fisher_z(axial_r),
        "axial_circ_corr_pixel_p_approx": axial_p,
        "mean_cos2_alignment": float(np.nanmean(cos2_vals)) if cos2_vals.size else math.nan,
        "std_cos2_alignment": float(np.nanstd(cos2_vals, ddof=1)) if cos2_vals.size > 1 else math.nan,
        "sem_cos2_alignment": float(stats.sem(cos2_vals, nan_policy="omit")) if cos2_vals.size > 1 else math.nan,
        "weighted_mean_cos2_alignment": weighted_cos2,
        "mean_signed_angle_diff_rad": float(np.nanmean(angle_vals)) if angle_vals.size else math.nan,
        "mean_abs_angle_diff_rad": float(np.nanmean(np.abs(angle_vals))) if angle_vals.size else math.nan,
        "std_angle_diff_rad": float(np.nanstd(angle_vals, ddof=1)) if angle_vals.size > 1 else math.nan,
        "angle_samples": int(cos2_vals.size),
    }
    for col in ["contrast", "condition", "role", "subid", "hemisphere", "emb_condition"]:
        plot_df[col] = row[col]
    return row, plot_df


def paired_deltas(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in subject_df.groupby(["contrast", "subid", "hemisphere"], sort=False):
        pivot = sub.pivot_table(index="subid", columns="role", values=list(ALL_METRICS), aggfunc="mean")
        if "Drug" not in pivot.columns.get_level_values(1) or "PCB" not in pivot.columns.get_level_values(1):
            continue
        for metric in ALL_METRICS:
            try:
                drug = float(pivot[(metric, "Drug")].iloc[0])
                pcb = float(pivot[(metric, "PCB")].iloc[0])
            except KeyError:
                continue
            rows.append(
                {
                    "contrast": keys[0],
                    "subid": keys[1],
                    "hemisphere": keys[2],
                    "metric": metric,
                    "metric_label": ALL_METRICS[metric],
                    "drug_value": drug,
                    "pcb_value": pcb,
                    "delta_drug_minus_pcb": drug - pcb if np.isfinite(drug) and np.isfinite(pcb) else math.nan,
                    "drug_r": float(np.tanh(drug)) if metric in RAW_R_COLUMNS and np.isfinite(drug) else math.nan,
                    "pcb_r": float(np.tanh(pcb)) if metric in RAW_R_COLUMNS and np.isfinite(pcb) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _mean_std_sem(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, math.nan, math.nan
    std = float(np.std(values, ddof=1))
    return mean, std, float(std / math.sqrt(values.size))


def group_correlation_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in subject_df.groupby(["contrast", "role", "hemisphere"], sort=False):
        contrast, role, hemi = keys
        for metric, label in ALL_METRICS.items():
            values = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=float)
            mean_value, std_value, sem_value = _mean_std_sem(values)
            if values.size >= 3:
                t_val, p_val = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
            else:
                t_val, p_val = math.nan, math.nan
            rows.append(
                {
                    "contrast": "one_sample_vs_zero",
                    "drug_contrast": contrast,
                    "role": role,
                    "hemisphere": hemi,
                    "metric": metric,
                    "metric_label": label,
                    "n": int(values.size),
                    "mean_value": mean_value,
                    "std_value": std_value,
                    "sem_value": sem_value,
                    "mean_z": mean_value if metric in RAW_R_COLUMNS else math.nan,
                    "std_z": std_value if metric in RAW_R_COLUMNS else math.nan,
                    "sem_z": sem_value if metric in RAW_R_COLUMNS else math.nan,
                    "group_r": float(np.tanh(mean_value)) if metric in RAW_R_COLUMNS and np.isfinite(mean_value) else math.nan,
                    "group_r_sem_approx": float(np.tanh(mean_value + sem_value) - np.tanh(mean_value)) if metric in RAW_R_COLUMNS and np.isfinite(mean_value) and np.isfinite(sem_value) else math.nan,
                    "t_vs_zero": float(t_val) if np.isfinite(t_val) else math.nan,
                    "p_vs_zero": float(p_val) if np.isfinite(p_val) else math.nan,
                    "cohen_dz": float(mean_value / std_value) if np.isfinite(std_value) and std_value > 0 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_ttest_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if delta_df.empty:
        return pd.DataFrame(rows)
    for keys, sub in delta_df.groupby(["contrast", "hemisphere", "metric"], sort=False):
        contrast, hemi, metric = keys
        vals = pd.to_numeric(sub["delta_drug_minus_pcb"], errors="coerce").dropna().to_numpy(dtype=float)
        mean_delta, std_delta, sem_delta = _mean_std_sem(vals)
        if vals.size >= 3:
            t_val, p_val = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
        else:
            t_val, p_val = math.nan, math.nan
        rows.append(
            {
                "contrast": "paired_drug_minus_pcb",
                "drug_contrast": contrast,
                "hemisphere": hemi,
                "metric": metric,
                "metric_label": ALL_METRICS.get(metric, metric),
                "n": int(vals.size),
                "mean_delta": mean_delta,
                "std_delta": std_delta,
                "sem_delta": sem_delta,
                "mean_delta_z": mean_delta if metric in RAW_R_COLUMNS else math.nan,
                "std_delta_z": std_delta if metric in RAW_R_COLUMNS else math.nan,
                "sem_delta_z": sem_delta if metric in RAW_R_COLUMNS else math.nan,
                "mean_delta_r_approx": float(np.tanh(mean_delta)) if metric in RAW_R_COLUMNS and np.isfinite(mean_delta) else math.nan,
                "t": float(t_val) if np.isfinite(t_val) else math.nan,
                "p": float(p_val) if np.isfinite(p_val) else math.nan,
                "cohen_dz": float(mean_delta / std_delta) if np.isfinite(std_delta) and std_delta > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def save_subject_correlation_plots(plot_df: pd.DataFrame, fig_dir: Path) -> None:
    if plot_df.empty:
        return
    out_dir = fig_dir / "subject_angle_correlations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for keys, sub in plot_df.groupby(["contrast", "role", "subid", "hemisphere"], sort=False):
        contrast, role, subid, hemi = keys
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        axes[0].scatter(sub["emb_angle"], sub["topmode_angle"], s=4, alpha=0.22, color=ROLE_PALETTE[role])
        axes[0].set_title("Circular")
        axes[0].set_xlabel("Embedding-gradient angle")
        axes[0].set_ylabel("Topmode angle")
        axes[1].scatter(sub["emb_angle_axial"], sub["topmode_angle_axial"], s=4, alpha=0.22, color=ROLE_PALETTE[role])
        axes[1].set_title("Axial doubled angles")
        axes[1].set_xlabel("2 x embedding angle")
        axes[1].set_ylabel("2 x topmode angle")
        for ax in axes:
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"{contrast} {role} S{subid} {hemi}")
        fig.tight_layout()
        fig.savefig(out_dir / f"angle_corr_{contrast}_{role}_S{subid}_{hemi}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_angle_diff_polar_plots(plot_df: pd.DataFrame, fig_dir: Path, bins: int = 72) -> None:
    if plot_df.empty:
        return
    out_dir = fig_dir / "angle_diff_polar"
    out_dir.mkdir(parents=True, exist_ok=True)
    for keys, sub in plot_df.groupby(["contrast", "role", "hemisphere"], sort=False):
        contrast, role, hemi = keys
        vals = pd.to_numeric(sub["angle_diff"], errors="coerce").dropna().to_numpy(dtype=float)
        axial_vals = pd.to_numeric(sub["angle_diff_axial"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0 and axial_vals.size == 0:
            continue
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection="polar")
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        if vals.size:
            ax1.hist(vals, bins=bins, color=ROLE_PALETTE[role], alpha=0.85)
        if axial_vals.size:
            ax2.hist(axial_vals, bins=bins, color=ROLE_PALETTE[role], alpha=0.85)
        ax1.set_title("Angle difference")
        ax2.set_title("Axial angle difference")
        fig.suptitle(f"{contrast} {role} {hemi}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(out_dir / f"angle_diff_polar_{contrast}_{role}_{hemi}.png", dpi=260, bbox_inches="tight")
        plt.close(fig)


def save_group_plots(subject_df: pd.DataFrame, delta_df: pd.DataFrame, paired_stats: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for contrast, cdf in subject_df.groupby("contrast", sort=False):
        contrast_dir = fig_dir / str(contrast)
        contrast_dir.mkdir(parents=True, exist_ok=True)
        for metric, label in ALL_METRICS.items():
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            sns.violinplot(data=cdf, x="hemisphere", y=metric, hue="role", palette=ROLE_PALETTE, cut=0, ax=ax)
            sns.stripplot(data=cdf, x="hemisphere", y=metric, hue="role", dodge=True, color="black", alpha=0.45, size=3, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], frameon=False, title="")
            ax.axhline(0.0, color="black", linewidth=0.9)
            ax.set_ylabel("Fisher z" if metric in RAW_R_COLUMNS else "Value")
            ax.set_title(f"{contrast} | {ALL_METRICS[metric]}")
            fig.tight_layout()
            fig.savefig(contrast_dir / f"group_distribution_{metric}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            for hemi, hdf in cdf.groupby("hemisphere", sort=False):
                pivot = hdf.pivot_table(index="subid", columns="role", values=metric, aggfunc="mean")
                if not {"PCB", "Drug"}.issubset(pivot.columns):
                    continue
                pivot = pivot[["PCB", "Drug"]].dropna()
                if pivot.empty:
                    continue
                tidy = pivot.reset_index().melt(id_vars="subid", var_name="role", value_name="value")
                fig, ax = plt.subplots(figsize=(5.2, 5.2))
                sns.violinplot(
                    data=tidy,
                    x="role",
                    y="value",
                    order=["PCB", "Drug"],
                    palette=ROLE_PALETTE,
                    cut=0,
                    ax=ax,
                )
                for subid, row in pivot.iterrows():
                    ax.plot([0, 1], [row["PCB"], row["Drug"]], color="#555555", alpha=0.45, linewidth=0.9)
                    ax.scatter(
                        [0, 1],
                        [row["PCB"], row["Drug"]],
                        color=[ROLE_PALETTE["PCB"], ROLE_PALETTE["Drug"]],
                        edgecolor="white",
                        linewidth=0.5,
                        s=30,
                        zorder=3,
                    )
                stat = paired_stats[
                    (paired_stats["drug_contrast"] == contrast)
                    & (paired_stats["hemisphere"] == hemi)
                    & (paired_stats["metric"] == metric)
                ]
                p_text = ""
                if not stat.empty and np.isfinite(float(stat.iloc[0]["p"])):
                    p_text = f"\np={float(stat.iloc[0]['p']):.3g}, dz={float(stat.iloc[0]['cohen_dz']):.3g}"
                ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
                ax.set_xlabel("")
                ax.set_ylabel("Fisher z" if metric in RAW_R_COLUMNS else "Value")
                ax.set_title(f"{contrast} {hemi} | {ALL_METRICS[metric]}{p_text}")
                fig.tight_layout()
                fig.savefig(contrast_dir / f"paired_lines_{metric}_{hemi}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

            if delta_df.empty or "metric" not in delta_df.columns:
                continue
            sub = delta_df[(delta_df["contrast"] == contrast) & (delta_df["metric"] == metric)].dropna(
                subset=["delta_drug_minus_pcb"]
            )
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.0, 5.0))
            sns.violinplot(data=sub, x="hemisphere", y="delta_drug_minus_pcb", color="#8f8f8f", cut=0, ax=ax)
            sns.stripplot(data=sub, x="hemisphere", y="delta_drug_minus_pcb", color="black", alpha=0.55, size=3, ax=ax)
            ax.axhline(0.0, color="black", linewidth=0.9)
            for x, hemi in enumerate(sorted(sub["hemisphere"].unique())):
                stat = paired_stats[
                    (paired_stats["drug_contrast"] == contrast)
                    & (paired_stats["hemisphere"] == hemi)
                    & (paired_stats["metric"] == metric)
                ]
                if not stat.empty and np.isfinite(float(stat.iloc[0]["p"])):
                    ax.text(x, ax.get_ylim()[1], f"p={float(stat.iloc[0]['p']):.3g}", ha="center", va="top", fontsize=10)
            ax.set_ylabel("Drug - PCB Fisher z" if metric in RAW_R_COLUMNS else "Drug - PCB")
            ax.set_title(f"{contrast} paired delta | {ALL_METRICS[metric]}")
            fig.tight_layout()
            fig.savefig(contrast_dir / f"paired_delta_{metric}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    rng = np.random.default_rng(args.random_seed)

    entries = discover_detect_entries(args)
    refs = load_condition_refs(args)
    fig_dir = args.out_dir / "figures"
    if not args.no_plots:
        save_reference_quiver_plots(refs, fig_dir, args.quiver_step)

    rows = []
    plot_rows = []
    failures = []
    for row in entries.itertuples(index=False):
        try:
            print(f"Processing {row.role} S{row.subid} {row.hemisphere}: {Path(row.detect_dir).name}")
            subject_row, plot_df = summarize_one_bundle(
                pd.Series(row._asdict()),
                refs=refs,
                field_method=args.field_method,
                svd_mode=args.svd_mode,
                plot_sample_points=args.plot_sample_points,
                rng=rng,
                topmode_fig_dir=None if args.no_plots else (fig_dir / "topmode_quiver"),
                quiver_step=args.quiver_step,
                overlay_fig_dir=None if args.no_plots else (fig_dir / "overlay_quiver"),
            )
            rows.append(subject_row)
            if not plot_df.empty:
                plot_rows.append(plot_df)
        except Exception as exc:
            failures.append(
                {
                    "contrast": row.contrast,
                    "condition": row.condition,
                    "role": row.role,
                    "subid": row.subid,
                    "hemisphere": row.hemisphere,
                    "detect_dir": str(row.detect_dir),
                    "error": str(exc),
                }
            )
            print(f"Failed {row.role} S{row.subid} {row.hemisphere}: {exc}", file=sys.stderr)

    if not rows:
        raise RuntimeError("No whole-topmode correlation records were generated.")

    subject_df = pd.DataFrame(rows).sort_values(["contrast", "role", "subid", "hemisphere"]).reset_index(drop=True)
    delta_df = paired_deltas(subject_df)
    group_summary = group_correlation_summary(subject_df)
    paired_summary = paired_ttest_summary(delta_df)

    subject_df.to_csv(args.out_dir / "subject_correlations.csv", index=False)
    delta_df.to_csv(args.out_dir / "paired_deltas.csv", index=False)
    group_summary.to_csv(args.out_dir / "group_correlation_summary.csv", index=False)
    paired_summary.to_csv(args.out_dir / "paired_ttests.csv", index=False)
    if plot_rows:
        pd.concat(plot_rows, ignore_index=True).to_csv(args.out_dir / "subject_plot_samples.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.out_dir / "failures.csv", index=False)

    if not args.no_plots:
        if plot_rows:
            plot_df_all = pd.concat(plot_rows, ignore_index=True)
            save_subject_correlation_plots(plot_df_all, fig_dir)
            save_angle_diff_polar_plots(plot_df_all, fig_dir)
        save_group_plots(subject_df, delta_df, paired_summary, fig_dir)

    contrast_specs = parse_contrasts(args)
    metadata = {
        "detect_root": str(args.detect_root),
        "drug_condition": args.drug_condition,
        "pcb_condition": args.pcb_condition,
        "contrasts": contrast_specs,
        "emb_root": str(args.emb_root),
        "refs": {
            item["label"]: {
                "drug": {hemi: str(condition_ref_path(args.emb_root, item["drug_condition"], hemi)) for hemi in ["left", "right"]},
                "pcb": {hemi: str(condition_ref_path(args.emb_root, item["pcb_condition"], hemi)) for hemi in ["left", "right"]},
            }
            for item in contrast_specs
        },
        "field_method": args.field_method,
        "svd_mode": args.svd_mode,
        "spacing": float(args.spacing),
        "edge_margin": int(args.edge_margin),
        "plot_sample_points": int(args.plot_sample_points),
        "quiver_step": int(args.quiver_step),
        "random_seed": int(args.random_seed),
        "n_entries": int(len(entries)),
        "n_failures": int(len(failures)),
        "metrics": ALL_METRICS,
        "correlation": "Jammalamadaka-Sengupta circular-circular correlation between embedding-gradient angle and topmode angle.",
        "axial_correlation": "The same correlation after doubling both angles, so parallel and anti-parallel axes are treated equivalently.",
        "cos2_alignment": "cos(2theta) is computed pointwise from the topmode and condition-matched embedding-gradient angle difference, then averaged per subject.",
        "group_inference": "Group r is tanh(mean Fisher z). t-tests are performed on subject-level Fisher z values.",
        "pixel_p_note": "Per-subject pixel p-values are approximate and descriptive because grid points are spatially autocorrelated.",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote whole-topmode embedding correlations to: {args.out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
