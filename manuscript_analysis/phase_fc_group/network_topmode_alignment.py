#!/usr/bin/env python
"""
Top-mode alignment analysis against a reference/base field.

Workflow
1. Discover paired phase-FC + detection bundles.
2. Build a vector-field cube from each phase cube.
3. Extract the top spatial mode for the whole brain and for each network.
4. Compare each top mode with the reference/base field via angle difference.
5. Run paired Drug vs PCB statistics and save plots.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


NETWORK_ORDER_7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_ORDER_17 = [
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicB",
    "LimbicA",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
]
CONDITION_RE = re.compile(r"^(?P<condition>[A-Za-z]+_[A-Za-z]+)_S(?P<subid>\d+)")
SUFFIX_RE = re.compile(r"(?P<code>[A-Za-z]+)(?P<subid>\d+)(?P<hemi>[LR])$")
DEFAULT_ALIGNMENT_METADATA = Path("analysis_outputs/phase_gradient_alignment_dmt/run_metadata.json")
ROLE_PALETTE = {"PCB": "#4575b4", "Drug": "#d73027"}
METRIC_LABELS = {
    "mean_abs_angle_diff_rad": "Mean |angle diff| (rad)",
    "mean_signed_angle_diff_rad": "Mean signed angle diff (rad)",
    "mean_cos_alignment": "Mean cos(theta)",
    "mean_cos2_alignment": "Mean cos(2theta)",
    "top1_energy": "Top-1 energy",
}


@dataclass(frozen=True)
class BundleEntry:
    condition: str
    role: str
    subid: str
    hemisphere: str
    fc_dir: Path
    detect_dir: Path
    grid_path: Path
    parcel_meta_path: Path
    phase_cube: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whole-brain and network-wise top-mode alignment against a reference field."
    )
    parser.add_argument(
        "--phase-fc-root",
        default=Path("analysis_outputs/phase_fc_recon_7networks"),
        type=Path,
    )
    parser.add_argument(
        "--detect-root",
        default=Path("detect_results/DMT"),
        type=Path,
    )
    parser.add_argument("--drug-condition", default="DMT_DMT")
    parser.add_argument("--pcb-condition", default="DMT_PCB")
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/network_topmode_alignment"),
        type=Path,
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--ref-left",
        default=None,
        type=Path,
        help="Reference/base field map for left hemisphere.",
    )
    parser.add_argument(
        "--ref-right",
        default=None,
        type=Path,
        help="Reference/base field map for right hemisphere.",
    )
    parser.add_argument(
        "--alignment-metadata",
        default=DEFAULT_ALIGNMENT_METADATA,
        type=Path,
    )
    parser.add_argument(
        "--field-method",
        choices=["phase_gradient", "optical_flow"],
        default="phase_gradient",
    )
    parser.add_argument(
        "--svd-mode",
        choices=["real_svd", "complex_svd"],
        default="real_svd",
    )
    parser.add_argument("--spacing", default=1.0, type=float)
    parser.add_argument("--edge-margin", default=2, type=int)
    parser.add_argument("--quiver-step", default=6, type=int)
    parser.add_argument("--max-bundles", default=None, type=int)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def canonical_condition(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def infer_network_order(networks: Iterable[str]) -> list[str]:
    values = [str(network) for network in networks if pd.notna(network)]
    available = set(values)
    for known in (NETWORK_ORDER_7, NETWORK_ORDER_17):
        ordered = [network for network in known if network in available]
        if ordered:
            extras = [network for network in values if network not in set(known)]
            return ordered + list(dict.fromkeys(extras))
    return list(dict.fromkeys(values))


def parse_fc_dir_name(name: str, drug_condition: str, pcb_condition: str) -> tuple[str, str, str, str] | None:
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


def load_default_refs(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    ref_left = args.ref_left
    ref_right = args.ref_right
    if (ref_left is None or ref_right is None) and args.alignment_metadata.exists():
        meta = json.loads(args.alignment_metadata.read_text(encoding="utf-8"))
        ref_left = ref_left or Path(meta.get("ref_left", ""))
        ref_right = ref_right or Path(meta.get("ref_right", ""))
    if ref_left is not None and not ref_left.exists():
        ref_left = None
    if ref_right is not None and not ref_right.exists():
        ref_right = None
    return ref_left, ref_right


def discover_entries(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    hemispheres = {"left", "right"} if args.hemisphere == "both" else {args.hemisphere}
    for fc_dir in sorted(p for p in args.phase_fc_root.glob("*") if p.is_dir() and p.name != "atlas_metadata"):
        parsed = parse_fc_dir_name(fc_dir.name, args.drug_condition, args.pcb_condition)
        if parsed is None:
            continue
        condition, role, subid, hemi = parsed
        if hemi not in hemispheres:
            continue
        detect_dir = args.detect_root / fc_dir.name
        entry = BundleEntry(
            condition=condition,
            role=role,
            subid=subid,
            hemisphere=hemi,
            fc_dir=fc_dir,
            detect_dir=detect_dir,
            grid_path=fc_dir / "grid_labels.npy",
            parcel_meta_path=fc_dir / "parcel_metadata.csv",
            phase_cube=detect_dir / "phase_cube.npy",
        )
        if all(path.exists() for path in [entry.grid_path, entry.parcel_meta_path, entry.phase_cube]):
            rows.append(entry.__dict__)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matching phase-FC/detection bundles found.")
    df = df.sort_values(["role", "subid", "hemisphere"]).reset_index(drop=True)
    if args.max_bundles is not None:
        df = df.head(args.max_bundles).copy()
    return df


def load_network_grid(grid_path: Path, parcel_meta_path: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    grid_labels = np.load(grid_path)
    parcels = pd.read_csv(parcel_meta_path)
    if not {"parcel_id", "network"}.issubset(parcels.columns):
        raise ValueError(f"parcel metadata missing parcel_id/network columns: {parcel_meta_path}")
    network_order = infer_network_order(parcels["network"])
    network_to_idx = {network: idx for idx, network in enumerate(network_order)}
    parcel_to_network = dict(zip(parcels["parcel_id"].astype(int), parcels["network"].astype(str)))
    network_grid = np.full(grid_labels.shape, -1, dtype=np.int16)
    finite = np.isfinite(grid_labels)
    for parcel_id in np.unique(grid_labels[finite]).astype(int):
        network = parcel_to_network.get(int(parcel_id))
        if network is None:
            continue
        network_grid[grid_labels == parcel_id] = network_to_idx[network]
    areas = []
    for network, idx in network_to_idx.items():
        areas.append({"network": network, "network_index": idx, "network_area_pixels": int(np.sum(network_grid == idx))})
    return network_grid, pd.DataFrame(areas), network_order


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
    raise ValueError(f"Cannot align {name} shape {arr.shape} to {target_shape}")


def load_reference(path: Path, spacing: float, edge_margin: int) -> dict[str, np.ndarray]:
    gmap = np.load(path).astype(np.float64)
    gy, gx = np.gradient(gmap, spacing)
    mag = np.hypot(gx, gy)
    valid = np.isfinite(gmap) & np.isfinite(gx) & np.isfinite(gy) & (mag > 0)
    valid = apply_border_mask(valid, edge_margin)
    return {"gx": gx, "gy": gy, "mag": mag, "valid": valid}


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
        dst[axis] = slice(1, n - 1)
        src_a[axis] = slice(2, n)
        src_b[axis] = slice(0, n - 2)
        deriv[tuple(dst)] = 0.5 * _phase_diff(arr[tuple(src_a)], arr[tuple(src_b)])

    finite = np.isfinite(arr)
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

    flow_alpha = 0.1
    flow_beta = 10.0
    flow_max_iter = 200
    flow_tol = 1.0e-4

    u = np.zeros_like(phase_a, dtype=np.float64) if u0 is None else np.where(np.isfinite(u0), u0, 0.0)
    v = np.zeros_like(phase_a, dtype=np.float64) if v0 is None else np.where(np.isfinite(v0), v0, 0.0)
    alpha2 = flow_alpha * flow_alpha
    beta2 = flow_beta * flow_beta

    for _ in range(flow_max_iter):
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
        if np.isfinite(step) and step < flow_tol:
            break
    return np.where(valid, u, np.nan), np.where(valid, v, np.nan)


def build_field_cube(phase_cube: np.ndarray, field_method: str) -> np.ndarray:
    if field_method == "phase_gradient":
        gx = _phase_derivative_axis(phase_cube, axis=1)
        gy = _phase_derivative_axis(phase_cube, axis=0)
        return gx + 1j * gy
    if field_method == "optical_flow":
        rows, cols, frames = phase_cube.shape
        u_cube = np.full((rows, cols, frames - 1), np.nan, dtype=np.float64)
        v_cube = np.full((rows, cols, frames - 1), np.nan, dtype=np.float64)
        u_prev = None
        v_prev = None
        for t in range(frames - 1):
            u_frame, v_frame = _compute_pairwise_velocity(phase_cube[:, :, t], phase_cube[:, :, t + 1], u_prev, v_prev)
            u_cube[:, :, t] = u_frame
            v_cube[:, :, t] = v_frame
            u_prev = u_frame
            v_prev = v_frame
        return u_cube + 1j * v_cube
    raise ValueError(f"Unsupported field method: {field_method}")


def extract_top_mode(complex_field_cube: np.ndarray, support_mask: np.ndarray, svd_mode: str) -> tuple[np.ndarray, dict[str, float]]:
    rows, cols, frames = complex_field_cube.shape
    flat = complex_field_cube.reshape(rows * cols, frames)
    support_flat = support_mask.reshape(-1)
    valid_flat = support_flat & np.isfinite(flat).all(axis=1)
    if not np.any(valid_flat):
        mode_map = np.full((rows, cols), np.nan + 1j * np.nan, dtype=np.complex128)
        return mode_map, {"top1_energy": math.nan, "n_mode_voxels": 0, "field_frames": float(frames)}

    data_complex = flat[valid_flat].T
    if data_complex.shape[0] < 2 or data_complex.shape[1] < 1:
        mode_map = np.full((rows, cols), np.nan + 1j * np.nan, dtype=np.complex128)
        return mode_map, {"top1_energy": math.nan, "n_mode_voxels": float(np.sum(valid_flat)), "field_frames": float(data_complex.shape[0])}

    if svd_mode == "complex_svd":
        _, s, vh = np.linalg.svd(data_complex, full_matrices=False)
        spatial_mode = vh[0]
    elif svd_mode == "real_svd":
        data_real = np.concatenate([np.real(data_complex), np.imag(data_complex)], axis=1)
        _, s, vh = np.linalg.svd(data_real, full_matrices=False)
        n_vox = data_complex.shape[1]
        spatial_mode = vh[0, :n_vox] - 1j * vh[0, n_vox:]
    else:
        raise ValueError(f"Unsupported svd mode: {svd_mode}")

    energy = s * s
    total = float(np.sum(energy))
    top1 = float((energy[0] / total)) if total > 0 else math.nan
    mode_flat = np.full(rows * cols, np.nan + 1j * np.nan, dtype=np.complex128)
    mode_flat[valid_flat] = spatial_mode
    mode_map = mode_flat.reshape(rows, cols)
    return mode_map, {"top1_energy": top1, "n_mode_voxels": float(np.sum(valid_flat)), "field_frames": float(data_complex.shape[0])}


def align_mode_to_reference(
    mode_map: np.ndarray,
    ref_gx: np.ndarray,
    ref_gy: np.ndarray,
    ref_valid: np.ndarray,
    svd_mode: str,
) -> np.ndarray:
    out = np.array(mode_map, copy=True)
    ref_complex = ref_gx + 1j * ref_gy
    valid = ref_valid & np.isfinite(out) & np.isfinite(ref_complex)
    if not np.any(valid):
        return out
    if svd_mode == "complex_svd":
        denom = np.vdot(ref_complex[valid], out[valid])
        if denom != 0:
            out *= np.exp(-1j * np.angle(denom))
    else:
        score = np.real(np.vdot(ref_complex[valid], out[valid]))
        if np.isfinite(score) and score < 0:
            out *= -1.0
    return out


def summarize_mode_alignment(
    mode_map: np.ndarray,
    ref_gx: np.ndarray,
    ref_gy: np.ndarray,
    ref_valid: np.ndarray,
    extra: dict[str, float],
) -> dict[str, float]:
    vx = np.real(mode_map)
    vy = np.imag(mode_map)
    ref_mag = np.hypot(ref_gx, ref_gy)
    tgt_mag = np.hypot(vx, vy)
    valid = ref_valid & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(ref_mag) & (ref_mag > 0) & (tgt_mag > 0)
    if not np.any(valid):
        return {
            "mean_abs_angle_diff_rad": math.nan,
            "mean_signed_angle_diff_rad": math.nan,
            "mean_cos_alignment": math.nan,
            "mean_cos2_alignment": math.nan,
            "alignment_samples": 0,
            **extra,
        }

    cos_theta = np.divide(
        ref_gx * vx + ref_gy * vy,
        ref_mag * tgt_mag,
        out=np.full(ref_gx.shape, np.nan, dtype=np.float64),
        where=valid,
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    cross_z = ref_gx * vy - ref_gy * vx
    angle = np.where(cross_z < 0, -angle, angle)
    return {
        "mean_abs_angle_diff_rad": float(np.nanmean(np.abs(angle[valid]))),
        "mean_signed_angle_diff_rad": float(np.nanmean(angle[valid])),
        "mean_cos_alignment": float(np.nanmean(cos_theta[valid])),
        "mean_cos2_alignment": float(np.nanmean(np.cos(2.0 * angle[valid]))),
        "alignment_samples": int(np.sum(valid)),
        **extra,
    }


def paired_deltas(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [m for m in METRIC_LABELS if m in set(subject_df.columns)]
    group_cols = ["subid", "hemisphere", "network", "network_index"]
    for keys, sub in subject_df.groupby(group_cols, sort=False):
        hemi_sub = sub.pivot(index="subid", columns="role", values=metrics)
        if "Drug" not in hemi_sub.columns.get_level_values(1) or "PCB" not in hemi_sub.columns.get_level_values(1):
            continue
        for metric in metrics:
            try:
                drug = float(hemi_sub[(metric, "Drug")].iloc[0])
                pcb = float(hemi_sub[(metric, "PCB")].iloc[0])
            except KeyError:
                continue
            rows.append(
                {
                    "subid": keys[0],
                    "hemisphere": keys[1],
                    "network": keys[2],
                    "network_index": int(keys[3]),
                    "metric": metric,
                    "drug_value": drug,
                    "pcb_value": pcb,
                    "delta_drug_minus_pcb": drug - pcb if np.isfinite(drug) and np.isfinite(pcb) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_delta_summary(delta_df: pd.DataFrame, min_pairs: int = 3) -> pd.DataFrame:
    rows = []
    if delta_df.empty:
        return pd.DataFrame(rows)
    for keys, sub in delta_df.groupby(["hemisphere", "network", "network_index", "metric"], sort=False):
        values = pd.to_numeric(sub["delta_drug_minus_pcb"], errors="coerce").dropna().to_numpy(dtype=float)
        hemi, network, network_index, metric = keys
        if values.size == 0:
            mean = std = sem = t_val = p_val = math.nan
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
            sem = float(std / math.sqrt(values.size)) if values.size > 1 else math.nan
            if values.size >= min_pairs:
                t_val, p_val = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
                t_val = float(t_val)
                p_val = float(p_val)
            else:
                t_val = math.nan
                p_val = math.nan
        rows.append(
            {
                "hemisphere": hemi,
                "network": network,
                "network_index": int(network_index),
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "n_paired": int(values.size),
                "mean_delta_drug_minus_pcb": mean,
                "std_delta": std,
                "sem_delta": sem,
                "t_1sample_delta": t_val,
                "p_1sample_delta": p_val,
                "cohen_dz": mean / std if np.isfinite(mean) and np.isfinite(std) and std > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def save_group_mode_npz(mode_records: list[dict], out_path: Path) -> None:
    arrays = {}
    keys = sorted(
        {
            (str(rec["hemisphere"]), str(rec["role"]), str(rec["network"]))
            for rec in mode_records
        }
    )
    for hemi, role, network in keys:
        matches = [
            np.asarray(rec["mode_map"], dtype=np.complex64)
            for rec in mode_records
            if str(rec["hemisphere"]) == hemi and str(rec["role"]) == role and str(rec["network"]) == network
        ]
        if matches:
            arrays[f"{hemi}__{role}__{network}"] = np.nanmean(np.stack(matches, axis=0), axis=0)
    np.savez_compressed(out_path, **arrays)


def plot_quiver(ax: plt.Axes, field: np.ndarray, title: str, step: int, color: str) -> None:
    vx = np.real(field)
    vy = np.imag(field)
    mag = np.hypot(vx, vy)
    valid = np.isfinite(vx) & np.isfinite(vy) & (mag > 0)
    if np.any(valid):
        vx_n = np.divide(vx, mag, out=np.zeros_like(vx), where=valid)
        vy_n = np.divide(vy, mag, out=np.zeros_like(vy), where=valid)
        yy, xx = np.mgrid[0 : field.shape[0], 0 : field.shape[1]]
        sample = valid.copy()
        sample[::step, ::step] &= True
        sample &= False
        sample[::step, ::step] = valid[::step, ::step]
        ax.quiver(xx[sample], yy[sample], vx_n[sample], vy_n[sample], color=color, angles="xy", scale_units="xy", scale=0.25)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def save_whole_field_panels(mode_records: list[dict], refs: dict[str, dict[str, np.ndarray]], fig_dir: Path, quiver_step: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for hemi in sorted(refs):
        ref_field = refs[hemi]["gx"] + 1j * refs[hemi]["gy"]
        mean_fields = {}
        for role in ["PCB", "Drug"]:
            matches = [rec["mode_map"] for rec in mode_records if rec["hemisphere"] == hemi and rec["role"] == role and rec["network"] == "whole"]
            if matches:
                mean_fields[role] = np.nanmean(np.stack(matches, axis=0), axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        plot_quiver(axes[0], ref_field, f"Base field ({hemi})", quiver_step, "#222222")
        for ax, role in zip(axes[1:], ["PCB", "Drug"]):
            if role in mean_fields:
                plot_quiver(ax, mean_fields[role], f"{role} mean top mode ({hemi})", quiver_step, ROLE_PALETTE[role])
            else:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / f"whole_field_panel_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_network_field_panels(
    mode_records: list[dict],
    network_order: list[str],
    fig_dir: Path,
    quiver_step: int,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for hemi in sorted({rec["hemisphere"] for rec in mode_records}):
        for role in ["PCB", "Drug"]:
            ncols = 4
            nrows = int(math.ceil(len(network_order) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows), squeeze=False)
            for ax, network in zip(axes.flat, network_order):
                matches = [rec["mode_map"] for rec in mode_records if rec["hemisphere"] == hemi and rec["role"] == role and rec["network"] == network]
                if matches:
                    mean_field = np.nanmean(np.stack(matches, axis=0), axis=0)
                    plot_quiver(ax, mean_field, network, quiver_step, ROLE_PALETTE[role])
                else:
                    ax.axis("off")
            for ax in axes.flat[len(network_order) :]:
                ax.axis("off")
            fig.suptitle(f"{role} mean network top modes ({hemi})", y=1.02)
            fig.tight_layout()
            fig.savefig(fig_dir / f"network_field_panel_{role}_{hemi}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def save_metric_plots(subject_df: pd.DataFrame, delta_summary: pd.DataFrame, fig_dir: Path, network_order: list[str]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in METRIC_LABELS if m in set(subject_df.columns)]
    for hemi in sorted(subject_df["hemisphere"].dropna().unique()):
        whole_sub = subject_df[(subject_df["hemisphere"] == hemi) & (subject_df["network"] == "whole")]
        if not whole_sub.empty:
            for metric in metrics:
                pivot = whole_sub.pivot(index="subid", columns="role", values=metric).dropna(subset=["PCB", "Drug"], how="any")
                if pivot.empty:
                    continue
                tidy = pivot.reset_index().melt(id_vars="subid", var_name="role", value_name="value")
                fig, ax = plt.subplots(figsize=(5.2, 5.2))
                sns.violinplot(data=tidy, x="role", y="value", order=["PCB", "Drug"], palette=ROLE_PALETTE, cut=0, ax=ax)
                for _, row in pivot.iterrows():
                    ax.plot([0, 1], [row["PCB"], row["Drug"]], color="gray", alpha=0.45, linewidth=0.9)
                stat = stats.ttest_rel(pivot["Drug"], pivot["PCB"], nan_policy="omit")
                ax.set_title(f"{METRIC_LABELS.get(metric, metric)} | whole | {hemi}\np={stat.pvalue:.3g}")
                ax.set_xlabel("")
                fig.tight_layout()
                fig.savefig(fig_dir / f"whole_paired_{metric}_{hemi}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

        net_sub = delta_summary[(delta_summary["hemisphere"] == hemi) & (delta_summary["network"] != "whole")]
        if net_sub.empty:
            continue
        for metric in metrics:
            mdf = net_sub[net_sub["metric"] == metric].set_index("network").reindex(network_order).reset_index()
            if mdf.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(network_order)), 5.0))
            ax.bar(mdf["network"], mdf["cohen_dz"], color="#4c78a8")
            ax.axhline(0.0, color="black", linewidth=0.9)
            ax.set_ylabel("Cohen dz")
            ax.set_title(f"{METRIC_LABELS.get(metric, metric)} | network deltas | {hemi}")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(fig_dir / f"network_effect_{metric}_{hemi}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        heat_rows = []
        for metric in metrics:
            mdf = net_sub[net_sub["metric"] == metric].set_index("network").reindex(network_order)
            heat_rows.append(mdf["mean_delta_drug_minus_pcb"].rename(METRIC_LABELS.get(metric, metric)))
        if heat_rows:
            mat = pd.DataFrame(heat_rows)
            vals = mat.to_numpy(dtype=float)
            vmax = np.nanpercentile(np.abs(vals), 95) if np.isfinite(vals).any() else 1.0
            vmax = max(float(vmax), 1e-6)
            fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(network_order)), 4.8))
            sns.heatmap(mat, cmap="coolwarm", center=0.0, vmin=-vmax, vmax=vmax, ax=ax, cbar_kws={"label": "Drug - PCB"})
            ax.set_title(f"Network paired deltas ({hemi})")
            fig.tight_layout()
            fig.savefig(fig_dir / f"network_delta_heatmap_{hemi}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def summarize_bundle(
    entry: pd.Series,
    refs: dict[str, dict[str, np.ndarray]],
    field_method: str,
    svd_mode: str,
) -> tuple[pd.DataFrame, list[dict]]:
    network_grid, network_areas, network_order = load_network_grid(Path(entry.grid_path), Path(entry.parcel_meta_path))
    phase_cube = np.asarray(np.load(entry.phase_cube, mmap_mode="r"), dtype=np.float64)
    field_cube = build_field_cube(phase_cube, field_method)
    h, w, _ = field_cube.shape
    ref = refs[entry.hemisphere]
    ref_gx = align_2d(ref["gx"], (h, w), "ref_gx")
    ref_gy = align_2d(ref["gy"], (h, w), "ref_gy")
    ref_valid = align_2d(ref["valid"], (h, w), "ref_valid").astype(bool)

    rows = []
    mode_records: list[dict] = []
    whole_support = np.isfinite(field_cube).all(axis=2)
    whole_mode, whole_extra = extract_top_mode(field_cube, whole_support, svd_mode)
    whole_mode = align_mode_to_reference(whole_mode, ref_gx, ref_gy, ref_valid & whole_support, svd_mode)
    rows.append(
        {
            "condition": entry.condition,
            "role": entry.role,
            "subid": str(entry.subid),
            "hemisphere": entry.hemisphere,
            "network": "whole",
            "network_index": -1,
            "network_area_pixels": int(np.sum(whole_support)),
            **summarize_mode_alignment(whole_mode, ref_gx, ref_gy, ref_valid & whole_support, whole_extra),
        }
    )
    mode_records.append({"role": entry.role, "subid": str(entry.subid), "hemisphere": entry.hemisphere, "network": "whole", "mode_map": whole_mode})

    for area_row in network_areas.itertuples(index=False):
        idx = int(area_row.network_index)
        support = (network_grid == idx) & np.isfinite(field_cube).all(axis=2)
        mode_map, extra = extract_top_mode(field_cube, support, svd_mode)
        mode_map = align_mode_to_reference(mode_map, ref_gx, ref_gy, ref_valid & support, svd_mode)
        rows.append(
            {
                "condition": entry.condition,
                "role": entry.role,
                "subid": str(entry.subid),
                "hemisphere": entry.hemisphere,
                "network": area_row.network,
                "network_index": idx,
                "network_area_pixels": int(area_row.network_area_pixels),
                **summarize_mode_alignment(mode_map, ref_gx, ref_gy, ref_valid & support, extra),
            }
        )
        mode_records.append({"role": entry.role, "subid": str(entry.subid), "hemisphere": entry.hemisphere, "network": area_row.network, "mode_map": mode_map})
    return pd.DataFrame(rows), mode_records


def wide_to_long(subject_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["condition", "role", "subid", "hemisphere", "network", "network_index", "network_area_pixels"]
    value_cols = [c for c in METRIC_LABELS if c in subject_df.columns]
    return subject_df.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    entries = discover_entries(args)
    ref_left, ref_right = load_default_refs(args)
    if ref_left is None or ref_right is None:
        raise RuntimeError("Both ref-left and ref-right must resolve to existing base-field files.")
    refs = {
        "left": load_reference(ref_left, spacing=args.spacing, edge_margin=args.edge_margin),
        "right": load_reference(ref_right, spacing=args.spacing, edge_margin=args.edge_margin),
    }

    records = []
    mode_records: list[dict] = []
    failures = []
    network_order_global: list[str] = []
    for row in entries.itertuples(index=False):
        try:
            print(f"Processing {row.role} S{row.subid} {row.hemisphere}: {Path(row.fc_dir).name}")
            bundle_df, bundle_modes = summarize_bundle(pd.Series(row._asdict()), refs, args.field_method, args.svd_mode)
            records.append(bundle_df)
            mode_records.extend(bundle_modes)
            if not network_order_global:
                _, _, network_order_global = load_network_grid(Path(row.grid_path), Path(row.parcel_meta_path))
        except Exception as exc:
            failures.append(
                {
                    "condition": row.condition,
                    "role": row.role,
                    "subid": row.subid,
                    "hemisphere": row.hemisphere,
                    "fc_dir": str(row.fc_dir),
                    "detect_dir": str(row.detect_dir),
                    "error": str(exc),
                }
            )
            print(f"Failed {row.role} S{row.subid} {row.hemisphere}: {exc}", file=sys.stderr)

    if not records:
        raise RuntimeError("No subject top-mode records were generated.")

    subject_df = pd.concat(records, ignore_index=True)
    network_order = ["whole"] + network_order_global
    subject_df["network"] = pd.Categorical(subject_df["network"], categories=network_order, ordered=True)
    subject_df = subject_df.sort_values(["role", "subid", "hemisphere", "network"]).reset_index(drop=True)
    long_df = wide_to_long(subject_df)
    delta_df = paired_deltas(subject_df)
    delta_summary = paired_delta_summary(delta_df)

    subject_df.to_csv(args.out_dir / "subject_topmode_metrics_wide.csv", index=False)
    long_df.to_csv(args.out_dir / "subject_topmode_metrics_long.csv", index=False)
    delta_df.to_csv(args.out_dir / "paired_deltas_long.csv", index=False)
    delta_summary.to_csv(args.out_dir / "paired_delta_summary.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.out_dir / "failures.csv", index=False)
    save_group_mode_npz(mode_records, args.out_dir / "group_mean_mode_fields.npz")

    if not args.no_plots:
        fig_dir = args.out_dir / "figures"
        save_whole_field_panels(mode_records, refs, fig_dir / "whole_fields", args.quiver_step)
        save_network_field_panels(mode_records, network_order_global, fig_dir / "network_fields", args.quiver_step)
        save_metric_plots(subject_df, delta_summary, fig_dir / "stats", network_order_global)

    metadata = {
        "phase_fc_root": str(args.phase_fc_root),
        "detect_root": str(args.detect_root),
        "drug_condition": args.drug_condition,
        "pcb_condition": args.pcb_condition,
        "ref_left": str(ref_left),
        "ref_right": str(ref_right),
        "field_method": args.field_method,
        "svd_mode": args.svd_mode,
        "spacing": float(args.spacing),
        "edge_margin": int(args.edge_margin),
        "quiver_step": int(args.quiver_step),
        "n_entries": int(len(entries)),
        "n_failures": int(len(failures)),
        "network_order": network_order,
        "metrics": METRIC_LABELS,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote top-mode alignment outputs to: {args.out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
