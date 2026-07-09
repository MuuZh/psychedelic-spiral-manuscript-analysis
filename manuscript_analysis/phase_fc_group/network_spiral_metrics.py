#!/usr/bin/env python
"""
Network-wise spiral metrics using Schaefer grid labels.

This script combines detection bundles with the Schaefer grid projections
created by scripts/phase_fc_recon_batch.py / phase_fc_single_subject.py. Spiral
frames are assigned to a network by majority footprint overlap, then summarized
at subject x hemisphere x network level. KDE plots use subjects as samples.
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
    from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))
    from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field


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

SPIRAL_METRIC_LABELS = {
    "spiral_count_per_frame": "Pattern count / frame",
    "spiral_count_per_network_px": "Pattern count / network px",
    "mean_spiral_size": "Mean spiral size",
    "mean_network_footprint_px": "Mean footprint in network (px)",
    "mean_network_footprint_fraction": "Mean footprint fraction of network",
    "mean_spiral_power": "Mean spiral power",
    "mean_expansion_radius": "Mean expansion radius",
    "mean_cos2_alignment": "Mean cos(2theta)",
    "weighted_mean_cos2_alignment": "Weighted mean cos(2theta)",
}
ROLE_PALETTE = {"PCB": "#4575b4", "Drug": "#d73027"}


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
    frame_index: Path
    coords: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schaefer-network spiral density/size/power/alignment analysis."
    )
    parser.add_argument(
        "--phase-fc-root",
        default=Path("analysis_outputs/phase_fc_recon_7networks"),
        type=Path,
        help="Root containing per-bundle phase-FC outputs with grid_labels.npy.",
    )
    parser.add_argument(
        "--detect-root",
        default=Path("detect_results/DMT"),
        type=Path,
        help="Root containing detection bundles with frame_index.parquet and coords.feather.",
    )
    parser.add_argument("--drug-condition", default="DMT_DMT")
    parser.add_argument("--pcb-condition", default="DMT_PCB")
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/network_spiral_metrics"),
        type=Path,
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument("--min-overlap", default=0.5, type=float)
    parser.add_argument(
        "--overlap-denominator",
        choices=["labeled", "all"],
        default="labeled",
        help="Denominator for majority overlap. 'labeled' ignores non-atlas pixels in the footprint.",
    )
    parser.add_argument(
        "--ref-left",
        default=None,
        type=Path,
        help="Reference cortical-gradient map for left hemisphere. Defaults to prior alignment metadata if present.",
    )
    parser.add_argument(
        "--ref-right",
        default=None,
        type=Path,
        help="Reference cortical-gradient map for right hemisphere. Defaults to prior alignment metadata if present.",
    )
    parser.add_argument(
        "--alignment-metadata",
        default=DEFAULT_ALIGNMENT_METADATA,
        type=Path,
        help="Metadata JSON to read default ref-left/ref-right from.",
    )
    parser.add_argument("--spacing", default=1.0, type=float)
    parser.add_argument("--edge-margin", default=2, type=int)
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip cos(2theta) metrics.",
    )
    parser.add_argument(
        "--alignment-cache-dir",
        default=None,
        type=Path,
        help="Directory for cached per-bundle network cos(2theta) CSVs. Defaults to <out-dir>/cache/alignment.",
    )
    parser.add_argument(
        "--overwrite-alignment-cache",
        action="store_true",
        help="Recompute cached alignment CSVs.",
    )
    parser.add_argument(
        "--max-bundles",
        default=None,
        type=int,
        help="Debug limit after filtering conditions/hemispheres.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read existing CSV outputs in --out-dir and regenerate figures without recomputing metrics.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Read existing CSV outputs in --out-dir and regenerate group stats/t-tests/RM-ANOVA only.",
    )
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
            frame_index=detect_dir / "frame_index.parquet",
            coords=detect_dir / "coords.feather",
        )
        if all(
            path.exists()
            for path in [entry.grid_path, entry.parcel_meta_path, entry.phase_cube, entry.frame_index, entry.coords]
        ):
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


def read_frame_count(detect_dir: Path, phase_cube: Path) -> int:
    meta_path = detect_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "frame_count" in meta:
                return int(meta["frame_count"])
        except Exception:
            pass
    cube = np.load(phase_cube, mmap_mode="r")
    return int(cube.shape[2])


def assign_spiral_frames(
    frame_index: pd.DataFrame,
    coords: pd.DataFrame,
    network_grid: np.ndarray,
    network_order: list[str],
    min_overlap: float,
    denominator_mode: str,
) -> pd.DataFrame:
    rows = []
    coord_y = coords["y"].to_numpy(dtype=np.int64, copy=False)
    coord_x = coords["x"].to_numpy(dtype=np.int64, copy=False)
    h, w = network_grid.shape

    for row in frame_index.itertuples(index=False):
        start = int(row.coord_start)
        end = int(row.coord_end)
        y = coord_y[start:end]
        x = coord_x[start:end]
        in_bounds = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        if not np.any(in_bounds):
            continue
        y = y[in_bounds]
        x = x[in_bounds]
        net_idx = network_grid[y, x]
        labeled = net_idx >= 0
        if not np.any(labeled):
            continue
        counts = np.bincount(net_idx[labeled], minlength=len(network_order))
        best_idx = int(np.argmax(counts))
        best_count = int(counts[best_idx])
        denom = int(np.sum(labeled)) if denominator_mode == "labeled" else int(len(net_idx))
        if denom <= 0:
            continue
        overlap = best_count / denom
        if overlap < min_overlap:
            continue
        size = float(getattr(row, "instantaneous_size", math.nan))
        power = float(getattr(row, "instantaneous_power", math.nan))
        rows.append(
            {
                "network": network_order[best_idx],
                "pattern_id": int(row.pattern_id),
                "abs_time": int(row.abs_time),
                "overlap_fraction": float(overlap),
                "network_footprint_px": best_count,
                "labeled_footprint_px": int(np.sum(labeled)),
                "instantaneous_size": size,
                "instantaneous_power": power,
                "instantaneous_peak_amp": float(getattr(row, "instantaneous_peak_amp", math.nan)),
                "instantaneous_width": float(getattr(row, "instantaneous_width", math.nan)),
                "expansion_radius": float(getattr(row, "expansion_radius", math.nan)),
                "power_per_spiral_pixel": power / size if np.isfinite(power) and np.isfinite(size) and size > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def assign_spiral_patterns(
    frame_index: pd.DataFrame,
    coords: pd.DataFrame,
    network_grid: np.ndarray,
    network_order: list[str],
    min_overlap: float,
    denominator_mode: str,
) -> pd.DataFrame:
    rows = []
    coord_y = coords["y"].to_numpy(dtype=np.int64, copy=False)
    coord_x = coords["x"].to_numpy(dtype=np.int64, copy=False)
    h, w = network_grid.shape

    for pattern_id, sub in frame_index.groupby("pattern_id", sort=False):
        footprint_parts = []
        total_in_bounds = 0
        for row in sub.itertuples(index=False):
            start = int(row.coord_start)
            end = int(row.coord_end)
            y = coord_y[start:end]
            x = coord_x[start:end]
            in_bounds = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            if not np.any(in_bounds):
                continue
            y = y[in_bounds]
            x = x[in_bounds]
            total_in_bounds += int(len(y))
            footprint_parts.append(y * w + x)
        if not footprint_parts:
            continue
        footprint = np.unique(np.concatenate(footprint_parts))
        y = footprint // w
        x = footprint % w
        net_idx = network_grid[y, x]
        labeled = net_idx >= 0
        if not np.any(labeled):
            continue
        counts = np.bincount(net_idx[labeled], minlength=len(network_order))
        best_idx = int(np.argmax(counts))
        best_count = int(counts[best_idx])
        denom = int(np.sum(labeled)) if denominator_mode == "labeled" else int(len(net_idx))
        if denom <= 0:
            continue
        overlap = best_count / denom
        if overlap < min_overlap:
            continue
        rows.append(
            {
                "network": network_order[best_idx],
                "pattern_id": int(pattern_id),
                "overlap_fraction": float(overlap),
                "network_footprint_px": best_count,
                "labeled_footprint_px": int(np.sum(labeled)),
                "lifetime_footprint_px": int(len(net_idx)),
                "lifetime_coord_observations": total_in_bounds,
                "frame_observations": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


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


def compute_network_alignment(
    phase_cube_path: Path,
    network_grid: np.ndarray,
    network_areas: pd.DataFrame,
    ref: dict[str, np.ndarray],
    spacing: float,
) -> pd.DataFrame:
    phase_cube = np.load(phase_cube_path, mmap_mode="r")
    h, w, _ = phase_cube.shape
    ref_gx = align_2d(ref["gx"], (h, w), "ref_gx")
    ref_gy = align_2d(ref["gy"], (h, w), "ref_gy")
    ref_mag = align_2d(ref["mag"], (h, w), "ref_mag")
    ref_valid = align_2d(ref["valid"], (h, w), "ref_valid").astype(bool)

    grad_x, grad_y = compute_phase_gradient(np.asarray(phase_cube), spacing=spacing, show_progress=False)
    phase_ux, phase_uy, phase_mag = normalize_vector_field(grad_x, grad_y)

    rows = []
    ref_norm = np.hypot(ref_gx, ref_gy)
    for area_row in network_areas.itertuples(index=False):
        idx = int(area_row.network_index)
        net_mask = network_grid == idx
        valid = (
            net_mask[:, :, None]
            & ref_valid[:, :, None]
            & np.isfinite(phase_ux)
            & np.isfinite(phase_uy)
            & np.isfinite(phase_mag)
            & (phase_mag > 0)
            & (ref_norm[:, :, None] > 0)
        )
        if not np.any(valid):
            rows.append(
                {
                    "network": area_row.network,
                    "mean_cos2_alignment": math.nan,
                    "weighted_mean_cos2_alignment": math.nan,
                    "alignment_samples": 0,
                }
            )
            continue
        dot = ref_gx[:, :, None] * phase_ux + ref_gy[:, :, None] * phase_uy
        cos_theta = np.divide(
            dot,
            ref_norm[:, :, None],
            out=np.full(phase_ux.shape, np.nan, dtype=np.float64),
            where=valid,
        )
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        cross_z = ref_gx[:, :, None] * phase_uy - ref_gy[:, :, None] * phase_ux
        angle = np.where(cross_z < 0, -angle, angle)
        cos2 = np.cos(2.0 * angle)
        weights = ref_mag[:, :, None] * phase_mag
        good = valid & np.isfinite(cos2) & np.isfinite(weights)
        denom = float(np.sum(weights[good])) if np.any(good) else 0.0
        rows.append(
            {
                "network": area_row.network,
                "mean_cos2_alignment": float(np.nanmean(cos2[good])) if np.any(good) else math.nan,
                "weighted_mean_cos2_alignment": float(np.sum(weights[good] * cos2[good]) / denom)
                if denom > 0
                else math.nan,
                "alignment_samples": int(np.sum(good)),
            }
        )
    return pd.DataFrame(rows)


def cached_network_alignment(
    entry: pd.Series,
    network_grid: np.ndarray,
    network_areas: pd.DataFrame,
    ref: dict[str, np.ndarray],
    spacing: float,
    cache_dir: Path,
    overwrite: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{Path(entry.fc_dir).name}_network_alignment.csv"
    required = {"network", "mean_cos2_alignment", "weighted_mean_cos2_alignment", "alignment_samples"}
    if cache_path.exists() and not overwrite:
        cached = pd.read_csv(cache_path)
        if required.issubset(cached.columns):
            return cached
    align = compute_network_alignment(Path(entry.phase_cube), network_grid, network_areas, ref=ref, spacing=spacing)
    align.to_csv(cache_path, index=False)
    return align


def summarize_bundle(
    entry: pd.Series,
    min_overlap: float,
    denominator_mode: str,
    ref: dict[str, np.ndarray] | None,
    spacing: float,
    alignment_cache_dir: Path | None = None,
    overwrite_alignment_cache: bool = False,
) -> pd.DataFrame:
    network_grid, network_areas, network_order = load_network_grid(Path(entry.grid_path), Path(entry.parcel_meta_path))
    frame_count = read_frame_count(Path(entry.detect_dir), Path(entry.phase_cube))
    frame_cols = [
        "pattern_id",
        "abs_time",
        "coord_start",
        "coord_end",
        "instantaneous_size",
        "instantaneous_power",
        "instantaneous_peak_amp",
        "instantaneous_width",
        "expansion_radius",
    ]
    frame_index = pd.read_parquet(entry.frame_index, columns=frame_cols)
    coords = pd.read_feather(entry.coords)
    assigned_patterns = assign_spiral_patterns(
        frame_index=frame_index,
        coords=coords,
        network_grid=network_grid,
        network_order=network_order,
        min_overlap=min_overlap,
        denominator_mode=denominator_mode,
    )
    frame_metrics = frame_index[
        [
            "pattern_id",
            "abs_time",
            "instantaneous_size",
            "instantaneous_power",
            "instantaneous_peak_amp",
            "instantaneous_width",
            "expansion_radius",
        ]
    ].copy()
    if assigned_patterns.empty:
        assigned_frames = pd.DataFrame(columns=[*frame_metrics.columns, "network"])
    else:
        assigned_frames = frame_metrics.merge(
            assigned_patterns[["pattern_id", "network"]],
            on="pattern_id",
            how="inner",
        )

    rows = []
    for area_row in network_areas.itertuples(index=False):
        net = area_row.network
        area = int(area_row.network_area_pixels)
        pattern_sub = assigned_patterns[assigned_patterns["network"] == net] if not assigned_patterns.empty else assigned_patterns
        frame_sub = assigned_frames[assigned_frames["network"] == net] if not assigned_frames.empty else assigned_frames
        n_patterns = int(len(pattern_sub))
        n_frame_events = int(len(frame_sub))
        count_per_frame = n_patterns / frame_count if frame_count > 0 else math.nan
        rows.append(
            {
                "condition": entry.condition,
                "role": entry.role,
                "subid": str(entry.subid),
                "hemisphere": entry.hemisphere,
                "network": net,
                "network_index": int(area_row.network_index),
                "network_area_pixels": area,
                "frame_count": frame_count,
                "assigned_spiral_pattern_count": n_patterns,
                "assigned_spiral_frame_count": n_frame_events,
                "spiral_count_per_frame": count_per_frame,
                "spiral_count_per_network_px": n_patterns / area if area > 0 else math.nan,
                "mean_spiral_size": float(frame_sub["instantaneous_size"].mean()) if n_frame_events else math.nan,
                "mean_network_footprint_px": float(pattern_sub["network_footprint_px"].mean()) if n_patterns else math.nan,
                "mean_network_footprint_fraction": float(pattern_sub["network_footprint_px"].mean()) / area
                if n_patterns and area > 0
                else math.nan,
                "mean_spiral_power": float(frame_sub["instantaneous_power"].mean()) if n_frame_events else math.nan,
                "mean_expansion_radius": float(frame_sub["expansion_radius"].mean()) if n_frame_events else math.nan,
                "mean_overlap_fraction": float(pattern_sub["overlap_fraction"].mean()) if n_patterns else math.nan,
            }
        )
    out = pd.DataFrame(rows)
    if ref is not None:
        if alignment_cache_dir is not None:
            align = cached_network_alignment(
                entry=entry,
                network_grid=network_grid,
                network_areas=network_areas,
                ref=ref,
                spacing=spacing,
                cache_dir=alignment_cache_dir,
                overwrite=overwrite_alignment_cache,
            )
        else:
            align = compute_network_alignment(Path(entry.phase_cube), network_grid, network_areas, ref=ref, spacing=spacing)
        out = out.merge(align, on="network", how="left")
    return out


def wide_to_long(subject_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [col for col in SPIRAL_METRIC_LABELS if col in subject_df.columns]
    id_cols = [
        "condition",
        "role",
        "subid",
        "hemisphere",
        "network",
        "network_index",
        "network_area_pixels",
        "frame_count",
    ]
    long = subject_df.melt(id_vars=id_cols, value_vars=metrics, var_name="metric", value_name="value")
    long["metric_label"] = long["metric"].map(SPIRAL_METRIC_LABELS).fillna(long["metric"])
    return long


def paired_deltas(subject_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [col for col in SPIRAL_METRIC_LABELS if col in subject_df.columns]
    rows = []
    for keys, sub in subject_df.groupby(
        ["subid", "hemisphere", "network", "network_index"],
        sort=False,
        observed=True,
    ):
        pivot = sub.pivot_table(index="subid", columns="role", values=metrics, aggfunc="mean")
        if pivot.empty or not {"Drug", "PCB"}.issubset(set(pivot.columns.get_level_values(1))):
            continue
        subid, hemi, net, net_idx = keys
        for metric in metrics:
            try:
                drug = float(pivot[(metric, "Drug")].iloc[0])
                pcb = float(pivot[(metric, "PCB")].iloc[0])
            except KeyError:
                continue
            rows.append(
                {
                    "subid": subid,
                    "hemisphere": hemi,
                    "network": net,
                    "network_index": net_idx,
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
                "metric_label": SPIRAL_METRIC_LABELS.get(metric, metric),
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


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return math.nan, math.nan
    mean = float(np.mean(finite))
    sem = float(stats.sem(finite, nan_policy="omit"))
    tcrit = float(stats.t.ppf((1.0 + confidence) / 2.0, finite.size - 1))
    return mean - tcrit * sem, mean + tcrit * sem


def group_basic_stats_paired_only(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if delta_df.empty:
        return pd.DataFrame(rows)
    for keys, sub in delta_df.groupby(["hemisphere", "network", "network_index", "metric"], sort=False):
        hemi, network, network_index, metric = keys
        paired = sub[
            np.isfinite(pd.to_numeric(sub["drug_value"], errors="coerce"))
            & np.isfinite(pd.to_numeric(sub["pcb_value"], errors="coerce"))
        ].copy()
        for role, value_col in [("Drug", "drug_value"), ("PCB", "pcb_value")]:
            values = pd.to_numeric(paired[value_col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            ci_low, ci_high = _mean_ci(finite)
            rows.append(
                {
                    "hemisphere": hemi,
                    "network": network,
                    "network_index": int(network_index),
                    "metric": metric,
                    "metric_label": SPIRAL_METRIC_LABELS.get(metric, metric),
                    "role": role,
                    "n_paired": int(finite.size),
                    "mean": float(np.mean(finite)) if finite.size else math.nan,
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan,
                    "sem": float(stats.sem(finite, nan_policy="omit")) if finite.size > 1 else math.nan,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def paired_ttest_summary(delta_df: pd.DataFrame, min_pairs: int = 3) -> pd.DataFrame:
    base = paired_delta_summary(delta_df, min_pairs=min_pairs)
    if base.empty:
        return base
    rename = {
        "n_paired": "n",
        "mean_delta_drug_minus_pcb": "mean_delta",
        "std_delta": "std_delta",
        "sem_delta": "sem_delta",
        "t_1sample_delta": "t",
        "p_1sample_delta": "p",
        "cohen_dz": "dz",
    }
    out = base.rename(columns=rename).copy()
    ci_rows = []
    for keys, sub in delta_df.groupby(["hemisphere", "network", "network_index", "metric"], sort=False):
        values = pd.to_numeric(sub["delta_drug_minus_pcb"], errors="coerce").dropna().to_numpy(dtype=float)
        ci_low, ci_high = _mean_ci(values)
        ci_rows.append((*keys, ci_low, ci_high))
    ci_df = pd.DataFrame(
        ci_rows,
        columns=["hemisphere", "network", "network_index", "metric", "delta_ci95_low", "delta_ci95_high"],
    )
    out = out.merge(ci_df, on=["hemisphere", "network", "network_index", "metric"], how="left")
    cols = [
        "hemisphere",
        "network",
        "network_index",
        "metric",
        "metric_label",
        "n",
        "mean_delta",
        "std_delta",
        "sem_delta",
        "delta_ci95_low",
        "delta_ci95_high",
        "t",
        "p",
        "dz",
    ]
    return out[cols]


def two_way_rm_anova(subject_df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.anova import AnovaRM

    metrics = [col for col in SPIRAL_METRIC_LABELS if col in subject_df.columns]
    network_order = infer_network_order(subject_df["network"])
    rows = []
    for hemi in sorted(subject_df["hemisphere"].dropna().unique()):
        hemi_df = subject_df[subject_df["hemisphere"] == hemi]
        for metric in metrics:
            cols = ["subid", "role", "network", "network_index", metric]
            data = hemi_df[cols].rename(columns={metric: "value"}).copy()
            data["value"] = pd.to_numeric(data["value"], errors="coerce")
            data = data.dropna(subset=["value"])
            data = (
                data.groupby(["subid", "role", "network"], as_index=False, observed=True)["value"]
                .mean()
                .sort_values(["subid", "role", "network"])
            )
            pivot = data.pivot_table(
                index="subid",
                columns=["role", "network"],
                values="value",
                aggfunc="mean",
                observed=True,
            )
            required_cols = pd.MultiIndex.from_product([["Drug", "PCB"], network_order], names=["role", "network"])
            complete = pivot.reindex(columns=required_cols).dropna()
            n_subjects = int(len(complete))
            effects = ["role", "network", "role:network"]
            if n_subjects < 2:
                for effect in effects:
                    rows.append(
                        {
                            "hemisphere": hemi,
                            "metric": metric,
                            "metric_label": SPIRAL_METRIC_LABELS.get(metric, metric),
                            "effect": effect,
                            "n_subjects": n_subjects,
                            "df_num": math.nan,
                            "df_den": math.nan,
                            "F": math.nan,
                            "p": math.nan,
                            "partial_eta_sq": math.nan,
                            "error": "fewer than 2 complete paired subjects",
                        }
                    )
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*stack.*")
                anova_data = complete.stack(["role", "network"]).rename("value").reset_index()
            try:
                result = AnovaRM(anova_data, depvar="value", subject="subid", within=["role", "network"]).fit()
                table = result.anova_table.reset_index().rename(columns={"index": "effect"})
                for _, arow in table.iterrows():
                    effect = str(arow["effect"])
                    f_val = float(arow["F Value"])
                    df_num = float(arow["Num DF"])
                    df_den = float(arow["Den DF"])
                    p_val = float(arow["Pr > F"])
                    rows.append(
                        {
                            "hemisphere": hemi,
                            "metric": metric,
                            "metric_label": SPIRAL_METRIC_LABELS.get(metric, metric),
                            "effect": effect,
                            "n_subjects": n_subjects,
                            "df_num": df_num,
                            "df_den": df_den,
                            "F": f_val,
                            "p": p_val,
                            "partial_eta_sq": (f_val * df_num) / (f_val * df_num + df_den)
                            if np.isfinite(f_val) and np.isfinite(df_num) and np.isfinite(df_den)
                            else math.nan,
                            "error": "",
                        }
                    )
            except Exception as exc:
                for effect in effects:
                    rows.append(
                        {
                            "hemisphere": hemi,
                            "metric": metric,
                            "metric_label": SPIRAL_METRIC_LABELS.get(metric, metric),
                            "effect": effect,
                            "n_subjects": n_subjects,
                            "df_num": math.nan,
                            "df_den": math.nan,
                            "F": math.nan,
                            "p": math.nan,
                            "partial_eta_sq": math.nan,
                            "error": str(exc),
                        }
                    )
    return pd.DataFrame(rows)


def three_way_rm_anova(subject_df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.anova import AnovaRM

    metrics = [col for col in SPIRAL_METRIC_LABELS if col in subject_df.columns]
    network_order = infer_network_order(subject_df["network"])
    hemisphere_order = [hemi for hemi in ["left", "right"] if hemi in set(subject_df["hemisphere"])]
    if not hemisphere_order:
        hemisphere_order = list(dict.fromkeys(subject_df["hemisphere"].dropna().astype(str)))
    rows = []
    effects = [
        "role",
        "hemisphere",
        "network",
        "role:hemisphere",
        "role:network",
        "hemisphere:network",
        "role:hemisphere:network",
    ]
    for metric in metrics:
        cols = ["subid", "role", "hemisphere", "network", metric]
        data = subject_df[cols].rename(columns={metric: "value"}).copy()
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["value"])
        data = (
            data.groupby(["subid", "role", "hemisphere", "network"], as_index=False, observed=True)["value"]
            .mean()
            .sort_values(["subid", "role", "hemisphere", "network"])
        )
        pivot = data.pivot_table(
            index="subid",
            columns=["role", "hemisphere", "network"],
            values="value",
            aggfunc="mean",
            observed=True,
        )
        required_cols = pd.MultiIndex.from_product(
            [["Drug", "PCB"], hemisphere_order, network_order],
            names=["role", "hemisphere", "network"],
        )
        complete = pivot.reindex(columns=required_cols).dropna()
        n_subjects = int(len(complete))
        if n_subjects < 2:
            for effect in effects:
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": SPIRAL_METRIC_LABELS.get(metric, metric),
                        "effect": effect,
                        "n_subjects": n_subjects,
                        "df_num": math.nan,
                        "df_den": math.nan,
                        "F": math.nan,
                        "p": math.nan,
                        "partial_eta_sq": math.nan,
                        "error": "fewer than 2 complete paired subjects",
                    }
                )
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*stack.*")
            anova_data = complete.stack(["role", "hemisphere", "network"]).rename("value").reset_index()
        try:
            result = AnovaRM(
                anova_data,
                depvar="value",
                subject="subid",
                within=["role", "hemisphere", "network"],
            ).fit()
            table = result.anova_table.reset_index().rename(columns={"index": "effect"})
            for _, arow in table.iterrows():
                effect = str(arow["effect"])
                f_val = float(arow["F Value"])
                df_num = float(arow["Num DF"])
                df_den = float(arow["Den DF"])
                p_val = float(arow["Pr > F"])
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": SPIRAL_METRIC_LABELS.get(metric, metric),
                        "effect": effect,
                        "n_subjects": n_subjects,
                        "df_num": df_num,
                        "df_den": df_den,
                        "F": f_val,
                        "p": p_val,
                        "partial_eta_sq": (f_val * df_num) / (f_val * df_num + df_den)
                        if np.isfinite(f_val) and np.isfinite(df_num) and np.isfinite(df_den)
                        else math.nan,
                        "error": "",
                    }
                )
        except Exception as exc:
            for effect in effects:
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": SPIRAL_METRIC_LABELS.get(metric, metric),
                        "effect": effect,
                        "n_subjects": n_subjects,
                        "df_num": math.nan,
                        "df_den": math.nan,
                        "F": math.nan,
                        "p": math.nan,
                        "partial_eta_sq": math.nan,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def save_existing_result_stats(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_df = pd.read_csv(out_dir / "subject_network_metrics_wide.csv")
    delta_df = paired_deltas(subject_df)
    basic = group_basic_stats_paired_only(delta_df)
    paired = paired_ttest_summary(delta_df)
    anova = two_way_rm_anova(subject_df)
    three_way = three_way_rm_anova(subject_df)
    delta_df.to_csv(out_dir / "paired_deltas_long.csv", index=False)
    paired_delta_summary(delta_df).to_csv(out_dir / "paired_delta_summary.csv", index=False)
    basic.to_csv(out_dir / "group_basic_stats_paired_only.csv", index=False)
    paired.to_csv(out_dir / "paired_ttest_summary.csv", index=False)
    anova.to_csv(out_dir / "two_way_rm_anova.csv", index=False)
    three_way.to_csv(out_dir / "three_way_rm_anova.csv", index=False)
    return basic, paired, anova, three_way


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


def save_metric_panels(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    network_order: list[str],
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics = [m for m in SPIRAL_METRIC_LABELS if m in set(long_df["metric"])]
    for hemi in sorted(long_df["hemisphere"].dropna().unique()):
        sub = long_df[long_df["hemisphere"] == hemi]
        if sub.empty:
            continue
        ncols = 3
        nrows = int(math.ceil(len(plot_metrics) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.9 * ncols, 5.0 * nrows), squeeze=False)
        for ax, metric in zip(axes.flat, plot_metrics):
            mdf = sub[sub["metric"] == metric].dropna(subset=["value"])
            if mdf.empty:
                ax.set_title(SPIRAL_METRIC_LABELS.get(metric, metric), pad=10)
                continue
            xmin = float(mdf["value"].min())
            xmax = float(mdf["value"].max())
            pad = (xmax - xmin) * 0.08 if xmax > xmin else 1.0
            x_grid = np.linspace(xmin - pad, xmax + pad, 256)
            row_gap = 1.0
            ridge_height = 0.72
            y_positions = {network: (len(network_order) - 1 - idx) * row_gap for idx, network in enumerate(network_order)}
            for network in network_order:
                y0 = y_positions[network]
                for role in ["PCB", "Drug"]:
                    vals = mdf[(mdf["network"] == network) & (mdf["role"] == role)]["value"].to_numpy(dtype=float)
                    if vals.size >= 2 and np.nanstd(vals) > 0:
                        try:
                            kde = stats.gaussian_kde(vals)
                            dens = kde(x_grid)
                            max_dens = float(np.nanmax(dens))
                            if max_dens > 0:
                                dens = dens / max_dens * ridge_height
                                ax.plot(x_grid, y0 + dens, color=ROLE_PALETTE[role], linewidth=1.5, label=role)
                                ax.fill_between(x_grid, y0, y0 + dens, color=ROLE_PALETTE[role], alpha=0.16)
                        except Exception:
                            ax.scatter(vals, np.full(vals.shape, y0), color=ROLE_PALETTE[role], s=8, alpha=0.7)
                    elif vals.size:
                        ax.scatter(vals, np.full(vals.shape, y0), color=ROLE_PALETTE[role], s=12, alpha=0.75, label=role)
                ax.hlines(y0, xmin - pad, xmax + pad, color="#888888", linewidth=0.45, alpha=0.45)
                srow = summary_df[
                    (summary_df["hemisphere"] == hemi)
                    & (summary_df["network"] == network)
                    & (summary_df["metric"] == metric)
                ]
                if not srow.empty:
                    stars = p_to_stars(float(srow.iloc[0]["p_1sample_delta"]))
                    if stars:
                        ax.text(xmax + pad * 0.72, y0 + ridge_height * 0.35, stars, ha="left", va="center", fontsize=10)
            ax.set_yticks([y_positions[n] for n in network_order])
            ax.set_yticklabels(network_order)
            ax.set_xlim(xmin - pad, xmax + pad * 1.25)
            ax.set_ylim(-0.35, max(y_positions.values()) + ridge_height + 0.35)
            ax.set_title(SPIRAL_METRIC_LABELS.get(metric, metric), pad=10)
            ax.set_ylabel("")
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelbottom=False)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", visible=False)
        for ax in axes.flat[len(plot_metrics) :]:
            ax.axis("off")
        handles = [
            plt.Line2D([0], [0], color=ROLE_PALETTE["PCB"], linewidth=2, label="PCB"),
            plt.Line2D([0], [0], color=ROLE_PALETTE["Drug"], linewidth=2, label="Drug"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False)
        fig.suptitle(f"Subject-level network density ridgelines | {hemi}", y=1.015)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(fig_dir / f"metric_ridgelines_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_network_facets_by_metric(long_df: pd.DataFrame, fig_dir: Path, network_order: list[str]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics = [m for m in SPIRAL_METRIC_LABELS if m in set(long_df["metric"])]
    for hemi in sorted(long_df["hemisphere"].dropna().unique()):
        for metric in plot_metrics:
            sub = long_df[(long_df["hemisphere"] == hemi) & (long_df["metric"] == metric)].dropna(subset=["value"])
            if sub.empty:
                continue
            ncols = 4
            nrows = int(math.ceil(len(network_order) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.5 * nrows), squeeze=False)
            for ax, network in zip(axes.flat, network_order):
                ndf = sub[sub["network"] == network]
                for role in ["PCB", "Drug"]:
                    vals = ndf[ndf["role"] == role]["value"].to_numpy(dtype=float)
                    if vals.size >= 2 and np.nanstd(vals) > 0:
                        sns.kdeplot(vals, ax=ax, label=role, color=ROLE_PALETTE[role], linewidth=1.8)
                    elif vals.size:
                        ax.axvline(float(vals[0]), color=ROLE_PALETTE[role], linewidth=1.2, alpha=0.7)
                ax.set_title(network)
                ax.set_xlabel("")
                ax.set_ylabel("Density")
            for ax in axes.flat[len(network_order) :]:
                ax.axis("off")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False)
            fig.suptitle(f"{SPIRAL_METRIC_LABELS.get(metric, metric)} | {hemi}", y=1.03)
            fig.tight_layout(rect=[0, 0, 1, 0.91])
            fig.savefig(fig_dir / f"network_facets_{metric}_{hemi}.png", dpi=300)
            plt.close(fig)


def save_delta_heatmaps(delta_df: pd.DataFrame, fig_dir: Path, network_order: list[str]) -> None:
    if delta_df.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    for hemi in sorted(delta_df["hemisphere"].dropna().unique()):
        sub = delta_df[delta_df["hemisphere"] == hemi]
        metrics = [m for m in SPIRAL_METRIC_LABELS if m in set(sub["metric"])]
        mat = pd.DataFrame(index=[SPIRAL_METRIC_LABELS[m] for m in metrics], columns=network_order, dtype=float)
        for metric in metrics:
            mdf = sub[sub["metric"] == metric]
            means = mdf.groupby("network")["delta_drug_minus_pcb"].mean()
            for network in network_order:
                mat.loc[SPIRAL_METRIC_LABELS[metric], network] = means.get(network, math.nan)
        vals = mat.to_numpy(dtype=float)
        vmax = np.nanpercentile(np.abs(vals), 95) if np.isfinite(vals).any() else 1.0
        vmax = max(float(vmax), 1e-6)
        plt.figure(figsize=(10, max(5, 0.45 * len(metrics))))
        sns.heatmap(mat, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, cbar_kws={"label": "Drug - PCB"})
        plt.title(f"Mean paired delta by network ({hemi})")
        plt.tight_layout()
        plt.savefig(fig_dir / f"paired_delta_heatmap_{hemi}.png", dpi=300)
        plt.close()


def _metric_short_labels(metrics: list[str]) -> dict[str, str]:
    return {
        metric: SPIRAL_METRIC_LABELS.get(metric, metric)
        .replace("Spiral ", "")
        .replace("Mean ", "")
        .replace(" in network", "")
        .replace(" fraction of network", " fraction")
        .replace("Weighted mean ", "W. ")
        for metric in metrics
    }


def _annotate_effect_stars(ax: plt.Axes, x_positions: dict[object, float], df: pd.DataFrame, x_col: str) -> None:
    y_vals = pd.to_numeric(df["cohen_dz"], errors="coerce").to_numpy(dtype=float)
    finite = y_vals[np.isfinite(y_vals)]
    span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else 1.0
    offset = max(0.035 * span, 0.035)
    for row in df.itertuples(index=False):
        stars = p_to_stars(float(row.p_1sample_delta))
        y = float(row.cohen_dz)
        if not stars or not np.isfinite(y):
            continue
        x = x_positions[getattr(row, x_col)]
        va = "bottom" if y >= 0 else "top"
        ax.text(x, y + offset if y >= 0 else y - offset, stars, ha="center", va=va, fontsize=10, color="black")


def save_effect_size_plots(summary_df: pd.DataFrame, fig_dir: Path, network_order: list[str]) -> None:
    if summary_df.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in SPIRAL_METRIC_LABELS if m in set(summary_df["metric"])]
    metric_labels = _metric_short_labels(metrics)
    net_palette = dict(zip(network_order, sns.color_palette("tab10", n_colors=len(network_order))))
    metric_palette = dict(zip(metrics, sns.color_palette("husl", n_colors=len(metrics))))

    for hemi in sorted(summary_df["hemisphere"].dropna().unique()):
        sub = summary_df[summary_df["hemisphere"] == hemi].copy()
        sub = sub[sub["metric"].isin(metrics)]
        if sub.empty:
            continue

        metric_x = {metric: idx for idx, metric in enumerate(metrics)}
        fig, ax = plt.subplots(figsize=(max(11, 1.35 * len(metrics)), 6.5))
        for network in network_order:
            ndf = sub[sub["network"] == network].sort_values("metric", key=lambda s: s.map(metric_x))
            if ndf.empty:
                continue
            x = [metric_x[m] for m in ndf["metric"]]
            ax.plot(x, ndf["cohen_dz"], marker="o", linewidth=1.6, label=network, color=net_palette[network])
            _annotate_effect_stars(ax, metric_x, ndf, "metric")
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.8)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=35, ha="right")
        ax.set_ylabel("Cohen dz (Drug - PCB)")
        ax.set_title(f"Effect size by metric | networks as lines | {hemi}", pad=12)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=min(7, len(network_order)),
            frameon=False,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.78])
        fig.savefig(fig_dir / f"effect_size_metrics_x_network_lines_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        network_x = {network: idx for idx, network in enumerate(network_order)}
        fig, ax = plt.subplots(figsize=(max(11, 1.15 * len(network_order)), 6.5))
        for metric in metrics:
            mdf = sub[sub["metric"] == metric].sort_values("network_index")
            if mdf.empty:
                continue
            x = [network_x[n] for n in mdf["network"]]
            ax.plot(x, mdf["cohen_dz"], marker="o", linewidth=1.6, label=metric_labels[metric], color=metric_palette[metric])
            _annotate_effect_stars(ax, network_x, mdf, "network")
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.8)
        ax.set_xticks(range(len(network_order)))
        ax.set_xticklabels(network_order, rotation=25, ha="right")
        ax.set_ylabel("Cohen dz (Drug - PCB)")
        ax.set_title(f"Effect size by network | metrics as lines | {hemi}", pad=12)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=3,
            frameon=False,
            fontsize=9,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.74])
        fig.savefig(fig_dir / f"effect_size_networks_x_metric_lines_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    alignment_cache_dir = args.alignment_cache_dir or (args.out_dir / "cache" / "alignment")

    if args.stats_only:
        basic, paired, anova, three_way = save_existing_result_stats(args.out_dir)
        print(
            "Regenerated stats from existing CSVs: "
            f"{len(basic)} group rows, {len(paired)} paired t-test rows, "
            f"{len(anova)} two-way ANOVA rows, {len(three_way)} three-way ANOVA rows."
        )
        return 0

    if args.plot_only:
        basic, paired, anova, three_way = save_existing_result_stats(args.out_dir)
        subject_df = pd.read_csv(args.out_dir / "subject_network_metrics_wide.csv")
        long_df = pd.read_csv(args.out_dir / "subject_network_metrics_long.csv")
        delta_df = pd.read_csv(args.out_dir / "paired_deltas_long.csv")
        delta_summary = pd.read_csv(args.out_dir / "paired_delta_summary.csv")
        network_order = infer_network_order(subject_df["network"])
        fig_dir = args.out_dir / "figures"
        save_metric_panels(long_df, delta_summary, fig_dir / "metric_panels", network_order)
        save_network_facets_by_metric(long_df, fig_dir / "network_facets_by_metric", network_order)
        save_delta_heatmaps(delta_df, fig_dir, network_order)
        save_effect_size_plots(delta_summary, fig_dir / "effect_size", network_order)
        print(
            f"Regenerated plots and stats from existing CSVs in: {args.out_dir} "
            f"({len(basic)} group rows, {len(paired)} paired t-test rows, "
            f"{len(anova)} two-way ANOVA rows, {len(three_way)} three-way ANOVA rows)."
        )
        return 0

    entries = discover_entries(args)
    ref_left, ref_right = load_default_refs(args)
    refs = {}
    if not args.skip_alignment:
        if ref_left is not None:
            refs["left"] = load_reference(ref_left, spacing=args.spacing, edge_margin=args.edge_margin)
        if ref_right is not None:
            refs["right"] = load_reference(ref_right, spacing=args.spacing, edge_margin=args.edge_margin)

    records = []
    failures = []
    for row in entries.itertuples(index=False):
        try:
            print(f"Processing {row.role} S{row.subid} {row.hemisphere}: {Path(row.fc_dir).name}")
            ref = refs.get(row.hemisphere)
            records.append(
                summarize_bundle(
                    pd.Series(row._asdict()),
                    min_overlap=args.min_overlap,
                    denominator_mode=args.overlap_denominator,
                    ref=ref,
                    spacing=args.spacing,
                    alignment_cache_dir=alignment_cache_dir if ref is not None else None,
                    overwrite_alignment_cache=args.overwrite_alignment_cache,
                )
            )
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
        raise RuntimeError("No subject-network records were generated.")

    subject_df = pd.concat(records, ignore_index=True)
    network_order = infer_network_order(subject_df["network"])
    subject_df["network"] = pd.Categorical(subject_df["network"], categories=network_order, ordered=True)
    subject_df = subject_df.sort_values(["role", "subid", "hemisphere", "network"]).reset_index(drop=True)
    long_df = wide_to_long(subject_df)
    delta_df = paired_deltas(subject_df)
    delta_summary = paired_delta_summary(delta_df)
    basic_stats = group_basic_stats_paired_only(delta_df)
    ttest_summary = paired_ttest_summary(delta_df)
    anova_summary = two_way_rm_anova(subject_df)
    three_way_anova_summary = three_way_rm_anova(subject_df)

    subject_df.to_csv(args.out_dir / "subject_network_metrics_wide.csv", index=False)
    long_df.to_csv(args.out_dir / "subject_network_metrics_long.csv", index=False)
    delta_df.to_csv(args.out_dir / "paired_deltas_long.csv", index=False)
    delta_summary.to_csv(args.out_dir / "paired_delta_summary.csv", index=False)
    basic_stats.to_csv(args.out_dir / "group_basic_stats_paired_only.csv", index=False)
    ttest_summary.to_csv(args.out_dir / "paired_ttest_summary.csv", index=False)
    anova_summary.to_csv(args.out_dir / "two_way_rm_anova.csv", index=False)
    three_way_anova_summary.to_csv(args.out_dir / "three_way_rm_anova.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.out_dir / "failures.csv", index=False)

    if not args.no_plots:
        fig_dir = args.out_dir / "figures"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            save_metric_panels(long_df, delta_summary, fig_dir / "metric_panels", network_order)
            save_network_facets_by_metric(long_df, fig_dir / "network_facets_by_metric", network_order)
            save_delta_heatmaps(delta_df, fig_dir, network_order)
            save_effect_size_plots(delta_summary, fig_dir / "effect_size", network_order)

    metadata = {
        "phase_fc_root": str(args.phase_fc_root),
        "detect_root": str(args.detect_root),
        "drug_condition": args.drug_condition,
        "pcb_condition": args.pcb_condition,
        "min_overlap": float(args.min_overlap),
        "overlap_denominator": args.overlap_denominator,
        "ref_left": str(ref_left) if ref_left is not None else None,
        "ref_right": str(ref_right) if ref_right is not None else None,
        "skip_alignment": bool(args.skip_alignment),
        "alignment_cache_dir": str(alignment_cache_dir),
        "overwrite_alignment_cache": bool(args.overwrite_alignment_cache),
        "spacing": float(args.spacing),
        "edge_margin": int(args.edge_margin),
        "n_entries": int(len(entries)),
        "n_failures": int(len(failures)),
        "network_order": network_order,
        "metrics": SPIRAL_METRIC_LABELS,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote network spiral metrics to: {args.out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
