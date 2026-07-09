from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt

from .dfr_core import align_2d
from .dfr_stats import mean_error_fields, proportion_error_fields


def parcel_to_network_grid(grid_labels: np.ndarray, parcel_meta_path: Path | None) -> np.ndarray:
    if parcel_meta_path is None or not Path(parcel_meta_path).exists():
        return np.asarray(grid_labels)
    meta = pd.read_csv(parcel_meta_path)
    if "parcel_id" not in meta.columns or "network" not in meta.columns:
        return np.asarray(grid_labels)
    networks = {name: idx + 1 for idx, name in enumerate(dict.fromkeys(meta["network"].astype(str)))}
    parcel_to_network = {int(row.parcel_id): networks[str(row.network)] for row in meta.itertuples(index=False)}
    out = np.full(np.asarray(grid_labels).shape, np.nan, dtype=float)
    finite = np.isfinite(grid_labels)
    for parcel_id in np.unique(grid_labels[finite]).astype(int):
        if parcel_id in parcel_to_network:
            out[grid_labels == parcel_id] = parcel_to_network[parcel_id]
    return out


def boundary_from_labels(labels: np.ndarray, tolerance: int = 0) -> np.ndarray:
    valid = np.isfinite(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    diff_right = (labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    boundary[:, :-1] |= diff_right
    boundary[:, 1:] |= diff_right
    diff_down = (labels[:-1, :] != labels[1:, :]) & valid[:-1, :] & valid[1:, :]
    boundary[:-1, :] |= diff_down
    boundary[1:, :] |= diff_down
    if tolerance > 0:
        boundary = binary_dilation(boundary, iterations=tolerance) & valid
    return boundary


def overlap_metrics(dfr_boundary: np.ndarray, atlas_boundary: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    dfr = dfr_boundary & valid_mask
    atlas = atlas_boundary & valid_mask
    intersection = dfr & atlas
    dfr_n = int(dfr.sum())
    atlas_n = int(atlas.sum())
    valid_n = int(valid_mask.sum())
    non_atlas = valid_mask & ~atlas
    p_atlas = float(intersection.sum() / atlas_n) if atlas_n else math.nan
    p_non = float((dfr & non_atlas).sum() / non_atlas.sum()) if non_atlas.sum() else math.nan
    return {
        "intersection_pixels": int(intersection.sum()),
        "DFR_boundary_pixels": dfr_n,
        "atlas_boundary_pixels": atlas_n,
        "precision": float(intersection.sum() / dfr_n) if dfr_n else math.nan,
        "recall": float(intersection.sum() / atlas_n) if atlas_n else math.nan,
        "dice": float(2 * intersection.sum() / (dfr_n + atlas_n)) if (dfr_n + atlas_n) else math.nan,
        "enrichment": float(p_atlas / p_non) if p_non and np.isfinite(p_non) else math.nan,
        "valid_pixels": valid_n,
    }


def distance_metrics(dfr_boundary: np.ndarray, atlas_boundary: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    dfr = dfr_boundary & valid_mask
    if not np.any(dfr) or not np.any(atlas_boundary & valid_mask):
        return {
            "mean_distance": math.nan,
            "median_distance": math.nan,
            "fraction_within_1": math.nan,
            "fraction_within_2": math.nan,
            "fraction_within_3": math.nan,
            "background_mean_distance": math.nan,
            "background_median_distance": math.nan,
        }
    dist = distance_transform_edt(~(atlas_boundary & valid_mask))
    dvals = dist[dfr]
    bvals = dist[valid_mask]
    out = {
        **mean_error_fields(dvals, "distance"),
        "median_distance": float(np.median(dvals)),
        "fraction_within_1": float(np.mean(dvals <= 1)),
        "fraction_within_2": float(np.mean(dvals <= 2)),
        "fraction_within_3": float(np.mean(dvals <= 3)),
        **mean_error_fields(bvals, "background_distance"),
        "background_median_distance": float(np.median(bvals)),
    }
    out["mean_distance"] = out["distance_mean"]
    out["background_mean_distance"] = out["background_distance_mean"]
    for radius in (1, 2, 3):
        out.update(proportion_error_fields(int(np.sum(dvals <= radius)), int(len(dvals)), f"fraction_within_{radius}"))
    return out


def dfr_boundary_from_occupancy(
    avg_boundary: np.ndarray,
    valid_mask: np.ndarray,
    mean_boundary_count: float | None,
    mode: str,
    occupancy_threshold: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    valid_values = avg_boundary[valid_mask]
    if mode == "occupancy-threshold":
        threshold = float(occupancy_threshold)
        boundary = (avg_boundary >= threshold) & valid_mask
        return boundary, {
            "dfr_boundary_selection": mode,
            "dfr_occupancy_threshold": threshold,
            "dfr_top_mean_target_pixels": math.nan,
        }
    if mode != "top-mean":
        raise ValueError(f"Unknown network overlap mode: {mode}")
    if valid_values.size == 0:
        return np.zeros_like(valid_mask, dtype=bool), {
            "dfr_boundary_selection": mode,
            "dfr_occupancy_threshold": math.nan,
            "dfr_top_mean_target_pixels": 0,
        }
    target = int(round(float(mean_boundary_count))) if mean_boundary_count is not None and np.isfinite(mean_boundary_count) else 0
    target = max(0, min(target, int(valid_mask.sum())))
    boundary = np.zeros_like(valid_mask, dtype=bool)
    if target > 0:
        flat_valid = np.flatnonzero(valid_mask.ravel())
        flat_values = avg_boundary.ravel()[flat_valid]
        if target >= len(flat_valid):
            chosen = flat_valid
        else:
            order = np.argpartition(flat_values, -target)[-target:]
            chosen = flat_valid[order]
        boundary.ravel()[chosen] = True
    threshold = float(np.min(avg_boundary[boundary])) if np.any(boundary) else math.nan
    return boundary, {
        "dfr_boundary_selection": mode,
        "dfr_occupancy_threshold": threshold,
        "dfr_top_mean_target_pixels": target,
    }


def compute_network_overlap_rows(
    row: pd.Series,
    avg_boundary: np.ndarray,
    valid_mask: np.ndarray,
    tolerance: int,
    dfr_mode: str = "top-mean",
    occupancy_threshold: float = 0.5,
) -> tuple[dict[str, float] | None, dict[str, float] | None, np.ndarray | None]:
    grid_path = row.get("grid_path")
    if pd.isna(grid_path) or not grid_path:
        return None, None, None
    labels = np.load(Path(grid_path))
    labels = align_2d(labels, avg_boundary.shape, "grid_labels")
    network_grid = parcel_to_network_grid(labels, Path(row["parcel_meta_path"]) if row.get("parcel_meta_path") else None)
    atlas_boundary = boundary_from_labels(network_grid, tolerance=tolerance)
    dfr_boundary, selection = dfr_boundary_from_occupancy(
        avg_boundary,
        valid_mask,
        row.get("mean_boundary_count"),
        dfr_mode,
        occupancy_threshold,
    )
    overlap = overlap_metrics(dfr_boundary, atlas_boundary, valid_mask)
    overlap.update(selection)
    distance = distance_metrics(dfr_boundary, atlas_boundary, valid_mask)
    distance.update(selection)
    return overlap, distance, atlas_boundary
