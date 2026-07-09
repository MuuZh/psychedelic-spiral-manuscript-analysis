from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, label
from tqdm import tqdm

from .dfr_stats import add_legacy_mean_alias, mean_error_fields

try:
    from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))
    from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field


MAGNITUDE_THRESHOLD = 0.2
BOUNDARY_THICKNESS = 2
MIN_REGION_SIZE = 10
EDGE_MARGIN = 2


@dataclass(frozen=True)
class DfrParams:
    magnitude_threshold: float = MAGNITUDE_THRESHOLD
    boundary_thickness: int = BOUNDARY_THICKNESS
    min_region_size: int = MIN_REGION_SIZE
    edge_margin: int = EDGE_MARGIN
    spacing: float = 1.0


@dataclass
class FrameDfr:
    frame: int
    boundary_mask: np.ndarray
    labeled_regions: np.ndarray
    region_count: int
    boundary_count: int
    grad_mag: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    unit_x: np.ndarray
    unit_y: np.ndarray


def align_2d(arr: np.ndarray, target_shape: tuple[int, int], name: str = "array") -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    if arr.shape[0] == target_shape[0] + 1 and arr.shape[1] == target_shape[1]:
        return arr[:-1, :]
    if arr.shape[0] + 1 == target_shape[0] and arr.shape[1] == target_shape[1]:
        out = np.full(target_shape, np.nan, dtype=float)
        out[:-1, :] = arr
        return out
    raise ValueError(f"Cannot align {name} shape {arr.shape} to {target_shape}")


def apply_edge_margin(mask: np.ndarray, margin: int) -> np.ndarray:
    out = mask.copy()
    if margin <= 0:
        return out
    out[:margin, :] = False
    out[-margin:, :] = False
    out[:, :margin] = False
    out[:, -margin:] = False
    return out


def valid_mask_from_phase_and_grid(phase_cube: np.ndarray, grid_labels: np.ndarray | None, edge_margin: int) -> np.ndarray:
    base = np.isfinite(phase_cube).any(axis=2)
    if grid_labels is not None:
        grid = align_2d(np.asarray(grid_labels), phase_cube.shape[:2], "grid_labels")
        base &= np.isfinite(grid)
    return apply_edge_margin(base, edge_margin)


def compute_dfr_fields(phase_cube: np.ndarray, params: DfrParams) -> dict[str, np.ndarray]:
    grad_x, grad_y = compute_phase_gradient(phase_cube, spacing=params.spacing, show_progress=False)
    unit_x, unit_y, grad_mag = normalize_vector_field(grad_x, grad_y)
    return {
        "grad_x": grad_x,
        "grad_y": grad_y,
        "unit_x": unit_x,
        "unit_y": unit_y,
        "grad_mag": grad_mag,
    }


def label_regions_from_boundary(
    boundary_mask: np.ndarray,
    valid_mask: np.ndarray,
    params: DfrParams,
) -> tuple[np.ndarray, int, np.ndarray]:
    boundary = boundary_mask & valid_mask
    if params.boundary_thickness > 0:
        boundary = binary_dilation(boundary, iterations=params.boundary_thickness) & valid_mask

    candidates = (~boundary) & valid_mask
    candidates = binary_closing(candidates, iterations=2) & valid_mask
    raw_labels, n_raw = label(candidates)

    labels = np.zeros_like(raw_labels, dtype=np.int32)
    next_label = 1
    for region_id in range(1, n_raw + 1):
        region = raw_labels == region_id
        if int(region.sum()) >= params.min_region_size:
            labels[region] = next_label
            next_label += 1

    final_boundary = (labels == 0) & valid_mask
    return labels, next_label - 1, final_boundary


def compute_frame_dfr(
    fields: dict[str, np.ndarray],
    frame: int,
    valid_mask: np.ndarray,
    params: DfrParams,
) -> FrameDfr:
    grad_mag = fields["grad_mag"][:, :, frame]
    boundary = (grad_mag > params.magnitude_threshold) & valid_mask & np.isfinite(grad_mag)
    labels, region_count, final_boundary = label_regions_from_boundary(boundary, valid_mask, params)
    return FrameDfr(
        frame=frame,
        boundary_mask=final_boundary,
        labeled_regions=labels,
        region_count=int(region_count),
        boundary_count=int(final_boundary.sum()),
        grad_mag=grad_mag,
        grad_x=fields["grad_x"][:, :, frame],
        grad_y=fields["grad_y"][:, :, frame],
        unit_x=fields["unit_x"][:, :, frame],
        unit_y=fields["unit_y"][:, :, frame],
    )


def summarize_phase_cube(
    phase_cube: np.ndarray,
    valid_mask: np.ndarray,
    params: DfrParams,
    qc_frames: list[int] | None = None,
    show_progress: bool = False,
    progress_desc: str = "DFR frames",
) -> tuple[dict[str, float], dict[int, FrameDfr], np.ndarray, np.ndarray]:
    fields = compute_dfr_fields(np.asarray(phase_cube), params)
    n_frames = int(phase_cube.shape[2])
    region_counts: list[int] = []
    boundary_counts: list[int] = []
    occupancy = np.zeros(phase_cube.shape[:2], dtype=np.float64)
    boundary_stack = np.zeros((*phase_cube.shape[:2], n_frames), dtype=bool)
    qc_set = {int(f) for f in (qc_frames or []) if 0 <= int(f) < n_frames}
    qc: dict[int, FrameDfr] = {}

    iterator = range(n_frames)
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc, unit="frame", leave=False)
    for frame in iterator:
        frame_dfr = compute_frame_dfr(fields, frame, valid_mask, params)
        region_counts.append(frame_dfr.region_count)
        boundary_counts.append(frame_dfr.boundary_count)
        occupancy += frame_dfr.boundary_mask.astype(np.float64)
        boundary_stack[:, :, frame] = frame_dfr.boundary_mask
        if frame in qc_set:
            qc[frame] = frame_dfr

    summary = {
        "n_frames": n_frames,
        **add_legacy_mean_alias(mean_error_fields(region_counts, "region_count"), "region_count", "mean_region_count"),
        **add_legacy_mean_alias(mean_error_fields(boundary_counts, "boundary_count"), "boundary_count", "mean_boundary_count"),
    }
    return summary, qc, occupancy / max(n_frames, 1), boundary_stack


def default_qc_frames(n_frames: int) -> list[int]:
    if n_frames <= 0:
        return []
    return sorted(set([0, n_frames // 4, n_frames // 2, (3 * n_frames) // 4]))
