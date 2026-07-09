from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dfr_core import DfrParams, FrameDfr


CACHE_VERSION = "dfr-cache-v2"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def cache_key(
    phase_path: Path,
    grid_path: Path | None,
    params: DfrParams,
    qc_frames: list[int],
) -> str:
    phase = phase_path.resolve()
    payload: dict[str, Any] = {
        "version": CACHE_VERSION,
        "phase_path": str(phase),
        "phase_mtime_ns": phase.stat().st_mtime_ns,
        "phase_size": phase.stat().st_size,
        "grid_path": None,
        "grid_mtime_ns": None,
        "grid_size": None,
        "params": params.__dict__,
        "qc_frames": sorted(int(f) for f in qc_frames),
    }
    if grid_path is not None:
        grid = grid_path.resolve()
        if grid.exists():
            payload.update(
                {
                    "grid_path": str(grid),
                    "grid_mtime_ns": grid.stat().st_mtime_ns,
                    "grid_size": grid.stat().st_size,
                }
            )
    text = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.npz"


def load_cached_summary(cache_file: Path) -> tuple[dict[str, float], dict[int, FrameDfr], np.ndarray, np.ndarray] | None:
    if not cache_file.exists():
        return None
    data = np.load(cache_file, allow_pickle=True)
    summary = json.loads(str(data["summary_json"]))
    avg_boundary = data["avg_boundary"]
    boundary_stack = data["boundary_stack"].astype(bool, copy=False)
    qc: dict[int, FrameDfr] = {}
    qc_frames = data["qc_frames"].astype(int).tolist()
    for frame in qc_frames:
        prefix = f"qc_{frame}_"
        qc[frame] = FrameDfr(
            frame=frame,
            boundary_mask=data[prefix + "boundary_mask"].astype(bool, copy=False),
            labeled_regions=data[prefix + "labeled_regions"].astype(np.int32, copy=False),
            region_count=int(data[prefix + "region_count"]),
            boundary_count=int(data[prefix + "boundary_count"]),
            grad_mag=data[prefix + "grad_mag"],
            grad_x=data[prefix + "grad_x"],
            grad_y=data[prefix + "grad_y"],
            unit_x=data[prefix + "unit_x"],
            unit_y=data[prefix + "unit_y"],
        )
    return summary, qc, avg_boundary, boundary_stack


def write_cached_summary(
    cache_file: Path,
    summary: dict[str, float],
    qc: dict[int, FrameDfr],
    avg_boundary: np.ndarray,
    boundary_stack: np.ndarray,
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "summary_json": json.dumps(summary, sort_keys=True),
        "avg_boundary": avg_boundary,
        "boundary_stack": boundary_stack.astype(bool, copy=False),
        "qc_frames": np.array(sorted(qc), dtype=np.int32),
    }
    for frame, item in qc.items():
        prefix = f"qc_{frame}_"
        arrays[prefix + "boundary_mask"] = item.boundary_mask
        arrays[prefix + "labeled_regions"] = item.labeled_regions
        arrays[prefix + "region_count"] = np.array(item.region_count, dtype=np.int32)
        arrays[prefix + "boundary_count"] = np.array(item.boundary_count, dtype=np.int32)
        arrays[prefix + "grad_mag"] = item.grad_mag
        arrays[prefix + "grad_x"] = item.grad_x
        arrays[prefix + "grad_y"] = item.grad_y
        arrays[prefix + "unit_x"] = item.unit_x
        arrays[prefix + "unit_y"] = item.unit_y
    np.savez_compressed(cache_file, **arrays)
