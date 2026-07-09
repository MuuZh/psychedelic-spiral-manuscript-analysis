#!/usr/bin/env python
"""
Batch runner: full preprocessing + windowed spiral detection + curl-threshold sweep.

Key behavior:
- Reads a subject manifest (CSV/TSV/JSON; required column: cifti_file).
- For each subject/hemisphere, performs full preprocessing once and reuses the
  resulting phase/curl fields across all curl-threshold and optflow runs.
- Detects spirals only inside a selected frame window to reduce runtime.
- Uses the same frame window for all curl-threshold runs within a subject.
- Supports manual or random window selection per subject.

Manifest optional columns:
- hemisphere: left/right/both (overrides --hemisphere)
- subject_id: output folder label
- frame_start: manual window start (overrides global/manual/random)
- frame_length: window length for this row
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from matphase.analysis.storage import save_subject_bundle_from_detection
from matphase.config import load_config
from matphase.config.schema import MatPhaseConfig
from matphase.detect.expansion import (
    apply_expanded_masks_to_detection,
    compute_phase_alignment_mask,
    expand_spiral_patterns,
)
from matphase.detect.phase_field import compute_phase_field
from matphase.detect.spirals import (
    SpiralDetectionResult,
    SpiralPattern,
    detect_spirals_directional,
    detect_spirals_from_masks,
)
from matphase.detect.thresholds import apply_combined_threshold
from matphase.io import load_cifti, load_surface
from matphase.io.parcellation import load_parcellation, parcellation_to_mask
from matphase.preprocess import interpolate_to_grid_batch
from matphase.preprocess.spatial import spatial_bandpass_filter
from matphase.preprocess.temporal import temporal_bandpass_filter
from matphase.utils import get_logger, setup_logging

from run_full_detection_bundle import (
    _apply_pass_map_to_patterns,
    _build_pattern_time_value_map,
    _compute_significant_metrics,
    _compute_surrogate_compatibility_distributions,
    _filter_pattern_masks_by_pass_map,
    _hemisphere_ranges,
    _hilbert_phase_cube,
    _resolve_detection_thresholds,
    _resolve_sampling_rate,
    _sanitize_bundle_name,
)

logger = get_logger(__name__)


# ============================ USER CONFIG (edit here) ============================
DEFAULT_CONFIG = Path("configs/defaults.yaml")
DEFAULT_CIFTI_FILE = Path("")
DEFAULT_OUTDIR = Path("output/curl_threshold_sweep")
DEFAULT_CURL_THRESHOLDS: List[float] = [0.8, 1.0, 1.2, 1.4]
# ===============================================================================


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() in {".json"}:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON manifest must be a list of objects.")
        return [dict(row) for row in data]

    delimiter = "\t" if path.suffix.lower() in {".tsv"} else ","
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]


def _iter_rows(manifest: List[Dict[str, str]]) -> Iterable[Dict[str, str]]:
    for idx, row in enumerate(manifest, start=1):
        if not row.get("cifti_file"):
            raise ValueError(f"Row {idx} missing required field: cifti_file")
        yield row


def _parse_curl_thresholds(raw_thresholds: Optional[List[float]]) -> List[float]:
    thresholds = raw_thresholds if raw_thresholds else DEFAULT_CURL_THRESHOLDS
    if not thresholds:
        raise ValueError(
            "No curl thresholds available. Set DEFAULT_CURL_THRESHOLDS in this script "
            "or provide --curl-threshold-series."
        )
    parsed = [float(v) for v in thresholds]
    if any(v <= 0 for v in parsed):
        raise ValueError(
            f"Invalid curl threshold list: {parsed}. All values must be > 0."
        )
    return parsed


def _run_command(cmd: List[str]) -> int:
    proc = subprocess.Popen(cmd)
    return proc.wait()


def _format_curl_tag(threshold: float) -> str:
    token = f"{threshold:.6g}".replace("-", "m").replace(".", "p")
    return f"curl_{token}"


def _enabled_rotation_dirs(rotation_mode: str) -> set[str]:
    if rotation_mode == "both":
        return {"ccw", "cw"}
    if rotation_mode in {"ccw", "cw"}:
        return {rotation_mode}
    raise ValueError(f"Unsupported rotation_mode: {rotation_mode}")


def _paint_patch(
    mask_2d: np.ndarray,
    *,
    row: int,
    col: int,
    radius: int,
    valid_2d: Optional[np.ndarray] = None,
) -> None:
    h, w = mask_2d.shape
    y0 = max(row - radius, 0)
    y1 = min(row + radius, h - 1)
    x0 = max(col - radius, 0)
    x1 = min(col + radius, w - 1)
    if valid_2d is None:
        mask_2d[y0: y1 + 1, x0: x1 + 1] = True
    else:
        patch = valid_2d[y0: y1 + 1, x0: x1 + 1]
        mask_2d[y0: y1 + 1, x0: x1 + 1] |= patch


def _build_optflow_focus_masks(
    *,
    phase_cube_full: np.ndarray,
    frame_start: int,
    curl_field: np.ndarray,
    valid_mask: Optional[np.ndarray],
    rotation_mode: str,
    alpha: float,
    beta: float,
    max_iter: int,
    tol: float,
    winding_radius: int,
    winding_min: float,
    stable_only: bool,
    merge_distance: float,
    seed_patch_radius: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    from spiral_optflow_focus.spiral_wave_detector import SpiralWaveDetector

    rows, cols, frames = curl_field.shape
    masks: Dict[str, np.ndarray] = {
        "ccw": np.zeros((rows, cols, frames), dtype=bool),
        "cw": np.zeros((rows, cols, frames), dtype=bool),
    }
    raw_foci_rows: List[Dict[str, object]] = []
    enabled_dirs = _enabled_rotation_dirs(rotation_mode)
    detector = SpiralWaveDetector(
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
        winding_radius=winding_radius,
        winding_min=winding_min,
        stable_only=stable_only,
        merge_distance=merge_distance,
    )
    patch_radius = max(int(seed_patch_radius), 0)

    for frame_idx in range(frames):
        abs_frame_idx = int(frame_start) + frame_idx
        foci = detector.detect_from_phase_cube(
            phase_cube_full,
            frame_idx=abs_frame_idx,
        )
        if not foci:
            continue
        valid_2d = valid_mask[:, :, frame_idx] if valid_mask is not None else None
        for focus in foci:
            row = int(round(float(focus.row)))
            col = int(round(float(focus.col)))
            if row < 0 or row >= rows or col < 0 or col >= cols:
                continue
            if valid_2d is not None and not bool(valid_2d[row, col]):
                continue
            curl_value = float(curl_field[row, col, frame_idx])
            if not np.isfinite(curl_value) or curl_value == 0.0:
                continue
            direction = "ccw" if curl_value > 0 else "cw"
            if direction not in enabled_dirs:
                continue
            raw_foci_rows.append(
                {
                    "abs_time": int(abs_frame_idx),
                    "window_frame": int(frame_idx),
                    "row": float(focus.row),
                    "col": float(focus.col),
                    "y": float(focus.row),
                    "x": float(focus.col),
                    "row_rounded": int(row),
                    "col_rounded": int(col),
                    "direction": direction,
                    "curl_value_at_rounded": float(curl_value),
                    "determinant": float(focus.determinant),
                    "trace": float(focus.trace),
                    "stable": bool(focus.stable),
                    "winding_number": float(focus.winding_number),
                }
            )
            _paint_patch(
                masks[direction][:, :, frame_idx],
                row=row,
                col=col,
                radius=patch_radius,
                valid_2d=valid_2d,
            )

    return masks, raw_foci_rows


def _subject_tag(row: Dict[str, str], cifti_path: Path) -> str:
    if row.get("subject_id"):
        return str(row["subject_id"])
    stem = cifti_path.name.replace(".", "_")
    return stem


def _resolve_cifti_path(data_dir: Path, cifti_file: str) -> Path:
    p = Path(cifti_file)
    return p if p.is_absolute() else data_dir / p


def _parse_frame_int(value: Optional[str]) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _frame_count_from_cifti(cifti_path: Path, hemisphere: str) -> int:
    structure = "CORTEX_LEFT" if hemisphere == "left" else "CORTEX_RIGHT"
    ts = load_cifti(str(cifti_path))
    data = ts.get_full_surface_data(structure)
    if data.ndim != 2:
        raise ValueError(f"Unexpected surface data shape: {data.shape}")
    return int(data.shape[1])


def _deterministic_seed(base_seed: int, subject_key: str) -> int:
    h = hashlib.md5(subject_key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) ^ int(base_seed)


def _choose_window(
    *,
    frame_count: int,
    row: Dict[str, str],
    mode: str,
    default_length: int,
    manual_start: Optional[int],
    random_seed: int,
    random_min_start: int,
    random_max_start: Optional[int],
    avoid_head: int,
    avoid_tail: int,
    subject_key: str,
) -> Tuple[int, int]:
    row_start = _parse_frame_int(row.get("frame_start"))
    row_len = _parse_frame_int(row.get("frame_length"))
    length = row_len if row_len is not None else default_length
    if length <= 0:
        raise ValueError("frame length must be > 0")

    # Per-row explicit start overrides global mode.
    if row_start is not None:
        start = int(row_start)
    elif mode == "manual":
        if manual_start is None:
            raise ValueError(
                "manual mode requires --frame-start (or frame_start in manifest).")
        start = int(manual_start)
    else:
        low = max(int(random_min_start), int(avoid_head))
        max_by_length = frame_count - length
        max_by_tail = frame_count - length - int(avoid_tail)
        high = min(max_by_length, max_by_tail)
        if random_max_start is not None:
            high = min(high, int(random_max_start))
        if high < low:
            raise ValueError(
                f"No valid random window range: low={low}, high={high}, "
                f"frame_count={frame_count}, length={length}"
            )
        rng = random.Random(_deterministic_seed(random_seed, subject_key))
        start = rng.randint(low, high)

    end = start + length
    if start < 0 or end > frame_count:
        raise ValueError(
            f"Invalid window [{start}, {end}) for frame_count={frame_count}"
        )
    return start, end


def _offset_result_to_full_frames(
    result: SpiralDetectionResult,
    *,
    full_frame_count: int,
    frame_start: int,
) -> SpiralDetectionResult:
    h, w, tw = result.input_shape
    full_shape = (h, w, full_frame_count)

    if result.num_patterns == 0:
        return SpiralDetectionResult(
            patterns=[],
            num_patterns=0,
            labeled_volume=np.zeros(full_shape, dtype=int),
            input_shape=full_shape,
            detection_params=result.detection_params,
            statistics=result.statistics,
            rotation_direction=result.rotation_direction,
            curl_sign=result.curl_sign,
        )

    new_patterns: List[SpiralPattern] = []
    for p in result.patterns:
        ys, xs, ts = np.unravel_index(p.voxel_indices, result.input_shape)
        ts_full = ts + frame_start
        voxel_full = np.ravel_multi_index((ys, xs, ts_full), full_shape)
        bx0, bx1, by0, by1, bt0, bt1 = p.bounding_box

        new_patterns.append(
            replace(
                p,
                start_time=int(p.start_time + frame_start),
                end_time=int(p.end_time + frame_start),
                absolute_times=(p.absolute_times + frame_start).astype(int),
                voxel_indices=voxel_full,
                bounding_box=(bx0, bx1, by0, by1, bt0 +
                              frame_start, bt1 + frame_start),
            )
        )

    labeled = np.zeros(full_shape, dtype=int)
    for p in new_patterns:
        mask = np.unravel_index(p.voxel_indices, full_shape)
        labeled[mask] = p.pattern_id

    return SpiralDetectionResult(
        patterns=new_patterns,
        num_patterns=len(new_patterns),
        labeled_volume=labeled,
        input_shape=full_shape,
        detection_params=result.detection_params,
        statistics=result.statistics,
        rotation_direction=result.rotation_direction,
        curl_sign=result.curl_sign,
    )


@dataclass
class WindowedHemispherePrecompute:
    phase_cube_full: np.ndarray
    phase_cube: np.ndarray
    spatial_bandpass_window: np.ndarray
    raw_bandpass_window: np.ndarray
    valid_mask: np.ndarray
    phase_alignment_mask: Optional[np.ndarray]
    phase_field: object
    curl_field: np.ndarray
    full_frame_count: int
    surrogate_compatibility_thresholds: Optional[Dict[int, float]]


def precompute_hemisphere_windowed(
    *,
    hemisphere: str,
    config: MatPhaseConfig,
    sampling_rate_hz: float,
    show_progress: bool,
    phase_field_spacing: float,
    frame_start: int,
    frame_end: int,
) -> WindowedHemispherePrecompute:
    structure = "CORTEX_LEFT" if hemisphere == "left" else "CORTEX_RIGHT"
    data_root = Path(config.paths.data_dir)

    cifti_path_cfg = Path(config.paths.cifti_file)
    cifti_path = cifti_path_cfg if cifti_path_cfg.is_absolute() else (data_root /
                                                                      cifti_path_cfg)

    surface_path = data_root / (
        config.paths.surface_left if hemisphere == "left" else config.paths.surface_right
    )
    parcellation_path = data_root / (
        config.paths.parcellation_left if hemisphere == "left" else config.paths.parcellation_right
    )

    logger.info("Processing %s hemisphere", hemisphere.upper())
    logger.info("  CIFTI: %s", cifti_path)
    logger.info("  Window: [%d, %d)", frame_start, frame_end)

    cifti_ts = load_cifti(str(cifti_path))
    surface = load_surface(str(surface_path))

    parcellation_mask = None
    if parcellation_path.exists():
        parcellation = load_parcellation(str(parcellation_path))
        parcellation_mask = parcellation_to_mask(parcellation)

    coords = surface.vertices[:, :2]
    surface_data = cifti_ts.get_full_surface_data(structure)
    hemi_ranges = _hemisphere_ranges(config, hemisphere)

    grid = interpolate_to_grid_batch(
        signal=surface_data,
        positions=coords,
        faces=surface.faces if config.preprocessing.interpolation_method == "tri_linear" else None,
        x_range=hemi_ranges["x_range"],
        y_range=hemi_ranges["y_range"],
        downsample_rate=config.preprocessing.downsample_rate,
        method=config.preprocessing.interpolation_method,
        coordinate_system=config.preprocessing.interpolation_coordinate_system,
        parcellation_mask=parcellation_mask,
        n_jobs=1,
        show_progress=show_progress,
    )

    temporal = temporal_bandpass_filter(
        grid,
        sampling_rate=sampling_rate_hz,
        freq_low=config.preprocessing.filter_low_freq,
        freq_high=config.preprocessing.filter_high_freq,
        filter_order=config.preprocessing.filter_order,
        phase_method=config.preprocessing.phase_extraction_method,
        filter_range=config.preprocessing.gp_filter_range,
        phase_correction_threshold=config.preprocessing.gp_phase_correction_threshold,
        neg_freq_extension=config.preprocessing.gp_neg_freq_extension,
        return_inst_freq=config.preprocessing.gp_return_inst_freq,
        return_neg_freq_mask=config.preprocessing.gp_return_neg_freq_mask,
        demean=config.preprocessing.temporal_demean,
        show_progress=show_progress and config.preprocessing.show_temporal_progress,
    )

    spatial = spatial_bandpass_filter(
        temporal.bandpassed,
        sigma_scales=config.preprocessing.sigma_scales,
        downsample_rate=config.preprocessing.downsample_rate,
        mode=config.preprocessing.spatial_filter_mode,
        show_progress=show_progress and config.preprocessing.show_spatial_progress,
    )

    spatial_bandpass = spatial.bandpass[:, :, :, 0]
    valid_mask_full = np.isfinite(spatial_bandpass)

    phase_cube_full = _hilbert_phase_cube(
        spatial_bandpass, show_progress=show_progress)

    use_phase_alignment = (
        config.detection.phase_difference_threshold is not None
        and config.detection.use_surrogate_thresholds
    )
    raw_phase_cube_full = None
    if use_phase_alignment:
        raw_phase_cube_full = _hilbert_phase_cube(
            temporal.bandpassed, show_progress=show_progress)

    full_frame_count = int(phase_cube_full.shape[2])
    if frame_start < 0 or frame_end > full_frame_count or frame_start >= frame_end:
        raise ValueError(
            f"Invalid detection window [{frame_start}, {frame_end}) for frame_count={full_frame_count}"
        )

    sl = slice(frame_start, frame_end)
    phase_cube = phase_cube_full[:, :, sl]
    spatial_bandpass_window = spatial_bandpass[:, :, sl]
    raw_bandpass_window = temporal.bandpassed[:, :, sl]
    valid_mask = valid_mask_full[:, :, sl]

    phase_alignment_mask = None
    if use_phase_alignment and raw_phase_cube_full is not None:
        phase_alignment_mask = compute_phase_alignment_mask(
            raw_phase=raw_phase_cube_full[:, :, sl],
            smooth_phase=phase_cube,
            threshold=config.detection.phase_difference_threshold,
        )

    phase_field = compute_phase_field(
        phase_cube,
        spacing=phase_field_spacing,
        show_progress=show_progress,
    )
    curl_field = phase_field.curl

    surrogate_compatibility_thresholds: Optional[Dict[int, float]] = None
    if config.detection.use_surrogate_thresholds:
        surrogate_compatibility_thresholds = _compute_surrogate_compatibility_distributions(
            raw_bandpassed=raw_bandpass_window,
            spatial_bandpass=spatial_bandpass_window,
            config=config,
            phase_field_spacing=phase_field_spacing,
            show_progress=show_progress,
        )

    return WindowedHemispherePrecompute(
        phase_cube_full=phase_cube_full,
        phase_cube=phase_cube,
        spatial_bandpass_window=spatial_bandpass_window,
        raw_bandpass_window=raw_bandpass_window,
        valid_mask=valid_mask,
        phase_alignment_mask=phase_alignment_mask,
        phase_field=phase_field,
        curl_field=curl_field,
        full_frame_count=full_frame_count,
        surrogate_compatibility_thresholds=surrogate_compatibility_thresholds,
    )


def run_detection_on_precomputed(
    *,
    precomputed: WindowedHemispherePrecompute,
    config: MatPhaseConfig,
    show_progress: bool,
    frame_start: int,
    seed_method: str = "curl_threshold",
    optflow_focus_params: Optional[Dict[str, object]] = None,
) -> SpiralDetectionResult:
    phase_cube = precomputed.phase_cube
    spatial_bandpass_window = precomputed.spatial_bandpass_window
    valid_mask = precomputed.valid_mask
    phase_alignment_mask = precomputed.phase_alignment_mask
    phase_field = precomputed.phase_field
    curl_field = precomputed.curl_field
    surrogate_compatibility_thresholds = precomputed.surrogate_compatibility_thresholds
    raw_foci_rows: List[Dict[str, object]] = []

    thresholds = _resolve_detection_thresholds(
        spatial_bandpass_window,
        config,
        show_progress=show_progress,
    )
    curl_threshold = thresholds["curl"]
    expansion_threshold = thresholds.get("expansion")
    phase_coherence_threshold = thresholds.get("phase_coherence")

    if seed_method == "curl_threshold":
        expansion_threshold_filter = None if config.detection.enable_spiral_expansion else expansion_threshold

        filtered_curl, threshold_details = apply_combined_threshold(
            curl_field,
            curl_threshold=curl_threshold,
            phase_gradient_x=phase_field.gradient_x,
            phase_gradient_y=phase_field.gradient_y,
            expansion_threshold=expansion_threshold_filter,
            phase_coherence_threshold=phase_coherence_threshold,
            fill_value=0.0,
        )

        for name, result in threshold_details.items():
            logger.info(
                "%s threshold %.3f pass_fraction=%.4f",
                name.capitalize(),
                result.threshold_value,
                result.pass_fraction,
            )

        detection_result = detect_spirals_directional(
            filtered_curl,
            signal_amplitude=curl_field,
            curl_threshold=curl_threshold,
            rotation_mode=config.detection.rotation_mode,
            min_duration=config.detection.min_pattern_duration,
            min_size=config.detection.min_pattern_size,
            connectivity=config.detection.connectivity,
            use_weighted_centroids=config.detection.use_weighted_centroids,
            show_progress=show_progress,
        )
    elif seed_method == "optflow_focus":
        params = optflow_focus_params or {}
        focus_masks, raw_foci_rows = _build_optflow_focus_masks(
            phase_cube_full=precomputed.phase_cube_full,
            frame_start=frame_start,
            curl_field=curl_field,
            valid_mask=valid_mask,
            rotation_mode=config.detection.rotation_mode,
            alpha=float(params.get("alpha", 0.1)),
            beta=float(params.get("beta", 10.0)),
            max_iter=int(params.get("max_iter", 200)),
            tol=float(params.get("tol", 1e-4)),
            winding_radius=int(params.get("winding_radius", 2)),
            winding_min=float(params.get("winding_min", 0.8)),
            stable_only=bool(params.get("stable_only", True)),
            merge_distance=float(params.get("merge_distance", 2.0)),
            seed_patch_radius=int(params.get("seed_patch_radius", 1)),
        )
        detection_result = detect_spirals_from_masks(
            focus_masks,
            signal_amplitude=curl_field,
            min_duration=config.detection.min_pattern_duration,
            min_size=config.detection.min_pattern_size,
            connectivity=config.detection.connectivity,
            use_weighted_centroids=config.detection.use_weighted_centroids,
        )
    else:
        raise ValueError(f"Unsupported seed_method: {seed_method}")

    pattern_masks_for_metrics: Optional[Dict[int,
                                             Dict[int, np.ndarray]]] = None
    compatibility_pass_map: Optional[Dict[int, Dict[int, bool]]] = None
    final_result = detection_result

    if config.detection.enable_spiral_expansion and detection_result.num_patterns > 0:
        expanded_masks, pattern_mask_map, expansion_radii_map = expand_spiral_patterns(
            detection_result,
            phase_field.normalized_x,
            phase_field.normalized_y,
            valid_mask=valid_mask,
            phase_alignment_mask=None,
            angle_center=config.detection.angle_window_center,
            angle_half_width=config.detection.angle_window_half_width,
            expansion_threshold=(
                config.detection.expansion_threshold
                if config.detection.expansion_threshold is not None
                else 1.0
            ),
            radius_min=config.detection.expansion_radius_min,
            radius_max=config.detection.expansion_radius_max,
            radius_step=config.detection.expansion_radius_step,
            center_patch_radius=config.detection.center_patch_radius,
            show_progress=show_progress,
        )
        final_result = apply_expanded_masks_to_detection(
            detection_result,
            pattern_mask_map,
            amplitude_field=curl_field,
            expansion_radii=expansion_radii_map,
        )
        if pattern_mask_map:
            pattern_masks_for_metrics = pattern_mask_map

    if config.detection.use_surrogate_thresholds and final_result.num_patterns > 0:
        from matphase.detect.compatibility import (
            apply_compatibility_ratios_to_patterns,
            filter_patterns_by_compatibility,
        )

        updated_patterns = apply_compatibility_ratios_to_patterns(
            patterns=final_result.patterns,
            phase_field_vx=phase_field.normalized_x,
            phase_field_vy=phase_field.normalized_y,
            phase_alignment_mask=phase_alignment_mask,
            show_progress=show_progress,
        )

        labeled_volume = np.zeros(final_result.input_shape, dtype=int)
        for pattern in updated_patterns:
            mask = np.unravel_index(
                pattern.voxel_indices, final_result.input_shape)
            labeled_volume[mask] = pattern.pattern_id

        final_result = SpiralDetectionResult(
            patterns=updated_patterns,
            num_patterns=len(updated_patterns),
            labeled_volume=labeled_volume,
            input_shape=final_result.input_shape,
            detection_params=final_result.detection_params,
            statistics=final_result.statistics,
            rotation_direction=final_result.rotation_direction,
            curl_sign=final_result.curl_sign,
        )

        compatibility_ratio_map = _build_pattern_time_value_map(
            updated_patterns, "compatibility_ratios")
        expansion_radius_map = _build_pattern_time_value_map(
            updated_patterns, "expansion_radii")

        if surrogate_compatibility_thresholds and compatibility_ratio_map and expansion_radius_map:
            compatibility_pass_map = filter_patterns_by_compatibility(
                updated_patterns,
                compatibility_ratios=compatibility_ratio_map,
                expansion_radii=expansion_radius_map,
                surrogate_thresholds=surrogate_compatibility_thresholds,
                show_progress=show_progress,
            )
            if compatibility_pass_map:
                if pattern_masks_for_metrics:
                    pattern_masks_for_metrics, _, _ = _filter_pattern_masks_by_pass_map(
                        pattern_masks_for_metrics,
                        compatibility_pass_map,
                    )
                updated_patterns = _apply_pass_map_to_patterns(
                    updated_patterns, compatibility_pass_map)
                final_result = replace(
                    final_result, patterns=updated_patterns, num_patterns=len(updated_patterns))

    significance_metrics = _compute_significant_metrics(
        detection_result,
        pattern_masks_for_metrics,
        compatibility_pass_map,
    )
    final_result.statistics.update(significance_metrics)

    # Convert window-local detection time indices back to full-frame absolute indices.
    final_result = _offset_result_to_full_frames(
        final_result,
        full_frame_count=precomputed.full_frame_count,
        frame_start=frame_start,
    )
    if seed_method == "optflow_focus":
        final_result.raw_optflow_foci = raw_foci_rows

    return final_result


def run_pipeline_for_hemisphere_windowed(
    *,
    hemisphere: str,
    config: MatPhaseConfig,
    sampling_rate_hz: float,
    show_progress: bool,
    phase_field_spacing: float,
    frame_start: int,
    frame_end: int,
    phase_cube_output_path: Optional[Path],
    seed_method: str = "curl_threshold",
    optflow_focus_params: Optional[Dict[str, object]] = None,
) -> Tuple[SpiralDetectionResult, Optional[Dict[int, float]]]:
    precomputed = precompute_hemisphere_windowed(
        hemisphere=hemisphere,
        config=config,
        sampling_rate_hz=sampling_rate_hz,
        show_progress=show_progress,
        phase_field_spacing=phase_field_spacing,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    if config.output.save_phase_cube and phase_cube_output_path is not None:
        phase_cube_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(phase_cube_output_path, precomputed.phase_cube_full)

    result = run_detection_on_precomputed(
        precomputed=precomputed,
        config=config,
        show_progress=show_progress,
        frame_start=frame_start,
        seed_method=seed_method,
        optflow_focus_params=optflow_focus_params,
    )
    return result, precomputed.surrogate_compatibility_thresholds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch curl-threshold sweep runner with full preprocessing and windowed detection.",
    )
    parser.add_argument("--manifest", type=Path, required=False,
                        help="CSV/TSV/JSON with at least cifti_file")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/defaults.yaml"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument("--max-workers", type=int,
                        default=1, help="Concurrent subjects")

    parser.add_argument("--bundle-suffix", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--curl-threshold-series",
        nargs="+",
        type=float,
        required=False,
        help="Optional override curl thresholds to sweep; if omitted, uses DEFAULT_CURL_THRESHOLDS in this script.",
    )
    parser.add_argument(
        "--include-optflow-focus-baseline",
        action="store_true",
        help="Also run one extra detection branch seeded by optical-flow spiral foci.",
    )
    parser.add_argument("--optflow-alpha", type=float, default=0.1)
    parser.add_argument("--optflow-beta", type=float, default=10.0)
    parser.add_argument("--optflow-max-iter", type=int, default=200)
    parser.add_argument("--optflow-tol", type=float, default=1e-4)
    parser.add_argument(
        "--optflow-winding-radius",
        type=int,
        default=2,
        help="Square-ring radius in pixels for optical-flow Poincare index validation.",
    )
    parser.add_argument(
        "--optflow-winding-min",
        type=float,
        default=0.8,
        help="Minimum absolute winding number required for optical-flow focus seeds.",
    )
    parser.add_argument(
        "--optflow-merge-distance",
        type=float,
        default=2.0,
        help="Merge stable optical-flow foci within this grid-pixel distance.",
    )
    parser.add_argument("--optflow-seed-patch-radius", type=int, default=1)
    parser.add_argument(
        "--optflow-stable-only",
        action="store_true",
        default=True,
        help="Use only stable foci from optical-flow detector (default: enabled).",
    )
    parser.add_argument("--min-duration", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--rotation-mode",
                        choices=["both", "ccw", "cw"], default=None)
    parser.add_argument("--phase-field-spacing", type=float, default=1.0)
    parser.add_argument("--sampling-rate", type=float, default=None)
    parser.add_argument("--tr", type=float, default=None)
    parser.add_argument("--use-surrogate-thresholds", action="store_true")
    parser.add_argument("--surrogate-percentile", type=float, default=None)
    parser.add_argument("--n-surrogates-threshold", type=int, default=None)

    parser.add_argument(
        "--window-mode", choices=["manual", "random"], default="random")
    parser.add_argument("--frame-length", type=int, required=True)
    parser.add_argument("--frame-start", type=int,
                        default=None, help="Used in manual mode")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-min-start", type=int, default=0)
    parser.add_argument("--random-max-start", type=int, default=None)
    parser.add_argument("--avoid-head", type=int, default=0)
    parser.add_argument("--avoid-tail", type=int, default=0)

    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    # Internal worker mode (used by parent subprocess scheduler).
    parser.add_argument("--single-cifti-file", type=str, default=None)
    parser.add_argument("--single-subject-id", type=str, default=None)
    parser.add_argument(
        "--single-hemisphere", choices=["left", "right", "both"], default=None)
    parser.add_argument("--single-frame-start", type=int, default=None)
    parser.add_argument("--single-frame-length", type=int, default=None)
    return parser.parse_args()


def _apply_common_overrides(config: MatPhaseConfig, args: argparse.Namespace) -> None:
    if args.data_dir is not None:
        config.paths.data_dir = args.data_dir
    if args.output_dir is not None:
        config.paths.output_dir = args.output_dir

    if args.sampling_rate is not None and args.tr is not None:
        raise ValueError("Use either --sampling-rate or --tr, not both.")
    if args.sampling_rate is not None:
        if args.sampling_rate <= 0:
            raise ValueError("--sampling-rate must be positive")
        config.preprocessing.temporal_sampling_rate = args.sampling_rate
    if args.tr is not None:
        if args.tr <= 0:
            raise ValueError("--tr must be positive")
        config.preprocessing.temporal_sampling_rate = 1.0 / args.tr

    if args.min_duration is not None:
        config.detection.min_pattern_duration = args.min_duration
    if args.min_size is not None:
        config.detection.min_pattern_size = args.min_size
    if args.rotation_mode is not None:
        config.detection.rotation_mode = args.rotation_mode
    if args.use_surrogate_thresholds:
        config.detection.use_surrogate_thresholds = True
    if args.surrogate_percentile is not None:
        config.detection.surrogate_percentile = args.surrogate_percentile
    if args.n_surrogates_threshold is not None:
        config.detection.n_surrogates_threshold = args.n_surrogates_threshold


def _process_subject(
    args: argparse.Namespace,
    row: Dict[str, str],
    curl_thresholds: Sequence[float],
) -> List[Path]:
    config0 = load_config(args.config, apply_env=True)
    _apply_common_overrides(config0, args)

    data_root = Path(config0.paths.data_dir)
    cifti_path = _resolve_cifti_path(data_root, row["cifti_file"])
    if not cifti_path.exists():
        raise FileNotFoundError(f"CIFTI not found: {cifti_path}")

    hemi = row.get("hemisphere") or args.hemisphere
    hemispheres = ["left", "right"] if hemi == "both" else [hemi]
    if any(h not in {"left", "right"} for h in hemispheres):
        raise ValueError(f"Invalid hemisphere in row: {row.get('hemisphere')}")

    subject_key = _subject_tag(row, cifti_path)
    frame_count = _frame_count_from_cifti(cifti_path, hemispheres[0])
    frame_start, frame_end = _choose_window(
        frame_count=frame_count,
        row=row,
        mode=args.window_mode,
        default_length=args.frame_length,
        manual_start=args.frame_start,
        random_seed=args.random_seed,
        random_min_start=args.random_min_start,
        random_max_start=args.random_max_start,
        avoid_head=args.avoid_head,
        avoid_tail=args.avoid_tail,
        subject_key=subject_key,
    )

    logger.info(
        "Subject %s | CIFTI=%s | frames=%d | window=[%d,%d)",
        subject_key,
        cifti_path,
        frame_count,
        frame_start,
        frame_end,
    )

    subject_root = args.outdir / subject_key
    subject_root.mkdir(parents=True, exist_ok=True)

    preprocess_config = load_config(args.config, apply_env=True)
    _apply_common_overrides(preprocess_config, args)
    preprocess_config.paths.cifti_file = str(cifti_path)

    cifti_ts = load_cifti(str(cifti_path))
    metadata_sampling_rate = getattr(cifti_ts.metadata, "sampling_rate", None)
    if args.sampling_rate is not None or args.tr is not None:
        metadata_sampling_rate = None
    sampling_rate_hz = _resolve_sampling_rate(
        preprocess_config, metadata_sampling_rate)
    precomputed_by_hemi: Optional[Dict[str, WindowedHemispherePrecompute]] = None

    def _bundle_complete(bundle_dir: Path, seed_method: str) -> bool:
        required = [
            bundle_dir / "coords.feather",
            bundle_dir / "frame_index.parquet",
            bundle_dir / "metadata.json",
            bundle_dir / "patterns.parquet",
            bundle_dir / "phase_cube.npy",
        ]
        if seed_method == "optflow_focus":
            required.append(bundle_dir / "optflow_raw_foci.parquet")
        if not all(p.exists() for p in required):
            return False
        if seed_method == "optflow_focus":
            try:
                frame_index = pd.read_parquet(bundle_dir / "frame_index.parquet")
                raw_foci = pd.read_parquet(bundle_dir / "optflow_raw_foci.parquet")
            except Exception:
                return False
            if len(frame_index) > 0 and len(raw_foci) == 0:
                return False
        return True

    run_specs: List[Dict[str, object]] = [
        {
            "tag": _format_curl_tag(curl_threshold),
            "seed_method": "curl_threshold",
            "curl_threshold": float(curl_threshold),
        }
        for curl_threshold in curl_thresholds
    ]
    if args.include_optflow_focus_baseline:
        run_specs.append(
            {
                "tag": "optflow_focus",
                "seed_method": "optflow_focus",
                "curl_threshold": None,
            }
        )

    saved: List[Path] = []
    for run_spec in run_specs:
        config = load_config(args.config, apply_env=True)
        _apply_common_overrides(config, args)
        if run_spec["curl_threshold"] is not None:
            config.detection.curl_threshold = float(run_spec["curl_threshold"])
        config.paths.cifti_file = str(cifti_path)

        run_tag = str(run_spec["tag"])
        seed_method = str(run_spec["seed_method"])
        curl_root = subject_root / run_tag
        curl_root.mkdir(parents=True, exist_ok=True)

        # Skip this threshold run if all target hemisphere bundles are complete.
        target_bundle_dirs: Dict[str, Path] = {}
        for hemi_sel in hemispheres:
            suffix = args.bundle_suffix or hemi_sel
            bundle_name = _sanitize_bundle_name(cifti_path, suffix=suffix)
            target_bundle_dirs[hemi_sel] = curl_root / bundle_name
        if all(_bundle_complete(p, seed_method) for p in target_bundle_dirs.values()):
            logger.info(
                "Skip subject=%s curl=%s (existing complete bundles found).",
                subject_key,
                run_tag,
            )
            saved.extend(target_bundle_dirs.values())
            continue

        if precomputed_by_hemi is None:
            logger.info(
                "Precomputing shared phase/curl fields once for subject=%s.",
                subject_key,
            )
            precomputed_by_hemi = {}
            for hemi_sel in hemispheres:
                precomputed_by_hemi[hemi_sel] = precompute_hemisphere_windowed(
                    hemisphere=hemi_sel,
                    config=preprocess_config,
                    sampling_rate_hz=sampling_rate_hz,
                    show_progress=args.show_progress,
                    phase_field_spacing=args.phase_field_spacing,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )

        results: Dict[str, SpiralDetectionResult] = {}
        surrogate_by_hemi: Dict[str, Optional[Dict[int, float]]] = {}

        for hemi_sel in hemispheres:
            suffix = args.bundle_suffix or hemi_sel
            precomputed = precomputed_by_hemi[hemi_sel]
            phase_cube_path = (
                curl_root /
                _sanitize_bundle_name(
                    cifti_path, suffix=suffix) / config.output.phase_cube_filename
                if config.output.save_phase_cube
                else None
            )
            if phase_cube_path is not None:
                phase_cube_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(phase_cube_path, precomputed.phase_cube_full)

            res = run_detection_on_precomputed(
                precomputed=precomputed,
                config=config,
                show_progress=args.show_progress,
                frame_start=frame_start,
                seed_method=seed_method,
                optflow_focus_params={
                    "alpha": args.optflow_alpha,
                    "beta": args.optflow_beta,
                    "max_iter": args.optflow_max_iter,
                    "tol": args.optflow_tol,
                    "winding_radius": args.optflow_winding_radius,
                    "winding_min": args.optflow_winding_min,
                    "merge_distance": args.optflow_merge_distance,
                    "seed_patch_radius": args.optflow_seed_patch_radius,
                    "stable_only": args.optflow_stable_only,
                },
            )
            results[hemi_sel] = res
            surrogate_by_hemi[hemi_sel] = precomputed.surrogate_compatibility_thresholds

        for hemi_sel, detection_result in results.items():
            suffix = args.bundle_suffix or hemi_sel
            extra_metadata = {
                "hemisphere": hemi_sel,
                "rotation_mode": config.detection.rotation_mode,
                "curl_threshold": (
                    config.detection.curl_threshold
                    if seed_method == "curl_threshold"
                    else None
                ),
                "detection_seed_method": seed_method,
                "min_duration": config.detection.min_pattern_duration,
                "min_size": config.detection.min_pattern_size,
                "sigma_scales": list(config.preprocessing.sigma_scales),
                "detection_window_start": int(frame_start),
                "detection_window_end": int(frame_end),
                "detection_window_mode": args.window_mode,
            }
            if seed_method == "optflow_focus":
                extra_metadata["optflow_focus_params"] = {
                    "alpha": float(args.optflow_alpha),
                    "beta": float(args.optflow_beta),
                    "max_iter": int(args.optflow_max_iter),
                    "tol": float(args.optflow_tol),
                    "winding_radius": int(args.optflow_winding_radius),
                    "winding_min": float(args.optflow_winding_min),
                    "merge_distance": float(args.optflow_merge_distance),
                    "seed_patch_radius": int(args.optflow_seed_patch_radius),
                    "stable_only": bool(args.optflow_stable_only),
                }
            surrogate = surrogate_by_hemi.get(hemi_sel)
            if surrogate:
                extra_metadata["surrogate_compatibility_thresholds"] = surrogate
                extra_metadata["surrogate_percentile"] = config.detection.surrogate_percentile
                extra_metadata["n_surrogates_threshold"] = config.detection.n_surrogates_threshold

            bundle_dir = save_subject_bundle_from_detection(
                detection_result=detection_result,
                cifti_file=cifti_path,
                output_root=curl_root,
                extra_metadata=extra_metadata,
                processing_config={
                    "paths": config.paths.model_dump(),
                    "preprocessing": config.preprocessing.model_dump(),
                    "detection": config.detection.model_dump(),
                },
                suffix=suffix,
                overwrite=True,
                show_progress=args.show_progress,
            )
            if seed_method == "optflow_focus":
                raw_foci_rows = getattr(detection_result, "raw_optflow_foci", [])
                raw_foci_columns = [
                    "abs_time",
                    "window_frame",
                    "row",
                    "col",
                    "y",
                    "x",
                    "row_rounded",
                    "col_rounded",
                    "direction",
                    "curl_value_at_rounded",
                    "determinant",
                    "trace",
                    "stable",
                    "winding_number",
                ]
                raw_foci_df = pd.DataFrame(raw_foci_rows, columns=raw_foci_columns)
                raw_foci_df.to_parquet(bundle_dir / "optflow_raw_foci.parquet", index=False)
            saved.append(bundle_dir)

    return saved


def _build_subject_command(args: argparse.Namespace, row: Dict[str, str]) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--outdir",
        str(args.outdir),
        "--hemisphere",
        args.hemisphere,
        "--max-workers",
        "1",
        "--window-mode",
        args.window_mode,
        "--frame-length",
        str(args.frame_length),
        "--random-seed",
        str(args.random_seed),
        "--random-min-start",
        str(args.random_min_start),
        "--avoid-head",
        str(args.avoid_head),
        "--avoid-tail",
        str(args.avoid_tail),
        "--single-cifti-file",
        str(row["cifti_file"]),
    ]

    if args.bundle_suffix:
        cmd += ["--bundle-suffix", str(args.bundle_suffix)]
    if args.data_dir:
        cmd += ["--data-dir", str(args.data_dir)]
    if args.output_dir:
        cmd += ["--output-dir", str(args.output_dir)]
    if args.include_optflow_focus_baseline:
        cmd.append("--include-optflow-focus-baseline")
    cmd += ["--optflow-alpha", str(args.optflow_alpha)]
    cmd += ["--optflow-beta", str(args.optflow_beta)]
    cmd += ["--optflow-max-iter", str(args.optflow_max_iter)]
    cmd += ["--optflow-tol", str(args.optflow_tol)]
    cmd += ["--optflow-winding-radius", str(args.optflow_winding_radius)]
    cmd += ["--optflow-winding-min", str(args.optflow_winding_min)]
    cmd += ["--optflow-merge-distance", str(args.optflow_merge_distance)]
    cmd += ["--optflow-seed-patch-radius", str(args.optflow_seed_patch_radius)]
    if args.optflow_stable_only:
        cmd.append("--optflow-stable-only")
    if args.min_duration is not None:
        cmd += ["--min-duration", str(args.min_duration)]
    if args.min_size is not None:
        cmd += ["--min-size", str(args.min_size)]
    if args.rotation_mode:
        cmd += ["--rotation-mode", str(args.rotation_mode)]
    if args.phase_field_spacing is not None:
        cmd += ["--phase-field-spacing", str(args.phase_field_spacing)]
    if args.sampling_rate is not None:
        cmd += ["--sampling-rate", str(args.sampling_rate)]
    if args.tr is not None:
        cmd += ["--tr", str(args.tr)]
    if args.use_surrogate_thresholds:
        cmd.append("--use-surrogate-thresholds")
    if args.surrogate_percentile is not None:
        cmd += ["--surrogate-percentile", str(args.surrogate_percentile)]
    if args.n_surrogates_threshold is not None:
        cmd += ["--n-surrogates-threshold", str(args.n_surrogates_threshold)]
    if args.frame_start is not None:
        cmd += ["--frame-start", str(args.frame_start)]
    if args.random_max_start is not None:
        cmd += ["--random-max-start", str(args.random_max_start)]
    if args.show_progress:
        cmd.append("--show-progress")
    if args.log_file is not None:
        cmd += ["--log-file", str(args.log_file)]

    if args.curl_threshold_series:
        cmd += ["--curl-threshold-series", *[str(v) for v in args.curl_threshold_series]]

    if row.get("subject_id"):
        cmd += ["--single-subject-id", str(row["subject_id"])]
    row_hemi = row.get("hemisphere")
    if row_hemi:
        cmd += ["--single-hemisphere", str(row_hemi)]
    row_frame_start = row.get("frame_start")
    if row_frame_start is not None and str(row_frame_start).strip() != "":
        cmd += ["--single-frame-start", str(row_frame_start)]
    row_frame_length = row.get("frame_length")
    if row_frame_length is not None and str(row_frame_length).strip() != "":
        cmd += ["--single-frame-length", str(row_frame_length)]

    return cmd


def main() -> int:
    args = _parse_args()
    curl_thresholds = _parse_curl_thresholds(args.curl_threshold_series)

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file if args.log_file is not None else (
        args.outdir / "run_windowed_curl_batch.log")
    setup_logging(log_file=log_path)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    # Worker mode: process one subject in current process.
    if args.single_cifti_file:
        row: Dict[str, str] = {"cifti_file": args.single_cifti_file}
        if args.single_subject_id:
            row["subject_id"] = args.single_subject_id
        if args.single_hemisphere:
            row["hemisphere"] = args.single_hemisphere
        if args.single_frame_start is not None:
            row["frame_start"] = str(args.single_frame_start)
        if args.single_frame_length is not None:
            row["frame_length"] = str(args.single_frame_length)
        saved = _process_subject(args, row, curl_thresholds)
        logger.info("Worker done. Bundles processed: %s", ", ".join(str(p) for p in saved))
        return 0

    if args.manifest is None:
        raise ValueError("--manifest is required unless --single-cifti-file is used.")
    manifest = _load_manifest(args.manifest)
    if not manifest:
        raise ValueError("Manifest is empty")
    rows = list(_iter_rows(manifest))

    # Parent mode: subprocess-based parallelism (same style as run_detection_batch.py).
    cmds = [_build_subject_command(args, row) for row in rows]
    if args.max_workers == 1 or len(cmds) == 1:
        exit_codes = [_run_command(cmd) for cmd in cmds]
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_run_command, cmd) for cmd in cmds]
            exit_codes = [f.result() for f in as_completed(futures)]

    failed = sum(code != 0 for code in exit_codes)
    if failed:
        raise SystemExit(f"{failed} subject run(s) failed.")
    logger.info("Done. %d subject run(s) completed.", len(exit_codes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
