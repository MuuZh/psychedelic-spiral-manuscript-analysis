from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import convolve
from scipy import stats
from tqdm import tqdm

from .config import Config
from .utils import (
    build_group_summary_df,
    paired_t,
    plot_paired_violin,
    resolve_bundle_dir,
    save_fig,
    write_table,
)

TR_SECONDS_DEFAULT = 2.0
MIN_FRAMES_FOR_ANGULAR_VELOCITY_DEFAULT = 3

SOBEL_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def _estimate_omega_from_wavefront(
    center_x: float,
    center_y: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> float:
    if x_coords.size == 0 or y_coords.size == 0:
        return math.nan
    x_rel = x_coords.astype(float) - float(center_x)
    y_rel = y_coords.astype(float) - float(center_y)
    r = np.sqrt(x_rel ** 2 + y_rel ** 2)
    for radius in (3.0, 5.0):
        mask = r <= radius
        if int(mask.sum()) <= 2:
            continue
        theta = np.arctan2(y_rel[mask], x_rel[mask])
        return float(np.arctan2(np.sum(np.sin(theta)), np.sum(np.cos(theta))))
    return math.nan


def _precompute_wavefront_points(phase_cube: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if phase_cube.ndim != 3:
        return []
    points: list[tuple[np.ndarray, np.ndarray]] = []
    for t in range(phase_cube.shape[2]):
        phase_map = np.asarray(phase_cube[:, :, t], dtype=float)
        edge_x = convolve(phase_map, SOBEL_KX, mode="constant", cval=0.0)
        edge_y = convolve(phase_map, SOBEL_KY, mode="constant", cval=0.0)
        wavefront = np.sqrt(edge_x ** 2 + edge_y ** 2)
        y_coords, x_coords = np.where(wavefront > 2 * np.pi)
        points.append((x_coords.astype(np.int16, copy=False), y_coords.astype(np.int16, copy=False)))
    return points


def _select_xy(frame_df: pd.DataFrame) -> pd.DataFrame:
    out = frame_df.copy()
    out["x"] = pd.to_numeric(out["weighted_centroid_x"], errors="coerce")
    out["y"] = pd.to_numeric(out["weighted_centroid_y"], errors="coerce")
    missing = out["x"].isna() | out["y"].isna()
    if missing.any():
        out.loc[missing, "x"] = pd.to_numeric(out.loc[missing, "centroid_x"], errors="coerce")
        out.loc[missing, "y"] = pd.to_numeric(out.loc[missing, "centroid_y"], errors="coerce")
    out["abs_time"] = pd.to_numeric(out["abs_time"], errors="coerce")
    out = out.dropna(subset=["x", "y", "abs_time"])
    out["abs_time"] = out["abs_time"].astype(int)
    return out.sort_values("abs_time").reset_index(drop=True)


def _compute_phase_angular_velocity(
    frame_df: pd.DataFrame,
    wavefront_points: list[tuple[np.ndarray, np.ndarray]],
    rotation_direction: str,
    tr_seconds: float,
    min_frames_for_angular_velocity: int,
) -> dict[str, float]:
    if len(frame_df) < max(2, int(min_frames_for_angular_velocity)):
        return {
            "phase_angular_velocity_mean_rad_per_frame": math.nan,
            "phase_angular_velocity_abs_mean_rad_per_frame": math.nan,
            "phase_angular_velocity_mean_rad_per_second": math.nan,
            "phase_angular_velocity_abs_mean_rad_per_second": math.nan,
            "phase_omega_sample_count": 0,
        }

    omegas: list[float] = []
    omega_times: list[int] = []
    for row in frame_df.itertuples(index=False):
        t = int(row.abs_time)
        if t < 0 or t >= len(wavefront_points):
            continue
        x_coords, y_coords = wavefront_points[t]
        omega = _estimate_omega_from_wavefront(float(row.x), float(row.y), x_coords, y_coords)
        if np.isfinite(omega):
            omegas.append(omega)
            omega_times.append(t)

    if len(omegas) < 2:
        return {
            "phase_angular_velocity_mean_rad_per_frame": math.nan,
            "phase_angular_velocity_abs_mean_rad_per_frame": math.nan,
            "phase_angular_velocity_mean_rad_per_second": math.nan,
            "phase_angular_velocity_abs_mean_rad_per_second": math.nan,
            "phase_omega_sample_count": int(len(omegas)),
        }

    omegas_arr = np.asarray(omegas, dtype=float)
    times_arr = np.asarray(omega_times, dtype=float)
    dtheta = np.angle(np.exp(1j * np.diff(omegas_arr)))
    dt_frames = np.diff(times_arr)
    valid = np.isfinite(dtheta) & np.isfinite(dt_frames) & (dt_frames > 0)
    if not np.any(valid):
        return {
            "phase_angular_velocity_mean_rad_per_frame": math.nan,
            "phase_angular_velocity_abs_mean_rad_per_frame": math.nan,
            "phase_angular_velocity_mean_rad_per_second": math.nan,
            "phase_angular_velocity_abs_mean_rad_per_second": math.nan,
            "phase_omega_sample_count": int(len(omegas)),
        }

    vel_per_frame = dtheta[valid] / dt_frames[valid]
    vel_per_second = vel_per_frame / tr_seconds
    return {
        "phase_angular_velocity_mean_rad_per_frame": float(np.mean(vel_per_frame)),
        "phase_angular_velocity_abs_mean_rad_per_frame": float(np.mean(np.abs(vel_per_frame))),
        "phase_angular_velocity_mean_rad_per_second": float(np.mean(vel_per_second)),
        "phase_angular_velocity_abs_mean_rad_per_second": float(np.mean(np.abs(vel_per_second))),
        "phase_omega_sample_count": int(len(omegas)),
    }


def run_pattern_dynamics(cfg: Config, summary: List[Dict]) -> None:
    patterns_path = cfg.combined_dir / "combined_patterns.parquet"
    frames_path = cfg.combined_dir / "combined_frame_index.parquet"
    if not patterns_path.exists() or not frames_path.exists():
        logging.info("pattern_dynamics skipped: missing combined parquet inputs")
        return

    t0 = time.perf_counter()
    tr_seconds = float(getattr(cfg, "tr_seconds", TR_SECONDS_DEFAULT) or TR_SECONDS_DEFAULT)
    min_frames_for_angular_velocity = int(
        getattr(cfg, "min_frames_for_angular_velocity", MIN_FRAMES_FOR_ANGULAR_VELOCITY_DEFAULT)
        or MIN_FRAMES_FOR_ANGULAR_VELOCITY_DEFAULT
    )
    logging.info("pattern_dynamics: loading patterns from %s", patterns_path)
    patterns = pd.read_parquet(patterns_path)
    logging.info("pattern_dynamics: loading frames from %s", frames_path)
    frames = pd.read_parquet(frames_path)

    target_groups = {cfg.group_pcb, cfg.group_drug}
    patterns = patterns[patterns["group"].isin(target_groups)].copy()
    frames = frames[frames["group"].isin(target_groups)].copy()
    if patterns.empty or frames.empty:
        logging.info("pattern_dynamics skipped: no rows after group filtering")
        return
    logging.info(
        "pattern_dynamics: filtered rows patterns=%d frames=%d groups=%s",
        len(patterns),
        len(frames),
        ",".join(sorted(set(patterns["group"].astype(str)))),
    )

    pattern_keys = ["group", "subid", "hemisphere", "pattern_id", "bundle_dir"]
    frame_cols = pattern_keys + [
        "abs_time",
        "centroid_x",
        "centroid_y",
        "weighted_centroid_x",
        "weighted_centroid_y",
    ]
    missing_cols = [col for col in frame_cols if col not in frames.columns]
    if missing_cols:
        logging.warning("pattern_dynamics skipped: missing frame columns %s", missing_cols)
        return

    frames = frames[frame_cols].copy()
    logging.info("pattern_dynamics: merging pattern rows with frame rows")
    merged = patterns.merge(
        frames,
        on=pattern_keys,
        how="left",
        suffixes=("", "_frame"),
    )
    if merged.empty:
        logging.info("pattern_dynamics skipped: merged table is empty")
        return
    logging.info("pattern_dynamics: merged rows=%d", len(merged))

    out_dir = cfg.output_root / cfg.results_prefix / "pattern_dynamics"
    bundle_wavefront_cache: dict[Path, list[tuple[np.ndarray, np.ndarray]]] = {}
    pattern_records: list[dict[str, object]] = []

    grouped = merged.groupby(pattern_keys + ["rotation_direction"], dropna=False, sort=False)
    total_patterns = grouped.ngroups
    logging.info(
        "pattern_dynamics: computing %d patterns (min_frames_for_angular_velocity=%d)",
        total_patterns,
        min_frames_for_angular_velocity,
    )
    angle_velocity_count = 0
    wavefront_bundle_count = 0
    for idx, (keys, pattern_df) in enumerate(
        tqdm(grouped, total=total_patterns, desc="pattern dynamics", file=sys.stdout, dynamic_ncols=True),
        start=1,
    ):
        group, subid, hemisphere, pattern_id, bundle_dir_raw, rotation_direction = keys
        frame_df = _select_xy(pattern_df)
        duration_series = pd.to_numeric(pattern_df.get("duration"), errors="coerce").dropna()
        duration_val = float(duration_series.iloc[0]) if not duration_series.empty else math.nan
        row: dict[str, object] = {
            "group": group,
            "subid": str(subid),
            "hemisphere": hemisphere,
            "pattern_id": int(pattern_id) if pd.notna(pattern_id) else pattern_id,
            "bundle_dir": bundle_dir_raw,
            "rotation_direction": rotation_direction,
            "frame_count": int(len(frame_df)),
            "duration": duration_val,
        }

        bundle_row = pd.Series(
            {
                "group": group,
                "subid": subid,
                "hemisphere": hemisphere,
                "bundle_dir": bundle_dir_raw,
            }
        )
        bundle_dir = resolve_bundle_dir(bundle_row, cfg)
        if bundle_dir and bundle_dir.exists():
            wavefront_points = bundle_wavefront_cache.get(bundle_dir)
            if wavefront_points is None:
                cube_path = bundle_dir / "phase_cube.npy"
                if cube_path.exists():
                    logging.info("pattern_dynamics: precomputing wavefronts for bundle %s", bundle_dir)
                    cube = np.load(cube_path, mmap_mode="r")
                    wavefront_points = _precompute_wavefront_points(cube)
                    bundle_wavefront_cache[bundle_dir] = wavefront_points
                    wavefront_bundle_count += 1
            if wavefront_points is not None:
                ang_vel = _compute_phase_angular_velocity(
                    frame_df=frame_df,
                    wavefront_points=wavefront_points,
                    rotation_direction=str(rotation_direction),
                    tr_seconds=tr_seconds,
                    min_frames_for_angular_velocity=min_frames_for_angular_velocity,
                )
                row.update(ang_vel)
                if int(ang_vel.get("phase_omega_sample_count", 0)) > 1:
                    angle_velocity_count += 1
            else:
                row.update(
                    _compute_phase_angular_velocity(
                        frame_df.iloc[0:0],
                        [],
                        "",
                        tr_seconds,
                        min_frames_for_angular_velocity,
                    )
                )
        else:
            row.update(
                _compute_phase_angular_velocity(
                    frame_df.iloc[0:0],
                    [],
                    "",
                    tr_seconds,
                    min_frames_for_angular_velocity,
                )
            )

        pattern_records.append(row)
    if not pattern_records:
        logging.info("pattern_dynamics finished with no pattern records")
        return

    pattern_df = pd.DataFrame(pattern_records)
    write_table(pattern_df, out_dir / "per_pattern.csv")

    metric_cols = [
        "phase_angular_velocity_mean_rad_per_frame",
        "phase_angular_velocity_abs_mean_rad_per_frame",
        "phase_angular_velocity_mean_rad_per_second",
        "phase_angular_velocity_abs_mean_rad_per_second",
    ]
    subj_df = (
        pattern_df.groupby(["group", "subid", "hemisphere"], as_index=False)[metric_cols]
        .mean()
    )
    write_table(subj_df, out_dir / "per_subject.csv")

    group_summary_df = build_group_summary_df(subj_df, metric_cols, cfg)
    for hemi in ["left", "right"]:
        hemi_df = subj_df[subj_df["hemisphere"] == hemi]
        for metric in metric_cols:
            drug = hemi_df[hemi_df["group"] == cfg.group_drug].set_index("subid")[metric]
            pcb = hemi_df[hemi_df["group"] == cfg.group_pcb].set_index("subid")[metric]
            paired = drug.index.intersection(pcb.index)
            if len(paired) > 0:
                pair_stats = paired_t(drug.loc[paired], pcb.loc[paired])
                summary.append(
                    {
                        "section": "pattern_dynamics",
                        "metric": metric,
                        "hemisphere": hemi,
                        "comparison": "paired_drug_vs_pcb",
                        "n": int(pair_stats.get("n", len(paired))),
                        "t": float(pair_stats.get("t", math.nan)),
                        "p": float(pair_stats.get("p", math.nan)),
                        "dz": float(pair_stats.get("dz", math.nan)),
                    }
                )
            else:
                summary.append(
                    {
                        "section": "pattern_dynamics",
                        "metric": metric,
                        "hemisphere": hemi,
                        "comparison": "paired_drug_vs_pcb",
                        "n": 0,
                        "t": math.nan,
                        "p": math.nan,
                        "dz": math.nan,
                    }
                )
            unpaired = stats.ttest_ind(drug.dropna(), pcb.dropna(), equal_var=False, nan_policy="omit")
            summary.append(
                {
                    "section": "pattern_dynamics",
                    "metric": metric,
                    "hemisphere": hemi,
                    "comparison": "unpaired_drug_vs_pcb",
                    "n1": int(drug.dropna().shape[0]),
                    "n2": int(pcb.dropna().shape[0]),
                    "t": float(unpaired.statistic) if np.isfinite(unpaired.statistic) else math.nan,
                    "p": float(unpaired.pvalue) if np.isfinite(unpaired.pvalue) else math.nan,
                }
            )

    write_table(group_summary_df, out_dir / "group_summary.csv")
    logging.info(
        "pattern_dynamics: wrote per_pattern=%d per_subject=%d in %.1fs",
        len(pattern_df),
        len(subj_df),
        time.perf_counter() - t0,
    )
    logging.info(
        "pattern_dynamics: angular_velocity_computed=%d wavefront_bundles=%d",
        angle_velocity_count,
        wavefront_bundle_count,
    )

    sns.set_theme(style="whitegrid", context="talk")
    plot_metrics = [
        "phase_angular_velocity_abs_mean_rad_per_frame",
        "phase_angular_velocity_abs_mean_rad_per_second",
    ]
    for hemi in ["left", "right"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes_flat = np.ravel(axes)
        for ax, metric in zip(axes_flat, plot_metrics):
            tidy = subj_df[subj_df["hemisphere"] == hemi][["subid", "group", metric]]
            sns.violinplot(
                data=tidy,
                x="group",
                y=metric,
                order=[cfg.group_pcb, cfg.group_drug],
                ax=ax,
                cut=0,
            )
            ax.set_title(f"{metric} ({hemi})")
        for ax in axes_flat[len(plot_metrics):]:
            ax.axis("off")
        fig.suptitle(f"Pattern dynamics ({hemi})")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, out_dir / f"violin_{hemi}.png", cfg.save_plots)
        for metric in plot_metrics:
            tidy = subj_df[subj_df["hemisphere"] == hemi][["subid", "group", metric]]
            plot_paired_violin(
                tidy,
                metric,
                hemi,
                f"Pattern dynamics {metric} paired",
                out_dir / f"paired_{metric}_{hemi}.png",
                cfg,
            )
