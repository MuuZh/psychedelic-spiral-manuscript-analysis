#!/usr/bin/env python
"""
Compare curl-threshold=1.0 vs optflow_focus using spiral centers only.

Matching rule:
- same subject
- same hemisphere
- same absolute frame
- center distance <= match_radius

Outputs:
- frame_summary.csv
- subject_summary.csv
- overall_summary.csv
- example plots with center markers only
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from matphase.detect.phase_field import compute_phase_field

matplotlib.use("Agg")


GROUP_SUB_RE = re.compile(r"^[A-Za-z]+_(DMT|PCB)_S(\d+)", re.IGNORECASE)


@dataclass
class BundlePair:
    subject_dir: Path
    subject_name: str
    group: Optional[str]
    subid: Optional[str]
    hemisphere: str
    curl_bundle: Path
    optflow_bundle: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare curl_1 and optflow_focus by spiral centers only.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/curl_threshold_sweep"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs") / "curl1_vs_optflow_center_overlap",
    )
    parser.add_argument("--curl-tag", type=str, default="curl_1")
    parser.add_argument("--optflow-tag", type=str, default="optflow_focus")
    parser.add_argument(
        "--optflow-center-source",
        choices=["frame_index", "raw_foci"],
        default="frame_index",
        help=(
            "Use saved optflow pattern centroids from frame_index.parquet, "
            "or raw optical-flow foci from optflow_raw_foci.parquet."
        ),
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="left")
    parser.add_argument("--groups", nargs="*", default=["DMT", "PCB"])
    parser.add_argument(
        "--match-radius",
        type=float,
        default=3.0,
        help="Center-distance tolerance in grid pixels.",
    )
    parser.add_argument("--max-example-frames", type=int, default=6)
    parser.add_argument("--phase-field-spacing", type=float, default=1.0)
    parser.add_argument("--quiver-step", type=int, default=8)
    parser.add_argument("--phase-cmap", type=str, default="RdYlBu_r")
    return parser.parse_args()


def _parse_group_subid(subject_dir_name: str, cifti_file: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    match = GROUP_SUB_RE.search(subject_dir_name)
    if match:
        return match.group(1).upper(), str(int(match.group(2)))
    if cifti_file:
        alt = GROUP_SUB_RE.search(Path(cifti_file).stem.replace(".", "_"))
        if alt:
            return alt.group(1).upper(), str(int(alt.group(2)))
    return None, None


def _bundle_hemi(bundle_dir: Path) -> Optional[str]:
    meta_path = bundle_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    hemi = metadata.get("extra_metadata", {}).get("hemisphere")
    if hemi in {"left", "right"}:
        return hemi
    low = bundle_dir.name.lower()
    if low.endswith("_left"):
        return "left"
    if low.endswith("_right"):
        return "right"
    return None


def _iter_bundle_pairs(
    input_root: Path,
    curl_tag: str,
    optflow_tag: str,
    hemisphere: str,
    groups: Sequence[str],
) -> Iterable[BundlePair]:
    groups_norm = {g.upper() for g in groups}
    for subject_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        curl_root = subject_dir / curl_tag
        opt_root = subject_dir / optflow_tag
        if not (curl_root.exists() and opt_root.exists()):
            continue

        opt_bundles = {p.name: p for p in opt_root.iterdir() if p.is_dir()}
        for curl_bundle in sorted(p for p in curl_root.iterdir() if p.is_dir()):
            hemi = _bundle_hemi(curl_bundle)
            if hemi is None:
                continue
            if hemisphere != "both" and hemi != hemisphere:
                continue

            opt_bundle = opt_bundles.get(curl_bundle.name)
            if opt_bundle is None:
                continue

            meta = json.loads((curl_bundle / "metadata.json").read_text(encoding="utf-8"))
            group, subid = _parse_group_subid(subject_dir.name, meta.get("cifti_file"))
            if groups_norm and group is not None and group.upper() not in groups_norm:
                continue

            yield BundlePair(
                subject_dir=subject_dir,
                subject_name=subject_dir.name,
                group=group,
                subid=subid,
                hemisphere=hemi,
                curl_bundle=curl_bundle,
                optflow_bundle=opt_bundle,
            )


def _load_centers(bundle_dir: Path) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict]:
    frame_index = pd.read_parquet(bundle_dir / "frame_index.parquet")
    phase_cube = np.load(bundle_dir / "phase_cube.npy")
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))

    if "weighted_centroid_x" in frame_index.columns and "weighted_centroid_y" in frame_index.columns:
        x_col = "weighted_centroid_x"
        y_col = "weighted_centroid_y"
    else:
        x_col = "centroid_x"
        y_col = "centroid_y"

    centers: Dict[int, List[List[float]]] = {}
    for row in frame_index.itertuples(index=False):
        t = int(row.abs_time)
        x = float(getattr(row, x_col))
        y = float(getattr(row, y_col))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        centers.setdefault(t, []).append([x, y])

    center_arrays = {
        frame: np.asarray(points, dtype=float)
        for frame, points in centers.items()
    }
    return center_arrays, phase_cube, metadata


def _load_optflow_centers(bundle_dir: Path, source: str) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict]:
    if source == "frame_index":
        return _load_centers(bundle_dir)
    if source != "raw_foci":
        raise ValueError(f"Unsupported optflow center source: {source}")

    raw_path = bundle_dir / "optflow_raw_foci.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw optflow foci table: {raw_path}. "
            "Re-run run_detection_batch_curl_window.py with --include-optflow-focus-baseline "
            "after the raw-foci saving change."
        )

    raw_foci = pd.read_parquet(raw_path)
    phase_cube = np.load(bundle_dir / "phase_cube.npy")
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))

    required = {"abs_time", "col", "row"}
    missing = required.difference(raw_foci.columns)
    if missing:
        raise ValueError(f"{raw_path} missing required columns: {sorted(missing)}")

    centers: Dict[int, List[List[float]]] = {}
    for row in raw_foci.itertuples(index=False):
        t = int(getattr(row, "abs_time"))
        x = float(getattr(row, "col"))
        y = float(getattr(row, "row"))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        centers.setdefault(t, []).append([x, y])

    center_arrays = {
        frame: np.asarray(points, dtype=float)
        for frame, points in centers.items()
    }
    return center_arrays, phase_cube, metadata


def _safe_ratio(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _match_flags(source: np.ndarray, target: np.ndarray, radius: float) -> np.ndarray:
    if source.size == 0:
        return np.zeros(0, dtype=bool)
    if target.size == 0:
        return np.zeros(len(source), dtype=bool)
    diff = source[:, None, :] - target[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.any(dist2 <= (radius * radius), axis=1)


def _summarize_pair(
    pair: BundlePair,
    match_radius: float,
    phase_field_spacing: float,
    optflow_center_source: str,
) -> Tuple[List[Dict], Dict[int, Dict]]:
    curl_centers, curl_phase, curl_meta = _load_centers(pair.curl_bundle)
    opt_centers, opt_phase, _ = _load_optflow_centers(pair.optflow_bundle, optflow_center_source)
    if curl_phase.shape != opt_phase.shape:
        raise ValueError(f"Phase cube shape mismatch: {pair.curl_bundle} vs {pair.optflow_bundle}")
    phase_field = compute_phase_field(curl_phase, spacing=phase_field_spacing, compute_curl=False, show_progress=False)

    frames = sorted(set(curl_centers) | set(opt_centers))
    rows: List[Dict] = []
    examples: Dict[int, Dict] = {}
    for frame in frames:
        curl_pts = curl_centers.get(frame, np.empty((0, 2), dtype=float))
        opt_pts = opt_centers.get(frame, np.empty((0, 2), dtype=float))

        curl_matched = _match_flags(curl_pts, opt_pts, match_radius)
        opt_matched = _match_flags(opt_pts, curl_pts, match_radius)

        curl_count = int(len(curl_pts))
        opt_count = int(len(opt_pts))
        curl_only_count = int((~curl_matched).sum())
        optflow_only_count = int((~opt_matched).sum())
        matched_curl_count = int(curl_matched.sum())
        matched_optflow_count = int(opt_matched.sum())

        rows.append(
            {
                "subject": pair.subject_name,
                "group": pair.group,
                "subid": pair.subid,
                "hemisphere": pair.hemisphere,
                "frame": int(frame),
                "window_start": curl_meta.get("extra_metadata", {}).get("detection_window_start"),
                "window_end": curl_meta.get("extra_metadata", {}).get("detection_window_end"),
                "match_radius": float(match_radius),
                "optflow_center_source": optflow_center_source,
                "curl_center_count": curl_count,
                "optflow_center_count": opt_count,
                "curl_center_count_per_frame": float(curl_count),
                "optflow_center_count_per_frame": float(opt_count),
                "curl_only_center_count": curl_only_count,
                "optflow_only_center_count": optflow_only_count,
                "matched_curl_center_count": matched_curl_count,
                "matched_optflow_center_count": matched_optflow_count,
                "curl_only_ratio_in_curl": _safe_ratio(curl_only_count, curl_count),
                "optflow_only_ratio_in_optflow": _safe_ratio(optflow_only_count, opt_count),
                "matched_curl_ratio": _safe_ratio(matched_curl_count, curl_count),
                "matched_optflow_ratio": _safe_ratio(matched_optflow_count, opt_count),
            }
        )

        examples[frame] = {
            "phase": curl_phase[:, :, frame],
            "vx": np.asarray(phase_field.normalized_x[:, :, frame], dtype=float),
            "vy": np.asarray(phase_field.normalized_y[:, :, frame], dtype=float),
            "curl_pts": curl_pts,
            "opt_pts": opt_pts,
            "curl_matched": curl_matched,
            "opt_matched": opt_matched,
            "optflow_center_source": optflow_center_source,
        }

    return rows, examples


def _aggregate_subject(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for keys, sub in frame_df.groupby(["subject", "group", "subid", "hemisphere"], dropna=False):
        curl_count = int(sub["curl_center_count"].sum())
        opt_count = int(sub["optflow_center_count"].sum())
        curl_only_count = int(sub["curl_only_center_count"].sum())
        optflow_only_count = int(sub["optflow_only_center_count"].sum())
        matched_curl_count = int(sub["matched_curl_center_count"].sum())
        matched_optflow_count = int(sub["matched_optflow_center_count"].sum())
        rows.append(
            {
                "subject": keys[0],
                "group": keys[1],
                "subid": keys[2],
                "hemisphere": keys[3],
                "n_frames_compared": int(len(sub)),
                "match_radius": float(sub["match_radius"].iloc[0]),
                "optflow_center_source": str(sub["optflow_center_source"].iloc[0]),
                "curl_center_count": curl_count,
                "optflow_center_count": opt_count,
                "curl_center_count_per_frame": _safe_ratio(curl_count, int(len(sub))),
                "optflow_center_count_per_frame": _safe_ratio(opt_count, int(len(sub))),
                "curl_only_center_count": curl_only_count,
                "optflow_only_center_count": optflow_only_count,
                "matched_curl_center_count": matched_curl_count,
                "matched_optflow_center_count": matched_optflow_count,
                "curl_only_ratio_in_curl": _safe_ratio(curl_only_count, curl_count),
                "optflow_only_ratio_in_optflow": _safe_ratio(optflow_only_count, opt_count),
                "matched_curl_ratio": _safe_ratio(matched_curl_count, curl_count),
                "matched_optflow_ratio": _safe_ratio(matched_optflow_count, opt_count),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_overall(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for (group, hemisphere), sub in frame_df.groupby(["group", "hemisphere"], dropna=False):
        curl_count = int(sub["curl_center_count"].sum())
        opt_count = int(sub["optflow_center_count"].sum())
        curl_only_count = int(sub["curl_only_center_count"].sum())
        optflow_only_count = int(sub["optflow_only_center_count"].sum())
        matched_curl_count = int(sub["matched_curl_center_count"].sum())
        matched_optflow_count = int(sub["matched_optflow_center_count"].sum())
        rows.append(
            {
                "group": group,
                "hemisphere": hemisphere,
                "n_subjects": int(sub["subject"].nunique()),
                "n_frames_compared": int(len(sub)),
                "match_radius": float(sub["match_radius"].iloc[0]),
                "optflow_center_source": str(sub["optflow_center_source"].iloc[0]),
                "curl_center_count": curl_count,
                "optflow_center_count": opt_count,
                "curl_only_center_count": curl_only_count,
                "optflow_only_center_count": optflow_only_count,
                "matched_curl_center_count": matched_curl_count,
                "matched_optflow_center_count": matched_optflow_count,
                "curl_only_ratio_in_curl": _safe_ratio(curl_only_count, curl_count),
                "optflow_only_ratio_in_optflow": _safe_ratio(optflow_only_count, opt_count),
                "matched_curl_ratio": _safe_ratio(matched_curl_count, curl_count),
                "matched_optflow_ratio": _safe_ratio(matched_optflow_count, opt_count),
            }
        )

    rows.append(
        {
            "group": "ALL",
            "hemisphere": "ALL",
            "n_subjects": int(frame_df["subject"].nunique()),
            "n_frames_compared": int(len(frame_df)),
            "match_radius": float(frame_df["match_radius"].iloc[0]),
            "optflow_center_source": str(frame_df["optflow_center_source"].iloc[0]),
            "curl_center_count": int(frame_df["curl_center_count"].sum()),
            "optflow_center_count": int(frame_df["optflow_center_count"].sum()),
            "curl_center_count_per_frame": _safe_ratio(
                int(frame_df["curl_center_count"].sum()),
                int(len(frame_df)),
            ),
            "optflow_center_count_per_frame": _safe_ratio(
                int(frame_df["optflow_center_count"].sum()),
                int(len(frame_df)),
            ),
            "curl_only_center_count": int(frame_df["curl_only_center_count"].sum()),
            "optflow_only_center_count": int(frame_df["optflow_only_center_count"].sum()),
            "matched_curl_center_count": int(frame_df["matched_curl_center_count"].sum()),
            "matched_optflow_center_count": int(frame_df["matched_optflow_center_count"].sum()),
            "curl_only_ratio_in_curl": _safe_ratio(
                int(frame_df["curl_only_center_count"].sum()),
                int(frame_df["curl_center_count"].sum()),
            ),
            "optflow_only_ratio_in_optflow": _safe_ratio(
                int(frame_df["optflow_only_center_count"].sum()),
                int(frame_df["optflow_center_count"].sum()),
            ),
            "matched_curl_ratio": _safe_ratio(
                int(frame_df["matched_curl_center_count"].sum()),
                int(frame_df["curl_center_count"].sum()),
            ),
            "matched_optflow_ratio": _safe_ratio(
                int(frame_df["matched_optflow_center_count"].sum()),
                int(frame_df["optflow_center_count"].sum()),
            ),
        }
    )
    return pd.DataFrame(rows)


PAIRED_METRICS = [
    ("center_count", "curl_center_count", "optflow_center_count"),
    ("center_count_per_frame", "curl_center_count_per_frame", "optflow_center_count_per_frame"),
    ("only_center_count", "curl_only_center_count", "optflow_only_center_count"),
    ("matched_center_count", "matched_curl_center_count", "matched_optflow_center_count"),
    ("matched_ratio", "matched_curl_ratio", "matched_optflow_ratio"),
]

RATIO_METRICS = [
    "curl_only_ratio_in_curl",
    "optflow_only_ratio_in_optflow",
    "matched_curl_ratio",
    "matched_optflow_ratio",
]


def _series_stats(values: pd.Series) -> Dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(len(arr))
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": float(arr.mean()),
        "std": std,
        "sem": float(std / math.sqrt(n)) if n > 1 else float("nan"),
        "median": float(arr.median()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _mean_ci95(mean: float, sem: float) -> Tuple[float, float]:
    if not (math.isfinite(mean) and math.isfinite(sem)):
        return float("nan"), float("nan")
    delta = 1.96 * sem
    return float(mean - delta), float(mean + delta)


def _ratio_error_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    group_slices: List[Tuple[str, pd.DataFrame]] = []

    groups = subject_df["group"].dropna().astype(str).str.upper()
    for group in sorted(groups.unique()):
        group_slices.append((group, subject_df[subject_df["group"].astype(str).str.upper() == group]))
    group_slices.append(
        ("DMT+PCB", subject_df[subject_df["group"].astype(str).str.upper().isin(["DMT", "PCB"])])
    )
    group_slices.append(("ALL", subject_df))

    hemispheres = [h for h in sorted(subject_df["hemisphere"].dropna().astype(str).unique())]
    hemisphere_slices = [(hemi, subject_df[subject_df["hemisphere"].astype(str) == hemi]) for hemi in hemispheres]
    hemisphere_slices.append(("ALL", subject_df))

    for group_name, group_df in group_slices:
        if group_df.empty:
            continue
        for hemisphere, hemi_df in hemisphere_slices:
            sub = group_df if hemisphere == "ALL" else group_df[group_df["hemisphere"].astype(str) == hemisphere]
            if sub.empty:
                continue
            for metric in RATIO_METRICS:
                stats_row = _series_stats(sub[metric])
                ci95_lo, ci95_hi = _mean_ci95(stats_row["mean"], stats_row["sem"])
                rows.append(
                    {
                        "group": group_name,
                        "hemisphere": hemisphere,
                        "metric": metric,
                        "paired_unit": "subject_hemisphere",
                        "n_subjects": int(sub["subject"].nunique()),
                        "n_values": stats_row["n"],
                        "mean_ratio": stats_row["mean"],
                        "std_ratio": stats_row["std"],
                        "sem_ratio": stats_row["sem"],
                        "ci95_lo": ci95_lo,
                        "ci95_hi": ci95_hi,
                        "median_ratio": stats_row["median"],
                        "min_ratio": stats_row["min"],
                        "max_ratio": stats_row["max"],
                    }
                )

    return pd.DataFrame(rows)


def _paired_ttest(curl: pd.Series, optflow: pd.Series) -> Tuple[int, float, float]:
    paired = pd.DataFrame({"curl": curl, "optflow": optflow}).dropna()
    if len(paired) < 2:
        return int(len(paired)), float("nan"), float("nan")
    diff = paired["curl"].astype(float) - paired["optflow"].astype(float)
    if np.allclose(diff, 0.0, equal_nan=False):
        return int(len(paired)), 0.0, 1.0
    result = stats.ttest_rel(paired["curl"], paired["optflow"], nan_policy="omit")
    return int(len(paired)), float(result.statistic), float(result.pvalue)


def _paired_effect_size(diff_stats: Dict[str, float], n_pairs: int, t_stat: float) -> Tuple[float, float]:
    diff_std = diff_stats["std"]
    if math.isfinite(diff_std) and diff_std != 0:
        cohen_dz = float(diff_stats["mean"] / diff_std)
    elif math.isfinite(diff_stats["mean"]) and diff_stats["mean"] == 0:
        cohen_dz = 0.0
    else:
        cohen_dz = float("nan")

    if n_pairs > 0 and math.isfinite(t_stat):
        t_based_dz = float(t_stat / math.sqrt(n_pairs))
    else:
        t_based_dz = float("nan")
    return cohen_dz, t_based_dz


def _comparison_slices(subject_df: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
    groups = subject_df["group"].dropna().astype(str).str.upper()
    for group in sorted(groups.unique()):
        sub = subject_df[subject_df["group"].astype(str).str.upper() == group]
        if not sub.empty:
            yield group, sub
    yield "DMT+PCB", subject_df[subject_df["group"].astype(str).str.upper().isin(["DMT", "PCB"])]


def _paired_metric_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    hemispheres = [h for h in sorted(subject_df["hemisphere"].dropna().astype(str).unique())]
    hemisphere_slices = [(hemi, subject_df[subject_df["hemisphere"].astype(str) == hemi]) for hemi in hemispheres]
    hemisphere_slices.append(("ALL", subject_df))

    for hemisphere, hemi_df in hemisphere_slices:
        if hemi_df.empty:
            continue
        for comparison_group, group_df in _comparison_slices(hemi_df):
            if group_df.empty:
                continue
            for metric, curl_col, optflow_col in PAIRED_METRICS:
                curl_stats = _series_stats(group_df[curl_col])
                optflow_stats = _series_stats(group_df[optflow_col])
                diff = pd.to_numeric(group_df[curl_col], errors="coerce") - pd.to_numeric(
                    group_df[optflow_col], errors="coerce"
                )
                diff_stats = _series_stats(diff)
                n_pairs, t_stat, p_value = _paired_ttest(group_df[curl_col], group_df[optflow_col])
                cohen_dz, t_based_dz = _paired_effect_size(diff_stats, n_pairs, t_stat)

                rows.append(
                    {
                        "comparison_group": comparison_group,
                        "hemisphere": hemisphere,
                        "metric": metric,
                        "curl_column": curl_col,
                        "optflow_column": optflow_col,
                        "paired_unit": "subject_hemisphere",
                        "n_pairs": n_pairs,
                        "n_subjects": int(group_df["subject"].nunique()),
                        "curl_mean": curl_stats["mean"],
                        "curl_std": curl_stats["std"],
                        "curl_sem": curl_stats["sem"],
                        "curl_median": curl_stats["median"],
                        "curl_min": curl_stats["min"],
                        "curl_max": curl_stats["max"],
                        "optflow_mean": optflow_stats["mean"],
                        "optflow_std": optflow_stats["std"],
                        "optflow_sem": optflow_stats["sem"],
                        "optflow_median": optflow_stats["median"],
                        "optflow_min": optflow_stats["min"],
                        "optflow_max": optflow_stats["max"],
                        "diff_mean_curl_minus_optflow": diff_stats["mean"],
                        "diff_std": diff_stats["std"],
                        "diff_sem": diff_stats["sem"],
                        "diff_median": diff_stats["median"],
                        "diff_min": diff_stats["min"],
                        "diff_max": diff_stats["max"],
                        "paired_t_stat": t_stat,
                        "paired_p_value": p_value,
                        "cohen_dz_curl_minus_optflow": cohen_dz,
                        "cohen_dz_from_t": t_based_dz,
                    }
                )

    return pd.DataFrame(rows)


def _plot_example(
    out_path: Path,
    *,
    subject: str,
    group: Optional[str],
    hemisphere: str,
    frame: int,
    example: Dict,
    metric_name: str,
    metric_value: int,
    quiver_step: int,
    phase_cmap: str,
) -> None:
    phase = example["phase"]
    vx = example["vx"]
    vy = example["vy"]
    curl_pts = example["curl_pts"]
    opt_pts = example["opt_pts"]
    curl_matched = example["curl_matched"]
    opt_matched = example["opt_matched"]
    optflow_center_source = example.get("optflow_center_source", "frame_index")

    yy, xx = np.mgrid[0:phase.shape[0], 0:phase.shape[1]]
    sample = (
        np.isfinite(vx)
        & np.isfinite(vy)
        & ((yy % max(quiver_step, 1)) == 0)
        & ((xx % max(quiver_step, 1)) == 0)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    axes[0].imshow(phase, cmap=phase_cmap, origin="lower", vmin=-np.pi, vmax=np.pi)
    axes[0].quiver(
        xx[sample], yy[sample], vx[sample], vy[sample],
        color="#111111", angles="xy", scale_units="xy", scale=0.35,
        width=0.0022, alpha=0.75,
    )
    if len(curl_pts):
        axes[0].scatter(curl_pts[:, 0], curl_pts[:, 1], s=38, c="#d73027", marker="x")
    axes[0].set_title(f"curl_1 centers\nn={len(curl_pts)}")

    axes[1].imshow(phase, cmap=phase_cmap, origin="lower", vmin=-np.pi, vmax=np.pi)
    axes[1].quiver(
        xx[sample], yy[sample], vx[sample], vy[sample],
        color="#111111", angles="xy", scale_units="xy", scale=0.35,
        width=0.0022, alpha=0.75,
    )
    if len(opt_pts):
        axes[1].scatter(opt_pts[:, 0], opt_pts[:, 1], s=38, c="#2c7fb8", marker="x")
    axes[1].set_title(f"optflow {optflow_center_source}\nn={len(opt_pts)}")

    axes[2].imshow(phase, cmap=phase_cmap, origin="lower", vmin=-np.pi, vmax=np.pi)
    axes[2].quiver(
        xx[sample], yy[sample], vx[sample], vy[sample],
        color="#111111", angles="xy", scale_units="xy", scale=0.35,
        width=0.0022, alpha=0.6,
    )
    if len(curl_pts):
        axes[2].scatter(
            curl_pts[curl_matched, 0],
            curl_pts[curl_matched, 1],
            s=34,
            c="#31a354",
            marker="o",
            facecolors="none",
            linewidths=1.2,
        )
        axes[2].scatter(
            curl_pts[~curl_matched, 0],
            curl_pts[~curl_matched, 1],
            s=42,
            c="#d73027",
            marker="x",
        )
    if len(opt_pts):
        axes[2].scatter(
            opt_pts[opt_matched, 0],
            opt_pts[opt_matched, 1],
            s=34,
            c="#31a354",
            marker="s",
            facecolors="none",
            linewidths=1.2,
        )
        axes[2].scatter(
            opt_pts[~opt_matched, 0],
            opt_pts[~opt_matched, 1],
            s=42,
            c="#2c7fb8",
            marker="+",
        )
    axes[2].set_title("green=matched, red=curl only, blue=optflow only")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{subject} | group={group} | hemi={hemisphere} | frame={frame} | {metric_name}={metric_value}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_examples(
    frame_df: pd.DataFrame,
    example_store: Dict[Tuple[str, str, int], Dict],
    output_root: Path,
    max_example_frames: int,
    quiver_step: int,
    phase_cmap: str,
) -> None:
    for metric, folder in [
        ("curl_only_center_count", "top_curl_only"),
        ("optflow_only_center_count", "top_optflow_only"),
    ]:
        sub = frame_df[frame_df[metric] > 0].sort_values(by=metric, ascending=False).head(max_example_frames)
        if sub.empty:
            continue
        out_dir = output_root / "examples" / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(sub.itertuples(index=False), start=1):
            key = (str(row.subject), str(row.hemisphere), int(row.frame))
            payload = example_store.get(key)
            if payload is None:
                continue
            filename = (
                f"{idx:02d}_{row.subject}_{row.hemisphere}_frame{int(row.frame):03d}_{metric}_{int(getattr(row, metric))}.png"
            )
            _plot_example(
                out_dir / filename,
                subject=str(row.subject),
                group=getattr(row, "group"),
                hemisphere=str(row.hemisphere),
                frame=int(row.frame),
                example=payload,
                metric_name=metric,
                metric_value=int(getattr(row, metric)),
                quiver_step=quiver_step,
                phase_cmap=phase_cmap,
            )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pairs = list(
        _iter_bundle_pairs(
            input_root=args.input_root,
            curl_tag=args.curl_tag,
            optflow_tag=args.optflow_tag,
            hemisphere=args.hemisphere,
            groups=args.groups,
        )
    )
    if not pairs:
        raise SystemExit("No matching curl/optflow bundle pairs found.")

    frame_rows: List[Dict] = []
    example_store: Dict[Tuple[str, str, int], Dict] = {}
    for pair in pairs:
        rows, examples = _summarize_pair(
            pair,
            args.match_radius,
            args.phase_field_spacing,
            args.optflow_center_source,
        )
        frame_rows.extend(rows)
        for frame, payload in examples.items():
            example_store[(pair.subject_name, pair.hemisphere, int(frame))] = payload

    frame_df = pd.DataFrame(frame_rows).sort_values(
        by=["group", "subject", "hemisphere", "frame"],
        na_position="last",
    )
    subject_df = _aggregate_subject(frame_df)
    overall_df = _aggregate_overall(frame_df)
    ratio_error_df = _ratio_error_summary(subject_df)
    paired_stats_df = _paired_metric_summary(subject_df)

    frame_df.to_csv(args.output_root / "frame_summary.csv", index=False)
    subject_df.to_csv(args.output_root / "subject_summary.csv", index=False)
    overall_df.to_csv(args.output_root / "overall_summary.csv", index=False)
    ratio_error_df.to_csv(args.output_root / "ratio_error_summary.csv", index=False)
    paired_stats_df.to_csv(args.output_root / "paired_ttest_summary.csv", index=False)

    _write_examples(
        frame_df,
        example_store,
        args.output_root,
        args.max_example_frames,
        args.quiver_step,
        args.phase_cmap,
    )

    print(f"Compared bundle pairs: {len(pairs)}")
    print(f"Frame rows: {len(frame_df)}")
    print(f"Paired t-test rows: {len(paired_stats_df)}")
    print(f"Outputs written to: {args.output_root}")


if __name__ == "__main__":
    main()
