#!/usr/bin/env python
"""
Compute per-pattern MSD beta from combined MatPhase pattern outputs.

For each pattern, the script:
  1. filters patterns shorter than --min-duration (default 10 frames),
  2. builds the centroid trajectory,
  3. computes mean squared displacement (MSD) for each temporal lag,
  4. searches all contiguous log-log MSD windows with at least
     --min-linear-points points,
  5. saves the slope of the most linear window as msd_beta.

Example:
    python scripts/pattern_msd_beta.py ^
        --combined-dir combined_outputs/DMT ^
        --out-dir analysis_outputs/pattern_msd_beta/dmt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


PATTERN_KEYS = ["group", "subid", "hemisphere", "bundle_dir", "pattern_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-pattern MSD beta.")
    parser.add_argument(
        "--combined-dir",
        required=True,
        type=Path,
        help="Directory containing combined_patterns.parquet and combined_frame_index.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for per-pattern beta tables and optional QC plots.",
    )
    parser.add_argument(
        "--min-duration",
        default=10,
        type=int,
        help="Minimum pattern duration/frame count to analyze. Default: 10.",
    )
    parser.add_argument(
        "--min-linear-points",
        default=5,
        type=int,
        help="Minimum number of lag points in the selected log-log linear window. Default: 5.",
    )
    parser.add_argument(
        "--centroid",
        choices=["weighted", "unweighted"],
        default="weighted",
        help="Use weighted or unweighted centroid trajectory. Default: weighted.",
    )
    parser.add_argument(
        "--max-window-points",
        default=None,
        type=int,
        help="Optional maximum number of points in candidate linear windows.",
    )
    parser.add_argument(
        "--save-msd-curves",
        action="store_true",
        help="Also save all per-pattern MSD lag curves. This can be large.",
    )
    parser.add_argument(
        "--plot-qc",
        action="store_true",
        help="Save a histogram of beta values and selected-window R2.",
    )
    return parser.parse_args()


def _choose_xy_columns(frames: pd.DataFrame, centroid: str) -> tuple[str, str]:
    if centroid == "weighted":
        weighted = ("weighted_centroid_x", "weighted_centroid_y")
        if all(col in frames.columns for col in weighted):
            return weighted
    needed = ("centroid_x", "centroid_y")
    if not all(col in frames.columns for col in needed):
        raise ValueError("combined_frame_index.parquet lacks centroid columns.")
    return needed


def compute_msd(times: np.ndarray, xy: np.ndarray) -> pd.DataFrame:
    order = np.argsort(times)
    times = times[order].astype(float)
    xy = xy[order].astype(float)
    finite = np.isfinite(times) & np.isfinite(xy).all(axis=1)
    times = times[finite]
    xy = xy[finite]
    if times.size < 2:
        return pd.DataFrame(columns=["lag_frames", "n_pairs", "msd"])

    by_lag: dict[float, list[float]] = {}
    for start in range(times.size - 1):
        dt = times[start + 1 :] - times[start]
        disp = xy[start + 1 :] - xy[start]
        sq = np.sum(disp * disp, axis=1)
        valid = (dt > 0) & np.isfinite(sq)
        for lag, sq_val in zip(dt[valid], sq[valid]):
            by_lag.setdefault(float(lag), []).append(float(sq_val))

    rows: list[dict[str, float]] = []
    for lag, sq_vals in by_lag.items():
        arr = np.asarray(sq_vals, dtype=float)
        rows.append({"lag_frames": float(lag), "n_pairs": int(arr.size), "msd": float(np.mean(arr))})
    return pd.DataFrame(rows).sort_values("lag_frames").reset_index(drop=True)


def fit_best_loglog_window(
    msd_df: pd.DataFrame,
    min_linear_points: int,
    max_window_points: int | None,
) -> dict[str, float]:
    work = msd_df.copy()
    work = work[(work["lag_frames"] > 0) & (work["msd"] > 0)].copy()
    work["log_lag"] = np.log(work["lag_frames"].to_numpy(dtype=float))
    work["log_msd"] = np.log(work["msd"].to_numpy(dtype=float))
    work = work[np.isfinite(work["log_lag"]) & np.isfinite(work["log_msd"])].reset_index(drop=True)
    n = int(len(work))
    if n < min_linear_points:
        return {
            "msd_beta": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
            "p": math.nan,
            "stderr": math.nan,
            "linear_start_lag": math.nan,
            "linear_end_lag": math.nan,
            "linear_n_points": n,
            "linear_window_rank_score": math.nan,
        }

    best: dict[str, float] | None = None
    max_len = max_window_points if max_window_points is not None else n
    max_len = max(min_linear_points, min(max_len, n))
    for start in range(n):
        for end in range(start + min_linear_points, min(n, start + max_len) + 1):
            sub = work.iloc[start:end]
            fit = stats.linregress(sub["log_lag"], sub["log_msd"])
            if not np.isfinite(fit.slope) or not np.isfinite(fit.rvalue):
                continue
            r2 = float(fit.rvalue * fit.rvalue)
            window_len = int(end - start)
            lag_span = float(sub["lag_frames"].iloc[-1] - sub["lag_frames"].iloc[0])
            rank_score = r2 + 1e-6 * window_len + 1e-9 * lag_span
            candidate = {
                "msd_beta": float(fit.slope),
                "intercept": float(fit.intercept),
                "r2": r2,
                "p": float(fit.pvalue),
                "stderr": float(fit.stderr) if fit.stderr is not None else math.nan,
                "linear_start_lag": float(sub["lag_frames"].iloc[0]),
                "linear_end_lag": float(sub["lag_frames"].iloc[-1]),
                "linear_n_points": window_len,
                "linear_window_rank_score": rank_score,
            }
            if best is None or candidate["linear_window_rank_score"] > best["linear_window_rank_score"]:
                best = candidate
    if best is None:
        return {
            "msd_beta": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
            "p": math.nan,
            "stderr": math.nan,
            "linear_start_lag": math.nan,
            "linear_end_lag": math.nan,
            "linear_n_points": n,
            "linear_window_rank_score": math.nan,
        }
    return best


def plot_qc(per_pattern: pd.DataFrame, out_dir: Path) -> None:
    valid = per_pattern[np.isfinite(pd.to_numeric(per_pattern["msd_beta"], errors="coerce"))].copy()
    if valid.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(valid["msd_beta"], bins=40, color="#4c78a8", edgecolor="white")
    axes[0].set_xlabel("MSD beta")
    axes[0].set_ylabel("Pattern count")
    axes[1].hist(valid["r2"].dropna(), bins=40, color="#f58518", edgecolor="white")
    axes[1].set_xlabel("Selected window R2")
    axes[1].set_ylabel("Pattern count")
    fig.tight_layout()
    fig.savefig(out_dir / "msd_beta_qc_histograms.png", dpi=240)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patterns_path = args.combined_dir / "combined_patterns.parquet"
    frames_path = args.combined_dir / "combined_frame_index.parquet"
    if not patterns_path.exists() or not frames_path.exists():
        raise FileNotFoundError(f"Missing combined parquet inputs under {args.combined_dir}")

    patterns = pd.read_parquet(patterns_path)
    frames = pd.read_parquet(frames_path)
    missing = [col for col in PATTERN_KEYS if col not in patterns.columns or col not in frames.columns]
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")
    if "duration" not in patterns.columns:
        raise ValueError("combined_patterns.parquet must contain duration.")

    x_col, y_col = _choose_xy_columns(frames, args.centroid)
    pattern_cols = PATTERN_KEYS + [
        col
        for col in ["rotation_direction", "curl_sign", "duration", "start_frame", "end_frame"]
        if col in patterns.columns
    ]
    eligible = patterns[pattern_cols].copy()
    eligible["duration"] = pd.to_numeric(eligible["duration"], errors="coerce")
    eligible = eligible[eligible["duration"] >= args.min_duration].copy()
    if eligible.empty:
        raise RuntimeError(f"No patterns meet --min-duration {args.min_duration}.")

    frame_cols = PATTERN_KEYS + ["abs_time", x_col, y_col]
    merged = frames[frame_cols].merge(eligible, on=PATTERN_KEYS, how="inner")

    records: list[dict[str, object]] = []
    msd_curves: list[pd.DataFrame] = []
    group_cols = PATTERN_KEYS + [
        col for col in ["rotation_direction", "curl_sign", "duration", "start_frame", "end_frame"] if col in eligible.columns
    ]
    grouped = merged.groupby(group_cols, dropna=False, sort=False)
    iterator = grouped
    if tqdm is not None:
        iterator = tqdm(grouped, total=grouped.ngroups, desc="MSD beta", dynamic_ncols=True)
    for keys, pattern_frames in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        traj = pattern_frames[["abs_time", x_col, y_col]].dropna()
        traj = traj.drop_duplicates(subset=["abs_time"]).sort_values("abs_time")
        msd_df = compute_msd(
            traj["abs_time"].to_numpy(dtype=float),
            traj[[x_col, y_col]].to_numpy(dtype=float),
        )
        fit = fit_best_loglog_window(msd_df, args.min_linear_points, args.max_window_points)
        base.update(fit)
        base["frame_count"] = int(traj["abs_time"].nunique())
        base["msd_point_count"] = int(len(msd_df))
        base["centroid_mode"] = args.centroid
        base["x_column"] = x_col
        base["y_column"] = y_col
        records.append(base)
        if args.save_msd_curves and not msd_df.empty:
            tagged = msd_df.copy()
            for col in PATTERN_KEYS:
                tagged[col] = base[col]
            msd_curves.append(tagged)

    per_pattern = pd.DataFrame(records)
    per_pattern.to_csv(args.out_dir / "per_pattern_msd_beta.csv", index=False)
    per_subject = (
        per_pattern.groupby(["group", "subid", "hemisphere"], as_index=False)
        .agg(
            msd_beta_mean=("msd_beta", "mean"),
            msd_beta_median=("msd_beta", "median"),
            msd_beta_std=("msd_beta", "std"),
            selected_window_r2_mean=("r2", "mean"),
            n_patterns=("msd_beta", "count"),
            n_patterns_input=("pattern_id", "count"),
            mean_duration=("duration", "mean"),
        )
    )
    per_subject.to_csv(args.out_dir / "per_subject_msd_beta.csv", index=False)
    if msd_curves:
        pd.concat(msd_curves, ignore_index=True).to_csv(args.out_dir / "per_pattern_msd_curves.csv", index=False)
    if args.plot_qc:
        plot_qc(per_pattern, args.out_dir)

    metadata = {
        "combined_dir": str(args.combined_dir),
        "min_duration": int(args.min_duration),
        "min_linear_points": int(args.min_linear_points),
        "max_window_points": args.max_window_points,
        "centroid": args.centroid,
        "x_column": x_col,
        "y_column": y_col,
        "n_patterns_input": int(len(patterns)),
        "n_patterns_duration_eligible": int(len(eligible)),
        "n_patterns_output": int(len(per_pattern)),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote per-pattern MSD beta results to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
