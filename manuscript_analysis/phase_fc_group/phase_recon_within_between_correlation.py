#!/usr/bin/env python
"""
Correlate original phase-FC within/between summaries with reconstructed phase-FC.

Inputs are paired_within_between outputs, specifically:

    <root>/paired_within_between/<comparison>/left|right/network_subject_deltas.csv

For each drug, hemisphere, network row, FC type, and value kind, this script
correlates:

    x = original phase-FC within/between value
    y = reconstructed phase-FC within/between value

The default value kinds are:
    - delta: Drug - PCB
    - drug: drug-condition value
    - placebo: placebo-condition value

CSV outputs include Pearson/Spearman correlation, r^2, regression parameters,
and common reconstruction error metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


VALUE_KIND_TO_COLUMN = {
    "delta": "delta",
    "drug": "drug_value",
    "placebo": "placebo_value",
}
FC_TYPES = ["within", "between"]


@dataclass(frozen=True)
class DatasetSpec:
    drug: str
    original_root: Path
    recon_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate original and reconstructed phase-FC within/between outputs."
    )
    parser.add_argument(
        "--dmt-original-root",
        default=Path("analysis_outputs/phase_fc_group_corr_7networks"),
        type=Path,
    )
    parser.add_argument(
        "--dmt-recon-root",
        default=Path("analysis_outputs/phase_fc_group_corr_recon_7networks"),
        type=Path,
    )
    parser.add_argument(
        "--lsd-original-root",
        default=Path("analysis_outputs/phase_fc_group_corr_7networks_LSD"),
        type=Path,
    )
    parser.add_argument(
        "--lsd-recon-root",
        default=Path("analysis_outputs/phase_fc_group_corr_recon_7networks_LSD"),
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/phase_recon_within_between_correlation"),
        type=Path,
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--value-kind",
        choices=sorted(VALUE_KIND_TO_COLUMN),
        action="append",
        default=None,
        help="Value kind to analyze. Can be passed multiple times. Defaults to delta, drug, placebo.",
    )
    parser.add_argument(
        "--min-n",
        default=5,
        type=int,
        help="Minimum paired rows required to compute correlations/regression.",
    )
    return parser.parse_args()


def normalize_subid(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("S"):
        text = text[1:]
    if text.isdigit():
        return f"{int(text):02d}"
    return text


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


def r_confidence_interval(r_value: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r_value) or n <= 3 or abs(r_value) >= 1:
        return math.nan, math.nan
    z = np.arctanh(r_value)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo = float(np.tanh(z - zcrit * se))
    hi = float(np.tanh(z + zcrit * se))
    return lo, hi


def discover_comparison_dirs(root: Path) -> list[Path]:
    base = root / "paired_within_between"
    if not base.exists():
        base = root
    dirs = sorted(path for path in base.iterdir() if path.is_dir() and "_minus_" in path.name)
    if not dirs:
        raise RuntimeError(f"No comparison directories found under {base}")
    return dirs


def load_network_subject_deltas(root: Path, hemispheres: list[str]) -> pd.DataFrame:
    frames = []
    for comparison_dir in discover_comparison_dirs(root):
        for hemisphere in hemispheres:
            path = comparison_dir / hemisphere / "network_subject_deltas.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            required = {
                "network_a",
                "network_b",
                "type",
                "drug_value",
                "placebo_value",
                "delta",
                "subid",
                "hemisphere",
                "comparison",
            }
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            df = df.copy()
            df["subid"] = df["subid"].map(normalize_subid)
            df["hemisphere"] = df["hemisphere"].astype(str).str.lower()
            df["type"] = df["type"].astype(str).str.lower()
            df["comparison_dir"] = comparison_dir.name
            df["source_file"] = str(path)
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No network_subject_deltas.csv files found under {root}")
    return pd.concat(frames, ignore_index=True)


def build_joined(
    original: pd.DataFrame,
    recon: pd.DataFrame,
    drug: str,
    value_kinds: list[str],
) -> pd.DataFrame:
    key_cols = ["comparison", "comparison_dir", "hemisphere", "subid", "network_a", "network_b", "type"]
    rows = []
    for value_kind in value_kinds:
        value_col = VALUE_KIND_TO_COLUMN[value_kind]
        left = original[key_cols + [value_col]].rename(columns={value_col: "original_value"})
        right = recon[key_cols + [value_col]].rename(columns={value_col: "recon_value"})
        joined = left.merge(right, on=key_cols, how="inner")
        joined["drug"] = drug
        joined["value_kind"] = value_kind
        rows.append(joined)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["error"] = out["recon_value"] - out["original_value"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    return out


def finite_xy(group: pd.DataFrame) -> pd.DataFrame:
    work = group[["original_value", "recon_value"]].apply(pd.to_numeric, errors="coerce")
    return work.replace([np.inf, -np.inf], np.nan).dropna()


def signed_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def stats_for_group(group: pd.DataFrame, keys: tuple[object, ...], key_names: list[str], min_n: int) -> dict[str, object]:
    row = dict(zip(key_names, keys))
    work = finite_xy(group)
    x = work["original_value"].to_numpy(dtype=float)
    y = work["recon_value"].to_numpy(dtype=float)
    n = int(len(work))
    error = y - x
    abs_error = np.abs(error)
    squared_error = error**2

    row.update(
        {
            "n": n,
            "original_mean": float(np.mean(x)) if n else math.nan,
            "original_sd": float(np.std(x, ddof=1)) if n > 1 else math.nan,
            "recon_mean": float(np.mean(y)) if n else math.nan,
            "recon_sd": float(np.std(y, ddof=1)) if n > 1 else math.nan,
            "bias_mean_error": float(np.mean(error)) if n else math.nan,
            "mean_abs_error": float(np.mean(abs_error)) if n else math.nan,
            "median_abs_error": float(np.median(abs_error)) if n else math.nan,
            "max_abs_error": float(np.max(abs_error)) if n else math.nan,
            "mse": float(np.mean(squared_error)) if n else math.nan,
            "rmse": float(np.sqrt(np.mean(squared_error))) if n else math.nan,
            "error_sd": float(np.std(error, ddof=1)) if n > 1 else math.nan,
            "mean_abs_original": float(np.mean(np.abs(x))) if n else math.nan,
        }
    )
    denom = np.abs(x) + np.abs(y)
    smape_terms = np.divide(2 * abs_error, denom, out=np.full_like(abs_error, np.nan), where=denom > 0)
    row["smape"] = float(np.nanmean(smape_terms)) if n and np.isfinite(smape_terms).any() else math.nan
    row["nrmse_by_original_sd"] = (
        row["rmse"] / row["original_sd"]
        if np.isfinite(row["rmse"]) and np.isfinite(row["original_sd"]) and row["original_sd"] > 0
        else math.nan
    )
    row["nrmse_by_original_abs_mean"] = (
        row["rmse"] / row["mean_abs_original"]
        if np.isfinite(row["rmse"]) and np.isfinite(row["mean_abs_original"]) and row["mean_abs_original"] > 0
        else math.nan
    )
    row["explained_variance_score"] = (
        1.0 - float(np.var(error, ddof=0)) / float(np.var(x, ddof=0))
        if n > 1 and np.var(x, ddof=0) > 0
        else math.nan
    )
    row["identity_r2_score"] = signed_r2(x, y) if n > 1 else math.nan

    if n >= min_n and np.std(x, ddof=1) > 0 and np.std(y, ddof=1) > 0:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        slope, intercept, reg_r, reg_p, slope_stderr = stats.linregress(x, y)
        ci_low, ci_high = r_confidence_interval(float(pearson_r), n)
        row.update(
            {
                "pearson_r": float(pearson_r),
                "pearson_r2": float(pearson_r**2),
                "pearson_p": float(pearson_p),
                "pearson_r_ci95_low": ci_low,
                "pearson_r_ci95_high": ci_high,
                "pearson_stars": p_to_stars(float(pearson_p)),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "slope": float(slope),
                "intercept": float(intercept),
                "regression_r2": float(reg_r**2),
                "regression_p": float(reg_p),
                "slope_stderr": float(slope_stderr),
            }
        )
    else:
        row.update(
            {
                "pearson_r": math.nan,
                "pearson_r2": math.nan,
                "pearson_p": math.nan,
                "pearson_r_ci95_low": math.nan,
                "pearson_r_ci95_high": math.nan,
                "pearson_stars": "",
                "spearman_r": math.nan,
                "spearman_p": math.nan,
                "slope": math.nan,
                "intercept": math.nan,
                "regression_r2": math.nan,
                "regression_p": math.nan,
                "slope_stderr": math.nan,
            }
        )
    return row


def correlation_stats(joined: pd.DataFrame, min_n: int) -> pd.DataFrame:
    key_names = ["drug", "comparison", "hemisphere", "value_kind", "type", "network_a", "network_b"]
    rows = []
    for keys, group in joined.groupby(key_names, sort=False):
        rows.append(stats_for_group(group, keys, key_names, min_n=min_n))
    return pd.DataFrame(rows)


def aggregate_stats(joined: pd.DataFrame, min_n: int) -> pd.DataFrame:
    key_names = ["drug", "comparison", "hemisphere", "value_kind", "type"]
    rows = []
    for keys, group in joined.groupby(key_names, sort=False):
        row = stats_for_group(group, keys, key_names, min_n=min_n)
        row["network_a"] = "ALL"
        row["network_b"] = "ALL"
        rows.append(row)

    hemi_key_names = ["drug", "comparison", "hemisphere", "value_kind"]
    for keys, group in joined.groupby(hemi_key_names, sort=False):
        row = stats_for_group(group, keys, hemi_key_names, min_n=min_n)
        row["type"] = "all"
        row["network_a"] = "ALL"
        row["network_b"] = "ALL"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hemispheres = ["left", "right"] if args.hemisphere == "both" else [args.hemisphere]
    value_kinds = args.value_kind or ["delta", "drug", "placebo"]
    specs = [
        DatasetSpec("DMT", args.dmt_original_root, args.dmt_recon_root),
        DatasetSpec("LSD", args.lsd_original_root, args.lsd_recon_root),
    ]

    joined_frames = []
    metadata = {
        "hemispheres": hemispheres,
        "value_kinds": value_kinds,
        "min_n": int(args.min_n),
        "datasets": [],
    }
    for spec in specs:
        original = load_network_subject_deltas(spec.original_root, hemispheres)
        recon = load_network_subject_deltas(spec.recon_root, hemispheres)
        joined = build_joined(original, recon, spec.drug, value_kinds)
        if joined.empty:
            raise RuntimeError(f"No joined rows for {spec.drug}")
        joined_frames.append(joined)
        metadata["datasets"].append(
            {
                "drug": spec.drug,
                "original_root": str(spec.original_root),
                "recon_root": str(spec.recon_root),
                "n_original_rows": int(len(original)),
                "n_recon_rows": int(len(recon)),
                "n_joined_rows": int(len(joined)),
            }
        )

    joined_all = pd.concat(joined_frames, ignore_index=True)
    row_stats = correlation_stats(joined_all, min_n=args.min_n)
    agg_stats = aggregate_stats(joined_all, min_n=args.min_n)
    all_stats = pd.concat([row_stats, agg_stats], ignore_index=True)

    joined_all.to_csv(args.out_dir / "original_recon_within_between_joined.csv", index=False)
    row_stats.to_csv(args.out_dir / "original_recon_within_between_correlations_by_network.csv", index=False)
    agg_stats.to_csv(args.out_dir / "original_recon_within_between_correlations_aggregate.csv", index=False)
    all_stats.to_csv(args.out_dir / "original_recon_within_between_correlations_all.csv", index=False)

    metadata.update(
        {
            "n_joined_rows": int(len(joined_all)),
            "n_network_stats_rows": int(len(row_stats)),
            "n_aggregate_stats_rows": int(len(agg_stats)),
            "x": "original phase-FC value",
            "y": "reconstructed phase-FC value",
            "error": "recon_value - original_value",
        }
    )
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote original/reconstructed within-between correlations to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
