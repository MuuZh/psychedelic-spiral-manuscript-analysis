from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .dfr_stats import mean_error_fields, pearson_ci, slope_ci


def paired_stats(per_subject: pd.DataFrame, min_pairs: int = 3) -> pd.DataFrame:
    rows = []
    metrics = ["mean_region_count", "mean_boundary_count"]
    group_cols = ["source", "drug", "comparison", "hemisphere"]
    for keys, group in per_subject.groupby(group_cols, dropna=False):
        for metric in metrics:
            pivot = group.pivot_table(index="subid", columns="role", values=metric, aggfunc="mean")
            if not {"Drug", "PCB"}.issubset(pivot.columns):
                continue
            pivot = pivot[["PCB", "Drug"]].dropna()
            delta = pivot["Drug"].to_numpy(float) - pivot["PCB"].to_numpy(float)
            if len(pivot) >= max(2, min_pairs):
                t_val, p_val = stats.ttest_rel(pivot["Drug"], pivot["PCB"], nan_policy="omit")
                sd = float(np.std(delta, ddof=1))
            else:
                t_val = p_val = sd = math.nan
            rows.append(
                {
                    "source": keys[0],
                    "drug": keys[1],
                    "comparison": keys[2],
                    "hemisphere": keys[3],
                    "metric": metric,
                    "n": int(len(pivot)),
                    **mean_error_fields(pivot["PCB"].to_numpy(float), "pcb"),
                    "mean_pcb": float(pivot["PCB"].mean()) if len(pivot) else math.nan,
                    "sd_pcb": float(pivot["PCB"].std(ddof=1)) if len(pivot) > 1 else math.nan,
                    **mean_error_fields(pivot["Drug"].to_numpy(float), "drug"),
                    "mean_drug": float(pivot["Drug"].mean()) if len(pivot) else math.nan,
                    "sd_drug": float(pivot["Drug"].std(ddof=1)) if len(pivot) > 1 else math.nan,
                    **mean_error_fields(delta, "delta"),
                    "delta_mean": float(np.mean(delta)) if len(delta) else math.nan,
                    "delta_sd": sd,
                    "t": float(t_val) if np.isfinite(t_val) else math.nan,
                    "p": float(p_val) if np.isfinite(p_val) else math.nan,
                    "cohen_dz": float(np.mean(delta) / sd) if sd and np.isfinite(sd) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def delta_table(per_subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in per_subject.groupby(["source", "drug", "comparison", "hemisphere"], dropna=False):
        pivot = group.pivot_table(
            index="subid",
            columns="role",
            values=["mean_region_count", "mean_boundary_count"],
            aggfunc="mean",
        )
        needed = [
            ("mean_region_count", "Drug"),
            ("mean_region_count", "PCB"),
            ("mean_boundary_count", "Drug"),
            ("mean_boundary_count", "PCB"),
        ]
        if not all(col in pivot.columns for col in needed):
            continue
        pivot = pivot.dropna(subset=needed)
        for subid, vals in pivot.iterrows():
            rows.append(
                {
                    "source": keys[0],
                    "drug": keys[1],
                    "comparison": keys[2],
                    "subid": str(subid),
                    "hemisphere": keys[3],
                    "delta_region_count": float(vals[("mean_region_count", "Drug")] - vals[("mean_region_count", "PCB")]),
                    "delta_boundary_count": float(vals[("mean_boundary_count", "Drug")] - vals[("mean_boundary_count", "PCB")]),
                }
            )
    return pd.DataFrame(rows)


def correlation_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    metrics: list[str],
    min_n: int,
) -> pd.DataFrame:
    rows = []
    keys = ["drug", "comparison", "hemisphere"]
    if left.empty or right.empty or not set(keys).issubset(left.columns) or not set(keys).issubset(right.columns):
        return pd.DataFrame()
    for group_keys, lgrp in left.groupby(keys, dropna=False):
        rgrp = right
        for key, value in zip(keys, group_keys):
            rgrp = rgrp[rgrp[key] == value]
        merged = lgrp.merge(rgrp, on=["drug", "comparison", "subid", "hemisphere"], suffixes=(f"_{left_name}", f"_{right_name}"))
        for metric in metrics:
            x = merged[f"{metric}_{left_name}"].to_numpy(float) if f"{metric}_{left_name}" in merged else np.array([])
            y = merged[f"{metric}_{right_name}"].to_numpy(float) if f"{metric}_{right_name}" in merged else np.array([])
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if len(x) >= max(3, min_n) and np.std(x) > 0 and np.std(y) > 0:
                pr, pp = stats.pearsonr(x, y)
                sr, sp = stats.spearmanr(x, y)
                slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
                pearson_se, pearson_low, pearson_high, r2_low, r2_high = pearson_ci(pr, len(x))
                spearman_se, spearman_low, spearman_high, _, _ = pearson_ci(sr, len(x))
                slope_low, slope_high = slope_ci(slope, stderr, len(x))
            else:
                pr = pp = sr = sp = slope = r_value = stderr = math.nan
                pearson_se = pearson_low = pearson_high = r2_low = r2_high = math.nan
                spearman_se = spearman_low = spearman_high = math.nan
                slope_low = slope_high = math.nan
            rows.append(
                {
                    "drug": group_keys[0],
                    "comparison": group_keys[1],
                    "hemisphere": group_keys[2],
                    "metric": metric,
                    "n": int(len(x)),
                    "pearson_r": float(pr) if np.isfinite(pr) else math.nan,
                    "pearson_r_se": float(pearson_se) if np.isfinite(pearson_se) else math.nan,
                    "pearson_r_ci95_low": float(pearson_low) if np.isfinite(pearson_low) else math.nan,
                    "pearson_r_ci95_high": float(pearson_high) if np.isfinite(pearson_high) else math.nan,
                    "pearson_p": float(pp) if np.isfinite(pp) else math.nan,
                    "spearman_rho": float(sr) if np.isfinite(sr) else math.nan,
                    "spearman_rho_se": float(spearman_se) if np.isfinite(spearman_se) else math.nan,
                    "spearman_rho_ci95_low": float(spearman_low) if np.isfinite(spearman_low) else math.nan,
                    "spearman_rho_ci95_high": float(spearman_high) if np.isfinite(spearman_high) else math.nan,
                    "spearman_p": float(sp) if np.isfinite(sp) else math.nan,
                    "slope": float(slope) if np.isfinite(slope) else math.nan,
                    "slope_se": float(stderr) if np.isfinite(stderr) else math.nan,
                    "slope_ci95_low": float(slope_low) if np.isfinite(slope_low) else math.nan,
                    "slope_ci95_high": float(slope_high) if np.isfinite(slope_high) else math.nan,
                    "r_squared": float(r_value * r_value) if np.isfinite(r_value) else math.nan,
                    "r_squared_ci95_low": float(r2_low) if np.isfinite(r2_low) else math.nan,
                    "r_squared_ci95_high": float(r2_high) if np.isfinite(r2_high) else math.nan,
                }
            )
    return pd.DataFrame(rows)
