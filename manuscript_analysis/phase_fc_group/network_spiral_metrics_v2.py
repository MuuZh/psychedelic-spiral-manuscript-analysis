#!/usr/bin/env python
"""
V2 network-wise spiral metric ln-ratio analysis.

This script is intentionally a post-processing layer over
network_spiral_metrics.py outputs. It preserves the original network assignment
and metric computation, then keeps the selected metrics and converts paired
Drug-vs-PCB values to subject-level ln(Drug / PCB) for ridgeline plots.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats

from network_spiral_metrics import infer_network_order


SELECTED_METRIC_LABELS = {
    "spiral_count_per_network_px": "Pattern count / network px",
    "mean_spiral_size": "Mean spiral size",
    "mean_spiral_power": "Mean spiral power",
    "weighted_mean_cos2_alignment": "Weighted mean cos(2theta)",
}

STUDY_INPUT_DEFAULTS = {
    "DMT": Path("analysis_outputs/phase_fc_group/network_spiral_metrics/subject_network_metrics_wide.csv"),
    "LSD": Path("analysis_outputs/phase_fc_group/network_spiral_metrics_LSD/subject_network_metrics_wide.csv"),
}

STUDY_STYLE = {
    "DMT": {"color": "#d62728", "linewidth": 2.4},
    "LSD": {"color": "#1f77b4", "linewidth": 2.4},
}
HEMI_LINESTYLE = {"left": "-", "right": "--"}
DIST_ALPHA = 0.92
MEAN_ALPHA = 0.68
MEAN_LINEWIDTH = 1.25
MEAN_TICK_FRACTION = 0.28
FONT_SIZES = {
    "suptitle": 28,
    "axis_title": 24,
    "axis_label": 18,
    "tick": 17,
    "legend": 18,
}


def parse_study_input(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Study inputs must use STUDY=path format.")
    study, path = text.split("=", 1)
    study = study.strip()
    if not study:
        raise argparse.ArgumentTypeError("Study name cannot be empty.")
    return study, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and summarize Drug-vs-PCB ln-ratio distributions for selected network spiral metrics."
    )
    parser.add_argument(
        "--study-input",
        action="append",
        type=parse_study_input,
        default=None,
        metavar="STUDY=CSV",
        help=(
            "Subject-level wide CSV from network_spiral_metrics.py. Can be repeated. "
            "Defaults to existing DMT and LSD outputs."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/network_spiral_metrics_v2"),
        type=Path,
    )
    parser.add_argument("--min-pairs", default=3, type=int)
    parser.add_argument("--kde-points", default=256, type=int)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_subject_tables(study_inputs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    missing = []
    required_base = {"role", "subid", "hemisphere", "network", "network_index"}
    for study, path in study_inputs:
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        required = required_base | set(SELECTED_METRIC_LABELS)
        absent = sorted(required - set(df.columns))
        if absent:
            raise ValueError(f"{path} missing required columns: {absent}")
        df = df.copy()
        df["study"] = study
        rows.append(df)
    if missing:
        raise FileNotFoundError("Missing input CSV(s): " + ", ".join(missing))
    if not rows:
        raise RuntimeError("No input tables loaded.")
    return pd.concat(rows, ignore_index=True)


def compute_paired_values(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = list(SELECTED_METRIC_LABELS)
    group_cols = ["study", "subid", "hemisphere", "network", "network_index"]
    for keys, sub in subject_df.groupby(group_cols, sort=False, observed=True):
        pivot = sub.pivot_table(index="subid", columns="role", values=metrics, aggfunc="mean")
        if pivot.empty or not {"Drug", "PCB"}.issubset(set(pivot.columns.get_level_values(1))):
            continue
        study, subid, hemi, network, network_index = keys
        for metric in metrics:
            try:
                drug = float(pivot[(metric, "Drug")].iloc[0])
                pcb = float(pivot[(metric, "PCB")].iloc[0])
            except KeyError:
                continue
            delta = drug - pcb if np.isfinite(drug) and np.isfinite(pcb) else math.nan
            rows.append(
                {
                    "study": study,
                    "subid": str(subid),
                    "hemisphere": hemi,
                    "network": network,
                    "network_index": int(network_index),
                    "metric": metric,
                    "metric_label": SELECTED_METRIC_LABELS[metric],
                    "drug_value": drug,
                    "pcb_value": pcb,
                    "delta_drug_minus_pcb": delta,
                }
            )
    return pd.DataFrame(rows)


def compute_log_ratio(paired_df: pd.DataFrame) -> pd.DataFrame:
    if paired_df.empty:
        return paired_df.copy()

    rows = []
    for row in paired_df.itertuples(index=False):
        drug = float(row.drug_value)
        pcb = float(row.pcb_value)
        if np.isfinite(drug) and np.isfinite(pcb) and drug > 0 and pcb > 0:
            log_ratio = float(np.log(drug / pcb))
            equivalent_pct = float(100.0 * np.expm1(log_ratio))
            invalid_reason = ""
        else:
            log_ratio = math.nan
            equivalent_pct = math.nan
            invalid_reason = "nonpositive_or_missing_value"
        rows.append(
            {
                "study": row.study,
                "subid": str(row.subid),
                "hemisphere": row.hemisphere,
                "network": row.network,
                "network_index": int(row.network_index),
                "metric": row.metric,
                "metric_label": row.metric_label,
                "drug_value": drug,
                "pcb_value": pcb,
                "delta_drug_minus_pcb": float(row.delta_drug_minus_pcb),
                "log_ratio_drug_vs_pcb": log_ratio,
                "equivalent_percent_change": equivalent_pct,
                "log_ratio_invalid_reason": invalid_reason,
            }
        )
    return pd.DataFrame(rows)


def compute_standardized_difference(paired_df: pd.DataFrame) -> pd.DataFrame:
    if paired_df.empty:
        return paired_df.copy()

    base = paired_df.copy()
    pcb_sd = (
        base.groupby(["study", "hemisphere", "network", "network_index", "metric"], observed=True)["pcb_value"]
        .std(ddof=1)
        .rename("pcb_sd")
        .reset_index()
    )
    base = base.merge(pcb_sd, on=["study", "hemisphere", "network", "network_index", "metric"], how="left")

    rows = []
    for row in base.itertuples(index=False):
        delta = float(row.delta_drug_minus_pcb)
        pcb_sd_value = float(row.pcb_sd) if np.isfinite(row.pcb_sd) else math.nan
        if np.isfinite(delta) and np.isfinite(pcb_sd_value) and pcb_sd_value > 0:
            z_value = float(delta / pcb_sd_value)
            invalid_reason = ""
        else:
            z_value = math.nan
            invalid_reason = "missing_or_zero_pcb_sd"
        rows.append(
            {
                "study": row.study,
                "subid": str(row.subid),
                "hemisphere": row.hemisphere,
                "network": row.network,
                "network_index": int(row.network_index),
                "metric": row.metric,
                "metric_label": row.metric_label,
                "drug_value": float(row.drug_value),
                "pcb_value": float(row.pcb_value),
                "delta_drug_minus_pcb": delta,
                "pcb_sd": pcb_sd_value,
                "standardized_difference": z_value,
                "standardized_difference_invalid_reason": invalid_reason,
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


def summarize_log_ratio(log_df: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    rows = []
    group_cols = ["study", "hemisphere", "network", "network_index", "metric"]
    for keys, sub in log_df.groupby(group_cols, sort=False, observed=True):
        study, hemi, network, network_index, metric = keys
        values = pd.to_numeric(sub["log_ratio_drug_vs_pcb"], errors="coerce").dropna().to_numpy(dtype=float)
        mean = float(np.mean(values)) if values.size else math.nan
        std = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
        sem = float(stats.sem(values, nan_policy="omit")) if values.size > 1 else math.nan
        ci_low, ci_high = _mean_ci(values)
        rows.append(
            {
                "study": study,
                "hemisphere": hemi,
                "network": network,
                "network_index": int(network_index),
                "metric": metric,
                "metric_label": SELECTED_METRIC_LABELS[metric],
                "n_paired": int(values.size),
                "mean_log_ratio": mean,
                "std_log_ratio": std,
                "sem_log_ratio": sem,
                "log_ratio_ci95_low": ci_low,
                "log_ratio_ci95_high": ci_high,
                "geometric_mean_percent_change": _log_ratio_to_percent(mean),
                "geometric_percent_change_ci95_low": _log_ratio_to_percent(ci_low),
                "geometric_percent_change_ci95_high": _log_ratio_to_percent(ci_high),
            }
        )
    return pd.DataFrame(rows)


def summarize_standardized_difference(std_df: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    rows = []
    group_cols = ["study", "hemisphere", "network", "network_index", "metric"]
    for keys, sub in std_df.groupby(group_cols, sort=False, observed=True):
        study, hemi, network, network_index, metric = keys
        values = pd.to_numeric(sub["standardized_difference"], errors="coerce").dropna().to_numpy(dtype=float)
        mean = float(np.mean(values)) if values.size else math.nan
        std = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
        sem = float(stats.sem(values, nan_policy="omit")) if values.size > 1 else math.nan
        ci_low, ci_high = _mean_ci(values)
        rows.append(
            {
                "study": study,
                "hemisphere": hemi,
                "network": network,
                "network_index": int(network_index),
                "metric": metric,
                "metric_label": SELECTED_METRIC_LABELS[metric],
                "n_paired": int(values.size),
                "mean_standardized_difference": mean,
                "std_standardized_difference": std,
                "sem_standardized_difference": sem,
                "standardized_difference_ci95_low": ci_low,
                "standardized_difference_ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def group_basic_stats_paired_only(paired_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if paired_df.empty:
        return pd.DataFrame(rows)
    group_cols = ["study", "hemisphere", "network", "network_index", "metric"]
    for keys, sub in paired_df.groupby(group_cols, sort=False, observed=True):
        study, hemi, network, network_index, metric = keys
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
                    "study": study,
                    "hemisphere": hemi,
                    "network": network,
                    "network_index": int(network_index),
                    "metric": metric,
                    "metric_label": SELECTED_METRIC_LABELS[metric],
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


def paired_ttest_raw_summary(paired_df: pd.DataFrame, min_pairs: int = 3) -> pd.DataFrame:
    rows = []
    if paired_df.empty:
        return pd.DataFrame(rows)
    group_cols = ["study", "hemisphere", "network", "network_index", "metric"]
    for keys, sub in paired_df.groupby(group_cols, sort=False, observed=True):
        study, hemi, network, network_index, metric = keys
        drug = pd.to_numeric(sub["drug_value"], errors="coerce").to_numpy(dtype=float)
        pcb = pd.to_numeric(sub["pcb_value"], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(drug) & np.isfinite(pcb)
        drug = drug[finite_mask]
        pcb = pcb[finite_mask]
        delta = drug - pcb
        n = int(delta.size)
        drug_ci_low, drug_ci_high = _mean_ci(drug)
        pcb_ci_low, pcb_ci_high = _mean_ci(pcb)
        delta_ci_low, delta_ci_high = _mean_ci(delta)
        delta_mean = float(np.mean(delta)) if n else math.nan
        delta_std = float(np.std(delta, ddof=1)) if n > 1 else math.nan
        if n >= min_pairs:
            t_val, p_val = stats.ttest_rel(drug, pcb, nan_policy="omit")
            t_val = float(t_val)
            p_val = float(p_val)
        else:
            t_val = math.nan
            p_val = math.nan
        rows.append(
            {
                "study": study,
                "hemisphere": hemi,
                "network": network,
                "network_index": int(network_index),
                "metric": metric,
                "metric_label": SELECTED_METRIC_LABELS[metric],
                "n": n,
                "drug_mean": float(np.mean(drug)) if n else math.nan,
                "drug_std": float(np.std(drug, ddof=1)) if n > 1 else math.nan,
                "drug_sem": float(stats.sem(drug, nan_policy="omit")) if n > 1 else math.nan,
                "drug_ci95_low": drug_ci_low,
                "drug_ci95_high": drug_ci_high,
                "pcb_mean": float(np.mean(pcb)) if n else math.nan,
                "pcb_std": float(np.std(pcb, ddof=1)) if n > 1 else math.nan,
                "pcb_sem": float(stats.sem(pcb, nan_policy="omit")) if n > 1 else math.nan,
                "pcb_ci95_low": pcb_ci_low,
                "pcb_ci95_high": pcb_ci_high,
                "mean_delta": delta_mean,
                "std_delta": delta_std,
                "sem_delta": float(stats.sem(delta, nan_policy="omit")) if n > 1 else math.nan,
                "delta_ci95_low": delta_ci_low,
                "delta_ci95_high": delta_ci_high,
                "t": t_val,
                "p": p_val,
                "dz": delta_mean / delta_std if np.isfinite(delta_mean) and np.isfinite(delta_std) and delta_std > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def two_way_rm_anova(subject_df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.anova import AnovaRM

    metrics = [col for col in SELECTED_METRIC_LABELS if col in subject_df.columns]
    network_order = infer_network_order(subject_df["network"])
    rows = []
    effects = ["role", "network", "role:network"]
    for study in sorted(subject_df["study"].dropna().unique()):
        study_df = subject_df[subject_df["study"] == study]
        for hemi in sorted(study_df["hemisphere"].dropna().unique()):
            hemi_df = study_df[study_df["hemisphere"] == hemi]
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
                if n_subjects < 2:
                    for effect in effects:
                        rows.append(
                            {
                                "study": study,
                                "hemisphere": hemi,
                                "metric": metric,
                                "metric_label": SELECTED_METRIC_LABELS[metric],
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
                                "study": study,
                                "hemisphere": hemi,
                                "metric": metric,
                                "metric_label": SELECTED_METRIC_LABELS[metric],
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
                                "study": study,
                                "hemisphere": hemi,
                                "metric": metric,
                                "metric_label": SELECTED_METRIC_LABELS[metric],
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

    metrics = [col for col in SELECTED_METRIC_LABELS if col in subject_df.columns]
    network_order = infer_network_order(subject_df["network"])
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
    for study in sorted(subject_df["study"].dropna().unique()):
        study_df = subject_df[subject_df["study"] == study]
        hemisphere_order = [hemi for hemi in ["left", "right"] if hemi in set(study_df["hemisphere"])]
        if not hemisphere_order:
            hemisphere_order = list(dict.fromkeys(study_df["hemisphere"].dropna().astype(str)))
        for metric in metrics:
            cols = ["subid", "role", "hemisphere", "network", metric]
            data = study_df[cols].rename(columns={metric: "value"}).copy()
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
                            "study": study,
                            "metric": metric,
                            "metric_label": SELECTED_METRIC_LABELS[metric],
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
                            "study": study,
                            "metric": metric,
                            "metric_label": SELECTED_METRIC_LABELS[metric],
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
                            "study": study,
                            "metric": metric,
                            "metric_label": SELECTED_METRIC_LABELS[metric],
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


def _log_ratio_to_percent(value: float) -> float:
    if not np.isfinite(value):
        return math.nan
    return float(100.0 * np.expm1(value))


def _finite_axis_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if lo == hi:
        pad = max(abs(lo) * 0.1, 0.05)
    else:
        pad = max((hi - lo) * 0.08, 0.05)
    return lo - pad, hi + pad


def _log_ratio_to_percent_label(x: float, _pos: int) -> str:
    pct = _log_ratio_to_percent(x)
    if not np.isfinite(pct):
        return ""
    if abs(pct) >= 100:
        return f"{pct:.0f}%"
    if abs(pct) >= 10:
        return f"{pct:.0f}%"
    if abs(pct) >= 1:
        return f"{pct:.1f}%"
    return f"{pct:.2f}%"


def _plot_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    x_grid: np.ndarray,
    y0: float,
    height: float,
    color: str,
    linestyle: str,
    linewidth: float,
    direction: float,
) -> None:
    finite = values[np.isfinite(values)]
    mean_value = float(np.mean(finite)) if finite.size else math.nan
    if finite.size >= 2 and np.nanstd(finite) > 0:
        try:
            kde = stats.gaussian_kde(finite)
            dens = kde(x_grid)
            max_dens = float(np.nanmax(dens))
            if max_dens > 0:
                dens = dens / max_dens * height
                ax.plot(
                    x_grid,
                    y0 + direction * dens,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=DIST_ALPHA,
                )
                if np.isfinite(mean_value):
                    ax.vlines(
                        mean_value,
                        y0,
                        y0 + direction * height * MEAN_TICK_FRACTION,
                        color=color,
                        linestyle="-",
                        linewidth=MEAN_LINEWIDTH,
                        alpha=MEAN_ALPHA,
                    )
                return
        except Exception:
            pass
    if finite.size:
        jitter = direction * np.linspace(0.08, height * 0.65, finite.size)
        ax.scatter(finite, y0 + jitter, color=color, marker="o", s=14, alpha=0.75, linewidths=0)
        if np.isfinite(mean_value):
            ax.vlines(
                mean_value,
                y0,
                y0 + direction * height * MEAN_TICK_FRACTION,
                color=color,
                linestyle="-",
                linewidth=MEAN_LINEWIDTH,
                alpha=MEAN_ALPHA,
            )


def save_combined_ridgeline_figure(
    plot_df: pd.DataFrame,
    fig_dir: Path,
    network_order: list[str],
    kde_points: int,
    value_col: str,
    filename: str,
    xlabel: str,
    x_formatter: FuncFormatter | None = None,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [metric for metric in SELECTED_METRIC_LABELS if metric in set(plot_df["metric"])]
    fig, axes = plt.subplots(2, 2, figsize=(19, 15), squeeze=False)
    axes_flat = axes.flat
    row_gap = 1.65
    ridge_height = 0.62
    hemi_direction = {"left": 1.0, "right": -1.0}
    y_positions = {network: (len(network_order) - 1 - idx) * row_gap for idx, network in enumerate(network_order)}

    for ax, metric in zip(axes_flat, metrics):
        mdf = plot_df[plot_df["metric"] == metric].copy()
        x_min, x_max = _finite_axis_limits(mdf[value_col])
        x_grid = np.linspace(x_min, x_max, kde_points)
        for network in network_order:
            base_y = y_positions[network]
            ax.hlines(base_y, x_min, x_max, color="#777777", linewidth=0.6, alpha=0.45)
            for hemi, direction in hemi_direction.items():
                for study, style in STUDY_STYLE.items():
                    vals = mdf[
                        (mdf["network"] == network)
                        & (mdf["study"] == study)
                        & (mdf["hemisphere"] == hemi)
                    ][value_col].to_numpy(dtype=float)
                    _plot_distribution(
                        ax=ax,
                        values=vals,
                        x_grid=x_grid,
                        y0=base_y,
                        height=ridge_height,
                        color=style["color"],
                        linestyle=HEMI_LINESTYLE[hemi],
                        linewidth=style["linewidth"],
                        direction=direction,
                    )
        ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.85)
        ax.set_title(SELECTED_METRIC_LABELS[metric], pad=12, fontsize=FONT_SIZES["axis_title"])
        ax.set_yticks([y_positions[n] for n in network_order])
        ax.set_yticklabels(network_order, fontsize=FONT_SIZES["tick"])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-ridge_height - 0.2, max(y_positions.values()) + ridge_height + 0.2)
        ax.set_xlabel(
            xlabel,
            fontsize=FONT_SIZES["axis_label"],
        )
        ax.set_ylabel("")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        if x_formatter is not None:
            ax.xaxis.set_major_formatter(x_formatter)
        ax.grid(axis="x", visible=True, linewidth=0.5, alpha=0.35)
        ax.grid(axis="y", visible=False)

    for ax in list(axes_flat)[len(metrics) :]:
        ax.axis("off")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=STUDY_STYLE["DMT"]["color"],
            linewidth=STUDY_STYLE["DMT"]["linewidth"],
            label="DMT",
        ),
        plt.Line2D(
            [0],
            [0],
            color=STUDY_STYLE["LSD"]["color"],
            linewidth=STUDY_STYLE["LSD"]["linewidth"],
            label="LSD",
        ),
        plt.Line2D([0], [0], color="#444444", linewidth=2.4, linestyle=HEMI_LINESTYLE["left"], label="left: above"),
        plt.Line2D([0], [0], color="#444444", linewidth=2.4, linestyle=HEMI_LINESTYLE["right"], label="right: below"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZES["legend"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(fig_dir / f"{filename}.pdf", bbox_inches="tight")
    except PermissionError as exc:
        print(f"Warning: could not overwrite PDF figure, likely open in another program: {exc}")
    plt.close(fig)


def save_combined_log_ratio_figure(
    log_df: pd.DataFrame,
    fig_dir: Path,
    network_order: list[str],
    kde_points: int,
) -> None:
    save_combined_ridgeline_figure(
        plot_df=log_df,
        fig_dir=fig_dir,
        network_order=network_order,
        kde_points=kde_points,
        value_col="log_ratio_drug_vs_pcb",
        filename="combined_ln_ratio_metric_panels",
        xlabel="Drug vs PCB change (ticks relabeled from ln ratio)",
        x_formatter=FuncFormatter(_log_ratio_to_percent_label),
    )


def save_combined_standardized_difference_figure(
    std_df: pd.DataFrame,
    fig_dir: Path,
    network_order: list[str],
    kde_points: int,
) -> None:
    save_combined_ridgeline_figure(
        plot_df=std_df,
        fig_dir=fig_dir,
        network_order=network_order,
        kde_points=kde_points,
        value_col="standardized_difference",
        filename="combined_standardized_difference_metric_panels",
        xlabel="Standardized difference: (Drug - PCB) / SD_PCB",
        x_formatter=None,
    )


def main() -> int:
    args = parse_args()
    study_inputs = args.study_input if args.study_input is not None else list(STUDY_INPUT_DEFAULTS.items())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subject_df = load_subject_tables(study_inputs)
    network_order = infer_network_order(subject_df["network"])
    paired_df = compute_paired_values(subject_df)
    log_df = compute_log_ratio(paired_df)
    std_df = compute_standardized_difference(paired_df)
    metric_order = list(SELECTED_METRIC_LABELS)
    paired_df["network"] = pd.Categorical(paired_df["network"], categories=network_order, ordered=True)
    paired_df["metric"] = pd.Categorical(paired_df["metric"], categories=metric_order, ordered=True)
    paired_df = paired_df.sort_values(["metric", "network", "study", "hemisphere", "subid"]).reset_index(drop=True)
    log_df["network"] = pd.Categorical(log_df["network"], categories=network_order, ordered=True)
    log_df["metric"] = pd.Categorical(log_df["metric"], categories=metric_order, ordered=True)
    log_df = log_df.sort_values(["metric", "network", "study", "hemisphere", "subid"]).reset_index(drop=True)
    std_df["network"] = pd.Categorical(std_df["network"], categories=network_order, ordered=True)
    std_df["metric"] = pd.Categorical(std_df["metric"], categories=metric_order, ordered=True)
    std_df = std_df.sort_values(["metric", "network", "study", "hemisphere", "subid"]).reset_index(drop=True)
    summary_df = summarize_log_ratio(log_df, min_pairs=args.min_pairs)
    summary_df["network"] = pd.Categorical(summary_df["network"], categories=network_order, ordered=True)
    summary_df["metric"] = pd.Categorical(summary_df["metric"], categories=metric_order, ordered=True)
    summary_df = summary_df.sort_values(["metric", "network", "study", "hemisphere"]).reset_index(drop=True)
    std_summary_df = summarize_standardized_difference(std_df, min_pairs=args.min_pairs)
    std_summary_df["network"] = pd.Categorical(std_summary_df["network"], categories=network_order, ordered=True)
    std_summary_df["metric"] = pd.Categorical(std_summary_df["metric"], categories=metric_order, ordered=True)
    std_summary_df = std_summary_df.sort_values(["metric", "network", "study", "hemisphere"]).reset_index(drop=True)
    basic_stats_df = group_basic_stats_paired_only(paired_df)
    basic_stats_df["network"] = pd.Categorical(basic_stats_df["network"], categories=network_order, ordered=True)
    basic_stats_df["metric"] = pd.Categorical(basic_stats_df["metric"], categories=metric_order, ordered=True)
    basic_stats_df = basic_stats_df.sort_values(["metric", "network", "study", "hemisphere", "role"]).reset_index(drop=True)
    paired_ttest_df = paired_ttest_raw_summary(paired_df, min_pairs=args.min_pairs)
    paired_ttest_df["network"] = pd.Categorical(paired_ttest_df["network"], categories=network_order, ordered=True)
    paired_ttest_df["metric"] = pd.Categorical(paired_ttest_df["metric"], categories=metric_order, ordered=True)
    paired_ttest_df = paired_ttest_df.sort_values(["metric", "network", "study", "hemisphere"]).reset_index(drop=True)
    two_way_anova_df = two_way_rm_anova(subject_df)
    two_way_anova_df["metric"] = pd.Categorical(two_way_anova_df["metric"], categories=metric_order, ordered=True)
    two_way_anova_df = two_way_anova_df.sort_values(["metric", "study", "hemisphere", "effect"]).reset_index(drop=True)
    three_way_anova_df = three_way_rm_anova(subject_df)
    three_way_anova_df["metric"] = pd.Categorical(three_way_anova_df["metric"], categories=metric_order, ordered=True)
    three_way_anova_df = three_way_anova_df.sort_values(["metric", "study", "effect"]).reset_index(drop=True)

    paired_df.to_csv(args.out_dir / "subject_paired_values_wide.csv", index=False)
    log_df.to_csv(args.out_dir / "subject_log_ratio_long.csv", index=False)
    summary_df.to_csv(args.out_dir / "log_ratio_summary.csv", index=False)
    std_df.to_csv(args.out_dir / "subject_standardized_difference_long.csv", index=False)
    std_summary_df.to_csv(args.out_dir / "standardized_difference_summary.csv", index=False)
    basic_stats_df.to_csv(args.out_dir / "group_basic_stats_paired_only.csv", index=False)
    paired_ttest_df.to_csv(args.out_dir / "paired_ttest_raw_summary.csv", index=False)
    two_way_anova_df.to_csv(args.out_dir / "two_way_rm_anova.csv", index=False)
    three_way_anova_df.to_csv(args.out_dir / "three_way_rm_anova.csv", index=False)

    metadata = {
        "study_inputs": {study: str(path) for study, path in study_inputs},
        "selected_metrics": SELECTED_METRIC_LABELS,
        "log_ratio_formula": "ln(drug_value / pcb_value)",
        "tick_percent_formula": "100 * (exp(log_ratio_drug_vs_pcb) - 1)",
        "ratio_validity": "Rows with nonpositive or missing Drug/PCB values have NaN log_ratio.",
        "standardized_difference_formula": "(drug_value - pcb_value) / SD_PCB",
        "sd_pcb_scope": "study x hemisphere x network x metric across paired PCB subject values",
        "paired_ttest": "Within each study x hemisphere x network x metric, scipy.stats.ttest_rel(Drug, PCB) on raw metric values.",
        "rm_anova": "Within each study: two-way role x network repeated-measures ANOVA per hemisphere; three-way role x hemisphere x network repeated-measures ANOVA.",
        "plot_layout": "Left hemisphere KDEs are drawn above each network baseline; right hemisphere KDEs below.",
        "network_order": network_order,
        "n_subject_paired_rows": int(len(paired_df)),
        "n_subject_log_ratio_rows": int(len(log_df)),
        "n_subject_standardized_difference_rows": int(len(std_df)),
        "n_group_basic_stats_rows": int(len(basic_stats_df)),
        "n_paired_ttest_rows": int(len(paired_ttest_df)),
        "n_two_way_rm_anova_rows": int(len(two_way_anova_df)),
        "n_three_way_rm_anova_rows": int(len(three_way_anova_df)),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.no_plots:
        save_combined_log_ratio_figure(
            log_df=log_df,
            fig_dir=args.out_dir / "figures",
            network_order=network_order,
            kde_points=args.kde_points,
        )
        save_combined_standardized_difference_figure(
            std_df=std_df,
            fig_dir=args.out_dir / "figures",
            network_order=network_order,
            kde_points=args.kde_points,
        )

    print(f"Wrote v2 ln-ratio, standardized-difference, paired t-test, and RM-ANOVA outputs to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
