from __future__ import annotations

import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import pingouin as pg
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.use("Agg")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

PALETTE_DEFAULT = {"PCB": "#4575b4", "DMT": "#d73027"}
OUTPUT_PREFIX_DEFAULT = ""
OUTPUT_DATE_DEFAULT = ""


def _clean_name_token(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


OUTPUT_PREFIX = OUTPUT_PREFIX_DEFAULT
OUTPUT_DATE = OUTPUT_DATE_DEFAULT


class TqdmConsoleHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg, file=self.stream)
            else:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def set_output_naming(prefix: str | None = None, date_stamp: str | None = None) -> None:
    global OUTPUT_PREFIX, OUTPUT_DATE
    OUTPUT_PREFIX = _clean_name_token(prefix or "")
    OUTPUT_DATE = _clean_name_token(
        date_stamp or datetime.now().strftime("%Y%m%d")
    )


def build_prefixed_dated_path(path: Path) -> Path | None:
    if not OUTPUT_PREFIX and not OUTPUT_DATE:
        return None
    stem = path.stem
    prefix_part = f"{OUTPUT_PREFIX}_" if OUTPUT_PREFIX else ""
    date_part = f"_{OUTPUT_DATE}" if OUTPUT_DATE else ""
    return path.with_name(f"{prefix_part}{stem}{date_part}{path.suffix}")


def resolve_reference_gmap(cfg, hemisphere: str | None) -> Path | None:
    hemi = str(hemisphere or "").lower()
    if hemi == "left" and getattr(cfg, "reference_gmap_left", None):
        return Path(cfg.reference_gmap_left)
    if hemi == "right" and getattr(cfg, "reference_gmap_right", None):
        return Path(cfg.reference_gmap_right)
    ref = getattr(cfg, "reference_gmap", None)
    return Path(ref) if ref else None


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run_all_metrics.log"
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            TqdmConsoleHandler(stream=sys.stderr),
        ],
        force=True,
    )
    logging.info("Logging to %s", log_file)


def ci95(mean: float, sem: float) -> tuple[float, float]:
    if math.isnan(mean) or math.isnan(sem):
        return (math.nan, math.nan)
    return (mean - 1.96 * sem, mean + 1.96 * sem)


def summarize_series(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "sem": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    lo, hi = ci95(mean, sem)
    return {"n": n, "mean": mean, "std": std, "sem": sem, "ci95_lo": lo, "ci95_hi": hi}


def paired_t(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return {"n": 0, "t": math.nan, "p": math.nan, "dz": math.nan}
    diff = (a[mask] - b[mask]).to_numpy(dtype=float)
    t, p = stats.ttest_rel(a[mask], b[mask], nan_policy="omit")
    diff_std = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    if diff.size == 0 or diff_std == 0.0:
        dz = math.nan
    else:
        dz = float(np.mean(diff) / diff_std)
    return {"n": int(mask.sum()), "t": float(t), "p": float(p), "dz": dz}


def unpaired_t(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return {"n1": len(a), "n2": len(b), "t": math.nan, "p": math.nan}
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return {"n1": len(a), "n2": len(b), "t": float(t), "p": float(p)}


def save_fig(fig: plt.Figure, path: Path, save: bool) -> None:
    if not save:
        plt.close(fig)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    alias_path = build_prefixed_dated_path(path)
    if alias_path is not None and alias_path != path:
        df.to_csv(alias_path, index=False)


def add_tests(rows: list[dict], hemi: str, metric: str, drug: pd.Series, pcb: pd.Series, cfg) -> None:
    """Append paired/unpaired test results into summary rows for CSV output.

    Align indices for paired tests to avoid zero overlap when group Series
    come from different subjects or have disjoint integer indices.
    """
    # align by shared index (e.g., subid) if available
    common_idx = drug.index.intersection(pcb.index)
    drug_aligned = drug.loc[common_idx]
    pcb_aligned = pcb.loc[common_idx]

    # fall back to positional alignment if no shared index names
    if common_idx.empty and len(drug) and len(pcb):
        min_len = min(len(drug), len(pcb))
        drug_aligned = drug.reset_index(drop=True).iloc[:min_len]
        pcb_aligned = pcb.reset_index(drop=True).iloc[:min_len]

    pt = paired_t(drug_aligned, pcb_aligned)
    pair_mask = drug_aligned.notna() & pcb_aligned.notna()
    n_pair = int(pair_mask.sum())

    pt_row = {
        "hemisphere": hemi,
        "group": f"{cfg.group_drug}_vs_{cfg.group_pcb}",
        "metric": metric,
        "comparison": "paired",
        "n1": n_pair,
        "n2": n_pair,
    }
    pt_row.update(pt)
    rows.append(pt_row)

    ut = unpaired_t(drug, pcb)
    ut_row = {
        "hemisphere": hemi,
        "group": f"{cfg.group_drug}_vs_{cfg.group_pcb}",
        "metric": metric,
        "comparison": "unpaired",
    }
    ut_row.update(ut)
    rows.append(ut_row)


def plot_abs_effect_size_bars(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Absolute Cohen's dz by metric",
    max_items: int = 40,
    save: bool = True,
) -> None:
    if summary_df.empty or "dz" not in summary_df.columns:
        return
    plot_df = summary_df.copy()
    plot_df["dz"] = pd.to_numeric(plot_df["dz"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["dz"])].copy()
    if plot_df.empty:
        return
    if "comparison" in plot_df.columns:
        plot_df = plot_df[plot_df["comparison"] == "paired_drug_vs_pcb"].copy()
    if plot_df.empty:
        return
    plot_df["abs_dz"] = plot_df["dz"].abs()
    hemi = plot_df.get("hemisphere", pd.Series([""] * len(plot_df))).fillna("").astype(str)
    section = plot_df.get("section", pd.Series([""] * len(plot_df))).fillna("").astype(str)
    metric = plot_df.get("metric", pd.Series([""] * len(plot_df))).fillna("").astype(str)
    plot_df["label"] = section + ":" + metric + np.where(hemi.ne(""), " (" + hemi + ")", "")
    plot_df = plot_df.sort_values("abs_dz", ascending=False).head(max_items)
    if plot_df.empty:
        return
    fig_h = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.barplot(data=plot_df, x="abs_dz", y="label", color="#4c78a8", ax=ax)
    ax.set_xlabel("|Cohen's dz|")
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_path, save)


def get_palette(cfg) -> Dict[str, str]:
    pal = dict(PALETTE_DEFAULT)
    pal.setdefault(cfg.group_pcb, PALETTE_DEFAULT.get("PCB", "#4575b4"))
    pal.setdefault(cfg.group_drug, PALETTE_DEFAULT.get("DMT", "#d73027"))
    pal.setdefault(cfg.group_drug, PALETTE_DEFAULT.get("LSD", "#d73027"))
    pal[cfg.group_pcb] = pal[cfg.group_pcb]
    pal[cfg.group_drug] = pal[cfg.group_drug]
    return pal


def resolve_bundle_dir(row: pd.Series, cfg) -> Path | None:
    bdir_val = row.get("bundle_dir")
    if bdir_val is None or (isinstance(bdir_val, float) and math.isnan(bdir_val)):
        return None
    bdir = Path(str(bdir_val))
    if not bdir.is_absolute():
        if bdir.exists():
            return bdir.resolve()
        cand = Path(cfg.detect_results_dir) / bdir
        if cand.exists():
            return cand
        grp = row.get("group")
        if grp:
            cand2 = Path(cfg.detect_results_dir) / str(grp) / bdir
            if cand2.exists():
                return cand2
        return None
    return bdir


def plot_paired_violin(tidy: pd.DataFrame, metric: str, hemi: str, title: str, out_path: Path, cfg) -> None:
    if tidy.empty:
        return
    pivot = tidy.pivot(index="subid", columns="group", values=metric)
    if cfg.group_pcb not in pivot.columns or cfg.group_drug not in pivot.columns:
        return
    pivot = pivot[[cfg.group_pcb, cfg.group_drug]].dropna()
    if pivot.empty:
        return
    pal = get_palette(cfg)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(data=tidy, x="group", y=metric, order=[
                   cfg.group_pcb, cfg.group_drug], palette=pal, cut=0, ax=ax)
    x_lookup = {cfg.group_pcb: 0, cfg.group_drug: 1}
    for _, row in pivot.iterrows():
        ax.plot([x_lookup[cfg.group_pcb], x_lookup[cfg.group_drug]], [
                row[cfg.group_pcb], row[cfg.group_drug]], color="gray", alpha=0.5, linewidth=0.9, zorder=1)
        ax.scatter([x_lookup[cfg.group_pcb], x_lookup[cfg.group_drug]], [row[cfg.group_pcb], row[cfg.group_drug]], color=[
                   pal[cfg.group_pcb], pal[cfg.group_drug]], edgecolor="white", linewidth=0.5, s=25, zorder=2)
    ax.set_title(f"{title} ({hemi})")
    ax.set_xlabel("")
    fig.tight_layout()
    save_fig(fig, out_path, cfg.save_plots)


def build_group_summary_df(
    df: pd.DataFrame,
    metric_cols: list[str],
    cfg,
    group_col: str = "group",
    pair_col: str = "subid",
    within_col: str = "hemisphere",
) -> pd.DataFrame:
    if df.empty or not metric_cols:
        return pd.DataFrame()

    out = df.copy()
    out[group_col] = out[group_col].astype(str)
    if pair_col in out.columns:
        out[pair_col] = out[pair_col].astype(str)
    if within_col in out.columns:
        out[within_col] = out[within_col].astype(str)
    groups = [g for g in [cfg.group_pcb, cfg.group_drug] if g in set(out[group_col])]
    if len(groups) < 2:
        groups = list(dict.fromkeys(out[group_col].dropna().astype(str).tolist()))
    if len(groups) < 2:
        return pd.DataFrame()

    def paired_only_slice(work: pd.DataFrame) -> pd.DataFrame:
        if pair_col not in work.columns:
            return work.iloc[0:0].copy()
        counts = work.groupby(pair_col, dropna=False)[group_col].nunique(dropna=True)
        keep_ids = counts[counts == len(groups)].index
        return work[work[pair_col].isin(keep_ids)].copy()

    rows: list[dict[str, object]] = []
    for hemi in ["left", "right", "combined"]:
        slice_df = out if hemi == "combined" or within_col not in out.columns else out[out[within_col] == hemi].copy()
        if slice_df.empty:
            continue
        paired_df = paired_only_slice(slice_df)
        for subset_name, subset_df in [("all_subjects", slice_df), ("paired_only", paired_df)]:
            for grp in groups:
                grp_df = subset_df[subset_df[group_col] == grp]
                for metric in metric_cols:
                    stat_row = {
                        "row_type": "summary",
                        "subset": subset_name,
                        "metric": metric,
                        "hemisphere": hemi,
                        "group": grp,
                        "comparison": "",
                        "test_type": "",
                    }
                    stat_row.update(summarize_series(grp_df[metric]))
                    rows.append(stat_row)

        if len(groups) >= 2:
            group_a, group_b = groups[:2]
            for metric in metric_cols:
                if pair_col in slice_df.columns:
                    pivot = (
                        slice_df[[pair_col, group_col, metric]]
                        .pivot_table(index=pair_col, columns=group_col, values=metric, aggfunc="mean")
                    )
                    if group_b in pivot.columns and group_a in pivot.columns:
                        pair_stats = paired_t(pivot[group_b], pivot[group_a])
                        pair_n = pair_stats.get("n", math.nan)
                    else:
                        pair_stats = {"n": 0, "t": math.nan, "p": math.nan}
                        pair_n = 0
                else:
                    pair_stats = {"n": 0, "t": math.nan, "p": math.nan}
                    pair_n = 0
                rows.append(
                    {
                        "row_type": "test",
                        "subset": "paired_only",
                        "metric": metric,
                        "hemisphere": hemi,
                        "group": "",
                        "comparison": f"{group_b}_vs_{group_a}",
                        "test_type": "paired_ttest",
                        "n1": pair_n,
                        "n2": pair_n,
                        "t": pair_stats.get("t", math.nan),
                        "p": pair_stats.get("p", math.nan),
                        "dz": pair_stats.get("dz", math.nan),
                    }
                )
                unpair = unpaired_t(
                    slice_df[slice_df[group_col] == group_b][metric],
                    slice_df[slice_df[group_col] == group_a][metric],
                )
                rows.append(
                    {
                        "row_type": "test",
                        "subset": "all_subjects",
                        "metric": metric,
                        "hemisphere": hemi,
                        "group": "",
                        "comparison": f"{group_b}_vs_{group_a}",
                        "test_type": "unpaired_ttest",
                        **unpair,
                    }
                )

        if hemi == "combined" and within_col in out.columns:
            for metric in metric_cols:
                work = out[[pair_col, group_col, within_col, metric]].copy()
                work[metric] = pd.to_numeric(work[metric], errors="coerce")
                work = work.dropna(subset=[pair_col, group_col, within_col, metric])
                if work.empty or work[group_col].nunique() < 2 or work[within_col].nunique() < 2:
                    continue
                expected = int(work[group_col].nunique() * work[within_col].nunique())
                counts = work.groupby(pair_col, dropna=False).apply(
                    lambda x: x[[group_col, within_col]].drop_duplicates().shape[0],
                    include_groups=False,
                )
                keep_ids = counts[counts == expected].index
                work = work[work[pair_col].isin(keep_ids)].copy()
                if work[pair_col].nunique() < 2:
                    continue
                work = work.groupby([pair_col, group_col, within_col], as_index=False)[metric].mean()
                try:
                    aov = pg.rm_anova(data=work, dv=metric, within=[group_col, within_col], subject=pair_col, detailed=True)
                except Exception:
                    continue
                for _, effect_row in aov.iterrows():
                    f_val = float(effect_row["F"]) if pd.notna(effect_row.get("F")) else math.nan
                    ddof1 = float(effect_row["ddof1"]) if pd.notna(effect_row.get("ddof1")) else math.nan
                    ddof2 = float(effect_row["ddof2"]) if pd.notna(effect_row.get("ddof2")) else math.nan
                    eta_p2 = float((f_val * ddof1) / ((f_val * ddof1) + ddof2)) if pd.notna(f_val) and pd.notna(ddof1) and pd.notna(ddof2) and ((f_val * ddof1) + ddof2) != 0 else math.nan
                    rows.append(
                        {
                            "row_type": "test",
                            "subset": "paired_only",
                            "metric": metric,
                            "hemisphere": hemi,
                            "group": "",
                            "comparison": f"{group_col}_x_{within_col}",
                            "test_type": "two_way_rm_anova",
                            "effect": effect_row.get("Source", ""),
                            "n": int(work[pair_col].nunique()),
                            "p": float(effect_row["p-unc"]) if pd.notna(effect_row.get("p-unc")) else math.nan,
                            "F": f_val,
                            "ddof1": ddof1,
                            "ddof2": ddof2,
                            "eta_p2": eta_p2,
                            "ng2": float(effect_row["ng2"]) if pd.notna(effect_row.get("ng2")) else math.nan,
                            "eps": float(effect_row["eps"]) if pd.notna(effect_row.get("eps")) else math.nan,
                            "anova_within_col": within_col,
                        }
                    )
    return pd.DataFrame(rows)
