from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats


DEFAULT_GROUP_COL = "group"
DEFAULT_PAIR_COL = "subid"
DEFAULT_OUTPUT_NAME = "recomputed_group_summary.csv"
COMBINED_LABEL = "combined"
PLOT_DIRNAME = "interaction_plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute summary statistics for every per_subject.csv under a root path. "
            "For each file, save all-subject summaries, paired-only summaries, "
            "unpaired t-tests, and paired t-tests beside the source file."
        )
    )
    parser.add_argument("root", type=Path, help="Root directory to scan recursively.")
    parser.add_argument(
        "--filename",
        default="per_subject.csv",
        help="Target per-subject filename to scan for. Default: per_subject.csv",
    )
    parser.add_argument(
        "--group-col",
        default=DEFAULT_GROUP_COL,
        help=f"Group column name. Default: {DEFAULT_GROUP_COL}",
    )
    parser.add_argument(
        "--pair-col",
        default=DEFAULT_PAIR_COL,
        help=f"Pairing column name. Default: {DEFAULT_PAIR_COL}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename written next to each per_subject file. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--anova-within-col",
        default=None,
        help=(
            "Second within-subject factor for two-way repeated-measures ANOVA. "
            "If omitted, the script will try to auto-detect it, preferring 'hemisphere'."
        ),
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional prefix added to dated output filenames. Defaults to root folder name.",
    )
    parser.add_argument(
        "--date-stamp",
        default=None,
        help="Optional YYYYMMDD-style suffix for dated output filenames. Defaults to today.",
    )
    return parser.parse_args()


def build_dated_output_path(path: Path, prefix: str | None, date_stamp: str | None) -> Path:
    clean_prefix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(prefix or "")).strip("_")
    clean_date = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(date_stamp or "")).strip("_")
    stem = path.stem
    prefix_part = f"{clean_prefix}_" if clean_prefix else ""
    date_part = f"_{clean_date}" if clean_date else ""
    return path.with_name(f"{prefix_part}{stem}{date_part}{path.suffix}")


def save_csv_with_alias(df: pd.DataFrame, out_path: Path, prefix: str | None, date_stamp: str | None) -> Path:
    df.to_csv(out_path, index=False)
    alias = build_dated_output_path(out_path, prefix=prefix, date_stamp=date_stamp)
    if alias != out_path:
        df.to_csv(alias, index=False)
    return alias


def summarize_numeric(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "ci95_lo": math.nan,
            "ci95_hi": math.nan,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_lo": mean - 1.96 * sem,
        "ci95_hi": mean + 1.96 * sem,
    }


def sem_numeric(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(arr.size)
    if n == 0:
        return math.nan
    if n == 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(n))


def unpaired_ttest(a: pd.Series, b: pd.Series) -> dict[str, float]:
    arr_a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if arr_a.size == 0 or arr_b.size == 0:
        return {"n1": int(arr_a.size), "n2": int(arr_b.size), "t": math.nan, "p": math.nan}
    t_stat, p_val = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
    return {"n1": int(arr_a.size), "n2": int(arr_b.size), "t": float(t_stat), "p": float(p_val)}


def paired_ttest(a: pd.Series, b: pd.Series) -> dict[str, float]:
    arr_a = pd.to_numeric(a, errors="coerce")
    arr_b = pd.to_numeric(b, errors="coerce")
    mask = arr_a.notna() & arr_b.notna()
    if int(mask.sum()) == 0:
        return {"n1": 0, "n2": 0, "t": math.nan, "p": math.nan}
    t_stat, p_val = stats.ttest_rel(arr_a[mask], arr_b[mask], nan_policy="omit")
    n = int(mask.sum())
    return {"n1": n, "n2": n, "t": float(t_stat), "p": float(p_val)}


def detect_metric_columns(df: pd.DataFrame, reserved: set[str]) -> list[str]:
    metrics: list[str] = []
    for col in df.columns:
        if col in reserved:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            metrics.append(col)
    return metrics


def normalize_frame(df: pd.DataFrame, group_col: str, pair_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in {group_col, pair_col}:
            out[col] = out[col].astype(str)
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().any():
            out[col] = numeric
        else:
            out[col] = out[col].astype(str)
    return out


def iter_slices(df: pd.DataFrame, strata_cols: list[str]) -> Iterable[tuple[dict[str, str], pd.DataFrame]]:
    if not strata_cols:
        yield {}, df
        return

    grouped = df.groupby(strata_cols, dropna=False, sort=False)
    for keys, sub_df in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        strata = {}
        for col, val in zip(strata_cols, keys):
            strata[col] = "" if pd.isna(val) else str(val)
        yield strata, sub_df.copy()

    combined = {col: COMBINED_LABEL for col in strata_cols}
    yield combined, df.copy()


def paired_subset(df: pd.DataFrame, pair_col: str, group_col: str, groups: list[str]) -> pd.DataFrame:
    if pair_col not in df.columns:
        return df.iloc[0:0].copy()
    valid = df[df[pair_col].notna()].copy()
    if valid.empty:
        return valid
    counts = (
        valid.groupby(pair_col, dropna=False)[group_col]
        .nunique(dropna=True)
    )
    keep_ids = counts[counts == len(groups)].index
    return valid[valid[pair_col].isin(keep_ids)].copy()


def build_summary_rows(
    df: pd.DataFrame,
    subset_name: str,
    metric_cols: list[str],
    strata: dict[str, str],
    group_col: str,
    groups: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        for group in groups:
            values = df.loc[df[group_col] == group, metric]
            row: dict[str, object] = {
                "row_type": "summary",
                "subset": subset_name,
                "metric": metric,
                "group": group,
                "comparison": "",
                "test_type": "",
                "n1": math.nan,
                "n2": math.nan,
                "t": math.nan,
                "p": math.nan,
            }
            row.update(strata)
            row.update(summarize_numeric(values))
            rows.append(row)
    return rows


def build_test_rows(
    all_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    metric_cols: list[str],
    strata: dict[str, str],
    group_col: str,
    pair_col: str,
    groups: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if len(groups) != 2:
        return rows

    group_a, group_b = groups
    comparison = f"{group_a}_vs_{group_b}"
    for metric in metric_cols:
        row_unpaired: dict[str, object] = {
            "row_type": "test",
            "subset": "all_subjects",
            "metric": metric,
            "group": "",
            "comparison": comparison,
            "test_type": "unpaired_ttest",
            "n": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "ci95_lo": math.nan,
            "ci95_hi": math.nan,
        }
        row_unpaired.update(strata)
        row_unpaired.update(
            unpaired_ttest(
                all_df.loc[all_df[group_col] == group_a, metric],
                all_df.loc[all_df[group_col] == group_b, metric],
            )
        )
        rows.append(row_unpaired)

        pivot = (
            paired_df[[pair_col, group_col, metric]]
            .pivot_table(index=pair_col, columns=group_col, values=metric, aggfunc="mean")
        )
        if group_a in pivot.columns and group_b in pivot.columns:
            pair_stats = paired_ttest(pivot[group_a], pivot[group_b])
        else:
            pair_stats = {"n1": 0, "n2": 0, "t": math.nan, "p": math.nan}
        row_paired: dict[str, object] = {
            "row_type": "test",
            "subset": "paired_only",
            "metric": metric,
            "group": "",
            "comparison": comparison,
            "test_type": "paired_ttest",
            "n": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "ci95_lo": math.nan,
            "ci95_hi": math.nan,
        }
        row_paired.update(strata)
        row_paired.update(pair_stats)
        rows.append(row_paired)
    return rows


def choose_anova_within_col(strata_cols: list[str], requested: str | None) -> str | None:
    if requested:
        return requested if requested in strata_cols else None
    if "hemisphere" in strata_cols:
        return "hemisphere"
    if len(strata_cols) == 1:
        return strata_cols[0]
    return None


def build_two_way_rm_anova_rows(
    df: pd.DataFrame,
    metric_cols: list[str],
    strata: dict[str, str],
    group_col: str,
    pair_col: str,
    anova_within_col: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not anova_within_col or anova_within_col not in df.columns or pair_col not in df.columns:
        return rows

    if len(df[group_col].dropna().unique()) < 2 or len(df[anova_within_col].dropna().unique()) < 2:
        return rows

    for metric in metric_cols:
        work = df[[pair_col, group_col, anova_within_col, metric]].copy()
        work = work.dropna(subset=[pair_col, group_col, anova_within_col])
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[metric])
        if work.empty:
            continue

        # Keep only subjects with a full repeated-measures cell set.
        n_group = int(work[group_col].nunique())
        n_within = int(work[anova_within_col].nunique())
        expected_cells = n_group * n_within
        counts = (
            work.groupby(pair_col, dropna=False)
            .apply(lambda x: x[[group_col, anova_within_col]].drop_duplicates().shape[0], include_groups=False)
        )
        keep_ids = counts[counts == expected_cells].index
        work = work[work[pair_col].isin(keep_ids)].copy()
        if work[pair_col].nunique() < 2:
            continue

        work = (
            work.groupby([pair_col, group_col, anova_within_col], as_index=False)[metric]
            .mean()
        )
        try:
            aov = pg.rm_anova(
                data=work,
                dv=metric,
                within=[group_col, anova_within_col],
                subject=pair_col,
                detailed=True,
            )
        except Exception:
            continue

        for _, effect_row in aov.iterrows():
            f_val = float(effect_row["F"]) if pd.notna(effect_row.get("F")) else math.nan
            ddof1 = float(effect_row["ddof1"]) if pd.notna(effect_row.get("ddof1")) else math.nan
            ddof2 = float(effect_row["ddof2"]) if pd.notna(effect_row.get("ddof2")) else math.nan
            if pd.notna(f_val) and pd.notna(ddof1) and pd.notna(ddof2) and ((f_val * ddof1) + ddof2) != 0:
                eta_p2 = float((f_val * ddof1) / ((f_val * ddof1) + ddof2))
            else:
                eta_p2 = math.nan
            row: dict[str, object] = {
                "row_type": "test",
                "subset": "paired_only",
                "metric": metric,
                "group": "",
                "comparison": f"{group_col}_x_{anova_within_col}",
                "test_type": "two_way_rm_anova",
                "effect": effect_row.get("Source", ""),
                "n": int(work[pair_col].nunique()),
                "n1": math.nan,
                "n2": math.nan,
                "t": math.nan,
                "p": float(effect_row["p-unc"]) if pd.notna(effect_row.get("p-unc")) else math.nan,
                "mean": math.nan,
                "std": math.nan,
                "sem": math.nan,
                "ci95_lo": math.nan,
                "ci95_hi": math.nan,
                "F": f_val,
                "ddof1": ddof1,
                "ddof2": ddof2,
                "eta_p2": eta_p2,
                "ng2": float(effect_row["ng2"]) if pd.notna(effect_row.get("ng2")) else math.nan,
                "eps": float(effect_row["eps"]) if pd.notna(effect_row.get("eps")) else math.nan,
                "anova_within_col": anova_within_col,
            }
            row.update(strata)
            rows.append(row)
    return rows


def sanitize_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "plot"


def save_interaction_plot(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    pair_col: str,
    anova_within_col: str,
    out_dir: Path,
) -> bool:
    work = df[[pair_col, group_col, anova_within_col, metric]].copy()
    work = work.dropna(subset=[pair_col, group_col, anova_within_col])
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        return False

    n_group = int(work[group_col].nunique())
    n_within = int(work[anova_within_col].nunique())
    if n_group < 2 or n_within < 2:
        return False

    expected_cells = n_group * n_within
    counts = (
        work.groupby(pair_col, dropna=False)
        .apply(lambda x: x[[group_col, anova_within_col]].drop_duplicates().shape[0], include_groups=False)
    )
    keep_ids = counts[counts == expected_cells].index
    work = work[work[pair_col].isin(keep_ids)].copy()
    if work[pair_col].nunique() < 2:
        return False

    work = work.groupby([pair_col, group_col, anova_within_col], as_index=False)[metric].mean()
    summary = (
        work.groupby([group_col, anova_within_col], as_index=False)[metric]
        .agg(["mean", sem_numeric, "count"])
        .reset_index()
        .rename(columns={"sem_numeric": "sem", "count": "n"})
    )
    if summary.empty:
        return False

    group_order = list(dict.fromkeys(work[group_col].astype(str).tolist()))
    within_order = list(dict.fromkeys(work[anova_within_col].astype(str).tolist()))
    x = np.arange(len(within_order), dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap("tab10")
    for idx, group_name in enumerate(group_order):
        grp = summary[summary[group_col].astype(str) == group_name].copy()
        grp[anova_within_col] = grp[anova_within_col].astype(str)
        grp = grp.set_index(anova_within_col).reindex(within_order).reset_index()
        ax.errorbar(
            x,
            grp["mean"].to_numpy(dtype=float),
            yerr=grp["sem"].to_numpy(dtype=float),
            label=str(group_name),
            color=cmap(idx % 10),
            marker="o",
            linewidth=2.0,
            markersize=6,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(within_order)
    ax.set_xlabel(anova_within_col)
    ax.set_ylabel(f"{metric} (Mean ± SEM)")
    ax.set_title(f"{metric}: {group_col} × {anova_within_col}")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_stem = sanitize_filename(f"{metric}_{group_col}_x_{anova_within_col}_interaction")
    fig.savefig(out_dir / f"{plot_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def recompute_one_file(
    path: Path,
    group_col: str,
    pair_col: str,
    output_name: str,
    anova_within_col: str | None,
    output_prefix: str | None = None,
    date_stamp: str | None = None,
) -> tuple[bool, str]:
    df_raw = pd.read_csv(path)
    if df_raw.empty:
        return False, "empty file"
    if group_col not in df_raw.columns:
        return False, f"missing group column: {group_col}"

    df = normalize_frame(df_raw, group_col=group_col, pair_col=pair_col)
    reserved = {group_col, pair_col}
    metric_cols = detect_metric_columns(df, reserved=reserved)
    if not metric_cols:
        return False, "no numeric metric columns found"

    strata_cols = [col for col in df.columns if col not in set(metric_cols) | {group_col, pair_col}]
    resolved_anova_within_col = choose_anova_within_col(strata_cols, anova_within_col)
    groups = list(dict.fromkeys(df[group_col].dropna().astype(str).tolist()))
    if len(groups) < 2:
        return False, "fewer than two groups found"

    rows: list[dict[str, object]] = []
    interaction_plot_count = 0
    for strata, slice_df in iter_slices(df, strata_cols):
        slice_groups = [g for g in groups if g in set(slice_df[group_col].astype(str))]
        if len(slice_groups) < 2:
            continue
        paired_df = paired_subset(slice_df, pair_col=pair_col, group_col=group_col, groups=slice_groups)
        rows.extend(
            build_summary_rows(
                df=slice_df,
                subset_name="all_subjects",
                metric_cols=metric_cols,
                strata=strata,
                group_col=group_col,
                groups=slice_groups,
            )
        )
        rows.extend(
            build_summary_rows(
                df=paired_df,
                subset_name="paired_only",
                metric_cols=metric_cols,
                strata=strata,
                group_col=group_col,
                groups=slice_groups,
            )
        )
        rows.extend(
            build_test_rows(
                all_df=slice_df,
                paired_df=paired_df,
                metric_cols=metric_cols,
                strata=strata,
                group_col=group_col,
                pair_col=pair_col,
                groups=slice_groups[:2],
            )
        )
        active_anova_within = None
        if resolved_anova_within_col:
            if not strata_cols:
                active_anova_within = None
            elif strata.get(resolved_anova_within_col) == COMBINED_LABEL:
                active_anova_within = resolved_anova_within_col
        rows.extend(
            build_two_way_rm_anova_rows(
                df=slice_df,
                metric_cols=metric_cols,
                strata=strata,
                group_col=group_col,
                pair_col=pair_col,
                anova_within_col=active_anova_within,
            )
        )
        if active_anova_within:
            strata_suffix = "_".join(
                f"{k}_{v}" for k, v in strata.items() if v != COMBINED_LABEL
            )
            plot_dir = path.parent / PLOT_DIRNAME
            if strata_suffix:
                plot_dir = plot_dir / sanitize_filename(strata_suffix)
            for metric in metric_cols:
                if save_interaction_plot(
                    df=slice_df,
                    metric=metric,
                    group_col=group_col,
                    pair_col=pair_col,
                    anova_within_col=active_anova_within,
                    out_dir=plot_dir,
                ):
                    interaction_plot_count += 1

    if not rows:
        return False, "no valid strata with at least two groups"

    out_path = path.parent / output_name
    alias = save_csv_with_alias(pd.DataFrame(rows), out_path, prefix=output_prefix, date_stamp=date_stamp)
    return True, f"{out_path} | alias={alias.name} | interaction_plots={interaction_plot_count}"


def recompute_tree(
    root: Path,
    filename: str = "per_subject.csv",
    group_col: str = DEFAULT_GROUP_COL,
    pair_col: str = DEFAULT_PAIR_COL,
    output_name: str = DEFAULT_OUTPUT_NAME,
    anova_within_col: str | None = None,
    output_prefix: str | None = None,
    date_stamp: str | None = None,
) -> list[tuple[Path, bool, str]]:
    root = root.resolve()
    targets = sorted(root.rglob(filename))
    results: list[tuple[Path, bool, str]] = []
    if not targets:
        return results
    effective_prefix = output_prefix or root.name
    effective_date = date_stamp or datetime.now().strftime("%Y%m%d")
    for path in targets:
        ok, msg = recompute_one_file(
            path=path,
            group_col=group_col,
            pair_col=pair_col,
            output_name=output_name,
            anova_within_col=anova_within_col,
            output_prefix=effective_prefix,
            date_stamp=effective_date,
        )
        results.append((path, ok, msg))
    return results


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    results = recompute_tree(
        root=root,
        filename=args.filename,
        group_col=args.group_col,
        pair_col=args.pair_col,
        output_name=args.output_name,
        anova_within_col=args.anova_within_col,
        output_prefix=args.prefix or root.name,
        date_stamp=args.date_stamp or datetime.now().strftime("%Y%m%d"),
    )
    if not results:
        print(f"No {args.filename} found under {root}")
        return

    ok_count = 0
    for path, ok, msg in results:
        status = "OK" if ok else "SKIP"
        print(f"[{status}] {path} -> {msg}")
        if ok:
            ok_count += 1
    print(f"Finished: {ok_count}/{len(results)} files written.")


if __name__ == "__main__":
    main()
