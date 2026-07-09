#!/usr/bin/env python
"""
Compare original and reconstructed parcel-FC matrices.

For every matched condition/subject/hemisphere output, this script correlates
the upper triangle of the parcel FC matrix between two phase-FC batch roots
(for example original vs reconstructed). It Fisher-z transforms each
subject-level r and runs paired condition tests on z values.

It also computes mean parcel delta matrices for each Drug:Placebo comparison:

    mean_delta = mean_subjects(Drug FC - Placebo FC)

Then it correlates original-vs-reconstructed mean delta matrices. No
inference is run for the mean-delta correlations.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


CONDITION_RE = re.compile(r"^(?P<condition>[A-Za-z]+_[A-Za-z]+)_S(?P<subid>\d+)")
META_COLUMNS = ["parcel_id", "parcel_name", "hemi", "network"]


@dataclass(frozen=True)
class FcEntry:
    condition: str
    subid: str
    hemisphere: str
    out_dir: Path
    fc_path: Path
    meta_path: Path


@dataclass(frozen=True)
class PconnEntry:
    condition: str
    subid: str
    subject_id: str
    hemisphere: str
    pconn_path: Path
    manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate original and reconstructed parcel-FC outputs."
    )
    parser.add_argument(
        "--orig-root",
        required=True,
        type=Path,
        help="Original phase-FC batch root, e.g. analysis_outputs/phase_fc_batch_phase_corr_7networks.",
    )
    parser.add_argument(
        "--recon-root",
        required=True,
        type=Path,
        help="Reconstructed phase-FC batch root, e.g. analysis_outputs/phase_fc_recon_7networks.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for correlation tables and mean-delta matrices.",
    )
    parser.add_argument(
        "--fc-file",
        default="parcel_plv.npy",
        help="FC matrix filename in each bundle directory, e.g. parcel_plv.npy or parcel_phase_corr.npy.",
    )
    parser.add_argument(
        "--true-fc-root",
        default=None,
        type=Path,
        help=(
            "Optional wb_command FC root containing per-condition manifests. "
            "When provided, also correlates original and reconstructed parcel-FC matrices with true pconn FC."
        ),
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=None,
        help="Drug:Placebo condition pair, e.g. DMT_DMT:DMT_PCB. Can be passed multiple times.",
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--min-pairs",
        default=3,
        type=int,
        help="Minimum paired subjects required for condition-level paired t-tests.",
    )
    parser.add_argument(
        "--min-edges",
        default=10,
        type=int,
        help="Minimum finite matrix edges required to compute a correlation.",
    )
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Do not fail when original/recon parcel metadata differ. Use only when you know matrices are comparable.",
    )
    return parser.parse_args()


def infer_hemisphere_from_meta(meta_path: Path) -> str:
    meta = pd.read_csv(meta_path, usecols=["hemi"])
    hemi = str(meta["hemi"].iloc[0])
    if hemi == "LH":
        return "left"
    if hemi == "RH":
        return "right"
    raise ValueError(f"Unexpected hemi value {hemi!r} in {meta_path}")


def parse_entry(out_dir: Path, fc_file: str) -> FcEntry | None:
    match = CONDITION_RE.search(out_dir.name)
    if not match:
        return None
    fc_path = out_dir / fc_file
    meta_path = out_dir / "parcel_metadata.csv"
    if not fc_path.exists() or not meta_path.exists():
        return None
    return FcEntry(
        condition=match.group("condition"),
        subid=match.group("subid"),
        hemisphere=infer_hemisphere_from_meta(meta_path),
        out_dir=out_dir,
        fc_path=fc_path,
        meta_path=meta_path,
    )


def parse_subject_id(subject_id: str) -> tuple[str, str]:
    match = CONDITION_RE.search(subject_id)
    if not match:
        raise ValueError(f"Cannot parse condition/subid from subject_id {subject_id!r}")
    return match.group("condition"), match.group("subid")


def discover_entries(root: Path, fc_file: str) -> pd.DataFrame:
    rows = []
    for out_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "atlas_metadata"):
        entry = parse_entry(out_dir, fc_file)
        if entry is not None:
            rows.append(entry.__dict__)
    if not rows:
        raise RuntimeError(f"No usable {fc_file} outputs found under {root}")
    return pd.DataFrame(rows)


def discover_pconn_entries(root: Path) -> pd.DataFrame:
    rows = []
    manifests = sorted(root.glob("*/manifests/fc_batch_hemisphere_manifest.csv"))
    if not manifests:
        raise RuntimeError(f"No wb_command hemisphere manifests found under {root}")

    for manifest_path in manifests:
        manifest = pd.read_csv(manifest_path)
        required = {"subject_id", "hemisphere", "pconn", "status"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        ok_rows = manifest[manifest["status"].astype(str).str.lower() == "ok"]
        for row in ok_rows.itertuples(index=False):
            pconn_path = Path(row.pconn)
            if not pconn_path.exists():
                continue
            condition, subid = parse_subject_id(str(row.subject_id))
            rows.append(
                PconnEntry(
                    condition=condition,
                    subid=subid,
                    subject_id=str(row.subject_id),
                    hemisphere=str(row.hemisphere).lower(),
                    pconn_path=pconn_path,
                    manifest_path=manifest_path,
                ).__dict__
            )

    if not rows:
        raise RuntimeError(f"No usable pconn outputs found under {root}")
    return pd.DataFrame(rows)


def key_index(entries: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    out: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in entries.iterrows():
        key = (str(row["condition"]), str(row["subid"]), str(row["hemisphere"]))
        if key in out:
            raise RuntimeError(f"Duplicate entry for key={key}")
        out[key] = row
    return out


def default_comparisons(conditions: set[str]) -> list[tuple[str, str]]:
    candidates = [("DMT_DMT", "DMT_PCB"), ("LSD_LSD", "LSD_PCB")]
    return [pair for pair in candidates if pair[0] in conditions and pair[1] in conditions]


def parse_comparisons(args: argparse.Namespace, conditions: set[str]) -> list[tuple[str, str]]:
    if args.comparison:
        pairs = []
        for spec in args.comparison:
            if ":" not in spec:
                raise ValueError(f"Comparison must be Drug:Placebo, got {spec!r}")
            drug, placebo = spec.split(":", 1)
            pairs.append((drug, placebo))
        return pairs
    pairs = default_comparisons(conditions)
    if not pairs:
        raise RuntimeError("No default comparisons found. Pass --comparison Drug:Placebo explicitly.")
    return pairs


def assert_metadata_match(orig_meta_path: Path, recon_meta_path: Path) -> None:
    orig = pd.read_csv(orig_meta_path, usecols=META_COLUMNS).reset_index(drop=True)
    recon = pd.read_csv(recon_meta_path, usecols=META_COLUMNS).reset_index(drop=True)
    if not orig.equals(recon):
        raise ValueError(f"Parcel metadata mismatch: {orig_meta_path} vs {recon_meta_path}")


def upper_triangle_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"Matrix shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square 2D matrices, got {a.shape}")
    idx = np.triu_indices_from(a, k=1)
    x = np.asarray(a[idx], dtype=float)
    y = np.asarray(b[idx], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def correlate_matrices(a: np.ndarray, b: np.ndarray, min_edges: int) -> dict[str, float]:
    x, y = upper_triangle_pair(a, b)
    n_edges = int(x.size)
    if n_edges < min_edges or np.std(x) == 0 or np.std(y) == 0:
        return {
            "n_edges": n_edges,
            "r": np.nan,
            "fisher_z": np.nan,
            "p": np.nan,
            "mean_abs_diff": np.nan,
            "rmse": np.nan,
        }
    r, p = stats.pearsonr(x, y)
    r = float(np.clip(r, -0.999999, 0.999999))
    diff = x - y
    return {
        "n_edges": n_edges,
        "r": r,
        "fisher_z": float(np.arctanh(r)),
        "p": float(p),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def mean_center_columns(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    return x - mean


def zscore_columns(x: np.ndarray) -> np.ndarray:
    std = np.std(x, axis=0, ddof=1, keepdims=True)
    valid = std > 0
    x = x[:, valid.ravel()]
    std = std[:, valid.ravel()]
    return x / std


def compute_ngsc_matrix(x: np.ndarray, *, zscore: bool = True) -> float:
    """Entropy of singular-value energy, normalized by log(k)."""
    if x.ndim != 2:
        return float("nan")
    valid_cols = np.all(np.isfinite(x), axis=0)
    x = x[:, valid_cols]
    if x.size == 0:
        return float("nan")
    x = mean_center_columns(x)
    if zscore:
        x = zscore_columns(x)
    if x.size == 0:
        return float("nan")
    n_time, n_nodes = x.shape
    if n_time < 2 or n_nodes < 1:
        return float("nan")
    try:
        _, svals, _ = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    energy = svals**2
    total = np.sum(energy)
    if total <= 0:
        return float("nan")
    p = energy / total
    p = p[p > 0]
    if p.size <= 1:
        return float("nan")
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(float(p.size)))


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(n))


def mean_ci95(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "se": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan}
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if n > 1 else np.nan
    if n > 1 and np.isfinite(se):
        crit = float(stats.t.ppf(0.975, n - 1))
        ci95_lo = mean - crit * se
        ci95_hi = mean + crit * se
    else:
        ci95_lo = np.nan
        ci95_hi = np.nan
    return {"n": n, "mean": mean, "sd": sd, "se": se, "ci95_lo": ci95_lo, "ci95_hi": ci95_hi}


def r2_interval(r_lo: float, r_hi: float) -> tuple[float, float]:
    if not np.isfinite(r_lo) or not np.isfinite(r_hi):
        return np.nan, np.nan
    vals = [r_lo**2, r_hi**2]
    lo = 0.0 if r_lo <= 0 <= r_hi else float(min(vals))
    hi = float(max(vals))
    return lo, hi


def fisher_r_summary(values: pd.Series | np.ndarray) -> dict[str, float]:
    r = np.asarray(values, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "n": 0,
            "mean_fisher_z": np.nan,
            "sd_fisher_z": np.nan,
            "se_fisher_z": np.nan,
            "ci95_z_lo": np.nan,
            "ci95_z_hi": np.nan,
            "mean_r": np.nan,
            "mean_r2": np.nan,
            "ci95_r_lo": np.nan,
            "ci95_r_hi": np.nan,
            "ci95_r2_lo": np.nan,
            "ci95_r2_hi": np.nan,
        }
    z_stats = mean_ci95(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    mean_r = float(np.tanh(z_stats["mean"]))
    ci95_r_lo = float(np.tanh(z_stats["ci95_lo"])) if np.isfinite(z_stats["ci95_lo"]) else np.nan
    ci95_r_hi = float(np.tanh(z_stats["ci95_hi"])) if np.isfinite(z_stats["ci95_hi"]) else np.nan
    ci95_r2_lo, ci95_r2_hi = r2_interval(ci95_r_lo, ci95_r_hi)
    return {
        "n": z_stats["n"],
        "mean_fisher_z": z_stats["mean"],
        "sd_fisher_z": z_stats["sd"],
        "se_fisher_z": z_stats["se"],
        "ci95_z_lo": z_stats["ci95_lo"],
        "ci95_z_hi": z_stats["ci95_hi"],
        "mean_r": mean_r,
        "mean_r2": float(mean_r**2),
        "ci95_r_lo": ci95_r_lo,
        "ci95_r_hi": ci95_r_hi,
        "ci95_r2_lo": ci95_r2_lo,
        "ci95_r2_hi": ci95_r2_hi,
    }


def summarize_correlations_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_cols, as_index=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(fisher_r_summary(group["r"]))
        row.update(
            {
                "mean_abs_diff": float(group["mean_abs_diff"].mean()),
                "rmse": float(group["rmse"].mean()),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols)


def paired_t(values_a: np.ndarray, values_b: np.ndarray, min_pairs: int) -> dict[str, float]:
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    a = np.asarray(values_a[finite], dtype=float)
    b = np.asarray(values_b[finite], dtype=float)
    n = int(a.size)
    if n < min_pairs:
        return {
            "n": n,
            "mean_z_a": np.nan,
            "mean_z_b": np.nan,
            "mean_z_diff": np.nan,
            "mean_r_a": np.nan,
            "mean_r_b": np.nan,
            "mean_r2_a": np.nan,
            "mean_r2_b": np.nan,
            "se_z_a": np.nan,
            "se_z_b": np.nan,
            "ci95_r_a_lo": np.nan,
            "ci95_r_a_hi": np.nan,
            "ci95_r2_a_lo": np.nan,
            "ci95_r2_a_hi": np.nan,
            "ci95_r_b_lo": np.nan,
            "ci95_r_b_hi": np.nan,
            "ci95_r2_b_lo": np.nan,
            "ci95_r2_b_hi": np.nan,
            "se_z_diff": np.nan,
            "ci95_z_diff_lo": np.nan,
            "ci95_z_diff_hi": np.nan,
            "t": np.nan,
            "p": np.nan,
            "cohen_dz": np.nan,
        }
    diff = a - b
    sd = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    if n > 1 and sd > 0:
        t_stat, p_val = stats.ttest_rel(a, b, nan_policy="omit")
        dz = float(np.mean(diff) / sd)
    else:
        t_stat, p_val, dz = np.nan, np.nan, np.nan
    mean_z_a = float(np.mean(a))
    mean_z_b = float(np.mean(b))
    a_stats = mean_ci95(a)
    b_stats = mean_ci95(b)
    diff_stats = mean_ci95(diff)
    mean_r_a = float(np.tanh(mean_z_a))
    mean_r_b = float(np.tanh(mean_z_b))
    ci95_r_a_lo = float(np.tanh(a_stats["ci95_lo"])) if np.isfinite(a_stats["ci95_lo"]) else np.nan
    ci95_r_a_hi = float(np.tanh(a_stats["ci95_hi"])) if np.isfinite(a_stats["ci95_hi"]) else np.nan
    ci95_r_b_lo = float(np.tanh(b_stats["ci95_lo"])) if np.isfinite(b_stats["ci95_lo"]) else np.nan
    ci95_r_b_hi = float(np.tanh(b_stats["ci95_hi"])) if np.isfinite(b_stats["ci95_hi"]) else np.nan
    ci95_r2_a_lo, ci95_r2_a_hi = r2_interval(ci95_r_a_lo, ci95_r_a_hi)
    ci95_r2_b_lo, ci95_r2_b_hi = r2_interval(ci95_r_b_lo, ci95_r_b_hi)
    return {
        "n": n,
        "mean_z_a": mean_z_a,
        "mean_z_b": mean_z_b,
        "mean_z_diff": float(np.mean(diff)),
        "mean_r_a": mean_r_a,
        "mean_r_b": mean_r_b,
        "mean_r2_a": float(mean_r_a**2),
        "mean_r2_b": float(mean_r_b**2),
        "se_z_a": a_stats["se"],
        "se_z_b": b_stats["se"],
        "ci95_r_a_lo": ci95_r_a_lo,
        "ci95_r_a_hi": ci95_r_a_hi,
        "ci95_r2_a_lo": ci95_r2_a_lo,
        "ci95_r2_a_hi": ci95_r2_a_hi,
        "ci95_r_b_lo": ci95_r_b_lo,
        "ci95_r_b_hi": ci95_r_b_hi,
        "ci95_r2_b_lo": ci95_r2_b_lo,
        "ci95_r2_b_hi": ci95_r2_b_hi,
        "se_z_diff": diff_stats["se"],
        "ci95_z_diff_lo": diff_stats["ci95_lo"],
        "ci95_z_diff_hi": diff_stats["ci95_hi"],
        "t": float(t_stat),
        "p": float(p_val),
        "cohen_dz": dz,
    }


def load_fc(row: pd.Series) -> np.ndarray:
    fc = np.load(Path(row["fc_path"]))
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"Expected square FC matrix, got {fc.shape}: {row['fc_path']}")
    return np.asarray(fc, dtype=np.float32)


def load_pconn(path: Path, expected_n: int) -> np.ndarray:
    data = np.asarray(nib.load(path).get_fdata(dtype=np.float32), dtype=np.float32)
    data = np.squeeze(data)
    if data.shape != (expected_n, expected_n):
        raise ValueError(f"Expected {expected_n}x{expected_n} pconn matrix, got {data.shape}: {path}")
    data = data.copy()
    np.fill_diagonal(data, np.nan)
    return data


def condition_correlations(
    orig_entries: pd.DataFrame,
    recon_entries: pd.DataFrame,
    hemispheres: list[str],
    min_edges: int,
    allow_metadata_mismatch: bool,
) -> pd.DataFrame:
    orig_by_key = key_index(orig_entries)
    recon_by_key = key_index(recon_entries)
    rows = []
    for key in sorted(set(orig_by_key) & set(recon_by_key)):
        condition, subid, hemisphere = key
        if hemisphere not in hemispheres:
            continue
        orig = orig_by_key[key]
        recon = recon_by_key[key]
        if not allow_metadata_mismatch:
            assert_metadata_match(Path(orig["meta_path"]), Path(recon["meta_path"]))
        stats_row = correlate_matrices(load_fc(orig), load_fc(recon), min_edges=min_edges)
        rows.append(
            {
                "condition": condition,
                "subid": subid,
                "hemisphere": hemisphere,
                **stats_row,
                "orig_fc": str(orig["fc_path"]),
                "recon_fc": str(recon["fc_path"]),
            }
        )
    if not rows:
        raise RuntimeError("No matched original/recon entries found.")
    return pd.DataFrame(rows)


def true_fc_correlations(
    orig_entries: pd.DataFrame,
    recon_entries: pd.DataFrame,
    true_entries: pd.DataFrame,
    hemispheres: list[str],
    min_edges: int,
    allow_metadata_mismatch: bool,
) -> pd.DataFrame:
    orig_by_key = key_index(orig_entries)
    recon_by_key = key_index(recon_entries)
    true_by_key = key_index(true_entries)
    rows = []
    for key in sorted((set(orig_by_key) | set(recon_by_key)) & set(true_by_key)):
        condition, subid, hemisphere = key
        if hemisphere not in hemispheres:
            continue
        true = true_by_key[key]
        phase_rows = []
        if key in orig_by_key:
            phase_rows.append(("orig", orig_by_key[key]))
        if key in recon_by_key:
            phase_rows.append(("recon", recon_by_key[key]))

        for source, phase in phase_rows:
            if source == "recon" and key in orig_by_key and not allow_metadata_mismatch:
                assert_metadata_match(Path(orig_by_key[key]["meta_path"]), Path(phase["meta_path"]))
            phase_fc = load_fc(phase)
            true_fc = load_pconn(Path(true["pconn_path"]), expected_n=phase_fc.shape[0])
            stats_row = correlate_matrices(phase_fc, true_fc, min_edges=min_edges)
            rows.append(
                {
                    "condition": condition,
                    "subid": subid,
                    "hemisphere": hemisphere,
                    "source": source,
                    **stats_row,
                    "phase_fc": str(phase["fc_path"]),
                    "true_pconn": str(true["pconn_path"]),
                }
            )
    if not rows:
        raise RuntimeError("No matched phase/true-FC entries found.")
    return pd.DataFrame(rows)


def filter_to_condition_paired_subjects(
    df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    hemispheres: list[str],
    *,
    source_col: str | None = None,
) -> pd.DataFrame:
    """Keep only subjects with both comparison conditions within each analysis stratum."""
    if df.empty:
        return df.copy()
    keep_idx: set[int] = set()
    source_values = [None]
    if source_col is not None and source_col in df.columns:
        source_values = list(df[source_col].dropna().unique())

    for source in source_values:
        source_df = df if source is None else df[df[source_col] == source]
        for drug, placebo in comparisons:
            for hemisphere in hemispheres:
                subset = source_df[
                    (source_df["hemisphere"] == hemisphere)
                    & (source_df["condition"].isin([drug, placebo]))
                ]
                if subset.empty:
                    continue
                wide = subset.pivot_table(
                    index="subid",
                    columns="condition",
                    values="hemisphere",
                    aggfunc="count",
                    fill_value=0,
                )
                if drug not in wide.columns or placebo not in wide.columns:
                    continue
                paired_subids = wide.index[(wide[drug] > 0) & (wide[placebo] > 0)]
                keep_idx.update(subset[subset["subid"].isin(paired_subids)].index.tolist())

    if not keep_idx:
        return df.iloc[0:0].copy()
    return df.loc[sorted(keep_idx)].copy()


def condition_paired_tests(
    corr_df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    hemispheres: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    for drug, placebo in comparisons:
        for hemisphere in hemispheres:
            subset = corr_df[
                (corr_df["hemisphere"] == hemisphere)
                & (corr_df["condition"].isin([drug, placebo]))
            ]
            wide = subset.pivot(index="subid", columns="condition", values="fisher_z")
            if drug not in wide.columns or placebo not in wide.columns:
                result = paired_t(np.asarray([], dtype=float), np.asarray([], dtype=float), min_pairs)
            else:
                result = paired_t(
                    wide[drug].to_numpy(dtype=float),
                    wide[placebo].to_numpy(dtype=float),
                    min_pairs=min_pairs,
                )
            rows.append(
                {
                    "comparison": f"{drug}-{placebo}",
                    "condition_a": drug,
                    "condition_b": placebo,
                    "hemisphere": hemisphere,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def true_condition_paired_tests(
    true_corr: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    hemispheres: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    for source, source_df in true_corr.groupby("source", sort=False):
        for drug, placebo in comparisons:
            for hemisphere in hemispheres:
                subset = source_df[
                    (source_df["hemisphere"] == hemisphere)
                    & (source_df["condition"].isin([drug, placebo]))
                ]
                wide = subset.pivot(index="subid", columns="condition", values="fisher_z")
                if drug not in wide.columns or placebo not in wide.columns:
                    result = paired_t(np.asarray([], dtype=float), np.asarray([], dtype=float), min_pairs)
                else:
                    result = paired_t(
                        wide[drug].to_numpy(dtype=float),
                        wide[placebo].to_numpy(dtype=float),
                        min_pairs=min_pairs,
                    )
                rows.append(
                    {
                        "source": source,
                        "comparison": f"{drug}-{placebo}",
                        "condition_a": drug,
                        "condition_b": placebo,
                        "hemisphere": hemisphere,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def true_source_paired_tests(
    true_corr: pd.DataFrame,
    hemispheres: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    for condition in sorted(true_corr["condition"].dropna().unique()):
        for hemisphere in hemispheres:
            subset = true_corr[
                (true_corr["condition"] == condition)
                & (true_corr["hemisphere"] == hemisphere)
                & (true_corr["source"].isin(["orig", "recon"]))
            ]
            wide = subset.pivot(index="subid", columns="source", values="fisher_z")
            if "orig" not in wide.columns or "recon" not in wide.columns:
                result = paired_t(np.asarray([], dtype=float), np.asarray([], dtype=float), min_pairs)
            else:
                result = paired_t(
                    wide["recon"].to_numpy(dtype=float),
                    wide["orig"].to_numpy(dtype=float),
                    min_pairs=min_pairs,
                )
            rows.append(
                {
                    "condition": condition,
                    "hemisphere": hemisphere,
                    "source_a": "recon",
                    "source_b": "orig",
                    **result,
                }
            )
    return pd.DataFrame(rows)


def paired_subjects(
    orig_by_key: dict[tuple[str, str, str], pd.Series],
    recon_by_key: dict[tuple[str, str, str], pd.Series],
    drug: str,
    placebo: str,
    hemisphere: str,
) -> list[str]:
    subjects = {
        subid
        for condition, subid, hemi in set(orig_by_key) & set(recon_by_key)
        if hemi == hemisphere and condition in {drug, placebo}
    }
    paired = []
    for subid in subjects:
        required = [
            (drug, subid, hemisphere),
            (placebo, subid, hemisphere),
        ]
        if all(key in orig_by_key and key in recon_by_key for key in required):
            paired.append(subid)
    return sorted(paired)


def mean_delta_correlations(
    orig_entries: pd.DataFrame,
    recon_entries: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    hemispheres: list[str],
    out_dir: Path,
    min_edges: int,
) -> pd.DataFrame:
    orig_by_key = key_index(orig_entries)
    recon_by_key = key_index(recon_entries)
    rows = []
    for drug, placebo in comparisons:
        for hemisphere in hemispheres:
            subjects = paired_subjects(orig_by_key, recon_by_key, drug, placebo, hemisphere)
            orig_deltas = []
            recon_deltas = []
            for subid in subjects:
                orig_drug = load_fc(orig_by_key[(drug, subid, hemisphere)])
                orig_placebo = load_fc(orig_by_key[(placebo, subid, hemisphere)])
                recon_drug = load_fc(recon_by_key[(drug, subid, hemisphere)])
                recon_placebo = load_fc(recon_by_key[(placebo, subid, hemisphere)])
                orig_deltas.append((orig_drug - orig_placebo).astype(np.float32, copy=False))
                recon_deltas.append((recon_drug - recon_placebo).astype(np.float32, copy=False))

            comp_name = f"{drug}_minus_{placebo}"
            hemi_out = out_dir / "mean_delta_matrices" / comp_name / hemisphere
            hemi_out.mkdir(parents=True, exist_ok=True)
            if not orig_deltas:
                rows.append(
                    {
                        "comparison": f"{drug}-{placebo}",
                        "condition_a": drug,
                        "condition_b": placebo,
                        "hemisphere": hemisphere,
                        "n_subjects": 0,
                        "n_edges": 0,
                        "r": np.nan,
                        "fisher_z": np.nan,
                        "p": np.nan,
                        "mean_abs_diff": np.nan,
                        "rmse": np.nan,
                        "orig_delta_mean": "",
                        "recon_delta_mean": "",
                    }
                )
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
                orig_delta_mean = np.nanmean(np.stack(orig_deltas, axis=0), axis=0).astype(np.float32, copy=False)
                recon_delta_mean = np.nanmean(np.stack(recon_deltas, axis=0), axis=0).astype(np.float32, copy=False)
            orig_path = hemi_out / "orig_parcel_delta_mean.npy"
            recon_path = hemi_out / "recon_parcel_delta_mean.npy"
            np.save(orig_path, orig_delta_mean)
            np.save(recon_path, recon_delta_mean)
            np.save(hemi_out / "recon_minus_orig_parcel_delta_mean.npy", recon_delta_mean - orig_delta_mean)
            stats_row = correlate_matrices(orig_delta_mean, recon_delta_mean, min_edges=min_edges)
            rows.append(
                {
                    "comparison": f"{drug}-{placebo}",
                    "condition_a": drug,
                    "condition_b": placebo,
                    "hemisphere": hemisphere,
                    "n_subjects": int(len(subjects)),
                    **stats_row,
                    "orig_delta_mean": str(orig_path),
                    "recon_delta_mean": str(recon_path),
                }
            )
    return pd.DataFrame(rows)


def resolve_phase_cube_path(row: pd.Series) -> Path | None:
    metadata_path = Path(row["out_dir"]) / "run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    phase_cube = metadata.get("phase_cube")
    if not phase_cube:
        return None
    path = Path(phase_cube)
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        path,
        Path.cwd() / path,
        metadata_path.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_phase_ngsc(row: pd.Series) -> tuple[float, str]:
    phase_path = resolve_phase_cube_path(row)
    if phase_path is None:
        return np.nan, ""
    try:
        phase_cube = np.load(phase_path, mmap_mode="r")
        if phase_cube.ndim != 3:
            return np.nan, str(phase_path)
        phase_x = np.asarray(phase_cube, dtype=float).reshape(-1, phase_cube.shape[2]).T
        return compute_ngsc_matrix(phase_x, zscore=True), str(phase_path)
    except Exception:
        return np.nan, str(phase_path)


def phase_ngsc_by_source(
    orig_entries: pd.DataFrame,
    recon_entries: pd.DataFrame,
    hemispheres: list[str],
) -> pd.DataFrame:
    rows = []
    for source, entries in [("orig", orig_entries), ("recon", recon_entries)]:
        for row in entries.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            if str(row_series["hemisphere"]) not in hemispheres:
                continue
            phase_ngsc, phase_cube = load_phase_ngsc(row_series)
            rows.append(
                {
                    "source": source,
                    "condition": str(row_series["condition"]),
                    "subid": str(row_series["subid"]),
                    "hemisphere": str(row_series["hemisphere"]),
                    "phase_ngsc": phase_ngsc,
                    "phase_cube": phase_cube,
                }
            )
    return pd.DataFrame(rows)


def scalar_correlation(x_values: np.ndarray, y_values: np.ndarray, min_pairs: int) -> dict[str, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = int(x.size)
    if n < min_pairs or np.std(x) == 0 or np.std(y) == 0:
        return {
            "n": n,
            "r": np.nan,
            "r2": np.nan,
            "fisher_z": np.nan,
            "se_fisher_z": np.nan,
            "ci95_z_lo": np.nan,
            "ci95_z_hi": np.nan,
            "ci95_r_lo": np.nan,
            "ci95_r_hi": np.nan,
            "ci95_r2_lo": np.nan,
            "ci95_r2_hi": np.nan,
            "p": np.nan,
        }
    r, p = stats.pearsonr(x, y)
    r = float(np.clip(r, -0.999999, 0.999999))
    fisher_z = float(np.arctanh(r))
    if n > 3:
        se_z = float(1.0 / np.sqrt(n - 3))
        z_lo = fisher_z - 1.96 * se_z
        z_hi = fisher_z + 1.96 * se_z
        r_lo = float(np.tanh(z_lo))
        r_hi = float(np.tanh(z_hi))
        r2_lo, r2_hi = r2_interval(r_lo, r_hi)
    else:
        se_z = np.nan
        z_lo = np.nan
        z_hi = np.nan
        r_lo = np.nan
        r_hi = np.nan
        r2_lo = np.nan
        r2_hi = np.nan
    return {
        "n": n,
        "r": r,
        "r2": float(r**2),
        "fisher_z": fisher_z,
        "se_fisher_z": se_z,
        "ci95_z_lo": z_lo,
        "ci95_z_hi": z_hi,
        "ci95_r_lo": r_lo,
        "ci95_r_hi": r_hi,
        "ci95_r2_lo": r2_lo,
        "ci95_r2_hi": r2_hi,
        "p": float(p),
    }


def phase_ngsc_condition_correlations(
    ngsc_df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    hemispheres: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    if ngsc_df.empty:
        return pd.DataFrame(rows)
    for source, source_df in ngsc_df.groupby("source", sort=False):
        for drug, placebo in comparisons:
            for hemisphere in hemispheres:
                subset = source_df[
                    (source_df["hemisphere"] == hemisphere)
                    & (source_df["condition"].isin([drug, placebo]))
                ]
                wide = subset.pivot(index="subid", columns="condition", values="phase_ngsc")
                if drug in wide.columns and placebo in wide.columns:
                    result = scalar_correlation(wide[drug].to_numpy(), wide[placebo].to_numpy(), min_pairs)
                else:
                    result = scalar_correlation(np.asarray([], dtype=float), np.asarray([], dtype=float), min_pairs)
                rows.append(
                    {
                        "source": source,
                        "comparison": f"{drug}-{placebo}",
                        "condition_a": drug,
                        "condition_b": placebo,
                        "hemisphere": hemisphere,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def phase_ngsc_source_correlations(
    ngsc_df: pd.DataFrame,
    hemispheres: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    if ngsc_df.empty:
        return pd.DataFrame(rows)
    for condition in sorted(ngsc_df["condition"].dropna().unique()):
        for hemisphere in hemispheres:
            subset = ngsc_df[
                (ngsc_df["condition"] == condition)
                & (ngsc_df["hemisphere"] == hemisphere)
                & (ngsc_df["source"].isin(["orig", "recon"]))
            ]
            wide = subset.pivot(index="subid", columns="source", values="phase_ngsc")
            if "orig" in wide.columns and "recon" in wide.columns:
                result = scalar_correlation(wide["recon"].to_numpy(), wide["orig"].to_numpy(), min_pairs)
            else:
                result = scalar_correlation(np.asarray([], dtype=float), np.asarray([], dtype=float), min_pairs)
            rows.append(
                {
                    "condition": condition,
                    "hemisphere": hemisphere,
                    "source_a": "recon",
                    "source_b": "orig",
                    **result,
                }
            )
    return pd.DataFrame(rows)


def phase_ngsc_delta_correlations(
    ngsc_df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    hemispheres: list[str],
    min_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_rows = []
    corr_rows = []
    if ngsc_df.empty:
        return pd.DataFrame(delta_rows), pd.DataFrame(corr_rows)

    for drug, placebo in comparisons:
        for hemisphere in hemispheres:
            source_deltas = {}
            for source, source_df in ngsc_df.groupby("source", sort=False):
                subset = source_df[
                    (source_df["hemisphere"] == hemisphere)
                    & (source_df["condition"].isin([drug, placebo]))
                ]
                wide = subset.pivot(index="subid", columns="condition", values="phase_ngsc")
                if drug not in wide.columns or placebo not in wide.columns:
                    source_deltas[source] = pd.Series(dtype=float)
                    continue
                delta = wide[drug] - wide[placebo]
                source_deltas[source] = delta
                for subid, value in delta.dropna().items():
                    delta_rows.append(
                        {
                            "source": source,
                            "comparison": f"{drug}-{placebo}",
                            "condition_a": drug,
                            "condition_b": placebo,
                            "hemisphere": hemisphere,
                            "subid": str(subid),
                            "phase_ngsc_delta": float(value),
                        }
                    )

            orig_delta = source_deltas.get("orig", pd.Series(dtype=float))
            recon_delta = source_deltas.get("recon", pd.Series(dtype=float))
            common = orig_delta.index.intersection(recon_delta.index)
            result = scalar_correlation(
                recon_delta.loc[common].to_numpy(dtype=float),
                orig_delta.loc[common].to_numpy(dtype=float),
                min_pairs=min_pairs,
            )
            corr_rows.append(
                {
                    "comparison": f"{drug}-{placebo}",
                    "condition_a": drug,
                    "condition_b": placebo,
                    "hemisphere": hemisphere,
                    "source_a": "recon_delta",
                    "source_b": "orig_delta",
                    **result,
                }
            )

    return pd.DataFrame(delta_rows), pd.DataFrame(corr_rows)


def p_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=nan"
    if p_value < 1e-3:
        return f"p={p_value:.1e}"
    return f"p={p_value:.3f}"


def plot_subject_correlations(
    corr_df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "figures" / "subject_correlations"
    fig_dir.mkdir(parents=True, exist_ok=True)

    corr_df = corr_df.copy()
    corr_df["subid"] = corr_df["subid"].astype(str)
    for drug, placebo in comparisons:
        comp_df = corr_df[corr_df["condition"].isin([drug, placebo])].copy()
        if comp_df.empty:
            continue
        for hemisphere, hemi_df in comp_df.groupby("hemisphere", sort=False):
            hemi_df["condition"] = pd.Categorical(
                hemi_df["condition"],
                categories=[placebo, drug],
                ordered=True,
            )
            hemi_df = hemi_df.sort_values(["condition", "subid"])
            plt.figure(figsize=(7, 5))
            ax = sns.boxplot(
                data=hemi_df,
                x="condition",
                y="r",
                hue="condition",
                order=[placebo, drug],
                palette="Set2",
                width=0.45,
                fliersize=0,
                legend=False,
            )
            sns.stripplot(
                data=hemi_df,
                x="condition",
                y="r",
                order=[placebo, drug],
                color="black",
                alpha=0.75,
                jitter=0.08,
                size=4,
                ax=ax,
            )
            wide = hemi_df.pivot(index="subid", columns="condition", values="r")
            if drug in wide.columns and placebo in wide.columns:
                for _, row in wide.dropna(subset=[placebo, drug]).iterrows():
                    ax.plot([0, 1], [row[placebo], row[drug]], color="0.55", alpha=0.45, linewidth=1)
            ax.set_title(f"Original vs recon parcel-FC correlation ({drug} vs {placebo}, {hemisphere})")
            ax.set_xlabel("")
            ax.set_ylabel("Pearson r across parcel-FC edges")
            ax.set_ylim(max(0.0, float(np.nanmin(hemi_df["r"])) - 0.05), min(1.0, float(np.nanmax(hemi_df["r"])) + 0.05))
            plt.tight_layout()
            plt.savefig(fig_dir / f"{drug}_vs_{placebo}_{hemisphere}_subject_r.png", dpi=240)
            plt.close()


def plot_condition_ttests(tests: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if tests.empty:
        return

    plot_rows = []
    for row in tests.itertuples(index=False):
        plot_rows.append(
            {
                "comparison": row.comparison,
                "hemisphere": row.hemisphere,
                "condition": row.condition_a,
                "mean_r": row.mean_r_a,
                "p": row.p,
                "n": row.n,
            }
        )
        plot_rows.append(
            {
                "comparison": row.comparison,
                "hemisphere": row.hemisphere,
                "condition": row.condition_b,
                "mean_r": row.mean_r_b,
                "p": row.p,
                "n": row.n,
            }
        )
    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        return

    g = sns.catplot(
        data=plot_df,
        x="hemisphere",
        y="mean_r",
        hue="condition",
        col="comparison",
        kind="bar",
        palette="Set2",
        height=4.5,
        aspect=1.0,
        sharey=True,
    )
    g.set_axis_labels("", "Mean r from Fisher-z average")
    g.set_titles("{col_name}")
    for ax, comparison in zip(g.axes.flat, plot_df["comparison"].drop_duplicates()):
        sub = tests[tests["comparison"] == comparison]
        comp_values = plot_df[plot_df["comparison"] == comparison]["mean_r"].to_numpy(dtype=float)
        ymax = float(np.nanmax(comp_values)) if np.isfinite(comp_values).any() else 1.0
        y = min(1.0, ymax + 0.04)
        for idx, row in enumerate(sub.itertuples(index=False)):
            ax.text(idx, y, f"n={row.n}\n{p_label(row.p)}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, min(1.05, y + 0.08))
    plt.tight_layout()
    g.savefig(fig_dir / "condition_mean_correlation_paired_ttests.png", dpi=240)
    plt.close(g.figure)


def paired_ylim(values: pd.Series, pad: float = 0.05) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return 0.0, 1.0
    lo = max(-1.0, float(finite.min()) - pad)
    hi = min(1.0, float(finite.max()) + pad)
    if lo == hi:
        lo = max(-1.0, lo - pad)
        hi = min(1.0, hi + pad)
    return lo, hi


def plot_true_subject_correlations(
    true_corr: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "figures" / "true_fc_subject_correlations"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if true_corr.empty:
        return

    true_corr = true_corr.copy()
    true_corr["subid"] = true_corr["subid"].astype(str)
    source_order = [source for source in ["orig", "recon"] if source in set(true_corr["source"])]
    for drug, placebo in comparisons:
        comp_df = true_corr[true_corr["condition"].isin([drug, placebo])].copy()
        if comp_df.empty:
            continue
        for hemisphere, hemi_df in comp_df.groupby("hemisphere", sort=False):
            hemi_df["condition"] = pd.Categorical(
                hemi_df["condition"],
                categories=[placebo, drug],
                ordered=True,
            )
            hemi_df["source"] = pd.Categorical(hemi_df["source"], categories=source_order, ordered=True)
            g = sns.catplot(
                data=hemi_df,
                x="condition",
                y="r",
                hue="condition",
                col="source",
                col_order=source_order,
                order=[placebo, drug],
                kind="box",
                palette="Set2",
                height=4.5,
                aspect=0.9,
                fliersize=0,
                legend=False,
                sharey=True,
            )
            for ax, source in zip(g.axes.flat, source_order):
                sub = hemi_df[hemi_df["source"].astype(str) == source]
                sns.stripplot(
                    data=sub,
                    x="condition",
                    y="r",
                    order=[placebo, drug],
                    color="black",
                    alpha=0.75,
                    jitter=0.08,
                    size=4,
                    ax=ax,
                )
                wide = sub.pivot(index="subid", columns="condition", values="r")
                if drug in wide.columns and placebo in wide.columns:
                    for _, row in wide.dropna(subset=[placebo, drug]).iterrows():
                        ax.plot([0, 1], [row[placebo], row[drug]], color="0.55", alpha=0.45, linewidth=1)
                ax.set_title(f"{source} vs true")
                ax.set_xlabel("")
            g.set_axis_labels("", "Pearson r across parcel-FC edges")
            ymin, ymax = paired_ylim(hemi_df["r"])
            for ax in g.axes.flat:
                ax.set_ylim(ymin, ymax)
            g.figure.suptitle(f"Phase-FC vs true FC correlation ({drug} vs {placebo}, {hemisphere})", y=1.03)
            g.savefig(fig_dir / f"{drug}_vs_{placebo}_{hemisphere}_true_fc_subject_r.png", dpi=240, bbox_inches="tight")
            plt.close(g.figure)


def plot_true_condition_ttests(tests: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if tests.empty:
        return

    plot_rows = []
    for row in tests.itertuples(index=False):
        plot_rows.append(
            {
                "source": row.source,
                "comparison": row.comparison,
                "hemisphere": row.hemisphere,
                "condition": row.condition_a,
                "mean_r": row.mean_r_a,
                "p": row.p,
                "n": row.n,
            }
        )
        plot_rows.append(
            {
                "source": row.source,
                "comparison": row.comparison,
                "hemisphere": row.hemisphere,
                "condition": row.condition_b,
                "mean_r": row.mean_r_b,
                "p": row.p,
                "n": row.n,
            }
        )
    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        return

    for comparison, comp_df in plot_df.groupby("comparison", sort=False):
        g = sns.catplot(
            data=comp_df,
            x="hemisphere",
            y="mean_r",
            hue="condition",
            col="source",
            kind="bar",
            palette="Set2",
            height=4.5,
            aspect=0.95,
            sharey=True,
        )
        g.set_axis_labels("", "Mean r from Fisher-z average")
        g.set_titles("{col_name} vs true")
        ymax = float(np.nanmax(comp_df["mean_r"])) if np.isfinite(comp_df["mean_r"]).any() else 1.0
        y = min(1.0, ymax + 0.04)
        for ax, source in zip(g.axes.flat, comp_df["source"].drop_duplicates()):
            sub = tests[(tests["comparison"] == comparison) & (tests["source"] == source)]
            for idx, row in enumerate(sub.itertuples(index=False)):
                ax.text(idx, y, f"n={row.n}\n{p_label(row.p)}", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(0, min(1.05, y + 0.08))
        g.figure.suptitle(f"Phase-FC vs true FC condition test ({comparison})", y=1.04)
        out_name = f"{comparison.replace('-', '_vs_')}_true_fc_condition_paired_ttests.png"
        g.savefig(fig_dir / out_name, dpi=240, bbox_inches="tight")
        plt.close(g.figure)


def plot_true_source_ttests(tests: pd.DataFrame, true_corr: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if tests.empty or true_corr.empty:
        return

    plot_df = (
        true_corr.groupby(["condition", "hemisphere", "source"], as_index=False)
        .agg(mean_fisher_z=("fisher_z", "mean"))
        .assign(mean_r=lambda df: np.tanh(df["mean_fisher_z"]))
    )
    plot_df = plot_df[plot_df["source"].isin(["orig", "recon"])]
    if plot_df.empty:
        return

    g = sns.catplot(
        data=plot_df,
        x="condition",
        y="mean_r",
        hue="source",
        col="hemisphere",
        kind="bar",
        palette="Set2",
        height=4.8,
        aspect=1.1,
        sharey=True,
    )
    g.set_axis_labels("", "Mean r with true FC from Fisher-z average")
    g.set_titles("{col_name}")
    for ax, hemisphere in zip(g.axes.flat, plot_df["hemisphere"].drop_duplicates()):
        sub = tests[tests["hemisphere"] == hemisphere]
        ymax = float(np.nanmax(plot_df[plot_df["hemisphere"] == hemisphere]["mean_r"]))
        y = min(1.0, ymax + 0.04)
        for idx, row in enumerate(sub.itertuples(index=False)):
            ax.text(idx, y, f"n={row.n}\n{p_label(row.p)}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, min(1.05, y + 0.08))
    g.figure.suptitle("Recon vs original correlation with true FC", y=1.04)
    g.savefig(fig_dir / "true_fc_orig_recon_paired_ttests.png", dpi=240, bbox_inches="tight")
    plt.close(g.figure)


def plot_delta_correlations(delta_corr: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if delta_corr.empty:
        return

    plt.figure(figsize=(7, 4.8))
    ax = sns.barplot(
        data=delta_corr,
        x="hemisphere",
        y="r",
        hue="comparison",
        palette="Set2",
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)
    ax.set_title("Original vs recon mean parcel-delta FC correlation")
    ax.set_xlabel("")
    ax.set_ylabel("Pearson r across parcel-delta edges")
    finite_r = delta_corr["r"].to_numpy(dtype=float)
    ymax = float(np.nanmax(finite_r)) if np.isfinite(finite_r).any() else 1.0
    ax.set_ylim(0, min(1.05, max(1.0, ymax + 0.08)))
    plt.tight_layout()
    plt.savefig(fig_dir / "mean_delta_correlation_bar.png", dpi=240)
    plt.close()


def plot_delta_matrix_triptychs(delta_corr: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "mean_delta_matrices"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for row in delta_corr.itertuples(index=False):
        if not row.orig_delta_mean or not row.recon_delta_mean:
            continue
        orig_path = Path(row.orig_delta_mean)
        recon_path = Path(row.recon_delta_mean)
        if not orig_path.exists() or not recon_path.exists():
            continue
        orig = np.load(orig_path)
        recon = np.load(recon_path)
        diff = recon - orig
        vmax_delta = np.nanmax(np.abs(np.stack([orig, recon], axis=0)))
        vmax_delta = max(float(vmax_delta), 1e-6)
        vmax_diff = np.nanmax(np.abs(diff))
        vmax_diff = max(float(vmax_diff), 1e-6)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        panels = [
            ("Original mean delta", orig, vmax_delta),
            ("Recon mean delta", recon, vmax_delta),
            ("Recon - original", diff, vmax_diff),
        ]
        for ax, (title, mat, vmax) in zip(axes, panels):
            sns.heatmap(
                mat,
                ax=ax,
                cmap="coolwarm",
                center=0,
                vmin=-vmax,
                vmax=vmax,
                square=True,
                xticklabels=False,
                yticklabels=False,
                cbar_kws={"shrink": 0.75},
            )
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.suptitle(f"{row.comparison} {row.hemisphere}: r={row.r:.3f}, n={row.n_subjects}")
        out_name = f"{row.condition_a}_minus_{row.condition_b}_{row.hemisphere}_mean_delta_matrices.png"
        fig.savefig(fig_dir / out_name, dpi=240)
        plt.close(fig)


def write_summary_report(
    out_dir: Path,
    corr_df: pd.DataFrame,
    tests: pd.DataFrame,
    delta_corr: pd.DataFrame,
    true_corr: pd.DataFrame | None = None,
    true_condition_tests: pd.DataFrame | None = None,
    true_source_tests: pd.DataFrame | None = None,
    phase_ngsc: pd.DataFrame | None = None,
    phase_ngsc_condition_corr: pd.DataFrame | None = None,
    phase_ngsc_source_corr: pd.DataFrame | None = None,
    phase_ngsc_delta: pd.DataFrame | None = None,
    phase_ngsc_delta_corr: pd.DataFrame | None = None,
) -> None:
    summary_by_condition = summarize_correlations_by_group(corr_df, ["condition", "hemisphere"])
    summary_by_condition.to_csv(out_dir / "summary_by_condition_hemisphere.csv", index=False)
    if true_corr is not None and not true_corr.empty:
        true_summary = summarize_correlations_by_group(true_corr, ["source", "condition", "hemisphere"])
        true_summary.to_csv(out_dir / "summary_true_fc_by_source_condition_hemisphere.csv", index=False)
    else:
        true_summary = pd.DataFrame()
    if phase_ngsc is not None and not phase_ngsc.empty:
        ngsc_rows = []
        for key, group in phase_ngsc.groupby(["source", "condition", "hemisphere"], sort=True):
            stats_row = mean_ci95(group["phase_ngsc"].to_numpy(dtype=float))
            ngsc_rows.append(
                {
                    "source": key[0],
                    "condition": key[1],
                    "hemisphere": key[2],
                    "n": stats_row["n"],
                    "mean_phase_ngsc": stats_row["mean"],
                    "sd_phase_ngsc": stats_row["sd"],
                    "se_phase_ngsc": stats_row["se"],
                    "ci95_phase_ngsc_lo": stats_row["ci95_lo"],
                    "ci95_phase_ngsc_hi": stats_row["ci95_hi"],
                }
            )
        ngsc_summary = pd.DataFrame(ngsc_rows).sort_values(["source", "condition", "hemisphere"])
        ngsc_summary.to_csv(out_dir / "summary_phase_ngsc_by_source_condition_hemisphere.csv", index=False)
    else:
        ngsc_summary = pd.DataFrame()
    if phase_ngsc_delta is not None and not phase_ngsc_delta.empty:
        delta_summary_rows = []
        for key, group in phase_ngsc_delta.groupby(["source", "comparison", "hemisphere"], sort=True):
            stats_row = mean_ci95(group["phase_ngsc_delta"].to_numpy(dtype=float))
            delta_summary_rows.append(
                {
                    "source": key[0],
                    "comparison": key[1],
                    "hemisphere": key[2],
                    "n": stats_row["n"],
                    "mean_phase_ngsc_delta": stats_row["mean"],
                    "sd_phase_ngsc_delta": stats_row["sd"],
                    "se_phase_ngsc_delta": stats_row["se"],
                    "ci95_phase_ngsc_delta_lo": stats_row["ci95_lo"],
                    "ci95_phase_ngsc_delta_hi": stats_row["ci95_hi"],
                }
            )
        ngsc_delta_summary = pd.DataFrame(delta_summary_rows).sort_values(["source", "comparison", "hemisphere"])
        ngsc_delta_summary.to_csv(out_dir / "summary_phase_ngsc_drug_minus_pcb_delta.csv", index=False)
    else:
        ngsc_delta_summary = pd.DataFrame()

    lines = [
        "# Recon Parcel-FC Correlation Summary",
        "",
        "## Per-Condition Summary",
        "",
        summary_by_condition.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired Condition Tests on Fisher z",
        "",
        tests.to_markdown(index=False, floatfmt=".4f") if not tests.empty else "No paired tests.",
        "",
        "## Mean Delta Correlations",
        "",
        delta_corr.to_markdown(index=False, floatfmt=".4f") if not delta_corr.empty else "No mean delta correlations.",
        "",
        "## True FC Correlations",
        "",
        true_summary.to_markdown(index=False, floatfmt=".4f") if not true_summary.empty else "No true FC root provided.",
        "",
        "## True FC Drug-Placebo Tests on Fisher z",
        "",
        true_condition_tests.to_markdown(index=False, floatfmt=".4f")
        if true_condition_tests is not None and not true_condition_tests.empty
        else "No true FC Drug-Placebo tests.",
        "",
        "## True FC Recon-Original Tests on Fisher z",
        "",
        true_source_tests.to_markdown(index=False, floatfmt=".4f")
        if true_source_tests is not None and not true_source_tests.empty
        else "No true FC recon-original tests.",
        "",
        "## Phase NGSC Summary",
        "",
        ngsc_summary.to_markdown(index=False, floatfmt=".4f") if not ngsc_summary.empty else "No phase NGSC records.",
        "",
        "## Phase NGSC Drug-Placebo Correlations",
        "",
        phase_ngsc_condition_corr.to_markdown(index=False, floatfmt=".4f")
        if phase_ngsc_condition_corr is not None and not phase_ngsc_condition_corr.empty
        else "No phase NGSC condition correlations.",
        "",
        "## Phase NGSC Recon-Original Correlations",
        "",
        phase_ngsc_source_corr.to_markdown(index=False, floatfmt=".4f")
        if phase_ngsc_source_corr is not None and not phase_ngsc_source_corr.empty
        else "No phase NGSC source correlations.",
        "",
        "## Phase NGSC Drug-Placebo Delta Correlations",
        "",
        "Delta is computed per subject as Drug - PCB separately in orig and recon, then recon_delta is correlated with orig_delta across subjects.",
        "",
        "### Delta Summary",
        "",
        ngsc_delta_summary.to_markdown(index=False, floatfmt=".4f")
        if not ngsc_delta_summary.empty
        else "No phase NGSC Drug-Placebo delta summary.",
        "",
        "### Delta Correlation",
        "",
        phase_ngsc_delta_corr.to_markdown(index=False, floatfmt=".4f")
        if phase_ngsc_delta_corr is not None and not phase_ngsc_delta_corr.empty
        else "No phase NGSC Drug-Placebo delta correlations.",
        "",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots_and_report(
    out_dir: Path,
    corr_df: pd.DataFrame,
    tests: pd.DataFrame,
    delta_corr: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    true_corr: pd.DataFrame | None = None,
    true_condition_tests: pd.DataFrame | None = None,
    true_source_tests: pd.DataFrame | None = None,
    phase_ngsc: pd.DataFrame | None = None,
    phase_ngsc_condition_corr: pd.DataFrame | None = None,
    phase_ngsc_source_corr: pd.DataFrame | None = None,
    phase_ngsc_delta: pd.DataFrame | None = None,
    phase_ngsc_delta_corr: pd.DataFrame | None = None,
) -> None:
    sns.set_theme(style="whitegrid")
    plot_subject_correlations(corr_df, comparisons, out_dir)
    plot_condition_ttests(tests, out_dir)
    plot_delta_correlations(delta_corr, out_dir)
    plot_delta_matrix_triptychs(delta_corr, out_dir)
    if true_corr is not None:
        plot_true_subject_correlations(true_corr, comparisons, out_dir)
    if true_condition_tests is not None:
        plot_true_condition_ttests(true_condition_tests, out_dir)
    if true_source_tests is not None and true_corr is not None:
        plot_true_source_ttests(true_source_tests, true_corr, out_dir)
    write_summary_report(
        out_dir,
        corr_df,
        tests,
        delta_corr,
        true_corr=true_corr,
        true_condition_tests=true_condition_tests,
        true_source_tests=true_source_tests,
        phase_ngsc=phase_ngsc,
        phase_ngsc_condition_corr=phase_ngsc_condition_corr,
        phase_ngsc_source_corr=phase_ngsc_source_corr,
        phase_ngsc_delta=phase_ngsc_delta,
        phase_ngsc_delta_corr=phase_ngsc_delta_corr,
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hemispheres = ["left", "right"] if args.hemisphere == "both" else [args.hemisphere]

    orig_entries = discover_entries(args.orig_root, args.fc_file)
    recon_entries = discover_entries(args.recon_root, args.fc_file)
    true_entries = discover_pconn_entries(args.true_fc_root) if args.true_fc_root is not None else None
    conditions = set(orig_entries["condition"]) & set(recon_entries["condition"])
    if true_entries is not None:
        conditions &= set(true_entries["condition"])
    comparisons = parse_comparisons(args, conditions)

    corr_df = condition_correlations(
        orig_entries=orig_entries,
        recon_entries=recon_entries,
        hemispheres=hemispheres,
        min_edges=args.min_edges,
        allow_metadata_mismatch=bool(args.allow_metadata_mismatch),
    )
    corr_df = filter_to_condition_paired_subjects(corr_df, comparisons, hemispheres)
    corr_df.to_csv(args.out_dir / "per_condition_subject_parcel_fc_correlations.csv", index=False)

    tests = condition_paired_tests(
        corr_df=corr_df,
        comparisons=comparisons,
        hemispheres=hemispheres,
        min_pairs=args.min_pairs,
    )
    tests.to_csv(args.out_dir / "condition_paired_ttests_fisher_z.csv", index=False)

    delta_corr = mean_delta_correlations(
        orig_entries=orig_entries,
        recon_entries=recon_entries,
        comparisons=comparisons,
        hemispheres=hemispheres,
        out_dir=args.out_dir,
        min_edges=args.min_edges,
    )
    delta_corr.to_csv(args.out_dir / "mean_delta_parcel_fc_correlations.csv", index=False)

    phase_ngsc = phase_ngsc_by_source(
        orig_entries=orig_entries,
        recon_entries=recon_entries,
        hemispheres=hemispheres,
    )
    phase_ngsc = filter_to_condition_paired_subjects(
        phase_ngsc,
        comparisons,
        hemispheres,
        source_col="source",
    )
    phase_ngsc.to_csv(args.out_dir / "per_condition_subject_phase_ngsc.csv", index=False)
    phase_ngsc_condition_corr = phase_ngsc_condition_correlations(
        ngsc_df=phase_ngsc,
        comparisons=comparisons,
        hemispheres=hemispheres,
        min_pairs=args.min_pairs,
    )
    phase_ngsc_condition_corr.to_csv(args.out_dir / "phase_ngsc_condition_correlations.csv", index=False)
    phase_ngsc_source_corr = phase_ngsc_source_correlations(
        ngsc_df=phase_ngsc,
        hemispheres=hemispheres,
        min_pairs=args.min_pairs,
    )
    phase_ngsc_source_corr.to_csv(args.out_dir / "phase_ngsc_orig_recon_correlations.csv", index=False)
    phase_ngsc_delta, phase_ngsc_delta_corr = phase_ngsc_delta_correlations(
        ngsc_df=phase_ngsc,
        comparisons=comparisons,
        hemispheres=hemispheres,
        min_pairs=args.min_pairs,
    )
    phase_ngsc_delta.to_csv(args.out_dir / "per_subject_phase_ngsc_drug_minus_pcb_delta.csv", index=False)
    phase_ngsc_delta_corr.to_csv(args.out_dir / "phase_ngsc_drug_minus_pcb_delta_orig_recon_correlations.csv", index=False)

    if true_entries is not None:
        true_corr = true_fc_correlations(
            orig_entries=orig_entries,
            recon_entries=recon_entries,
            true_entries=true_entries,
            hemispheres=hemispheres,
            min_edges=args.min_edges,
            allow_metadata_mismatch=bool(args.allow_metadata_mismatch),
        )
        true_corr = filter_to_condition_paired_subjects(
            true_corr,
            comparisons,
            hemispheres,
            source_col="source",
        )
        true_corr.to_csv(args.out_dir / "per_condition_subject_true_fc_correlations.csv", index=False)
        true_condition_tests = true_condition_paired_tests(
            true_corr=true_corr,
            comparisons=comparisons,
            hemispheres=hemispheres,
            min_pairs=args.min_pairs,
        )
        true_condition_tests.to_csv(args.out_dir / "true_fc_condition_paired_ttests_fisher_z.csv", index=False)
        true_source_tests = true_source_paired_tests(
            true_corr=true_corr,
            hemispheres=hemispheres,
            min_pairs=args.min_pairs,
        )
        true_source_tests.to_csv(args.out_dir / "true_fc_orig_recon_paired_ttests_fisher_z.csv", index=False)
    else:
        true_corr = None
        true_condition_tests = None
        true_source_tests = None
    make_plots_and_report(
        args.out_dir,
        corr_df,
        tests,
        delta_corr,
        comparisons,
        true_corr=true_corr,
        true_condition_tests=true_condition_tests,
        true_source_tests=true_source_tests,
        phase_ngsc=phase_ngsc,
        phase_ngsc_condition_corr=phase_ngsc_condition_corr,
        phase_ngsc_source_corr=phase_ngsc_source_corr,
        phase_ngsc_delta=phase_ngsc_delta,
        phase_ngsc_delta_corr=phase_ngsc_delta_corr,
    )

    summary = {
        "orig_root": str(args.orig_root),
        "recon_root": str(args.recon_root),
        "true_fc_root": str(args.true_fc_root) if args.true_fc_root is not None else None,
        "fc_file": args.fc_file,
        "comparisons": [{"condition_a": a, "condition_b": b} for a, b in comparisons],
        "hemispheres": hemispheres,
        "n_orig_entries": int(len(orig_entries)),
        "n_recon_entries": int(len(recon_entries)),
        "n_true_entries": int(len(true_entries)) if true_entries is not None else 0,
        "n_matched_correlations": int(len(corr_df)),
        "n_true_fc_correlations": int(len(true_corr)) if true_corr is not None else 0,
        "n_true_fc_condition_tests": int(len(true_condition_tests)) if true_condition_tests is not None else 0,
        "n_true_fc_orig_recon_tests": int(len(true_source_tests)) if true_source_tests is not None else 0,
        "n_phase_ngsc_records": int(len(phase_ngsc)),
        "n_phase_ngsc_condition_correlations": int(len(phase_ngsc_condition_corr)),
        "n_phase_ngsc_orig_recon_correlations": int(len(phase_ngsc_source_corr)),
        "n_phase_ngsc_drug_minus_pcb_delta_records": int(len(phase_ngsc_delta)),
        "n_phase_ngsc_drug_minus_pcb_delta_orig_recon_correlations": int(len(phase_ngsc_delta_corr)),
        "min_pairs": int(args.min_pairs),
        "min_edges": int(args.min_edges),
        "allow_metadata_mismatch": bool(args.allow_metadata_mismatch),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote recon parcel-FC correlations to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
