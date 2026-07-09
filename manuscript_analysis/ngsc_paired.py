#!/usr/bin/env python
"""
Compute hemisphere-level NGSC from CIFTI dtseries and run paired condition tests.

NGSC is computed from the PCA/SVD variance spectrum of a time by spatial-node
matrix, following normalized spatial complexity definitions used in Jia et al.
2018 and Siegel et al. 2024:

    NGSC = -sum_i(p_i * log(p_i)) / log(N)

where p_i is the normalized eigenvalue/variance contribution and N is the
number of valid spatial nodes entering the hemisphere-level calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

# Ensure project root and src are on sys.path when running from analysis/.
project_root = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)
src_root = project_root / "src"
for path in (project_root, src_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from matphase.io.cifti import load_cifti  # noqa: E402


# ----------------------------- Config ---------------------------------

DATASET = "DMT"
CONTROL_GROUP = "PCB"
TREATMENT_GROUP = "DMT"
DETECT_RESULTS_DIR = project_root / "detect_results" / DATASET
OUT_DIR = project_root / "analysis_outputs" / "ngsc"

GROUP_ORDER = [CONTROL_GROUP, TREATMENT_GROUP]
HEMISPHERES = ["left", "right"]
STRUCTURE_MAP = {
    "left": "CORTEX_LEFT",
    "right": "CORTEX_RIGHT",
}

Z_SCORE = False
CENTER = True
DENOMINATOR = "n_nodes"
SHOW_PROGRESS = True
RUN_SANITY_TESTS = True
SAVE_FIGS = True
INLINE_PREVIEW = False
RANDOM_SEED = 202406


# ----------------------------- Data types ------------------------------


@dataclass
class NgscResult:
    ngsc: float
    n_timepoints: int
    n_nodes: int
    n_components: int
    total_variance: float
    zscore: bool
    warning: str = ""


@dataclass
class OrientedMatrix:
    x: np.ndarray
    raw_shape: tuple[int, int]
    final_shape: tuple[int, int]
    orientation: str
    warning: str = ""


# ----------------------------- Helpers ---------------------------------


def parse_group_subid(cifti_path: str) -> tuple[str | None, str | None]:
    """Parse group and subject ID from a CIFTI filename."""
    name = Path(cifti_path).name
    subid_match = re.search(r"_S(\d+)", name, flags=re.IGNORECASE)
    if not subid_match:
        return None, None

    group_match = re.search(r"(?:^|_)([^_]+)_S\d+", name, flags=re.IGNORECASE)
    if not group_match:
        return None, None

    group = group_match.group(1).upper()
    allowed_groups = {CONTROL_GROUP.upper(), TREATMENT_GROUP.upper()}
    if group not in allowed_groups:
        return None, None

    subid = f"S{subid_match.group(1)}"
    return group, subid


def mean_center_columns(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)


def zscore_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = np.std(x, axis=0, ddof=1, keepdims=True)
    keep = std.ravel() > 0
    if not np.any(keep):
        return x[:, :0], keep
    return x[:, keep] / std[:, keep], keep


def orient_structure_matrix(data: np.ndarray, n_timepoints: int) -> OrientedMatrix:
    """Return structure data as (time, nodes), recording any ambiguity."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D CIFTI structure data, got shape {data.shape}")

    raw_shape = tuple(int(v) for v in data.shape)
    axis0_is_time = data.shape[0] == n_timepoints
    axis1_is_time = data.shape[1] == n_timepoints

    if axis0_is_time and not axis1_is_time:
        x = data
        orientation = "as_is_time_by_nodes"
        warning = ""
    elif axis1_is_time and not axis0_is_time:
        x = data.T
        orientation = "transposed_nodes_by_time"
        warning = ""
    elif axis0_is_time and axis1_is_time:
        x = data
        orientation = "ambiguous_square_as_is"
        warning = (
            f"Both axes match n_timepoints={n_timepoints}; kept data as "
            "(time, nodes). Verify this square matrix manually."
        )
    else:
        raise ValueError(
            f"Cannot orient shape {raw_shape}: neither axis matches "
            f"CIFTI n_timepoints={n_timepoints}"
        )

    return OrientedMatrix(
        x=np.asarray(x),
        raw_shape=raw_shape,
        final_shape=tuple(int(v) for v in x.shape),
        orientation=orientation,
        warning=warning,
    )


def compute_ngsc(
    x: np.ndarray,
    *,
    zscore: bool = False,
    center: bool = True,
    denominator: str = "n_nodes",
) -> NgscResult:
    """
    Compute paper-style NGSC / normalized spatial complexity.

    Parameters
    ----------
    x : array, shape (time, nodes)
        Time by spatial-node matrix.
    zscore : bool
        If False, use covariance-style PCA, the default main analysis.
        If True, z-score each node time course as a sensitivity analysis.
    center : bool
        Mean-center each node time course before SVD.
    denominator : str
        Must default to "n_nodes". This implementation does not normalize by
        rank or number of nonzero components.
    """
    if denominator != "n_nodes":
        raise ValueError("Only denominator='n_nodes' is supported for paper-style NGSC")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix with shape (time, nodes), got {x.shape}")

    x = np.asarray(x, dtype=np.float64)
    valid_cols = np.all(np.isfinite(x), axis=0)
    x = x[:, valid_cols]

    if center and x.size:
        x = mean_center_columns(x)

    if zscore and x.size:
        x, _ = zscore_columns(x)
    elif x.size:
        keep = np.sum(x * x, axis=0) > 0
        x = x[:, keep]

    n_timepoints, n_nodes = (int(x.shape[0]), int(x.shape[1]))
    base = {
        "n_timepoints": n_timepoints,
        "n_nodes": n_nodes,
        "zscore": bool(zscore),
    }

    if n_nodes <= 1:
        return NgscResult(np.nan, **base, n_components=0, total_variance=np.nan, warning="n_nodes <= 1")
    if n_timepoints < 2:
        return NgscResult(np.nan, **base, n_components=0, total_variance=np.nan, warning="n_timepoints < 2")

    try:
        svals = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        return NgscResult(
            np.nan,
            **base,
            n_components=0,
            total_variance=np.nan,
            warning=f"SVD failed: {exc}",
        )

    energy = svals**2
    total_variance = float(np.sum(energy))
    n_components = int(svals.size)
    if total_variance <= 0 or not np.isfinite(total_variance):
        return NgscResult(
            np.nan,
            **base,
            n_components=n_components,
            total_variance=total_variance,
            warning="total_variance <= 0 or non-finite",
        )

    p = energy / total_variance
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))
    ngsc = entropy / np.log(float(n_nodes))
    return NgscResult(
        ngsc=float(ngsc),
        **base,
        n_components=n_components,
        total_variance=total_variance,
    )


def run_ngsc_sanity_tests() -> None:
    """Small synthetic tests for expected NGSC behavior."""
    rng = np.random.default_rng(RANDOM_SEED)

    t = 600
    latent = rng.normal(size=(t, 1))
    identical = np.repeat(latent, 25, axis=1)
    independent = rng.normal(size=(t, 25))
    scaled = independent * 7.5
    low_rank = rng.normal(size=(t, 3)) @ rng.normal(size=(3, 25))

    identical_result = compute_ngsc(identical, zscore=False)
    independent_result = compute_ngsc(independent, zscore=False)
    scaled_result = compute_ngsc(scaled, zscore=False)
    low_rank_result = compute_ngsc(low_rank, zscore=False)
    with_zero_col = np.column_stack([independent, np.zeros(t)])
    zero_col_result = compute_ngsc(with_zero_col, zscore=False)

    checks = [
        (identical_result.ngsc < 0.05, "identical nodes should have NGSC near 0"),
        (independent_result.ngsc > 0.80, "independent nodes should have high NGSC"),
        (
            np.isclose(independent_result.ngsc, scaled_result.ngsc, atol=1e-12),
            "multiplying the full matrix by a constant should not change NGSC",
        ),
        (
            zero_col_result.n_nodes == independent_result.n_nodes,
            "zero-variance columns should be removed before n_nodes denominator is set",
        ),
        (
            low_rank_result.ngsc < independent_result.ngsc,
            "low-rank data should have lower NGSC than high-rank random data",
        ),
    ]
    failures = [message for ok, message in checks if not ok]
    if failures:
        raise AssertionError("NGSC sanity tests failed: " + "; ".join(failures))

    print(
        "NGSC sanity tests passed: "
        f"identical={identical_result.ngsc:.4f}, "
        f"independent={independent_result.ngsc:.4f}, "
        f"low_rank={low_rank_result.ngsc:.4f}"
    )


def scan_cifti_paths(detect_results_dir: Path) -> pd.DataFrame:
    """Scan detect_results metadata and build a subject/condition table."""
    if not detect_results_dir.exists():
        raise FileNotFoundError(f"detect_results not found: {detect_results_dir}")

    metadata_paths = sorted(detect_results_dir.glob("*/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata.json found under {detect_results_dir}")

    records = []
    for meta_path in metadata_paths:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.warn(f"Skipping unreadable metadata {meta_path}: {exc}")
            continue

        cifti_file = meta.get("cifti_file")
        if not cifti_file:
            continue
        group, subid = parse_group_subid(cifti_file)
        if group is None or subid is None:
            continue
        hemi = meta.get("extra_metadata", {}).get("hemisphere")
        if hemi not in HEMISPHERES:
            continue
        records.append(
            {
                "group": group,
                "subid": subid,
                "hemisphere": hemi,
                "cifti_file": str(cifti_file),
                "metadata_file": str(meta_path),
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No valid metadata records found.")

    df = df.drop_duplicates(subset=["group", "subid", "hemisphere"])
    return df.sort_values(["subid", "group", "hemisphere"]).reset_index(drop=True)


def compute_ngsc_table(subject_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute NGSC per subject/condition/hemisphere plus QC shape records."""
    records = []
    qc_records = []
    iterator = subject_map.itertuples(index=False)
    if SHOW_PROGRESS:
        iterator = tqdm(list(iterator), desc="CIFTI subjects", unit="subject")

    for row in iterator:
        cifti_path = Path(row.cifti_file)
        context = f"{row.subid} {row.group} {row.hemisphere} {cifti_path}"
        if not cifti_path.exists():
            warnings.warn(f"Missing CIFTI for {context}")
            continue

        try:
            ts = load_cifti(cifti_path)
        except Exception as exc:
            warnings.warn(f"Failed to load CIFTI for {context}: {exc}")
            continue

        structure = STRUCTURE_MAP[row.hemisphere]
        try:
            data = ts.get_structure_data(structure)
            oriented = orient_structure_matrix(data, ts.metadata.n_timepoints)
        except Exception as exc:
            warnings.warn(f"Failed to extract/orient {structure} for {context}: {exc}")
            continue

        if oriented.warning:
            warnings.warn(f"{context}: {oriented.warning}")

        result = compute_ngsc(
            oriented.x,
            zscore=Z_SCORE,
            center=CENTER,
            denominator=DENOMINATOR,
        )
        if result.warning:
            warnings.warn(f"{context}: {result.warning}")

        print(
            f"{row.subid} {row.group} {row.hemisphere} {structure}: "
            f"raw_shape={oriented.raw_shape}, final_shape={oriented.final_shape}, "
            f"n_nodes={result.n_nodes}, ngsc={result.ngsc:.6g}"
        )

        records.append(
            {
                "subid": row.subid,
                "group": row.group,
                "hemisphere": row.hemisphere,
                "structure": structure,
                "ngsc": result.ngsc,
                "n_timepoints": result.n_timepoints,
                "n_nodes": result.n_nodes,
                "n_components": result.n_components,
                "total_variance": result.total_variance,
                "zscore": result.zscore,
                "center": CENTER,
                "denominator": DENOMINATOR,
                "cifti_file": str(cifti_path),
                "metadata_file": row.metadata_file,
                "warning": result.warning,
            }
        )
        qc_records.append(
            {
                "subid": row.subid,
                "group": row.group,
                "hemisphere": row.hemisphere,
                "structure": structure,
                "cifti_file": str(cifti_path),
                "cifti_n_timepoints": int(ts.metadata.n_timepoints),
                "raw_shape_0": oriented.raw_shape[0],
                "raw_shape_1": oriented.raw_shape[1],
                "final_timepoints": oriented.final_shape[0],
                "final_nodes_before_filter": oriented.final_shape[1],
                "n_nodes_after_filter": result.n_nodes,
                "n_components": result.n_components,
                "orientation": oriented.orientation,
                "warning": "; ".join(v for v in [oriented.warning, result.warning] if v),
            }
        )

    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(qc_records)


def paired_table(df: pd.DataFrame, hemi: str) -> pd.DataFrame:
    subset = df[df["hemisphere"] == hemi].copy()
    pivot = subset.pivot(index="subid", columns="group", values="ngsc").reset_index()
    missing = [group for group in GROUP_ORDER if group not in pivot.columns]
    if missing:
        return pd.DataFrame(columns=["subid", *GROUP_ORDER])
    return pivot.dropna(subset=GROUP_ORDER, how="any")


def summarize_paired_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Run paired treatment-control tests separately per hemisphere."""
    rows = []
    control_group, treatment_group = GROUP_ORDER
    diff_direction = f"{treatment_group} - {control_group}"
    for hemi in HEMISPHERES:
        pivot = paired_table(df, hemi)
        structure = STRUCTURE_MAP[hemi]
        if pivot.empty:
            rows.append(
                {
                    "hemisphere": hemi,
                    "structure": structure,
                    "control_group": control_group,
                    "treatment_group": treatment_group,
                    "diff_direction": diff_direction,
                    "n_pairs": 0,
                }
            )
            continue

        control = pivot[control_group].to_numpy(dtype=float)
        treatment = pivot[treatment_group].to_numpy(dtype=float)
        diff = treatment - control
        n = int(diff.size)

        if n >= 2:
            t_val, p_val = stats.ttest_rel(treatment, control, nan_policy="omit")
            diff_mean = float(np.mean(diff))
            diff_sd = float(np.std(diff, ddof=1))
            sem = diff_sd / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, df=n - 1)
            ci_low = diff_mean - tcrit * sem
            ci_high = diff_mean + tcrit * sem
            dz = diff_mean / diff_sd if diff_sd > np.finfo(float).eps else np.nan
        else:
            t_val = p_val = ci_low = ci_high = dz = np.nan
            diff_mean = float(np.mean(diff))
            diff_sd = np.nan

        rows.append(
            {
                "hemisphere": hemi,
                "structure": structure,
                "control_group": control_group,
                "treatment_group": treatment_group,
                "diff_direction": diff_direction,
                "n_pairs": n,
                "control_mean": float(np.mean(control)),
                "control_sd": float(np.std(control, ddof=1)) if n >= 2 else np.nan,
                "treatment_mean": float(np.mean(treatment)),
                "treatment_sd": float(np.std(treatment, ddof=1)) if n >= 2 else np.nan,
                "mean_diff": diff_mean,
                "diff_sd": diff_sd,
                "paired_t": float(t_val),
                "p_value": float(p_val),
                "cohens_dz": float(dz),
                "ci95_diff_low": float(ci_low),
                "ci95_diff_high": float(ci_high),
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_violin_paired(df: pd.DataFrame, hemi: str) -> plt.Figure | None:
    subset = df[df["hemisphere"] == hemi].copy()
    if subset.empty:
        print(f"No data for hemisphere={hemi}")
        return None

    structure = STRUCTURE_MAP[hemi]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if sns is not None:
        sns.violinplot(
            data=subset,
            x="group",
            y="ngsc",
            order=GROUP_ORDER,
            cut=0,
            inner="quartile",
            color="#d8e2dc",
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="group",
            y="ngsc",
            order=GROUP_ORDER,
            color="black",
            size=4,
            alpha=0.75,
            jitter=0.08,
            ax=ax,
        )
    else:
        values = [
            subset.loc[subset["group"] == group, "ngsc"].dropna().to_numpy()
            for group in GROUP_ORDER
        ]
        ax.violinplot(values, positions=[0, 1], showmeans=False, showmedians=True)
        rng = np.random.default_rng(RANDOM_SEED)
        for idx, vals in enumerate(values):
            jitter = rng.uniform(-0.05, 0.05, size=len(vals))
            ax.scatter(
                np.full(len(vals), idx) + jitter,
                vals,
                color="black",
                s=18,
                alpha=0.75,
                zorder=3,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(GROUP_ORDER)

    pivot = paired_table(df, hemi)
    for _, row in pivot.iterrows():
        ax.plot(
            [0, 1],
            [row[GROUP_ORDER[0]], row[GROUP_ORDER[1]]],
            color="gray",
            alpha=0.45,
            linewidth=0.9,
            zorder=1,
        )

    if len(pivot) >= 2:
        control_group, treatment_group = GROUP_ORDER
        t_val, p_val = stats.ttest_rel(
            pivot[treatment_group], pivot[control_group], nan_policy="omit"
        )
        ax.text(
            0.02,
            0.95,
            f"n={len(pivot)}, paired p={p_val:.3g}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
        )

    ax.set_title(f"NGSC {structure} (zscore={Z_SCORE})")
    ax.set_xlabel("")
    ax.set_ylabel("NGSC")
    fig.tight_layout()

    if SAVE_FIGS:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "zscore" if Z_SCORE else "no_zscore"
        fig.savefig(OUT_DIR / f"ngsc_paired_{hemi}_{suffix}.png", dpi=300)
    if INLINE_PREVIEW:
        return fig
    plt.close(fig)
    return None


def write_outputs(
    subject_map: pd.DataFrame,
    ngsc_table: pd.DataFrame,
    stats_table: pd.DataFrame,
    qc_table: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ngsc_table.to_csv(OUT_DIR / "ngsc_subject_hemisphere_values.csv", index=False)
    stats_table.to_csv(OUT_DIR / "ngsc_paired_stats.csv", index=False)
    qc_table.to_csv(OUT_DIR / "ngsc_qc_shapes.csv", index=False)

    config = {
        "dataset": DATASET,
        "detect_results_dir": str(DETECT_RESULTS_DIR),
        "out_dir": str(OUT_DIR),
        "control_group": CONTROL_GROUP,
        "treatment_group": TREATMENT_GROUP,
        "group_order": GROUP_ORDER,
        "hemispheres": HEMISPHERES,
        "structure_map": STRUCTURE_MAP,
        "zscore": Z_SCORE,
        "center": CENTER,
        "denominator": DENOMINATOR,
        "run_sanity_tests": RUN_SANITY_TESTS,
        "n_metadata_records": int(len(subject_map)),
        "n_ngsc_records": int(len(ngsc_table)),
        "outputs": {
            "subject_values": "ngsc_subject_hemisphere_values.csv",
            "paired_stats": "ngsc_paired_stats.csv",
            "qc_shapes": "ngsc_qc_shapes.csv",
            "config": "ngsc_config.json",
            "plots": [
                f"ngsc_paired_{hemi}_{'zscore' if Z_SCORE else 'no_zscore'}.png"
                for hemi in HEMISPHERES
            ],
        },
    }
    (OUT_DIR / "ngsc_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hemisphere-level NGSC and paired condition statistics."
    )
    parser.add_argument("--dataset", default=DATASET, help="Dataset label, e.g. DMT or LSD.")
    parser.add_argument("--control-group", default=CONTROL_GROUP)
    parser.add_argument("--treatment-group", default=TREATMENT_GROUP)
    parser.add_argument(
        "--detect-results-dir",
        type=Path,
        default=None,
        help="Directory containing detection bundle metadata. Defaults to detect_results/<dataset>.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to analysis_outputs/ngsc for DMT, analysis_outputs/ngsc_<dataset> otherwise.",
    )
    parser.add_argument("--zscore", action="store_true", help="Use per-node z-score sensitivity analysis.")
    parser.add_argument("--skip-sanity-tests", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--inline-preview", action="store_true")
    parser.add_argument("--no-save-figs", action="store_true")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global DATASET, CONTROL_GROUP, TREATMENT_GROUP, GROUP_ORDER
    global DETECT_RESULTS_DIR, OUT_DIR, Z_SCORE, SHOW_PROGRESS
    global RUN_SANITY_TESTS, INLINE_PREVIEW, SAVE_FIGS

    DATASET = args.dataset.upper()
    CONTROL_GROUP = args.control_group.upper()
    TREATMENT_GROUP = args.treatment_group.upper()
    if CONTROL_GROUP == TREATMENT_GROUP:
        raise ValueError("--control-group and --treatment-group must differ")
    GROUP_ORDER = [CONTROL_GROUP, TREATMENT_GROUP]

    DETECT_RESULTS_DIR = (
        args.detect_results_dir
        if args.detect_results_dir is not None
        else project_root / "detect_results" / DATASET
    )
    OUT_DIR = (
        args.out_dir
        if args.out_dir is not None
        else (
            project_root / "analysis_outputs" / "ngsc"
            if DATASET == "DMT"
            else project_root / "analysis_outputs" / f"ngsc_{DATASET.lower()}"
        )
    )
    Z_SCORE = bool(args.zscore)
    RUN_SANITY_TESTS = not args.skip_sanity_tests
    SHOW_PROGRESS = not args.no_progress
    INLINE_PREVIEW = bool(args.inline_preview)
    SAVE_FIGS = not args.no_save_figs


def main() -> int:
    args = parse_args()
    apply_args(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    if RUN_SANITY_TESTS:
        run_ngsc_sanity_tests()

    subject_map = scan_cifti_paths(DETECT_RESULTS_DIR)
    print(f"Metadata records found: {len(subject_map)}")

    ngsc_table, qc_table = compute_ngsc_table(subject_map)
    if ngsc_table.empty:
        raise RuntimeError("No NGSC records were computed.")

    stats_table = summarize_paired_stats(ngsc_table)
    write_outputs(subject_map, ngsc_table, stats_table, qc_table)

    for hemi in HEMISPHERES:
        plot_violin_paired(ngsc_table, hemi)

    print(f"\nPaired statistics (diff = {GROUP_ORDER[1]} - {GROUP_ORDER[0]}):")
    print(stats_table.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
