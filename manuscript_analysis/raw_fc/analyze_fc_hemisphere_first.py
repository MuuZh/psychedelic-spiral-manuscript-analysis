#!/usr/bin/env python3
"""Compute hemisphere-first FC from Workbench ptseries and compare two groups.

This differs from analyze_fc_results_v2.py in the ordering of operations:
1. load each whole-brain ptseries
2. split parcel time series into left/right hemispheres
3. compute FC separately within each hemisphere

The existing Workbench pipeline computes a whole-brain pconn first and the
analysis script then slices LL/RR blocks from that matrix. For Pearson
correlation those LL/RR values are numerically equivalent, but this script
materializes the hemisphere-first matrices and keeps them separate on disk.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.cifti2 import Cifti2Image
from nibabel.cifti2.cifti2_axes import ParcelsAxis
from tqdm import tqdm

from analyze_fc_results_v2 import (
    LEFT_DEFAULT,
    RIGHT_DEFAULT,
    compare_groups,
    compute_metrics,
    ensure_dir,
    finite_upper_triangle,
    get_parcel_names,
    hemisphere_indices,
    infer_pair_id,
    make_matrix_visuals,
    make_summary_plots,
    maybe_compare_with_vortex,
    save_parcel_names,
    setup_logger,
    strip_known_suffixes,
)


@dataclass(frozen=True)
class PtseriesGroupSpec:
    label: str
    ptseries_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute left/right hemisphere FC from ptseries first, then run the same "
            "subject-level and group-level summaries as the pconn analysis."
        )
    )
    parser.add_argument(
        "--group-a-dir",
        required=True,
        type=Path,
        help="Group A directory. May be a ptseries directory or an fc_out group root containing ptseries/.",
    )
    parser.add_argument(
        "--group-b-dir",
        required=True,
        type=Path,
        help="Group B directory. May be a ptseries directory or an fc_out group root containing ptseries/.",
    )
    parser.add_argument("--group-a-label", default="drug")
    parser.add_argument("--group-b-label", default="pcb")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--left-regex", default=LEFT_DEFAULT)
    parser.add_argument("--right-regex", default=RIGHT_DEFAULT)
    parser.add_argument(
        "--test-mode",
        choices=["independent", "paired"],
        default="independent",
        help="Use independent-samples test or pair subjects by pair ID and use paired t-test.",
    )
    parser.add_argument(
        "--pair-id-regex",
        default=None,
        help="Optional regex with one capture group used to extract pair IDs from ptseries filenames.",
    )
    parser.add_argument(
        "--multiple-comparison",
        choices=["none", "fdr"],
        default="none",
        help="How to handle multiple comparisons in summary/edgewise outputs.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--vortex-csv", type=Path)
    parser.add_argument("--save-subject-heatmaps", action="store_true")
    parser.add_argument("--max-subject-heatmaps", type=int, default=12)
    parser.add_argument(
        "--no-save-hemi-pconn",
        action="store_true",
        help="Do not save subject-level left/right pconn CIFTI files.",
    )
    return parser.parse_args()


def collect_ptseries_files(path: Path) -> List[Path]:
    files = sorted(path.glob("*.ptseries.nii"))
    if not files:
        nested = path / "ptseries"
        files = sorted(nested.glob("*.ptseries.nii"))
    return files


def fisher_z_correlation(timeseries: np.ndarray) -> Tuple[np.ndarray, int]:
    data = np.asarray(timeseries, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D time-by-parcel array, got shape {data.shape}")
    if data.shape[0] < 2:
        raise ValueError("Need at least 2 time points to compute FC.")

    finite = np.isfinite(data)
    valid = finite.all(axis=0) & (np.nanstd(data, axis=0, ddof=1) > 0)
    invalid_count = int((~valid).sum())

    corr = np.full((data.shape[1], data.shape[1]), np.nan, dtype=float)
    if valid.sum() >= 2:
        valid_corr = np.corrcoef(data[:, valid], rowvar=False)
        corr[np.ix_(valid, valid)] = valid_corr
    elif valid.sum() == 1:
        corr[np.ix_(valid, valid)] = 1.0

    corr = np.asarray(corr, dtype=float)
    corr = np.clip(corr, -0.999999, 0.999999)
    return np.arctanh(corr), invalid_count


def save_hemi_pconn(matrix: np.ndarray, axis: ParcelsAxis, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    img = Cifti2Image(np.asarray(matrix, dtype=np.float32), header=(axis, axis))
    nib.save(img, str(out_path))


def load_subject_metrics_and_matrices_from_ptseries(
    group: PtseriesGroupSpec,
    left_regex: str,
    right_regex: str,
    pair_id_regex: Optional[str],
    out_dir: Path,
    save_hemi_pconn_files: bool,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, List[np.ndarray]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, np.ndarray]]]:
    files = collect_ptseries_files(group.ptseries_dir)
    if not files:
        raise FileNotFoundError(f"No .ptseries.nii files found in {group.ptseries_dir}")

    rows: List[Dict[str, object]] = []
    matrices: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
    subjects_by_hemi: Dict[str, List[str]] = {"left": [], "right": []}
    parcels_by_hemi: Dict[str, List[str]] = {}
    matrices_by_pair: Dict[str, Dict[str, np.ndarray]] = {"left": {}, "right": {}}

    for file_path in tqdm(files, desc=f"Loading {group.label}", unit="file"):
        img = nib.load(str(file_path))
        data = np.asarray(img.dataobj, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D ptseries in {file_path}, got shape {data.shape}")

        parcel_axis_number = None
        parcel_axis: Optional[ParcelsAxis] = None
        for axis_number in range(2):
            axis = img.header.get_axis(axis_number)
            if isinstance(axis, ParcelsAxis):
                parcel_axis_number = axis_number
                parcel_axis = axis
                break
        if parcel_axis_number is None or parcel_axis is None:
            raise ValueError(f"Expected one ParcelsAxis in {file_path}")

        names = get_parcel_names(parcel_axis)
        hemi_map = hemisphere_indices(names, left_regex=left_regex, right_regex=right_regex)

        if parcel_axis_number == 0:
            time_by_parcel = data.T
        else:
            time_by_parcel = data
        if time_by_parcel.shape[1] != len(names):
            raise ValueError(
                f"Parcel axis length mismatch in {file_path}: data shape {data.shape}, parcels={len(names)}"
            )

        subject_id = strip_known_suffixes(file_path.name)
        pair_id = infer_pair_id(subject_id, pair_id_regex)

        for hemi in ["left", "right"]:
            idx = hemi_map[hemi]
            hemi_names = [names[i] for i in idx]
            hemi_axis = parcel_axis[idx]
            hemi_matrix, invalid_count = fisher_z_correlation(time_by_parcel[:, idx])
            if invalid_count:
                logger.warning(
                    "%s %s %s: %d parcel(s) had non-finite or zero-variance time series; related FC edges are NaN.",
                    group.label,
                    subject_id,
                    hemi,
                    invalid_count,
                )

            if hemi not in parcels_by_hemi:
                parcels_by_hemi[hemi] = hemi_names
            elif parcels_by_hemi[hemi] != hemi_names:
                raise ValueError(
                    f"Parcel ordering mismatch within group {group.label} for hemisphere {hemi}. File: {file_path}"
                )

            if save_hemi_pconn_files:
                save_hemi_pconn(
                    hemi_matrix,
                    hemi_axis,
                    out_dir / "hemisphere_pconn" / group.label / hemi / f"{subject_id}_{hemi}.pconn.nii",
                )

            tri = finite_upper_triangle(hemi_matrix)
            metric_row = {
                "group": group.label,
                "subject_id": subject_id,
                "pair_id": pair_id,
                "hemisphere": hemi,
                "n_parcels": int(len(idx)),
                "n_edges": int(tri.size),
            }
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                metric_row.update(compute_metrics(tri, hemi_matrix))
            rows.append(metric_row)

            matrices[hemi].append(hemi_matrix)
            subjects_by_hemi[hemi].append(subject_id)
            matrices_by_pair[hemi][pair_id] = hemi_matrix

    logger.info("Loaded %d ptseries files for group %s.", len(files), group.label)
    return pd.DataFrame(rows), matrices, subjects_by_hemi, parcels_by_hemi, matrices_by_pair


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "logs" / "analyze_fc_hemisphere_first.log")

    logger.info("Hemisphere-first FC from ptseries")
    logger.info("test_mode=%s | multiple_comparison=%s | alpha=%.4f", args.test_mode, args.multiple_comparison, args.alpha)
    if args.test_mode == "paired":
        logger.info("paired analysis enabled; pair_id_regex=%s", args.pair_id_regex or "<auto>")

    group_a = PtseriesGroupSpec(label=args.group_a_label, ptseries_dir=args.group_a_dir)
    group_b = PtseriesGroupSpec(label=args.group_b_label, ptseries_dir=args.group_b_dir)
    save_hemi_pconn_files = not args.no_save_hemi_pconn

    logger.info("Loading group A ptseries from %s", group_a.ptseries_dir)
    df_a, mats_a, subjects_a, parcels_a, pair_mats_a = load_subject_metrics_and_matrices_from_ptseries(
        group_a,
        args.left_regex,
        args.right_regex,
        args.pair_id_regex,
        args.out_dir,
        save_hemi_pconn_files,
        logger,
    )
    logger.info("Loading group B ptseries from %s", group_b.ptseries_dir)
    df_b, mats_b, subjects_b, parcels_b, pair_mats_b = load_subject_metrics_and_matrices_from_ptseries(
        group_b,
        args.left_regex,
        args.right_regex,
        args.pair_id_regex,
        args.out_dir,
        save_hemi_pconn_files,
        logger,
    )

    for hemi in ["left", "right"]:
        if hemi in parcels_a and hemi in parcels_b and parcels_a[hemi] != parcels_b[hemi]:
            raise ValueError(f"Parcel ordering mismatch between groups for hemisphere '{hemi}'.")

    if args.test_mode == "paired":
        common_ids = sorted(set(df_a["pair_id"]) & set(df_b["pair_id"]))
        logger.info("Found %d paired IDs shared by both groups. Example IDs: %s", len(common_ids), common_ids[:10])
        if len(common_ids) < 2:
            raise ValueError("Paired mode requested, but fewer than 2 shared pair IDs were found across groups.")

    parcel_dir = args.out_dir / "parcel_info"
    save_parcel_names(parcels_a, parcel_dir, f"{args.group_a_label}")
    save_parcel_names(parcels_b, parcel_dir, f"{args.group_b_label}")

    df = pd.concat([df_a, df_b], ignore_index=True)
    subject_csv = args.out_dir / "subject_level_fc_metrics.csv"
    df.to_csv(subject_csv, index=False)
    logger.info("Wrote subject-level metrics to %s", subject_csv)

    stats_df = compare_groups(df, args.group_a_label, args.group_b_label, args.test_mode, args.multiple_comparison, args.alpha)
    stats_csv = args.out_dir / "group_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Wrote group statistics to %s", stats_csv)

    method_df = maybe_compare_with_vortex(stats_df, args.vortex_csv, logger, args.test_mode, args.multiple_comparison, args.alpha)
    if method_df is not None:
        method_csv = args.out_dir / "method_comparison.csv"
        method_df.to_csv(method_csv, index=False)
        logger.info("Wrote method comparison table to %s", method_csv)

    make_summary_plots(df, stats_df, args.out_dir, args.group_a_label, args.group_b_label, args.test_mode, args.multiple_comparison, args.alpha)
    make_matrix_visuals(
        args.group_a_label,
        args.group_b_label,
        mats_a,
        mats_b,
        subjects_a,
        subjects_b,
        pair_mats_a,
        pair_mats_b,
        args.out_dir,
        args.save_subject_heatmaps,
        args.max_subject_heatmaps,
        args.test_mode,
        args.multiple_comparison,
        args.alpha,
    )

    if save_hemi_pconn_files:
        logger.info("Saved subject hemisphere pconn files under %s", args.out_dir / "hemisphere_pconn")
    logger.info("Saved plots under %s", args.out_dir / "plots")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
