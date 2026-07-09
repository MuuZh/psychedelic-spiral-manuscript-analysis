from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dfr_cache import cache_key, cache_path, load_cached_summary, write_cached_summary
from .dfr_core import DfrParams, default_qc_frames, summarize_phase_cube, valid_mask_from_phase_and_grid
from .dfr_io import discover_phase_entries, ensure_outdirs, load_grid
from .dfr_network_overlap import compute_network_overlap_rows
from .dfr_plots import save_frame_qc, save_occupancy_map, save_paired_plot, save_scatter
from .dfr_recon_compare import correlation_rows, delta_table, paired_stats
from .dfr_spiral_overlap import (
    compute_spiral_boundary_overlap,
    dfr_spiral_delta_correlations,
    load_spiral_deltas,
    spiral_overlap_delta_table,
)


DRUG_DEFAULTS = {
    "DMT": {
        "drug_condition": "DMT_DMT",
        "pcb_condition": "DMT_PCB",
        "empirical_root": Path("analysis_outputs/phase_fc_batch_phase_corr_7networks"),
        "recon_root": Path("analysis_outputs/phase_fc_recon_7networks"),
    },
    "LSD": {
        "drug_condition": "LSD_LSD",
        "pcb_condition": "LSD_PCB",
        "empirical_root": Path("analysis_outputs/phase_fc_batch_phase_corr_7networks_LSD"),
        "recon_root": Path("analysis_outputs/phase_fc_recon_7networks_LSD"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic functional region analysis from phase cubes.")
    parser.add_argument("--drug", choices=["DMT", "LSD", "both"], default="both")
    parser.add_argument("--drug-condition", default=None, help="Override when running one drug.")
    parser.add_argument("--pcb-condition", default=None, help="Override when running one drug.")
    parser.add_argument("--dmt-empirical-root", type=Path, default=DRUG_DEFAULTS["DMT"]["empirical_root"])
    parser.add_argument("--dmt-recon-root", type=Path, default=DRUG_DEFAULTS["DMT"]["recon_root"])
    parser.add_argument("--lsd-empirical-root", type=Path, default=DRUG_DEFAULTS["LSD"]["empirical_root"])
    parser.add_argument("--lsd-recon-root", type=Path, default=DRUG_DEFAULTS["LSD"]["recon_root"])
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/dfr_analysis"))
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument("--magnitude-threshold", type=float, default=0.2)
    parser.add_argument("--boundary-thickness", type=int, default=2)
    parser.add_argument("--min-region-size", type=int, default=10)
    parser.add_argument("--edge-margin", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--network-boundary-tolerance", type=int, default=1)
    parser.add_argument("--network-overlap-mode", choices=["top-mean", "occupancy-threshold"], default="top-mean")
    parser.add_argument("--network-dfr-occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--min-corr-n", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per source/drug after discovery.")
    parser.add_argument("--save-frame-qc", action="store_true")
    parser.add_argument("--qc-subjects-per-condition", type=int, default=3)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("analysis_outputs/dfr_analysis/cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--spiral-bundle-root",
        type=Path,
        action="append",
        default=[Path("detect_results")],
        help="Root containing detection bundles with frame_index.parquet. Can be repeated.",
    )
    parser.add_argument(
        "--spiral-root",
        type=Path,
        action="append",
        default=[Path("analysis_outputs/phase_fc_group/network_spiral_metrics"), Path("analysis_outputs/path_entropy")],
        help="Root or CSV containing spiral/path-entropy delta tables. Can be repeated.",
    )
    return parser.parse_args()


def drug_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    drugs = ["DMT", "LSD"] if args.drug == "both" else [args.drug]
    specs = []
    for drug in drugs:
        default = DRUG_DEFAULTS[drug]
        specs.append(
            {
                "drug": drug,
                "drug_condition": args.drug_condition if args.drug != "both" and args.drug_condition else default["drug_condition"],
                "pcb_condition": args.pcb_condition if args.drug != "both" and args.pcb_condition else default["pcb_condition"],
                "empirical_roots": [args.dmt_empirical_root if drug == "DMT" else args.lsd_empirical_root],
                "recon_roots": [args.dmt_recon_root if drug == "DMT" else args.lsd_recon_root],
            }
        )
    return specs


def should_save_qc(qc_counts: dict[tuple[str, str, str, str], int], row: pd.Series, max_subjects: int) -> bool:
    key = (str(row["source"]), str(row["condition"]), str(row["hemisphere"]), str(row["role"]))
    if qc_counts.get(key, 0) >= max_subjects:
        return False
    qc_counts[key] = qc_counts.get(key, 0) + 1
    return True


def process_entry(
    row: pd.Series,
    params: DfrParams,
    out_paths: dict[str, Path],
    save_qc: bool,
    no_plots: bool,
    network_tolerance: int,
    network_overlap_mode: str,
    network_dfr_occupancy_threshold: float,
    spiral_bundle_roots: list[Path],
    cache_dir: Path,
    use_cache: bool,
    show_progress: bool,
) -> tuple[
    dict[str, object],
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
    list[dict[str, str]],
]:
    failures: list[dict[str, str]] = []
    phase_path = Path(row["phase_cube"])
    phase_cube = np.load(phase_path, mmap_mode="r")
    if phase_cube.ndim != 3:
        raise ValueError(f"Expected phase cube shape (H, W, T), got {phase_cube.shape}: {phase_path}")
    grid = load_grid(row)
    valid_mask = valid_mask_from_phase_and_grid(phase_cube, grid, params.edge_margin)
    qc_frames = default_qc_frames(phase_cube.shape[2]) if save_qc else []
    grid_path = Path(row["grid_path"]) if row.get("grid_path") and not pd.isna(row.get("grid_path")) else None
    cfile = cache_path(cache_dir, cache_key(phase_path, grid_path, params, qc_frames))
    cached = load_cached_summary(cfile) if use_cache else None
    cache_hit = cached is not None
    if cached is not None:
        summary, qc, avg_boundary, boundary_stack = cached
    else:
        summary, qc, avg_boundary, boundary_stack = summarize_phase_cube(
            phase_cube,
            valid_mask,
            params,
            qc_frames=qc_frames,
            show_progress=show_progress,
            progress_desc=f"DFR {row['source']} {row['condition']} S{row['subid']} {row['hemisphere']}",
        )
        if use_cache:
            write_cached_summary(cfile, summary, qc, avg_boundary, boundary_stack)

    base = {
        "source": row["source"],
        "drug": row["drug"],
        "comparison": row["comparison"],
        "condition": row["condition"],
        "role": row["role"],
        "subid": row["subid"],
        "hemisphere": row["hemisphere"],
        "phase_cube": str(phase_path),
        "bundle_dir": str(row["bundle_dir"]),
        "cache_hit": bool(cache_hit),
        **summary,
    }

    map_name = f"{row['source']}_{row['condition']}_S{row['subid']}_{row['hemisphere']}"
    np.savez_compressed(
        out_paths["maps"] / f"boundary_occupancy_{map_name}.npz",
        avg_boundary=avg_boundary,
        valid_mask=valid_mask,
    )

    overlap_row = None
    distance_row = None
    spiral_row = None
    atlas_boundary = None
    try:
        metric_row = row.copy()
        for key, value in summary.items():
            metric_row[key] = value
        overlap, distance, atlas_boundary = compute_network_overlap_rows(
            metric_row,
            avg_boundary,
            valid_mask,
            tolerance=network_tolerance,
            dfr_mode=network_overlap_mode,
            occupancy_threshold=network_dfr_occupancy_threshold,
        )
        if overlap is not None:
            overlap_row = {**base, **overlap}
            distance_row = {**base, **(distance or {})}
    except Exception as exc:
        failures.append({"bundle_dir": str(row["bundle_dir"]), "error": f"network_overlap_failed: {exc}"})

    spiral_centers_by_frame: dict[int, pd.DataFrame] = {}
    try:
        spiral, spiral_failures, spiral_centers_by_frame = compute_spiral_boundary_overlap(
            row,
            boundary_stack,
            spiral_bundle_roots,
            show_progress=show_progress,
        )
        failures.extend(spiral_failures)
        if spiral is not None:
            spiral_row = {**base, **spiral}
            if base["n_frames"]:
                spiral_row["spiral_centers_per_frame"] = float(spiral_row["n_spiral_centers"]) / float(base["n_frames"])
    except Exception as exc:
        failures.append({"bundle_dir": str(row["bundle_dir"]), "error": f"spiral_overlap_failed: {exc}"})

    if not no_plots:
        save_occupancy_map(
            avg_boundary,
            atlas_boundary,
            out_paths["figures"] / f"boundary_occupancy_maps_{map_name}.png",
            f"{row['condition']} S{row['subid']} {row['hemisphere']} {row['source']}",
        )

    if save_qc and not no_plots:
        for frame, frame_dfr in qc.items():
            save_frame_qc(
                np.asarray(phase_cube[:, :, frame]),
                frame_dfr,
                atlas_boundary,
                spiral_centers_by_frame.get(frame),
                out_paths["frame_qc"] / f"frame_qc_{map_name}_f{frame:04d}.png",
                f"{row['source']} {row['condition']} S{row['subid']} {row['hemisphere']} frame {frame}",
            )

    return base, overlap_row, distance_row, spiral_row, failures


def write_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and columns is not None:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    out_paths = ensure_outdirs(args.out_dir)
    params = DfrParams(
        magnitude_threshold=args.magnitude_threshold,
        boundary_thickness=args.boundary_thickness,
        min_region_size=args.min_region_size,
        edge_margin=args.edge_margin,
        spacing=args.spacing,
    )
    hemispheres = ["left", "right"] if args.hemisphere == "both" else [args.hemisphere]

    discovered = []
    failures: list[dict[str, str]] = []
    for spec in drug_specs(args):
        emp, fail = discover_phase_entries(spec["empirical_roots"], "empirical", spec["drug"], spec["drug_condition"], spec["pcb_condition"], hemispheres)
        rec, fail2 = discover_phase_entries(spec["recon_roots"], "recon", spec["drug"], spec["drug_condition"], spec["pcb_condition"], hemispheres)
        failures.extend(fail)
        failures.extend(fail2)
        for df in [emp, rec]:
            if not df.empty:
                discovered.append(df)

    entries = pd.concat(discovered, ignore_index=True) if discovered else pd.DataFrame()
    if args.limit is not None and not entries.empty:
        entries = entries.groupby(["source", "drug"], group_keys=False).head(args.limit)

    records = []
    overlap_records = []
    distance_records = []
    spiral_overlap_records = []
    qc_counts: dict[tuple[str, str, str, str], int] = {}
    row_iter = list(entries.itertuples(index=False))
    if row_iter and not args.no_progress:
        row_iter = tqdm(row_iter, desc="DFR entries", unit="entry")
    for row in row_iter:
        series = pd.Series(row._asdict())
        try:
            save_qc = args.save_frame_qc and should_save_qc(qc_counts, series, args.qc_subjects_per_condition)
            record, overlap, distance, spiral_overlap, local_failures = process_entry(
                series,
                params,
                out_paths,
                save_qc=save_qc,
                no_plots=args.no_plots,
                network_tolerance=args.network_boundary_tolerance,
                network_overlap_mode=args.network_overlap_mode,
                network_dfr_occupancy_threshold=args.network_dfr_occupancy_threshold,
                spiral_bundle_roots=args.spiral_bundle_root,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
                show_progress=not args.no_progress,
            )
            records.append(record)
            if overlap is not None:
                overlap_records.append(overlap)
            if distance is not None:
                distance_records.append(distance)
            if spiral_overlap is not None:
                spiral_overlap_records.append(spiral_overlap)
            failures.extend(local_failures)
            print(f"Processed {record['source']} {record['condition']} S{record['subid']} {record['hemisphere']}")
        except Exception as exc:
            failures.append({"bundle_dir": str(series.get("bundle_dir", "")), "error": str(exc)})
            print(f"Failed {series.get('bundle_dir', '')}: {exc}")

    per_subject = pd.DataFrame(records)
    deltas = delta_table(per_subject) if not per_subject.empty else pd.DataFrame()
    stats_df = paired_stats(per_subject, min_pairs=args.min_pairs) if not per_subject.empty else pd.DataFrame()

    empirical_delta = deltas[deltas["source"] == "empirical"] if "source" in deltas else pd.DataFrame()
    recon_delta = deltas[deltas["source"] == "recon"] if "source" in deltas else pd.DataFrame()
    emp_recon_corr = correlation_rows(
        empirical_delta,
        recon_delta,
        "empirical",
        "recon",
        ["delta_region_count", "delta_boundary_count"],
        min_n=args.min_corr_n,
    )

    spiral_delta, spiral_warnings = load_spiral_deltas(args.spiral_root)
    if spiral_delta.empty and spiral_overlap_records:
        spiral_delta = spiral_overlap_delta_table(pd.DataFrame(spiral_overlap_records))
    spiral_corr = dfr_spiral_delta_correlations(empirical_delta, spiral_delta, min_n=args.min_corr_n)
    failures.extend({"bundle_dir": "", "error": warning} for warning in spiral_warnings)

    write_csv(per_subject, args.out_dir / "per_condition_subject_dfr.csv")
    write_csv(stats_df, args.out_dir / "paired_dfr_stats.csv")
    write_csv(deltas, args.out_dir / "per_subject_dfr_drug_minus_pcb_delta.csv")
    write_csv(spiral_corr, args.out_dir / "dfr_spiral_delta_correlations.csv")
    write_csv(pd.DataFrame(spiral_overlap_records), args.out_dir / "dfr_spiral_boundary_overlap.csv")
    write_csv(pd.DataFrame(overlap_records), args.out_dir / "dfr_network_boundary_overlap.csv")
    write_csv(pd.DataFrame(distance_records), args.out_dir / "dfr_network_boundary_distance.csv")
    write_csv(emp_recon_corr, args.out_dir / "empirical_recon_dfr_delta_correlations.csv")
    if failures:
        write_csv(pd.DataFrame(failures), args.out_dir / "failures.csv")

    if not args.no_plots and not per_subject.empty:
        for (source, drug, comparison, hemi), group in per_subject.groupby(["source", "drug", "comparison", "hemisphere"]):
            tag = f"{source}_{drug}_{hemi}"
            save_paired_plot(group, "mean_region_count", args.out_dir / "figures" / f"paired_region_count_{tag}.png")
            save_paired_plot(group, "mean_boundary_count", args.out_dir / "figures" / f"paired_boundary_count_{tag}.png")

    if not args.no_plots and not empirical_delta.empty and not recon_delta.empty:
        merged = empirical_delta.merge(recon_delta, on=["drug", "comparison", "subid", "hemisphere"], suffixes=("_empirical", "_recon"))
        for metric in ["delta_region_count", "delta_boundary_count"]:
            for (drug, hemi), group in merged.groupby(["drug", "hemisphere"]):
                save_scatter(
                    group,
                    f"{metric}_empirical",
                    f"{metric}_recon",
                    args.out_dir / "figures" / f"empirical_recon_delta_scatter_{drug}_{hemi}_{metric}.png",
                    f"{drug} {hemi} {metric}",
                )

    metadata = {
        "parameters": params.__dict__,
        "network_boundary_tolerance": args.network_boundary_tolerance,
        "network_overlap_mode": args.network_overlap_mode,
        "network_dfr_occupancy_threshold": args.network_dfr_occupancy_threshold,
        "cache_dir": str(args.cache_dir),
        "cache_enabled": not args.no_cache,
        "gradient_method": "matphase.detect.phase_field.compute_phase_gradient circular central differences",
        "vector_convention": "normalize_vector_field returns -grad / |grad|; DFR boundary uses unnormalized circular gradient magnitude.",
        "old_interaction_outputs_used": False,
        "n_discovered_entries": int(len(entries)),
        "n_processed_entries": int(len(per_subject)),
        "n_spiral_overlap_entries": int(len(spiral_overlap_records)),
        "n_failures": int(len(failures)),
        "inputs": [
            {
                "source": str(row.source),
                "drug": str(row.drug),
                "condition": str(row.condition),
                "subid": str(row.subid),
                "hemisphere": str(row.hemisphere),
                "phase_cube": str(row.phase_cube),
                "grid_path": str(row.grid_path),
                "parcel_meta_path": str(row.parcel_meta_path),
            }
            for row in entries.itertuples(index=False)
        ],
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 1 if failures and per_subject.empty else 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        raise SystemExit(main())
