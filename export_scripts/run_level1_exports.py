import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.io_utils import read_csv, write_json, write_table
from common.path_config import EXPORT_ROOT, LOG_ROOT, OLD_ANALYSIS_OUTPUTS, ensure_new_roots
from common.provenance import sha256_file, source_record, utc_now
from common.stats_utils import numeric_summary
from common.validation_utils import validate_written_table

OLD_SCRIPTS = {
    "all_metrics_runner": r"<source_analysis>\all_metrics\runner.py",
    "gipr": r"<source_analysis>\added_analysis\GIPR\gipr.py",
    "network_spiral_metrics": r"<source_analysis>\phase_fc_group\network_spiral_metrics.py",
    "path_entropy": r"<source_analysis>\path_entropy_paired.py",
    "pattern_msd_beta": r"<source_analysis>\pattern_msd_beta_group.py",
    "phase_ngsc_recon_corr": r"<source_analysis>\phase_fc_group\recon_parcel_fc_correlations.py",
    "phase_recon_wbfc": r"<source_analysis>\phase_fc_group\phase_recon_edge_weighted_wbfc_delta.py",
    "cai_model_delta": r"<source_analysis>\added_analysis\SVD-CAI-model-delta-corr\svd_cai_model_delta_corr.py",
    "dfr": r"<source_analysis>\dfr_analysis\run_dfr.py",
    "gcor": r"<source_analysis>\all_metrics\run_gcor_all_metrics.py",
}


def old_script_provenance(family: str) -> dict:
    path = Path(OLD_SCRIPTS[family])
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "hash_status": "available" if path.is_file() else "old_script_not_accessible",
    }


def source(relative: str) -> Path:
    return OLD_ANALYSIS_OUTPUTS / relative


TABLE_SPECS = [
    ("EXP001", "all_metrics_runner", "subject_level", "pattern_stats_subject", "csv", [
        ("DMT", source("all_metrics/dmt-run/pattern_stats/per_subject.csv")),
        ("LSD", source("all_metrics/lsd-run/pattern_stats/per_subject.csv")),
    ]),
    ("EXP002", "all_metrics_runner", "frame_or_pattern_level", "pattern_dynamics", "parquet", [
        ("DMT", source("all_metrics/dmt-run/pattern_dynamics/per_pattern.csv")),
        ("LSD", source("all_metrics/lsd-run/pattern_dynamics/per_pattern.csv")),
    ]),
    ("EXP002", "all_metrics_runner", "frame_or_pattern_level", "pattern_dynamics_msd", "parquet", [
        ("DMT", source("all_metrics/dmt-run/pattern_dynamics/per_pattern_msd.csv")),
        ("LSD", source("all_metrics/lsd-run/pattern_dynamics/per_pattern_msd.csv")),
    ]),
    ("EXP006", "gipr", "subject_level", "gipr_subject", "csv", [
        ("DMT", source("added_analysis/GIPR/dmt_vs_pcb_run/per_subject.csv")),
        ("LSD", source("added_analysis/GIPR/lsd_vs_pcb_run/per_subject.csv")),
    ]),
    ("EXP008", "network_spiral_metrics", "network_level", "subject_network_metrics_wide", "csv", [
        ("DMT", source("phase_fc_group/network_spiral_metrics/subject_network_metrics_wide.csv")),
        ("LSD", source("phase_fc_group/network_spiral_metrics_LSD/subject_network_metrics_wide.csv")),
    ]),
    ("EXP008", "network_spiral_metrics", "network_level", "subject_network_metrics_long", "csv", [
        ("DMT", source("phase_fc_group/network_spiral_metrics/subject_network_metrics_long.csv")),
        ("LSD", source("phase_fc_group/network_spiral_metrics_LSD/subject_network_metrics_long.csv")),
    ]),
    ("EXP008", "network_spiral_metrics", "network_level", "paired_deltas_long", "csv", [
        ("DMT", source("phase_fc_group/network_spiral_metrics/paired_deltas_long.csv")),
        ("LSD", source("phase_fc_group/network_spiral_metrics_LSD/paired_deltas_long.csv")),
    ]),
    ("EXP010", "path_entropy", "subject_level", "path_entropy_subject", "csv", [
        ("DMT", source("path_entropy/dmt_vs_pcb/per_subject_with_whole.csv")),
        ("LSD", source("path_entropy/lsd_vs_pcb/per_subject_with_whole.csv")),
    ]),
    ("EXP010", "path_entropy", "subject_level", "path_entropy_paired_deltas", "csv", [
        ("DMT", source("path_entropy/dmt_vs_pcb/paired_deltas.csv")),
        ("LSD", source("path_entropy/lsd_vs_pcb/paired_deltas.csv")),
    ]),
    ("EXP012", "pattern_msd_beta", "frame_or_pattern_level", "per_pattern_msd_beta", "parquet", [
        ("DMT", source("pattern_msd_beta/dmt/per_pattern_msd_beta.csv")),
        ("LSD", source("pattern_msd_beta/lsd/per_pattern_msd_beta.csv")),
    ]),
    ("EXP012", "pattern_msd_beta", "subject_level", "per_subject_msd_beta", "csv", [
        ("DMT", source("pattern_msd_beta/dmt_group/per_subject_msd_beta.csv")),
        ("LSD", source("pattern_msd_beta/lsd_group/per_subject_msd_beta.csv")),
    ]),
    ("EXP013", "phase_ngsc_recon_corr", "model_correlation", "parcel_fc_correlation_points", "csv", [
        ("all", source("phase_fc_group/recon_parcel_fc_correlations/per_condition_subject_parcel_fc_correlations.csv")),
    ]),
    ("EXP013", "phase_ngsc_recon_corr", "model_correlation", "true_fc_correlation_points", "csv", [
        ("all", source("phase_fc_group/recon_parcel_fc_correlations/per_condition_subject_true_fc_correlations.csv")),
    ]),
    ("EXP013", "phase_ngsc_recon_corr", "model_correlation", "mean_delta_parcel_fc_correlations", "csv", [
        ("all", source("phase_fc_group/recon_parcel_fc_correlations/mean_delta_parcel_fc_correlations.csv")),
    ]),
    ("EXP014", "phase_recon_wbfc", "model_correlation", "original_recon_joined_values_z", "csv", [
        ("all", source("phase_fc_group/phase_recon_edge_weighted_wbfc_delta_v3/original_recon_joined_values_z.csv")),
    ]),
    ("EXP014", "phase_recon_wbfc", "model_correlation", "subject_condition_values_z", "csv", [
        ("all", source("phase_fc_group/phase_recon_edge_weighted_wbfc_delta_v3/subject_condition_values_z.csv")),
    ]),
    ("EXP014", "phase_recon_wbfc", "model_correlation", "subject_drug_placebo_delta_values_z", "csv", [
        ("all", source("phase_fc_group/phase_recon_edge_weighted_wbfc_delta_v3/subject_drug_placebo_delta_values_z.csv")),
    ]),
    ("EXP016", "cai_model_delta", "model_correlation", "subject_condition_metrics", "csv", [
        ("all", source("added_analysis/SVD-CAI-model-delta-corr/subject_condition_metrics.csv")),
    ]),
    ("EXP016", "cai_model_delta", "model_correlation", "subject_condition_values_long", "csv", [
        ("all", source("added_analysis/SVD-CAI-model-delta-corr/subject_condition_values_long.csv")),
    ]),
    ("EXP016", "cai_model_delta", "model_correlation", "empirical_model_joined_values", "csv", [
        ("all", source("added_analysis/SVD-CAI-model-delta-corr/empirical_model_joined_values.csv")),
    ]),
    ("EXP018", "dfr", "model_correlation", "per_condition_subject_dfr", "csv", [
        ("all", source("dfr_analysis/per_condition_subject_dfr.csv")),
    ]),
    ("EXP018", "dfr", "model_correlation", "per_subject_dfr_drug_minus_pcb_delta", "csv", [
        ("all", source("dfr_analysis/per_subject_dfr_drug_minus_pcb_delta.csv")),
    ]),
    ("EXP018", "dfr", "model_correlation", "empirical_recon_dfr_delta_correlations", "csv", [
        ("all", source("dfr_analysis/empirical_recon_dfr_delta_correlations.csv")),
    ]),
    ("EXP018", "dfr", "model_correlation", "dfr_spiral_delta_correlations", "csv", [
        ("all", source("dfr_analysis/dfr_spiral_delta_correlations.csv")),
    ]),
    ("EXP018", "dfr", "model_correlation", "dfr_network_boundary_distance", "csv", [
        ("all", source("dfr_analysis/dfr_network_boundary_distance.csv")),
    ]),
    ("EXP018", "dfr", "model_correlation", "dfr_network_boundary_overlap", "csv", [
        ("all", source("dfr_analysis/dfr_network_boundary_overlap.csv")),
    ]),
    ("EXP018", "dfr", "model_correlation", "dfr_spiral_boundary_overlap", "csv", [
        ("all", source("dfr_analysis/dfr_spiral_boundary_overlap.csv")),
    ]),
    ("EXP020", "gcor", "subject_level", "gcor_subject", "csv", [
        ("DMT", source("all_metrics_gcor/gcor_group_stats_dmt/gcor/subject_metrics.csv")),
        ("LSD", source("all_metrics_gcor/gcor_group_stats_lsd/gcor/subject_metrics.csv")),
    ]),
]

SUMMARY_SPECS = [
    ("EXP001", "all_metrics_runner", "pattern_stats_summary", [("DMT", source("all_metrics/dmt-run/pattern_stats/group_summary.csv")), ("LSD", source("all_metrics/lsd-run/pattern_stats/group_summary.csv"))]),
    ("EXP006", "gipr", "gipr_summary", [("DMT", source("added_analysis/GIPR/dmt_vs_pcb_run/group_summary.csv")), ("LSD", source("added_analysis/GIPR/lsd_vs_pcb_run/group_summary.csv"))]),
    ("EXP008", "network_spiral_metrics", "network_paired_ttest_summary", [("DMT", source("phase_fc_group/network_spiral_metrics/paired_ttest_summary.csv")), ("LSD", source("phase_fc_group/network_spiral_metrics_LSD/paired_ttest_summary.csv"))]),
    ("EXP010", "path_entropy", "path_entropy_summary", [("DMT", source("path_entropy/dmt_vs_pcb/group_summary.csv")), ("LSD", source("path_entropy/lsd_vs_pcb/group_summary.csv"))]),
    ("EXP012", "pattern_msd_beta", "msd_beta_group_summary", [("DMT", source("pattern_msd_beta/dmt_group/msd_beta_group_summary.csv")), ("LSD", source("pattern_msd_beta/lsd_group/msd_beta_group_summary.csv"))]),
    ("EXP013", "phase_ngsc_recon_corr", "ngsc_fc_summary", [("all", source("phase_fc_group/recon_parcel_fc_correlations/summary_by_condition_hemisphere.csv")), ("all", source("phase_fc_group/recon_parcel_fc_correlations/summary_true_fc_by_source_condition_hemisphere.csv"))]),
    ("EXP014", "phase_recon_wbfc", "phase_recon_wbfc_summary", [("all", source("phase_fc_group/phase_recon_edge_weighted_wbfc_delta_v3/drug_placebo_delta_tests_z.csv")), ("all", source("phase_fc_group/phase_recon_edge_weighted_wbfc_delta_v3/original_recon_delta_agreement_stats_z.csv"))]),
    ("EXP016", "cai_model_delta", "cai_model_delta_summary", [("all", source("added_analysis/SVD-CAI-model-delta-corr/drug_placebo_delta_tests.csv")), ("all", source("added_analysis/SVD-CAI-model-delta-corr/empirical_model_delta_agreement_stats.csv"))]),
    ("EXP018", "dfr", "dfr_summary", [("all", source("dfr_analysis/paired_dfr_stats.csv"))]),
    ("EXP020", "gcor", "gcor_summary", [("all", source("all_metrics_gcor/combined_gcor_summary.csv"))]),
]


def add_metadata(frame: pd.DataFrame, export_id: str, family: str, datasets: list[str], paths: list[Path]) -> pd.DataFrame:
    result = frame.copy()
    result["export_id"] = export_id
    result["script_family_id"] = family
    result["source_dataset"] = pd.Series(datasets, index=result.index, dtype="string")
    result["source_file"] = pd.Series([str(path) for path in paths], index=result.index, dtype="string")
    return result


def load_sources(items: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[dict], list[str], list[Path]]:
    frames, records, datasets, row_paths = [], [], [], []
    for dataset, path in items:
        frame = read_csv(path)
        frames.append(frame)
        records.append(source_record(path))
        datasets.extend([dataset] * len(frame))
        row_paths.extend([path] * len(frame))
    return pd.concat(frames, ignore_index=True, sort=False), records, datasets, row_paths


def manifest_row(export_id, family, quantity, output, sidecar, frame, status, sources, notes=""):
    return {
        "export_id": export_id,
        "script_family_id": family,
        "computed_quantity": quantity,
        "data_granularity": output.parent.name,
        "output_file": str(output),
        "sidecar_file": str(sidecar),
        "format": output.suffix.lstrip("."),
        "row_count": len(frame),
        "array_count": 0,
        "shape_summary": f"{len(frame)} rows x {len(frame.columns)} columns",
        "dataset": ";".join(sorted(set(frame.get("source_dataset", pd.Series(["all"])).astype(str)))),
        "condition_scope": "preserved_from_source",
        "hemisphere_scope": "preserved_from_source",
        "network_scope": "preserved_from_source",
        "old_script_path": OLD_SCRIPTS[family],
        "new_export_script": str(Path(__file__).resolve()),
        "old_reference_outputs": ";".join(str(item["path"]) for item in sources),
        "validation_status": status,
        "notes": notes,
    }


def export_table(spec, overwrite, dry_run, manifest, validations):
    export_id, family, group, quantity, fmt, items = spec
    missing = [str(path) for _, path in items if not path.is_file()]
    if missing:
        validations.append({"export_id": export_id, "script_family_id": family, "output_file": "", "validation_status": "failed", "details": f"Missing sources: {missing}"})
        return
    raw, records, datasets, row_paths = load_sources(items)
    frame = add_metadata(raw, export_id, family, datasets, row_paths)
    output = EXPORT_ROOT / family / group / f"{export_id.lower()}_{quantity}.{fmt}"
    sidecar = output.with_suffix(output.suffix + ".metadata.json")
    if dry_run:
        logging.info("DRY RUN %s -> %s rows=%d", quantity, output, len(frame))
        return
    write_table(frame, output, overwrite)
    status, details = validate_written_table(raw, output)
    metadata = {
        "export_id": export_id,
        "script_family_id": family,
        "old_script_path": OLD_SCRIPTS[family],
        "old_script_provenance": old_script_provenance(family),
        "old_output_directories": sorted({str(path.parent) for _, path in items}),
        "new_export_script": str(Path(__file__).resolve()),
        "created_at": utc_now(),
        "dataset": sorted(set(datasets)),
        "inputs": records,
        "old_reference_outputs": [str(path) for _, path in items],
        "format": fmt,
        "row_unit_or_array_unit": group,
        "row_count": len(frame),
        "columns": list(frame.columns),
        "numeric_summary": numeric_summary(raw),
        "validation_status": status,
        "validation_details": details,
        "notes": "Original columns preserved; standardized provenance columns appended.",
    }
    write_json(metadata, sidecar, overwrite)
    manifest.append(manifest_row(export_id, family, quantity, output, sidecar, frame, status, records))
    validations.append({"export_id": export_id, "script_family_id": family, "output_file": str(output), "validation_status": status, "details": details})
    logging.info("%s %s rows=%d validation=%s", export_id, quantity, len(frame), status)


def export_dfr_npz_manifest(overwrite, dry_run, manifest, validations):
    export_id, family = "EXP019", "dfr"
    root = source("dfr_analysis")
    files = sorted(root.rglob("*.npz"))
    if not files:
        validations.append({"export_id": export_id, "script_family_id": family, "output_file": "", "validation_status": "failed", "details": "No DFR NPZ files found"})
        return
    if dry_run:
        logging.info("DRY RUN EXP019 would inventory %d NPZ files", len(files))
        return
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as arrays:
            keys = list(arrays.files)
            shapes = {key: list(arrays[key].shape) for key in keys}
            dtypes = {key: str(arrays[key].dtype) for key in keys}
        name = path.stem
        rows.append({
            "array_key": name,
            "source_npz": str(path),
            "relative_source_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "npz_keys": json.dumps(keys),
            "shape_by_key": json.dumps(shapes),
            "dtype_by_key": json.dumps(dtypes),
            "export_id": export_id,
            "script_family_id": family,
            "copy_status": "not_copied_manifest_only",
        })
    frame = pd.DataFrame(rows)
    output = EXPORT_ROOT / family / "spatial_maps" / "exp019_dfr_npz_manifest.csv"
    sidecar = output.with_suffix(output.suffix + ".metadata.json")
    write_table(frame, output, overwrite)
    metadata = {
        "export_id": export_id,
        "script_family_id": family,
        "old_script_path": OLD_SCRIPTS[family],
        "old_script_provenance": old_script_provenance(family),
        "old_output_directories": [str(root)],
        "new_export_script": str(Path(__file__).resolve()),
        "created_at": utc_now(),
        "inputs": [{"path": str(root), "npz_count": len(files), "total_size_bytes": int(frame["size_bytes"].sum())}],
        "format": "csv manifest referencing immutable old NPZ files",
        "row_unit_or_array_unit": "one row per NPZ map file",
        "row_count": len(frame),
        "array_count": len(frame),
        "validation_status": "pass",
        "validation_details": "Every discovered NPZ was opened read-only; keys, shapes, dtypes, size, and SHA-256 recorded.",
        "notes": "Maps were not copied because the 840 source NPZ files total about 601 MB.",
    }
    write_json(metadata, sidecar, overwrite)
    row = manifest_row(export_id, family, "DFR NPZ map inventory", output, sidecar, frame, "pass", metadata["inputs"], metadata["notes"])
    row["array_count"] = len(frame)
    row["shape_summary"] = "Per-file key shapes recorded in manifest"
    manifest.append(row)
    validations.append({"export_id": export_id, "script_family_id": family, "output_file": str(output), "validation_status": "pass", "details": metadata["validation_details"]})
    logging.info("EXP019 inventoried %d NPZ files (%d bytes)", len(frame), int(frame["size_bytes"].sum()))


def write_reports(manifest, validations, overwrite):
    manifest_frame = pd.DataFrame(manifest)
    validation_frame = pd.DataFrame(validations)
    write_table(manifest_frame, EXPORT_ROOT / "metadata" / "master_export_manifest.csv", overwrite)
    write_table(validation_frame, EXPORT_ROOT / "summary_validation" / "validation_summary.csv", overwrite)
    counts = validation_frame["validation_status"].value_counts().to_dict()
    lines = [
        "# Level 1 Validation Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "Validation compares every original source column, row, and value against the written export. "
        "Canonical summary tables are exported separately as validation targets.",
        "",
        f"- pass: {counts.get('pass', 0)}",
        f"- failed: {counts.get('failed', 0)}",
        f"- not_applicable: {counts.get('not_applicable', 0)}",
        "",
        "## Results",
        "",
    ]
    for row in validations:
        lines.append(f"- {row['export_id']} `{Path(row['output_file']).name if row['output_file'] else 'no output'}`: **{row['validation_status']}** - {row['details']}")
    report = EXPORT_ROOT / "summary_validation" / "validation_report.md"
    if report.exists() and not overwrite:
        raise FileExistsError(report)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run only Phase 2 Level 1 automatic exporters.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-new-output", action="store_true")
    args = parser.parse_args()
    ensure_new_roots()
    log_path = LOG_ROOT / "rerun_logs" / "level1_exporters.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()])
    manifest, validations = [], []
    for spec in TABLE_SPECS:
        export_table(spec, args.overwrite_new_output, args.dry_run, manifest, validations)
    for export_id, family, quantity, items in SUMMARY_SPECS:
        export_table((export_id, family, "summary_validation", quantity, "csv", items), args.overwrite_new_output, args.dry_run, manifest, validations)
    export_dfr_npz_manifest(args.overwrite_new_output, args.dry_run, manifest, validations)
    if not args.dry_run:
        write_reports(manifest, validations, args.overwrite_new_output)
    logging.info("Completed Level 1 run: manifest_entries=%d validations=%d", len(manifest), len(validations))


if __name__ == "__main__":
    main()
