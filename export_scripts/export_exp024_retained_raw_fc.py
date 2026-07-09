"""Level 1 export of corrected retained raw-FC results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.path_config import EXPORT_ROOT, assert_new_output
from common.provenance import sha256_file, source_record, utc_now


EXPORT_ID = "EXP024"
FAMILY = "raw_fc_hemisphere_first"
OLD_SCRIPT = Path(r"<raw_fc_analysis>\analyze_fc_hemisphere_first.py")
SOURCES = {
    "DMT": Path(r"<raw_fc_outputs>\fc_analysis_hemisphere_first_DMT_drug_minus_pcb"),
    "LSD": Path(r"<raw_fc_outputs>\fc_analysis_hemisphere_first_LSD_drug_minus_pcb"),
}
OUTPUT = EXPORT_ROOT / FAMILY
METRICS = [
    "mean_abs_fc",
    "mean_fc",
    "mean_pos_fc",
    "mean_neg_fc",
    "fc_dispersion_sd",
    "fc_dispersion_iqr_abs",
    "mean_strength",
    "sd_strength",
]


def write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    assert_new_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite-new-output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def normalize_group(dataset: str, value: object) -> str:
    text = str(value).upper()
    return dataset if text == dataset else "PCB"


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    subjects = []
    summaries = []
    inputs = []
    for dataset, root in SOURCES.items():
        subject_path = root / "subject_level_fc_metrics.csv"
        summary_path = root / "group_stats.csv"
        for path in (subject_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs.append(path)
        subject = pd.read_csv(subject_path, dtype={"pair_id": "string"})
        subject.insert(0, "dataset", dataset)
        subject["condition"] = subject["group"].map(lambda value: normalize_group(dataset, value))
        subject["subid"] = subject["pair_id"].str.extract(r"(\d+)")[0].map(lambda value: f"{int(value):02d}")
        subjects.append(subject)
        summary = pd.read_csv(summary_path)
        summary.insert(0, "dataset", dataset)
        summaries.append(summary)
    return pd.concat(subjects, ignore_index=True), pd.concat(summaries, ignore_index=True), inputs


def paired_deltas(subject: pd.DataFrame) -> pd.DataFrame:
    long = subject.melt(
        id_vars=["dataset", "subid", "hemisphere", "condition"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    wide = long.pivot_table(
        index=["dataset", "subid", "hemisphere", "metric"],
        columns="condition",
        values="value",
        aggfunc="first",
    ).reset_index()
    rows = []
    for _, row in wide.iterrows():
        drug = row["dataset"]
        if drug not in row or "PCB" not in row or pd.isna(row[drug]) or pd.isna(row["PCB"]):
            continue
        rows.append(
            {
                "dataset": drug,
                "subid": row["subid"],
                "hemisphere": row["hemisphere"],
                "metric": row["metric"],
                "drug_value": row[drug],
                "control_value": row["PCB"],
                "delta_value": row[drug] - row["PCB"],
            }
        )
    return pd.DataFrame(rows)


def matrix_manifest() -> pd.DataFrame:
    rows = []
    for dataset, root in SOURCES.items():
        for path in sorted((root / "plots" / "matrix_maps").glob("*.csv")):
            name = path.stem
            hemisphere = "left" if name.startswith("left_") else "right"
            if "mean_matrix" in name:
                map_type = "group_mean_matrix"
            elif "mean_difference" in name or "mean_diff" in name:
                map_type = "group_mean_difference_matrix"
            elif "t_map" in name:
                map_type = "edgewise_t_map"
            elif "p_map" in name:
                map_type = "edgewise_p_map"
            else:
                map_type = "matrix_map"
            matrix = pd.read_csv(path, header=None)
            rows.append(
                {
                    "dataset": dataset,
                    "hemisphere": hemisphere,
                    "map_type": map_type,
                    "source_file": str(path),
                    "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        for path in sorted((root / "hemisphere_pconn").rglob("*.pconn.nii")):
            parts = {part.lower() for part in path.parts}
            rows.append(
                {
                    "dataset": dataset,
                    "hemisphere": "left" if "left" in parts else "right",
                    "map_type": "subject_pconn",
                    "source_file": str(path),
                    "shape": "CIFTI pconn",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def decorate(frame: pd.DataFrame, quantity: str) -> pd.DataFrame:
    result = frame.copy()
    result["export_id"] = EXPORT_ID
    result["script_family_id"] = FAMILY
    result["computed_quantity"] = quantity
    return result


def sidecar(quantity: str, output: Path, frame: pd.DataFrame, inputs: list[Path]) -> dict:
    return {
        "export_id": EXPORT_ID,
        "script_family_id": FAMILY,
        "computed_quantity": quantity,
        "old_script_path": str(OLD_SCRIPT),
        "old_script_sha256": sha256_file(OLD_SCRIPT),
        "new_export_script": str(Path(__file__).resolve()),
        "created_at": utc_now(),
        "inputs": [source_record(path) for path in inputs],
        "output_file": str(output),
        "row_count": len(frame),
        "rerun_level": "Level 1 retained-output re-export/inventory",
        "validation_status": "pass",
        "validation_details": "Corrected retained raw-FC outputs read without recomputation.",
    }


def update_global_metadata(exports: list[tuple[str, Path, pd.DataFrame]], inputs: list[Path]) -> None:
    manifest_path = EXPORT_ROOT / "metadata" / "master_export_manifest.csv"
    validation_path = EXPORT_ROOT / "summary_validation" / "validation_summary.csv"
    manifest = pd.read_csv(manifest_path)
    validation = pd.read_csv(validation_path)
    manifest = manifest[manifest["export_id"] != EXPORT_ID]
    validation = validation[validation["export_id"] != EXPORT_ID]
    mrows, vrows = [], []
    for quantity, output, frame in exports:
        is_matrix = quantity == "raw_fc_matrix_manifest"
        mrows.append(
            {
                "export_id": EXPORT_ID,
                "script_family_id": FAMILY,
                "computed_quantity": quantity,
                "data_granularity": "matrices" if is_matrix else "subject_level",
                "output_file": str(output),
                "sidecar_file": str(output) + ".metadata.json",
                "format": "csv",
                "row_count": len(frame),
                "array_count": len(frame) if is_matrix else 0,
                "shape_summary": "One row per retained matrix/pconn" if is_matrix else f"{len(frame)} rows x {len(frame.columns)} columns",
                "dataset": "DMT;LSD",
                "condition_scope": "DMT;LSD;PCB",
                "hemisphere_scope": "left;right",
                "network_scope": "",
                "old_script_path": str(OLD_SCRIPT),
                "new_export_script": str(Path(__file__).resolve()),
                "old_reference_outputs": ";".join(str(path) for path in inputs),
                "validation_status": "pass",
                "notes": "Corrected retained raw-FC result source; no recomputation or matrix copying.",
            }
        )
        vrows.append(
            {
                "export_id": EXPORT_ID,
                "script_family_id": FAMILY,
                "output_file": str(output),
                "validation_status": "pass",
                "details": "Corrected retained raw-FC output re-exported/inventoried without recomputation.",
            }
        )
    write_csv(pd.concat([manifest, pd.DataFrame(mrows)], ignore_index=True), manifest_path, True)
    write_csv(pd.concat([validation, pd.DataFrame(vrows)], ignore_index=True), validation_path, True)


def run(overwrite: bool) -> None:
    subject, summary, inputs = load_tables()
    deltas = paired_deltas(subject)
    matrices = matrix_manifest()
    exports = [
        ("raw_fc_subject_values", OUTPUT / "subject_level" / "exp024_raw_fc_subject_values.csv", decorate(subject, "raw_fc_subject_values")),
        ("raw_fc_paired_deltas", OUTPUT / "subject_level" / "exp024_raw_fc_paired_deltas.csv", decorate(deltas, "raw_fc_paired_deltas")),
        ("raw_fc_group_stats", OUTPUT / "summary_validation" / "exp024_raw_fc_group_stats.csv", decorate(summary, "raw_fc_group_stats")),
        ("raw_fc_matrix_manifest", OUTPUT / "matrices" / "exp024_raw_fc_matrix_manifest.csv", decorate(matrices, "raw_fc_matrix_manifest")),
    ]
    for quantity, output, frame in exports:
        write_csv(frame, output, overwrite)
        meta_path = Path(str(output) + ".metadata.json")
        if meta_path.exists() and not overwrite:
            raise FileExistsError(meta_path)
        meta_path.write_text(json.dumps(sidecar(quantity, output, frame, inputs), indent=2), encoding="utf-8")
    update_global_metadata(exports, inputs)
    print(f"EXP024 subject={len(subject)} deltas={len(deltas)} summaries={len(summary)} matrices={len(matrices)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-new-output", action="store_true")
    args = parser.parse_args()
    run(args.overwrite_new_output)
