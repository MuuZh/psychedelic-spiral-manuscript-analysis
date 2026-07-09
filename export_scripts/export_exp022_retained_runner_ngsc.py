"""Level 1 export of retained raw-NGSC and phase-NGSC runner outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.path_config import EXPORT_ROOT, OLD_ANALYSIS_OUTPUTS, assert_new_output
from common.provenance import sha256_file, source_record, utc_now


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = EXPORT_ROOT / "all_metrics_runner" / "ngsc"
OLD_SCRIPT = Path(r"<source_analysis>\all_metrics\ngsc.py")
SOURCES = {
    "DMT": OLD_ANALYSIS_OUTPUTS / "all_metrics" / "dmt-run" / "ngsc",
    "LSD": OLD_ANALYSIS_OUTPUTS / "all_metrics" / "lsd-run" / "ngsc",
}
EXPORT_ID = "EXP022"


def write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    assert_new_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite-new-output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(data: dict, path: Path, overwrite: bool) -> None:
    assert_new_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite-new-output: {path}")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def load_retained() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    subject_frames = []
    summary_frames = []
    inputs = []
    for dataset, directory in SOURCES.items():
        subject_path = directory / "per_subject.csv"
        summary_path = directory / "group_summary.csv"
        for path in (subject_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs.append(path)
        subject = pd.read_csv(subject_path, dtype={"subid": "string"})
        subject.insert(0, "dataset", dataset)
        subject["subid"] = subject["subid"].map(lambda value: f"{int(str(value)):02d}")
        subject = subject.rename(columns={"group": "condition"})
        subject_frames.append(subject)
        summary = pd.read_csv(summary_path)
        summary.insert(0, "dataset", dataset)
        summary_frames.append(summary)
    return (
        pd.concat(subject_frames, ignore_index=True, sort=False),
        pd.concat(summary_frames, ignore_index=True, sort=False),
        inputs,
    )


def paired_deltas(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, hemisphere), group in subject.groupby(["dataset", "hemisphere"], sort=True):
        drug = dataset
        wide = group.pivot(index="subid", columns="condition", values=["ngsc", "phase_ngsc"])
        required = [
            ("ngsc", drug),
            ("ngsc", "PCB"),
            ("phase_ngsc", drug),
            ("phase_ngsc", "PCB"),
        ]
        complete = wide.dropna(subset=required)
        for subid, values in complete.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "subid": subid,
                    "hemisphere": hemisphere,
                    "drug_condition": drug,
                    "control_condition": "PCB",
                    "drug_ngsc": values[("ngsc", drug)],
                    "control_ngsc": values[("ngsc", "PCB")],
                    "delta_ngsc": values[("ngsc", drug)] - values[("ngsc", "PCB")],
                    "drug_phase_ngsc": values[("phase_ngsc", drug)],
                    "control_phase_ngsc": values[("phase_ngsc", "PCB")],
                    "delta_phase_ngsc": values[("phase_ngsc", drug)]
                    - values[("phase_ngsc", "PCB")],
                }
            )
    return pd.DataFrame(rows)


def add_provenance(frame: pd.DataFrame, quantity: str) -> pd.DataFrame:
    result = frame.copy()
    result["export_id"] = EXPORT_ID
    result["script_family_id"] = "all_metrics_runner"
    result["computed_quantity"] = quantity
    result["old_script"] = str(OLD_SCRIPT)
    return result


def metadata(output: Path, inputs: list[Path], quantity: str, rows: int) -> dict:
    return {
        "export_id": EXPORT_ID,
        "script_family_id": "all_metrics_runner",
        "computed_quantity": quantity,
        "old_script_path": str(OLD_SCRIPT),
        "old_script_sha256": sha256_file(OLD_SCRIPT),
        "new_export_script": str(Path(__file__).resolve()),
        "created_at": utc_now(),
        "inputs": [source_record(path) for path in inputs],
        "output_file": str(output),
        "row_count": rows,
        "rerun_level": "Level 1 retained-output re-export",
        "validation_status": "pass",
        "validation_details": "Retained canonical runner NGSC tables read without recomputation.",
        "notes": "Raw NGSC is column ngsc; phase NGSC is column phase_ngsc.",
    }


def update_global_metadata(outputs: list[tuple[str, Path, pd.DataFrame]], inputs: list[Path]) -> None:
    manifest_path = EXPORT_ROOT / "metadata" / "master_export_manifest.csv"
    validation_path = EXPORT_ROOT / "summary_validation" / "validation_summary.csv"
    manifest = pd.read_csv(manifest_path)
    validation = pd.read_csv(validation_path)
    manifest = manifest[manifest["export_id"] != EXPORT_ID]
    validation = validation[validation["export_id"] != EXPORT_ID]
    manifest_rows = []
    validation_rows = []
    for quantity, output, frame in outputs:
        manifest_rows.append(
            {
                "export_id": EXPORT_ID,
                "script_family_id": "all_metrics_runner",
                "computed_quantity": quantity,
                "data_granularity": "summary_validation"
                if quantity == "ngsc_summary_validation"
                else "subject_level",
                "output_file": str(output),
                "sidecar_file": str(output) + ".metadata.json",
                "format": "csv",
                "row_count": len(frame),
                "array_count": 0,
                "shape_summary": f"{len(frame)} rows x {len(frame.columns)} columns",
                "dataset": "DMT;LSD",
                "condition_scope": "DMT;LSD;PCB",
                "hemisphere_scope": "left;right",
                "network_scope": "",
                "old_script_path": str(OLD_SCRIPT),
                "new_export_script": str(Path(__file__).resolve()),
                "old_reference_outputs": ";".join(str(path) for path in inputs),
                "validation_status": "pass",
                "notes": "Level 1 retained canonical runner NGSC export; no recomputation.",
            }
        )
        validation_rows.append(
            {
                "export_id": EXPORT_ID,
                "script_family_id": "all_metrics_runner",
                "output_file": str(output),
                "validation_status": "pass",
                "details": "Retained canonical runner NGSC values re-exported without recomputation.",
            }
        )
    write_csv(pd.concat([manifest, pd.DataFrame(manifest_rows)], ignore_index=True), manifest_path, True)
    write_csv(pd.concat([validation, pd.DataFrame(validation_rows)], ignore_index=True), validation_path, True)


def run(overwrite: bool) -> None:
    subject, summary, inputs = load_retained()
    deltas = paired_deltas(subject)
    tables = [
        ("ngsc_subject_values", OUTPUT / "exp022_ngsc_subject_values.csv", add_provenance(subject, "ngsc_subject_values")),
        ("ngsc_paired_deltas", OUTPUT / "exp022_ngsc_paired_deltas.csv", add_provenance(deltas, "ngsc_paired_deltas")),
        (
            "ngsc_summary_validation",
            OUTPUT / "exp022_ngsc_summary_validation.csv",
            add_provenance(summary, "ngsc_summary_validation"),
        ),
    ]
    for quantity, output, frame in tables:
        write_csv(frame, output, overwrite)
        write_json(metadata(output, inputs, quantity, len(frame)), Path(str(output) + ".metadata.json"), overwrite)
    update_global_metadata(tables, inputs)
    print(f"EXP022 subject_rows={len(subject)} paired_delta_rows={len(deltas)} summary_rows={len(summary)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-new-output", action="store_true")
    args = parser.parse_args()
    run(args.overwrite_new_output)
