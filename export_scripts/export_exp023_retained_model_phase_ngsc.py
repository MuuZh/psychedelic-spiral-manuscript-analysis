"""Level 1 export of retained empirical/model phase-NGSC outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.path_config import EXPORT_ROOT, OLD_ANALYSIS_OUTPUTS, assert_new_output
from common.provenance import sha256_file, source_record, utc_now


EXPORT_ID = "EXP023"
OLD_SCRIPT = Path(
    r"<source_analysis>\phase_fc_group\recon_parcel_fc_correlations.py"
)
SOURCE_ROOT = OLD_ANALYSIS_OUTPUTS / "phase_fc_recon_correlation_7networks_with_NGSC"
OUTPUT = EXPORT_ROOT / "phase_ngsc_model"
FILES = {
    "phase_ngsc_condition_values": "per_condition_subject_phase_ngsc.csv",
    "phase_ngsc_paired_deltas": "per_subject_phase_ngsc_drug_minus_pcb_delta.csv",
    "phase_ngsc_orig_recon_correlations": "phase_ngsc_orig_recon_correlations.csv",
    "phase_ngsc_delta_orig_recon_correlations": "phase_ngsc_drug_minus_pcb_delta_orig_recon_correlations.csv",
    "phase_ngsc_summary": "summary_phase_ngsc_by_source_condition_hemisphere.csv",
    "phase_ngsc_delta_summary": "summary_phase_ngsc_drug_minus_pcb_delta.csv",
}


def write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    assert_new_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite-new-output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_quantity(filename: str) -> tuple[pd.DataFrame, list[Path]]:
    frames = []
    inputs = []
    for dataset in ("DMT", "LSD"):
        path = SOURCE_ROOT / dataset / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"subid": "string"})
        frame.insert(0, "dataset", dataset)
        if "subid" in frame:
            frame["subid"] = frame["subid"].map(lambda value: f"{int(str(value)):02d}")
        frames.append(frame)
        inputs.append(path)
    return pd.concat(frames, ignore_index=True, sort=False), inputs


def sidecar(quantity: str, output: Path, frame: pd.DataFrame, inputs: list[Path]) -> dict:
    return {
        "export_id": EXPORT_ID,
        "script_family_id": "phase_ngsc_model",
        "computed_quantity": quantity,
        "old_script_path": str(OLD_SCRIPT),
        "old_script_sha256": sha256_file(OLD_SCRIPT),
        "new_export_script": str(Path(__file__).resolve()),
        "created_at": utc_now(),
        "inputs": [source_record(path) for path in inputs],
        "output_file": str(output),
        "row_count": len(frame),
        "rerun_level": "Level 1 retained-output re-export",
        "validation_status": "pass",
        "validation_details": "Retained phase-NGSC empirical/model outputs re-exported without recomputation.",
    }


def update_global_metadata(exports: list[tuple[str, Path, pd.DataFrame, list[Path]]]) -> None:
    manifest_path = EXPORT_ROOT / "metadata" / "master_export_manifest.csv"
    validation_path = EXPORT_ROOT / "summary_validation" / "validation_summary.csv"
    manifest = pd.read_csv(manifest_path)
    validation = pd.read_csv(validation_path)
    manifest = manifest[manifest["export_id"] != EXPORT_ID]
    validation = validation[validation["export_id"] != EXPORT_ID]
    manifest_rows = []
    validation_rows = []
    for quantity, output, frame, inputs in exports:
        is_summary = "summary" in quantity or "correlations" in quantity
        manifest_rows.append(
            {
                "export_id": EXPORT_ID,
                "script_family_id": "phase_ngsc_model",
                "computed_quantity": quantity,
                "data_granularity": "summary_validation" if is_summary else "model_correlation",
                "output_file": str(output),
                "sidecar_file": str(output) + ".metadata.json",
                "format": "csv",
                "row_count": len(frame),
                "array_count": 0,
                "shape_summary": f"{len(frame)} rows x {len(frame.columns)} columns",
                "dataset": "DMT;LSD",
                "condition_scope": "DMT_DMT;DMT_PCB;LSD_LSD;LSD_PCB",
                "hemisphere_scope": "left;right",
                "network_scope": "",
                "old_script_path": str(OLD_SCRIPT),
                "new_export_script": str(Path(__file__).resolve()),
                "old_reference_outputs": ";".join(str(path) for path in inputs),
                "validation_status": "pass",
                "notes": "Retained orig/recon phase-NGSC output; no recomputation.",
            }
        )
        validation_rows.append(
            {
                "export_id": EXPORT_ID,
                "script_family_id": "phase_ngsc_model",
                "output_file": str(output),
                "validation_status": "pass",
                "details": "Retained empirical/model phase-NGSC output re-exported without recomputation.",
            }
        )
    write_csv(pd.concat([manifest, pd.DataFrame(manifest_rows)], ignore_index=True), manifest_path, True)
    write_csv(pd.concat([validation, pd.DataFrame(validation_rows)], ignore_index=True), validation_path, True)


def run(overwrite: bool) -> None:
    exports = []
    for quantity, filename in FILES.items():
        frame, inputs = load_quantity(filename)
        frame["export_id"] = EXPORT_ID
        frame["script_family_id"] = "phase_ngsc_model"
        output = OUTPUT / f"exp023_{quantity}.csv"
        write_csv(frame, output, overwrite)
        metadata_path = Path(str(output) + ".metadata.json")
        if metadata_path.exists() and not overwrite:
            raise FileExistsError(metadata_path)
        metadata_path.write_text(
            json.dumps(sidecar(quantity, output, frame, inputs), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        exports.append((quantity, output, frame, inputs))
    update_global_metadata(exports)
    print("EXP023 " + " ".join(f"{quantity}={len(frame)}" for quantity, _, frame, _ in exports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-new-output", action="store_true")
    args = parser.parse_args()
    run(args.overwrite_new_output)
