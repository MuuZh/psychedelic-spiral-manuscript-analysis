"""Export subject-balanced CAI signed-angle distributions for polar figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.path_config import EXPORT_ROOT, assert_new_output
from common.provenance import utc_now


EXPORT_ID = "EXP017"
FAMILY = "cai_polar"
TOOLBOX_ROOT = Path(r"<toolbox_root>")
SOURCES = {
    "DMT": TOOLBOX_ROOT / "analysis_outputs" / "phase_gradient_alignment_dmt",
    "LSD": TOOLBOX_ROOT / "analysis_outputs" / "phase_gradient_alignment_lsd",
}
OUTPUT_DIR = EXPORT_ROOT / FAMILY / "angle_distributions"
SUMMARY_PATH = OUTPUT_DIR / "exp017_subject_circular_summaries.csv"
NPZ_PATH = OUTPUT_DIR / "exp017_subject_angle_histograms.npz"
PAIRED_STATS_PATH = OUTPUT_DIR / "exp017_paired_bin_statistics.csv"
N_BINS = 72


def output_guard(path: Path, overwrite: bool) -> None:
    assert_new_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite-new-output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_source_rows() -> pd.DataFrame:
    frames = []
    for dataset, root in SOURCES.items():
        metrics_path = root / "subject_metrics.csv"
        frame = pd.read_csv(metrics_path, dtype={"subid": "string"})
        frame.insert(0, "dataset", dataset)
        frame["source_root"] = str(root)
        frame["source_file"] = frame["angle_cube"].map(lambda value: str(TOOLBOX_ROOT / value))
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result[result["hemisphere"].isin(["left", "right"])].copy()
    result["condition"] = result["role"].map({"PCB": "PCB", "Drug": "Drug"})
    result["subid"] = result["subid"].map(lambda value: f"{int(value):02d}")
    return result.sort_values(["dataset", "hemisphere", "subid", "condition"]).reset_index(drop=True)


def summarize_angles(source_rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    summaries = []
    densities = []
    for index, row in source_rows.iterrows():
        source = Path(row["source_file"])
        if not source.is_file():
            raise FileNotFoundError(source)
        cube = np.load(source, mmap_mode="r")
        angles = np.asarray(cube[np.isfinite(cube)], dtype=np.float64)
        counts, _ = np.histogram(angles, bins=edges)
        density = counts / counts.sum()
        mean_cos = float(np.mean(np.cos(angles)))
        mean_sin = float(np.mean(np.sin(angles)))
        resultant_length = float(np.hypot(mean_cos, mean_sin))
        summaries.append(
            {
                "subject_index": index,
                "dataset": row["dataset"],
                "condition": row["condition"],
                "condition_code": row["condition_code"],
                "subid": row["subid"],
                "hemisphere": row["hemisphere"],
                "n_angles": int(angles.size),
                "mean_angle": float(np.arctan2(mean_sin, mean_cos)),
                "resultant_length": resultant_length,
                "circular_variance": 1.0 - resultant_length,
                "mean_cos": mean_cos,
                "mean_sin": mean_sin,
                "mean_cos2": float(np.mean(np.cos(2.0 * angles))),
                "weighted_mean_cos2": float(row["weighted_mean_cos2_alignment"]),
                "binary_array_key": f"subject_density[{index}]",
                "source_file": str(source),
                "source_shape": "x".join(str(value) for value in cube.shape),
                "source_size_bytes": source.stat().st_size,
                "export_id": EXPORT_ID,
                "script_family_id": FAMILY,
            }
        )
        densities.append(density.astype(np.float32))
        print(f"[{index + 1}/{len(source_rows)}] {source.name}: n={angles.size}")
    return pd.DataFrame(summaries), np.asarray(densities), edges


def paired_bin_statistics(
    summary: pd.DataFrame, densities: np.ndarray, edges: np.ndarray
) -> pd.DataFrame:
    centers = (edges[:-1] + edges[1:]) / 2.0
    rows = []
    for dataset in ("DMT", "LSD"):
        for hemisphere in ("left", "right"):
            panel = summary[
                (summary["dataset"] == dataset) & (summary["hemisphere"] == hemisphere)
            ]
            paired = panel.pivot(index="subid", columns="condition", values="subject_index").dropna()
            delta = densities[paired["Drug"].astype(int)] - densities[paired["PCB"].astype(int)]
            if not np.allclose(delta.sum(axis=1), 0.0, atol=1e-7):
                raise ValueError(f"Paired histogram deltas do not sum to zero: {dataset} {hemisphere}")
            test = ttest_1samp(delta, popmean=0.0, axis=0)
            reject, q_values, _, _ = multipletests(test.pvalue, alpha=0.05, method="fdr_bh")
            mean = delta.mean(axis=0)
            sem = delta.std(axis=0, ddof=1) / np.sqrt(len(delta))
            cos2_delta = np.sum(delta * np.cos(2.0 * centers), axis=1)
            cos2_test = ttest_1samp(cos2_delta, popmean=0.0)
            for bin_index in range(N_BINS):
                rows.append(
                    {
                        "dataset": dataset,
                        "hemisphere": hemisphere,
                        "bin_index": bin_index,
                        "bin_left_rad": edges[bin_index],
                        "bin_right_rad": edges[bin_index + 1],
                        "bin_center_rad": centers[bin_index],
                        "n_pairs": len(delta),
                        "mean_delta_drug_minus_pcb": mean[bin_index],
                        "sem_delta_drug_minus_pcb": sem[bin_index],
                        "t_statistic": test.statistic[bin_index],
                        "p_value": test.pvalue[bin_index],
                        "q_value_bh_fdr": q_values[bin_index],
                        "fdr_significant_0p05": bool(reject[bin_index]),
                        "panel_mean_cos2_delta": float(cos2_delta.mean()),
                        "panel_mean_cos2_delta_sem": float(cos2_delta.std(ddof=1) / np.sqrt(len(delta))),
                        "panel_mean_cos2_delta_t": float(cos2_test.statistic),
                        "panel_mean_cos2_delta_p": float(cos2_test.pvalue),
                        "export_id": EXPORT_ID,
                        "script_family_id": FAMILY,
                    }
                )
    return pd.DataFrame(rows)


def update_metadata(summary: pd.DataFrame, paired_stats: pd.DataFrame) -> None:
    manifest_path = EXPORT_ROOT / "metadata" / "master_export_manifest.csv"
    validation_path = EXPORT_ROOT / "summary_validation" / "validation_summary.csv"
    manifest = pd.read_csv(manifest_path)
    validation = pd.read_csv(validation_path)
    manifest = manifest[manifest["export_id"] != EXPORT_ID]
    validation = validation[validation["export_id"] != EXPORT_ID]
    manifest_rows = [{
        "export_id": EXPORT_ID,
        "script_family_id": FAMILY,
        "computed_quantity": "cai_signed_angle_distribution",
        "data_granularity": "subject_level_histogram_and_circular_summary",
        "output_file": str(NPZ_PATH),
        "sidecar_file": str(NPZ_PATH) + ".metadata.json",
        "format": "npz;csv",
        "row_count": len(summary),
        "array_count": 2,
        "shape_summary": f"subject_density={len(summary)}x{N_BINS}; bin_edges={N_BINS + 1}",
        "dataset": "DMT;LSD",
        "condition_scope": "Drug;PCB",
        "hemisphere_scope": "left;right",
        "network_scope": "",
        "old_script_path": str(TOOLBOX_ROOT / "analysis" / "phase_gradient_alignment.py"),
        "new_export_script": str(Path(__file__).resolve()),
        "old_reference_outputs": ";".join(str(path / "subject_metrics.csv") for path in SOURCES.values()),
        "validation_status": "pass",
        "notes": "Full retained angle cubes reduced to equal-subject probability histograms; raw angle cubes remain referenced by path.",
    }, {
        "export_id": EXPORT_ID,
        "script_family_id": FAMILY,
        "computed_quantity": "cai_paired_bin_statistics",
        "data_granularity": "study_hemisphere_angle_bin",
        "output_file": str(PAIRED_STATS_PATH),
        "sidecar_file": str(PAIRED_STATS_PATH) + ".metadata.json",
        "format": "csv",
        "row_count": len(paired_stats),
        "array_count": 0,
        "shape_summary": f"4 panels x {N_BINS} angle bins",
        "dataset": "DMT;LSD",
        "condition_scope": "Drug-minus-PCB paired delta",
        "hemisphere_scope": "left;right",
        "network_scope": "",
        "old_script_path": str(TOOLBOX_ROOT / "analysis" / "phase_gradient_alignment.py"),
        "new_export_script": str(Path(__file__).resolve()),
        "old_reference_outputs": str(NPZ_PATH),
        "validation_status": "pass",
        "notes": "Paired one-sample t tests per raw 5-degree bin; BH-FDR applied independently within each study x hemisphere panel.",
    }]
    validation_rows = [{
        "export_id": EXPORT_ID,
        "script_family_id": FAMILY,
        "output_file": str(NPZ_PATH),
        "validation_status": "pass",
        "details": "Every retained subject-condition-hemisphere angle cube produced a normalized 72-bin distribution.",
    }, {
        "export_id": EXPORT_ID,
        "script_family_id": FAMILY,
        "output_file": str(PAIRED_STATS_PATH),
        "validation_status": "pass",
        "details": "Complete pairs only; every subject delta sums to zero; BH-FDR computed separately across 72 bins per panel.",
    }]
    pd.concat([manifest, pd.DataFrame(manifest_rows)], ignore_index=True).to_csv(manifest_path, index=False)
    pd.concat([validation, pd.DataFrame(validation_rows)], ignore_index=True).to_csv(validation_path, index=False)


def run(overwrite: bool) -> None:
    for path in (
        SUMMARY_PATH,
        NPZ_PATH,
        PAIRED_STATS_PATH,
        Path(str(SUMMARY_PATH) + ".metadata.json"),
        Path(str(NPZ_PATH) + ".metadata.json"),
        Path(str(PAIRED_STATS_PATH) + ".metadata.json"),
    ):
        output_guard(path, overwrite)
    source_rows = load_source_rows()
    summary, densities, edges = summarize_angles(source_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    np.savez_compressed(NPZ_PATH, bin_edges_rad=edges, subject_density=densities)
    paired_stats = paired_bin_statistics(summary, densities, edges)
    paired_stats.to_csv(PAIRED_STATS_PATH, index=False)
    metadata = {
        "export_id": EXPORT_ID,
        "script_family_id": FAMILY,
        "created_at": utc_now(),
        "source_analysis": str(TOOLBOX_ROOT / "analysis" / "phase_gradient_alignment.py"),
        "source_roots": {key: str(value) for key, value in SOURCES.items()},
        "summary_file": str(SUMMARY_PATH),
        "binary_file": str(NPZ_PATH),
        "paired_bin_statistics_file": str(PAIRED_STATS_PATH),
        "n_subject_distributions": len(summary),
        "n_bins": N_BINS,
        "angle_range_rad": [-float(np.pi), float(np.pi)],
        "normalization": "Each subject-condition-hemisphere histogram sums to one before group averaging.",
        "paired_statistics": "Paired one-sample t test per raw 5-degree bin; BH-FDR independently within each study x hemisphere panel.",
        "validation_status": "pass",
    }
    Path(str(SUMMARY_PATH) + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path(str(NPZ_PATH) + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path(str(PAIRED_STATS_PATH) + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    update_metadata(summary, paired_stats)
    print(f"EXP017 wrote {len(summary)} subject distributions and {len(paired_stats)} paired-bin rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-new-output", action="store_true")
    args = parser.parse_args()
    run(args.overwrite_new_output)
