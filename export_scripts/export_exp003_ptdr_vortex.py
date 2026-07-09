from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common.path_config import EXPORT_ROOT, LOG_ROOT, PROJECT_ROOT, assert_new_output


OLD_SCRIPT = Path(r"<source_analysis>\all_metrics\vortex.py")
DATASETS = {
    "DMT": {
        "drug": "DMT",
        "control": "PCB",
        "bundle_root": Path(r"<detect_results>\DMT"),
        "canonical_subject": Path(
            r"<analysis_outputs>\all_metrics\dmt-run\vortex_occupancy\per_subject.csv"
        ),
        "canonical_summary": Path(
            r"<analysis_outputs>\all_metrics\dmt-run\vortex_occupancy\group_summary.csv"
        ),
    },
    "LSD": {
        "drug": "LSD",
        "control": "PCB",
        "bundle_root": Path(r"<detect_results>\LSD"),
        "canonical_subject": Path(
            r"<analysis_outputs>\all_metrics\lsd-run\vortex_occupancy\per_subject.csv"
        ),
        "canonical_summary": Path(
            r"<analysis_outputs>\all_metrics\lsd-run\vortex_occupancy\group_summary.csv"
        ),
    },
}
METRICS = ["occupancy_mean", "occupancy_p95", "occupancy_p5", "occupancy_p95_p5_diff"]
OUTPUT_NAMES = {
    "subject": "exp003_ptdr_subject_values.csv",
    "deltas": "exp003_ptdr_paired_deltas.csv",
    "validation": "exp003_ptdr_summary_validation.csv",
    "maps": "exp003_vortex_occupancy_maps.npz",
    "manifest": "exp003_vortex_occupancy_map_manifest.csv",
}
ABS_TOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled EXP003 PTDR/vortex occupancy exporter.")
    parser.add_argument("--dataset", choices=["DMT", "LSD", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--overwrite-new-output", action="store_true")
    parser.add_argument("--output-root", type=Path, default=EXPORT_ROOT)
    return parser.parse_args()


def resolve_output_root(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = candidate.resolve()
    export_root = EXPORT_ROOT.resolve()
    if resolved != export_root and export_root not in resolved.parents:
        raise ValueError(f"Refusing output root outside project 02_exports: {resolved}")
    return resolved


def setup_logging(dry_run: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if not dry_run:
        log_path = LOG_ROOT / "rerun_logs" / "exp003_ptdr_vortex.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def infer_bundle_identity(bundle_dir: Path, dataset: str, meta: dict) -> tuple[str | None, str | None, str | None]:
    group = meta.get("group")
    hemisphere = meta.get("hemisphere")
    subid = meta.get("subject_id") or meta.get("subid") or meta.get("subject")
    match = re.search(r"_([A-Za-z]+)(\d+)([LR])$", bundle_dir.name)
    if match:
        raw_group, suffix_subid, hemi_token = match.groups()
        raw_group_lower = raw_group.lower()
        if group is None:
            if raw_group_lower.endswith(dataset.lower()):
                group = dataset
            elif raw_group_lower.endswith("pcb"):
                group = "PCB"
        subid = subid or suffix_subid
        hemisphere = hemisphere or ("left" if hemi_token.upper() == "L" else "right")
    if group is not None:
        group = str(group).upper()
    if hemisphere is not None:
        hemisphere = str(hemisphere).lower()
    return group, str(subid) if subid is not None else None, hemisphere


def inventory_bundles(dataset: str) -> pd.DataFrame:
    cfg = DATASETS[dataset]
    rows = []
    for bundle_dir in sorted(cfg["bundle_root"].iterdir()):
        if not bundle_dir.is_dir():
            continue
        required = {
            "metadata": bundle_dir / "metadata.json",
            "frame_index": bundle_dir / "frame_index.parquet",
            "coords": bundle_dir / "coords.feather",
        }
        if not all(path.is_file() for path in required.values()):
            continue
        try:
            meta = json.loads(required["metadata"].read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Skipping unreadable metadata %s: %s", required["metadata"], exc)
            continue
        group, subid, hemisphere = infer_bundle_identity(bundle_dir, dataset, meta)
        if group not in {cfg["drug"], cfg["control"]} or hemisphere not in {"left", "right"} or subid is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "condition": group,
                "subid": subid,
                "hemisphere": hemisphere,
                "bundle_dir": bundle_dir,
                **required,
                "old_cache": bundle_dir / "vortex_occupancy.npy",
            }
        )
    return pd.DataFrame(rows)


def select_smoke_subset(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    cfg = DATASETS[dataset]
    left = frame[frame["hemisphere"] == "left"]
    pivot = left.pivot_table(index="subid", columns="condition", values="bundle_dir", aggfunc="first")
    paired = pivot.dropna(subset=[cfg["drug"], cfg["control"]]).index.astype(str).tolist()
    keep = set(sorted(paired)[:2])
    return left[left["subid"].isin(keep)].copy()


def file_state(paths: list[Path]) -> dict[str, tuple[int, int]]:
    state = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            state[str(path)] = (stat.st_size, stat.st_mtime_ns)
        else:
            state[str(path)] = (-1, -1)
    return state


def build_occupancy(bundle: pd.Series) -> tuple[np.ndarray, int]:
    meta = json.loads(Path(bundle["metadata"]).read_text(encoding="utf-8"))
    height = int(meta.get("grid_height", 0))
    width = int(meta.get("grid_width", 0))
    frame_count_meta = int(meta.get("frame_count", 0))
    frame_index = pd.read_parquet(bundle["frame_index"], columns=["abs_time", "coord_start", "coord_end"])
    frame_count = frame_count_meta if frame_count_meta > 0 else int(frame_index["abs_time"].nunique())
    if height <= 0 or width <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid grid/frame metadata in {bundle['bundle_dir']}")
    coords = pd.read_feather(bundle["coords"], columns=["y", "x"]).to_numpy()
    occupancy = np.zeros((height, width), dtype=np.int64)
    for _, frame_rows in frame_index.groupby("abs_time", sort=True):
        slices = []
        for row in frame_rows.itertuples(index=False):
            start, end = int(row.coord_start), int(row.coord_end)
            if end > start:
                slices.append(coords[start:end])
        if not slices:
            continue
        points = slices[0] if len(slices) == 1 else np.concatenate(slices, axis=0)
        if points.size == 0:
            continue
        unique = np.unique(points, axis=0)
        ys = unique[:, 0].astype(int)
        xs = unique[:, 1].astype(int)
        valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        occupancy[ys[valid], xs[valid]] += 1
    return occupancy.astype(np.float64) / frame_count, frame_count


def scalar_record(bundle: pd.Series, occupancy: np.ndarray, frame_count: int) -> dict:
    nonzero = occupancy[np.isfinite(occupancy) & (occupancy != 0)]
    if nonzero.size:
        p95 = float(np.nanpercentile(nonzero, 95))
        p5 = float(np.nanpercentile(nonzero, 5))
        mean = float(nonzero.mean())
        ptdr = p95 - p5
    else:
        mean = p95 = p5 = ptdr = math.nan
    return {
        "dataset": bundle["dataset"],
        "condition": bundle["condition"],
        "subid": bundle["subid"],
        "hemisphere": bundle["hemisphere"],
        "occupancy_mean": mean,
        "occupancy_p95": p95,
        "occupancy_p5": p5,
        "occupancy_p95_p5_diff": ptdr,
        "ptdr": ptdr,
        "frame_count": frame_count,
        "nonzero_pixel_count": int(nonzero.size),
        "grid_height": int(occupancy.shape[0]),
        "grid_width": int(occupancy.shape[1]),
        "source_bundle": str(bundle["bundle_dir"]),
        "old_script": str(OLD_SCRIPT),
    }


def subject_array_key(row: pd.Series) -> str:
    return f"subject__{row['dataset']}__{row['condition']}__sub{row['subid']}__{row['hemisphere']}"


def compute_exports(bundles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    records = []
    arrays: dict[str, np.ndarray] = {}
    manifest_rows = []
    total = len(bundles)
    for number, (_, bundle) in enumerate(bundles.iterrows(), start=1):
        logging.info("Computing occupancy %d/%d: %s", number, total, bundle["bundle_dir"])
        occupancy, frame_count = build_occupancy(bundle)
        record = scalar_record(bundle, occupancy, frame_count)
        records.append(record)
        key = subject_array_key(bundle)
        arrays[key] = occupancy
        manifest_rows.append(manifest_record(key, bundle["dataset"], bundle["condition"], bundle["subid"], bundle["hemisphere"], "subject_occupancy_fraction", occupancy, bundle["bundle_dir"]))

    subject = pd.DataFrame(records).sort_values(["dataset", "condition", "subid", "hemisphere"]).reset_index(drop=True)
    deltas = build_paired_deltas(subject)
    for (dataset, condition, hemisphere), rows in subject.groupby(["dataset", "condition", "hemisphere"], sort=True):
        keys = [subject_array_key(row) for _, row in rows.iterrows()]
        shapes = {arrays[key].shape for key in keys}
        if len(shapes) != 1:
            logging.warning("Skipping group mean for inconsistent shapes: %s %s %s", dataset, condition, hemisphere)
            continue
        group_mean = np.mean(np.stack([arrays[key] for key in keys]), axis=0)
        key = f"group_mean__{dataset}__{condition}__{hemisphere}"
        arrays[key] = group_mean
        manifest_rows.append(manifest_record(key, dataset, condition, "", hemisphere, "group_mean_occupancy_fraction", group_mean, "computed_from_subject_maps"))
    for dataset in sorted(subject["dataset"].unique()):
        drug, control = DATASETS[dataset]["drug"], DATASETS[dataset]["control"]
        for hemisphere in ["left", "right"]:
            drug_key = f"group_mean__{dataset}__{drug}__{hemisphere}"
            control_key = f"group_mean__{dataset}__{control}__{hemisphere}"
            if drug_key in arrays and control_key in arrays and arrays[drug_key].shape == arrays[control_key].shape:
                delta = arrays[drug_key] - arrays[control_key]
                key = f"group_delta__{dataset}__{drug}_minus_{control}__{hemisphere}"
                arrays[key] = delta
                manifest_rows.append(manifest_record(key, dataset, f"{drug}_minus_{control}", "", hemisphere, "group_mean_drug_minus_pcb", delta, "computed_from_group_mean_maps"))
    return subject, deltas, arrays, pd.DataFrame(manifest_rows)


def manifest_record(key: str, dataset: str, condition: str, subid: str, hemisphere: str, map_type: str, array: np.ndarray, source: Path | str) -> dict:
    return {
        "array_key": key,
        "dataset": dataset,
        "condition": condition,
        "subid": subid,
        "hemisphere": hemisphere,
        "map_type": map_type,
        "shape": "x".join(map(str, array.shape)),
        "dtype": str(array.dtype),
        "unit": "fraction_of_frames",
        "binary_file": OUTPUT_NAMES["maps"],
        "source": str(source),
        "old_script": str(OLD_SCRIPT),
        "new_export_script": str(Path(__file__).resolve()),
    }


def build_paired_deltas(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, hemisphere), frame in subject.groupby(["dataset", "hemisphere"], sort=True):
        drug, control = DATASETS[dataset]["drug"], DATASETS[dataset]["control"]
        pivot = frame.pivot_table(index="subid", columns="condition", values=METRICS, aggfunc="mean")
        for subid, values in pivot.dropna().iterrows():
            row = {"dataset": dataset, "subid": subid, "hemisphere": hemisphere, "drug_condition": drug, "control_condition": control}
            for metric in METRICS:
                row[f"drug_{metric}"] = values[(metric, drug)]
                row[f"control_{metric}"] = values[(metric, control)]
                row[f"delta_{metric}"] = values[(metric, drug)] - values[(metric, control)]
            row["delta_ptdr"] = row["delta_occupancy_p95_p5_diff"]
            rows.append(row)
    return pd.DataFrame(rows)


def close_enough(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=0.0, atol=ABS_TOL, equal_nan=True))


def validate_subject_values(subject: pd.DataFrame, dataset: str, allow_subset: bool = False) -> tuple[str, str]:
    old = pd.read_csv(DATASETS[dataset]["canonical_subject"], dtype={"subid": str})
    old = old.rename(columns={"group": "condition"})
    new = subject[subject["dataset"] == dataset].copy()
    keys = ["condition", "subid", "hemisphere"]
    if allow_subset:
        old = old.merge(new[keys].drop_duplicates(), on=keys, how="inner")
    merged = new.merge(old, on=keys, suffixes=("_new", "_old"), how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        return "failed", "Canonical/new subject keys differ"
    mismatches = []
    for metric in METRICS:
        mask = ~np.isclose(merged[f"{metric}_new"], merged[f"{metric}_old"], rtol=0.0, atol=ABS_TOL, equal_nan=True)
        if mask.any():
            mismatches.append(f"{metric}:{int(mask.sum())}")
    return ("pass", "All canonical subject scalar values reproduced") if not mismatches else ("failed", "Subject scalar mismatches " + ", ".join(mismatches))


def paired_stats(subject: pd.DataFrame, dataset: str, hemisphere: str) -> dict:
    cfg = DATASETS[dataset]
    frame = subject[(subject["dataset"] == dataset) & (subject["hemisphere"] == hemisphere)]
    pivot = frame.pivot_table(index="subid", columns="condition", values="occupancy_p95_p5_diff", aggfunc="mean")
    pivot = pivot.dropna(subset=[cfg["drug"], cfg["control"]])
    drug = pivot[cfg["drug"]].to_numpy(float)
    control = pivot[cfg["control"]].to_numpy(float)
    delta = drug - control
    t_value, p_value = stats.ttest_rel(drug, control, nan_policy="omit")
    return {
        "n_pairs": int(len(pivot)),
        "drug_mean": float(np.mean(drug)),
        "drug_sd": float(np.std(drug, ddof=1)),
        "pcb_mean": float(np.mean(control)),
        "pcb_sd": float(np.std(control, ddof=1)),
        "paired_delta_mean": float(np.mean(delta)),
        "t": float(t_value),
        "p": float(p_value),
        "cohens_dz": float(np.mean(delta) / np.std(delta, ddof=1)),
    }


def old_validation_targets(dataset: str, hemisphere: str) -> dict:
    cfg = DATASETS[dataset]
    old = pd.read_csv(cfg["canonical_summary"])
    summary = old[
        (old["row_type"] == "summary")
        & (old["subset"] == "paired_only")
        & (old["metric"] == "occupancy_p95_p5_diff")
        & (old["hemisphere"] == hemisphere)
    ]
    test = old[
        (old["row_type"] == "test")
        & (old["subset"] == "paired_only")
        & (old["metric"] == "occupancy_p95_p5_diff")
        & (old["hemisphere"] == hemisphere)
        & (old["test_type"] == "paired_ttest")
    ].iloc[0]
    drug = summary[summary["group"] == cfg["drug"]].iloc[0]
    control = summary[summary["group"] == cfg["control"]].iloc[0]
    return {
        "old_n_pairs": int(test["n1"]),
        "old_drug_mean": float(drug["mean"]),
        "old_drug_sd": float(drug["std"]),
        "old_pcb_mean": float(control["mean"]),
        "old_pcb_sd": float(control["std"]),
        "old_t": float(test["t"]),
        "old_p": float(test["p"]),
        "old_cohens_dz": float(test["dz"]),
    }


def build_validation(subject: pd.DataFrame, datasets: list[str], smoke_test: bool, validate: bool) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        subject_status, subject_reason = ("not_applicable", "Validation not requested")
        if validate:
            subject_status, subject_reason = validate_subject_values(subject, dataset, allow_subset=smoke_test)
        if smoke_test:
            rows.append(
                {
                    "dataset": dataset,
                    "hemisphere": "left",
                    "metric": "occupancy_p95_p5_diff",
                    "validation_status": subject_status,
                    "reason": f"Smoke subject validation: {subject_reason}; group/test validation deferred",
                }
            )
            continue
        for hemisphere in ["left", "right"]:
            computed = paired_stats(subject, dataset, hemisphere)
            old = old_validation_targets(dataset, hemisphere)
            checks = {
                "n_pairs": close_enough(computed["n_pairs"], old["old_n_pairs"]),
                "drug_mean": close_enough(computed["drug_mean"], old["old_drug_mean"]),
                "drug_sd": close_enough(computed["drug_sd"], old["old_drug_sd"]),
                "pcb_mean": close_enough(computed["pcb_mean"], old["old_pcb_mean"]),
                "pcb_sd": close_enough(computed["pcb_sd"], old["old_pcb_sd"]),
                "t": close_enough(computed["t"], old["old_t"]),
                "p": close_enough(computed["p"], old["old_p"]),
                "cohens_dz": close_enough(computed["cohens_dz"], old["old_cohens_dz"]),
            }
            status = "not_applicable" if not validate else ("pass" if subject_status == "pass" and all(checks.values()) else "failed")
            reason = "Validation not requested" if not validate else f"{subject_reason}; canonical summary fields matched={sum(checks.values())}/{len(checks)}"
            rows.append(
                {
                    "dataset": dataset,
                    "hemisphere": hemisphere,
                    "metric": "occupancy_p95_p5_diff",
                    **computed,
                    **old,
                    "validation_status": status,
                    "reason": reason,
                    "canonical_subject_csv": str(DATASETS[dataset]["canonical_subject"]),
                    "canonical_summary_csv": str(DATASETS[dataset]["canonical_summary"]),
                }
            )
    return pd.DataFrame(rows)


def check_targets(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    targets = {name: assert_new_output(output_dir / filename) for name, filename in OUTPUT_NAMES.items()}
    sidecars = {name: assert_new_output(Path(str(path) + ".metadata.json")) for name, path in targets.items()}
    if not overwrite:
        existing = [path for path in [*targets.values(), *sidecars.values()] if path.exists()]
        if existing:
            raise FileExistsError("New outputs exist; pass --overwrite-new-output: " + "; ".join(map(str, existing)))
    return targets


def write_outputs(
    targets: dict[str, Path],
    subject: pd.DataFrame,
    deltas: pd.DataFrame,
    validation: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    metadata: dict,
) -> None:
    targets["subject"].parent.mkdir(parents=True, exist_ok=True)
    subject.to_csv(targets["subject"], index=False)
    deltas.to_csv(targets["deltas"], index=False)
    validation.to_csv(targets["validation"], index=False)
    manifest.to_csv(targets["manifest"], index=False)
    np.savez_compressed(targets["maps"], **arrays)
    for name, path in targets.items():
        sidecar = Path(str(path) + ".metadata.json")
        payload = {**metadata, "output_type": name, "output_file": str(path)}
        sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(args.dry_run)
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "all_metrics_runner" / "ptdr_vortex"
    if args.smoke_test:
        if len(datasets) != 1:
            datasets = [datasets[0]]
            logging.info("Smoke test bounded to dataset %s", datasets[0])
        output_dir = output_dir / "smoke_test"

    inventories = []
    for dataset in datasets:
        inventory = inventory_bundles(dataset)
        if args.smoke_test:
            inventory = select_smoke_subset(inventory, dataset)
        logging.info("Inventory %s: %d usable bundles", dataset, len(inventory))
        inventories.append(inventory)
    bundles = pd.concat(inventories, ignore_index=True)
    if bundles.empty:
        raise RuntimeError("No usable bundles found")

    target_paths = check_targets(output_dir, args.overwrite_new_output) if not args.dry_run else {}
    logging.info("Output directory: %s", output_dir)
    logging.info("Old cache policy: ignored; no old cache reads or writes")
    if args.dry_run:
        logging.info("Dry-run complete: would process %d bundles; no outputs written", len(bundles))
        return

    monitored = []
    for column in ["metadata", "frame_index", "coords", "old_cache"]:
        monitored.extend(Path(path) for path in bundles[column])
    before = file_state(monitored)
    subject, deltas, arrays, manifest = compute_exports(bundles)
    validation = build_validation(subject, datasets, args.smoke_test, args.validate)
    after = file_state(monitored)
    old_inputs_unchanged = before == after
    if not old_inputs_unchanged:
        raise RuntimeError("Old input or cache state changed during controlled export")
    if args.smoke_test and args.validate and not (validation["validation_status"] == "pass").all():
        raise RuntimeError("Smoke-test subject validation failed; refusing smoke outputs")

    metadata = {
        "export_id": "EXP003",
        "script_family_id": "all_metrics_runner",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "conditions": {dataset: [DATASETS[dataset]["drug"], DATASETS[dataset]["control"]] for dataset in datasets},
        "hemispheres": sorted(subject["hemisphere"].unique().tolist()),
        "subjects": {dataset: sorted(subject[subject["dataset"] == dataset]["subid"].unique().tolist()) for dataset in datasets},
        "old_script": str(OLD_SCRIPT),
        "new_export_script": str(Path(__file__).resolve()),
        "canonical_subject_csvs": [str(DATASETS[dataset]["canonical_subject"]) for dataset in datasets],
        "canonical_summary_csvs": [str(DATASETS[dataset]["canonical_summary"]) for dataset in datasets],
        "metric_definition": "PTDR = nonzero occupancy-fraction 95th percentile minus 5th percentile",
        "input_files": ["metadata.json", "frame_index.parquet", "coords.feather"],
        "old_cache_policy": "Existing vortex_occupancy.npy files ignored and unchanged",
        "old_inputs_unchanged": old_inputs_unchanged,
        "row_unit": "dataset x condition x subject x hemisphere",
        "array_unit": "subject/group occupancy fraction map",
        "smoke_test": args.smoke_test,
        "validation_requested": args.validate,
        "subject_row_count": int(len(subject)),
        "paired_delta_row_count": int(len(deltas)),
        "array_count": int(len(arrays)),
    }
    write_outputs(target_paths, subject, deltas, validation, arrays, manifest, metadata)
    logging.info(
        "Completed EXP003: subject_rows=%d paired_rows=%d arrays=%d validation=%s old_inputs_unchanged=%s",
        len(subject),
        len(deltas),
        len(arrays),
        validation["validation_status"].value_counts().to_dict(),
        old_inputs_unchanged,
    )


if __name__ == "__main__":
    main()
