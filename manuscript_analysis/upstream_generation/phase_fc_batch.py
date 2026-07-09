#!/usr/bin/env python
"""
Batch wrapper for scripts/phase_fc_single_subject.py.

It recursively finds phase_cube.npy files under one or more bundle roots, infers
hemisphere from the bundle directory name, and writes one output directory per
subject/hemisphere bundle.

Example:
    python scripts/phase_fc_batch.py ^
        --bundle-root detect_results/DMT ^
        --bundle-root detect_results/LSD ^
        --dlabel <derived_data>/atlases/Schaefer2018_400Parcels_7Networks_order.dlabel.nii ^
        --surface-left testdata/L.flat.32k_fs_LR.surf.gii ^
        --surface-right testdata/R.flat.32k_fs_LR.surf.gii ^
        --config configs/defaults.yaml ^
        --out-root analysis_outputs/phase_fc_batch
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch phase-FC over MatPhase phase_cube.npy bundles.")
    parser.add_argument(
        "--bundle-root",
        action="append",
        required=True,
        type=Path,
        help="Root directory to recursively search for phase_cube.npy. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dlabel",
        default=Path(
            os.environ.get(
                "PSYCHEDELIC_SPIRAL_ATLAS_DLABEL",
                "data/atlases/Schaefer2018_400Parcels_7Networks_order.dlabel.nii",
            )
        ),
        type=Path,
    )
    parser.add_argument("--surface-left", required=True, type=Path)
    parser.add_argument("--surface-right", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/defaults.yaml"), type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--min-pixels", default=3, type=int)
    parser.add_argument("--min-valid-fraction", default=0.95, type=float)
    parser.add_argument("--limit", default=None, type=int, help="Optional max number of cubes to process.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run bundles with existing output CSV.")
    parser.add_argument(
        "--save-parcel-phase",
        action="store_true",
        help="Pass through to the single-subject script.",
    )
    parser.add_argument(
        "--compute-phase-corr",
        action="store_true",
        help="Also compute Pearson correlation on parcel phase time series.",
    )
    parser.add_argument(
        "--skip-atlas-metadata",
        action="store_true",
        help="Do not build shared atlas metadata/QC after the batch finishes.",
    )
    return parser.parse_args()


def infer_hemisphere(bundle_dir: Path) -> str:
    name = bundle_dir.name
    upper = name.upper()
    if upper.endswith("L") or "_L_" in upper or "LEFT" in upper:
        return "left"
    if upper.endswith("R") or "_R_" in upper or "RIGHT" in upper:
        return "right"
    raise ValueError(f"Could not infer hemisphere from bundle name: {bundle_dir}")


def collect_phase_cubes(roots: list[Path]) -> list[Path]:
    cubes: list[Path] = []
    for root in roots:
        cubes.extend(sorted(root.rglob("phase_cube.npy")))
    return sorted(dict.fromkeys(cubes))


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "phase_fc_single_subject.py"
    manifest_path = args.out_root / "phase_fc_batch_manifest.csv"

    cubes = collect_phase_cubes(args.bundle_root)
    if args.limit is not None:
        cubes = cubes[: args.limit]
    if not cubes:
        raise RuntimeError("No phase_cube.npy files found.")

    rows = []
    failures = 0
    for idx, cube in enumerate(cubes, start=1):
        bundle_dir = cube.parent
        try:
            hemisphere = infer_hemisphere(bundle_dir)
            surface = args.surface_left if hemisphere == "left" else args.surface_right
            out_dir = args.out_root / bundle_dir.name
            required_outputs = [out_dir / "network_within_between_plv.csv"]
            if args.compute_phase_corr:
                required_outputs.append(out_dir / "network_within_between_phase_corr.csv")
                required_outputs.append(out_dir / "parcel_phase_corr.npy")
            if all(path.exists() for path in required_outputs) and not args.overwrite:
                status = "skipped_existing"
                print(f"[{idx}/{len(cubes)}] skip {bundle_dir.name}")
            else:
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--phase-cube",
                    str(cube),
                    "--hemisphere",
                    hemisphere,
                    "--dlabel",
                    str(args.dlabel),
                    "--surface",
                    str(surface),
                    "--config",
                    str(args.config),
                    "--out-dir",
                    str(out_dir),
                    "--min-pixels",
                    str(args.min_pixels),
                    "--min-valid-fraction",
                    str(args.min_valid_fraction),
                ]
                if args.save_parcel_phase:
                    cmd.append("--save-parcel-phase")
                if args.compute_phase_corr:
                    cmd.append("--compute-phase-corr")
                print(f"[{idx}/{len(cubes)}] run {bundle_dir.name} ({hemisphere})")
                subprocess.run(cmd, check=True)
                status = "ok"
            rows.append(
                {
                    "bundle_dir": str(bundle_dir),
                    "phase_cube": str(cube),
                    "hemisphere": hemisphere,
                    "out_dir": str(out_dir),
                    "status": status,
                }
            )
        except Exception as exc:
            failures += 1
            print(f"[{idx}/{len(cubes)}] failed {bundle_dir}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "bundle_dir": str(bundle_dir),
                    "phase_cube": str(cube),
                    "hemisphere": "",
                    "out_dir": "",
                    "status": f"failed: {exc}",
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_dir", "phase_cube", "hemisphere", "out_dir", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest: {manifest_path}")
    if not args.skip_atlas_metadata:
        metadata_script = Path(__file__).resolve().parent / "phase_fc_build_atlas_metadata.py"
        metadata_cmd = [
            sys.executable,
            str(metadata_script),
            "--phase-fc-root",
            str(args.out_root),
            "--out-dir",
            str(args.out_root / "atlas_metadata"),
        ]
        print("Building shared atlas metadata/QC...")
        subprocess.run(metadata_cmd, check=False)
    if failures:
        print(f"Finished with {failures} failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
