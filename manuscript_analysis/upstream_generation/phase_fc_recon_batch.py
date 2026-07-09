#!/usr/bin/env python
"""
Batch phase-FC over reconstructed phase cubes.

The brainmodel reconstruction pipeline saves one .npy file per bundle, for
example:

    phase_cube_recon_DMT_DMT_S01_Atlas_s0_dtseries_nii_DMTDMT01L.npy

This wrapper adapts those files to the output layout produced by
scripts/phase_fc_batch.py: one output directory per bundle name, plus atlas
metadata/QC after the batch.

Example:
    python scripts/phase_fc_recon_batch.py ^
        --recon-root brainmodel/results/phase_recon/DMT ^
        --dlabel <derived_data>/atlases/Schaefer2018_400Parcels_17Networks_order.dlabel.nii ^
        --surface-left testdata/L.flat.32k_fs_LR.surf.gii ^
        --surface-right testdata/R.flat.32k_fs_LR.surf.gii ^
        --config configs/defaults.yaml ^
        --out-root analysis_outputs/phase_fc_recon_17networks ^
        --compute-phase-corr
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


RECON_PREFIX = "phase_cube_recon_"
HEMI_RE = re.compile(r"(?P<hemi>[LR])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch phase-FC over brainmodel reconstructed phase cubes."
    )
    parser.add_argument(
        "--recon-root",
        action="append",
        required=True,
        type=Path,
        help="Directory to recursively search for reconstructed .npy phase cubes. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pattern",
        default=f"{RECON_PREFIX}*.npy",
        help="Glob pattern used under each --recon-root.",
    )
    parser.add_argument(
        "--strip-prefix",
        default=RECON_PREFIX,
        help="Filename stem prefix to remove when forming the output bundle directory name.",
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
        help="CIFTI dlabel atlas path.",
    )
    parser.add_argument("--surface-left", required=True, type=Path)
    parser.add_argument("--surface-right", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/defaults.yaml"), type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--min-pixels", default=3, type=int)
    parser.add_argument("--min-valid-fraction", default=0.95, type=float)
    parser.add_argument("--limit", default=None, type=int, help="Optional max number of recon cubes to process.")
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
    parser.add_argument(
        "--no-parcellation-comparison",
        action="store_true",
        help="Pass through to the single-subject script to skip comparison plotting.",
    )
    return parser.parse_args()


def collect_recon_cubes(roots: list[Path], pattern: str) -> list[Path]:
    cubes: list[Path] = []
    for root in roots:
        cubes.extend(sorted(path for path in root.rglob(pattern) if path.is_file()))
    return sorted(dict.fromkeys(cubes))


def bundle_name_from_recon(path: Path, strip_prefix: str) -> str:
    stem = path.stem
    if strip_prefix and stem.startswith(strip_prefix):
        stem = stem[len(strip_prefix) :]
    return stem


def infer_hemisphere(bundle_name: str) -> str:
    match = HEMI_RE.search(bundle_name.upper())
    if not match:
        raise ValueError(f"Could not infer hemisphere from bundle name: {bundle_name}")
    return "left" if match.group("hemi") == "L" else "right"


def expected_outputs(out_dir: Path, compute_phase_corr: bool) -> list[Path]:
    outputs = [out_dir / "network_within_between_plv.csv"]
    if compute_phase_corr:
        outputs.extend(
            [
                out_dir / "network_within_between_phase_corr.csv",
                out_dir / "parcel_phase_corr.npy",
            ]
        )
    return outputs


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "phase_fc_single_subject.py"
    manifest_path = args.out_root / "phase_fc_batch_manifest.csv"

    cubes = collect_recon_cubes(args.recon_root, args.pattern)
    if args.limit is not None:
        cubes = cubes[: args.limit]
    if not cubes:
        raise RuntimeError(f"No reconstructed phase cubes found for pattern {args.pattern!r}.")

    rows = []
    failures = 0
    for idx, cube in enumerate(cubes, start=1):
        bundle_name = bundle_name_from_recon(cube, args.strip_prefix)
        try:
            hemisphere = infer_hemisphere(bundle_name)
            surface = args.surface_left if hemisphere == "left" else args.surface_right
            out_dir = args.out_root / bundle_name
            if all(path.exists() for path in expected_outputs(out_dir, args.compute_phase_corr)) and not args.overwrite:
                status = "skipped_existing"
                print(f"[{idx}/{len(cubes)}] skip {bundle_name}")
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
                if args.no_parcellation_comparison:
                    cmd.append("--no-parcellation-comparison")
                print(f"[{idx}/{len(cubes)}] run {bundle_name} ({hemisphere})")
                subprocess.run(cmd, check=True)
                status = "ok"
            rows.append(
                {
                    "bundle_dir": str(cube.parent),
                    "phase_cube": str(cube),
                    "hemisphere": hemisphere,
                    "out_dir": str(out_dir),
                    "status": status,
                }
            )
        except Exception as exc:
            failures += 1
            print(f"[{idx}/{len(cubes)}] failed {cube}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "bundle_dir": str(cube.parent),
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
