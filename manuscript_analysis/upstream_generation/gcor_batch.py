#!/usr/bin/env python
"""
Batch wrapper for analysis/gcor_from_cifti.py.

It finds CIFTI time series files from one or more roots or metadata.json files,
runs the single-script GCOR computation for each CIFTI, and writes:

    <out-root>/gcor_batch_manifest.csv
    <out-root>/all_gcor_by_subject.csv
    <out-root>/<subject-stem>/gcor_by_subject.csv

Example:
    python scripts/gcor_batch.py ^
        --cifti-root <derived_data>/dtseries ^
        --out-root analysis_outputs/gcor_batch

Or from existing detect_results metadata:
    python scripts/gcor_batch.py ^
        --metadata-root detect_results ^
        --out-root analysis_outputs/gcor_batch
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_RE = re.compile(
    r"(?P<drug>[A-Za-z]+)_(?P<condition>[A-Za-z]+)_S(?P<id>\d+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch GCOR computation for CIFTI time series files.")
    parser.add_argument(
        "--cifti-root",
        action="append",
        default=None,
        type=Path,
        help="Root directory to recursively search for CIFTI files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--cifti",
        action="append",
        default=None,
        type=Path,
        help="Explicit CIFTI file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--metadata-root",
        action="append",
        default=None,
        type=Path,
        help="Root to scan for metadata.json files containing cifti_file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern for --cifti-root. Default: *.dtseries.nii and *.ptseries.nii.",
    )
    parser.add_argument("--out-root", required=True, type=Path, help="Batch output root.")
    parser.add_argument("--limit", default=None, type=int, help="Optional maximum number of CIFTI files to process.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute files with existing per-subject output.")
    return parser.parse_args()


def resolve_existing_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = RELEASE_ROOT / path
        if candidate.exists():
            return candidate
    return path


def scan_metadata(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"metadata root not found: {root}")

    paths: list[Path] = []
    for meta_path in sorted(root.rglob("metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cifti_file = meta.get("cifti_file")
        if cifti_file:
            paths.append(resolve_existing_path(cifti_file))
    return paths


def scan_cifti_roots(roots: list[Path], patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"CIFTI root not found: {root}")
        for pattern in patterns:
            paths.extend(sorted(root.rglob(pattern)))
    return paths


def discover_cifti_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.cifti:
        paths.extend(resolve_existing_path(path) for path in args.cifti)
    if args.cifti_root:
        patterns = args.pattern or ["*.dtseries.nii", "*.ptseries.nii"]
        paths.extend(scan_cifti_roots(args.cifti_root, patterns))
    if args.metadata_root:
        for root in args.metadata_root:
            paths.extend(scan_metadata(root))
    if not paths:
        paths.extend(scan_metadata(RELEASE_ROOT / "detect_results"))

    unique: dict[str, Path] = {}
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        unique[key] = path
    discovered = sorted(unique.values(), key=lambda p: str(p).lower())
    if args.limit is not None:
        discovered = discovered[: args.limit]
    return discovered


def safe_output_name(cifti_path: Path) -> str:
    match = SUBJECT_RE.search(cifti_path.name)
    if match:
        drug = match.group("drug").upper()
        condition = match.group("condition").upper()
        subject_id = f"S{int(match.group('id')):02d}"
        return f"{drug}_{condition}_{subject_id}"
    name = cifti_path.name
    for suffix in (".dtseries.nii", ".ptseries.nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        name = cifti_path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_subject_fields(cifti_file: str) -> dict[str, str]:
    match = SUBJECT_RE.search(Path(cifti_file).name)
    if not match:
        return {"drug": "", "condition": "", "id": ""}
    return {
        "drug": match.group("drug").upper(),
        "condition": match.group("condition").upper(),
        "id": f"S{int(match.group('id')):02d}",
    }


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    cifti_paths = discover_cifti_paths(args)
    if not cifti_paths:
        raise RuntimeError("No CIFTI files found.")

    gcor_script = MANUSCRIPT_ANALYSIS_ROOT / "gcor_from_cifti.py"
    manifest_path = args.out_root / "gcor_batch_manifest.csv"
    combined_path = args.out_root / "all_gcor_by_subject.csv"

    rows = []
    failures = 0
    for idx, cifti_path in enumerate(cifti_paths, start=1):
        subject_name = safe_output_name(cifti_path)
        subject_out = args.out_root / subject_name
        subject_csv = subject_out / "gcor_by_subject.csv"
        subject_out.mkdir(parents=True, exist_ok=True)

        try:
            if subject_csv.exists() and not args.overwrite:
                status = "skipped_existing"
                print(f"[{idx}/{len(cifti_paths)}] skip {subject_name}")
            else:
                cmd = [
                    sys.executable,
                    str(gcor_script),
                    "--cifti",
                    str(cifti_path),
                    "--out-dir",
                    str(subject_out),
                    "--output-name",
                    "gcor_by_subject.csv",
                    "--no-progress",
                ]
                print(f"[{idx}/{len(cifti_paths)}] run {subject_name}")
                subprocess.run(cmd, check=True)
                status = "ok"
            rows.append(
                {
                    "cifti_file": str(cifti_path),
                    "subject_out": str(subject_out),
                    "gcor_csv": str(subject_csv),
                    "status": status,
                }
            )
        except Exception as exc:
            failures += 1
            print(f"[{idx}/{len(cifti_paths)}] failed {cifti_path}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "cifti_file": str(cifti_path),
                    "subject_out": str(subject_out),
                    "gcor_csv": str(subject_csv),
                    "status": f"failed: {exc}",
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cifti_file", "subject_out", "gcor_csv", "status"])
        writer.writeheader()
        writer.writerows(rows)

    tables = []
    for row in rows:
        csv_path = Path(row["gcor_csv"])
        if csv_path.exists():
            tables.append(pd.read_csv(csv_path))
    if tables:
        combined = pd.concat(tables, ignore_index=True)
        parsed = combined["cifti_file"].map(parse_subject_fields).apply(pd.Series)
        for column in ("drug", "condition", "id"):
            combined[column] = parsed[column]
        summary_columns = ["drug", "condition", "id", "hemisphere", "gcor"]
        combined = combined[combined["status"].astype(str).str.lower() == "ok"]
        combined = combined.loc[:, summary_columns].sort_values(["drug", "condition", "id", "hemisphere"])
        combined.to_csv(combined_path, index=False)
        print(f"Combined GCOR table: {combined_path}")
    else:
        print("No per-subject GCOR CSV files were available to combine.", file=sys.stderr)

    print(f"Manifest: {manifest_path}")
    if failures:
        print(f"Finished with {failures} failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
