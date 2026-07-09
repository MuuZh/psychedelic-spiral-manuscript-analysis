#!/usr/bin/env python
"""
Compute left/right hemisphere GCOR directly from CIFTI time series files.

For each CIFTI file, rows/columns are oriented to a (time, nodes) matrix, then
split by CIFTI brain structures CORTEX_LEFT and CORTEX_RIGHT. Each node time
series is demeaned and L2-normalized:

    u_i = (x_i - mean(x_i)) / ||x_i - mean(x_i)||

The subject/hemisphere GCOR is then:

    GCOR = || mean_i(u_i) ||^2

This is equivalent to the mean of all pairwise correlations including the
diagonal, without materializing the full correlation matrix.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel import cifti2
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


SUBJECT_RE = re.compile(
    r"(?P<drug>[A-Za-z]+)_(?P<condition>[A-Za-z]+)_S(?P<id>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GcorResult:
    gcor: float
    n_timepoints: int
    n_nodes_total: int
    n_nodes_used: int
    n_nodes_dropped_nonfinite: int
    n_nodes_dropped_zero_norm: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-subject GCOR from CIFTI time series files.")
    parser.add_argument(
        "--cifti",
        action="append",
        default=None,
        type=Path,
        help="CIFTI time series file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--cifti-root",
        action="append",
        default=None,
        type=Path,
        help="Root to recursively search for CIFTI time series files.",
    )
    parser.add_argument(
        "--metadata-root",
        default=PROJECT_ROOT / "detect_results",
        type=Path,
        help="Root to scan for metadata.json files containing cifti_file. Used when --cifti/--cifti-root are omitted.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern used with --cifti-root. Default: *.dtseries.nii and *.ptseries.nii.",
    )
    parser.add_argument(
        "--out-dir",
        default=PROJECT_ROOT / "analysis_outputs" / "gcor_from_cifti",
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name",
        default="gcor_by_subject.csv",
        help="Output CSV file name.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return parser.parse_args()


def parse_subject_fields(path: Path) -> dict[str, str]:
    match = SUBJECT_RE.search(path.name)
    if not match:
        return {"drug": "", "condition": "", "id": "", "subject_id": path.stem}
    drug = match.group("drug").upper()
    condition = match.group("condition").upper()
    subject_id = f"S{int(match.group('id')):02d}"
    return {
        "drug": drug,
        "condition": condition,
        "id": subject_id,
        "subject_id": f"{drug}_{condition}_{subject_id}",
    }


def resolve_existing_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
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
        if not cifti_file:
            continue
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
    if not paths:
        paths.extend(scan_metadata(args.metadata_root))

    unique: dict[str, Path] = {}
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        unique[key] = path
    return sorted(unique.values(), key=lambda p: str(p).lower())


def indices_from_slice(slice_obj: slice | np.ndarray, size: int) -> np.ndarray:
    if isinstance(slice_obj, slice):
        return np.arange(
            slice_obj.start or 0,
            slice_obj.stop if slice_obj.stop is not None else size,
            slice_obj.step or 1,
        )
    return np.asarray(slice_obj)


def load_cifti_hemisphere_matrices(path: Path) -> dict[str, np.ndarray]:
    img = nib.load(path)
    if not isinstance(img, cifti2.Cifti2Image):
        raise ValueError(f"Not a CIFTI-2 image: {path}")
    if len(img.shape) != 2:
        raise ValueError(f"Expected a 2D CIFTI time series, got shape {img.shape}: {path}")

    axis0 = img.header.get_axis(0)
    axis1 = img.header.get_axis(1)
    if isinstance(axis0, cifti2.SeriesAxis):
        data = img.get_fdata(dtype=np.float32)
        node_axis = axis1
    elif isinstance(axis1, cifti2.SeriesAxis):
        data = img.get_fdata(dtype=np.float32).T
        node_axis = axis0
    else:
        raise ValueError(
            "CIFTI does not contain a SeriesAxis; expected dtseries/ptseries-like time series: "
            f"{path}"
        )

    if data.ndim != 2:
        raise ValueError(f"Expected 2D data after orientation, got {data.shape}: {path}")
    if not isinstance(node_axis, cifti2.BrainModelAxis):
        raise ValueError(
            "CIFTI node axis is not a BrainModelAxis, so left/right cortex cannot be split from the file: "
            f"{path}"
        )

    hemisphere_mats: dict[str, np.ndarray] = {}
    structure_map = {
        "CIFTI_STRUCTURE_CORTEX_LEFT": "left",
        "CORTEX_LEFT": "left",
        "CIFTI_STRUCTURE_CORTEX_RIGHT": "right",
        "CORTEX_RIGHT": "right",
    }
    for structure_name, slice_obj, _ in node_axis.iter_structures():
        hemisphere = structure_map.get(structure_name)
        if hemisphere is None:
            continue
        indices = indices_from_slice(slice_obj, node_axis.size)
        hemisphere_mats[hemisphere] = np.asarray(data[:, indices], dtype=np.float32)

    missing = sorted({"left", "right"} - set(hemisphere_mats))
    if missing:
        raise ValueError(f"Missing cortex structure(s) {missing} in CIFTI: {path}")
    return hemisphere_mats


def compute_gcor(time_by_nodes: np.ndarray) -> GcorResult:
    if time_by_nodes.ndim != 2:
        raise ValueError(f"Expected (time, nodes) matrix, got {time_by_nodes.shape}")

    n_timepoints, n_nodes_total = time_by_nodes.shape
    if n_timepoints < 2 or n_nodes_total < 1:
        return GcorResult(np.nan, n_timepoints, n_nodes_total, 0, n_nodes_total, 0)

    finite_cols = np.all(np.isfinite(time_by_nodes), axis=0)
    x = time_by_nodes[:, finite_cols]
    n_nonfinite = int(n_nodes_total - x.shape[1])
    if x.shape[1] == 0:
        return GcorResult(np.nan, n_timepoints, n_nodes_total, 0, n_nonfinite, 0)

    x = x - np.mean(x, axis=0, keepdims=True)
    norms = np.linalg.norm(x, axis=0)
    valid_norm = norms > 0
    x = x[:, valid_norm]
    norms = norms[valid_norm]
    n_zero_norm = int(np.count_nonzero(~valid_norm))
    if x.shape[1] == 0:
        return GcorResult(np.nan, n_timepoints, n_nodes_total, 0, n_nonfinite, n_zero_norm)

    u_mean = np.mean(x / norms[np.newaxis, :], axis=1)
    gcor = float(np.dot(u_mean, u_mean))
    return GcorResult(
        gcor=gcor,
        n_timepoints=int(n_timepoints),
        n_nodes_total=int(n_nodes_total),
        n_nodes_used=int(x.shape[1]),
        n_nodes_dropped_nonfinite=n_nonfinite,
        n_nodes_dropped_zero_norm=n_zero_norm,
    )


def compute_table(cifti_paths: list[Path], show_progress: bool) -> pd.DataFrame:
    rows = []
    iterator = cifti_paths
    if show_progress:
        iterator = tqdm(cifti_paths, desc="GCOR CIFTI", unit="file")

    for path in iterator:
        subject_fields = parse_subject_fields(path)
        row = {
            **subject_fields,
            "cifti_file": str(path),
            "status": "ok",
            "error": "",
        }
        if not path.exists():
            row.update({"status": "missing", "error": "file not found"})
            rows.append(row)
            continue
        try:
            hemi_data = load_cifti_hemisphere_matrices(path)
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc), "hemisphere": ""})
            rows.append(row)
            continue

        for hemisphere in ("left", "right"):
            hemi_row = {**row, "hemisphere": hemisphere}
            try:
                result = compute_gcor(hemi_data[hemisphere])
                hemi_row.update(result.__dict__)
            except Exception as exc:
                hemi_row.update({"status": "failed", "error": str(exc)})
            rows.append(hemi_row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cifti_paths = discover_cifti_paths(args)
    if not cifti_paths:
        raise RuntimeError("No CIFTI files found.")

    table = compute_table(cifti_paths, show_progress=not args.no_progress)
    out_csv = args.out_dir / args.output_name
    table.to_csv(out_csv, index=False)

    metadata = {
        "n_files": len(cifti_paths),
        "output_csv": str(out_csv),
        "formula": "GCOR = ||mean_i((x_i - mean(x_i)) / ||x_i - mean(x_i)||)||^2",
        "note": "Rows with status=ok are split by CORTEX_LEFT/CORTEX_RIGHT and use finite, non-constant nodes.",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote GCOR table: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
