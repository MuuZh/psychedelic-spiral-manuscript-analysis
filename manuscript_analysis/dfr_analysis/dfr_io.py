from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONDITION_RE = re.compile(r"^(?P<condition>[A-Za-z]+_[A-Za-z]+)_S(?P<subid>\d+)")


@dataclass(frozen=True)
class PhaseEntry:
    source: str
    drug: str
    comparison: str
    condition: str
    role: str
    subid: str
    hemisphere: str
    bundle_dir: Path
    phase_cube: Path
    grid_path: Path | None
    parcel_meta_path: Path | None
    run_metadata: Path | None


def normalize_subid(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("S"):
        text = text[1:]
    return f"{int(text):02d}" if text.isdigit() else text


def canonical_condition(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


def resolve_existing_path(path_like: str | Path | None, base: Path | None = None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    candidates = [path] if path.is_absolute() else [path, Path.cwd() / path]
    if base is not None and not path.is_absolute():
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_hemisphere(bundle_dir: Path, metadata: dict[str, object] | None, parcel_meta: Path | None) -> str | None:
    if metadata:
        hemi = str(metadata.get("hemisphere", "")).lower()
        if hemi in {"left", "right"}:
            return hemi
    if parcel_meta and parcel_meta.exists():
        try:
            meta = pd.read_csv(parcel_meta, usecols=["hemi"])
            value = str(meta["hemi"].iloc[0])
            if value == "LH":
                return "left"
            if value == "RH":
                return "right"
        except Exception:
            pass
    suffix = bundle_dir.name[-1:].upper()
    if suffix == "L":
        return "left"
    if suffix == "R":
        return "right"
    return None


def parse_phase_entry(
    bundle_dir: Path,
    source: str,
    drug: str,
    drug_condition: str,
    pcb_condition: str,
) -> PhaseEntry | None:
    match = CONDITION_RE.search(bundle_dir.name)
    if match is None:
        return None
    condition = match.group("condition")
    canon = canonical_condition(condition)
    if canon == canonical_condition(drug_condition):
        role = "Drug"
    elif canon == canonical_condition(pcb_condition):
        role = "PCB"
    else:
        return None

    metadata_path = bundle_dir / "run_metadata.json"
    metadata: dict[str, object] | None = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = None
    phase_cube = resolve_existing_path((metadata or {}).get("phase_cube"), metadata_path.parent if metadata_path.exists() else None)
    phase_cube = phase_cube or (bundle_dir / "phase_cube.npy" if (bundle_dir / "phase_cube.npy").exists() else None)
    if phase_cube is None:
        return None

    grid_path = bundle_dir / "grid_labels.npy"
    parcel_meta = bundle_dir / "parcel_metadata.csv"
    hemisphere = infer_hemisphere(bundle_dir, metadata, parcel_meta if parcel_meta.exists() else None)
    if hemisphere is None:
        return None

    return PhaseEntry(
        source=source,
        drug=drug,
        comparison=f"{drug_condition}_vs_{pcb_condition}",
        condition=condition,
        role=role,
        subid=normalize_subid(match.group("subid")),
        hemisphere=hemisphere,
        bundle_dir=bundle_dir,
        phase_cube=phase_cube,
        grid_path=grid_path if grid_path.exists() else None,
        parcel_meta_path=parcel_meta if parcel_meta.exists() else None,
        run_metadata=metadata_path if metadata_path.exists() else None,
    )


def discover_phase_entries(
    roots: Iterable[Path],
    source: str,
    drug: str,
    drug_condition: str,
    pcb_condition: str,
    hemispheres: Iterable[str],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    allowed_hemi = set(hemispheres)
    for root in roots:
        if not root.exists():
            failures.append({"source": source, "root": str(root), "error": "root_missing"})
            continue
        for bundle_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "atlas_metadata"):
            entry = parse_phase_entry(bundle_dir, source, drug, drug_condition, pcb_condition)
            if entry is not None and entry.hemisphere in allowed_hemi:
                rows.append(entry.__dict__)
    return pd.DataFrame(rows), failures


def load_grid(entry: pd.Series) -> np.ndarray | None:
    path = entry.get("grid_path")
    if pd.isna(path) or not path:
        return None
    return np.load(Path(path))


def ensure_outdirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "figures": out_dir / "figures",
        "frame_qc": out_dir / "figures" / "frame_qc",
        "maps": out_dir / "boundary_maps",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
