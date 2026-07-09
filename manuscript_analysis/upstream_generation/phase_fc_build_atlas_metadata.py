#!/usr/bin/env python
"""
Build shared parcel/network metadata from completed phase-FC outputs.

This does not recompute FC. It scans per-bundle outputs produced by
phase_fc_single_subject.py / phase_fc_batch.py, verifies that parcel metadata is
consistent across subjects within each hemisphere, and writes shared atlas-level
metadata for downstream group analysis.

Example:
    python scripts/phase_fc_build_atlas_metadata.py ^
        --phase-fc-root analysis_outputs/phase_fc_batch ^
        --out-dir analysis_outputs/phase_fc_batch/atlas_metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["parcel_id", "parcel_name", "hemi", "network"]
OPTIONAL_COLUMNS = ["grid_pixel_count", "finite_frames"]
NETWORK_ORDER_7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_ORDER_17 = [
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicB",
    "LimbicA",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
]


def infer_network_order(networks: pd.Series) -> list[str]:
    values = [str(network) for network in networks if pd.notna(network)]
    available = set(values)
    for known_order in (NETWORK_ORDER_7, NETWORK_ORDER_17):
        ordered = [network for network in known_order if network in available]
        if ordered:
            extras = [network for network in values if network not in set(known_order)]
            return ordered + list(dict.fromkeys(extras))
    return list(dict.fromkeys(values))


def infer_atlas_prefix(parcels: pd.DataFrame) -> str:
    names = parcels["parcel_name"].astype(str)
    first = next((name for name in names if "_" in name), "Schaefer2018_400Parcels")
    atlas_tag = first.split("_", 1)[0]
    n_parcels = int(parcels["parcel_id"].nunique())
    if atlas_tag.endswith("Networks"):
        return f"Schaefer2018_{n_parcels}Parcels_{atlas_tag}"
    return f"Schaefer2018_{n_parcels}Parcels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shared atlas metadata and QC tables from phase-FC outputs."
    )
    parser.add_argument(
        "--phase-fc-root",
        required=True,
        type=Path,
        help="Root containing one subdirectory per phase-FC output.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Output directory. Defaults to <phase-fc-root>/atlas_metadata.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any metadata/order/shape mismatch is found.",
    )
    return parser.parse_args()


def load_metadata_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.glob("*/parcel_metadata.csv")
        if p.parent.name != "atlas_metadata"
    )


def normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"parcel_metadata.csv missing columns: {missing}")
    out = df.copy()
    out["parcel_id"] = out["parcel_id"].astype(int)
    out["parcel_name"] = out["parcel_name"].astype(str)
    out["hemi"] = out["hemi"].astype(str)
    out["network"] = out["network"].astype(str)
    return out


def expected_hemi_from_ids(df: pd.DataFrame) -> str:
    min_id = int(df["parcel_id"].min())
    max_id = int(df["parcel_id"].max())
    if min_id >= 1 and max_id <= 200:
        return "LH"
    if min_id >= 201 and max_id <= 400:
        return "RH"
    return "mixed"


def compare_key_metadata(df: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    current = df[KEY_COLUMNS].reset_index(drop=True)
    reference = ref[KEY_COLUMNS].reset_index(drop=True)
    if len(current) != len(reference):
        issues.append(f"row_count {len(current)} != reference {len(reference)}")
        return issues
    if not current["parcel_id"].equals(reference["parcel_id"]):
        issues.append("parcel_id order differs from reference")
    for col in ["parcel_name", "hemi", "network"]:
        mismatch = current[col] != reference[col]
        if mismatch.any():
            first = int(np.flatnonzero(mismatch.to_numpy())[0])
            issues.append(
                f"{col} differs at row {first}: {current.loc[first, col]!r} != {reference.loc[first, col]!r}"
            )
    return issues


def add_indices(parcels: pd.DataFrame, network_order: list[str]) -> pd.DataFrame:
    out = parcels[KEY_COLUMNS].copy()
    out = out.sort_values("parcel_id").reset_index(drop=True)
    out["parcel_index_400"] = out["parcel_id"] - 1
    out["hemi_index_200"] = out.groupby("hemi").cumcount()
    network_rank = {network: idx for idx, network in enumerate(network_order)}
    out["network_index"] = out["network"].map(network_rank).astype("Int64")
    return out[
        [
            "parcel_id",
            "parcel_index_400",
            "hemi_index_200",
            "parcel_name",
            "hemi",
            "network",
            "network_index",
        ]
    ]


def build_networks(parcels: pd.DataFrame, network_order: list[str]) -> pd.DataFrame:
    rows = []
    for idx, network in enumerate(network_order):
        subset = parcels[parcels["network"] == network]
        if subset.empty:
            continue
        rows.append(
            {
                "network": network,
                "network_index": idx,
                "n_parcels": int(len(subset)),
                "n_lh": int(np.sum(subset["hemi"] == "LH")),
                "n_rh": int(np.sum(subset["hemi"] == "RH")),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    root = args.phase_fc_root
    out_dir = args.out_dir or (root / "atlas_metadata")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = load_metadata_files(root)
    if not meta_paths:
        raise RuntimeError(f"No parcel_metadata.csv files found under {root}")

    refs: dict[str, pd.DataFrame] = {}
    qc_rows = []
    pixel_rows = []

    for path in meta_paths:
        try:
            df = normalize_meta(pd.read_csv(path))
            hemi = expected_hemi_from_ids(df)
            issues: list[str] = []
            if hemi not in {"LH", "RH"}:
                issues.append(f"unexpected parcel_id range: {df['parcel_id'].min()}-{df['parcel_id'].max()}")
            if df["hemi"].nunique() != 1 or (hemi in {"LH", "RH"} and df["hemi"].iloc[0] != hemi):
                issues.append(f"hemi column mismatch: values={sorted(df['hemi'].unique())}, expected={hemi}")
            if not df["parcel_id"].is_monotonic_increasing:
                issues.append("parcel_id is not sorted ascending")
            if hemi in refs:
                issues.extend(compare_key_metadata(df, refs[hemi]))
            elif hemi in {"LH", "RH"}:
                refs[hemi] = df[KEY_COLUMNS].copy()

            plv_path = path.parent / "parcel_plv.npy"
            if plv_path.exists():
                plv_shape = tuple(np.load(plv_path, mmap_mode="r").shape)
                if plv_shape != (len(df), len(df)):
                    issues.append(f"parcel_plv shape {plv_shape} != ({len(df)}, {len(df)})")
            else:
                issues.append("missing parcel_plv.npy")

            for col in OPTIONAL_COLUMNS:
                if col in df.columns:
                    pixel_rows.append(
                        {
                            "bundle": path.parent.name,
                            "hemi": hemi,
                            "column": col,
                            "min": float(df[col].min()),
                            "median": float(df[col].median()),
                            "max": float(df[col].max()),
                        }
                    )

            qc_rows.append(
                {
                    "bundle": path.parent.name,
                    "parcel_metadata": str(path),
                    "hemi": hemi,
                    "n_parcels": int(len(df)),
                    "status": "ok" if not issues else "mismatch",
                    "issues": "; ".join(issues),
                }
            )
        except Exception as exc:
            qc_rows.append(
                {
                    "bundle": path.parent.name,
                    "parcel_metadata": str(path),
                    "hemi": "",
                    "n_parcels": 0,
                    "status": "error",
                    "issues": str(exc),
                }
            )

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out_dir / "metadata_qc.csv", index=False)
    if pixel_rows:
        pd.DataFrame(pixel_rows).to_csv(out_dir / "parcel_count_qc.csv", index=False)

    if "LH" not in refs or "RH" not in refs:
        print(
            "Warning: only found hemisphere metadata for "
            f"{sorted(refs)}; writing partial parcel metadata."
        )

    ordered_refs = [refs[hemi] for hemi in ("LH", "RH") if hemi in refs]
    raw_parcels = pd.concat(ordered_refs, ignore_index=True)
    network_order = infer_network_order(raw_parcels["network"])
    parcels = add_indices(raw_parcels, network_order)
    networks = build_networks(parcels, network_order)
    atlas_prefix = infer_atlas_prefix(parcels)
    parcel_filename = f"{atlas_prefix}_parcels.csv"
    network_filename = f"{atlas_prefix}_networks.csv"
    parcels.to_csv(out_dir / parcel_filename, index=False)
    networks.to_csv(out_dir / network_filename, index=False)

    summary = {
        "phase_fc_root": str(root),
        "n_metadata_files": int(len(meta_paths)),
        "n_ok": int(np.sum(qc["status"] == "ok")),
        "n_mismatch": int(np.sum(qc["status"] == "mismatch")),
        "n_error": int(np.sum(qc["status"] == "error")),
        "network_order": network_order,
        "parcel_metadata": parcel_filename,
        "network_metadata": network_filename,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Scanned {len(meta_paths)} metadata files")
    print(f"QC ok: {summary['n_ok']}; mismatch: {summary['n_mismatch']}; error: {summary['n_error']}")
    print(f"Wrote: {out_dir}")
    if args.strict and (summary["n_mismatch"] or summary["n_error"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
