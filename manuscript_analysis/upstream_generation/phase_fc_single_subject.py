#!/usr/bin/env python
"""
Single-subject phase-FC on the downsampled MatPhase grid.

This script projects a CIFTI dlabel atlas (for example Schaefer 400/7-network)
onto the same flattened, downsampled 2D grid used by phase_cube.npy, computes
parcel-level circular mean phase time series, then estimates phase-FC with PLV.

Example:
    python scripts/phase_fc_single_subject.py ^
        --phase-cube detect_results/DMT/.../phase_cube.npy ^
        --hemisphere left ^
        --dlabel <derived_data>/atlases/Schaefer2018_400Parcels_7Networks_order.dlabel.nii ^
        --surface testdata/L.flat.32k_fs_LR.surf.gii ^
        --config configs/defaults.yaml ^
        --out-dir analysis_outputs/phase_fc_single/sub01_left
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata

from matphase.config import load_config
from matphase.io.parcellation import load_parcellation
from matphase.io.surface import load_surface
from matphase.preprocess.interpolate import (
    generate_coordinate_grid,
    shift_coordinates_to_positive,
)


NETWORK_ORDER_7 = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
]
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
NETWORK_SUMMARY_COLUMNS = ["network_a", "network_b", "type", "n_a", "n_b", "mean_plv"]


def infer_network_order(networks: Iterable[str]) -> list[str]:
    """Return a stable plotting/summarization order for 7- or 17-network Schaefer labels."""
    available = [str(network) for network in networks if pd.notna(network)]
    available_set = set(available)
    for known_order in (NETWORK_ORDER_7, NETWORK_ORDER_17):
        ordered = [network for network in known_order if network in available_set]
        if ordered:
            extras = [network for network in available if network not in set(known_order)]
            return ordered + list(dict.fromkeys(extras))
    return list(dict.fromkeys(available))


@dataclass(frozen=True)
class GridSpec:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    downsample_rate: int
    coordinate_system: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project dlabel to a MatPhase phase grid and compute single-subject phase-FC."
    )
    parser.add_argument("--phase-cube", required=True, type=Path, help="Path to phase_cube.npy.")
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
    parser.add_argument(
        "--surface",
        required=True,
        type=Path,
        help="Flat surface .surf.gii for the same hemisphere as the phase cube.",
    )
    parser.add_argument("--hemisphere", required=True, choices=["left", "right"])
    parser.add_argument("--config", default=Path("configs/defaults.yaml"), type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--min-pixels",
        default=3,
        type=int,
        help="Minimum number of valid grid pixels required to keep a parcel.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        default=0.95,
        type=float,
        help="Minimum pairwise valid time fraction for PLV calculation.",
    )
    parser.add_argument(
        "--no-drop-auto",
        action="store_true",
        help="Disable automatic final-row drop when dlabel grid has one row more than phase cube.",
    )
    parser.add_argument(
        "--save-parcel-phase",
        action="store_true",
        help="Also save parcel circular-mean phase time series as .npy.",
    )
    parser.add_argument(
        "--compute-phase-corr",
        action="store_true",
        help="Also compute Pearson correlation on parcel circular-mean phase time series.",
    )
    parser.add_argument(
        "--parcellation",
        default=None,
        type=Path,
        help="Optional existing grid parcellation .npy for side-by-side visual comparison. "
        "Defaults to the hemisphere-specific path in the config when available.",
    )
    parser.add_argument(
        "--no-parcellation-comparison",
        action="store_true",
        help="Do not try to plot the Schaefer projection next to the existing grid parcellation.",
    )
    return parser.parse_args()


def grid_spec_from_config(config_path: Path, hemisphere: str) -> GridSpec:
    cfg = load_config(config_path)
    prep = cfg.preprocessing
    if hemisphere == "left":
        x_range = (float(prep.left_x_coord_min), float(prep.left_x_coord_max))
        y_range = (float(prep.left_y_coord_min), float(prep.left_y_coord_max))
    else:
        x_range = (float(prep.right_x_coord_min), float(prep.right_x_coord_max))
        y_range = (float(prep.right_y_coord_min), float(prep.right_y_coord_max))
    return GridSpec(
        x_range=x_range,
        y_range=y_range,
        downsample_rate=int(prep.downsample_rate),
        coordinate_system=str(prep.interpolation_coordinate_system),
    )


def parcellation_path_from_config(config_path: Path, hemisphere: str) -> Path | None:
    cfg = load_config(config_path)
    data_dir = Path(cfg.paths.data_dir)
    parc = cfg.paths.parcellation_left if hemisphere == "left" else cfg.paths.parcellation_right
    if parc is None:
        return None
    parc_path = Path(parc)
    if not parc_path.is_absolute():
        parc_path = data_dir / parc_path
    return parc_path


def read_label_table(dlabel_path: Path) -> pd.DataFrame:
    label_axis = nib.load(str(dlabel_path)).header.get_axis(0)
    _, table, _ = label_axis[0]
    rows = []
    for label_id, (name, rgba) in table.items():
        if int(label_id) == 0:
            continue
        parts = str(name).split("_")
        hemi = parts[1] if len(parts) > 1 else "unknown"
        network = parts[2] if len(parts) > 2 else "unknown"
        rows.append(
            {
                "parcel_id": int(label_id),
                "parcel_name": str(name),
                "hemi": hemi,
                "network": network,
                "rgba": tuple(float(v) for v in rgba),
            }
        )
    return pd.DataFrame(rows)


def dlabel_surface_labels(dlabel_path: Path, hemisphere: str) -> np.ndarray:
    img = nib.load(str(dlabel_path))
    data = np.asarray(img.get_fdata()).squeeze()
    if data.ndim != 1:
        data = np.asarray(data[0]).squeeze()

    brain_axis = None
    for axis_idx in range(len(img.shape)):
        axis = img.header.get_axis(axis_idx)
        if axis.__class__.__name__ == "BrainModelAxis":
            brain_axis = axis
            break
    if brain_axis is None:
        raise ValueError(f"No BrainModelAxis found in {dlabel_path}")

    structure = "CIFTI_STRUCTURE_CORTEX_LEFT" if hemisphere == "left" else "CIFTI_STRUCTURE_CORTEX_RIGHT"
    for name, slc, _ in brain_axis.iter_structures():
        if name != structure:
            continue
        n_vertices = int(brain_axis.nvertices[name])
        vertex_indices = np.asarray(brain_axis.vertex[slc], dtype=np.int64)
        values = np.asarray(data[slc], dtype=float)
        labels = np.full(n_vertices, np.nan, dtype=float)
        labels[vertex_indices] = values
        labels[labels == 0] = np.nan
        return labels
    raise ValueError(f"{structure} not found in {dlabel_path}")


def project_labels_to_grid(
    labels: np.ndarray,
    surface_vertices: np.ndarray,
    spec: GridSpec,
) -> np.ndarray:
    return_physical = spec.coordinate_system == "physical"
    x_coords, y_coords, x_grid, y_grid = generate_coordinate_grid(
        spec.x_range,
        spec.y_range,
        spec.downsample_rate,
        return_physical=return_physical,
    )

    xy = np.asarray(surface_vertices[:, :2], dtype=float)
    if spec.coordinate_system == "positive":
        xy, _ = shift_coordinates_to_positive(xy, spec.x_range, spec.y_range)
    elif spec.coordinate_system != "physical":
        raise ValueError(f"Unsupported coordinate system: {spec.coordinate_system}")

    valid = np.isfinite(labels) & np.isfinite(xy).all(axis=1)
    grid_labels = griddata(
        xy[valid],
        labels[valid],
        (x_grid, y_grid),
        method="nearest",
        fill_value=np.nan,
    )
    return grid_labels.astype(np.float32, copy=False)


def align_grid_to_phase(grid: np.ndarray, phase_shape: tuple[int, int], drop_auto: bool) -> np.ndarray:
    if grid.shape == phase_shape:
        return grid
    if drop_auto and grid.shape[0] == phase_shape[0] + 1 and grid.shape[1] == phase_shape[1]:
        return grid[:-1, :]
    raise ValueError(f"Projected dlabel grid shape {grid.shape} does not match phase grid {phase_shape}")


def align_2d_to_phase(name: str, grid: np.ndarray, phase_shape: tuple[int, int]) -> np.ndarray:
    if grid.shape == phase_shape:
        return grid
    if grid.shape[0] == phase_shape[0] + 1 and grid.shape[1] == phase_shape[1]:
        return grid[:-1, :]
    raise ValueError(f"{name} shape {grid.shape} does not match phase grid {phase_shape}")


def circular_mean_phase(phase_values: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.angle(np.nanmean(np.exp(1j * phase_values), axis=axis))


def compute_parcel_phase(
    phase_cube: np.ndarray,
    grid_labels: np.ndarray,
    parcels: Iterable[int],
    min_pixels: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    parcel_phase = []
    rows = []
    finite_phase_any = np.any(np.isfinite(phase_cube), axis=2)
    for parcel_id in parcels:
        mask = (grid_labels == parcel_id) & finite_phase_any
        pixel_count = int(np.sum(mask))
        if pixel_count < min_pixels:
            continue
        phase_ts = circular_mean_phase(phase_cube[mask, :], axis=0)
        finite_frames = int(np.sum(np.isfinite(phase_ts)))
        if finite_frames == 0:
            continue
        parcel_phase.append(phase_ts.astype(np.float32, copy=False))
        rows.append(
            {
                "parcel_id": int(parcel_id),
                "grid_pixel_count": pixel_count,
                "finite_frames": finite_frames,
            }
        )
    if not parcel_phase:
        raise RuntimeError("No parcels survived grid projection and min-pixels filtering.")
    return np.vstack(parcel_phase), pd.DataFrame(rows)


def plv_matrix(parcel_phase: np.ndarray, min_valid_fraction: float) -> np.ndarray:
    n_parcels, n_frames = parcel_phase.shape
    out = np.full((n_parcels, n_parcels), np.nan, dtype=np.float32)
    z = np.exp(1j * parcel_phase)
    finite = np.isfinite(parcel_phase)

    for i in range(n_parcels):
        out[i, i] = np.nan
        for j in range(i + 1, n_parcels):
            valid = finite[i] & finite[j]
            if np.mean(valid) < min_valid_fraction:
                continue
            value = np.abs(np.mean(z[i, valid] * np.conj(z[j, valid])))
            out[i, j] = out[j, i] = np.float32(value)
    return out


def phase_corr_matrix(parcel_phase: np.ndarray, min_valid_fraction: float) -> np.ndarray:
    """Pearson correlation between parcel phase time series in radians."""
    n_parcels, _ = parcel_phase.shape
    out = np.full((n_parcels, n_parcels), np.nan, dtype=np.float32)
    finite = np.isfinite(parcel_phase)
    for i in range(n_parcels):
        out[i, i] = np.nan
        for j in range(i + 1, n_parcels):
            valid = finite[i] & finite[j]
            if np.mean(valid) < min_valid_fraction or np.sum(valid) < 3:
                continue
            x = parcel_phase[i, valid]
            y = parcel_phase[j, valid]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            value = np.corrcoef(x, y)[0, 1]
            out[i, j] = out[j, i] = np.float32(value)
    return out


def summarize_networks(fc: np.ndarray, parcel_meta: pd.DataFrame, network_order: list[str]) -> pd.DataFrame:
    rows = []
    networks = [net for net in network_order if net in set(parcel_meta["network"])]
    network_rank = {network: idx for idx, network in enumerate(network_order)}
    for net_a in networks:
        ids_a = np.flatnonzero(parcel_meta["network"].to_numpy() == net_a)
        if ids_a.size >= 2:
            sub = fc[np.ix_(ids_a, ids_a)]
            rows.append(
                {
                    "network_a": net_a,
                    "network_b": net_a,
                    "type": "within",
                    "n_a": int(ids_a.size),
                    "n_b": int(ids_a.size),
                    "mean_plv": float(np.nanmean(sub[np.triu_indices_from(sub, k=1)])),
                }
            )
        for net_b in networks:
            if network_rank[net_b] <= network_rank[net_a]:
                continue
            ids_b = np.flatnonzero(parcel_meta["network"].to_numpy() == net_b)
            sub = fc[np.ix_(ids_a, ids_b)]
            rows.append(
                {
                    "network_a": net_a,
                    "network_b": net_b,
                    "type": "between",
                    "n_a": int(ids_a.size),
                    "n_b": int(ids_b.size),
                    "mean_plv": float(np.nanmean(sub)),
                }
            )
    return pd.DataFrame(rows, columns=NETWORK_SUMMARY_COLUMNS)


def plot_grid_labels(grid_labels: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    masked = np.ma.masked_invalid(grid_labels)
    plt.imshow(masked, interpolation="nearest", cmap="tab20")
    plt.title("Schaefer dlabel projected to phase grid")
    plt.axis("off")
    plt.colorbar(label="Parcel ID", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_label_comparison(
    grid_labels: np.ndarray,
    parcellation: np.ndarray,
    phase_valid_mask: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    panels = [
        ("Schaefer dlabel projected", grid_labels),
        ("Existing parcellation.npy", parcellation),
        ("Phase valid mask", phase_valid_mask.astype(float)),
    ]
    for ax, (title, data) in zip(axes, panels):
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, interpolation="nearest", cmap="tab20")
        ax.set_title(title)
        ax.axis("off")
        if title != "Phase valid mask":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_plv_heatmap(plv: np.ndarray, parcel_meta: pd.DataFrame, out_path: Path, network_order: list[str]) -> None:
    sort_key = parcel_meta["network"].map({net: i for i, net in enumerate(network_order)}).fillna(999)
    order = np.lexsort((parcel_meta["parcel_id"].to_numpy(), sort_key.to_numpy()))
    sorted_plv = plv[np.ix_(order, order)]
    sorted_meta = parcel_meta.iloc[order].reset_index(drop=True)

    plt.figure(figsize=(9, 8))
    sns.heatmap(sorted_plv, cmap="viridis", vmin=0, vmax=1, square=True, cbar_kws={"label": "PLV"})
    bounds = []
    labels = []
    start = 0
    for network, group in sorted_meta.groupby("network", sort=False):
        end = start + len(group)
        bounds.append((start, end))
        labels.append(network)
        start = end
    centers = [(a + b) / 2 for a, b in bounds]
    for _, b in bounds[:-1]:
        plt.axhline(b, color="white", linewidth=0.8)
        plt.axvline(b, color="white", linewidth=0.8)
    plt.xticks(centers, labels, rotation=45, ha="right")
    plt.yticks(centers, labels, rotation=0)
    plt.title("Parcel phase-FC (PLV)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_fc_heatmap(fc: np.ndarray, parcel_meta: pd.DataFrame, out_path: Path, title: str, label: str, *, vmin: float, vmax: float, center: float | None = None) -> None:
    network_order = infer_network_order(parcel_meta["network"])
    sort_key = parcel_meta["network"].map({net: i for i, net in enumerate(network_order)}).fillna(999)
    order = np.lexsort((parcel_meta["parcel_id"].to_numpy(), sort_key.to_numpy()))
    sorted_fc = fc[np.ix_(order, order)]
    sorted_meta = parcel_meta.iloc[order].reset_index(drop=True)

    plt.figure(figsize=(9, 8))
    cmap = "coolwarm" if center is not None else "viridis"
    sns.heatmap(
        sorted_fc,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        cbar_kws={"label": label},
    )
    bounds = []
    labels = []
    start = 0
    for network, group in sorted_meta.groupby("network", sort=False):
        end = start + len(group)
        bounds.append((start, end))
        labels.append(network)
        start = end
    centers = [(a + b) / 2 for a, b in bounds]
    for _, b in bounds[:-1]:
        plt.axhline(b, color="white", linewidth=0.8)
        plt.axvline(b, color="white", linewidth=0.8)
    plt.xticks(centers, labels, rotation=45, ha="right")
    plt.yticks(centers, labels, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_network_matrix(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Within/between network PLV",
    label: str = "Mean PLV",
    vmin: float = 0,
    vmax: float = 1,
    center: float | None = None,
) -> None:
    if summary.empty:
        plt.figure(figsize=(7, 6))
        plt.text(0.5, 0.5, "No network summary rows", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()
        return

    network_order = infer_network_order(list(summary["network_a"]) + list(summary["network_b"]))
    networks = [net for net in network_order if net in set(summary["network_a"]) | set(summary["network_b"])]
    mat = pd.DataFrame(np.nan, index=networks, columns=networks)
    for row in summary.itertuples(index=False):
        mat.loc[row.network_a, row.network_b] = row.mean_plv
        mat.loc[row.network_b, row.network_a] = row.mean_plv
    plt.figure(figsize=(7, 6))
    cmap = "coolwarm" if center is not None else "mako"
    sns.heatmap(mat, cmap=cmap, vmin=vmin, vmax=vmax, center=center, annot=True, fmt=".3f", cbar_kws={"label": label})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phase_cube = np.load(args.phase_cube, mmap_mode="r")
    if phase_cube.ndim != 3:
        raise ValueError(f"Expected phase cube with shape (rows, cols, frames), got {phase_cube.shape}")

    spec = grid_spec_from_config(args.config, args.hemisphere)
    surface = load_surface(args.surface, hemisphere=args.hemisphere)
    surface_labels = dlabel_surface_labels(args.dlabel, args.hemisphere)
    grid_labels = project_labels_to_grid(surface_labels, surface.vertices, spec)
    grid_labels = align_grid_to_phase(
        grid_labels,
        phase_shape=phase_cube.shape[:2],
        drop_auto=not args.no_drop_auto,
    )
    phase_valid_mask = np.any(np.isfinite(phase_cube), axis=2)
    grid_labels = grid_labels.copy()
    grid_labels[~phase_valid_mask] = np.nan

    label_table = read_label_table(args.dlabel)
    hemi_code = "LH" if args.hemisphere == "left" else "RH"
    label_table = label_table[label_table["hemi"] == hemi_code].copy()
    projected_ids = sorted(int(v) for v in np.unique(grid_labels[np.isfinite(grid_labels)]) if v > 0)
    candidate_meta = label_table[label_table["parcel_id"].isin(projected_ids)].copy()

    parcel_phase, parcel_counts = compute_parcel_phase(
        np.asarray(phase_cube),
        grid_labels,
        parcels=candidate_meta["parcel_id"].to_numpy(),
        min_pixels=args.min_pixels,
    )
    parcel_counts = parcel_counts.reset_index(drop=False).rename(columns={"index": "phase_row"})
    parcel_meta = candidate_meta.merge(parcel_counts, on="parcel_id", how="inner")
    parcel_meta = parcel_meta.sort_values("parcel_id").reset_index(drop=True)
    parcel_phase = parcel_phase[parcel_meta["phase_row"].to_numpy()]
    parcel_meta = parcel_meta.drop(columns=["phase_row"])
    network_order = infer_network_order(parcel_meta["network"])

    plv = plv_matrix(parcel_phase, min_valid_fraction=args.min_valid_fraction)
    summary = summarize_networks(plv, parcel_meta, network_order)
    phase_corr = None
    phase_corr_summary = None
    if args.compute_phase_corr:
        phase_corr = phase_corr_matrix(parcel_phase, min_valid_fraction=args.min_valid_fraction)
        phase_corr_summary = summarize_networks(phase_corr, parcel_meta, network_order)

    np.save(args.out_dir / "grid_labels.npy", grid_labels)
    np.save(args.out_dir / "parcel_plv.npy", plv)
    if phase_corr is not None:
        np.save(args.out_dir / "parcel_phase_corr.npy", phase_corr)
    if args.save_parcel_phase:
        np.save(args.out_dir / "parcel_phase.npy", parcel_phase)
    parcel_meta.to_csv(args.out_dir / "parcel_metadata.csv", index=False)
    summary.to_csv(args.out_dir / "network_within_between_plv.csv", index=False)
    if phase_corr_summary is not None:
        phase_corr_summary.to_csv(args.out_dir / "network_within_between_phase_corr.csv", index=False)
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "phase_cube": str(args.phase_cube),
                "dlabel": str(args.dlabel),
                "surface": str(args.surface),
                "hemisphere": args.hemisphere,
                "phase_shape": list(phase_cube.shape),
                "grid_spec": spec.__dict__,
                "n_projected_parcels": int(len(projected_ids)),
                "n_kept_parcels": int(len(parcel_meta)),
                "min_pixels": int(args.min_pixels),
                "min_valid_fraction": float(args.min_valid_fraction),
                "compute_phase_corr": bool(args.compute_phase_corr),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_grid_labels(grid_labels, args.out_dir / "grid_labels.png")
    comparison_path = None
    if not args.no_parcellation_comparison:
        parc_path = args.parcellation or parcellation_path_from_config(args.config, args.hemisphere)
        if parc_path is not None and parc_path.exists():
            parcellation = align_2d_to_phase(
                "parcellation",
                load_parcellation(parc_path),
                phase_shape=phase_cube.shape[:2],
            )
            parcellation = parcellation.astype(float, copy=True)
            parcellation[~phase_valid_mask] = np.nan
            comparison_path = args.out_dir / "dlabel_vs_parcellation.png"
            plot_label_comparison(grid_labels, parcellation, phase_valid_mask, comparison_path)
        else:
            print(f"Parcellation comparison skipped; file not found: {parc_path}")
    plot_plv_heatmap(plv, parcel_meta, args.out_dir / "parcel_plv_heatmap.png", network_order)
    plot_network_matrix(summary, args.out_dir / "network_plv_matrix.png")
    if phase_corr is not None and phase_corr_summary is not None:
        plot_fc_heatmap(
            phase_corr,
            parcel_meta,
            args.out_dir / "parcel_phase_corr_heatmap.png",
            "Parcel phase time-series correlation",
            "Pearson r",
            vmin=-1,
            vmax=1,
            center=0,
        )
        plot_network_matrix(
            phase_corr_summary,
            args.out_dir / "network_phase_corr_matrix.png",
            title="Within/between network phase correlation",
            label="Mean Pearson r",
            vmin=-1,
            vmax=1,
            center=0,
        )

    print(f"Saved outputs to: {args.out_dir}")
    print(f"Phase cube shape: {phase_cube.shape}")
    print(f"Projected parcels: {len(projected_ids)}; kept parcels: {len(parcel_meta)}")
    if comparison_path is not None:
        print(f"Comparison plot: {comparison_path}")
    print(summary.sort_values(["type", "network_a", "network_b"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
