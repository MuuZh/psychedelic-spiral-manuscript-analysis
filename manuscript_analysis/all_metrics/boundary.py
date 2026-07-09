from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.ndimage import binary_closing, binary_dilation, label
from tqdm import tqdm

from .config import Config
from .utils import build_group_summary_df, get_palette, paired_t, resolve_bundle_dir, save_fig, unpaired_t, write_table, plot_paired_violin

MAGNITUDE_THRESHOLD = 0.2
BOUNDARY_THICKNESS = 2
PARCEL_EDGE_THICKNESS = 0
MIN_REGION_SIZE = 10


def _load_parcellations(cfg: Config) -> Dict[str, np.ndarray]:
    cfg_path = Path(cfg.parcellation_config)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    paths = cfg_yaml.get("paths", {})
    data_dir = Path(cfg_path.parent.parent) / paths.get("data_dir", ".")
    left = paths.get("parcellation_left")
    right = paths.get("parcellation_right")
    parcellations = {}
    if left:
        parcellations["left"] = np.load(data_dir / left).astype(float)
    if right:
        parcellations["right"] = np.load(data_dir / right).astype(float)
    return parcellations


def _trim_last_row(arr: np.ndarray) -> np.ndarray:
    return arr[:-1, :] if arr.shape[0] > 1 else arr


def _compute_parcel_edges(parcellation: np.ndarray) -> np.ndarray:
    labels = parcellation.copy()
    valid = np.isfinite(labels)
    edges = np.zeros_like(labels, dtype=bool)
    diff_right = (labels[:, :-1] != labels[:, 1:]
                  ) & valid[:, :-1] & valid[:, 1:]
    edges[:, :-1] |= diff_right
    edges[:, 1:] |= diff_right
    diff_down = (labels[:-1, :] != labels[1:, :]
                 ) & valid[:-1, :] & valid[1:, :]
    edges[:-1, :] |= diff_down
    edges[1:, :] |= diff_down
    return edges


def _align_phase_slice(phase_slice: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if phase_slice.shape == target_shape:
        return phase_slice
    if phase_slice.shape[0] == target_shape[0] + 1 and phase_slice.shape[1] == target_shape[1]:
        return phase_slice[:-1, :]
    raise ValueError(
        f"Phase slice shape {phase_slice.shape} does not match {target_shape}.")


def _compute_boundary_mask(
    phase_slice: np.ndarray,
    valid_mask: np.ndarray,
    parcel_edges: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    gy, gx = np.gradient(phase_slice.astype(float))
    gx = gx.copy()
    gy = gy.copy()
    gx[~valid_mask] = np.nan
    gy[~valid_mask] = np.nan

    magnitude = np.hypot(gx, gy)
    boundary_mask = magnitude > MAGNITUDE_THRESHOLD
    nan_mask = np.isnan(gx) | np.isnan(gy)
    boundary_mask = boundary_mask | nan_mask
    if BOUNDARY_THICKNESS > 0:
        boundary_mask = binary_dilation(
            boundary_mask, iterations=BOUNDARY_THICKNESS)
    if parcel_edges is not None:
        edges = parcel_edges
        if PARCEL_EDGE_THICKNESS > 0:
            edges = binary_dilation(
                parcel_edges, iterations=PARCEL_EDGE_THICKNESS)
        boundary_mask = boundary_mask | edges

    regions = ~boundary_mask
    regions = binary_closing(regions, iterations=2)
    labeled_regions, initial_count = label(regions)
    valid_regions = np.zeros_like(labeled_regions, dtype=np.int32)
    new_label = 1
    region_count = 0
    for region_num in range(1, initial_count + 1):
        region_mask = labeled_regions == region_num
        if np.sum(region_mask) >= MIN_REGION_SIZE:
            valid_regions[region_mask] = new_label
            new_label += 1
            region_count += 1
    boundary_mask = (valid_regions == 0) & valid_mask
    return boundary_mask, region_count


def run_boundary_regions(cfg: Config, bundle_df: pd.DataFrame, summary: List[Dict]) -> None:
    frame_index_path = cfg.combined_dir / "combined_frame_index.parquet"
    if not frame_index_path.exists():
        return
    parcellations = _load_parcellations(cfg)
    if not parcellations:
        return

    frame_index = pd.read_parquet(frame_index_path, columns=[
                                  "group", "subid", "hemisphere", "bundle_dir"])
    bundle_map = frame_index.drop_duplicates(
        subset=["group", "subid", "hemisphere", "bundle_dir"])

    target_groups = {cfg.group_pcb, cfg.group_drug}
    present = set(bundle_map["group"].unique())
    active_groups = present if len(
        present & target_groups) < 2 else target_groups
    bundle_map = bundle_map[bundle_map["group"].isin(active_groups)]

    out_dir = cfg.output_root / cfg.results_prefix / "boundary_regions"

    subj_df: pd.DataFrame | None = None
    if cfg.reuse_cache:
        cached = out_dir / "per_subject.csv"
        if cached.exists():
            subj_df = pd.read_csv(cached)
            if not subj_df.empty:
                active_groups = set(subj_df["group"].unique())
                bundle_map = bundle_map[bundle_map["group"].isin(active_groups)]

    records: List[Dict] = []
    heat_sums: Dict[tuple, np.ndarray] = {}
    heat_counts: Dict[tuple, int] = {}
    saved_maps: Dict[tuple, dict] = {}

    if subj_df is None:
        for hemi, parcellation in parcellations.items():
            parcellation = _trim_last_row(parcellation)
            valid_mask = np.isfinite(parcellation)
            parcel_edges = _compute_parcel_edges(parcellation)

            hemi_df = bundle_map[bundle_map["hemisphere"] == hemi]
            if hemi_df.empty:
                continue

            for grp, grp_df in hemi_df.groupby("group"):
                grp_sum = np.zeros_like(valid_mask, dtype=np.float64)
                grp_frames = 0
                for subid, sub_df in grp_df.groupby("subid"):
                    row = sub_df.iloc[0]
                    bdir = resolve_bundle_dir(row, cfg)
                    if not bdir or not (bdir / "phase_cube.npy").exists():
                        records.append(
                            {"group": grp, "subid": str(subid), "hemisphere": hemi,
                             "mean_boundary_count": math.nan, "mean_region_count": math.nan, "n_frames": 0}
                        )
                        continue
                    cube = np.load(bdir / "phase_cube.npy", mmap_mode="r")
                    if cube.ndim != 3:
                        continue
                    boundary_counts: List[float] = []
                    region_counts: List[int] = []
                    for fidx in tqdm(range(cube.shape[-1]), desc=f"{hemi}-{grp}-sub{subid}", leave=False, file=sys.stdout, dynamic_ncols=True):
                        slice_frame = _align_phase_slice(
                            cube[:, :, fidx], valid_mask.shape)
                        boundary_mask, region_count = _compute_boundary_mask(
                            slice_frame, valid_mask=valid_mask, parcel_edges=parcel_edges
                        )
                        boundary_counts.append(float(boundary_mask.sum()))
                        region_counts.append(int(region_count))
                        grp_sum += boundary_mask.astype(np.float64)
                        grp_frames += 1

                    records.append(
                        {
                            "group": grp,
                            "subid": str(subid),
                            "hemisphere": hemi,
                            "mean_boundary_count": float(np.mean(boundary_counts)) if boundary_counts else math.nan,
                            "mean_region_count": float(np.mean(region_counts)) if region_counts else math.nan,
                            "n_frames": len(boundary_counts),
                        }
                    )

                key = (hemi, grp)
                heat_sums[key] = grp_sum
                heat_counts[key] = grp_frames
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    out_dir / f"boundary_maps_{hemi}_{grp}.npz",
                    sum_map=grp_sum,
                    avg_map=(grp_sum / grp_frames) if grp_frames > 0 else grp_sum,
                    frames=grp_frames,
                    mask=valid_mask,
                )
                saved_maps[key] = {"sum": grp_sum, "frames": grp_frames, "mask": valid_mask}

        subj_df = pd.DataFrame(records)
        if subj_df.empty:
            return
        write_table(subj_df, out_dir / "per_subject.csv")

    # Load any previously saved maps so plotting can reuse cached results on reruns
    for npz_path in out_dir.glob("boundary_maps_*.npz"):
        stem = npz_path.stem  # boundary_maps_<hemi>_<group>
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        hemi = parts[2]
        grp = parts[3]
        key = (hemi, grp)
        if key in saved_maps:
            continue  # prefer fresh compute
        data = np.load(npz_path)
        frames = int(data.get("frames", 0))
        if frames <= 0:
            continue
        saved_maps[key] = {
            "sum": data["sum_map"],
            "frames": frames,
            "mask": data["mask"],
        }
    for key, val in saved_maps.items():
        if key not in heat_sums and val["frames"] > 0:
            heat_sums[key] = val["sum"]
            heat_counts[key] = val["frames"]

    metrics = ["mean_boundary_count", "mean_region_count"]

    for hemi in ["left", "right"]:
        pivot = subj_df[subj_df["hemisphere"] == hemi].pivot(
            index="subid", columns="group", values=metrics)
        if pivot.empty:
            continue
        pivot.columns = ["_".join(col) for col in pivot.columns]
        for metric in metrics:
            drug = pivot.get(f"{metric}_{cfg.group_drug}",
                             pd.Series(dtype=float))
            pcb = pivot.get(f"{metric}_{cfg.group_pcb}",
                            pd.Series(dtype=float))
            summary.append({"section": "boundary_regions", "metric": metric, "hemisphere": hemi,
                            "comparison": "paired_drug_vs_pcb", **paired_t(drug, pcb)})

    for hemi in ["left", "right"]:
        hemi_df = subj_df[subj_df["hemisphere"] == hemi]
        for metric in metrics:
            drug = hemi_df[hemi_df["group"] == cfg.group_drug][metric]
            pcb = hemi_df[hemi_df["group"] == cfg.group_pcb][metric]
            summary.append({"section": "boundary_regions", "metric": metric, "hemisphere": hemi,
                            "comparison": "unpaired_drug_vs_pcb", **unpaired_t(drug, pcb)})

    write_table(build_group_summary_df(subj_df, metrics, cfg), out_dir / "group_summary.csv")

    sns.set_theme(style="whitegrid", context="talk")
    pal = get_palette(cfg)
    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, hemi in zip(axes, ["left", "right"]):
            tidy = subj_df[subj_df["hemisphere"] == hemi]
            sns.violinplot(
                data=tidy, x="group", y=metric, order=[cfg.group_pcb, cfg.group_drug],
                palette=pal, cut=0, ax=ax
            )
            ax.set_title(f"{metric} ({hemi})")
        fig.suptitle(f"Boundary regions: {metric}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, out_dir / f"violin_{metric}.png", cfg.save_plots)
        # Paired violin with subject dots and connecting lines (per hemisphere)
        for hemi in ["left", "right"]:
            tidy = subj_df[subj_df["hemisphere"] == hemi][["subid", "group", metric]]
            plot_paired_violin(
                tidy,
                metric,
                hemi,
                f"Boundary {metric} paired",
                out_dir / f"paired_{metric}_{hemi}.png",
                cfg,
            )

    for (hemi, grp), sum_map in heat_sums.items():
        frames = heat_counts.get((hemi, grp), 0)
        if frames <= 0:
            continue
        avg_map = sum_map / frames
        mask = np.isfinite(avg_map)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(avg_map * mask, cmap="magma")
        ax.invert_yaxis()
        ax.set_title(f"Average boundary map ({grp}, {hemi})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        save_fig(fig, out_dir /
                 f"boundary_avg_{hemi}_{grp}.png", cfg.save_plots)
