from __future__ import annotations

import math
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .config import Config
from .utils import build_group_summary_df, resolve_bundle_dir, save_fig, write_table, paired_t, unpaired_t, plot_paired_violin


def run_vortex_occupancy(cfg: Config, bundle_df: pd.DataFrame, summary: List[dict]) -> None:
    if bundle_df.empty:
        return
    EXCLUDE_ZERO = True
    target_groups = {cfg.group_pcb, cfg.group_drug}
    present = set(bundle_df["group"].unique())
    active_groups = present if len(
        present & target_groups) < 2 else target_groups
    df = bundle_df[bundle_df["group"].isin(active_groups)]

    records = []
    heat_sums: Dict[tuple, np.ndarray] = {}
    heat_counts: Dict[tuple, int] = {}

    def build_occupancy(bundle_dir: Path) -> tuple[np.ndarray, int]:
        meta_path = bundle_dir / "metadata.json"
        fi = bundle_dir / "frame_index.parquet"
        coords_path = bundle_dir / "coords.feather"
        if not (meta_path.exists() and fi.exists() and coords_path.exists()):
            return np.array([]), 0
        meta = json.loads(meta_path.read_text())
        h = int(meta.get("grid_height", 0))
        w = int(meta.get("grid_width", 0))
        frame_count_meta = int(meta.get("frame_count", 0))
        frame_index = pd.read_parquet(
            fi, columns=["abs_time", "coord_start", "coord_end"])
        n_frames = int(frame_index["abs_time"].nunique())
        frame_count = frame_count_meta if frame_count_meta > 0 else n_frames
        if h <= 0 or w <= 0 or frame_count <= 0:
            return np.array([]), 0
        coords_df = pd.read_feather(coords_path, columns=["y", "x"])
        coords = coords_df.to_numpy()
        occ = np.zeros((h, w), dtype=np.int64)
        for _, frame_rows in frame_index.groupby("abs_time", sort=True):
            slices = []
            for _, r in frame_rows.iterrows():
                s = int(r["coord_start"])
                e = int(r["coord_end"])
                if e > s:
                    slices.append(coords[s:e])
            if not slices:
                continue
            pts = slices[0] if len(
                slices) == 1 else np.concatenate(slices, axis=0)
            if pts.size == 0:
                continue
            uniq = np.unique(pts, axis=0)
            ys = uniq[:, 0].astype(int)
            xs = uniq[:, 1].astype(int)
            valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            occ[ys[valid], xs[valid]] += 1
        return occ, frame_count

    for _, row in tqdm(df.iterrows(), total=len(df), desc="occupancy per subject", file=sys.stdout, dynamic_ncols=True):
        bundle_dir = resolve_bundle_dir(row, cfg)
        if not bundle_dir or not bundle_dir.exists():
            records.append({"group": row["group"], "subid": row["subid"], "hemisphere": row["hemisphere"],
                           "occupancy_mean": math.nan, "occupancy_p95": math.nan, "occupancy_p5": math.nan, "occupancy_p95_p5_diff": math.nan})
            continue
        occ_counts, frames = build_occupancy(bundle_dir)
        if occ_counts.size == 0 or frames <= 0:
            records.append({"group": row["group"], "subid": row["subid"], "hemisphere": row["hemisphere"],
                           "occupancy_mean": math.nan, "occupancy_p95": math.nan, "occupancy_p5": math.nan, "occupancy_p95_p5_diff": math.nan})
            continue
        occ_frac = occ_counts.astype(float) / frames
        if cfg.reuse_cache:
            np.save(bundle_dir / "vortex_occupancy.npy", occ_frac)
        flat = occ_frac[np.isfinite(occ_frac)]
        if EXCLUDE_ZERO:
            flat = flat[flat != 0]
        if flat.size == 0:
            p95 = p5 = math.nan
            occ_mean = math.nan
        else:
            p95 = float(np.nanpercentile(flat, 95))
            p5 = float(np.nanpercentile(flat, 5))
            occ_mean = float(flat.mean())
        records.append({
            "group": row["group"], "subid": row["subid"], "hemisphere": row["hemisphere"],
            "occupancy_mean": occ_mean if EXCLUDE_ZERO else float(np.nanmean(occ_frac)),
            "occupancy_p95": p95,
            "occupancy_p5": p5,
            "occupancy_p95_p5_diff": p95 - p5 if (not math.isnan(p95) and not math.isnan(p5)) else math.nan,
        })
        key = (row["hemisphere"], row["group"])
        heat_sums.setdefault(key, np.zeros_like(occ_frac, dtype=np.float64))
        heat_counts.setdefault(key, 0)
        if heat_sums[key].shape == occ_frac.shape:
            heat_sums[key] += np.nan_to_num(occ_frac, nan=0.0)
            heat_counts[key] += 1

    subj_df = (
        pd.DataFrame(records)
        .groupby(["group", "subid", "hemisphere"], as_index=False)[
            ["occupancy_mean", "occupancy_p95",
                "occupancy_p5", "occupancy_p95_p5_diff"]
        ]
        .mean()
    )
    out_dir = cfg.output_root / cfg.results_prefix / "vortex_occupancy"
    write_table(subj_df, out_dir / "per_subject.csv")

    metrics = ["occupancy_mean", "occupancy_p95",
               "occupancy_p5", "occupancy_p95_p5_diff"]

    for hemi in ["left", "right"]:
        pivot = subj_df[subj_df["hemisphere"] == hemi].pivot(
            index="subid", columns="group", values=metrics)
        pivot.columns = ["_".join(col) for col in pivot.columns]
        pivot = pivot.dropna()
        for metric in metrics:
            drug = pivot.get(f"{metric}_{cfg.group_drug}",
                             pd.Series(dtype=float))
            pcb = pivot.get(f"{metric}_{cfg.group_pcb}",
                            pd.Series(dtype=float))
            summary.append({"section": "vortex_occupancy", "metric": metric, "hemisphere": hemi,
                            "comparison": "paired_drug_vs_pcb", **paired_t(drug, pcb)})

    for hemi in ["left", "right"]:
        hemi_df = subj_df[subj_df["hemisphere"] == hemi]
        for metric in metrics:
            drug = hemi_df[hemi_df["group"] == cfg.group_drug][metric]
            pcb = hemi_df[hemi_df["group"] == cfg.group_pcb][metric]
            summary.append({"section": "vortex_occupancy", "metric": metric, "hemisphere": hemi,
                            "comparison": "unpaired_drug_vs_pcb", **unpaired_t(drug, pcb)})

    write_table(build_group_summary_df(subj_df, metrics, cfg), out_dir / "group_summary.csv")

    sns.set_theme(style="whitegrid", context="talk")
    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, hemi in zip(axes, ["left", "right"]):
            tidy = subj_df[subj_df["hemisphere"] == hemi]
            sns.violinplot(data=tidy, x="group", y=metric, order=[
                           cfg.group_pcb, cfg.group_drug], ax=ax, cut=0)
            ax.set_title(f"{metric} ({hemi})")
        fig.suptitle(f"Vortex occupancy: {metric}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, out_dir / f"violin_{metric}.png", cfg.save_plots)
        for hemi in ["left", "right"]:
            tidy = subj_df[subj_df["hemisphere"] ==
                           hemi][["subid", "group", metric]]
            plot_paired_violin(
                tidy, metric, hemi, f"Vortex {metric} paired", out_dir / f"paired_{metric}_{hemi}.png", cfg)

    for hemi in ["left", "right"]:
        for grp in active_groups:
            key = (hemi, grp)
            if key in heat_sums and heat_counts.get(key, 0) > 0:
                mean_map = heat_sums[key] / max(heat_counts[key], 1)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(mean_map, cmap="magma")
                ax.invert_yaxis()
                ax.set_title(f"Occupancy fraction ({grp}, {hemi})")
                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.tight_layout()
                save_fig(fig, out_dir /
                         f"heatmap_{grp}_{hemi}.png", cfg.save_plots)
