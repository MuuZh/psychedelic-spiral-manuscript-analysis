from __future__ import annotations

import math
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


def compute_curl_maps(phase_cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_x = np.full_like(phase_cube, np.nan, dtype=np.float64)
    grad_y = np.full_like(phase_cube, np.nan, dtype=np.float64)
    grad_x[:, 1:-1, :] = (phase_cube[:, 2:, :] - phase_cube[:, :-2, :]) / 2.0
    grad_y[1:-1, :, :] = (phase_cube[2:, :, :] - phase_cube[:-2, :, :]) / 2.0
    curl = grad_x - grad_y
    return curl, np.abs(curl)


def run_curl_spatial(cfg: Config, bundle_df: pd.DataFrame, summary: List[dict]) -> None:
    if bundle_df.empty:
        return
    logging_info = f"Curl spatial: {len(bundle_df)} bundles"
    records = []
    heat_sums: Dict[tuple, np.ndarray] = {}
    heat_counts: Dict[tuple, int] = {}

    target_groups = {cfg.group_pcb, cfg.group_drug}
    present = set(bundle_df["group"].unique())
    active_groups = present if len(
        present & target_groups) < 2 else target_groups
    df = bundle_df[bundle_df["group"].isin(active_groups)]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="curl per subject", file=sys.stdout, dynamic_ncols=True):
        bundle_dir = resolve_bundle_dir(row, cfg)
        cube_path = bundle_dir / "phase_cube.npy" if bundle_dir and (
            bundle_dir / "phase_cube.npy").exists() else row.get("phase_cube", Path(""))
        if not cube_path.exists():
            records.append({"group": row["group"], "subid": row["subid"], "hemisphere": row["hemisphere"],
                           "curl_mean_abs": math.nan, "curl_p95_abs": math.nan, "curl_p5_abs": math.nan, "curl_p95_p5_diff": math.nan, "curl_var_abs": math.nan})
            continue
        cube = np.load(cube_path, mmap_mode="r")
        _, curl_abs = compute_curl_maps(cube)
        finite = np.isfinite(curl_abs)
        if not finite.any():
            records.append({"group": row["group"], "subid": row["subid"], "hemisphere": row["hemisphere"],
                           "curl_mean_abs": math.nan, "curl_p95_abs": math.nan, "curl_p5_abs": math.nan, "curl_p95_p5_diff": math.nan, "curl_var_abs": math.nan})
            continue
        with np.errstate(invalid="ignore"):
            curl_mean_map = np.nanmean(curl_abs, axis=2)
        p95 = float(np.nanpercentile(curl_abs, 95))
        p5 = float(np.nanpercentile(curl_abs, 5))
        map_flat = curl_mean_map[np.isfinite(curl_mean_map)]
        if map_flat.size == 0:
            map_mean = math.nan
            map_p95 = math.nan
            map_p5 = math.nan
            map_var = math.nan
        else:
            map_mean = float(np.nanmean(map_flat))
            map_p95 = float(np.nanpercentile(map_flat, 95))
            map_p5 = float(np.nanpercentile(map_flat, 5))
            map_var = float(np.nanvar(map_flat))
        records.append({
            "group": row["group"], "subid": row["subid"], "hemisphere": row["hemisphere"],
            "curl_mean_abs": float(np.nanmean(curl_abs)),
            "curl_p95_abs": p95,
            "curl_p5_abs": p5,
            "curl_p95_p5_diff": p95 - p5 if (not math.isnan(p95) and not math.isnan(p5)) else math.nan,
            "curl_var_abs": float(np.nanvar(curl_abs)),
            "curl_map_mean_abs": map_mean,
            "curl_map_p95_abs": map_p95,
            "curl_map_p5_abs": map_p5,
            "curl_map_p95_p5_diff": map_p95 - map_p5 if (not math.isnan(map_p95) and not math.isnan(map_p5)) else math.nan,
            "curl_map_var_abs": map_var,
        })
        key = (row["hemisphere"], row["group"])
        heat_sums.setdefault(key, np.zeros_like(
            curl_mean_map, dtype=np.float64))
        heat_counts.setdefault(key, 0)
        if heat_sums[key].shape == curl_mean_map.shape:
            heat_sums[key] += np.nan_to_num(curl_mean_map, nan=0.0)
            heat_counts[key] += 1

    subj_df = (
        pd.DataFrame(records)
        .groupby(["group", "subid", "hemisphere"], as_index=False)[
            [
                "curl_mean_abs",
                "curl_p95_abs",
                "curl_p5_abs",
                "curl_p95_p5_diff",
                "curl_var_abs",
                "curl_map_mean_abs",
                "curl_map_p95_abs",
                "curl_map_p5_abs",
                "curl_map_p95_p5_diff",
                "curl_map_var_abs",
            ]
        ]
        .mean()
    )
    out_dir = cfg.output_root / cfg.results_prefix / "curl_spatial"
    write_table(subj_df, out_dir / "per_subject.csv")

    metrics = [
        "curl_mean_abs",
        "curl_p95_abs",
        "curl_p5_abs",
        "curl_p95_p5_diff",
        "curl_var_abs",
        "curl_map_mean_abs",
        "curl_map_p95_abs",
        "curl_map_p5_abs",
        "curl_map_p95_p5_diff",
        "curl_map_var_abs",
    ]

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
            summary.append({"section": "curl_spatial", "metric": metric, "hemisphere": hemi,
                           "comparison": "paired_drug_vs_pcb", **paired_t(drug, pcb)})

    for hemi in ["left", "right"]:
        hemi_df = subj_df[subj_df["hemisphere"] == hemi]
        for metric in metrics:
            drug = hemi_df[hemi_df["group"] == cfg.group_drug][metric]
            pcb = hemi_df[hemi_df["group"] == cfg.group_pcb][metric]
            summary.append({"section": "curl_spatial", "metric": metric, "hemisphere": hemi,
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
        fig.suptitle(f"Curl stats: {metric}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, out_dir / f"violin_{metric}.png", cfg.save_plots)
        for hemi in ["left", "right"]:
            tidy = subj_df[subj_df["hemisphere"] ==
                           hemi][["subid", "group", metric]]
            plot_paired_violin(
                tidy, metric, hemi, f"Curl {metric} paired", out_dir / f"paired_{metric}_{hemi}.png", cfg)

    for hemi in ["left", "right"]:
        for grp in active_groups:
            key = (hemi, grp)
            if key in heat_sums and heat_counts.get(key, 0) > 0:
                mean_map = heat_sums[key] / max(heat_counts[key], 1)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(mean_map, cmap="magma")
                # invert y-axis to match brain orientation conventions
                ax.invert_yaxis()
                ax.set_title(f"Mean |curl| ({grp}, {hemi})")
                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.tight_layout()
                save_fig(fig, out_dir /
                         f"heatmap_{grp}_{hemi}.png", cfg.save_plots)
