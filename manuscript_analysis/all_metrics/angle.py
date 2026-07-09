from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .config import Config
from .utils import save_fig, write_table, plot_paired_violin, paired_t, unpaired_t, summarize_series, add_tests, resolve_reference_gmap

ANGLE_SAMPLE_PER_BUNDLE = 50000
MAX_ANGLE_SAMPLES_PER_GROUP = 200000
rng = np.random.default_rng(42)


def compute_reference_gradient(path: Path):
    gmap = np.load(path).astype(float)
    gy, gx = np.gradient(gmap)
    gmag = np.hypot(gx, gy)
    valid = np.isfinite(gmap) & np.isfinite(gx) & np.isfinite(gy) & (gmag > 0)
    valid[:2, :] = False
    valid[-2:, :] = False
    valid[:, :2] = False
    valid[:, -2:] = False
    return gx, gy, valid


def signed_angle(ref_gx: np.ndarray, ref_gy: np.ndarray, tgt_gx: np.ndarray, tgt_gy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ref_mag = np.hypot(ref_gx, ref_gy)
    tgt_mag = np.hypot(tgt_gx, tgt_gy)
    denom = ref_mag * tgt_mag
    cos_theta = np.divide(ref_gx * tgt_gx + ref_gy * tgt_gy, denom, out=np.full_like(ref_gx, np.nan), where=mask & (denom > 0))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    cross_z = ref_gx * tgt_gy - ref_gy * tgt_gx
    angle = np.where(cross_z < 0, -angle, angle)
    angle[~mask] = np.nan
    return angle


def mean_angle_and_samples(cube_path: Path, ref_gx: np.ndarray, ref_gy: np.ndarray, ref_mask: np.ndarray, sample_per_bundle: int) -> tuple[float, np.ndarray]:
    cube = np.load(cube_path, mmap_mode="r")
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D phase cube at {cube_path}, got {cube.shape}")
    means = []
    samples_all = []
    for t in range(cube.shape[2]):
        sl = cube[:, :, t]
        py, px = np.gradient(sl)
        tgt_mag = np.hypot(px, py)
        mask = ref_mask & np.isfinite(sl) & np.isfinite(px) & np.isfinite(py) & (tgt_mag > 0)
        if not mask.any():
            continue
        angles = signed_angle(ref_gx, ref_gy, px, py, mask)
        finite = angles[np.isfinite(angles)]
        if finite.size == 0:
            continue
        means.append(float(np.nanmean(finite)))
        if sample_per_bundle > 0:
            take = min(sample_per_bundle, finite.size)
            idx = rng.choice(finite.size, size=take, replace=False)
            samples_all.append(finite[idx])
    mean_angle = float(np.nanmean(means)) if means else math.nan
    samples = np.concatenate(samples_all, axis=0) if samples_all else np.array([], dtype=float)
    return mean_angle, samples


def run_angle_diff(cfg: Config, bundle_df: pd.DataFrame, summary: List[Dict]) -> None:
    df = bundle_df[bundle_df["group"].isin({cfg.group_pcb, cfg.group_drug})].copy()
    records = []
    angle_samples: Dict[tuple, List[np.ndarray]] = {}
    ref_cache: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="angle cubes"):
        hemi = str(row["hemisphere"])
        ref_path = resolve_reference_gmap(cfg, hemi)
        if ref_path is None or not ref_path.exists():
            records.append({"group": row["group"], "subid": row["subid"], "hemisphere": hemi, "angle_mean": math.nan})
            continue
        if hemi not in ref_cache:
            ref_cache[hemi] = compute_reference_gradient(ref_path)
        gx, gy, valid = ref_cache[hemi]
        cube_path = Path(row["phase_cube"])
        if not cube_path.exists():
            records.append({"group": row["group"], "subid": row["subid"], "hemisphere": hemi, "angle_mean": math.nan})
            continue
        mean_angle, samples = mean_angle_and_samples(cube_path, gx, gy, valid, ANGLE_SAMPLE_PER_BUNDLE)
        records.append({"group": row["group"], "subid": row["subid"], "hemisphere": hemi, "angle_mean": mean_angle})
        if samples.size:
            angle_samples.setdefault((row["group"], hemi), []).append(samples)

    subj_df = (
        pd.DataFrame(records)
        .groupby(["group", "subid", "hemisphere"], as_index=False)["angle_mean"]
        .mean()
    )
    out_dir = cfg.output_root / cfg.results_prefix / "angle_diff"
    write_table(subj_df, out_dir / "per_subject.csv")

    order = [cfg.group_pcb, cfg.group_drug]
    for hemi in ["left", "right"]:
        pivot = subj_df[subj_df["hemisphere"] == hemi].pivot(index="subid", columns="group", values="angle_mean")
        pivot = pivot.dropna()
        summary.append({"section": "angle_diff", "metric": "angle_mean", "hemisphere": hemi, "comparison": "paired_drug_vs_pcb",
                        **paired_t(pivot.get(cfg.group_drug, pd.Series(dtype=float)), pivot.get(cfg.group_pcb, pd.Series(dtype=float)))})
        drug = subj_df[(subj_df["hemisphere"] == hemi) & (subj_df["group"] == cfg.group_drug)]["angle_mean"]
        pcb = subj_df[(subj_df["hemisphere"] == hemi) & (subj_df["group"] == cfg.group_pcb)]["angle_mean"]
        summary.append({"section": "angle_diff", "metric": "angle_mean", "hemisphere": hemi, "comparison": "unpaired_drug_vs_pcb",
                        **unpaired_t(drug, pcb)})

    rows = []
    for hemi in ["left", "right", "combined"]:
        for grp in order:
            subset = subj_df[subj_df["group"] == grp] if hemi == "combined" else subj_df[(subj_df["group"] == grp) & (subj_df["hemisphere"] == hemi)]
            stats_dict = summarize_series(subset["angle_mean"])
            stats_dict.update({"hemisphere": hemi, "group": grp, "metric": "angle_mean"})
            rows.append(stats_dict)
    # add statistical tests into the same summary CSV
    for hemi in ["left", "right"]:
        hemi_df = subj_df[subj_df["hemisphere"] == hemi]
        drug = hemi_df[hemi_df["group"] == cfg.group_drug]["angle_mean"]
        pcb = hemi_df[hemi_df["group"] == cfg.group_pcb]["angle_mean"]
        add_tests(rows, hemi, "angle_mean", drug, pcb, cfg)
    write_table(pd.DataFrame(rows), out_dir / "group_summary.csv")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, hemi in zip(axes, ["left", "right"]):
        tidy = subj_df[subj_df["hemisphere"] == hemi]
        sns.violinplot(data=tidy, x="group", y="angle_mean", order=order, ax=ax, cut=0)
        ax.set_title(f"Angle diff ({hemi})")
    fig.suptitle("Mean signed angle vs reference")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_dir / "violin_angle.png", cfg.save_plots)
    for hemi in ["left", "right"]:
        tidy = subj_df[subj_df["hemisphere"] == hemi][["subid", "group", "angle_mean"]]
        plot_paired_violin(tidy, "angle_mean", hemi, "Angle diff paired", out_dir / f"paired_angle_{hemi}.png", cfg)

    # polar
    NBINS = 100
    for grp in order:
        for hemi in ["left", "right"]:
            key = (grp, hemi)
            if key not in angle_samples:
                continue
            samples = np.concatenate(angle_samples[key], axis=0)
            if samples.size == 0:
                continue
            if samples.size > MAX_ANGLE_SAMPLES_PER_GROUP:
                idx = rng.choice(samples.size, size=MAX_ANGLE_SAMPLES_PER_GROUP, replace=False)
                samples = samples[idx]
            hist, edges = np.histogram(samples, bins=NBINS, range=(-np.pi, np.pi), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="polar")
            ax.bar(centers, hist, width=(2 * np.pi / NBINS), bottom=0.0, color="steelblue", alpha=0.7)
            ax.set_title(f"Angle polar ({grp}, {hemi})")
            fig.tight_layout()
            save_fig(fig, out_dir / f"polar_{grp}_{hemi}.png", cfg.save_plots)
