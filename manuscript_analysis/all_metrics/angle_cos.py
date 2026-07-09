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

try:
    from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))
    from matphase.detect.phase_field import compute_phase_gradient, normalize_vector_field

from .config import Config
from .utils import (
    build_group_summary_df,
    paired_t,
    plot_paired_violin,
    resolve_reference_gmap,
    save_fig,
    unpaired_t,
    write_table,
)

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
    return gx, gy, gmag, valid


def weighted_cos2_alignment(
    ref_gx: np.ndarray,
    ref_gy: np.ndarray,
    ref_mag: np.ndarray,
    phase_ux: np.ndarray,
    phase_uy: np.ndarray,
    phase_mag: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    ref_norm = np.hypot(ref_gx, ref_gy)
    dot = ref_gx[:, :, None] * phase_ux + ref_gy[:, :, None] * phase_uy
    cos_theta = np.divide(
        dot,
        ref_norm[:, :, None],
        out=np.full(phase_ux.shape, np.nan, dtype=np.float64),
        where=mask,
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    cross_z = ref_gx[:, :, None] * phase_uy - ref_gy[:, :, None] * phase_ux
    angle = np.where(cross_z < 0, -angle, angle)
    cos2 = np.cos(2.0 * angle)
    weights = ref_mag[:, :, None] * phase_mag
    good = mask & np.isfinite(cos2) & np.isfinite(weights)
    denom = float(np.sum(weights[good])) if np.any(good) else 0.0
    mean_val = float(np.sum(weights[good] * cos2[good]) / denom) if denom > 0 else math.nan
    return mean_val, cos2[good]


def mean_alignment_and_samples(
    cube_path: Path,
    ref_gx: np.ndarray,
    ref_gy: np.ndarray,
    ref_mag: np.ndarray,
    ref_mask: np.ndarray,
    sample_per_bundle: int,
) -> tuple[float, np.ndarray]:
    cube = np.load(cube_path, mmap_mode="r")
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D phase cube at {cube_path}, got {cube.shape}")
    grad_x, grad_y = compute_phase_gradient(np.asarray(cube), spacing=1.0, show_progress=False)
    phase_ux, phase_uy, phase_mag = normalize_vector_field(grad_x, grad_y)
    mask = (
        ref_mask[:, :, None]
        & np.isfinite(phase_ux)
        & np.isfinite(phase_uy)
        & np.isfinite(phase_mag)
        & (phase_mag > 0)
        & (np.hypot(ref_gx, ref_gy)[:, :, None] > 0)
    )
    if not np.any(mask):
        return math.nan, np.array([], dtype=float)
    mean_val, finite = weighted_cos2_alignment(ref_gx, ref_gy, ref_mag, phase_ux, phase_uy, phase_mag, mask)
    if sample_per_bundle > 0 and finite.size:
        take = min(sample_per_bundle, finite.size)
        idx = rng.choice(finite.size, size=take, replace=False)
        samples = finite[idx]
    else:
        samples = np.array([], dtype=float)
    return mean_val, samples


def run_weighted_mean_cos2_alignment(cfg: Config, bundle_df: pd.DataFrame, summary: List[Dict]) -> None:
    df = bundle_df[bundle_df["group"].isin({cfg.group_pcb, cfg.group_drug})].copy()
    records = []
    samples_by_group: Dict[tuple, List[np.ndarray]] = {}
    ref_cache: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="weighted cos2 cubes", file=sys.stdout, dynamic_ncols=True):
        hemi = str(row["hemisphere"])
        ref_path = resolve_reference_gmap(cfg, hemi)
        if ref_path is None or not ref_path.exists():
            records.append(
                {
                    "group": row["group"],
                    "subid": row["subid"],
                    "hemisphere": hemi,
                    "weighted_mean_cos2_alignment": math.nan,
                }
            )
            continue
        if hemi not in ref_cache:
            ref_cache[hemi] = compute_reference_gradient(ref_path)
        gx, gy, gmag, valid = ref_cache[hemi]
        cube_path = Path(row["phase_cube"])
        if not cube_path.exists():
            records.append(
                {
                    "group": row["group"],
                    "subid": row["subid"],
                    "hemisphere": hemi,
                    "weighted_mean_cos2_alignment": math.nan,
                }
            )
            continue
        mean_val, samples = mean_alignment_and_samples(cube_path, gx, gy, gmag, valid, ANGLE_SAMPLE_PER_BUNDLE)
        records.append(
            {
                "group": row["group"],
                "subid": row["subid"],
                "hemisphere": hemi,
                "weighted_mean_cos2_alignment": mean_val,
            }
        )
        if samples.size:
            samples_by_group.setdefault((row["group"], hemi), []).append(samples)

    subj_df = (
        pd.DataFrame(records)
        .groupby(["group", "subid", "hemisphere"], as_index=False)["weighted_mean_cos2_alignment"]
        .mean()
    )
    out_dir = cfg.output_root / cfg.results_prefix / "weighted_mean_cos2_alignment"
    write_table(subj_df, out_dir / "per_subject.csv")

    order = [cfg.group_pcb, cfg.group_drug]
    for hemi in ["left", "right"]:
        pivot = subj_df[subj_df["hemisphere"] == hemi].pivot(index="subid", columns="group", values="weighted_mean_cos2_alignment")
        pivot = pivot.dropna()
        summary.append(
            {
                "section": "weighted_mean_cos2_alignment",
                "metric": "weighted_mean_cos2_alignment",
                "hemisphere": hemi,
                "comparison": "paired_drug_vs_pcb",
                **paired_t(pivot.get(cfg.group_drug, pd.Series(dtype=float)), pivot.get(cfg.group_pcb, pd.Series(dtype=float))),
            }
        )
        drug = subj_df[(subj_df["hemisphere"] == hemi) & (subj_df["group"] == cfg.group_drug)]["weighted_mean_cos2_alignment"]
        pcb = subj_df[(subj_df["hemisphere"] == hemi) & (subj_df["group"] == cfg.group_pcb)]["weighted_mean_cos2_alignment"]
        summary.append(
            {
                "section": "weighted_mean_cos2_alignment",
                "metric": "weighted_mean_cos2_alignment",
                "hemisphere": hemi,
                "comparison": "unpaired_drug_vs_pcb",
                **unpaired_t(drug, pcb),
            }
        )

    write_table(build_group_summary_df(subj_df, ["weighted_mean_cos2_alignment"], cfg), out_dir / "group_summary.csv")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, hemi in zip(axes, ["left", "right"]):
        tidy = subj_df[subj_df["hemisphere"] == hemi]
        sns.violinplot(data=tidy, x="group", y="weighted_mean_cos2_alignment", order=order, ax=ax, cut=0)
        ax.set_title(f"weighted mean cos(2theta) ({hemi})")
    fig.suptitle("Weighted mean cos(2theta) vs reference")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_dir / "violin_weighted_mean_cos2_alignment.png", cfg.save_plots)
    for hemi in ["left", "right"]:
        tidy = subj_df[subj_df["hemisphere"] == hemi][["subid", "group", "weighted_mean_cos2_alignment"]]
        plot_paired_violin(
            tidy,
            "weighted_mean_cos2_alignment",
            hemi,
            "weighted mean cos(2theta) paired",
            out_dir / f"paired_weighted_mean_cos2_alignment_{hemi}.png",
            cfg,
        )

    # Optional distribution plots per group/hemi.
    for grp in order:
        for hemi in ["left", "right"]:
            key = (grp, hemi)
            if key not in samples_by_group:
                continue
            samples = np.concatenate(samples_by_group[key], axis=0)
            if samples.size == 0:
                continue
            if samples.size > MAX_ANGLE_SAMPLES_PER_GROUP:
                idx = rng.choice(samples.size, size=MAX_ANGLE_SAMPLES_PER_GROUP, replace=False)
                samples = samples[idx]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(samples, bins=50, stat="density", ax=ax)
            ax.set_xlim(-1.0, 1.0)
            ax.set_xlabel("cos(2theta)")
            ax.set_title(f"cos(2theta) dist ({grp}, {hemi})")
            fig.tight_layout()
            save_fig(fig, out_dir / f"hist_cos2_alignment_{grp}_{hemi}.png", cfg.save_plots)


def run_angle_diff_abs_cos(cfg: Config, bundle_df: pd.DataFrame, summary: List[Dict]) -> None:
    run_weighted_mean_cos2_alignment(cfg, bundle_df, summary)
