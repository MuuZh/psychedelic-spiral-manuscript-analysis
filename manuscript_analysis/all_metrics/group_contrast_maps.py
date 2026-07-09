from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from .config import Config
from .curl import compute_curl_maps
from .loaders import load_bundles
from .utils import (
    build_group_summary_df,
    save_fig,
    set_output_naming,
    setup_logging,
    write_table,
)


METRIC_SPECS = {
    "curl_mean_abs_map": {
        "title": "Mean |curl|",
        "cmap": "magma",
        "value_prefix": "curl_map",
    },
    "vortex_occupancy_map": {
        "title": "Vortex occupancy",
        "cmap": "magma",
        "value_prefix": "occupancy",
    },
}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Group contrast analysis for curl and vortex occupancy maps."
    )
    parser.add_argument("--prefix", dest="results_prefix", default=None)
    parser.add_argument("--drug-label", dest="group_drug", default=None)
    parser.add_argument("--pcb-label", dest="group_pcb", default=None)
    parser.add_argument("--detect-dir", type=Path, default=None)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--cluster-alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.results_prefix:
        cfg.results_prefix = args.results_prefix
    if args.group_drug:
        cfg.group_drug = args.group_drug
    if args.group_pcb:
        cfg.group_pcb = args.group_pcb
    if args.detect_dir:
        cfg.detect_results_dir = args.detect_dir
    if args.no_cache:
        cfg.reuse_cache = False
    if args.no_plots:
        cfg.save_plots = False
    cfg.n_permutations = max(100, int(args.n_permutations))
    cfg.cluster_alpha = float(args.cluster_alpha)
    cfg.random_seed = int(args.random_seed)
    return cfg


def _safe_abs_max(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.max(np.abs(finite)))
    return vmax if vmax > 0 else 1.0


def resolve_bundle_dir(row: pd.Series, cfg: Config) -> Path | None:
    bundle_dir = Path(str(row["bundle_dir"]))
    if bundle_dir.exists():
        return bundle_dir
    alt = Path(cfg.detect_results_dir) / bundle_dir
    return alt if alt.exists() else None


def build_occupancy_map(bundle_dir: Path) -> tuple[np.ndarray | None, int]:
    occ_path = bundle_dir / "vortex_occupancy.npy"
    if occ_path.exists():
        occ = np.load(occ_path)
        if occ.ndim == 2:
            return occ.astype(np.float64, copy=False), 0

    meta_path = bundle_dir / "metadata.json"
    fi_path = bundle_dir / "frame_index.parquet"
    coords_path = bundle_dir / "coords.feather"
    if not (meta_path.exists() and fi_path.exists() and coords_path.exists()):
        return None, 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    h = int(meta.get("grid_height", 0))
    w = int(meta.get("grid_width", 0))
    frame_count_meta = int(meta.get("frame_count", 0))
    frame_index = pd.read_parquet(fi_path, columns=["abs_time", "coord_start", "coord_end"])
    n_frames = int(frame_index["abs_time"].nunique())
    frame_count = frame_count_meta if frame_count_meta > 0 else n_frames
    if h <= 0 or w <= 0 or frame_count <= 0:
        return None, 0

    coords_df = pd.read_feather(coords_path, columns=["y", "x"])
    coords = coords_df.to_numpy()
    occ = np.zeros((h, w), dtype=np.int64)
    for _, frame_rows in frame_index.groupby("abs_time", sort=True):
        slices = []
        for _, frame_row in frame_rows.iterrows():
            start = int(frame_row["coord_start"])
            end = int(frame_row["coord_end"])
            if end > start:
                slices.append(coords[start:end])
        if not slices:
            continue
        pts = slices[0] if len(slices) == 1 else np.concatenate(slices, axis=0)
        if pts.size == 0:
            continue
        uniq = np.unique(pts, axis=0)
        ys = uniq[:, 0].astype(int)
        xs = uniq[:, 1].astype(int)
        valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        occ[ys[valid], xs[valid]] += 1

    occ_frac = occ.astype(np.float64) / float(frame_count)
    return occ_frac, frame_count


def compute_subject_maps(cfg: Config, bundles: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Dict[str, List[dict]]]]:
    scalar_rows: list[dict] = []
    map_records: Dict[str, Dict[str, List[dict]]] = {
        metric: {"left": [], "right": []} for metric in METRIC_SPECS
    }

    for _, row in bundles.iterrows():
        group = str(row["group"])
        hemi = str(row["hemisphere"]).lower()
        subid = str(row["subid"])
        bundle_dir = resolve_bundle_dir(row, cfg)
        if bundle_dir is None:
            logging.warning("Missing bundle directory for %s %s %s", group, subid, hemi)
            continue

        scalar_row = {"group": group, "subid": subid, "hemisphere": hemi}

        phase_path = bundle_dir / "phase_cube.npy"
        if phase_path.exists():
            cube = np.load(phase_path, mmap_mode="r")
            _, curl_abs = compute_curl_maps(cube)
            with np.errstate(invalid="ignore"):
                curl_map = np.nanmean(curl_abs, axis=2)
            flat = curl_map[np.isfinite(curl_map)]
            scalar_row.update(
                {
                    "curl_map_mean_abs": float(np.nanmean(flat)) if flat.size else math.nan,
                    "curl_map_p95_abs": float(np.nanpercentile(flat, 95)) if flat.size else math.nan,
                    "curl_map_p5_abs": float(np.nanpercentile(flat, 5)) if flat.size else math.nan,
                    "curl_map_p95_p5_diff": (
                        float(np.nanpercentile(flat, 95) - np.nanpercentile(flat, 5))
                        if flat.size
                        else math.nan
                    ),
                    "curl_map_var_abs": float(np.nanvar(flat)) if flat.size else math.nan,
                }
            )
            map_records["curl_mean_abs_map"][hemi].append(
                {"group": group, "subid": subid, "map": np.asarray(curl_map, dtype=np.float64)}
            )
        else:
            scalar_row.update(
                {
                    "curl_map_mean_abs": math.nan,
                    "curl_map_p95_abs": math.nan,
                    "curl_map_p5_abs": math.nan,
                    "curl_map_p95_p5_diff": math.nan,
                    "curl_map_var_abs": math.nan,
                }
            )

        occ_map, _ = build_occupancy_map(bundle_dir)
        if occ_map is not None and occ_map.size:
            occ_flat = occ_map[np.isfinite(occ_map)]
            occ_nonzero = occ_flat[occ_flat != 0]
            base = occ_nonzero if occ_nonzero.size else occ_flat
            scalar_row.update(
                {
                    "occupancy_mean": float(np.nanmean(base)) if base.size else math.nan,
                    "occupancy_p95": float(np.nanpercentile(base, 95)) if base.size else math.nan,
                    "occupancy_p5": float(np.nanpercentile(base, 5)) if base.size else math.nan,
                    "occupancy_p95_p5_diff": (
                        float(np.nanpercentile(base, 95) - np.nanpercentile(base, 5))
                        if base.size
                        else math.nan
                    ),
                    "occupancy_nonzero_fraction": float(np.count_nonzero(occ_map) / occ_map.size),
                    "occupancy_var": float(np.nanvar(base)) if base.size else math.nan,
                }
            )
            map_records["vortex_occupancy_map"][hemi].append(
                {"group": group, "subid": subid, "map": np.asarray(occ_map, dtype=np.float64)}
            )
            if cfg.reuse_cache and not (bundle_dir / "vortex_occupancy.npy").exists():
                np.save(bundle_dir / "vortex_occupancy.npy", occ_map)
        else:
            scalar_row.update(
                {
                    "occupancy_mean": math.nan,
                    "occupancy_p95": math.nan,
                    "occupancy_p5": math.nan,
                    "occupancy_p95_p5_diff": math.nan,
                    "occupancy_nonzero_fraction": math.nan,
                    "occupancy_var": math.nan,
                }
            )

        scalar_rows.append(scalar_row)

    scalar_df = pd.DataFrame(scalar_rows)
    if not scalar_df.empty:
        scalar_df = scalar_df.groupby(["group", "subid", "hemisphere"], as_index=False).mean(numeric_only=True)
    return scalar_df, map_records


def _cohens_d_unpaired(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean_diff = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    n1 = np.sum(np.isfinite(a), axis=0)
    n2 = np.sum(np.isfinite(b), axis=0)
    var1 = np.nanvar(a, axis=0, ddof=1)
    var2 = np.nanvar(b, axis=0, ddof=1)
    denom = n1 + n2 - 2
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = mean_diff / pooled
    d[(denom <= 0) | (~np.isfinite(d))] = np.nan
    return d


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=np.float64)
    qvals = np.full_like(pvals, np.nan)
    finite = np.isfinite(pvals)
    if not finite.any():
        return qvals
    p = pvals[finite]
    order = np.argsort(p)
    ranked = p[order]
    n = ranked.size
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restore = np.empty_like(q)
    restore[order] = q
    qvals[finite] = restore
    return qvals


def _cluster_stat_map(t_map: np.ndarray, threshold_t: float) -> tuple[np.ndarray, list[dict]]:
    supra = np.isfinite(t_map) & (np.abs(t_map) >= threshold_t)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labels, n_labels = ndimage.label(supra, structure=structure)
    cluster_mass = np.zeros_like(t_map, dtype=np.float64)
    clusters: list[dict] = []
    for cluster_id in range(1, n_labels + 1):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        mass = float(np.nansum(np.abs(t_map[mask])))
        sign_mean = float(np.nanmean(np.sign(t_map[mask])))
        cluster_mass[mask] = mass
        ys, xs = np.where(mask)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": int(np.count_nonzero(mask)),
                "mass": mass,
                "sign": "positive" if sign_mean >= 0 else "negative",
                "y_min": int(ys.min()),
                "y_max": int(ys.max()),
                "x_min": int(xs.min()),
                "x_max": int(xs.max()),
            }
        )
    return labels, clusters


def _paired_t_map(diff: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(diff, axis=0)
        std = np.nanstd(diff, axis=0, ddof=1)
        n = np.sum(np.isfinite(diff), axis=0).astype(np.float64)
        sem = std / np.sqrt(n)
        t_map = mean / sem
    t_map[(n < 2) | (~np.isfinite(t_map))] = np.nan
    return t_map


def _unpaired_t_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        t_map, _ = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")
    return np.asarray(t_map, dtype=np.float64)


def _cluster_permutation(
    *,
    paired: bool,
    drug_stack: np.ndarray,
    pcb_stack: np.ndarray,
    t_map: np.ndarray,
    cluster_alpha: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, float]:
    if paired:
        n_subjects = drug_stack.shape[0]
        dfree = max(n_subjects - 1, 1)
    else:
        n1 = drug_stack.shape[0]
        n2 = pcb_stack.shape[0]
        dfree = max(min(n1, n2) - 1, 1)
    threshold_t = float(stats.t.ppf(1.0 - cluster_alpha / 2.0, df=dfree))
    labels, clusters = _cluster_stat_map(t_map, threshold_t)
    if not clusters:
        empty = np.zeros_like(t_map, dtype=np.float64)
        return (
            empty.astype(np.int32),
            np.full_like(t_map, np.nan, dtype=np.float64),
            np.zeros_like(t_map, dtype=np.uint8),
            pd.DataFrame(),
            threshold_t,
        )

    max_masses = np.zeros(n_permutations, dtype=np.float64)
    if paired:
        diff = drug_stack - pcb_stack
        for idx in range(n_permutations):
            flips = rng.choice(np.array([-1.0, 1.0]), size=(diff.shape[0], 1, 1))
            perm_t = _paired_t_map(diff * flips)
            _, perm_clusters = _cluster_stat_map(perm_t, threshold_t)
            max_masses[idx] = max((c["mass"] for c in perm_clusters), default=0.0)
    else:
        combined = np.concatenate([drug_stack, pcb_stack], axis=0)
        n1 = drug_stack.shape[0]
        total = combined.shape[0]
        for idx in range(n_permutations):
            perm_idx = rng.permutation(total)
            perm_a = combined[perm_idx[:n1]]
            perm_b = combined[perm_idx[n1:]]
            perm_t = _unpaired_t_map(perm_a, perm_b)
            _, perm_clusters = _cluster_stat_map(perm_t, threshold_t)
            max_masses[idx] = max((c["mass"] for c in perm_clusters), default=0.0)

    cluster_p_map = np.full_like(t_map, np.nan, dtype=np.float64)
    cluster_sig_mask = np.zeros_like(t_map, dtype=np.uint8)
    cluster_rows: list[dict] = []
    for cluster in clusters:
        mass = cluster["mass"]
        p_cluster = float((1.0 + np.count_nonzero(max_masses >= mass)) / (n_permutations + 1.0))
        mask = labels == cluster["cluster_id"]
        cluster_p_map[mask] = p_cluster
        if p_cluster < 0.05:
            cluster_sig_mask[mask] = 1
        cluster_rows.append({**cluster, "cluster_p": p_cluster, "threshold_t": threshold_t})

    return (
        labels.astype(np.int32),
        cluster_p_map,
        cluster_sig_mask,
        pd.DataFrame(cluster_rows),
        threshold_t,
    )


def run_map_contrasts(
    cfg: Config,
    out_dir: Path,
    map_records: Dict[str, Dict[str, List[dict]]],
) -> pd.DataFrame:
    summary_rows: list[dict] = []
    cluster_rows_all: list[pd.DataFrame] = []
    map_dir = out_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(getattr(cfg, "random_seed", 42))

    for metric_name, hemi_dict in map_records.items():
        spec = METRIC_SPECS[metric_name]
        for hemi, records in hemi_dict.items():
            if not records:
                continue

            shape_counts: Dict[tuple[int, int], int] = {}
            for rec in records:
                shape_counts[tuple(rec["map"].shape)] = shape_counts.get(tuple(rec["map"].shape), 0) + 1
            ref_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
            records = [rec for rec in records if tuple(rec["map"].shape) == ref_shape]

            drug_records = [rec for rec in records if rec["group"] == cfg.group_drug]
            pcb_records = [rec for rec in records if rec["group"] == cfg.group_pcb]
            if not drug_records or not pcb_records:
                logging.warning("Skip %s %s: missing target groups", metric_name, hemi)
                continue

            drug_ids = {rec["subid"] for rec in drug_records}
            pcb_ids = {rec["subid"] for rec in pcb_records}
            common_ids = sorted(drug_ids & pcb_ids)
            paired = len(common_ids) >= 2

            if paired:
                drug_stack = np.stack([next(rec["map"] for rec in drug_records if rec["subid"] == sid) for sid in common_ids], axis=0)
                pcb_stack = np.stack([next(rec["map"] for rec in pcb_records if rec["subid"] == sid) for sid in common_ids], axis=0)
                diff = drug_stack - pcb_stack
                with np.errstate(invalid="ignore"):
                    t_map, p_map = stats.ttest_1samp(diff, popmean=0.0, axis=0, nan_policy="omit")
                diff_std = np.nanstd(diff, axis=0, ddof=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    effect_map = np.nanmean(diff, axis=0) / diff_std
                effect_map[~np.isfinite(effect_map)] = np.nan
                test_type = "paired_ttest"
                n_used = len(common_ids)
                contrast_map = np.nanmean(diff, axis=0)
                drug_mean = np.nanmean(drug_stack, axis=0)
                pcb_mean = np.nanmean(pcb_stack, axis=0)
            else:
                drug_stack = np.stack([rec["map"] for rec in drug_records], axis=0)
                pcb_stack = np.stack([rec["map"] for rec in pcb_records], axis=0)
                with np.errstate(invalid="ignore"):
                    t_map, p_map = stats.ttest_ind(drug_stack, pcb_stack, axis=0, equal_var=False, nan_policy="omit")
                effect_map = _cohens_d_unpaired(drug_stack, pcb_stack)
                test_type = "welch_ttest"
                n_used = min(len(drug_records), len(pcb_records))
                drug_mean = np.nanmean(drug_stack, axis=0)
                pcb_mean = np.nanmean(pcb_stack, axis=0)
                contrast_map = drug_mean - pcb_mean

            q_map = _fdr_bh(np.ravel(p_map)).reshape(p_map.shape)
            sig_mask_unc = p_map < 0.05
            sig_mask_unc[~np.isfinite(p_map)] = False
            sig_mask_fdr = q_map < 0.05
            sig_mask_fdr[~np.isfinite(q_map)] = False
            (
                cluster_labels,
                cluster_p_map,
                cluster_sig_mask,
                cluster_df,
                threshold_t,
            ) = _cluster_permutation(
                paired=paired,
                drug_stack=drug_stack,
                pcb_stack=pcb_stack,
                t_map=np.asarray(t_map, dtype=np.float64),
                cluster_alpha=getattr(cfg, "cluster_alpha", 0.05),
                n_permutations=getattr(cfg, "n_permutations", 2000),
                rng=rng,
            )

            stem = f"{metric_name}_{hemi}"
            np.save(map_dir / f"{stem}_{cfg.group_drug}_mean.npy", drug_mean)
            np.save(map_dir / f"{stem}_{cfg.group_pcb}_mean.npy", pcb_mean)
            np.save(map_dir / f"{stem}_contrast.npy", contrast_map)
            np.save(map_dir / f"{stem}_t.npy", np.asarray(t_map, dtype=np.float64))
            np.save(map_dir / f"{stem}_p.npy", np.asarray(p_map, dtype=np.float64))
            np.save(map_dir / f"{stem}_q.npy", q_map)
            np.save(map_dir / f"{stem}_effect.npy", effect_map)
            np.save(map_dir / f"{stem}_sig_mask_unc_p005.npy", sig_mask_unc.astype(np.uint8))
            np.save(map_dir / f"{stem}_sig_mask_fdr_q005.npy", sig_mask_fdr.astype(np.uint8))
            np.save(map_dir / f"{stem}_cluster_labels.npy", cluster_labels)
            np.save(map_dir / f"{stem}_cluster_p.npy", cluster_p_map)
            np.save(map_dir / f"{stem}_cluster_sig_mask.npy", cluster_sig_mask)

            if not cluster_df.empty:
                cluster_df.insert(0, "metric", metric_name)
                cluster_df.insert(1, "hemisphere", hemi)
                cluster_df.insert(2, "test_type", test_type)
                cluster_rows_all.append(cluster_df)

            summary_rows.append(
                {
                    "metric": metric_name,
                    "hemisphere": hemi,
                    "test_type": test_type,
                    "n_drug": len(drug_records),
                    "n_pcb": len(pcb_records),
                    "n_used_for_test": n_used,
                    "n_paired": len(common_ids),
                    "shape_y": ref_shape[0],
                    "shape_x": ref_shape[1],
                    "n_sig_p_lt_005": int(np.count_nonzero(sig_mask_unc)),
                    "sig_fraction_p_lt_005": float(np.mean(sig_mask_unc)),
                    "n_sig_fdr_q_lt_005": int(np.count_nonzero(sig_mask_fdr)),
                    "sig_fraction_fdr_q_lt_005": float(np.mean(sig_mask_fdr)),
                    "cluster_alpha": float(getattr(cfg, "cluster_alpha", 0.05)),
                    "cluster_forming_t": threshold_t,
                    "n_permutations": int(getattr(cfg, "n_permutations", 2000)),
                    "n_clusters": int(cluster_df.shape[0]),
                    "n_sig_cluster_pixels": int(np.count_nonzero(cluster_sig_mask)),
                    "sig_cluster_fraction": float(np.mean(cluster_sig_mask)),
                    "mean_abs_contrast": float(np.nanmean(np.abs(contrast_map))),
                    "max_abs_contrast": float(np.nanmax(np.abs(contrast_map))),
                    "mean_abs_effect": float(np.nanmean(np.abs(effect_map))),
                    "max_abs_effect": float(np.nanmax(np.abs(effect_map))),
                    "mean_abs_t": float(np.nanmean(np.abs(t_map))),
                    "max_abs_t": float(np.nanmax(np.abs(t_map))),
                }
            )

            if cfg.save_plots:
                contrast_lim = _safe_abs_max(contrast_map)
                effect_lim = _safe_abs_max(effect_map)
                t_lim = _safe_abs_max(np.asarray(t_map, dtype=np.float64))
                q_lim = _safe_abs_max(np.asarray(q_map, dtype=np.float64))
                cluster_p_lim = _safe_abs_max(np.asarray(cluster_p_map, dtype=np.float64))
                fig, axes = plt.subplots(3, 4, figsize=(18, 13))
                ax = axes.ravel()
                im0 = ax[0].imshow(pcb_mean, cmap=spec["cmap"])
                ax[0].set_title(f"{cfg.group_pcb} mean")
                fig.colorbar(im0, ax=ax[0], shrink=0.8)
                im1 = ax[1].imshow(drug_mean, cmap=spec["cmap"])
                ax[1].set_title(f"{cfg.group_drug} mean")
                fig.colorbar(im1, ax=ax[1], shrink=0.8)
                im2 = ax[2].imshow(
                    contrast_map,
                    cmap="RdBu_r",
                    vmin=-contrast_lim,
                    vmax=contrast_lim,
                )
                ax[2].set_title(f"{cfg.group_drug} - {cfg.group_pcb}")
                fig.colorbar(im2, ax=ax[2], shrink=0.8)
                im3 = ax[3].imshow(
                    effect_map,
                    cmap="RdBu_r",
                    vmin=-effect_lim,
                    vmax=effect_lim,
                )
                ax[3].set_title("Effect size map")
                fig.colorbar(im3, ax=ax[3], shrink=0.8)
                im4 = ax[4].imshow(
                    t_map,
                    cmap="RdBu_r",
                    vmin=-t_lim,
                    vmax=t_lim,
                )
                ax[4].set_title(f"{test_type} t map")
                fig.colorbar(im4, ax=ax[4], shrink=0.8)
                im5 = ax[5].imshow(
                    q_map,
                    cmap="viridis_r",
                    vmin=0.0,
                    vmax=min(0.25, q_lim),
                )
                ax[5].set_title("FDR q map")
                fig.colorbar(im5, ax=ax[5], shrink=0.8)
                ax[6].imshow(sig_mask_unc, cmap="gray_r", vmin=0, vmax=1)
                ax[6].set_title("Uncorrected p < 0.05")
                ax[7].imshow(sig_mask_fdr, cmap="gray_r", vmin=0, vmax=1)
                ax[7].set_title("FDR q < 0.05")
                im8 = ax[8].imshow(
                    cluster_p_map,
                    cmap="viridis_r",
                    vmin=0.0,
                    vmax=min(0.25, cluster_p_lim),
                )
                ax[8].set_title("Cluster permutation p")
                fig.colorbar(im8, ax=ax[8], shrink=0.8)
                ax[9].imshow(cluster_sig_mask, cmap="gray_r", vmin=0, vmax=1)
                ax[9].set_title("Cluster p < 0.05")
                im10 = ax[10].imshow(cluster_labels, cmap="tab20", vmin=0)
                ax[10].set_title("Cluster labels")
                fig.colorbar(im10, ax=ax[10], shrink=0.8)
                top_cluster_text = "No clusters"
                if not cluster_df.empty:
                    top = cluster_df.sort_values("cluster_p").iloc[0]
                    top_cluster_text = (
                        f"Top cluster\n"
                        f"id={int(top['cluster_id'])}, p={float(top['cluster_p']):.4f}\n"
                        f"size={int(top['size'])}, mass={float(top['mass']):.2f}\n"
                        f"thr |t|>={threshold_t:.2f}"
                    )
                ax[11].axis("off")
                ax[11].text(0.02, 0.98, top_cluster_text, va="top", ha="left", fontsize=12)
                for item in ax:
                    if item is ax[11]:
                        continue
                    item.invert_yaxis()
                    item.set_xticks([])
                    item.set_yticks([])
                fig.suptitle(f"{spec['title']} contrast ({hemi})")
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                save_fig(fig, out_dir / f"{stem}_contrast_panel.png", cfg.save_plots)

    cluster_out = pd.concat(cluster_rows_all, ignore_index=True) if cluster_rows_all else pd.DataFrame()
    write_table(cluster_out, out_dir / "cluster_permutation_summary.csv")
    return pd.DataFrame(summary_rows)


def main() -> None:
    cfg = parse_args()
    out_dir = cfg.output_root / cfg.results_prefix / "group_contrast_maps"
    set_output_naming(cfg.results_prefix, datetime.now().strftime("%Y%m%d"))
    setup_logging(out_dir)

    bundles = load_bundles(cfg)
    bundles = bundles[bundles["group"].isin({cfg.group_drug, cfg.group_pcb})].copy()
    if bundles.empty:
        raise SystemExit("No bundles found for the requested groups.")

    logging.info("Bundles loaded for contrast analysis: %d", len(bundles))
    scalar_df, map_records = compute_subject_maps(cfg, bundles)
    if scalar_df.empty:
        raise SystemExit("No per-subject metrics could be computed.")

    scalar_metric_cols = [
        "curl_map_mean_abs",
        "curl_map_p95_abs",
        "curl_map_p5_abs",
        "curl_map_p95_p5_diff",
        "curl_map_var_abs",
        "occupancy_mean",
        "occupancy_p95",
        "occupancy_p5",
        "occupancy_p95_p5_diff",
        "occupancy_nonzero_fraction",
        "occupancy_var",
    ]
    scalar_metric_cols = [col for col in scalar_metric_cols if col in scalar_df.columns]
    write_table(scalar_df, out_dir / "per_subject_scalar_metrics.csv")
    write_table(
        build_group_summary_df(scalar_df, scalar_metric_cols, cfg),
        out_dir / "scalar_group_summary.csv",
    )

    map_summary_df = run_map_contrasts(cfg, out_dir, map_records)
    write_table(map_summary_df, out_dir / "map_contrast_summary.csv")
    logging.info("Finished group contrast analysis.")


if __name__ == "__main__":
    main()
