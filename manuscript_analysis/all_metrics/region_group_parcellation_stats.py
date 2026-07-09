from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from tqdm import tqdm

from .config import Config
from .loaders import load_bundles
from .utils import resolve_bundle_dir, resolve_reference_gmap, setup_logging, write_table

matplotlib.use("Agg")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regional (parcellation-wise) group analysis for angle/pattern/cSVD metrics."
    )
    parser.add_argument("--prefix", dest="results_prefix", default="dmt_vs_pcb_region_metrics")
    parser.add_argument("--drug-label", dest="group_drug", default=None)
    parser.add_argument("--pcb-label", dest="group_pcb", default=None)
    parser.add_argument("--detect-dir", type=Path, default=None)
    parser.add_argument("--analytic-dir", type=Path, default=None)
    parser.add_argument("--parcellation-config", type=Path, default=None)
    parser.add_argument("--reference-gmap", type=Path, default=None)
    parser.add_argument("--reference-gmap-left", type=Path, default=None)
    parser.add_argument("--reference-gmap-right", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--time-stride", type=int, default=1, help="Use every Nth frame for map metrics.")
    parser.add_argument("--max-bundles", type=int, default=0, help="For quick debug. 0 means all.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.results_prefix:
        cfg.results_prefix = args.results_prefix
    if args.group_drug:
        cfg.group_drug = args.group_drug
    if args.group_pcb:
        cfg.group_pcb = args.group_pcb
    if args.detect_dir:
        cfg.detect_results_dir = args.detect_dir
    if args.analytic_dir:
        cfg.analytic_dir = args.analytic_dir
    if args.parcellation_config:
        cfg.parcellation_config = args.parcellation_config
    if args.reference_gmap:
        cfg.reference_gmap = args.reference_gmap
    if args.reference_gmap_left:
        cfg.reference_gmap_left = args.reference_gmap_left
    if args.reference_gmap_right:
        cfg.reference_gmap_right = args.reference_gmap_right
    if args.output_root:
        cfg.output_root = args.output_root
    cfg.save_plots = not args.no_plots
    return cfg


def _load_parcellations(cfg: Config) -> Dict[str, np.ndarray]:
    cfg_path = Path(cfg.parcellation_config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    paths = cfg_yaml.get("paths", {})
    data_dir = Path(cfg_path.parent.parent) / paths.get("data_dir", ".")
    out: Dict[str, np.ndarray] = {}
    if paths.get("parcellation_left"):
        out["left"] = np.load(data_dir / paths["parcellation_left"]).astype(float)
    if paths.get("parcellation_right"):
        out["right"] = np.load(data_dir / paths["parcellation_right"]).astype(float)
    return out


def _align_rows(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    if arr.shape[0] == target_shape[0] + 1 and arr.shape[1] == target_shape[1]:
        return arr[:-1, :]
    raise ValueError(f"Cannot align shape {arr.shape} to {target_shape}")


def _prepare_reference_gradient(reference_gmap: Path, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not reference_gmap.exists():
        return None
    gmap = np.load(reference_gmap).astype(float)
    gmap = _align_rows(gmap, target_shape)
    gy, gx = np.gradient(gmap)
    gmag = np.hypot(gx, gy)
    valid = np.isfinite(gmap) & np.isfinite(gx) & np.isfinite(gy) & (gmag > 0)
    valid[:2, :] = False
    valid[-2:, :] = False
    valid[:, :2] = False
    valid[:, -2:] = False
    return gx, gy, valid


def _aggregate_by_parcel(
    map2d: np.ndarray,
    parcellation: np.ndarray,
    hemi: str,
    metric: str,
    group: str,
    subid: str,
) -> List[Dict]:
    rows: List[Dict] = []
    valid = np.isfinite(parcellation) & np.isfinite(map2d)
    if not valid.any():
        return rows
    labels = parcellation[valid].astype(int)
    values = map2d[valid]
    for rid in np.unique(labels):
        mask = labels == rid
        if not mask.any():
            continue
        rows.append(
            {
                "group": group,
                "subid": str(subid),
                "hemisphere": hemi,
                "region_id": int(rid),
                "metric": metric,
                "value": float(np.nanmean(values[mask])),
                "n_voxels": int(mask.sum()),
            }
        )
    return rows


def _resolve_analytic_path(bundle_dir: str, group: str, cfg: Config) -> Path | None:
    fname = f"{bundle_dir}__analytic_cube.npy"
    direct = Path(cfg.analytic_dir) / fname
    if direct.exists():
        return direct
    nested = Path(cfg.analytic_dir) / str(group) / fname
    if nested.exists():
        return nested
    return None


def _csvd_metrics_for_region(x: np.ndarray) -> Dict[str, float]:
    valid_cols = np.all(np.isfinite(x), axis=0)
    x = x[:, valid_cols]
    if x.size == 0 or x.shape[0] < 2 or x.shape[1] < 1:
        return {"csvd_top1_energy": math.nan, "csvd_top3_energy": math.nan, "csvd_entropy": math.nan, "csvd_k90": math.nan}
    x = x - np.mean(x, axis=0, keepdims=True)
    try:
        _, svals, _ = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"csvd_top1_energy": math.nan, "csvd_top3_energy": math.nan, "csvd_entropy": math.nan, "csvd_k90": math.nan}
    energy = np.abs(svals) ** 2
    total = float(np.sum(energy))
    if total <= 0:
        return {"csvd_top1_energy": math.nan, "csvd_top3_energy": math.nan, "csvd_entropy": math.nan, "csvd_k90": math.nan}
    frac = energy / total
    eps = 1.0e-12
    return {
        "csvd_top1_energy": float(frac[0]) if frac.size > 0 else math.nan,
        "csvd_top3_energy": float(np.sum(frac[:3])) if frac.size > 0 else math.nan,
        "csvd_entropy": float(-np.sum(frac * np.log(frac + eps))),
        "csvd_k90": float(int(np.searchsorted(np.cumsum(frac), 0.9) + 1)) if frac.size > 0 else math.nan,
    }


def _bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna()
    if valid.sum() == 0:
        return out
    pv = p[valid].to_numpy(dtype=float)
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    back = np.empty_like(adj)
    back[order] = adj
    out.loc[valid] = back
    return out


def _ttest_unpaired(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return math.nan, math.nan
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(t), float(p)


def _ttest_paired(a: pd.Series, b: pd.Series) -> Tuple[int, float, float, float]:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return 0, math.nan, math.nan, math.nan
    d = (a.loc[common] - b.loc[common]).dropna()
    if len(d) < 2:
        return len(d), math.nan, math.nan, math.nan
    t, p = stats.ttest_rel(a.loc[d.index], b.loc[d.index], nan_policy="omit")
    sd = float(np.std(d, ddof=1))
    dz = float(np.mean(d) / sd) if sd > 0 else math.nan
    return len(d), float(t), float(p), dz


def _mann_whitney(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return math.nan, math.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def _wilcoxon(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return math.nan, math.nan
    d = (a.loc[common] - b.loc[common]).dropna()
    if len(d) < 2 or np.allclose(d.to_numpy(), 0.0):
        return math.nan, math.nan
    w, p = stats.wilcoxon(d, zero_method="wilcox", correction=False)
    return float(w), float(p)


def compute_region_stats(values_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[Dict] = []
    for (hemi, metric, region_id), sub in values_df.groupby(["hemisphere", "metric", "region_id"], dropna=False):
        drug = (
            sub[sub["group"] == cfg.group_drug]
            .groupby("subid")["value"]
            .mean()
        )
        pcb = (
            sub[sub["group"] == cfg.group_pcb]
            .groupby("subid")["value"]
            .mean()
        )
        mean_drug = float(drug.mean()) if len(drug) else math.nan
        mean_pcb = float(pcb.mean()) if len(pcb) else math.nan
        std_drug = float(drug.std(ddof=1)) if len(drug) > 1 else math.nan
        std_pcb = float(pcb.std(ddof=1)) if len(pcb) > 1 else math.nan
        t_unp, p_unp = _ttest_unpaired(drug, pcb)
        u_mwu, p_mwu = _mann_whitney(drug, pcb)
        n_pair, t_pair, p_pair, dz = _ttest_paired(drug, pcb)
        w_wil, p_wil = _wilcoxon(drug, pcb)
        pooled = math.nan
        if len(drug) > 1 and len(pcb) > 1:
            num = (len(drug) - 1) * (std_drug ** 2) + (len(pcb) - 1) * (std_pcb ** 2)
            den = len(drug) + len(pcb) - 2
            pooled = math.sqrt(num / den) if den > 0 else math.nan
        d_unpaired = (mean_drug - mean_pcb) / pooled if pooled and pooled > 0 else math.nan
        rows.append(
            {
                "hemisphere": hemi,
                "metric": metric,
                "region_id": int(region_id),
                "n_drug": int(len(drug)),
                "n_pcb": int(len(pcb)),
                "n_paired": int(n_pair),
                "mean_drug": mean_drug,
                "mean_pcb": mean_pcb,
                "mean_diff_drug_minus_pcb": mean_drug - mean_pcb if (not math.isnan(mean_drug) and not math.isnan(mean_pcb)) else math.nan,
                "t_welch": t_unp,
                "p_welch": p_unp,
                "u_mannwhitney": u_mwu,
                "p_mannwhitney": p_mwu,
                "t_paired": t_pair,
                "p_paired": p_pair,
                "w_wilcoxon": w_wil,
                "p_wilcoxon": p_wil,
                "cohen_d_unpaired": d_unpaired,
                "cohen_dz_paired": dz,
            }
        )
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df
    for p_col in ["p_welch", "p_mannwhitney", "p_paired", "p_wilcoxon"]:
        stats_df[f"{p_col}_fdr"] = (
            stats_df.groupby(["hemisphere", "metric"], group_keys=False)[p_col]
            .apply(_bh_fdr)
        )
    return stats_df


def _plot_top_bars(stats_df: pd.DataFrame, out_dir: Path, alpha: float = 0.05, top_k: int = 15) -> None:
    for (hemi, metric), sub in stats_df.groupby(["hemisphere", "metric"]):
        choose = sub[sub["p_paired_fdr"] < alpha].copy()
        if choose.empty:
            choose = sub.copy()
        choose["abs_diff"] = choose["mean_diff_drug_minus_pcb"].abs()
        choose = choose.sort_values("abs_diff", ascending=False).head(top_k)
        if choose.empty:
            continue
        choose = choose.sort_values("mean_diff_drug_minus_pcb", ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(choose))))
        sns.barplot(data=choose, y="region_id", x="mean_diff_drug_minus_pcb", orient="h", ax=ax, color="#4c72b0")
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
        ax.set_title(f"Top regional diffs | {metric} | {hemi}")
        ax.set_xlabel("Drug - PCB")
        ax.set_ylabel("Region ID")
        fig.tight_layout()
        fig.savefig(out_dir / f"top_regions_{metric}_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_volcano(stats_df: pd.DataFrame, out_dir: Path, alpha: float = 0.05) -> None:
    for (hemi, metric), sub in stats_df.groupby(["hemisphere", "metric"]):
        plot_df = sub.copy()
        p = pd.to_numeric(plot_df["p_paired"], errors="coerce").clip(lower=1e-300)
        plot_df["neglog10p"] = -np.log10(p)
        sig = pd.to_numeric(plot_df["p_paired_fdr"], errors="coerce") < alpha
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            plot_df.loc[~sig, "cohen_dz_paired"],
            plot_df.loc[~sig, "neglog10p"],
            s=22,
            alpha=0.7,
            color="#888888",
            label="ns",
        )
        ax.scatter(
            plot_df.loc[sig, "cohen_dz_paired"],
            plot_df.loc[sig, "neglog10p"],
            s=30,
            alpha=0.9,
            color="#d62728",
            label="FDR<0.05",
        )
        ax.set_title(f"Volcano | {metric} | {hemi}")
        ax.set_xlabel("Paired effect size (Cohen dz)")
        ax.set_ylabel("-log10(paired p)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"volcano_{metric}_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_parcel_maps(stats_df: pd.DataFrame, parcellations: Dict[str, np.ndarray], out_dir: Path, alpha: float = 0.05) -> None:
    for (hemi, metric), sub in stats_df.groupby(["hemisphere", "metric"]):
        parc = parcellations.get(hemi)
        if parc is None:
            continue
        diff_map = np.full(parc.shape, np.nan, dtype=float)
        sig_map = np.full(parc.shape, np.nan, dtype=float)
        for row in sub.itertuples(index=False):
            mask = parc == float(row.region_id)
            diff_map[mask] = row.mean_diff_drug_minus_pcb
            sig_map[mask] = 1.0 if (pd.notna(row.p_paired_fdr) and row.p_paired_fdr < alpha) else 0.0

        fig, ax = plt.subplots(figsize=(7, 5))
        vmax = np.nanpercentile(np.abs(diff_map), 98) if np.isfinite(diff_map).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(diff_map, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(f"Regional diff map ({metric}, {hemi})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"parcel_diff_map_{metric}_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(sig_map, cmap="viridis", vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.set_title(f"Significance map FDR<0.05 ({metric}, {hemi})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"parcel_sig_map_{metric}_{hemi}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    out_root = cfg.output_root / cfg.results_prefix / "regional_parcellation"
    setup_logging(out_root)
    sns.set_theme(style="whitegrid", context="talk")

    bundles = load_bundles(cfg)
    bundles = bundles[bundles["group"].isin({cfg.group_drug, cfg.group_pcb})].copy()
    if args.max_bundles and args.max_bundles > 0:
        bundles = bundles.head(args.max_bundles).copy()
    if bundles.empty:
        raise SystemExit("No bundles found for target groups.")

    parcellations = _load_parcellations(cfg)
    if not parcellations:
        raise SystemExit("Failed to load parcellation files from parcellation-config.")

    # Align parcellation shape to actual phase cube shape for each hemisphere.
    shape_map: Dict[str, Tuple[int, int]] = {}
    for hemi in ["left", "right"]:
        row_hemi = bundles[bundles["hemisphere"] == hemi]
        if row_hemi.empty:
            continue
        sample_cube = np.load(Path(row_hemi.iloc[0]["phase_cube"]), mmap_mode="r")
        shape_map[hemi] = tuple(sample_cube.shape[:2])
        parcellations[hemi] = _align_rows(parcellations[hemi], shape_map[hemi])

    ref_grad = {}
    for hemi, target in shape_map.items():
        ref_path = resolve_reference_gmap(cfg, hemi)
        ref_grad[hemi] = _prepare_reference_gradient(ref_path, target) if ref_path is not None else None

    records: List[Dict] = []

    for row in tqdm(bundles.itertuples(index=False), total=len(bundles), desc="regional metrics"):
        hemi = row.hemisphere
        parc = parcellations.get(hemi)
        if parc is None:
            continue

        bundle_dir = resolve_bundle_dir(pd.Series(row._asdict()), cfg)
        if not bundle_dir:
            bundle_dir = Path(row.bundle_dir)
        cube_path = Path(row.phase_cube)
        if not cube_path.exists():
            continue

        cube = np.load(cube_path, mmap_mode="r")
        h, w, t = cube.shape
        if (h, w) != parc.shape:
            continue

        phase_sum = np.nansum(cube, axis=2).astype(float)
        phase_cnt = np.sum(np.isfinite(cube), axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            phase_mean = np.divide(
                phase_sum,
                phase_cnt,
                out=np.full((h, w), np.nan, dtype=float),
                where=phase_cnt > 0,
            )
        records.extend(_aggregate_by_parcel(phase_mean, parc, hemi, "angle_phase_mean", row.group, row.subid))

        angle_cos_abs_acc = np.zeros((h, w), dtype=np.float64)
        angle_cnt = np.zeros((h, w), dtype=np.float64)
        ref = ref_grad.get(hemi)

        stride = max(1, int(args.time_stride))
        for fidx in range(0, t, stride):
            sl = cube[:, :, fidx]
            if sl.shape != parc.shape:
                continue

            if ref is not None:
                rgx, rgy, rvalid = ref
                py, px = np.gradient(sl.astype(float))
                ref_mag = np.hypot(rgx, rgy)
                tgt_mag = np.hypot(px, py)
                denom = ref_mag * tgt_mag
                mask = rvalid & np.isfinite(sl) & np.isfinite(px) & np.isfinite(py) & (denom > 0)
                if np.any(mask):
                    cos_theta = np.divide(
                        rgx * px + rgy * py,
                        denom,
                        out=np.full_like(rgx, np.nan, dtype=float),
                        where=mask,
                    )
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_cos_abs_acc[mask] += np.abs(cos_theta[mask])
                    angle_cnt[mask] += 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            angle_abs_cos_mean = np.divide(
                angle_cos_abs_acc,
                angle_cnt,
                out=np.full_like(angle_cos_abs_acc, np.nan),
                where=angle_cnt > 0,
            )
        records.extend(_aggregate_by_parcel(angle_abs_cos_mean, parc, hemi, "angle_abs_cos_mean", row.group, row.subid))

        frame_path = bundle_dir / "frame_index.parquet"
        if frame_path.exists():
            fi = pd.read_parquet(
                frame_path,
                columns=[
                    "pattern_id",
                    "weighted_centroid_x",
                    "weighted_centroid_y",
                    "instantaneous_power",
                    "instantaneous_size",
                ],
            )
            if not fi.empty:
                pids = pd.to_numeric(fi["pattern_id"], errors="coerce").astype("Int64")
                xs = np.rint(pd.to_numeric(fi["weighted_centroid_x"], errors="coerce")).astype("Int64")
                ys = np.rint(pd.to_numeric(fi["weighted_centroid_y"], errors="coerce")).astype("Int64")
                pw = pd.to_numeric(fi["instantaneous_power"], errors="coerce")
                sz = pd.to_numeric(fi["instantaneous_size"], errors="coerce")
                region_rows: Dict[int, Dict[str, List[float] | set[int]]] = {}
                for pid, x, y, pwr, siz in zip(pids, xs, ys, pw, sz):
                    if pd.isna(x) or pd.isna(y):
                        continue
                    xi = int(x)
                    yi = int(y)
                    if yi < 0 or yi >= parc.shape[0] or xi < 0 or xi >= parc.shape[1]:
                        continue
                    rid = parc[yi, xi]
                    if not np.isfinite(rid):
                        continue
                    key = int(rid)
                    region_rows.setdefault(key, {"power": [], "size": [], "pattern_ids": set()})
                    if not pd.isna(pid):
                        region_rows[key]["pattern_ids"].add(int(pid))
                    if np.isfinite(pwr):
                        region_rows[key]["power"].append(float(pwr))
                    if np.isfinite(siz):
                        region_rows[key]["size"].append(float(siz))
                for rid, vals in region_rows.items():
                    records.append(
                        {
                            "group": row.group,
                            "subid": str(row.subid),
                            "hemisphere": hemi,
                            "region_id": rid,
                            "metric": "pattern_power_mean",
                            "value": float(np.mean(vals["power"])) if vals["power"] else math.nan,
                            "n_voxels": int(np.sum(parc == rid)),
                        }
                    )
                    records.append(
                        {
                            "group": row.group,
                            "subid": str(row.subid),
                            "hemisphere": hemi,
                            "region_id": rid,
                            "metric": "pattern_size_mean",
                            "value": float(np.mean(vals["size"])) if vals["size"] else math.nan,
                            "n_voxels": int(np.sum(parc == rid)),
                        }
                    )
                    records.append(
                        {
                            "group": row.group,
                            "subid": str(row.subid),
                            "hemisphere": hemi,
                            "region_id": rid,
                            "metric": "pattern_count",
                            "value": float(len(vals["pattern_ids"])),
                            "n_voxels": int(np.sum(parc == rid)),
                        }
                    )

        analytic_path = _resolve_analytic_path(str(bundle_dir.name), str(row.group), cfg)
        if analytic_path is not None and analytic_path.exists():
            acube = np.load(analytic_path, mmap_mode="r")
            if acube.ndim == 3 and acube.shape[:2] == parc.shape:
                flat = acube.reshape(-1, acube.shape[2])
                valid = np.isfinite(parc.ravel())
                labels = parc.ravel()[valid].astype(int)
                for rid in np.unique(labels):
                    mask = (parc.ravel() == rid)
                    x = flat[mask, :].T
                    csvd_m = _csvd_metrics_for_region(x)
                    for mname, mval in csvd_m.items():
                        records.append(
                            {
                                "group": row.group,
                                "subid": str(row.subid),
                                "hemisphere": hemi,
                                "region_id": int(rid),
                                "metric": mname,
                                "value": float(mval) if np.isfinite(mval) else math.nan,
                                "n_voxels": int(mask.sum()),
                            }
                        )

    values_df = pd.DataFrame(records)
    if values_df.empty:
        raise SystemExit("No regional records generated.")
    values_df = (
        values_df.groupby(["group", "subid", "hemisphere", "region_id", "metric"], as_index=False)
        .agg(value=("value", "mean"), n_voxels=("n_voxels", "max"))
    )
    write_table(values_df, out_root / "regional_values_long.csv")

    stats_df = compute_region_stats(values_df, cfg)
    write_table(stats_df, out_root / "regional_stats.csv")

    sig_df = stats_df[
        (pd.to_numeric(stats_df["p_paired_fdr"], errors="coerce") < 0.05)
        | (pd.to_numeric(stats_df["p_wilcoxon_fdr"], errors="coerce") < 0.05)
    ].copy()
    write_table(sig_df, out_root / "regional_significant.csv")

    if cfg.save_plots and not stats_df.empty:
        plot_dir = out_root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_top_bars(stats_df, plot_dir, alpha=0.05, top_k=15)
        _plot_volcano(stats_df, plot_dir, alpha=0.05)
        _plot_parcel_maps(stats_df, parcellations, plot_dir, alpha=0.05)

    summary_rows = []
    for (hemi, metric), sub in stats_df.groupby(["hemisphere", "metric"]):
        n_sig = int((pd.to_numeric(sub["p_paired_fdr"], errors="coerce") < 0.05).sum())
        best = sub.sort_values("p_paired", na_position="last").head(1)
        best_region = int(best["region_id"].iloc[0]) if not best.empty else -1
        best_p = float(best["p_paired"].iloc[0]) if not best.empty else math.nan
        summary_rows.append(
            {
                "hemisphere": hemi,
                "metric": metric,
                "n_significant_paired_fdr_lt_0.05": n_sig,
                "best_region_id_by_paired_p": best_region,
                "best_paired_p": best_p,
            }
        )
    write_table(pd.DataFrame(summary_rows), out_root / "summary_by_metric.csv")


if __name__ == "__main__":
    main()
