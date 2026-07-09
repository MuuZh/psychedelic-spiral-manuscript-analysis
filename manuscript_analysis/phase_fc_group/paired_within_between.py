#!/usr/bin/env python
"""
Paired group analysis for downsampled-grid phase-FC outputs.

Inputs are the per-subject/per-hemisphere outputs from scripts/phase_fc_batch.py.
For each paired subject and hemisphere, this script computes:

    delta = Drug - Placebo

Then it summarizes:
    - network within/between PLV deltas with paired t-tests against zero
    - parcel-level mean delta matrices, plus edge-wise paired t/p matrices
    - flat-grid parcel maps of each parcel's mean delta connectivity

No FDR correction is applied.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from segregation import compute_network_segregation


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
NETWORK_ORDER = NETWORK_ORDER_7 + NETWORK_ORDER_17
NETWORK_METRIC_COLUMNS = ["network_a", "network_b", "type", "value"]


def infer_network_order(networks: pd.Series | list[str]) -> list[str]:
    values = [str(network) for network in networks if pd.notna(network)]
    available = set(values)
    for known_order in (NETWORK_ORDER_7, NETWORK_ORDER_17):
        ordered = [network for network in known_order if network in available]
        if ordered:
            extras = [network for network in values if network not in set(known_order)]
            return ordered + list(dict.fromkeys(extras))
    return list(dict.fromkeys(values))
CONDITION_RE = re.compile(r"^(?P<condition>[A-Za-z]+_[A-Za-z]+)_S(?P<subid>\d+)")


@dataclass(frozen=True)
class FcEntry:
    condition: str
    subid: str
    hemisphere: str
    out_dir: Path
    fc_path: Path
    meta_path: Path
    grid_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired Drug-Placebo phase-FC analysis.")
    parser.add_argument(
        "--phase-fc-root",
        default=Path("analysis_outputs/phase_fc_batch"),
        type=Path,
        help="Root containing per-subject phase-FC output directories.",
    )
    parser.add_argument(
        "--atlas-parcels",
        default=Path("analysis_outputs/phase_fc_batch/atlas_metadata/Schaefer2018_400Parcels_7Networks_parcels.csv"),
        type=Path,
        help="Shared parcel metadata CSV from phase_fc_build_atlas_metadata.py.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=None,
        help="Drug:Placebo condition pair, e.g. DMT_DMT:DMT_PCB. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/paired_within_between"),
        type=Path,
    )
    parser.add_argument(
        "--fc-file",
        default="parcel_plv.npy",
        help="Per-subject FC matrix filename to analyze, e.g. parcel_plv.npy or parcel_phase_corr.npy.",
    )
    parser.add_argument(
        "--metric-label",
        default="PLV",
        help="Label used in figures for the FC metric.",
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--min-pairs",
        default=3,
        type=int,
        help="Minimum paired subjects required to run a t-test for a metric.",
    )
    return parser.parse_args()


def infer_hemisphere_from_meta(meta_path: Path) -> str:
    meta = pd.read_csv(meta_path, usecols=["hemi"])
    hemi = str(meta["hemi"].iloc[0])
    if hemi == "LH":
        return "left"
    if hemi == "RH":
        return "right"
    raise ValueError(f"Unexpected hemi value {hemi!r} in {meta_path}")


def parse_entry(out_dir: Path, fc_file: str) -> FcEntry | None:
    match = CONDITION_RE.search(out_dir.name)
    if not match:
        return None
    fc_path = out_dir / fc_file
    meta_path = out_dir / "parcel_metadata.csv"
    grid_path = out_dir / "grid_labels.npy"
    if not fc_path.exists() or not meta_path.exists():
        return None
    return FcEntry(
        condition=match.group("condition"),
        subid=match.group("subid"),
        hemisphere=infer_hemisphere_from_meta(meta_path),
        out_dir=out_dir,
        fc_path=fc_path,
        meta_path=meta_path,
        grid_path=grid_path,
    )


def discover_entries(root: Path, fc_file: str) -> pd.DataFrame:
    rows = []
    for out_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "atlas_metadata"):
        entry = parse_entry(out_dir, fc_file)
        if entry is None:
            continue
        rows.append(entry.__dict__)
    if not rows:
        raise RuntimeError(f"No phase-FC outputs found under {root}")
    return pd.DataFrame(rows)


def default_comparisons(conditions: set[str]) -> list[tuple[str, str]]:
    candidates = [("DMT_DMT", "DMT_PCB"), ("LSD_LSD", "LSD_PCB")]
    return [pair for pair in candidates if pair[0] in conditions and pair[1] in conditions]


def parse_comparisons(args: argparse.Namespace, conditions: set[str]) -> list[tuple[str, str]]:
    if args.comparison:
        pairs = []
        for spec in args.comparison:
            if ":" not in spec:
                raise ValueError(f"Comparison must be Drug:Placebo, got {spec!r}")
            drug, placebo = spec.split(":", 1)
            pairs.append((drug, placebo))
        return pairs
    pairs = default_comparisons(conditions)
    if not pairs:
        raise RuntimeError(
            "No default comparisons found. Pass --comparison Drug:Placebo explicitly."
        )
    return pairs


def load_hemi_parcels(atlas_parcels: Path, hemisphere: str) -> pd.DataFrame:
    parcels = pd.read_csv(atlas_parcels)
    hemi = "LH" if hemisphere == "left" else "RH"
    out = parcels[parcels["hemi"] == hemi].copy()
    if out.empty:
        raise RuntimeError(f"No {hemi} parcels found in {atlas_parcels}")
    return out.sort_values("parcel_id").reset_index(drop=True)


def assert_fc_order(meta_path: Path, hemi_parcels: pd.DataFrame) -> None:
    meta = pd.read_csv(meta_path)
    cols = ["parcel_id", "parcel_name", "hemi", "network"]
    current = meta[cols].reset_index(drop=True)
    expected = hemi_parcels[cols].reset_index(drop=True)
    if not current.equals(expected):
        raise ValueError(f"Parcel metadata order mismatch: {meta_path}")


def paired_subjects(entries: pd.DataFrame, drug: str, placebo: str, hemisphere: str) -> list[str]:
    subset = entries[entries["hemisphere"] == hemisphere]
    drug_subs = set(subset[subset["condition"] == drug]["subid"])
    pcb_subs = set(subset[subset["condition"] == placebo]["subid"])
    return sorted(drug_subs & pcb_subs)


def load_fc_for(entries: pd.DataFrame, condition: str, subid: str, hemisphere: str) -> tuple[np.ndarray, Path]:
    row = entries[
        (entries["condition"] == condition)
        & (entries["subid"] == subid)
        & (entries["hemisphere"] == hemisphere)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one FC for {condition} S{subid} {hemisphere}, found {len(row)}")
    fc_path = Path(row.iloc[0]["fc_path"])
    return np.load(fc_path), Path(row.iloc[0]["meta_path"])


def network_metric(fc: np.ndarray, hemi_parcels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    network_order = infer_network_order(hemi_parcels["network"])
    networks = [net for net in network_order if net in set(hemi_parcels["network"])]
    network_values = hemi_parcels["network"].to_numpy()
    for i, net_a in enumerate(networks):
        idx_a = np.flatnonzero(network_values == net_a)
        if idx_a.size >= 2:
            sub = fc[np.ix_(idx_a, idx_a)]
            rows.append(
                {
                    "network_a": net_a,
                    "network_b": net_a,
                    "type": "within",
                    "value": float(np.nanmean(sub[np.triu_indices_from(sub, k=1)])),
                }
            )
        for net_b in networks[i + 1 :]:
            idx_b = np.flatnonzero(network_values == net_b)
            sub = fc[np.ix_(idx_a, idx_b)]
            rows.append(
                {
                    "network_a": net_a,
                    "network_b": net_b,
                    "type": "between",
                    "value": float(np.nanmean(sub)),
                }
            )
    return pd.DataFrame(rows, columns=NETWORK_METRIC_COLUMNS)


def paired_t(values: np.ndarray, min_pairs: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < min_pairs:
        return {"n": n, "mean_delta": np.nan, "sd_delta": np.nan, "sem_delta": np.nan, "t": np.nan, "p": np.nan, "cohen_dz": np.nan}
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = float(sd / np.sqrt(n)) if n > 1 else np.nan
    if n > 1 and sd > 0:
        t_stat, p_val = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
        dz = mean / sd
    else:
        t_stat, p_val, dz = np.nan, np.nan, np.nan
    return {
        "n": n,
        "mean_delta": mean,
        "sd_delta": sd,
        "sem_delta": sem,
        "t": float(t_stat),
        "p": float(p_val),
        "cohen_dz": float(dz),
    }


def edgewise_t(delta_stack: np.ndarray, min_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    n_sub, n_parcels, _ = delta_stack.shape
    t_mat = np.full((n_parcels, n_parcels), np.nan, dtype=np.float32)
    p_mat = np.full((n_parcels, n_parcels), np.nan, dtype=np.float32)
    for i in range(n_parcels):
        for j in range(i + 1, n_parcels):
            vals = delta_stack[:, i, j]
            vals = vals[np.isfinite(vals)]
            if vals.size < min_pairs:
                continue
            if np.std(vals, ddof=1) == 0:
                continue
            t_stat, p_val = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
            t_mat[i, j] = t_mat[j, i] = np.float32(t_stat)
            p_mat[i, j] = p_mat[j, i] = np.float32(p_val)
    return t_mat, p_mat


def matrix_to_long(mat: np.ndarray, parcels: pd.DataFrame, value_name: str) -> pd.DataFrame:
    rows = []
    ids = parcels["parcel_id"].to_numpy()
    names = parcels["parcel_name"].to_numpy()
    networks = parcels["network"].to_numpy()
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[1]):
            rows.append(
                {
                    "parcel_i": int(ids[i]),
                    "parcel_j": int(ids[j]),
                    "parcel_i_name": names[i],
                    "parcel_j_name": names[j],
                    "network_i": networks[i],
                    "network_j": networks[j],
                    value_name: float(mat[i, j]) if np.isfinite(mat[i, j]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def network_matrix(stats_df: pd.DataFrame, value_col: str = "mean_delta") -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()
    network_order = infer_network_order(list(stats_df["network_a"]) + list(stats_df["network_b"]))
    networks = [net for net in network_order if net in set(stats_df["network_a"]) | set(stats_df["network_b"])]
    mat = pd.DataFrame(np.nan, index=networks, columns=networks)
    for row in stats_df.itertuples(index=False):
        mat.loc[row.network_a, row.network_b] = getattr(row, value_col)
        mat.loc[row.network_b, row.network_a] = getattr(row, value_col)
    return mat


def plot_network_heatmap(mat: pd.DataFrame, out_path: Path, title: str, metric_label: str) -> None:
    if mat.empty:
        return
    vals = mat.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    vmax = max(vmax, 1e-6)
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, annot=True, fmt=".3f", cbar_kws={"label": f"Drug - Placebo {metric_label}"})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_network_stat_heatmap(
    mat: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    diverging: bool,
    label: str,
) -> None:
    if mat.empty:
        return
    vals = mat.to_numpy(dtype=float)
    plt.figure(figsize=(7, 6))
    if diverging:
        vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
        vmax = max(vmax, 1e-6)
        sns.heatmap(
            mat,
            cmap="coolwarm",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            cbar_kws={"label": label},
        )
    else:
        sns.heatmap(mat, cmap="viridis_r", vmin=0, vmax=0.10, annot=True, fmt=".3f", cbar_kws={"label": label})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_segregation_summary(seg_stats: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    if seg_stats.empty:
        return
    network_order = infer_network_order(seg_stats["network"])
    variants = ["raw_unweighted", "raw_weighted", "normalized_unweighted", "normalized_weighted"]
    for variant in variants:
        subset = seg_stats[seg_stats["variant"] == variant].copy()
        if subset.empty:
            continue
        subset["network"] = pd.Categorical(subset["network"], categories=network_order, ordered=True)
        subset = subset.sort_values("network")
        plt.figure(figsize=(8, 4.8))
        ax = sns.barplot(data=subset, x="network", y="mean_delta", color="#4c72b0")
        ax.errorbar(
            np.arange(len(subset)),
            subset["mean_delta"].to_numpy(),
            yerr=subset["sem_delta"].to_numpy(),
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
        )
        for idx, row in enumerate(subset.itertuples(index=False)):
            if np.isfinite(row.p) and row.p < 0.05:
                y = row.mean_delta + (row.sem_delta if np.isfinite(row.sem_delta) else 0)
                ax.text(idx, y, "*", ha="center", va="bottom", fontsize=14)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{title_prefix} segregation delta: {variant}")
        ax.set_xlabel("")
        ax.set_ylabel("Drug - Placebo")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"segregation_{variant}_bar.png", dpi=240)
        plt.close()

    for value_col, suffix, diverging, label in [
        ("mean_delta", "mean_delta", True, "Drug - Placebo"),
        ("t", "t", True, "paired t"),
        ("p", "p", False, "p"),
    ]:
        mat = seg_stats.pivot(index="variant", columns="network", values=value_col)
        mat = mat.reindex(index=variants, columns=[n for n in network_order if n in mat.columns])
        plot_network_stat_heatmap(
            mat,
            out_dir / f"segregation_{suffix}_heatmap.png",
            f"{title_prefix} segregation {suffix}",
            diverging=diverging,
            label=label,
        )


def plot_parcel_heatmap(delta_mean: np.ndarray, parcels: pd.DataFrame, out_path: Path, title: str, metric_label: str) -> None:
    network_order = infer_network_order(parcels["network"])
    sort_key = parcels["network"].map({net: i for i, net in enumerate(network_order)}).fillna(999)
    order = np.lexsort((parcels["parcel_id"].to_numpy(), sort_key.to_numpy()))
    sorted_delta = delta_mean[np.ix_(order, order)]
    sorted_parcels = parcels.iloc[order].reset_index(drop=True)
    vmax = np.nanmax(np.abs(sorted_delta)) if np.isfinite(sorted_delta).any() else 1.0
    vmax = max(vmax, 1e-6)

    plt.figure(figsize=(9, 8))
    sns.heatmap(sorted_delta, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, square=True, cbar_kws={"label": f"Drug - Placebo {metric_label}"})
    bounds = []
    labels = []
    start = 0
    for network, group in sorted_parcels.groupby("network", sort=False):
        end = start + len(group)
        bounds.append((start, end))
        labels.append(network)
        start = end
    centers = [(a + b) / 2 for a, b in bounds]
    for _, b in bounds[:-1]:
        plt.axhline(b, color="black", linewidth=0.5)
        plt.axvline(b, color="black", linewidth=0.5)
    plt.xticks(centers, labels, rotation=45, ha="right")
    plt.yticks(centers, labels, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_parcel_stat_heatmap(
    mat: np.ndarray,
    parcels: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    diverging: bool,
    label: str,
) -> None:
    network_order = infer_network_order(parcels["network"])
    sort_key = parcels["network"].map({net: i for i, net in enumerate(network_order)}).fillna(999)
    order = np.lexsort((parcels["parcel_id"].to_numpy(), sort_key.to_numpy()))
    sorted_mat = mat[np.ix_(order, order)]
    sorted_parcels = parcels.iloc[order].reset_index(drop=True)

    plt.figure(figsize=(9, 8))
    if diverging:
        vmax = np.nanmax(np.abs(sorted_mat)) if np.isfinite(sorted_mat).any() else 1.0
        vmax = max(vmax, 1e-6)
        sns.heatmap(sorted_mat, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, square=True, cbar_kws={"label": label})
    else:
        sns.heatmap(sorted_mat, cmap="viridis_r", vmin=0, vmax=0.10, square=True, cbar_kws={"label": label})

    bounds = []
    labels = []
    start = 0
    for network, group in sorted_parcels.groupby("network", sort=False):
        end = start + len(group)
        bounds.append((start, end))
        labels.append(network)
        start = end
    centers = [(a + b) / 2 for a, b in bounds]
    for _, b in bounds[:-1]:
        plt.axhline(b, color="black", linewidth=0.5)
        plt.axvline(b, color="black", linewidth=0.5)
    plt.xticks(centers, labels, rotation=45, ha="right")
    plt.yticks(centers, labels, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def parcel_strength(delta_mean: np.ndarray) -> np.ndarray:
    work = delta_mean.copy()
    np.fill_diagonal(work, np.nan)
    return np.nanmean(work, axis=1)


def plot_flat_strength(grid_labels: np.ndarray, parcels: pd.DataFrame, values: np.ndarray, out_path: Path, title: str, metric_label: str) -> None:
    value_by_id = dict(zip(parcels["parcel_id"].astype(int), values))
    flat = np.full(grid_labels.shape, np.nan, dtype=float)
    finite = np.isfinite(grid_labels)
    grid_ids = np.zeros(grid_labels.shape, dtype=np.int32)
    grid_ids[finite] = grid_labels[finite].astype(np.int32)
    for parcel_id, value in value_by_id.items():
        flat[finite & (grid_ids == int(parcel_id))] = value
    vmax = np.nanmax(np.abs(flat)) if np.isfinite(flat).any() else 1.0
    vmax = max(vmax, 1e-6)
    plt.figure(figsize=(8, 5))
    plt.imshow(np.ma.masked_invalid(flat), cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(label=f"Mean edge delta {metric_label}", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def first_grid_for(entries: pd.DataFrame, condition: str, hemisphere: str) -> Path | None:
    rows = entries[(entries["condition"] == condition) & (entries["hemisphere"] == hemisphere)]
    for path in rows["grid_path"]:
        path = Path(path)
        if path.exists():
            return path
    return None


def analyze_comparison(
    entries: pd.DataFrame,
    atlas_parcels: Path,
    drug: str,
    placebo: str,
    hemisphere: str,
    out_dir: Path,
    min_pairs: int,
    metric_label: str,
) -> pd.DataFrame:
    hemi_out = out_dir / f"{drug}_minus_{placebo}" / hemisphere
    hemi_out.mkdir(parents=True, exist_ok=True)
    parcels = load_hemi_parcels(atlas_parcels, hemisphere)
    network_order = infer_network_order(parcels["network"])
    subjects = paired_subjects(entries, drug, placebo, hemisphere)
    if len(subjects) < min_pairs:
        print(f"Skipping {drug}-{placebo} {hemisphere}: only {len(subjects)} pairs")
        return pd.DataFrame()

    deltas = []
    network_rows = []
    segregation_rows = []
    subject_rows = []
    for subid in subjects:
        drug_fc, drug_meta = load_fc_for(entries, drug, subid, hemisphere)
        pcb_fc, pcb_meta = load_fc_for(entries, placebo, subid, hemisphere)
        assert_fc_order(drug_meta, parcels)
        assert_fc_order(pcb_meta, parcels)
        delta = drug_fc - pcb_fc
        deltas.append(delta.astype(np.float32, copy=False))

        drug_net = network_metric(drug_fc, parcels).rename(columns={"value": "drug_value"})
        pcb_net = network_metric(pcb_fc, parcels).rename(columns={"value": "placebo_value"})
        merged = drug_net.merge(pcb_net, on=["network_a", "network_b", "type"], how="inner")
        merged["delta"] = merged["drug_value"] - merged["placebo_value"]
        merged["subid"] = subid
        merged["hemisphere"] = hemisphere
        merged["comparison"] = f"{drug}-{placebo}"
        network_rows.append(merged)

        drug_seg = compute_network_segregation(
            drug_net.rename(columns={"drug_value": "value"}),
            parcels,
            network_order=network_order,
        ).rename(columns={"value": "drug_value", "within": "drug_within", "mean_between": "drug_mean_between"})
        pcb_seg = compute_network_segregation(
            pcb_net.rename(columns={"placebo_value": "value"}),
            parcels,
            network_order=network_order,
        ).rename(columns={"value": "placebo_value", "within": "placebo_within", "mean_between": "placebo_mean_between"})
        seg = drug_seg.merge(
            pcb_seg,
            on=["network", "variant", "weighted", "normalized", "n_network"],
            how="inner",
        )
        seg["delta"] = seg["drug_value"] - seg["placebo_value"]
        seg["subid"] = subid
        seg["hemisphere"] = hemisphere
        seg["comparison"] = f"{drug}-{placebo}"
        segregation_rows.append(seg)
        subject_rows.append({"comparison": f"{drug}-{placebo}", "hemisphere": hemisphere, "subid": subid})

    delta_stack = np.stack(deltas, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        delta_mean = np.nanmean(delta_stack, axis=0)
    t_mat, p_mat = edgewise_t(delta_stack, min_pairs=min_pairs)
    strength = parcel_strength(delta_mean)

    np.save(hemi_out / "parcel_delta_stack.npy", delta_stack)
    np.save(hemi_out / "parcel_delta_mean.npy", delta_mean)
    np.save(hemi_out / "parcel_delta_t.npy", t_mat)
    np.save(hemi_out / "parcel_delta_p.npy", p_mat)
    pd.DataFrame(subject_rows).to_csv(hemi_out / "paired_subjects.csv", index=False)
    matrix_to_long(delta_mean, parcels, "mean_delta").to_csv(hemi_out / "parcel_delta_mean_edges.csv", index=False)
    matrix_to_long(p_mat, parcels, "p").to_csv(hemi_out / "parcel_delta_p_edges.csv", index=False)
    parcels.assign(mean_delta_connectivity=strength).to_csv(hemi_out / "parcel_delta_strength.csv", index=False)

    all_net = pd.concat(network_rows, ignore_index=True)
    all_net.to_csv(hemi_out / "network_subject_deltas.csv", index=False)
    stats_rows = []
    for keys, group in all_net.groupby(["comparison", "hemisphere", "network_a", "network_b", "type"], sort=False):
        stats_dict = paired_t(group["delta"].to_numpy(), min_pairs=min_pairs)
        row = dict(zip(["comparison", "hemisphere", "network_a", "network_b", "type"], keys))
        row.update(stats_dict)
        row["mean_drug"] = float(np.mean(group["drug_value"]))
        row["mean_placebo"] = float(np.mean(group["placebo_value"]))
        stats_rows.append(row)
    net_stats = pd.DataFrame(stats_rows)
    net_stats.to_csv(hemi_out / "network_paired_ttests.csv", index=False)

    net_mat = network_matrix(net_stats, "mean_delta")
    net_mat.to_csv(hemi_out / "network_delta_matrix.csv")
    plot_network_heatmap(net_mat, hemi_out / "network_delta_matrix.png", f"{drug} - {placebo} network delta ({hemisphere})", metric_label)
    plot_network_stat_heatmap(
        network_matrix(net_stats, "t"),
        hemi_out / "network_t_matrix.png",
        f"{drug} - {placebo} network paired t ({hemisphere})",
        diverging=True,
        label="paired t",
    )
    plot_network_stat_heatmap(
        network_matrix(net_stats, "p"),
        hemi_out / "network_p_matrix.png",
        f"{drug} - {placebo} network paired p ({hemisphere})",
        diverging=False,
        label="p",
    )

    all_seg = pd.concat(segregation_rows, ignore_index=True)
    all_seg.to_csv(hemi_out / "segregation_subject_deltas.csv", index=False)
    seg_stats_rows = []
    for keys, group in all_seg.groupby(["comparison", "hemisphere", "network", "variant", "weighted", "normalized"], sort=False):
        stats_dict = paired_t(group["delta"].to_numpy(), min_pairs=min_pairs)
        row = dict(zip(["comparison", "hemisphere", "network", "variant", "weighted", "normalized"], keys))
        row.update(stats_dict)
        row["mean_drug"] = float(np.nanmean(group["drug_value"]))
        row["mean_placebo"] = float(np.nanmean(group["placebo_value"]))
        row["mean_drug_within"] = float(np.nanmean(group["drug_within"]))
        row["mean_placebo_within"] = float(np.nanmean(group["placebo_within"]))
        row["mean_drug_between"] = float(np.nanmean(group["drug_mean_between"]))
        row["mean_placebo_between"] = float(np.nanmean(group["placebo_mean_between"]))
        seg_stats_rows.append(row)
    seg_stats = pd.DataFrame(seg_stats_rows)
    seg_stats.to_csv(hemi_out / "segregation_paired_ttests.csv", index=False)
    plot_segregation_summary(seg_stats, hemi_out, f"{drug} - {placebo} ({hemisphere})")

    plot_parcel_heatmap(delta_mean, parcels, hemi_out / "parcel_delta_matrix.png", f"{drug} - {placebo} parcel delta ({hemisphere})", metric_label)
    plot_parcel_stat_heatmap(
        t_mat,
        parcels,
        hemi_out / "parcel_t_matrix.png",
        f"{drug} - {placebo} parcel paired t ({hemisphere})",
        diverging=True,
        label="paired t",
    )
    plot_parcel_stat_heatmap(
        p_mat,
        parcels,
        hemi_out / "parcel_p_matrix.png",
        f"{drug} - {placebo} parcel paired p ({hemisphere})",
        diverging=False,
        label="p",
    )

    grid_path = first_grid_for(entries, drug, hemisphere) or first_grid_for(entries, placebo, hemisphere)
    if grid_path is not None:
        grid_labels = np.load(grid_path)
        plot_flat_strength(grid_labels, parcels, strength, hemi_out / "flat_parcel_delta_strength.png", f"{drug} - {placebo} parcel mean delta ({hemisphere})", metric_label)

    return net_stats


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = discover_entries(args.phase_fc_root, args.fc_file)
    hemispheres = ["left", "right"] if args.hemisphere == "both" else [args.hemisphere]
    comparisons = parse_comparisons(args, set(entries["condition"]))

    all_stats = []
    for drug, placebo in comparisons:
        for hemisphere in hemispheres:
            stats_df = analyze_comparison(
                entries=entries,
                atlas_parcels=args.atlas_parcels,
                drug=drug,
                placebo=placebo,
                hemisphere=hemisphere,
                out_dir=args.out_dir,
                min_pairs=args.min_pairs,
                metric_label=args.metric_label,
            )
            if not stats_df.empty:
                all_stats.append(stats_df)

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        combined.to_csv(args.out_dir / "all_network_paired_ttests.csv", index=False)
    seg_files = sorted(args.out_dir.glob("*_minus_*/*/segregation_paired_ttests.csv"))
    if seg_files:
        pd.concat((pd.read_csv(path) for path in seg_files), ignore_index=True).to_csv(
            args.out_dir / "all_segregation_paired_ttests.csv",
            index=False,
        )
    summary = {
        "phase_fc_root": str(args.phase_fc_root),
        "atlas_parcels": str(args.atlas_parcels),
        "comparisons": [{"drug": d, "placebo": p} for d, p in comparisons],
        "hemispheres": hemispheres,
        "n_entries": int(len(entries)),
        "fc_file": args.fc_file,
        "metric_label": args.metric_label,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote group analysis to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
