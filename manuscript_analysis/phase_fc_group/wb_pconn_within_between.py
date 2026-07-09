#!/usr/bin/env python
"""
Paired group analysis for wb_command pconn FC outputs.

Inputs are the per-condition hemisphere outputs from the wb_command batch:

    <fc-root>/<condition-dir>/manifests/fc_batch_hemisphere_manifest.csv
    <fc-root>/<condition-dir>/pconn/<hemisphere>/*.pconn.nii

For each paired subject and hemisphere, this script computes:

    delta = Drug - Placebo

Then it summarizes:
    - network within/between FC deltas with paired t-tests against zero
    - network segregation deltas with paired t-tests against zero
    - parcel-level mean delta matrices, plus edge-wise paired t/p matrices

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
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

from paired_within_between import (
    edgewise_t,
    infer_network_order,
    load_hemi_parcels,
    matrix_to_long,
    network_matrix,
    network_metric,
    paired_t,
    parcel_strength,
    plot_network_stat_heatmap,
    plot_parcel_stat_heatmap,
    plot_segregation_summary,
)
from segregation import compute_network_segregation


CONDITION_RE = re.compile(r"^(?P<condition>[A-Za-z]+_[A-Za-z]+)_S(?P<subid>\d+)")


@dataclass(frozen=True)
class PconnEntry:
    condition: str
    subid: str
    subject_id: str
    hemisphere: str
    pconn_path: Path
    manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired Drug-Placebo analysis for wb_command pconn FC outputs.")
    parser.add_argument(
        "--fc-root",
        default=Path(r"<raw_fc_outputs>\fc_out_hemi"),
        type=Path,
        help="Root containing wb_command FC outputs split by condition.",
    )
    parser.add_argument(
        "--atlas-parcels",
        default=Path("analysis_outputs/phase_fc_batch/atlas_metadata/Schaefer2018_400Parcels_7Networks_parcels.csv"),
        type=Path,
        help="Shared Schaefer parcel metadata CSV.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=None,
        help="Drug:Placebo condition pair, e.g. DMT_DMT:DMT_PCB. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("analysis_outputs/phase_fc_group/wb_pconn_within_between"),
        type=Path,
        help="Output directory for paired group results.",
    )
    parser.add_argument("--hemisphere", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--min-pairs",
        default=3,
        type=int,
        help="Minimum paired subjects required to run a t-test for a metric.",
    )
    return parser.parse_args()


def parse_subject_id(subject_id: str) -> tuple[str, str]:
    match = CONDITION_RE.search(subject_id)
    if not match:
        raise ValueError(f"Cannot parse condition/subid from subject_id {subject_id!r}")
    return match.group("condition"), match.group("subid")


def discover_entries(root: Path) -> pd.DataFrame:
    rows = []
    manifests = sorted(root.glob("*/manifests/fc_batch_hemisphere_manifest.csv"))
    if not manifests:
        raise RuntimeError(f"No wb_command hemisphere manifests found under {root}")

    for manifest_path in manifests:
        manifest = pd.read_csv(manifest_path)
        required = {"subject_id", "hemisphere", "pconn", "status"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        ok_rows = manifest[manifest["status"].astype(str).str.lower() == "ok"]
        for row in ok_rows.itertuples(index=False):
            pconn_path = Path(row.pconn)
            if not pconn_path.exists():
                continue
            condition, subid = parse_subject_id(str(row.subject_id))
            rows.append(
                PconnEntry(
                    condition=condition,
                    subid=subid,
                    subject_id=str(row.subject_id),
                    hemisphere=str(row.hemisphere).lower(),
                    pconn_path=pconn_path,
                    manifest_path=manifest_path,
                ).__dict__
            )

    if not rows:
        raise RuntimeError(f"No usable pconn outputs found under {root}")
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
        raise RuntimeError("No default comparisons found. Pass --comparison Drug:Placebo explicitly.")
    return pairs


def paired_subjects(entries: pd.DataFrame, drug: str, placebo: str, hemisphere: str) -> list[str]:
    subset = entries[entries["hemisphere"] == hemisphere]
    drug_subs = set(subset[subset["condition"] == drug]["subid"])
    pcb_subs = set(subset[subset["condition"] == placebo]["subid"])
    return sorted(drug_subs & pcb_subs)


def load_pconn(path: Path, expected_n: int) -> np.ndarray:
    data = np.asarray(nib.load(path).get_fdata(dtype=np.float32), dtype=np.float32)
    data = np.squeeze(data)
    if data.shape != (expected_n, expected_n):
        raise ValueError(f"Expected {expected_n}x{expected_n} pconn matrix, got {data.shape}: {path}")
    data = data.copy()
    np.fill_diagonal(data, np.nan)
    return data


def load_fc_for(entries: pd.DataFrame, condition: str, subid: str, hemisphere: str, expected_n: int) -> tuple[np.ndarray, Path]:
    row = entries[
        (entries["condition"] == condition)
        & (entries["subid"] == subid)
        & (entries["hemisphere"] == hemisphere)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one pconn for {condition} S{subid} {hemisphere}, found {len(row)}")
    pconn_path = Path(row.iloc[0]["pconn_path"])
    return load_pconn(pconn_path, expected_n=expected_n), pconn_path


def plot_matrix_heatmap(mat: pd.DataFrame, out_path: Path, title: str, label: str) -> None:
    if mat.empty:
        return
    vals = mat.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    vmax = max(vmax, 1e-6)
    plt.figure(figsize=(7, 6))
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
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_parcel_delta_heatmap(delta_mean: np.ndarray, parcels: pd.DataFrame, out_path: Path, title: str) -> None:
    network_order = infer_network_order(parcels["network"])
    sort_key = parcels["network"].map({net: i for i, net in enumerate(network_order)}).fillna(999)
    order = np.lexsort((parcels["parcel_id"].to_numpy(), sort_key.to_numpy()))
    sorted_delta = delta_mean[np.ix_(order, order)]
    sorted_parcels = parcels.iloc[order].reset_index(drop=True)
    vmax = np.nanmax(np.abs(sorted_delta)) if np.isfinite(sorted_delta).any() else 1.0
    vmax = max(vmax, 1e-6)

    plt.figure(figsize=(9, 8))
    sns.heatmap(
        sorted_delta,
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        cbar_kws={"label": "Drug - Placebo FC"},
    )
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


def analyze_comparison(
    entries: pd.DataFrame,
    atlas_parcels: Path,
    drug: str,
    placebo: str,
    hemisphere: str,
    out_dir: Path,
    min_pairs: int,
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
        drug_fc, drug_path = load_fc_for(entries, drug, subid, hemisphere, expected_n=len(parcels))
        pcb_fc, pcb_path = load_fc_for(entries, placebo, subid, hemisphere, expected_n=len(parcels))
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

        subject_rows.append(
            {
                "comparison": f"{drug}-{placebo}",
                "hemisphere": hemisphere,
                "subid": subid,
                "drug_pconn": str(drug_path),
                "placebo_pconn": str(pcb_path),
            }
        )

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
        row["mean_drug"] = float(np.nanmean(group["drug_value"]))
        row["mean_placebo"] = float(np.nanmean(group["placebo_value"]))
        stats_rows.append(row)
    net_stats = pd.DataFrame(stats_rows)
    net_stats.to_csv(hemi_out / "network_paired_ttests.csv", index=False)

    net_mat = network_matrix(net_stats, "mean_delta")
    net_mat.to_csv(hemi_out / "network_delta_matrix.csv")
    plot_matrix_heatmap(
        net_mat,
        hemi_out / "network_delta_matrix.png",
        f"{drug} - {placebo} network delta ({hemisphere})",
        "Drug - Placebo FC",
    )
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

    plot_parcel_delta_heatmap(
        delta_mean,
        parcels,
        hemi_out / "parcel_delta_matrix.png",
        f"{drug} - {placebo} parcel delta ({hemisphere})",
    )
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

    return net_stats


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = discover_entries(args.fc_root)
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
            )
            if not stats_df.empty:
                all_stats.append(stats_df)

    if all_stats:
        pd.concat(all_stats, ignore_index=True).to_csv(args.out_dir / "all_network_paired_ttests.csv", index=False)

    seg_files = sorted(args.out_dir.glob("*_minus_*/*/segregation_paired_ttests.csv"))
    if seg_files:
        pd.concat((pd.read_csv(path) for path in seg_files), ignore_index=True).to_csv(
            args.out_dir / "all_segregation_paired_ttests.csv",
            index=False,
        )

    summary = {
        "fc_root": str(args.fc_root),
        "atlas_parcels": str(args.atlas_parcels),
        "comparisons": [{"drug": d, "placebo": p} for d, p in comparisons],
        "hemispheres": hemispheres,
        "n_entries": int(len(entries)),
        "note": "Diagonal pconn values are set to NaN before analysis.",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote wb_command pconn group analysis to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
