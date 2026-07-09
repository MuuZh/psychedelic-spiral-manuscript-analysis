from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .config import Config
from .utils import build_group_summary_df, save_fig, write_table, paired_t, unpaired_t

STRUCTURE_MAP = {"left": "CORTEX_LEFT", "right": "CORTEX_RIGHT"}


@dataclass
class NgscResult:
    value: float
    n_timepoints: int
    n_nodes: int
    n_components: int
    total_variance: float


def mean_center_columns(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    return x - mean


def zscore_columns(x: np.ndarray) -> np.ndarray:
    std = np.std(x, axis=0, ddof=1, keepdims=True)
    valid = std > 0
    x = x[:, valid.ravel()]
    std = std[:, valid.ravel()]
    return x / std


def compute_ngsc_result(x: np.ndarray, *, zscore: bool = False) -> NgscResult:
    """Paper-style NGSC: SVD energy entropy normalized by log(n_nodes)."""
    if x.ndim != 2:
        return NgscResult(float("nan"), 0, 0, 0, float("nan"))
    valid_cols = np.all(np.isfinite(x), axis=0)
    x = x[:, valid_cols]
    if x.size == 0:
        return NgscResult(float("nan"), int(x.shape[0]), 0, 0, float("nan"))
    x = mean_center_columns(x)
    if zscore:
        x = zscore_columns(x)
    else:
        keep = np.sum(x * x, axis=0) > 0
        x = x[:, keep]
    if x.size == 0:
        return NgscResult(float("nan"), int(x.shape[0]), 0, 0, float("nan"))
    n_time, n_nodes = x.shape
    if n_time < 2 or n_nodes <= 1:
        return NgscResult(float("nan"), int(n_time), int(n_nodes), 0, float("nan"))
    try:
        svals = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return NgscResult(float("nan"), int(n_time), int(n_nodes), 0, float("nan"))
    energy = svals ** 2
    total = float(np.sum(energy))
    n_components = int(svals.size)
    if total <= 0:
        return NgscResult(float("nan"), int(n_time), int(n_nodes), n_components, total)
    p = energy / total
    p = p[p > 0]
    if p.size == 0:
        return NgscResult(float("nan"), int(n_time), int(n_nodes), n_components, total)
    entropy = -np.sum(p * np.log(p))
    ngsc = entropy / np.log(float(n_nodes))
    return NgscResult(float(ngsc), int(n_time), int(n_nodes), n_components, total)


def compute_ngsc_matrix(x: np.ndarray, *, zscore: bool = False) -> float:
    """Backward-compatible scalar wrapper for paper-style NGSC."""
    return compute_ngsc_result(x, zscore=zscore).value


def run_ngsc(cfg: Config, bundle_df: pd.DataFrame, summary: List[Dict]) -> None:
    try:
        from matphase.io.cifti import load_cifti
    except ImportError as exc:
        logging.warning("NGSC skipped: matphase.io.cifti unavailable (%s)", exc)
        return

    if bundle_df.empty:
        return
    target_groups = {cfg.group_pcb, cfg.group_drug}
    present = set(bundle_df["group"].unique())
    active_groups = present if len(present & target_groups) < 2 else target_groups
    df = bundle_df[bundle_df["group"].isin(active_groups)]

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="NGSC bundles", file=sys.stdout, dynamic_ncols=True):
        meta_path = Path(row["bundle_dir"]) / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        cifti_file = meta.get("cifti_file")
        if not cifti_file or not Path(cifti_file).exists():
            continue
        try:
            ts = load_cifti(cifti_file)
        except Exception as exc:
            logging.warning("Failed to load %s: %s", cifti_file, exc)
            continue
        hemi = row["hemisphere"]
        structure = STRUCTURE_MAP.get(hemi)
        if structure is None:
            continue
        try:
            data = ts.get_structure_data(structure)
        except Exception as exc:
            logging.warning("Missing structure %s in %s: %s", structure, cifti_file, exc)
            continue
        data_shape = tuple(int(v) for v in np.asarray(data).shape)
        x = np.asarray(data).T  # (time, nodes)
        ngsc_result = compute_ngsc_result(x, zscore=False)
        ngsc_val = ngsc_result.value

        phase_cube_path = Path(row["phase_cube"])
        phase_ngsc_val = float("nan")
        phase_shape = None
        phase_result = NgscResult(float("nan"), 0, 0, 0, float("nan"))
        if phase_cube_path.exists():
            try:
                phase_cube = np.load(phase_cube_path, mmap_mode="r")
                phase_shape = tuple(int(v) for v in phase_cube.shape)
                if phase_cube.ndim == 3:
                    phase_x = np.asarray(phase_cube, dtype=float).reshape(-1, phase_cube.shape[2]).T
                    phase_result = compute_ngsc_result(phase_x, zscore=False)
                    phase_ngsc_val = phase_result.value
            except Exception as exc:
                logging.warning("Failed to load phase cube %s: %s", phase_cube_path, exc)
        records.append(
            {
                "group": row["group"],
                "subid": row["subid"],
                "hemisphere": hemi,
                "ngsc": ngsc_val,
                "ngsc_zscore": False,
                "ngsc_raw_shape": str(data_shape),
                "ngsc_final_shape": str((ngsc_result.n_timepoints, ngsc_result.n_nodes)),
                "ngsc_n_timepoints": ngsc_result.n_timepoints,
                "ngsc_n_nodes": ngsc_result.n_nodes,
                "ngsc_n_components": ngsc_result.n_components,
                "ngsc_total_variance": ngsc_result.total_variance,
                "phase_ngsc": phase_ngsc_val,
                "phase_ngsc_zscore": False,
                "phase_raw_shape": str(phase_shape) if phase_shape is not None else "",
                "phase_final_shape": str((phase_result.n_timepoints, phase_result.n_nodes)),
                "phase_n_timepoints": phase_result.n_timepoints,
                "phase_n_nodes": phase_result.n_nodes,
                "phase_n_components": phase_result.n_components,
                "phase_total_variance": phase_result.total_variance,
            }
        )

    if not records:
        logging.warning("No NGSC records computed.")
        return

    subj_df = pd.DataFrame(records)
    out_dir = cfg.output_root / cfg.results_prefix / "ngsc"
    write_table(subj_df, out_dir / "per_subject.csv")

    metrics = ["ngsc", "phase_ngsc"]
    for hemi in ["left", "right"]:
        hemi_df = subj_df[subj_df["hemisphere"] == hemi]
        for metric in metrics:
            pivot = hemi_df.pivot(index="subid", columns="group", values=metric).dropna()
            summary.append({"section": "ngsc", "metric": metric, "hemisphere": hemi, "comparison": "paired_drug_vs_pcb",
                            **paired_t(pivot.get(cfg.group_drug, pd.Series(dtype=float)), pivot.get(cfg.group_pcb, pd.Series(dtype=float)))})
            drug = hemi_df[hemi_df["group"] == cfg.group_drug][metric]
            pcb = hemi_df[hemi_df["group"] == cfg.group_pcb][metric]
            summary.append({"section": "ngsc", "metric": metric, "hemisphere": hemi, "comparison": "unpaired_drug_vs_pcb",
                            **unpaired_t(drug, pcb)})

    write_table(build_group_summary_df(subj_df, metrics, cfg), out_dir / "group_summary.csv")

    sns.set_theme(style="whitegrid", context="talk")
    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, hemi in zip(axes, ["left", "right"]):
            tidy = subj_df[subj_df["hemisphere"] == hemi]
            sns.violinplot(data=tidy, x="group", y=metric, order=[cfg.group_pcb, cfg.group_drug], ax=ax, cut=0)
            ax.set_title(f"{metric} ({hemi})")
        fig.suptitle(f"{metric} per hemisphere")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, out_dir / f"violin_{metric}.png", cfg.save_plots)

    for metric in metrics:
        for hemi in ["left", "right"]:
            tidy = subj_df[subj_df["hemisphere"] == hemi][["subid", "group", metric]]
            if tidy.empty:
                continue
            pivot = tidy.pivot(index="subid", columns="group", values=metric).dropna()
            if pivot.empty or cfg.group_pcb not in pivot or cfg.group_drug not in pivot:
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.violinplot(data=tidy, x="group", y=metric, order=[cfg.group_pcb, cfg.group_drug], cut=0, ax=ax)
            for _, row in pivot.iterrows():
                ax.plot([0, 1], [row[cfg.group_pcb], row[cfg.group_drug]], color="gray", alpha=0.5, linewidth=0.9, zorder=1)
                ax.scatter([0, 1], [row[cfg.group_pcb], row[cfg.group_drug]], color=["#4575b4", "#d73027"], edgecolor="white", linewidth=0.5, s=25, zorder=2)
            t_res = paired_t(pivot[cfg.group_drug], pivot[cfg.group_pcb])
            ax.set_title(f"{metric} paired ({hemi}) | p={t_res['p']:.3g}")
            ax.set_xlabel("")
            fig.tight_layout()
            save_fig(fig, out_dir / f"paired_{metric}_{hemi}.png", cfg.save_plots)
