from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dfr_core import FrameDfr


PALETTE = {"PCB": "#4575b4", "Drug": "#d73027"}


def save_paired_plot(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot_table(index="subid", columns="role", values=metric, aggfunc="mean")
    if not {"PCB", "Drug"}.issubset(pivot.columns):
        return
    pivot = pivot[["PCB", "Drug"]].dropna()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    ax.boxplot([pivot["PCB"], pivot["Drug"]], labels=["PCB", "Drug"], showfliers=False)
    for _, row in pivot.iterrows():
        ax.plot([1, 2], [row["PCB"], row["Drug"]], color="0.55", alpha=0.55, linewidth=0.9)
        ax.scatter([1, 2], [row["PCB"], row["Drug"]], color=[PALETTE["PCB"], PALETTE["Drug"]], s=22, zorder=3)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_occupancy_map(avg_boundary: np.ndarray, atlas_boundary: np.ndarray | None, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg_boundary, cmap="magma", origin="lower")
    if atlas_boundary is not None:
        yy, xx = np.where(atlas_boundary)
        ax.scatter(xx, yy, s=0.6, c="#00d5ff", alpha=0.55)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    if df.empty or x not in df or y not in df:
        return
    work = df[[x, y]].dropna()
    if work.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(work[x], work[y], s=28, alpha=0.8)
    if len(work) >= 2:
        coef = np.polyfit(work[x], work[y], 1)
        xs = np.linspace(work[x].min(), work[x].max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_frame_qc(
    phase_slice: np.ndarray,
    frame_dfr: FrameDfr,
    atlas_boundary: np.ndarray | None,
    spiral_centers: pd.DataFrame | None,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axs = axes.ravel()

    im0 = axs[0].imshow(phase_slice, cmap="twilight", origin="lower")
    axs[0].set_title("Phase")
    fig.colorbar(im0, ax=axs[0], shrink=0.75)

    step = max(1, min(phase_slice.shape) // 25)
    yy, xx = np.mgrid[0 : phase_slice.shape[0] : step, 0 : phase_slice.shape[1] : step]
    axs[1].imshow(phase_slice, cmap="gray", origin="lower", alpha=0.25)
    axs[1].quiver(xx, yy, frame_dfr.unit_x[::step, ::step], frame_dfr.unit_y[::step, ::step], scale=35)
    axs[1].set_title("Circular Gradient Field")

    im2 = axs[2].imshow(frame_dfr.grad_mag, cmap="viridis", origin="lower")
    axs[2].contour(frame_dfr.boundary_mask, levels=[0.5], colors="white", linewidths=0.7)
    axs[2].set_title("Gradient Magnitude")
    fig.colorbar(im2, ax=axs[2], shrink=0.75)

    axs[3].imshow(frame_dfr.boundary_mask, cmap="gray", origin="lower")
    axs[3].set_title("DFR Boundary")

    axs[4].imshow(frame_dfr.labeled_regions, cmap="tab20", origin="lower", interpolation="nearest")
    axs[4].set_title("Labeled Regions")

    axs[5].imshow(frame_dfr.boundary_mask, cmap="gray", origin="lower")
    if atlas_boundary is not None:
        ay, ax = np.where(atlas_boundary)
        axs[5].scatter(ax, ay, s=0.8, c="#00d5ff", alpha=0.65)
    if spiral_centers is not None and not spiral_centers.empty:
        sc = axs[5].scatter(
            spiral_centers["center_x"],
            spiral_centers["center_y"],
            c=spiral_centers.get("boundary_distance", 0.0),
            cmap="coolwarm",
            s=18,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.9,
        )
        fig.colorbar(sc, ax=axs[5], shrink=0.75)
    axs[5].set_title("Network Overlay")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
