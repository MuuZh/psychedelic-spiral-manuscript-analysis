"""Plot amplified paired Drug-minus-PCB CAI angular differences."""

from __future__ import annotations

import os
import sys
from pathlib import Path

TEXMFCACHE = Path(__file__).resolve().parents[2] / ".texlive-cache"
TEXMFCACHE.mkdir(exist_ok=True)
for variable in ("TEXMFCACHE", "TEXMFVAR", "TEXMFCONFIG"):
    os.environ[variable] = TEXMFCACHE.as_posix()

import matplotlib as mpl

mpl.use("pgf")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 7,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.labelcolor": "black",
    "axes.unicode_minus": False,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "figure.titlesize": 7,
    "pgf.texsystem": "lualatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{fontspec}\setsansfont{Arial}\setmainfont{Arial}\renewcommand{\familydefault}{\sfdefault}",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(COMMON))

from figure_utils import cli, panel_row, save_figure
from paths import export_path
from style import PCB_COLOR, STUDY_DRUG_COLORS, apply_style
from layout_config import MAIN_FIGURE_WIDTH_IN


STATS_REL = "02_exports/cai_polar/angle_distributions/exp017_paired_bin_statistics.csv"
SUMMARY_REL = "02_exports/cai_polar/angle_distributions/exp017_subject_circular_summaries.csv"
NPZ_REL = "02_exports/cai_polar/angle_distributions/exp017_subject_angle_histograms.npz"
STUDIES = ["DMT", "LSD"]
HEMISPHERES = ["left", "right"]
POSITIVE_COLOR = "#B2182B"
NEGATIVE_COLOR = "#2166AC"
PCB_CONTOUR_COLOR = "#777777"
PCB_CONTOUR_ALPHA = 0.72
DRUG_CONTOUR_ALPHA = 0.82
SMOOTHING_BINS = 3
DISPLAY_BINS = 288
FIGURE_SCALE = 0.90
CONTOUR_BOTTOM_FACTOR = 1.10
CONTOUR_HEIGHT_FACTOR = 0.16
DRUG_RELATIVE_TO_PCB_GAIN = 2.5
REFERENCE_CONTOUR_LINEWIDTH = 1.1
DIFFERENCE_OUTLINE_LINEWIDTH = 0.6


def circular_smooth(values: np.ndarray, window: int = SMOOTHING_BINS) -> np.ndarray:
    if window % 2 != 1:
        raise ValueError("Circular smoothing window must be odd")
    radius = window // 2
    return np.mean([np.roll(values, shift) for shift in range(-radius, radius + 1)], axis=0)


def periodic_interpolate(theta: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extended_theta = np.r_[theta, theta[0] + 2.0 * np.pi]
    extended_values = np.r_[values, values[0]]
    display_theta = np.linspace(theta[0], theta[0] + 2.0 * np.pi, DISPLAY_BINS, endpoint=False)
    return display_theta, CubicSpline(extended_theta, extended_values, bc_type="periodic")(display_theta)


def configure_polar_axis(ax, maximum: float) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(
        ["0", "π/4", "π/2", "3π/4", "-π (π)", "-3π/4", "-π/2", "-π/4"]
    )
    ax.tick_params(axis="x", pad=-2)
    ax.set_ylim(0, maximum * 1.30)
    radial_ticks = np.linspace(0, maximum, 4)[1:]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick * 1e4:.1f}" for tick in radial_ticks], fontsize=5)
    ax.set_rlabel_position(25)
    ax.tick_params(axis="y", pad=1)
    ax.grid(color="#C7C7C7", linewidth=0.6, alpha=0.7)


def reference_contours(
    summary: pd.DataFrame,
    density: np.ndarray,
    study: str,
    hemisphere: str,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    panel = summary[(summary.dataset == study) & (summary.hemisphere == hemisphere)]
    means = {}
    for condition in ("PCB", "Drug"):
        indices = panel[panel.condition == condition]["subject_index"].to_numpy(dtype=int)
        means[condition] = density[indices].mean(axis=0)
    bottom = maximum * CONTOUR_BOTTOM_FACTOR
    height = maximum * CONTOUR_HEIGHT_FACTOR
    pcb_low = float(means["PCB"].min())
    pcb_span = float(means["PCB"].max() - pcb_low)
    if pcb_span <= np.finfo(float).eps:
        return np.full_like(means["PCB"], bottom), np.full_like(means["Drug"], bottom)
    pcb_radius = bottom + (means["PCB"] - pcb_low) / pcb_span * height
    drug_radius = (
        pcb_radius
        + DRUG_RELATIVE_TO_PCB_GAIN
        * (means["Drug"] - means["PCB"])
        / pcb_span
        * height
    )
    return pcb_radius, drug_radius


def colored_outline(ax, theta: np.ndarray, radius: np.ndarray, signed_values: np.ndarray) -> None:
    closed_theta = np.r_[theta, theta[0] + 2.0 * np.pi]
    closed_radius = np.r_[radius, radius[0]]
    points = np.column_stack([closed_theta, closed_radius])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    positive = signed_values >= 0
    for mask, color, zorder in (
        (~positive, NEGATIVE_COLOR, 3),
        (positive, POSITIVE_COLOR, 4),
    ):
        ax.add_collection(
            LineCollection(
                segments[mask],
                colors=color,
                linewidths=DIFFERENCE_OUTLINE_LINEWIDTH,
                alpha=0.95,
                zorder=zorder,
            )
        )


def draw_panel(
    ax,
    panel: pd.DataFrame,
    maximum: float,
    pcb_contour: np.ndarray,
    drug_contour: np.ndarray,
) -> None:
    theta = panel["bin_center_rad"].to_numpy()
    raw_mean = panel["mean_delta_drug_minus_pcb"].to_numpy()
    smooth_mean = circular_smooth(raw_mean)
    display_theta, display_mean = periodic_interpolate(theta, smooth_mean)
    radius = np.abs(display_mean)
    width = 2.0 * np.pi / DISPLAY_BINS
    positive = display_mean >= 0
    ax.bar(
        display_theta[~positive],
        radius[~positive],
        width=width * 1.04,
        color=NEGATIVE_COLOR,
        alpha=0.74,
        edgecolor="none",
        zorder=1,
    )
    ax.bar(
        display_theta[positive],
        radius[positive],
        width=width * 1.04,
        color=POSITIVE_COLOR,
        alpha=0.74,
        edgecolor="none",
        zorder=2,
    )
    colored_outline(ax, display_theta, radius, display_mean)
    pcb_theta, pcb_display = periodic_interpolate(theta, pcb_contour)
    drug_theta, drug_display = periodic_interpolate(theta, drug_contour)
    ax.plot(
        np.r_[pcb_theta, pcb_theta[0] + 2.0 * np.pi],
        np.r_[pcb_display, pcb_display[0]],
        color=PCB_CONTOUR_COLOR,
        lw=REFERENCE_CONTOUR_LINEWIDTH,
        linestyle=(0, (4, 2)),
        alpha=PCB_CONTOUR_ALPHA,
        zorder=4,
    )
    ax.plot(
        np.r_[drug_theta, drug_theta[0] + 2.0 * np.pi],
        np.r_[drug_display, drug_display[0]],
        color=STUDY_DRUG_COLORS[str(panel.iloc[0]["dataset"])],
        lw=REFERENCE_CONTOUR_LINEWIDTH,
        alpha=DRUG_CONTOUR_ALPHA,
        zorder=5,
    )
    row = panel.iloc[0]
    configure_polar_axis(ax, maximum)
    ax.text(
        -0.08,
        1.08,
        f"{row['dataset']} | {str(row['hemisphere']).capitalize()}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="black",
        zorder=10,
        clip_on=False,
    )


def main() -> None:
    args = cli("Plot paired CAI angular-distribution differences.")
    panels = [(study, hemisphere) for study in STUDIES for hemisphere in HEMISPHERES]
    if args.list_panels:
        for study, hemisphere in panels:
            print(f"{study}_{hemisphere}")
        return
    stats = pd.read_csv(export_path(STATS_REL))
    summary = pd.read_csv(export_path(SUMMARY_REL))
    with np.load(export_path(NPZ_REL)) as arrays:
        density = arrays["subject_density"]
    stats["smoothed_mean"] = stats.groupby(["dataset", "hemisphere"])["mean_delta_drug_minus_pcb"].transform(
        lambda values: circular_smooth(values.to_numpy())
    )
    maximum = float(stats["smoothed_mean"].abs().max())
    apply_style()
    fig, axes = plt.subplots(
        2, 2,
        figsize=(
            MAIN_FIGURE_WIDTH_IN * FIGURE_SCALE,
            MAIN_FIGURE_WIDTH_IN * 5.3 / 6.5 * FIGURE_SCALE,
        ),
        subplot_kw={"projection": "polar"},
    )
    rows = []
    for ax, (study, hemisphere) in zip(axes.flat, panels):
        panel = stats[(stats.dataset == study) & (stats.hemisphere == hemisphere)].sort_values("bin_index")
        pcb_contour, drug_contour = reference_contours(summary, density, study, hemisphere, maximum)
        draw_panel(ax, panel, maximum, pcb_contour, drug_contour)
        rows.append(
            panel_row(
                f"cai_paired_difference_{study}_{hemisphere}",
                "EXP017",
                STATS_REL,
                len(panel),
                "paired CAI angular probability difference",
                dataset=study,
                hemisphere=hemisphere,
                value_scale="absolute Drug-minus-PCB probability difference x10^-4; color encodes sign",
                notes="3-bin circular smoothing and periodic interpolation to 288 display bins; colored outline follows difference sign.",
            )
        )
        used = summary[(summary.dataset == study) & (summary.hemisphere == hemisphere)]
        rows.append(
            panel_row(
                f"cai_absolute_reference_contours_{study}_{hemisphere}",
                "EXP017",
                NPZ_REL,
                f"subject_density rows={';'.join(map(str, used.subject_index))}",
                "PCB-scaled absolute-probability reference contours",
                dataset=study,
                hemisphere=hemisphere,
                value_scale="PCB-normalized outer reference band with Drug-minus-PCB separation amplified 2.5x relative to the PCB span; not the difference radial scale",
                notes="PCB gray dashed and normalized to a common display height; Drug uses study color and is displaced from PCB in proportion to the native local Drug-minus-PCB difference.",
            )
        )
    legend = [
        Line2D([0], [0], color=POSITIVE_COLOR, lw=8, alpha=0.74, label="Drug > PCB"),
        Line2D([0], [0], color=NEGATIVE_COLOR, lw=8, alpha=0.74, label="Drug < PCB"),
        Line2D([0], [0], color=PCB_COLOR, lw=REFERENCE_CONTOUR_LINEWIDTH, linestyle=(0, (4, 2)), alpha=PCB_CONTOUR_ALPHA, label="PCB contour"),
        Line2D([0], [0], color=STUDY_DRUG_COLORS["DMT"], lw=REFERENCE_CONTOUR_LINEWIDTH, alpha=DRUG_CONTOUR_ALPHA, label="DMT contour"),
        Line2D([0], [0], color=STUDY_DRUG_COLORS["LSD"], lw=REFERENCE_CONTOUR_LINEWIDTH, alpha=DRUG_CONTOUR_ALPHA, label="LSD contour"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.tight_layout(rect=[0, 0.065, 1, 1], w_pad=0.2, h_pad=0.6)
    fig.subplots_adjust(wspace=-0.28)
    save_figure(
        fig,
        "cai_paired_difference_polar",
        "FIG_CAI_PAIRED_DIFFERENCE_POLAR",
        rows,
        Path(__file__).name,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
