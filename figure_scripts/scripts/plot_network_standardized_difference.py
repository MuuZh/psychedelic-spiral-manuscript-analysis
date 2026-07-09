import sys
from pathlib import Path

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
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{fontspec}\setsansfont{Arial}\setmainfont{Arial}\renewcommand{\familydefault}{\sfdefault}",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import stats

COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(COMMON))
from figure_utils import cli, panel_row, save_figure
from loaders import NETWORK_ORDER, read_table
from style import STUDY_DRUG_COLORS, apply_style
from layout_config import MAIN_FIGURE_WIDTH_IN

SCRIPT = Path(__file__).name
SOURCE = "02_exports/network_spiral_metrics_v2/subject_standardized_difference_long.csv"
METRICS = [
    ("spiral_count_per_network_px", "Spiral density"),
    ("mean_spiral_size", "Spiral size"),
    ("weighted_mean_cos2_alignment", "Weighted mean cos(2theta)"),
]
COMBINED_METRICS = METRICS[:2]
COMBINED_METRIC_LABELS = {
    "spiral_count_per_network_px": "Spiral density",
    "mean_spiral_size": "Spiral size",
}
STUDIES = ["DMT", "LSD"]
COMBINED_FIGSIZE = (MAIN_FIGURE_WIDTH_IN, MAIN_FIGURE_WIDTH_IN * 2.35 / 5.7)
LEGEND_FIGSIZE = (2.6, 0.7)
NETWORK_ALIASES = {
    "Vis": "VIS",
    "SomMot": "SMN",
    "DorsAttn": "DAN",
    "SalVentAttn": "SAL",
    "Limbic": "LIM",
    "Cont": "FPN/Cont",
    "Default": "DMN",
}
HEMISPHERE_STYLE = {
    "left": {"direction": -1.0, "linestyle": "-"},
    "right": {"direction": 1.0, "linestyle": "--"},
}


def finite_limits(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    pad = max((hi - lo) * 0.2, 0.2)
    return lo - pad, hi + pad


def draw_distribution(ax, values, y_grid, baseline, direction, color, linestyle):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or np.std(finite) == 0:
        return
    density = stats.gaussian_kde(finite)(y_grid)
    density = density / density.max() * 0.32
    ax.plot(
        baseline + direction * density, y_grid, color=color, linestyle=linestyle,
        linewidth=1.35, alpha=0.92, zorder=3,
    )
    mean = float(finite.mean())
    ax.hlines(
        mean, baseline, baseline + direction * 0.10, color=color,
        linewidth=0.9, alpha=0.65, zorder=4,
    )


def draw_panel(ax, data, metric, show_x_labels):
    subset = data[data.metric == metric].copy()
    y_min, y_max = finite_limits(subset.standardized_difference)
    y_grid = np.linspace(y_min, y_max, 256)
    positions = {network: index for index, network in enumerate(NETWORK_ORDER)}
    for network in NETWORK_ORDER:
        baseline = positions[network]
        ax.vlines(baseline, y_min, y_max, color="#A5A5A5", linewidth=0.5, alpha=0.55, zorder=1)
        for hemisphere, hemi_style in HEMISPHERE_STYLE.items():
            for study in STUDIES:
                values = subset[
                    (subset.network == network)
                    & (subset.study == study)
                    & (subset.hemisphere == hemisphere)
                ].standardized_difference.to_numpy(dtype=float)
                draw_distribution(
                    ax, values, y_grid, baseline, hemi_style["direction"],
                    STUDY_DRUG_COLORS[study], hemi_style["linestyle"],
                )
    ax.axhline(0, color="#202020", linewidth=0.8, zorder=2)
    ax.set_xlim(-0.55, len(NETWORK_ORDER) - 0.45)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(
        [positions[network] for network in NETWORK_ORDER],
        [NETWORK_ALIASES[network] for network in NETWORK_ORDER],
    )
    ax.tick_params(axis="x", labelsize=6, length=0, labelbottom=show_x_labels)
    ax.tick_params(axis="y", labelsize=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", linewidth=0.45, alpha=0.32)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("")


def draw_vertical_legend(fig):
    legend_items = [
        ("DMT", STUDY_DRUG_COLORS["DMT"], "-", 0.875, 0.82),
        ("LSD", STUDY_DRUG_COLORS["LSD"], "-", 0.875, 0.66),
        ("LH", "#444444", "-", 0.875, 0.50),
        ("RH", "#444444", "--", 0.875, 0.34),
    ]
    for label, color, linestyle, x, y in legend_items:
        fig.add_artist(plt.Line2D(
            [x, x], [y - 0.055, y + 0.055], transform=fig.transFigure,
            color=color, linestyle=linestyle, linewidth=1.8,
        ))
        fig.text(
            x + 0.010, y, label, rotation=0, ha="left", va="center", fontsize=7,
        )


def draw_standalone_legend(fig):
    legend_items = [
        ("DMT", STUDY_DRUG_COLORS["DMT"], "-", 0.42, 0.86),
        ("LSD", STUDY_DRUG_COLORS["LSD"], "-", 0.42, 0.62),
        ("LH", "#444444", "-", 0.42, 0.38),
        ("RH", "#444444", "--", 0.42, 0.14),
    ]
    for label, color, linestyle, x, y in legend_items:
        fig.add_artist(plt.Line2D(
            [x, x], [y - 0.12, y + 0.12], transform=fig.transFigure,
            color=color, linestyle=linestyle, linewidth=1.8,
        ))
        fig.text(x + 0.035, y, label, ha="left", va="center", fontsize=7)


def main():
    args = cli("Plot selected network standardized-difference ridgelines")
    if args.list_panels:
        print("\n".join(metric for metric, _ in METRICS))
        return
    data = read_table(
        SOURCE,
        ["study", "subid", "hemisphere", "network", "metric", "standardized_difference"],
    )
    data = data[data.metric.isin([metric for metric, _ in METRICS])].copy()
    apply_style()
    plt.rcParams.update({
        "axes.grid": False,
        "axes.titleweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    combined_fig, combined_axes = plt.subplots(
        len(COMBINED_METRICS), 1, figsize=COMBINED_FIGSIZE, squeeze=False,
    )
    combined_rows = []
    for row_index, (metric, title) in enumerate(COMBINED_METRICS):
        subset = data[data.metric == metric]
        draw_panel(
            combined_axes[row_index, 0], data, metric,
            show_x_labels=row_index == len(COMBINED_METRICS) - 1,
        )
        combined_axes[row_index, 0].text(
            -0.085, 0.5, COMBINED_METRIC_LABELS[metric],
            transform=combined_axes[row_index, 0].transAxes,
            ha="center", va="center", rotation=90, fontsize=7, linespacing=0.9,
        )
        combined_rows.append(panel_row(
            metric, "EXP008_V2", SOURCE, len(subset), metric,
            dataset="DMT;LSD", hemisphere="left;right",
            value_scale="(Drug - PCB) / paired PCB SD",
            notes="Two continuous spiral-metric rows with vertical distributions across seven aliased network x positions; DMT red; LSD blue; left solid/left side; right dashed/right side",
        ))
    draw_vertical_legend(combined_fig)
    combined_fig.supylabel(
        "Standardized difference", fontsize=7, x=0.055, y=0.54,
    )
    combined_fig.subplots_adjust(
        left=0.15, right=0.83, bottom=0.12, top=0.96, hspace=0.04,
    )
    save_figure(
        combined_fig, "publication_network_standardized_difference_spiral_count_and_size",
        "PUB_NETWORK_STANDARDIZED_DIFFERENCE", combined_rows, SCRIPT, args.dry_run,
    )

    legend_fig = plt.figure(figsize=LEGEND_FIGSIZE)
    draw_standalone_legend(legend_fig)
    save_figure(
        legend_fig, "publication_network_standardized_difference_legend",
        "PUB_NETWORK_STANDARDIZED_DIFFERENCE_LEGEND",
        [panel_row(
            "legend", "EXP008_V2", SOURCE, 0, "legend",
            dataset="DMT;LSD", hemisphere="left;right",
            notes="standalone legend for network standardized-difference panels",
        )],
        SCRIPT, args.dry_run,
    )


if __name__ == "__main__":
    main()
