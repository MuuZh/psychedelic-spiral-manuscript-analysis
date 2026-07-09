"""Plot six model-delta correlations for both hemispheres as 12 panels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_fig4_model_delta_composite import (  # noqa: E402
    CORR_METRIC_STEMS,
    METRIC_BY_STEM,
    load_correlation_data,
    significance_stars,
)
from common.style import STUDY_DRUG_COLORS, STUDY_ORDER  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "extended_data"
PDF_PATH = OUTPUT_DIR / "extended_data_all_model_delta_correlations.pdf"
PNG_PATH = OUTPUT_DIR / "extended_data_all_model_delta_correlations.png"

# Matplotlib uses inches; PDF points are 1/72 inch.
FIGURE_WIDTH_IN = 500.0 / 72.0
FIGURE_HEIGHT_IN = 480.0 / 72.0
HEMISPHERES = ("left", "right")


def set_common_limits(ax, panel) -> None:
    values = np.concatenate([
        panel["empirical"].to_numpy(dtype=float),
        panel["model"].to_numpy(dtype=float),
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * 0.10, abs(hi) * 0.02, 1e-3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


ALL_METRIC_STEMS = ["ngsc", "csvd", "wfc", "bfc", "csd", "cai"]


def draw_panel(ax, stem: str, hemisphere: str) -> None:
    metric = METRIC_BY_STEM[stem]
    data = load_correlation_data(stem, metric)
    panel = data[data["hemisphere"].eq(hemisphere)].copy()

    for drug in STUDY_ORDER:
        drug_panel = panel[panel["drug"].eq(drug)]
        colour = STUDY_DRUG_COLORS[drug]
        ax.scatter(
            drug_panel["empirical"],
            drug_panel["model"],
            s=10,
            facecolor=colour,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.80,
            zorder=3,
        )
        if len(drug_panel) >= 2:
            sns.regplot(
                data=drug_panel,
                x="empirical",
                y="model",
                scatter=False,
                ci=95,
                n_boot=2000,
                seed=20260630,
                truncate=False,
                color=colour,
                line_kws={"linewidth": 0.9},
                ax=ax,
            )

    set_common_limits(ax, panel)
    ax.axline((0, 0), slope=1, color="#B5B5B5", linewidth=0.55,
              linestyle=":", zorder=1)
    hemi_label = "L" if hemisphere == "left" else "R"
    ax.set_title(f"{metric[1]} | {hemi_label}", fontsize=7, pad=2)
    ax.set_xlabel("Empirical Δ", fontsize=6)
    ax.set_ylabel("Model Δ", fontsize=6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1)

    annotation_y = 0.96
    for drug in STUDY_ORDER:
        drug_panel = panel[panel["drug"].eq(drug)]
        if len(drug_panel) >= 3:
            r_value, p_value = pearsonr(
                drug_panel["empirical"], drug_panel["model"]
            )
            annotation = (
                f"{drug}: R$^2$={r_value ** 2:.2f} "
                f"{significance_stars(p_value)}"
            )
        else:
            annotation = f"{drug}: n={len(drug_panel)}"
        ax.text(
            0.04,
            annotation_y,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.7,
            color=STUDY_DRUG_COLORS[drug],
        )
        annotation_y -= 0.10

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 7,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        squeeze=False,
    )
    panel_index = 0
    for stem in ALL_METRIC_STEMS:
        for hemisphere in HEMISPHERES:
            ax = axes.flat[panel_index]
            draw_panel(ax, stem, hemisphere)
            ax.text(
                -0.20,
                1.12,
                chr(ord("a") + panel_index),
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="top",
            )
            panel_index += 1

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.070,
        top=0.965,
        wspace=0.42,
        hspace=0.48,
    )
    pdf_path = args.output_dir / PDF_PATH.name
    png_path = args.output_dir / PNG_PATH.name
    # Do not use bbox_inches="tight": the PDF MediaBox must remain 500 x 480 pt.
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)

    for stem in ALL_METRIC_STEMS:
        for hemisphere in HEMISPHERES:
            panel_fig, panel_ax = plt.subplots(figsize=(1.65, 1.45))
            draw_panel(panel_ax, stem, hemisphere)
            panel_fig.subplots_adjust(
                left=0.22,
                right=0.98,
                bottom=0.20,
                top=0.86,
            )
            panel_fig.savefig(
                args.output_dir
                / f"drug_pcb_delta_correlation_{stem}_{hemisphere}.pdf",
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.close(panel_fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
