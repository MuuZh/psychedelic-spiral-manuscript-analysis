"""Physical export sizes and output routing for the rearranged figures."""

from pathlib import Path

POINTS_PER_INCH = 72.0

MAIN_FIGURE_WIDTH_PT = 500.0
SUPPLEMENT_GP_WIDTH_PT = 252.28
# The GP PDFs are saved with bbox_inches="tight"; this compensated canvas width
# yields a 252.28 pt tight-cropped comparison.pdf page.
SUPPLEMENT_GP_FIGSIZE_WIDTH_PT = 264.52
HALF_PANEL_WIDTH_PT = 244.0
COLUMN_GAP_PT = MAIN_FIGURE_WIDTH_PT - 2 * HALF_PANEL_WIDTH_PT


def inches(points: float) -> float:
    return points / POINTS_PER_INCH


MAIN_FIGURE_WIDTH_IN = inches(MAIN_FIGURE_WIDTH_PT)
SUPPLEMENT_GP_WIDTH_IN = inches(SUPPLEMENT_GP_WIDTH_PT)
SUPPLEMENT_GP_FIGSIZE_WIDTH_IN = inches(SUPPLEMENT_GP_FIGSIZE_WIDTH_PT)
HALF_PANEL_WIDTH_IN = inches(HALF_PANEL_WIDTH_PT)

VIOLIN_HEIGHT_IN = 1.55
DOUBLE_VIOLIN_HEIGHT_IN = 2 * VIOLIN_HEIGHT_IN + inches(COLUMN_GAP_PT)

LAYOUT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = LAYOUT_ROOT / "outputs"

STEM_TO_FIGURE = {
    "publication_paired_pattern_count_per_frame": "fig1",
    "publication_paired_mean_size": "fig1",
    "ptdr_dmt_left_dmt_pcb": "fig2",
    "publication_paired_gipr": "fig2",
    "publication_paired_occupancy_p95_p5_diff": "fig2",
    "publication_network_dz_inflated": "fig3",
    "publication_network_standardized_difference_spiral_count_and_size": "fig3",
    "publication_network_standardized_difference_legend": "fig3",
    "publication_paired_phase_ngsc": "fig4",
    "publication_paired_complex_svd_top_mode": "fig4",
    "publication_paired_global_phase_wfc": "fig4",
    "publication_paired_global_phase_bfc": "fig4",
    "publication_paired_gcor": "fig4",
    "publication_paired_dfr_mean_region_count": "fig4",
    "publication_paired_path_entropy": "fig4",
    "publication_paired_mean_duration": "fig4",
    "cai_paired_difference_polar": "fig6",
    "publication_paired_weighted_mean_cos2_alignment": "fig6",
}

def output_dir_for_stem(stem: str) -> Path:
    figure = STEM_TO_FIGURE.get(stem)
    if figure is None:
        raise KeyError(f"No rearranged-figure output route is defined for {stem!r}")
    return OUTPUT_ROOT / figure
