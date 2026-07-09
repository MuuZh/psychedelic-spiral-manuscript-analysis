import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 6,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.labelcolor": "black",
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 6,
    "figure.titlesize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pgf.rcfonts": False,
})

COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(COMMON))
from figure_utils import append_manifest, cli, log, panel_row
from loaders import read_table
from paths import OUTPUT_ROOT, ensure_output_dirs
from style import HEMISPHERE_ORDER, STUDY_ORDER, apply_style

SCRIPT = Path(__file__).name
FIGURE_ID = "FIG5_MODEL_EFFECT_HEATMAPS"
OUT_ROOT = OUTPUT_ROOT / "fig5"
PNG_PATH = OUT_ROOT / "fig5_model_effect_heatmaps_composite.png"
PDF_PATH = OUT_ROOT / "fig5_model_effect_heatmaps_composite.pdf"
BOTTOM_PNG_PATH = OUT_ROOT / "fig5_model_effect_heatmaps_composite_bottom_colorbars.png"
BOTTOM_PDF_PATH = OUT_ROOT / "fig5_model_effect_heatmaps_composite_bottom_colorbars.pdf"
RIGHT_PNG_PATH = OUT_ROOT / "fig5_model_effect_heatmaps_composite_right_colorbars.png"
RIGHT_PDF_PATH = OUT_ROOT / "fig5_model_effect_heatmaps_composite_right_colorbars.pdf"

ROW_ORDER = [
    ("NGSC", "NGSC"),
    ("cSVD top-1 energy", "cSVD"),
    ("wFC", "wFC"),
    ("bFC", "bFC"),
    ("CSD region count", "CSD"),
    ("CAI", "CAI"),
]
COLUMN_ORDER = [(study, hemisphere) for study in STUDY_ORDER for hemisphere in HEMISPHERE_ORDER]
COLUMN_LABELS = [f"{study}-{hemisphere[0].upper()}" for study, hemisphere in COLUMN_ORDER]
DZ_LIMIT_STEP = 0.5

SOURCES = {
    "NGSC": {
        "source_export_id": "EXP023",
        "source_file": "02_exports/phase_ngsc_model/exp023_phase_ngsc_paired_deltas.csv",
        "metric": "phase_ngsc_delta",
        "value_scale": "native",
    },
    "cSVD top-1 energy": {
        "source_export_id": "EXP016",
        "source_file": "02_exports/cai_model_delta/model_correlation/exp016_empirical_model_joined_values.csv",
        "metric": "csvd_complex_svd_top1_energy",
        "value_scale": "native",
    },
    "wFC": {
        "source_export_id": "EXP014",
        "source_file": "02_exports/phase_recon_wbfc/model_correlation/exp014_original_recon_joined_values_z.csv",
        "metric": "global_within_edge_weighted",
        "value_scale": "Fisher z",
    },
    "bFC": {
        "source_export_id": "EXP014",
        "source_file": "02_exports/phase_recon_wbfc/model_correlation/exp014_original_recon_joined_values_z.csv",
        "metric": "global_between_edge_weighted",
        "value_scale": "Fisher z",
    },
    "CSD region count": {
        "source_export_id": "EXP018",
        "source_file": "02_exports/dfr/model_correlation/exp018_per_subject_dfr_drug_minus_pcb_delta.csv",
        "metric": "delta_region_count",
        "value_scale": "native",
    },
    "CAI": {
        "source_export_id": "EXP016",
        "source_file": "02_exports/cai_model_delta/model_correlation/exp016_empirical_model_joined_values.csv",
        "metric": "cai_weighted_mean_cos2_alignment",
        "value_scale": "native",
    },
}


def cohen_dz(values):
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if len(series) < 2:
        return np.nan
    sd = series.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(series.mean() / sd)


def squared_correlation(x_values, y_values):
    frame = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(np.corrcoef(frame["x"], frame["y"])[0, 1] ** 2)


def symmetric_dz_limit(*matrices, step=DZ_LIMIT_STEP):
    """Return a shared symmetric limit that covers all finite effect sizes."""
    finite = np.concatenate([
        matrix.to_numpy(dtype=float).ravel() for matrix in matrices
    ])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(step)
    maximum = float(np.max(np.abs(finite)))
    return float(np.ceil(maximum / step) * step)


def empty_results():
    rows = [short for _, short in ROW_ORDER]
    columns = COLUMN_LABELS
    return (
        pd.DataFrame(np.nan, index=rows, columns=columns),
        pd.DataFrame(np.nan, index=rows, columns=columns),
        pd.DataFrame(np.nan, index=rows, columns=columns),
        pd.DataFrame(0, index=rows, columns=columns),
    )


def set_cell(results, row_key, study, hemisphere, empirical, model):
    r2_matrix, empirical_dz_matrix, model_dz_matrix, n_matrix = results
    short = dict(ROW_ORDER)[row_key]
    column = f"{study}-{hemisphere[0].upper()}"
    paired = pd.DataFrame({"empirical": empirical, "model": model}).dropna()
    r2_matrix.loc[short, column] = squared_correlation(paired["empirical"], paired["model"])
    empirical_dz_matrix.loc[short, column] = cohen_dz(paired["empirical"])
    model_dz_matrix.loc[short, column] = cohen_dz(paired["model"])
    n_matrix.loc[short, column] = len(paired)


def collect_matrices():
    results = empty_results()

    ngsc = read_table(SOURCES["NGSC"]["source_file"], ["dataset", "source", "hemisphere", "subid", "phase_ngsc_delta"])
    for study, hemisphere in COLUMN_ORDER:
        panel = ngsc[(ngsc["dataset"].eq(study)) & (ngsc["hemisphere"].eq(hemisphere))]
        wide = panel.pivot_table(index="subid", columns="source", values="phase_ngsc_delta").dropna()
        set_cell(results, "NGSC", study, hemisphere, wide["orig"], wide["recon"])

    cai_model = read_table(
        SOURCES["CAI"]["source_file"],
        ["drug", "subid", "hemisphere", "metric", "value_kind", "empirical_value", "model_value"],
    )
    for row_key in ["cSVD top-1 energy", "CAI"]:
        subset = cai_model[(cai_model["metric"].eq(SOURCES[row_key]["metric"])) & (cai_model["value_kind"].eq("delta"))]
        for study, hemisphere in COLUMN_ORDER:
            panel = subset[(subset["drug"].eq(study)) & (subset["hemisphere"].eq(hemisphere))]
            set_cell(results, row_key, study, hemisphere, panel["empirical_value"], panel["model_value"])

    fc = read_table(
        SOURCES["wFC"]["source_file"],
        ["drug", "subid", "hemisphere", "metric", "value_kind", "original_value_z", "recon_value_z"],
    )
    for row_key in ["wFC", "bFC"]:
        subset = fc[(fc["metric"].eq(SOURCES[row_key]["metric"])) & (fc["value_kind"].eq("delta"))]
        for study, hemisphere in COLUMN_ORDER:
            panel = subset[(subset["drug"].eq(study)) & (subset["hemisphere"].eq(hemisphere))]
            set_cell(results, row_key, study, hemisphere, panel["original_value_z"], panel["recon_value_z"])

    csd = read_table(SOURCES["CSD region count"]["source_file"], ["source", "drug", "subid", "hemisphere", "delta_region_count"])
    for study, hemisphere in COLUMN_ORDER:
        panel = csd[(csd["drug"].eq(study)) & (csd["hemisphere"].eq(hemisphere))]
        wide = panel.pivot_table(index="subid", columns="source", values="delta_region_count").dropna()
        set_cell(results, "CSD region count", study, hemisphere, wide["empirical"], wide["recon"])

    return results


def contrast_text_colors(ax, values, cmap_name, vmin, vmax):
    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    flat = values.to_numpy(dtype=float).ravel()
    for text, value in zip(ax.texts, flat):
        if not np.isfinite(value):
            text.set_color("#303030")
            continue
        red, green, blue, _ = cmap(norm(value))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        text.set_color("black" if luminance > 0.55 else "white")


def draw_matrix(ax, matrix, title, cmap_name, vmin, vmax, show_y):
    data = matrix.to_numpy(dtype=float)
    cmap = mpl.colormaps[cmap_name]
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, pad=4)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index if show_y else [])
    ax.tick_params(axis="both", length=0, pad=1)
    ax.set_xlim(-0.5, len(matrix.columns) - 0.5)
    ax.set_ylim(len(matrix.index) - 0.5, -0.5)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="#E6E6E6", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            value = data[row_index, col_index]
            label = "" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=5.4)
    contrast_text_colors(ax, matrix, cmap_name, vmin, vmax)
    return im


def style_endpoint_colorbar(
    colorbar,
    low_label,
    high_label,
    orientation,
    label,
    label_rotation=None,
    labelpad=1,
):
    colorbar.set_ticks([])
    colorbar.outline.set_visible(False)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=0)
    label_kwargs = {}
    if label_rotation is not None:
        label_kwargs["rotation"] = label_rotation
    colorbar.set_label(label, fontsize=6, labelpad=labelpad, **label_kwargs)
    if orientation == "vertical":
        colorbar.ax.text(
            0.5, 1.015, high_label,
            transform=colorbar.ax.transAxes,
            ha="center", va="bottom", fontsize=5.5, color="black",
        )
        colorbar.ax.text(
            0.5, -0.015, low_label,
            transform=colorbar.ax.transAxes,
            ha="center", va="top", fontsize=5.5, color="black",
        )
    else:
        colorbar.ax.text(
            0.0, 1.18, low_label,
            transform=colorbar.ax.transAxes,
            ha="left", va="bottom", fontsize=5.5, color="black",
        )
        colorbar.ax.text(
            1.0, 1.18, high_label,
            transform=colorbar.ax.transAxes,
            ha="right", va="bottom", fontsize=5.5, color="black",
        )


def add_styled_colorbar(fig, image, cax, kind, orientation):
    if kind == "r2":
        cbar = fig.colorbar(image, cax=cax, orientation=orientation)
        style_endpoint_colorbar(
            cbar,
            "0",
            "1",
            orientation,
            "R\u00b2",
            label_rotation=0 if orientation == "vertical" else None,
            labelpad=4 if orientation == "vertical" else 1,
        )
    else:
        cbar = fig.colorbar(image, cax=cax, orientation=orientation)
        limit = max(abs(float(image.norm.vmin)), abs(float(image.norm.vmax)))
        style_endpoint_colorbar(
            cbar, f"-{limit:g}", f"{limit:g}", orientation, "Cohen's dz"
        )
    return cbar


def panel_rows_for_matrix(n_matrix):
    rows = []
    short_to_full = {short: full for full, short in ROW_ORDER}
    for short in n_matrix.index:
        row_key = short_to_full[short]
        source = SOURCES[row_key]
        for study, hemisphere in COLUMN_ORDER:
            column = f"{study}-{hemisphere[0].upper()}"
            rows.append(panel_row(
                f"{short.replace(' ', '_')}_{study}_{hemisphere}",
                source["source_export_id"],
                source["source_file"],
                int(n_matrix.loc[short, column]),
                source["metric"],
                dataset=study,
                hemisphere=hemisphere,
                value_scale=source["value_scale"],
                notes="Fig5 compact heatmap composite for R-squared, empirical dz, and model dz.",
            ))
    return rows


def save_figure_targets(fig, targets, dry_run, rows=None):
    ensure_output_dirs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if dry_run:
        panel_count = 0 if rows is None else len(rows)
        log(SCRIPT, f"DRY RUN fig5_model_effect_heatmaps_composite: {panel_count} panels")
        plt.close(fig)
        return

    written = []
    for target in targets:
        try:
            fig.savefig(target, bbox_inches="tight", pad_inches=0.02)
            written.append(target)
        except PermissionError:
            log(SCRIPT, f"skipped locked output {target}")
    plt.close(fig)

    if rows is None:
        log(SCRIPT, f"wrote {', '.join(target.name for target in written)}")
        return

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest_rows = []
    for target in written:
        for row in rows:
            manifest_rows.append({
                "figure_file": str(target.relative_to(OUTPUT_ROOT)),
                "figure_id": FIGURE_ID,
                "generated_by_script": SCRIPT,
                "generated_at": generated_at,
                **row,
            })
    if manifest_rows:
        append_manifest(manifest_rows)
    log(SCRIPT, f"wrote {', '.join(target.name for target in written)}; panels={len(rows)}")


def save_composite(fig, rows, dry_run, targets):
    save_figure_targets(fig, targets, dry_run, rows)


def save_single_heatmap(matrix, stem, title, cmap_name, vmin, vmax, show_y, dry_run):
    fig, ax = plt.subplots(figsize=(1.65, 2.05))
    draw_matrix(ax, matrix, title, cmap_name, vmin, vmax, show_y)
    fig.subplots_adjust(left=0.28 if show_y else 0.06, right=0.98, bottom=0.17, top=0.88)
    save_figure_targets(fig, [OUT_ROOT / f"{stem}.png", OUT_ROOT / f"{stem}.pdf"], dry_run)


def save_single_colorbar(kind, orientation, dry_run, dz_limit=2.0):
    cmap_name = "YlOrRd" if kind == "r2" else "RdBu_r"
    vmin, vmax = (0, 1) if kind == "r2" else (-dz_limit, dz_limit)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    image = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap_name])
    if orientation == "vertical":
        fig = plt.figure(figsize=(0.55, 1.85))
        cax = fig.add_axes([0.34, 0.08, 0.18, 0.84])
    else:
        fig = plt.figure(figsize=(1.75, 0.42))
        cax = fig.add_axes([0.08, 0.34, 0.84, 0.18])
    add_styled_colorbar(fig, image, cax, kind, orientation)
    save_figure_targets(
        fig,
        [
            OUT_ROOT / f"fig5_colorbar_{kind}_{orientation}.png",
            OUT_ROOT / f"fig5_colorbar_{kind}_{orientation}.pdf",
        ],
        dry_run,
    )


def main():
    args = cli("Plot compact Fig5 model-effect heatmap composite")
    if args.list_panels:
        print("\n".join([f"{row} x {column}" for row, _ in ROW_ORDER for column in COLUMN_LABELS]))
        return

    apply_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 6,
        "axes.titleweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    r2_matrix, empirical_dz_matrix, model_dz_matrix, n_matrix = collect_matrices()
    dz_limit = symmetric_dz_limit(empirical_dz_matrix, model_dz_matrix)

    rows = panel_rows_for_matrix(n_matrix)

    fig_bottom = plt.figure(figsize=(5.85, 2.75))
    gs_bottom = fig_bottom.add_gridspec(
        2, 3,
        height_ratios=[1, 0.085],
        width_ratios=[1, 1, 1],
        hspace=0.16,
        wspace=0.055,
    )
    ax_r2 = fig_bottom.add_subplot(gs_bottom[0, 0])
    ax_emp = fig_bottom.add_subplot(gs_bottom[0, 1])
    ax_mod = fig_bottom.add_subplot(gs_bottom[0, 2])
    cax_r2 = fig_bottom.add_subplot(gs_bottom[1, 0])
    cax_dz = fig_bottom.add_subplot(gs_bottom[1, 2])
    im_r2 = draw_matrix(ax_r2, r2_matrix, "R\u00b2", "YlOrRd", 0, 1, True)
    draw_matrix(ax_emp, empirical_dz_matrix, "Empirical dz", "RdBu_r", -dz_limit, dz_limit, False)
    im_mod = draw_matrix(ax_mod, model_dz_matrix, "Model dz", "RdBu_r", -dz_limit, dz_limit, False)
    add_styled_colorbar(fig_bottom, im_r2, cax_r2, "r2", "horizontal")
    add_styled_colorbar(fig_bottom, im_mod, cax_dz, "dz", "horizontal")
    fig_bottom.subplots_adjust(left=0.105, right=0.992, bottom=0.12, top=0.90)
    save_composite(fig_bottom, rows, args.dry_run, [BOTTOM_PNG_PATH, BOTTOM_PDF_PATH, PNG_PATH, PDF_PATH])

    fig_right = plt.figure(figsize=(6.25, 2.90))
    gs_right = fig_right.add_gridspec(
        1, 5,
        width_ratios=[1, 0.035, 1, 1, 0.035],
        wspace=0.075,
    )
    ax_r2 = fig_right.add_subplot(gs_right[0, 0])
    cax_r2 = fig_right.add_subplot(gs_right[0, 1])
    ax_emp = fig_right.add_subplot(gs_right[0, 2])
    ax_mod = fig_right.add_subplot(gs_right[0, 3])
    cax_dz = fig_right.add_subplot(gs_right[0, 4])
    im_r2 = draw_matrix(ax_r2, r2_matrix, "R\u00b2", "YlOrRd", 0, 1, True)
    draw_matrix(ax_emp, empirical_dz_matrix, "Empirical dz", "RdBu_r", -dz_limit, dz_limit, False)
    im_mod = draw_matrix(ax_mod, model_dz_matrix, "Model dz", "RdBu_r", -dz_limit, dz_limit, False)
    add_styled_colorbar(fig_right, im_r2, cax_r2, "r2", "vertical")
    add_styled_colorbar(fig_right, im_mod, cax_dz, "dz", "vertical")
    fig_right.subplots_adjust(left=0.100, right=0.990, bottom=0.16, top=0.90)
    save_composite(fig_right, rows, args.dry_run, [RIGHT_PNG_PATH, RIGHT_PDF_PATH])

    save_single_heatmap(r2_matrix, "fig5_heatmap_r2", "R\u00b2", "YlOrRd", 0, 1, True, args.dry_run)
    save_single_heatmap(empirical_dz_matrix, "fig5_heatmap_empirical_dz", "Empirical dz", "RdBu_r", -dz_limit, dz_limit, True, args.dry_run)
    save_single_heatmap(model_dz_matrix, "fig5_heatmap_model_dz", "Model dz", "RdBu_r", -dz_limit, dz_limit, True, args.dry_run)
    for orientation in ("horizontal", "vertical"):
        save_single_colorbar("r2", orientation, args.dry_run, dz_limit)
        save_single_colorbar("dz", orientation, args.dry_run, dz_limit)


if __name__ == "__main__":
    main()
