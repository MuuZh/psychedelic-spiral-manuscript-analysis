import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black", "legend.labelcolor": "black", "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7, "figure.titlesize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "pgf.rcfonts": False})
import numpy as np
from matplotlib.colors import TwoSlopeNorm

COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(COMMON))
from figure_utils import cli, panel_row, save_figure
from loaders import read_manifest_array, read_table, standardize_condition, standardize_dataset, standardize_hemisphere
from style import apply_style
from layout_config import MAIN_FIGURE_WIDTH_IN

SCRIPT = Path(__file__).name
MANIFEST = "02_exports/all_metrics_runner/ptdr_vortex/exp003_vortex_occupancy_map_manifest.csv"


def crop_to_shared_valid_rectangle(arrays):
    """Crop all maps to the smallest rectangle containing any finite nonzero value."""
    if not arrays:
        raise ValueError("At least one PTDR map is required")
    reference_shape = arrays[0].shape
    if any(array.shape != reference_shape for array in arrays):
        raise ValueError(
            f"PTDR map shapes must match for shared cropping: "
            f"{[array.shape for array in arrays]}"
        )
    shared_valid = np.zeros(reference_shape, dtype=bool)
    for array in arrays:
        shared_valid |= np.isfinite(array) & (array != 0)
    valid_rows, valid_columns = np.where(shared_valid)
    if valid_rows.size == 0:
        raise ValueError("PTDR maps contain no finite nonzero values")
    row_slice = slice(int(valid_rows.min()), int(valid_rows.max()) + 1)
    column_slice = slice(int(valid_columns.min()), int(valid_columns.max()) + 1)
    return [array[row_slice, column_slice] for array in arrays], row_slice, column_slice


def main():
    args = cli("Plot left-hemisphere DMT study PTDR occupancy maps")
    manifest = read_table(
        MANIFEST,
        ["array_key", "dataset", "condition", "hemisphere", "map_type", "binary_file"],
    )
    selected = manifest[manifest.map_type == "group_mean_occupancy_fraction"].copy()
    selected["study"] = selected.dataset.map(standardize_dataset)
    selected["hemisphere"] = selected.hemisphere.map(standardize_hemisphere)
    selected["condition_role"] = selected.condition.map(standardize_condition)
    selected = selected[(selected.study == "DMT") & (selected.hemisphere == "left")]

    ordered_rows = []
    for condition in ("PCB", "Drug"):
        match = selected[selected.condition_role == condition]
        if len(match) != 1:
            raise ValueError(
                f"Expected one PTDR map for DMT study, left hemisphere, {condition}; "
                f"found {len(match)}"
            )
        ordered_rows.append(match.iloc[0])

    if args.list_panels:
        print("\n".join(row.array_key for row in ordered_rows))
        return

    arrays = [read_manifest_array(MANIFEST, row)[0] for row in ordered_rows]
    arrays, row_slice, column_slice = crop_to_shared_valid_rectangle(arrays)
    masked_arrays = [np.ma.masked_where((array == 0) | ~np.isfinite(array), array) for array in arrays]
    nonzero_values = [array.compressed() for array in masked_arrays if array.count()]
    if not nonzero_values:
        raise ValueError("The selected DMT and PCB maps contain no finite nonzero values")
    shared_values = np.concatenate(nonzero_values)
    vmin, vmax = float(shared_values.min()), float(shared_values.max())
    center = float(shared_values.mean())

    apply_style()
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(MAIN_FIGURE_WIDTH_IN, MAIN_FIGURE_WIDTH_IN * 2.62 / 5.45),
        squeeze=False,
    )
    source_rows = []
    image = None

    for ax, row, array in zip(axes.flat, ordered_rows, masked_arrays):
        image = ax.imshow(array, cmap=cmap, norm=norm, origin="lower")
        if array.count() and np.ptp(array.compressed()) > 0:
            ax.contour(array, levels=5, colors="black", linewidths=0.25, alpha=0.45, origin="lower")
        condition_label = "DRUG" if row.condition_role == "Drug" else "PCB"
        ax.set_title(condition_label, fontweight="normal")
        ax.axis("off")
        source_rows.append(
            panel_row(
                row.array_key,
                "EXP003",
                MANIFEST,
                row.array_key,
                "occupancy map",
                dataset="DMT",
                hemisphere="left",
                value_scale="occupancy probability",
                notes=(
                    "Zero and non-finite values masked as transparent; both maps cropped "
                    f"to shared minimum valid rectangle rows "
                    f"{row_slice.start}:{row_slice.stop}, columns "
                    f"{column_slice.start}:{column_slice.stop}; y-axis reversed with "
                    "origin=lower; shared RdBu_r scale centered on pooled mean"
                ),
            )
        )

    colorbar_ax = fig.add_axes([0.39, 0.17, 0.22, 0.025])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([])
    colorbar.set_label("Occupancy probability", fontsize=7, labelpad=1)
    colorbar.ax.tick_params(
        axis="x", which="both", labelbottom=False, labeltop=False,
        length=0, width=0,
    )
    colorbar.outline.set_visible(False)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    colorbar.ax.text(
        -0.035, 0.5, "0",
        transform=colorbar.ax.transAxes,
        ha="right", va="center", fontsize=6,
    )
    colorbar.ax.text(
        1.035, 0.5, f"{vmax:.3f}",
        transform=colorbar.ax.transAxes,
        ha="left", va="center", fontsize=6,
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.22, top=0.91, wspace=0.24)
    save_figure(fig, "ptdr_dmt_left_dmt_pcb", "FIG_PTDR_DMT_LEFT", source_rows, SCRIPT, args.dry_run)


if __name__ == "__main__":
    main()
