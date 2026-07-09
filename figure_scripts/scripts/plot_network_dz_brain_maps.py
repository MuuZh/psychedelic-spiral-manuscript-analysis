import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black", "legend.labelcolor": "black", "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7, "figure.titlesize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "pgf.rcfonts": False})
import nibabel as nib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(COMMON))
from figure_utils import cli, panel_row, save_figure
from loaders import read_table
from style import apply_style
from layout_config import MAIN_FIGURE_WIDTH_IN

SCRIPT = Path(__file__).name
SOURCE = "02_exports/network_spiral_metrics_v2/paired_ttest_raw_summary.csv"
TESTDATA = Path(r"<toolbox_root>\testdata")
SULC = TESTDATA / "sulc.32k_fs_LR.dscalar.nii"
DLABEL = Path(
    r"<cbig_root>\stable_projects\brain_parcellation"
    r"\Schaefer2018_LocalGlobal\Parcellations\HCP\fslr32k\cifti"
    r"\Schaefer2018_400Parcels_7Networks_order.dlabel.nii"
)
SURFACES = {
    "left": {
        "inflated": TESTDATA / "L.inflated.32k_fs_LR.surf.gii",
    },
    "right": {
        "inflated": TESTDATA / "R.inflated.32k_fs_LR.surf.gii",
    },
}
METRICS = [
    ("spiral_count_per_network_px", "Spiral density"),
    ("mean_spiral_size", "Spiral size"),
]
STUDIES = ["DMT", "LSD"]
HEMISPHERES = ["left", "right"]
CMAP = "RdBu_r"


def load_mesh(path):
    image = nib.load(path)
    arrays = [array.data for array in image.darrays]
    vertices = next(array for array in arrays if array.ndim == 2 and array.shape[1] == 3 and np.issubdtype(array.dtype, np.floating))
    faces = next(array for array in arrays if array.ndim == 2 and array.shape[1] == 3 and np.issubdtype(array.dtype, np.integer))
    return vertices, faces


def load_network_vertices():
    image = nib.load(DLABEL)
    values = np.asarray(image.get_fdata()).reshape(-1).astype(int)
    label_axis = image.header.get_axis(0)
    brain_axis = image.header.get_axis(1)
    label_table = label_axis.label[0]
    id_to_network = {}
    for parcel_id, (name, _rgba) in label_table.items():
        parts = str(name).split("_")
        if parcel_id > 0 and len(parts) >= 3:
            id_to_network[int(parcel_id)] = parts[2]
    result = {}
    for structure, data_slice, model in brain_axis.iter_structures():
        if "CORTEX_LEFT" in structure:
            hemisphere = "left"
        elif "CORTEX_RIGHT" in structure:
            hemisphere = "right"
        else:
            continue
        n_vertices = int(model.nvertices[structure])
        vertex_network = np.full(n_vertices, "", dtype=object)
        parcel_values = values[data_slice]
        for parcel_id in np.unique(parcel_values):
            network = id_to_network.get(int(parcel_id))
            if network:
                vertex_network[model.vertex[parcel_values == parcel_id]] = network
        result[hemisphere] = vertex_network
    return result


def load_cortical_scalar(path):
    image = nib.load(path)
    values = np.asarray(image.get_fdata()).reshape(-1)
    brain_axis = image.header.get_axis(1)
    result = {}
    for structure, data_slice, model in brain_axis.iter_structures():
        if "CORTEX_LEFT" in structure:
            hemisphere = "left"
        elif "CORTEX_RIGHT" in structure:
            hemisphere = "right"
        else:
            continue
        vertex_values = np.full(int(model.nvertices[structure]), np.nan, dtype=float)
        vertex_values[model.vertex] = values[data_slice]
        result[hemisphere] = vertex_values
    return result


def build_vertex_maps(summary, vertex_networks):
    maps = {}
    for metric, _ in METRICS:
        for study in STUDIES:
            for hemisphere in HEMISPHERES:
                rows = summary[
                    (summary.metric == metric)
                    & (summary.study == study)
                    & (summary.hemisphere == hemisphere)
                ]
                network_to_dz = dict(zip(rows.network, rows.dz))
                networks = vertex_networks[hemisphere]
                values = np.full(len(networks), np.nan, dtype=float)
                for network, dz in network_to_dz.items():
                    values[networks == network] = float(dz)
                maps[(metric, study, hemisphere)] = values
    return maps


def draw_flat_map(ax, mesh, values, vlim):
    vertices, faces = mesh
    face_vertex_values = values[faces]
    valid_counts = np.isfinite(face_vertex_values).sum(axis=1)
    face_values = np.full(len(faces), np.nan, dtype=float)
    valid = valid_counts > 0
    face_values[valid] = np.nansum(face_vertex_values[valid], axis=1) / valid_counts[valid]
    valid_faces = np.isfinite(face_values)
    collection = ax.tripcolor(
        vertices[:, 0], vertices[:, 1], faces[valid_faces], facecolors=face_values[valid_faces],
        cmap=CMAP, vmin=-vlim, vmax=vlim, edgecolors="none",
    )
    collection.set_rasterized(True)
    ax.set_aspect("equal")
    ax.set_axis_off()


def flat_figure(maps, meshes, vlim, args):
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.05), squeeze=False)
    rows = []
    for row, (metric, metric_label) in enumerate(METRICS):
        for study_index, study in enumerate(STUDIES):
            for hemi_index, hemisphere in enumerate(HEMISPHERES):
                column = study_index * 2 + hemi_index
                ax = axes[row, column]
                draw_flat_map(ax, meshes[(hemisphere, "flat")], maps[(metric, study, hemisphere)], vlim)
                ax.set_title(f"{study} | {hemisphere.capitalize()}", fontsize=7)
            axes[row, 0].text(
                -0.05, 0.5, metric_label, transform=axes[row, 0].transAxes,
                rotation=90, va="center", ha="right", fontsize=7,
            )
        rows.append(panel_row(
            f"flat_{metric}", "EXP008_V2", SOURCE, 28, metric,
            dataset="DMT;LSD", hemisphere="left;right", value_scale="Cohen's dz",
            notes="Schaefer 400 parcels mapped to 7-network dz on fsLR-32k flat surface",
        ))
    fig.suptitle("Network-specific spiral effects on flat cortical surfaces", fontsize=7, y=0.99)
    add_colorbar(fig, vlim, [0.25, 0.035, 0.5, 0.025])
    fig.tight_layout(rect=[0, 0.07, 1, 0.96], w_pad=0.3, h_pad=0.8)
    save_figure(fig, "publication_network_dz_flat", "PUB_NETWORK_DZ_FLAT", rows, SCRIPT, args.dry_run)


def draw_surface_map(ax, mesh, values, sulc, vlim, hemisphere, view, use_sulc_shading=True):
    vertices, faces = mesh
    triangles = vertices[faces]
    face_vertex_values = values[faces]
    valid_counts = np.isfinite(face_vertex_values).sum(axis=1)
    face_values = np.full(len(faces), np.nan, dtype=float)
    labeled = valid_counts > 0
    face_values[labeled] = np.nansum(face_vertex_values[labeled], axis=1) / valid_counts[labeled]

    face_colors = np.full((len(faces), 4), (0.72, 0.72, 0.72, 1.0))
    face_colors[labeled] = plt.get_cmap(CMAP)(Normalize(-vlim, vlim)(face_values[labeled]))

    face_sulc = np.nanmean(sulc[faces], axis=1)
    sulc_low, sulc_high = np.nanpercentile(sulc, [2, 98])
    sulc_shade = 0.78 + 0.22 * np.clip((face_sulc - sulc_low) / (sulc_high - sulc_low), 0, 1)

    # Directional light always adds readable 3D relief; sulcal depth shading is optional.
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_lengths[:, None], np.finfo(float).eps)

    lateral_camera_x = -1.0 if hemisphere == "left" else 1.0
    camera_x = lateral_camera_x if view == "lateral" else -lateral_camera_x
    light = np.array([camera_x, -0.35, 0.8])
    light /= np.linalg.norm(light)
    illumination = np.clip(normals @ light, 0, 1)
    directional_shade = 0.55 + 0.45 * illumination
    shade = sulc_shade * directional_shade if use_sulc_shading else directional_shade
    face_colors[:, :3] *= shade[:, None]

    collection = Poly3DCollection(
        triangles, facecolors=face_colors, edgecolors="none", linewidths=0,
        antialiased=False, rasterized=True, clip_on=False,
    )
    ax.add_collection3d(collection)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins, zoom=1.25)
    ax.view_init(elev=0, azim=180 if camera_x < 0 else 0)
    ax.set_axis_off()
    ax.patch.set_visible(False)


def surface_figure(maps, meshes, sulc_maps, vlim, args, surface):
    map_order = [(metric, label, study) for metric, label in METRICS for study in STUDIES]
    views = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]
    fig = plt.figure(
        figsize=(MAIN_FIGURE_WIDTH_IN, MAIN_FIGURE_WIDTH_IN * 3.38 / 5.7)
    )
    rows = []
    for row, (metric, metric_label, study) in enumerate(map_order):
        for column, (hemisphere, view) in enumerate(views):
            ax = fig.add_subplot(4, 4, row * 4 + column + 1, projection="3d")
            values = maps[(metric, study, hemisphere)]
            draw_surface_map(
                ax, meshes[(hemisphere, surface)], values, sulc_maps[hemisphere],
                vlim, hemisphere, view, use_sulc_shading=(surface != "inflated"),
            )
            if row == 0:
                ax.set_title(
                    f"{hemisphere.capitalize()} {view}",
                    fontsize=7,
                    fontweight="normal",
                    pad=0,
                )
        axes_in_row = fig.axes[row * 4:(row + 1) * 4]
        axes_in_row[0].text2D(-0.06, 0.5, f"{metric_label}\n{study}", transform=axes_in_row[0].transAxes, va="center", ha="right", fontsize=7)
        rows.append(panel_row(
            f"{surface}_{metric}_{study}", "EXP008_V2", SOURCE, 14, metric,
            dataset=study, hemisphere="left;right", value_scale="Cohen's dz",
            notes=f"Schaefer 400 parcels mapped to 7-network dz on fsLR-32k {surface} surface with directional lighting; medial wall retained",
        ))
    fig.suptitle(f"Network-specific spiral effects on {surface} cortical surfaces", fontsize=7, y=0.99)
    add_colorbar(fig, vlim, [0.35, 0.078, 0.30, 0.024])
    fig.subplots_adjust(left=0.08, right=0.995, top=0.93, bottom=0.12, wspace=-0.20, hspace=-0.25)
    save_figure(
        fig, f"publication_network_dz_{surface}", f"PUB_NETWORK_DZ_{surface.upper()}",
        rows, SCRIPT, args.dry_run,
    )


def add_colorbar(fig, vlim, rect):
    colorbar_ax = fig.add_axes(rect)
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-vlim, vmax=vlim), cmap=CMAP),
        cax=colorbar_ax, orientation="horizontal",
    )
    colorbar.set_ticks([])
    colorbar.outline.set_visible(False)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    colorbar.set_label("Cohen's dz", fontsize=7, labelpad=1)
    colorbar.ax.tick_params(
        axis="x", which="both", labelbottom=False, labeltop=False,
        length=0, width=0,
    )
    colorbar.ax.text(
        -0.035, 0.5, f"{-vlim:g}",
        transform=colorbar.ax.transAxes,
        ha="right", va="center", fontsize=6,
    )
    colorbar.ax.text(
        1.035, 0.5, f"{vlim:g}",
        transform=colorbar.ax.transAxes,
        ha="left", va="center", fontsize=6,
    )


def main():
    args = cli("Map network-specific spiral dz values to inflated cortical surfaces")
    if args.list_panels:
        print("inflated: Spiral density; Spiral size")
        return
    summary = read_table(SOURCE, ["study", "hemisphere", "network", "metric", "dz"])
    summary = summary[summary.metric.isin([metric for metric, _ in METRICS])].copy()
    summary["dz"] = summary["dz"].astype(float)
    vertex_networks = load_network_vertices()
    sulc_maps = load_cortical_scalar(SULC)
    maps = build_vertex_maps(summary, vertex_networks)
    surfaces = ("inflated",)
    meshes = {
        (hemisphere, surface): load_mesh(paths[surface])
        for hemisphere, paths in SURFACES.items()
        for surface in surfaces
    }
    vlim = float(np.ceil(summary.dz.abs().max() * 10) / 10)
    apply_style()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})
    surface_figure(maps, meshes, sulc_maps, vlim, args, "inflated")


if __name__ == "__main__":
    main()
