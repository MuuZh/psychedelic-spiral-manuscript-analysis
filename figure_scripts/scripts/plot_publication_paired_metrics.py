import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black", "legend.labelcolor": "black", "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7, "figure.titlesize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "pgf.rcfonts": False})

COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(COMMON))
from figure_utils import cli, draw_paired_half_violin_panel, panel_row, save_figure
from loaders import normalize_scalar, prepare_paired_plot_df, read_table
from paths import OUTPUT_ROOT, ensure_output_dirs
from stats_annotations import paired_significance_text
from style import HEMISPHERE_ORDER, STUDY_ORDER, apply_style
from layout_config import HALF_PANEL_WIDTH_IN, VIOLIN_HEIGHT_IN

SCRIPT = Path(__file__).name

STANDARD_FIGURE_WIDTH = 2.50
COMPACT_FIGURE_WIDTH = 1.35
FIGURE_HEIGHT = 1.50
MAIN_FIGURE_VIOLIN_WIDTH = HALF_PANEL_WIDTH_IN
MAIN_FIGURE_VIOLIN_HEIGHT = VIOLIN_HEIGHT_IN
AXES_LEFT_IN = 0.58
AXES_RIGHT_IN = 0.06
AXES_BOTTOM_IN = 0.28
AXES_TOP_IN = 0.35

MAIN_FIGURE_VIOLIN_STEMS = {
    "pattern_count_per_frame",
    "mean_size",
    "gipr",
    "occupancy_p95_p5_diff",
    "phase_ngsc",
    "complex_svd_top_mode",
    "global_phase_wfc",
    "global_phase_bfc",
    "gcor",
    "dfr_mean_region_count",
    "weighted_mean_cos2_alignment",
    "path_entropy",
    "mean_duration",
}

TARGET_STEMS = MAIN_FIGURE_VIOLIN_STEMS


def metric_specs():
    simple = [
        ("pattern_count_per_frame", "Spiral count / frame", "EXP001", "02_exports/all_metrics_runner/subject_level/exp001_pattern_stats_subject.csv", "source_dataset", "group", "pattern_count_per_frame", "Spirals per frame"),
        ("mean_size", "Spiral size", "EXP001", "02_exports/all_metrics_runner/subject_level/exp001_pattern_stats_subject.csv", "source_dataset", "group", "mean_size", "Mean size"),
        ("gipr", "GIPR", "EXP006", "02_exports/gipr/subject_level/exp006_gipr_subject.csv", "drug_set", "group", "gipr", "GIPR"),
        ("occupancy_p95_p5_diff", "PTDR", "EXP003", "02_exports/all_metrics_runner/ptdr_vortex/exp003_ptdr_subject_values.csv", "dataset", "condition", "occupancy_p95_p5_diff", "Occupancy P95-P5"),
        ("path_entropy", "Path entropy", "EXP010", "02_exports/path_entropy/subject_level/exp010_path_entropy_subject.csv", "source_dataset", "group", "entropy", "Entropy"),
        ("mean_duration", "Pattern duration", "EXP001", "02_exports/all_metrics_runner/subject_level/exp001_pattern_stats_subject.csv", "source_dataset", "group", "mean_duration", "Duration"),
        ("msd_beta", "MSD beta", "EXP012", "02_exports/pattern_msd_beta/subject_level/exp012_per_subject_msd_beta.csv", "source_dataset", "group", "beta", "MSD beta"),
        ("phase_ngsc", "Phase NGSC", "EXP022", "02_exports/all_metrics_runner/ngsc/exp022_ngsc_subject_values.csv", "dataset", "condition", "phase_ngsc", "Phase NGSC"),
        ("raw_ngsc", "Raw NGSC", "EXP022", "02_exports/all_metrics_runner/ngsc/exp022_ngsc_subject_values.csv", "dataset", "condition", "ngsc", "Raw NGSC"),
        ("gcor", "GCOR", "EXP020", "02_exports/gcor/subject_level/exp020_gcor_subject.csv", "source_dataset", "group", "gcor", "GCOR"),
    ]
    out = []
    for stem, title, export_id, path, dataset, condition, value, ylabel in simple:
        frame = read_table(path)
        data = prepare_paired_plot_df(normalize_scalar(frame, dataset, condition, value, stem, export_id))
        out.append((stem, title, data, export_id, path, ylabel, "native"))

    cai_path = "02_exports/cai_model_delta/model_correlation/exp016_subject_condition_values_long.csv"
    cai = read_table(cai_path, ["drug", "source", "condition_role", "metric", "value", "subid", "hemisphere"])
    for metric, stem, title, ylabel in [
        ("csvd_complex_svd_top1_energy", "complex_svd_top_mode", "Complex SVD top mode", "Top-1 energy"),
        ("cai_weighted_mean_cos2_alignment", "weighted_mean_cos2_alignment", "CAI", "CAI"),
    ]:
        frame = cai[(cai.source == "empirical") & (cai.metric == metric)].copy()
        data = prepare_paired_plot_df(normalize_scalar(frame, "drug", "condition_role", "value", metric, "EXP016"))
        out.append((stem, title, data, "EXP016", cai_path, ylabel, "native"))

    phase_path = "02_exports/phase_recon_wbfc/model_correlation/exp014_subject_condition_values_z.csv"
    phase = read_table(phase_path, ["drug", "source", "condition_role", "scope", "fc_type", "subid", "hemisphere", "value_z"])
    phase = phase[(phase.source == "original") & (phase.scope == "global")].copy()
    for fc_type, stem, title in [
        ("between", "global_phase_bfc", "Global phase bFC"),
        ("within", "global_phase_wfc", "Global phase wFC"),
    ]:
        frame = phase[phase.fc_type == fc_type].copy()
        data = prepare_paired_plot_df(normalize_scalar(frame, "drug", "condition_role", "value_z", stem, "EXP014"))
        out.append((stem, title, data, "EXP014", phase_path, "Fisher z", "Fisher z"))

    dfr_path = "02_exports/dfr/model_correlation/exp018_per_condition_subject_dfr.csv"
    dfr = read_table(dfr_path, ["source", "drug", "role", "subid", "hemisphere", "mean_region_count"])
    dfr = dfr[dfr.source == "empirical"].copy()
    data = prepare_paired_plot_df(normalize_scalar(dfr, "drug", "role", "mean_region_count", "mean_region_count", "EXP018"))
    out.append(("dfr_mean_region_count", "CSD", data, "EXP018", dfr_path, "Mean region count", "native"))
    return out


def write_coverage(items):
    ensure_output_dirs()
    path = OUTPUT_ROOT / "publication_paired_metric_coverage.csv"
    available = {item[0] for item in items}
    rows = [{"metric": stem, "status": "ready", "notes": ""} for stem in sorted(available)]
    rows.extend([
        {"metric": "global_raw_bfc", "status": "not_available", "notes": "EXP024 has no subject-level global between-network FC field."},
        {"metric": "global_raw_wfc", "status": "not_available", "notes": "EXP024 has no subject-level global within-network FC field."},
    ])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = cli("Plot one-row publication-style paired half violins")
    items = [item for item in metric_specs() if item[0] in TARGET_STEMS]
    if args.list_panels:
        print("\n".join(stem for stem, *_ in items))
        return
    apply_style()
    plt.rcParams.update({
        "axes.grid": False,
        "axes.titleweight": "normal",
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    for stem, title, data, export_id, source, ylabel, scale in items:
        compact = False
        if stem in MAIN_FIGURE_VIOLIN_STEMS:
            width = MAIN_FIGURE_VIOLIN_WIDTH
            height = MAIN_FIGURE_VIOLIN_HEIGHT
            axes_bottom_in = 0.25
            axes_top_in = 0.36
        else:
            width = COMPACT_FIGURE_WIDTH if compact else STANDARD_FIGURE_WIDTH
            height = FIGURE_HEIGHT
            axes_bottom_in = AXES_BOTTOM_IN
            axes_top_in = AXES_TOP_IN
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes([
            AXES_LEFT_IN / width,
            axes_bottom_in / height,
            (width - AXES_LEFT_IN - AXES_RIGHT_IN) / width,
            (height - axes_bottom_in - axes_top_in) / height,
        ])
        rows = []
        facets = [(s, h) for s in STUDY_ORDER for h in HEMISPHERE_ORDER]
        group_spacing = 0.16
        centers = [facet_index * group_spacing for facet_index in range(len(facets))]
        for facet_index, ((study, hemisphere), center) in enumerate(zip(facets, centers)):
            n_pairs = draw_paired_half_violin_panel(
                ax, data, study, hemisphere, x_center=center, configure_axis=False,
            )
            panel = data[(data.study == study) & (data.hemisphere == hemisphere)]
            facet_label = f"{study} {hemisphere[0].upper()}" if compact else f"{study}\n{hemisphere.capitalize()}"
            ax.text(center, 1.03, facet_label,
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6, linespacing=0.9)
            ax.text(center, 0.99, paired_significance_text(panel),
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, fontweight="bold")
            rows.append(panel_row(
                f"{stem}_{study}_{hemisphere}", export_id, source, n_pairs * 2, stem,
                dataset=study, hemisphere=hemisphere, value_scale=scale,
                notes="paired-only; PCB left half; Drug right half; translucent density",
            ))
        ax.set_xlim(centers[0] - 0.09, centers[-1] + 0.09)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.suptitle(title, fontsize=7, y=0.98)
        save_figure(
            fig,
            f"publication_paired_{stem}",
            f"PUB_PAIRED_{stem.upper()}",
            rows,
            SCRIPT,
            args.dry_run,
            tight=False,
        )
    if not args.dry_run:
        write_coverage(items)


if __name__ == "__main__":
    main()
