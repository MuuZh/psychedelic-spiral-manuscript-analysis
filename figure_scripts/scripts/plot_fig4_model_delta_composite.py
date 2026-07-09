import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

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
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.titlesize": 7,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pgf.rcfonts": False,
})

COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(COMMON))
from figure_utils import append_manifest, cli, log, panel_row
from loaders import read_table
from paths import OUTPUT_ROOT, ensure_output_dirs
from style import HEMISPHERE_ORDER, STUDY_DRUG_COLORS, STUDY_ORDER, apply_style

SCRIPT = Path(__file__).name
FIGURE_ID = "FIG4_MODEL_DELTA_COMPOSITE"
OUT_ROOT = OUTPUT_ROOT / "fig4"
PNG_PATH = OUT_ROOT / "fig4_model_delta_composite.png"
PDF_PATH = OUT_ROOT / "fig4_model_delta_composite.pdf"

BOX_METRICS = [
    ("ngsc", "NGSC", "Delta", "EXP023", "02_exports/phase_ngsc_model/exp023_phase_ngsc_paired_deltas.csv", None, "native"),
    ("csvd", "cSVD", "Delta", "EXP016", "02_exports/cai_model_delta/model_correlation/exp016_empirical_model_joined_values.csv", "csvd_complex_svd_top1_energy", "native"),
    ("wfc", "wFC", "Delta (z)", "EXP014", "02_exports/phase_recon_wbfc/model_correlation/exp014_original_recon_joined_values_z.csv", "global_within_edge_weighted", "Fisher z"),
    ("bfc", "bFC", "Delta (z)", "EXP014", "02_exports/phase_recon_wbfc/model_correlation/exp014_original_recon_joined_values_z.csv", "global_between_edge_weighted", "Fisher z"),
    ("csd", "CSD count", "Delta", "EXP018", "02_exports/dfr/model_correlation/exp018_per_subject_dfr_drug_minus_pcb_delta.csv", "delta_region_count", "native"),
    ("cai", "CAI", "Delta", "EXP016", "02_exports/cai_model_delta/model_correlation/exp016_empirical_model_joined_values.csv", "cai_weighted_mean_cos2_alignment", "native"),
]

CORR_METRIC_STEMS = ["bfc", "wfc", "ngsc", "cai"]
METRIC_BY_STEM = {item[0]: item for item in BOX_METRICS}


def ensure_custom_output_dirs():
    ensure_output_dirs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

class CleanScalarFormatter(mticker.ScalarFormatter):
    def __call__(self, x, pos=None):
        if abs(x) < 1e-12:
            return "0"
        return super().__call__(x, pos)
    
def scientific_formatter():
    formatter = CleanScalarFormatter(useMathText=True, useOffset=True)
    formatter.set_powerlimits((0, 0))
    formatter.set_scientific(True)
    return formatter


def apply_scientific_ticks(ax, x=False, y=False):
    if x:
        ax.xaxis.set_major_formatter(scientific_formatter())
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=True, useMathText=True)
    if y:
        ax.yaxis.set_major_formatter(scientific_formatter())
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=True, useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(6)
    ax.yaxis.get_offset_text().set_fontsize(6)
    
    ax.xaxis.get_offset_text().set_va('bottom')
    ax.xaxis.get_offset_text().set_ha('right')


def load_boxplot_data(stem, metric):
    if stem == "ngsc":
        frame = read_table(metric[4], ["dataset", "source", "hemisphere", "subid", "phase_ngsc_delta"])
        out = frame[frame["source"].eq("orig")].copy()
        out = out.rename(columns={"dataset": "drug", "subid": "subject", "phase_ngsc_delta": "delta"})
    elif stem in {"csvd", "cai"}:
        frame = read_table(metric[4], ["drug", "subid", "hemisphere", "metric", "value_kind", "empirical_value"])
        out = frame[(frame["metric"].eq(metric[5])) & (frame["value_kind"].eq("delta"))].copy()
        out = out.rename(columns={"subid": "subject", "empirical_value": "delta"})
    elif stem in {"wfc", "bfc"}:
        frame = read_table(metric[4], ["drug", "subid", "hemisphere", "metric", "value_kind", "original_value_z"])
        out = frame[(frame["metric"].eq(metric[5])) & (frame["value_kind"].eq("delta"))].copy()
        out = out.rename(columns={"subid": "subject", "original_value_z": "delta"})
    elif stem == "csd":
        frame = read_table(metric[4], ["source", "drug", "subid", "hemisphere", "delta_region_count"])
        out = frame[frame["source"].eq("empirical")].copy()
        out = out.rename(columns={"subid": "subject", "delta_region_count": "delta"})
    else:
        raise ValueError(f"Unsupported metric stem: {stem}")

    out = out[["drug", "hemisphere", "subject", "delta"]].copy()
    out["drug"] = out["drug"].astype(str).str.upper()
    out["hemisphere"] = out["hemisphere"].astype(str).str.lower()
    out["subject"] = out["subject"].astype(str)
    out["delta"] = pd.to_numeric(out["delta"], errors="coerce")
    out = out[out["drug"].isin(STUDY_ORDER) & out["hemisphere"].isin(HEMISPHERE_ORDER)]
    return out.dropna(subset=["delta"]).sort_values(["drug", "hemisphere", "subject"]).reset_index(drop=True)


def load_correlation_data(stem, metric):
    if stem == "ngsc":
        frame = read_table(metric[4], ["dataset", "source", "hemisphere", "subid", "phase_ngsc_delta"])
        wide = frame.pivot_table(index=["dataset", "subid", "hemisphere"], columns="source", values="phase_ngsc_delta")
        out = wide.dropna(subset=["orig", "recon"]).reset_index()
        out = out.rename(columns={"dataset": "drug", "subid": "subject", "orig": "empirical", "recon": "model"})
    elif stem in {"csvd", "cai"}:
        frame = read_table(metric[4], ["drug", "subid", "hemisphere", "metric", "value_kind", "empirical_value", "model_value"])
        out = frame[(frame["metric"].eq(metric[5])) & (frame["value_kind"].eq("delta"))].copy()
        out = out.rename(columns={"subid": "subject", "empirical_value": "empirical", "model_value": "model"})
    elif stem in {"wfc", "bfc"}:
        frame = read_table(metric[4], ["drug", "subid", "hemisphere", "metric", "value_kind", "original_value_z", "recon_value_z"])
        out = frame[(frame["metric"].eq(metric[5])) & (frame["value_kind"].eq("delta"))].copy()
        out = out.rename(columns={"subid": "subject", "original_value_z": "empirical", "recon_value_z": "model"})
    elif stem == "csd":
        frame = read_table(metric[4], ["source", "drug", "subid", "hemisphere", "delta_region_count"])
        wide = frame.pivot_table(index=["drug", "subid", "hemisphere"], columns="source", values="delta_region_count")
        out = wide.dropna(subset=["empirical", "recon"]).reset_index()
        out = out.rename(columns={"subid": "subject", "recon": "model"})
    else:
        raise ValueError(f"Unsupported metric stem: {stem}")

    out = out[["drug", "hemisphere", "subject", "empirical", "model"]].copy()
    out["drug"] = out["drug"].astype(str).str.upper()
    out["hemisphere"] = out["hemisphere"].astype(str).str.lower()
    out["subject"] = out["subject"].astype(str)
    out["empirical"] = pd.to_numeric(out["empirical"], errors="coerce")
    out["model"] = pd.to_numeric(out["model"], errors="coerce")
    out = out[out["drug"].isin(STUDY_ORDER) & out["hemisphere"].isin(HEMISPHERE_ORDER)]
    return out.dropna(subset=["empirical", "model"]).sort_values(["drug", "hemisphere", "subject"]).reset_index(drop=True)


def draw_boxplot(ax, stem, metric, data):
    palette = {drug: STUDY_DRUG_COLORS[drug] for drug in STUDY_ORDER}
    group_centers = {"left": 0.0, "right": 0.105}
    offset = 0.012
    box_width = 0.0125
    positions = {
        ("left", "DMT"): group_centers["left"] - offset,
        ("left", "LSD"): group_centers["left"] + offset,
        ("right", "DMT"): group_centers["right"] - offset,
        ("right", "LSD"): group_centers["right"] + offset,
    }
    rng = np.random.default_rng(20260627)
    box_data, box_positions, box_colors = [], [], []
    for hemisphere in HEMISPHERE_ORDER:
        for drug in STUDY_ORDER:
            panel = data[(data["hemisphere"].eq(hemisphere)) & (data["drug"].eq(drug))]
            box_data.append(panel["delta"].to_numpy(dtype=float))
            box_positions.append(positions[(hemisphere, drug)])
            box_colors.append(palette[drug])

    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#303030", "linewidth": 0.65},
        whiskerprops={"color": "#303030", "linewidth": 0.55},
        capprops={"color": "#303030", "linewidth": 0.55},
        boxprops={"linewidth": 0.55},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#303030")
        patch.set_alpha(0.78)

    for hemisphere in HEMISPHERE_ORDER:
        for drug in STUDY_ORDER:
            panel = data[(data["hemisphere"].eq(hemisphere)) & (data["drug"].eq(drug))]
            side_shift = -box_width * 0.95 if drug == "DMT" else box_width * 0.95
            x = np.full(len(panel), positions[(hemisphere, drug)] + side_shift) + rng.uniform(-0.0025, 0.0025, len(panel))
            ax.scatter(x, panel["delta"], s=6.2, facecolor=palette[drug], edgecolor="black", linewidth=0.15, alpha=0.52, zorder=3)

    ax.axhline(0, color="#BDBDBD", linewidth=0.45, zorder=1)
    ax.set_title(metric[1], pad=1.5)
    ax.set_ylabel("")
    ax.set_xticks([group_centers["left"], group_centers["right"]])
    ax.set_xticklabels(["L", "R"])
    ax.set_xlim(group_centers["left"] - 0.045, group_centers["right"] + 0.045)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    apply_scientific_ticks(ax, y=True)
    ax.tick_params(axis="both", length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)


def apply_axis_limits(ax, data):
    values = pd.concat([data["empirical"], data["model"]]).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return
    lo = finite.min()
    hi = finite.max()
    pad = max((hi - lo) * 0.10, abs(hi) * 0.02, 1e-3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def significance_stars(p_value):
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "NS"


def draw_correlation(ax, stem, metric, data):
    panel_data = data[data["hemisphere"].eq("left")].copy()
    for drug in STUDY_ORDER:
        panel = panel_data[panel_data["drug"].eq(drug)]
        if panel.empty:
            continue
        ax.scatter(
            panel["empirical"],
            panel["model"],
            s=11,
            marker="o",
            facecolor=STUDY_DRUG_COLORS[drug],
            edgecolor="black",
            linewidth=0.25,
            alpha=0.78,
            zorder=3,
        )
        sns.regplot(
            data=panel,
            x="empirical",
            y="model",
            scatter=False,
            ci=95,
            n_boot=2000,
            seed=20260630,
            truncate=False,
            color=STUDY_DRUG_COLORS[drug],
            line_kws={"linewidth": 0.85, "alpha": 0.95},
            ax=ax,
        )
    apply_axis_limits(ax, panel_data)
    # ax.axline((0, 0), slope=1, color="#B5B5B5", linewidth=0.55, linestyle=":", zorder=1)
    ax.set_title(f"{metric[1]} left", pad=1.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    apply_scientific_ticks(ax, x=True, y=True)
    ax.tick_params(axis="both", length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)

    y = 0.97
    for drug in STUDY_ORDER:
        panel = panel_data[panel_data["drug"].eq(drug)]
        if len(panel) < 3:
            label = f"{drug}: n={len(panel)}"
        else:
            r, p = pearsonr(panel["empirical"].to_numpy(), panel["model"].to_numpy())
            label = f"{drug}: R$^2$={r * r:.2f} {significance_stars(p)}"
        ax.text(0.03, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=6, color=STUDY_DRUG_COLORS[drug])
        y -= 0.09


def collect_rows():
    rows = []
    for stem, label, _, export_id, source_file, metric_name, value_scale in BOX_METRICS:
        data = load_boxplot_data(stem, METRIC_BY_STEM[stem])
        for drug in STUDY_ORDER:
            for hemisphere in HEMISPHERE_ORDER:
                panel = data[(data["drug"].eq(drug)) & (data["hemisphere"].eq(hemisphere))]
                rows.append(panel_row(
                    f"box_{stem}_{drug}_{hemisphere}",
                    export_id,
                    source_file,
                    len(panel),
                    metric_name or stem,
                    dataset=drug,
                    hemisphere=hemisphere,
                    value_scale=value_scale,
                    notes="Compact Fig4 Drug-PCB empirical delta boxplot.",
                ))
    for stem in CORR_METRIC_STEMS:
        metric = METRIC_BY_STEM[stem]
        data = load_correlation_data(stem, metric)
        for drug in STUDY_ORDER:
            panel = data[(data["drug"].eq(drug)) & (data["hemisphere"].eq("left"))]
            rows.append(panel_row(
                f"corr_{stem}_{drug}_left",
                metric[3],
                metric[4],
                len(panel),
                metric[5] or stem,
                dataset=drug,
                hemisphere="left",
                value_scale=metric[6],
                notes="Compact Fig4 left-hemisphere empirical versus model Drug-PCB delta correlation.",
            ))
    return rows


def save_composite(fig, rows, dry_run):
    ensure_custom_output_dirs()
    if dry_run:
        log(SCRIPT, f"DRY RUN fig4_model_delta_composite: {len(rows)} panels")
        plt.close(fig)
        return

    targets = [PNG_PATH, PDF_PATH]
    written = []
    for target in targets:
        try:
            fig.savefig(target, bbox_inches="tight", pad_inches=0.02)
            written.append(target)
        except PermissionError:
            log(SCRIPT, f"skipped locked output {target}")
    plt.close(fig)

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
        last_error = None
        for _ in range(3):
            try:
                append_manifest(manifest_rows)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.25)
        if last_error is not None:
            log(SCRIPT, f"skipped manifest update: {last_error}")
    log(SCRIPT, f"wrote {', '.join(t.name for t in written)}; panels={len(rows)}")


def save_panel(fig, stem, dry_run):
    ensure_custom_output_dirs()
    if dry_run:
        plt.close(fig)
        return
    targets = [OUT_ROOT / f"{stem}.png", OUT_ROOT / f"{stem}.pdf"]
    for target in targets:
        try:
            fig.savefig(target, bbox_inches="tight", pad_inches=0.02)
        except PermissionError:
            log(SCRIPT, f"skipped locked output {target}")
    plt.close(fig)


def save_individual_panels(dry_run=False):
    for metric in BOX_METRICS:
        stem = metric[0]
        fig, ax = plt.subplots(figsize=(1.15, 1.05))
        draw_boxplot(ax, stem, metric, load_boxplot_data(stem, metric))
        fig.subplots_adjust(left=0.24, right=0.98, bottom=0.20, top=0.84)
        save_panel(fig, f"fig4_box_{stem}", dry_run)

    for stem in CORR_METRIC_STEMS:
        metric = METRIC_BY_STEM[stem]
        fig, ax = plt.subplots(figsize=(1.65, 1.45))
        draw_correlation(ax, stem, metric, load_correlation_data(stem, metric))
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.86)
        save_panel(fig, f"fig4_corr_{stem}_left", dry_run)


def main():
    args = cli("Plot compact Fig4 model-delta composite panels")
    if args.list_panels:
        print("\n".join([f"box_{stem}" for stem, *_ in BOX_METRICS] + [f"corr_{stem}_left" for stem in CORR_METRIC_STEMS]))
        return

    apply_style()
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "axes.grid": False,
        "axes.titleweight": "normal",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    
    plt.rcParams.update({
    "axes.grid": False,
    "axes.titleweight": "normal",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "figure.titlesize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(6.95, 3.75))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.38, 0.62], wspace=0.16)
    box_grid = outer[0].subgridspec(3, 2, hspace=0.52, wspace=0.25)
    corr_grid = outer[1].subgridspec(2, 2, hspace=0.42, wspace=0.28)

    for index, metric in enumerate(BOX_METRICS):
        stem = metric[0]
        ax = fig.add_subplot(box_grid[index // 2, index % 2])
        draw_boxplot(ax, stem, metric, load_boxplot_data(stem, metric))
        # if index % 2 == 1:
        #     ax.tick_params(axis="y", labelleft=False)

    for index, stem in enumerate(CORR_METRIC_STEMS):
        metric = METRIC_BY_STEM[stem]
        ax = fig.add_subplot(corr_grid[index // 2, index % 2])
        draw_correlation(ax, stem, metric, load_correlation_data(stem, metric))

    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.075, top=0.955)
    save_composite(fig, collect_rows(), args.dry_run)
    save_individual_panels(args.dry_run)


if __name__ == "__main__":
    main()
