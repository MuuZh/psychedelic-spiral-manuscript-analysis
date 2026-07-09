import argparse
import csv
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

from paths import LOG_ROOT, OUTPUT_ROOT, PDF_ROOT, PNG_ROOT, SOURCE_MANIFEST, ensure_output_dirs
from layout_config import output_dir_for_stem
from stats_annotations import paired_test_text
from style import CONDITION_ORDER, paired_palette

MANIFEST_COLUMNS = [
    "figure_file", "figure_id", "panel_id", "source_export_id", "source_file",
    "rows_used_or_array_keys", "metric", "dataset", "hemisphere", "value_scale",
    "generated_by_script", "generated_at", "notes",
]


def cli(description, configure_parser=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-panels", action="store_true")
    parser.add_argument("--include-unaffected-previews", action="store_true")
    if configure_parser is not None:
        configure_parser(parser)
    return parser.parse_args()


def log(script, message):
    ensure_output_dirs()
    with (LOG_ROOT / f"{script}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def draw_paired_violin_panel(ax, data, study, hemisphere, title="", ylabel=None):
    panel = data[(data.study == study) & (data.hemisphere == hemisphere)].copy()
    palette = paired_palette(study)
    sns.violinplot(data=panel, x="condition", y="value", hue="condition", order=CONDITION_ORDER,
                   hue_order=CONDITION_ORDER, palette=palette, legend=False,
                   inner=None, cut=0, linewidth=0.8, ax=ax)
    wide = panel.pivot(index="subject", columns="condition", values="value").dropna()
    for _, pair in wide.iterrows():
        ax.plot([0, 1], [pair["PCB"], pair["Drug"]], color="#4A4A4A", alpha=0.35, lw=0.7, zorder=2)
    for x, condition in enumerate(CONDITION_ORDER):
        values = wide[condition]
        ax.scatter([x] * len(values), values, s=7, color=palette[condition], edgecolor="black",
                   linewidth=0.35, alpha=0.8, zorder=3)
    ax.set_title(f"{title}\n{study} study | {hemisphere}")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or "")
    ax.text(.03, .97, paired_test_text(panel), transform=ax.transAxes, va="top", fontsize=7)
    return len(wide)


def draw_paired_half_violin_panel(
        ax, data, study, hemisphere, title="", ylabel=None,
        violin_width=0.055, point_offset=0.025, point_jitter=0.006,
        x_center=0, configure_axis=True):
    panel = data[(data.study == study) & (data.hemisphere == hemisphere)].copy()
    wide = panel.pivot(index="subject", columns="condition", values="value").dropna()
    palette = paired_palette(study)
    point_positions = {"PCB": x_center - point_offset, "Drug": x_center + point_offset}
    rng = np.random.default_rng(20260611)

    for condition in CONDITION_ORDER:
        values = wide[condition].to_numpy(dtype=float)
        if len(values) >= 2 and np.ptp(values) > 0:
            padding = max(np.ptp(values) * 0.08, abs(np.mean(values)) * 0.005, 1e-9)
            y = np.linspace(values.min() - padding, values.max() + padding, 256)
            density = gaussian_kde(values)(y)
            density = density / density.max() * violin_width
            edge = x_center - density if condition == "PCB" else x_center + density
            ax.fill_betweenx(
                y, x_center, edge, facecolor=palette[condition], edgecolor=palette[condition],
                linewidth=0.9, alpha=0.32, zorder=1,
            )

    point_x = {
        condition: np.full(len(wide), point_positions[condition])
        + rng.normal(0, point_jitter, len(wide))
        for condition in CONDITION_ORDER
    }
    for pair_index, (_, pair) in enumerate(wide.iterrows()):
        ax.plot(
            [point_x["PCB"][pair_index], point_x["Drug"][pair_index]],
            [pair["PCB"], pair["Drug"]],
            color="#6B6B6B", alpha=0.28, lw=0.65, zorder=2,
        )
    for condition in CONDITION_ORDER:
        values = wide[condition].to_numpy(dtype=float)
        ax.scatter(
            point_x[condition], values, s=7.5,
            facecolor=palette[condition], edgecolor="white", linewidth=0.35,
            alpha=0.62, zorder=3,
        )

    if configure_axis:
        ax.set_xlim(x_center - 0.13, x_center + 0.13)
        ax.set_xticks([point_positions["PCB"], point_positions["Drug"]], CONDITION_ORDER)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{study} | {hemisphere.capitalize()}")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel or "")
        ax.text(.03, .97, paired_test_text(panel), transform=ax.transAxes, va="top", fontsize=7)
        sns.despine(ax=ax)
    return len(wide)


def save_figure(fig, stem, figure_id, panel_rows, script, dry_run=False, tight=True):
    ensure_output_dirs()
    if dry_run:
        log(script, f"DRY RUN {stem}: {len(panel_rows)} panels")
        plt.close(fig)
        return
    pdf_dir = output_dir_for_stem(stem)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / f"{stem}.pdf"
    written_targets = []
    for target in (pdf,):
        try:
            fig.savefig(target, bbox_inches="tight", pad_inches=0.02)
            written_targets.append(target)
        except PermissionError:
            log(script, f"skipped locked output {target.name}")
    plt.close(fig)
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for target in written_targets:
        for row in panel_rows:
            rows.append({
                "figure_file": str(target.relative_to(OUTPUT_ROOT)),
                "figure_id": figure_id,
                "generated_by_script": script,
                "generated_at": generated_at,
                **row,
            })
    if rows:
        append_manifest(rows)
    log(script, f"wrote {', '.join(target.name for target in written_targets)}; panels={len(panel_rows)}")


def append_manifest(rows):
    ensure_output_dirs()
    existing = []
    if SOURCE_MANIFEST.exists():
        with SOURCE_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
    prefix = f"{OUTPUT_ROOT.name}\\"
    for row in existing:
        if row["figure_file"].startswith(prefix):
            row["figure_file"] = row["figure_file"][len(prefix):]
    replaced_files = {r["figure_file"] for r in rows}
    existing = [r for r in existing if r["figure_file"] not in replaced_files]
    with SOURCE_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(existing + rows)


def panel_row(panel_id, export_id, source_file, used, metric, dataset="all",
              hemisphere="all", value_scale="native", notes=""):
    return {
        "panel_id": panel_id, "source_export_id": export_id, "source_file": source_file,
        "rows_used_or_array_keys": str(used), "metric": metric, "dataset": dataset,
        "hemisphere": hemisphere, "value_scale": value_scale, "notes": notes,
    }
