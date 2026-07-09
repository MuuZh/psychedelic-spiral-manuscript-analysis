from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .config import Config
    from .utils import (
        build_group_summary_df,
        get_palette,
        paired_t,
        plot_abs_effect_size_bars,
        plot_paired_violin,
        save_fig,
        set_output_naming,
        setup_logging,
        unpaired_t,
        write_table,
    )
except ImportError:  # Allow direct execution: python analysis/all_metrics/run_gcor_all_metrics.py
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from analysis.all_metrics.config import Config
    from analysis.all_metrics.utils import (
        build_group_summary_df,
        get_palette,
        paired_t,
        plot_abs_effect_size_bars,
        plot_paired_violin,
        save_fig,
        set_output_naming,
        setup_logging,
        unpaired_t,
        write_table,
    )

matplotlib.use("Agg")


@dataclass(frozen=True)
class BatchSpec:
    label: str
    gcor_dir: Path
    drug_label: str
    pcb_label: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all_metrics-style group statistics for GCOR batch outputs.",
    )
    parser.add_argument(
        "--batch",
        action="append",
        nargs=3,
        metavar=("LABEL", "GCOR_DIR", "DRUG_LABEL"),
        help=(
            "Batch to analyze. Can be repeated. Example: "
            "--batch DMT analysis_outputs/gcor_batch_DMT DMT"
        ),
    )
    parser.add_argument("--pcb-label", default="PCB")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs") / "all_metrics_gcor",
    )
    parser.add_argument("--results-prefix", default="gcor_group_stats")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _default_batches(pcb_label: str) -> list[BatchSpec]:
    return [
        BatchSpec(
            label="DMT",
            gcor_dir=Path("analysis_outputs") / "gcor_batch_DMT",
            drug_label="DMT",
            pcb_label=pcb_label,
        ),
        BatchSpec(
            label="LSD",
            gcor_dir=Path("analysis_outputs") / "gcor_batch_LSD",
            drug_label="LSD",
            pcb_label=pcb_label,
        ),
    ]


def _batches_from_args(args: argparse.Namespace) -> list[BatchSpec]:
    if not args.batch:
        return _default_batches(args.pcb_label)
    return [
        BatchSpec(
            label=str(label),
            gcor_dir=Path(gcor_dir),
            drug_label=str(drug_label),
            pcb_label=args.pcb_label,
        )
        for label, gcor_dir, drug_label in args.batch
    ]


def _make_cfg(args: argparse.Namespace, spec: BatchSpec) -> Config:
    cfg = Config()
    cfg.results_prefix = f"{args.results_prefix}_{spec.label.lower()}"
    cfg.group_drug = spec.drug_label
    cfg.group_pcb = spec.pcb_label
    cfg.output_root = args.output_root
    cfg.save_plots = not args.no_plots
    return cfg


def _load_subject_metrics(spec: BatchSpec) -> pd.DataFrame:
    csv_path = spec.gcor_dir / "all_gcor_by_subject.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"GCOR combined table not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"condition", "id", "hemisphere", "gcor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["group"] = out["condition"].astype(str).str.upper()
    out["subid"] = out["id"].astype(str)
    out["hemisphere"] = out["hemisphere"].astype(str).str.lower()
    out["gcor"] = pd.to_numeric(out["gcor"], errors="coerce")
    out = out[out["group"].isin([spec.drug_label, spec.pcb_label])].copy()
    out = out[out["hemisphere"].isin(["left", "right"])].copy()
    out = out.dropna(subset=["gcor"])
    return out.loc[:, ["group", "subid", "hemisphere", "gcor"]].sort_values(
        ["group", "subid", "hemisphere"]
    )


def _summary_rows(subject_df: pd.DataFrame, cfg: Config) -> list[dict]:
    rows: list[dict] = []
    for hemi in sorted(subject_df["hemisphere"].dropna().unique()):
        hemi_df = subject_df[subject_df["hemisphere"] == hemi]
        pivot = hemi_df.pivot_table(
            index="subid", columns="group", values="gcor", aggfunc="mean"
        )
        if cfg.group_drug in pivot.columns and cfg.group_pcb in pivot.columns:
            rows.append(
                {
                    "section": "gcor",
                    "metric": "gcor",
                    "hemisphere": hemi,
                    "comparison": "paired_drug_vs_pcb",
                    **paired_t(pivot[cfg.group_drug], pivot[cfg.group_pcb]),
                }
            )

        drug = hemi_df[hemi_df["group"] == cfg.group_drug]["gcor"]
        pcb = hemi_df[hemi_df["group"] == cfg.group_pcb]["gcor"]
        rows.append(
            {
                "section": "gcor",
                "metric": "gcor",
                "hemisphere": hemi,
                "comparison": "unpaired_drug_vs_pcb",
                **unpaired_t(drug, pcb),
            }
        )
    return rows


def _save_distribution_plots(subject_df: pd.DataFrame, out_dir: Path, cfg: Config) -> None:
    if subject_df.empty:
        return
    pal = get_palette(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, hemi in zip(axes, ["left", "right"]):
        tidy = subject_df[subject_df["hemisphere"] == hemi]
        if tidy.empty:
            ax.axis("off")
            continue
        sns.violinplot(
            data=tidy,
            x="group",
            y="gcor",
            order=[cfg.group_pcb, cfg.group_drug],
            palette=pal,
            cut=0,
            ax=ax,
        )
        sns.stripplot(
            data=tidy,
            x="group",
            y="gcor",
            order=[cfg.group_pcb, cfg.group_drug],
            color="black",
            alpha=0.55,
            size=3,
            ax=ax,
        )
        ax.set_title(f"gcor ({hemi})")
        ax.set_xlabel("")
    fig.tight_layout()
    save_fig(fig, out_dir / "violin_gcor.png", cfg.save_plots)

    for hemi in ["left", "right"]:
        tidy = subject_df[subject_df["hemisphere"] == hemi][
            ["subid", "group", "gcor"]
        ]
        plot_paired_violin(
            tidy,
            "gcor",
            hemi,
            "GCOR paired",
            out_dir / f"paired_gcor_{hemi}.png",
            cfg,
        )


def _run_batch(args: argparse.Namespace, spec: BatchSpec) -> pd.DataFrame:
    cfg = _make_cfg(args, spec)
    out_root = cfg.output_root / cfg.results_prefix
    set_output_naming(cfg.results_prefix, datetime.now().strftime("%Y%m%d"))
    setup_logging(out_root)
    logging.info("Loading GCOR batch: %s", spec.gcor_dir)

    subject_df = _load_subject_metrics(spec)
    if subject_df.empty:
        raise RuntimeError(f"No usable GCOR rows found in {spec.gcor_dir}")

    out_dir = out_root / "gcor"
    write_table(subject_df, out_dir / "subject_metrics.csv")
    write_table(build_group_summary_df(subject_df, ["gcor"], cfg), out_dir / "group_summary.csv")
    _save_distribution_plots(subject_df, out_dir, cfg)

    summary_df = pd.DataFrame(_summary_rows(subject_df, cfg))
    summary_dir = out_root / "summary"
    write_table(summary_df, summary_dir / "all_metrics_summary.csv")
    plot_abs_effect_size_bars(
        summary_df,
        summary_dir / "abs_cohens_dz_by_metric.png",
        title=f"Absolute Cohen's dz by metric ({spec.label} GCOR)",
        save=cfg.save_plots,
    )
    logging.info("Done. Summary rows: %d", len(summary_df))
    summary_df.insert(0, "batch", spec.label)
    return summary_df


def _write_combined_summary(summary_frames: Iterable[pd.DataFrame], output_root: Path) -> None:
    frames = [df for df in summary_frames if not df.empty]
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_root / "combined_gcor_summary.csv", index=False)


def main() -> None:
    args = _parse_args()
    summaries = []
    for spec in _batches_from_args(args):
        summaries.append(_run_batch(args, spec))
    _write_combined_summary(summaries, args.output_root)


if __name__ == "__main__":
    main()
