from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPORT_ROOT = PROJECT_ROOT / "02_exports"
PLOT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"
PNG_ROOT = OUTPUT_ROOT
PDF_ROOT = OUTPUT_ROOT
SOURCE_ROOT = OUTPUT_ROOT / "source_data_used"
LOG_ROOT = OUTPUT_ROOT / "logs"
SOURCE_MANIFEST = SOURCE_ROOT / "figure_source_data_manifest.csv"


def ensure_output_dirs():
    for path in (PDF_ROOT, SOURCE_ROOT, LOG_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def export_path(relative_path):
    path = (PROJECT_ROOT / relative_path).resolve()
    if EXPORT_ROOT.resolve() not in path.parents:
        raise ValueError(f"Primary data input must be under 02_exports: {path}")
    return path
