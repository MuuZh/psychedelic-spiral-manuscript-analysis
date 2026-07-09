from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "export_scripts"
EXPORT_ROOT = PROJECT_ROOT / "02_exports"
LOG_ROOT = PROJECT_ROOT / "03_logs"

OLD_ANALYSIS_OUTPUTS = Path(r"<analysis_outputs>")
OLD_TOOLBOX_ANALYSIS = Path(r"<source_analysis>")


def ensure_new_roots() -> None:
    for path in (
        EXPORT_ROOT / "metadata",
        EXPORT_ROOT / "summary_validation",
        LOG_ROOT / "rerun_logs",
        LOG_ROOT / "validation_logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def assert_new_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != EXPORT_ROOT.resolve() and EXPORT_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"Refusing to write outside new export root: {resolved}")
    return resolved
