from pathlib import Path

import numpy as np
import pandas as pd


def validate_preserved_columns(source: pd.DataFrame, exported: pd.DataFrame) -> tuple[str, str]:
    missing = [column for column in source.columns if column not in exported.columns]
    if missing:
        return "failed", f"Missing source columns: {missing}"
    candidate = exported.loc[:, source.columns]
    if len(source) != len(candidate):
        return "failed", f"Row count mismatch: source={len(source)}, export={len(candidate)}"
    for column in source.columns:
        left = source[column]
        right = candidate[column]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                equal_nan=True,
            ):
                return "failed", f"Numeric values differ in column {column}"
        elif not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
            return "failed", f"Values differ in column {column}"
    return "pass", "All original columns, rows, and values preserved"


def validate_written_table(source: pd.DataFrame, output: Path) -> tuple[str, str]:
    exported = pd.read_parquet(output) if output.suffix.lower() == ".parquet" else pd.read_csv(output)
    return validate_preserved_columns(source, exported)
