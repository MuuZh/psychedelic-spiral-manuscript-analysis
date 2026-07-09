import pandas as pd


def numeric_summary(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include="number")
    return {
        column: {
            "n": int(values.notna().sum()),
            "mean": float(values.mean()) if values.notna().any() else None,
            "std": float(values.std()) if values.notna().sum() > 1 else None,
        }
        for column, values in numeric.items()
    }
