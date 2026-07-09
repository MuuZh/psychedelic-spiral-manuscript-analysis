import numpy as np
import pandas as pd

from paths import EXPORT_ROOT, export_path

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def require_columns(frame, columns, source):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def read_table(relative_path, required=()):
    path = export_path(relative_path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    require_columns(frame, required, path)
    return frame


def read_manifest_array(manifest_relative_path, row, key_column="array_key"):
    manifest_path = export_path(manifest_relative_path)
    key = row[key_column]
    binary_value = str(row["binary_file"])
    binary = export_path(binary_value) if "/" in binary_value or "\\" in binary_value else (manifest_path.parent / binary_value).resolve()
    if EXPORT_ROOT.resolve() not in binary.parents:
        raise ValueError(f"Manifest array resolved outside 02_exports: {binary}")
    with np.load(binary, allow_pickle=False) as arrays:
        if key not in arrays:
            raise KeyError(f"{key} is not present in {binary}")
        return arrays[key].copy(), manifest_path, binary


def standardize_dataset(value):
    text = str(value).upper()
    if "DMT" in text:
        return "DMT"
    if "LSD" in text:
        return "LSD"
    raise ValueError(f"Cannot standardize dataset label: {value}")


def standardize_condition(value):
    text = str(value).lower()
    if "pcb" in text or "placebo" in text:
        return "PCB"
    if "drug" in text or "dmt" in text or "lsd" in text:
        return "Drug"
    raise ValueError(f"Cannot standardize condition label: {value}")


def standardize_hemisphere(value):
    text = str(value).lower()
    if text in {"left", "right"}:
        return text
    raise ValueError(f"Cannot standardize hemisphere label: {value}")


def normalize_scalar(frame, dataset_col, condition_col, value_col, metric, source_export_id):
    require_columns(frame, [dataset_col, condition_col, "subid", "hemisphere", value_col], metric)
    out = frame[[dataset_col, condition_col, "subid", "hemisphere", value_col]].copy()
    out.columns = ["study", "condition", "subject", "hemisphere", "value"]
    out = out[out["hemisphere"].astype(str).str.lower().isin(["left", "right"])].copy()
    out["study"] = out["study"].map(standardize_dataset)
    out["condition"] = out["condition"].map(standardize_condition)
    out["hemisphere"] = out["hemisphere"].map(standardize_hemisphere)
    out["subject"] = out["subject"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["metric"] = metric
    out["source_export_id"] = source_export_id
    return out


def prepare_paired_plot_df(frame):
    required = ["study", "condition", "subject", "hemisphere", "value"]
    require_columns(frame, required, "normalized scalar frame")
    out = frame.dropna(subset=required).copy()
    key = ["study", "hemisphere", "subject", "condition"]
    duplicates = out.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"Paired scalar frame contains duplicate pair cells: {out.loc[duplicates, key].head().to_dict('records')}")
    counts = out.groupby(["study", "hemisphere", "subject"])["condition"].agg(lambda x: set(x))
    paired_keys = counts[counts.map(lambda x: x == {"PCB", "Drug"})].index
    out = out.set_index(["study", "hemisphere", "subject"]).loc[paired_keys].reset_index()
    return out.sort_values(["study", "hemisphere", "subject", "condition"]).reset_index(drop=True)
