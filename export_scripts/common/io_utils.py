import json
from pathlib import Path

import pandas as pd

from common.path_config import assert_new_output


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_table(frame: pd.DataFrame, path: Path, overwrite: bool = False) -> None:
    assert_new_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"New output exists; pass --overwrite-new-output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")


def write_json(data: dict, path: Path, overwrite: bool = False) -> None:
    assert_new_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"New output exists; pass --overwrite-new-output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
