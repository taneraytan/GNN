from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xls", ".xlsx", ".parquet"}


@dataclass(frozen=True)
class LoadedTable:
    """Container for a loaded tabular dataset."""

    frame: pd.DataFrame
    source_name: str


def _read_delimited(file: BinaryIO | str | Path, suffix: str) -> pd.DataFrame:
    if suffix == ".txt":
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_csv(file)


def load_table(file: BinaryIO | str | Path, source_name: str | None = None) -> LoadedTable:
    """Load a supported local tabular file into a pandas DataFrame."""

    name = source_name or getattr(file, "name", None) or str(file)
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}.")

    if suffix in {".csv", ".txt"}:
        frame = _read_delimited(file, suffix)
    elif suffix in {".xls", ".xlsx"}:
        frame = pd.read_excel(file)
    else:
        frame = pd.read_parquet(file)

    frame = frame.reset_index(drop=True)
    return LoadedTable(frame=frame, source_name=Path(name).name)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric columns that can be modeled by the detector."""

    return frame.select_dtypes(include="number").columns.tolist()
