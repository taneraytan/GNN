from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
import warnings

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xls", ".xlsx", ".parquet"}


@dataclass(frozen=True)
class LoadedTable:
    """Container for a loaded tabular dataset."""

    frame: pd.DataFrame
    source_name: str


@dataclass(frozen=True)
class FeaturePreparation:
    """Automatically engineered model features and audit metadata."""

    frame: pd.DataFrame
    feature_columns: list[str]
    source_columns: list[str]
    dropped_columns: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


NUMERIC_PARSE_THRESHOLD = 0.8
DATETIME_PARSE_THRESHOLD = 0.8
MAX_CATEGORY_LEVELS = 20
MAX_FEATURES = 80


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


def prepare_features(frame: pd.DataFrame, max_features: int = MAX_FEATURES) -> FeaturePreparation:
    """Automatically preprocess a table and select model-ready features.

    The app is intentionally unsupervised, so feature preparation avoids target-aware
    selection. It keeps informative numeric signals, coerces numeric-looking text,
    expands usable datetime and low-cardinality categorical fields, removes constants
    and identifier-like columns, imputes missing values, and caps the final feature
    matrix by unsupervised variance so users do not need to hand-pick columns.
    """

    if frame.empty:
        return FeaturePreparation(frame=pd.DataFrame(index=frame.index), feature_columns=[], source_columns=[], notes=["Dataset has no rows."])

    prepared_parts: list[pd.DataFrame] = []
    source_columns: list[str] = []
    dropped_columns: dict[str, str] = {}
    notes: list[str] = []

    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        if non_null.empty:
            dropped_columns[str(column)] = "all values are missing"
            continue
        if non_null.nunique(dropna=True) <= 1:
            dropped_columns[str(column)] = "constant value"
            continue

        if _looks_like_identifier(series, str(column)):
            dropped_columns[str(column)] = "identifier-like high-cardinality column"
            continue

        part = _prepare_column(series, str(column), dropped_columns)
        if part is None or part.empty:
            continue
        prepared_parts.append(part)
        source_columns.append(str(column))

    if not prepared_parts:
        notes.append("No usable columns remained after automatic preprocessing.")
        return FeaturePreparation(frame=pd.DataFrame(index=frame.index), feature_columns=[], source_columns=[], dropped_columns=dropped_columns, notes=notes)

    prepared = pd.concat(prepared_parts, axis=1)
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    prepared = prepared.apply(pd.to_numeric, errors="coerce")
    prepared = prepared.fillna(prepared.median(numeric_only=True)).fillna(0.0)
    prepared = _deduplicate_columns(prepared)

    variable = prepared.nunique(dropna=False) > 1
    for column in prepared.columns[~variable]:
        dropped_columns[str(column)] = "engineered feature is constant"
    prepared = prepared.loc[:, variable]

    if prepared.empty:
        notes.append("Automatic preprocessing produced only constant features.")
        return FeaturePreparation(frame=prepared, feature_columns=[], source_columns=source_columns, dropped_columns=dropped_columns, notes=notes)

    prepared = _select_by_variance(prepared, max_features=max_features)
    if len(prepared.columns) == max_features:
        notes.append(f"Selected the top {max_features} engineered features by min-max-scaled dispersion.")
    notes.append(f"Prepared {len(prepared.columns)} model-ready features from {len(source_columns)} source columns.")

    duplicate_rows = int(prepared.duplicated().sum())
    if duplicate_rows:
        notes.append(
            f"{duplicate_rows} rows are exact duplicates in the engineered feature space; "
            "duplicates reconstruct each other perfectly and can mask anomalies."
        )

    return FeaturePreparation(
        frame=prepared,
        feature_columns=prepared.columns.tolist(),
        source_columns=source_columns,
        dropped_columns=dropped_columns,
        notes=notes,
    )


def _prepare_column(series: pd.Series, name: str, dropped_columns: dict[str, str]) -> pd.DataFrame | None:
    if pd.api.types.is_bool_dtype(series):
        return pd.DataFrame({name: series.astype("float32")}, index=series.index)

    if pd.api.types.is_numeric_dtype(series):
        return pd.DataFrame({name: pd.to_numeric(series, errors="coerce")}, index=series.index)

    numeric = pd.to_numeric(series, errors="coerce")
    if _parse_ratio(numeric, series) >= NUMERIC_PARSE_THRESHOLD:
        return pd.DataFrame({f"{name}__numeric": numeric}, index=series.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        datetime = pd.to_datetime(series, errors="coerce", utc=True)
    if _parse_ratio(datetime, series) >= DATETIME_PARSE_THRESHOLD:
        return _datetime_features(datetime, name)

    categories = series.astype("string").str.strip().replace("", pd.NA)
    cardinality = categories.nunique(dropna=True)
    if 2 <= cardinality <= MAX_CATEGORY_LEVELS:
        return pd.get_dummies(categories, prefix=name, dummy_na=True, dtype="float32")

    dropped_columns[name] = f"categorical cardinality ({cardinality}) is outside the automatic range"
    return None


def _parse_ratio(parsed: pd.Series, original: pd.Series) -> float:
    candidates = original.notna().sum()
    if candidates == 0:
        return 0.0
    return float(parsed.notna().sum() / candidates)


def _datetime_features(series: pd.Series, name: str) -> pd.DataFrame:
    timestamp = series.astype("int64").astype("float64")
    timestamp = timestamp.mask(series.isna(), np.nan)
    return pd.DataFrame(
        {
            f"{name}__timestamp": timestamp,
            f"{name}__month": series.dt.month.astype("float64"),
            f"{name}__dayofweek": series.dt.dayofweek.astype("float64"),
        },
        index=series.index,
    )


def _looks_like_identifier(series: pd.Series, name: str) -> bool:
    lowered = name.lower()
    non_null = series.dropna()
    if non_null.empty or len(non_null) < 3:
        return False
    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    id_name = lowered == "id" or lowered.endswith("_id") or lowered.endswith(" id") or lowered in {"uuid", "guid"}
    if id_name and unique_ratio > 0.9:
        return True
    row_number_name = lowered in {"row", "row_number", "index"}
    if row_number_name and pd.api.types.is_integer_dtype(series) and unique_ratio > 0.9 and non_null.is_monotonic_increasing:
        return True
    return False


def _deduplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    duplicated = frame.T.duplicated()
    if not duplicated.any():
        return frame
    return frame.loc[:, ~duplicated.to_numpy()]


def _select_by_variance(frame: pd.DataFrame, max_features: int) -> pd.DataFrame:
    if max_features <= 0 or frame.shape[1] <= max_features:
        return frame
    # Standardizing first would make every non-constant column's variance exactly 1,
    # so rank dispersion on min-max scaled values instead: scale-free, and columns
    # whose mass concentrates near one end of their range score lower.
    col_min = frame.min(axis=0)
    col_range = (frame.max(axis=0) - col_min).replace(0, 1)
    normalized = (frame - col_min) / col_range
    dispersion = normalized.var(axis=0, ddof=0).fillna(0)
    selected = dispersion.sort_values(ascending=False).head(max_features).index.tolist()
    return frame.loc[:, selected]
