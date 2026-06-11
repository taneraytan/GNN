from __future__ import annotations

from io import BytesIO

import pandas as pd

# Cells beginning with these characters execute as formulas when a CSV/Excel
# export is opened in a spreadsheet (OWASP CSV injection).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def attach_results(frame: pd.DataFrame, scores, is_outlier, explanations: pd.Series | None = None) -> pd.DataFrame:
    """Return a copy of the source table with anomaly results appended."""

    output = frame.copy()
    output["outlier_score"] = scores
    output["is_outlier"] = is_outlier
    if explanations is not None:
        output["top_outlier_features"] = explanations.values
    return output.sort_values("outlier_score", ascending=False)


def _escape_cell(value):
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def sanitize_for_spreadsheet(frame: pd.DataFrame) -> pd.DataFrame:
    """Neutralize spreadsheet formula injection in string cells and headers."""

    sanitized = frame.copy()
    for column in sanitized.columns:
        if sanitized[column].dtype == object or pd.api.types.is_string_dtype(sanitized[column]):
            sanitized[column] = sanitized[column].map(_escape_cell)
    sanitized.columns = [_escape_cell(str(column)) for column in sanitized.columns]
    return sanitized


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return sanitize_for_spreadsheet(frame).to_csv(index=False).encode("utf-8")


def to_excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sanitize_for_spreadsheet(frame).to_excel(writer, index=False, sheet_name="outliers")
    return buffer.getvalue()


def to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()
