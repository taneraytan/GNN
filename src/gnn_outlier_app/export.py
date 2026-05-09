from __future__ import annotations

from io import BytesIO

import pandas as pd


def attach_results(frame: pd.DataFrame, scores, is_outlier) -> pd.DataFrame:
    """Return a copy of the source table with anomaly results appended."""

    output = frame.copy()
    output["outlier_score"] = scores
    output["is_outlier"] = is_outlier
    return output.sort_values("outlier_score", ascending=False)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def to_excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="outliers")
    return buffer.getvalue()


def to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()
