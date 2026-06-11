from __future__ import annotations

from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from src.gnn_outlier_app.export import attach_results, sanitize_for_spreadsheet, to_csv_bytes, to_excel_bytes, to_parquet_bytes


def _injection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "note": ["=cmd|' /C calc'!A0", "+SUM(A1:A9)", "-2+3", "@payload", "safe text"],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_sanitize_neutralizes_formula_prefixes() -> None:
    sanitized = sanitize_for_spreadsheet(_injection_frame())
    assert sanitized["note"].tolist() == [
        "'=cmd|' /C calc'!A0",
        "'+SUM(A1:A9)",
        "'-2+3",
        "'@payload",
        "safe text",
    ]
    # Numeric columns are untouched.
    assert sanitized["amount"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_sanitize_escapes_malicious_headers() -> None:
    frame = pd.DataFrame({"=HYPERLINK(...)": [1, 2]})
    sanitized = sanitize_for_spreadsheet(frame)
    assert sanitized.columns.tolist() == ["'=HYPERLINK(...)"]


def test_csv_export_is_escaped() -> None:
    text = to_csv_bytes(_injection_frame()).decode("utf-8")
    for line in text.splitlines()[1:]:
        assert not line.startswith(("=", "+", "-", "@"))


def test_excel_export_is_escaped() -> None:
    workbook = load_workbook(BytesIO(to_excel_bytes(_injection_frame())))
    sheet = workbook["outliers"]
    assert sheet["A2"].value == "'=cmd|' /C calc'!A0"


def test_parquet_export_roundtrip_preserves_raw_values() -> None:
    frame = _injection_frame()
    restored = pd.read_parquet(BytesIO(to_parquet_bytes(frame)))
    assert restored["note"].tolist() == frame["note"].tolist()


def test_attach_results_includes_explanations() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3]})
    explanations = pd.Series(["x (100%)", "x (100%)", "x (100%)"])
    output = attach_results(frame, [0.1, 0.9, 0.2], [False, True, False], explanations)
    assert output.iloc[0]["outlier_score"] == 0.9
    assert "top_outlier_features" in output.columns
