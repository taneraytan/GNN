from __future__ import annotations

import numpy as np
import pandas as pd


def top_feature_contributions(
    feature_errors: np.ndarray,
    feature_columns: list[str],
    top_k: int = 3,
) -> pd.Series:
    """Summarize which features drive each row's anomaly score.

    Returns one human-readable string per row, e.g.
    ``"amount (62%), created_at__timestamp (21%), k (9%)"`` where the percentage is
    that feature's share of the row's total reconstruction error (or robust
    deviation when the scikit-learn fallback ran).
    """

    errors = np.asarray(feature_errors, dtype=float)
    if errors.ndim != 2 or errors.shape[0] == 0 or not feature_columns:
        return pd.Series([""] * (errors.shape[0] if errors.ndim == 2 else 0), name="top_outlier_features", dtype="object")

    top_k = max(1, min(top_k, errors.shape[1]))
    totals = errors.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    shares = errors / totals
    order = np.argsort(-errors, axis=1)[:, :top_k]

    labels: list[str] = []
    for row, columns in enumerate(order):
        parts = [
            f"{feature_columns[column]} ({shares[row, column]:.0%})"
            for column in columns
            if shares[row, column] > 0
        ]
        labels.append(", ".join(parts))
    return pd.Series(labels, name="top_outlier_features", dtype="object")
