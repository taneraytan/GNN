from __future__ import annotations

import numpy as np

from src.gnn_outlier_app.explain import top_feature_contributions


def test_top_feature_contributions_orders_by_error_share() -> None:
    errors = np.array(
        [
            [9.0, 1.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    labels = top_feature_contributions(errors, ["a", "b", "c"], top_k=2)

    assert labels[0] == "a (90%), b (10%)"
    assert labels[1] == "c (100%)"


def test_top_feature_contributions_handles_empty_inputs() -> None:
    assert top_feature_contributions(np.empty((0, 3)), ["a", "b", "c"]).empty
    assert top_feature_contributions(np.ones((2, 2)), []).tolist() == ["", ""]


def test_top_feature_contributions_caps_top_k() -> None:
    errors = np.array([[1.0, 2.0]])
    labels = top_feature_contributions(errors, ["a", "b"], top_k=10)
    assert labels[0] == "b (67%), a (33%)"
