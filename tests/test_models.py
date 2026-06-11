from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.gnn_outlier_app.graph import build_knn_graph
from src.gnn_outlier_app.models import (
    FALLBACK_DETECTOR_NAMES,
    MIN_ROWS,
    _compute_threshold,
    _rank_normalize,
    _run_sklearn_fallback,
    run_detection,
)


def _toy_graph(rows: int = 30, with_outlier: bool = True):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"x": rng.normal(0, 1, rows), "y": rng.normal(0, 1, rows)})
    if with_outlier:
        frame.loc[rows - 1] = [50.0, 50.0]
    return frame, build_knn_graph(frame, ["x", "y"], k=3)


def test_run_detection_rejects_empty_architectures() -> None:
    _, graph = _toy_graph()
    with pytest.raises(ValueError, match="at least one"):
        run_detection(graph.features, graph.edge_index, [], epochs=1)
    with pytest.raises(ValueError, match="at least one"):
        run_detection(graph.features, graph.edge_index, ["NotAModel"], epochs=1)


def test_run_detection_rejects_tiny_datasets() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    graph = build_knn_graph(frame, ["x", "y"], k=1)
    with pytest.raises(ValueError, match=f"At least {MIN_ROWS} rows"):
        run_detection(graph.features, graph.edge_index, ["GCN"], epochs=1)


def test_run_detection_rejects_unknown_threshold_strategy() -> None:
    _, graph = _toy_graph()
    with pytest.raises(ValueError, match="threshold strategy"):
        run_detection(graph.features, graph.edge_index, ["GCN"], epochs=1, threshold_strategy="magic")


def test_fallback_reports_honest_detector_names() -> None:
    _, graph = _toy_graph()
    result = _run_sklearn_fallback(graph.features, contamination=0.1, threshold_strategy="contamination", seed=42, progress_callback=None)
    assert set(result.detector_scores) == set(FALLBACK_DETECTOR_NAMES)
    assert "IsolationForest" in result.used_backend
    assert "GCN" not in result.used_backend
    assert result.feature_errors.shape == graph.features.shape


def test_fallback_is_deterministic() -> None:
    _, graph = _toy_graph()
    first = _run_sklearn_fallback(graph.features, 0.1, "contamination", 42, None)
    second = _run_sklearn_fallback(graph.features, 0.1, "contamination", 42, None)
    assert np.allclose(first.scores, second.scores)


def test_mad_strategy_flags_extreme_row() -> None:
    frame, graph = _toy_graph(rows=40, with_outlier=True)
    result = run_detection(graph.features, graph.edge_index, ["GCN"], epochs=1, threshold_strategy="mad")
    extreme_position = len(frame) - 1
    assert bool(result.is_outlier[extreme_position])
    assert result.threshold_strategy == "mad"


def test_compute_threshold_mad_maps_magnitude_decision_to_score_scale() -> None:
    scores = np.linspace(0, 1, 100)
    magnitudes = np.concatenate([np.tile([0.4, 0.5, 0.6], 33), [50.0]])
    threshold = _compute_threshold(scores, magnitudes, contamination=0.05, strategy="mad")
    assert threshold == pytest.approx(scores[-1])

    calm = np.tile([0.4, 0.5, 0.6], 33)
    threshold_none = _compute_threshold(scores[: len(calm)], calm, contamination=0.05, strategy="mad")
    assert threshold_none > scores[: len(calm)].max()


def test_rank_normalize_bounds_and_robustness() -> None:
    values = np.array([1.0, 2.0, 3.0, 1_000_000.0])
    ranked = _rank_normalize(values)
    assert ranked.min() == 0.0
    assert ranked.max() == 1.0
    # An extreme value cannot compress the spacing of the others.
    assert np.allclose(np.diff(ranked), 1 / 3)


def test_progress_callback_is_invoked() -> None:
    _, graph = _toy_graph()
    calls: list[tuple[str, int, int]] = []
    run_detection(
        graph.features,
        graph.edge_index,
        ["GCN", "GAT"],
        epochs=1,
        progress_callback=lambda label, done, total: calls.append((label, done, total)),
    )
    assert calls
    assert calls[-1][1] == calls[-1][2]
