from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from src.gnn_outlier_app.data import load_table, numeric_columns, prepare_features
from src.gnn_outlier_app.database import get_engine, recent_runs, save_run
from src.gnn_outlier_app.export import attach_results
from src.gnn_outlier_app.graph import build_knn_graph
from src.gnn_outlier_app.models import ARCHITECTURE_DESCRIPTIONS, FALLBACK_DETECTOR_NAMES, _torch_geometric_available, run_detection


def test_load_table_csv_and_numeric_columns() -> None:
    payload = BytesIO(b"a,b,c\n1,2,x\n3,4,y\n")
    loaded = load_table(payload, "sample.csv")

    assert loaded.source_name == "sample.csv"
    assert loaded.frame.shape == (2, 3)
    assert numeric_columns(loaded.frame) == ["a", "b"]


def test_prepare_features_automates_preprocessing_and_selection() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "amount": [10.0, None, 15.5, 999.0],
            "numeric_text": ["1", "2", None, "4"],
            "segment": ["retail", "business", "retail", "enterprise"],
            "created_at": ["2024-01-01", "2024-01-02", "2024-01-03", None],
            "constant": ["same", "same", "same", "same"],
        }
    )

    prepared = prepare_features(frame)

    assert "customer_id" in prepared.dropped_columns
    assert "constant" in prepared.dropped_columns
    assert "amount" in prepared.feature_columns
    assert "numeric_text__numeric" in prepared.feature_columns
    assert any(column.startswith("segment_") for column in prepared.feature_columns)
    assert any(column.startswith("created_at__") for column in prepared.feature_columns)
    assert prepared.frame.isna().sum().sum() == 0


def test_prepare_features_empty_frame() -> None:
    prepared = prepare_features(pd.DataFrame())
    assert prepared.feature_columns == []
    assert prepared.notes == ["Dataset has no rows."]


def test_prepare_features_all_high_cardinality_categorical() -> None:
    frame = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(30)]})
    prepared = prepare_features(frame)
    assert prepared.feature_columns == []
    assert "email" in prepared.dropped_columns


def test_prepare_features_notes_duplicate_rows() -> None:
    frame = pd.DataFrame({"x": [1.0, 1.0, 1.0, 5.0], "y": [2.0, 2.0, 2.0, 9.0]})
    prepared = prepare_features(frame)
    assert any("duplicates" in note for note in prepared.notes)


def test_select_by_variance_actually_selects_dispersed_columns() -> None:
    rng = np.random.default_rng(0)
    # 'spread' fills its range evenly; 'spike' concentrates near zero with one extreme.
    frame = pd.DataFrame(
        {
            "spike": np.concatenate([rng.normal(0, 0.01, 99), [100.0]]),
            "spread": np.linspace(0, 1, 100),
        }
    )
    prepared = prepare_features(frame, max_features=1)
    assert prepared.feature_columns == ["spread"]


def test_build_knn_graph_is_symmetric() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 10.0], "y": [0.0, 1.0, 10.0]})
    graph = build_knn_graph(frame, ["x", "y"], k=1)
    edges = {tuple(edge) for edge in graph.edge_index.T.tolist()}

    assert graph.features.shape == (3, 2)
    assert all((target, source) in edges for source, target in edges)
    assert all(source != target for source, target in edges)


def test_detection_exports_and_database_roundtrip(tmp_path) -> None:
    frame = pd.DataFrame({"x": [0, 1, 2, 3, 4, 100], "y": [0, 1, 2, 3, 4, 100]})
    graph = build_knn_graph(frame, ["x", "y"], k=2)
    result = run_detection(graph.features, graph.edge_index, ["GCN", "GAT"], contamination=0.25, epochs=1)
    output = attach_results(frame, result.scores, result.is_outlier)

    assert len(result.scores) == len(frame)
    assert "outlier_score" in output.columns
    assert "is_outlier" in output.columns
    assert result.feature_errors.shape == graph.features.shape
    if _torch_geometric_available():
        assert set(result.detector_scores) == {"GCN", "GAT"}
    else:
        assert set(result.detector_scores) == set(FALLBACK_DETECTOR_NAMES)

    engine = get_engine(tmp_path / "runs.sqlite")
    run_id = save_run(
        engine,
        "unit.csv",
        frame,
        pd.Series(result.scores),
        pd.Series(result.is_outlier),
        ["GCN", "GAT"],
        result.used_backend,
        result.threshold,
        feature_count=2,
        params={"k": 2, "contamination": 0.25, "epochs": 1, "hidden_dim": 32, "seed": 42, "threshold_strategy": "contamination"},
        persist_mode="all",
    )
    history = recent_runs(engine)

    assert run_id == 1
    assert history.loc[0, "source_name"] == "unit.csv"
    assert np.isfinite(history.loc[0, "threshold"])
    assert history.loc[0, "feature_count"] == 2
    assert history.loc[0, "k"] == 2
    assert history.loc[0, "threshold_strategy"] == "contamination"


def test_run_detection_uses_all_selected_architectures_keys() -> None:
    assert set(ARCHITECTURE_DESCRIPTIONS) == {"GCN", "GraphSAGE", "GAT", "GIN", "ChebNet", "APPNP"}
