from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from src.gnn_outlier_app.data import load_table, numeric_columns, prepare_features
from src.gnn_outlier_app.database import get_engine, recent_runs, save_run
from src.gnn_outlier_app.export import attach_results
from src.gnn_outlier_app.graph import build_knn_graph
from src.gnn_outlier_app.models import run_detection


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


def test_build_knn_graph_is_symmetric() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 10.0], "y": [0.0, 1.0, 10.0]})
    graph = build_knn_graph(frame, ["x", "y"], k=1)
    edges = {tuple(edge) for edge in graph.edge_index.T.tolist()}

    assert graph.features.shape == (3, 2)
    assert all((target, source) in edges for source, target in edges)


def test_detection_exports_and_database_roundtrip(tmp_path) -> None:
    frame = pd.DataFrame({"x": [0, 1, 2, 100], "y": [0, 1, 2, 100]})
    graph = build_knn_graph(frame, ["x", "y"], k=2)
    result = run_detection(graph.features, graph.edge_index, ["GCN", "GAT"], contamination=0.25, epochs=1)
    output = attach_results(frame, result.scores, result.is_outlier)

    assert len(result.scores) == len(frame)
    assert "outlier_score" in output.columns
    assert "is_outlier" in output.columns

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
    )
    history = recent_runs(engine)

    assert run_id == 1
    assert history.loc[0, "source_name"] == "unit.csv"
    assert np.isfinite(history.loc[0, "threshold"])
