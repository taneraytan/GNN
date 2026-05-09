from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, insert
from sqlalchemy.engine import Engine

metadata = MetaData()

runs = Table(
    "runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("source_name", String, nullable=False),
    Column("row_count", Integer, nullable=False),
    Column("feature_count", Integer, nullable=False),
    Column("architectures", String, nullable=False),
    Column("backend", String, nullable=False),
    Column("threshold", Float, nullable=False),
)

outlier_rows = Table(
    "outlier_rows",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, nullable=False),
    Column("row_index", Integer, nullable=False),
    Column("score", Float, nullable=False),
    Column("is_outlier", Boolean, nullable=False),
    Column("payload_json", String, nullable=False),
)


def get_engine(db_path: str | Path = "outlier_runs.sqlite") -> Engine:
    """Create a SQLite engine for local persistence."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", future=True)
    metadata.create_all(engine)
    return engine


def save_run(
    engine: Engine,
    source_name: str,
    frame: pd.DataFrame,
    scores: pd.Series,
    is_outlier: pd.Series,
    architectures: list[str],
    backend: str,
    threshold: float,
) -> int:
    """Persist run metadata and row-level anomaly decisions in SQLite."""

    with engine.begin() as connection:
        result = connection.execute(
            insert(runs).values(
                created_at=datetime.now(timezone.utc),
                source_name=source_name,
                row_count=len(frame),
                feature_count=len(frame.columns),
                architectures=", ".join(architectures),
                backend=backend,
                threshold=float(threshold),
            )
        )
        run_id = int(result.inserted_primary_key[0])
        payload = frame.copy()
        payload["outlier_score"] = scores.values
        payload["is_outlier"] = is_outlier.values
        records = [
            {
                "run_id": run_id,
                "row_index": int(index),
                "score": float(row["outlier_score"]),
                "is_outlier": bool(row["is_outlier"]),
                "payload_json": row.to_json(date_format="iso"),
            }
            for index, row in payload.iterrows()
        ]
        if records:
            connection.execute(insert(outlier_rows), records)
    return run_id


def recent_runs(engine: Engine, limit: int = 10) -> pd.DataFrame:
    """Return recent detection runs from the local database."""

    query = f"SELECT * FROM runs ORDER BY created_at DESC LIMIT {int(limit)}"
    return pd.read_sql_query(query, engine)
