from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    URL,
    create_engine,
    delete,
    insert,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine

DATA_ROOT = Path("data")
WORKSPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

PERSIST_OUTLIERS = "outliers"
PERSIST_ALL = "all"
PERSIST_MODES = (PERSIST_OUTLIERS, PERSIST_ALL)

VERDICT_UNREVIEWED = "Unreviewed"
VERDICT_CONFIRMED = "Confirmed"
VERDICT_FALSE_POSITIVE = "False positive"
VERDICTS = (VERDICT_UNREVIEWED, VERDICT_CONFIRMED, VERDICT_FALSE_POSITIVE)

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
    Column("k", Integer, nullable=True),
    Column("contamination", Float, nullable=True),
    Column("epochs", Integer, nullable=True),
    Column("hidden_dim", Integer, nullable=True),
    Column("seed", Integer, nullable=True),
    Column("threshold_strategy", String, nullable=True),
    Column("persisted_rows", String, nullable=True),
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
    Column("verdict", String, nullable=True),
)


def resolve_database_path(workspace: str, data_root: str | Path = DATA_ROOT) -> Path:
    """Map a user-supplied workspace name to a SQLite path locked under the data root.

    Rejects path separators, traversal, absolute paths, and hidden-file prefixes so a
    name typed into the UI can never create files outside ``data/``.
    """

    name = str(workspace).strip()
    if name.endswith(".sqlite"):
        name = name[: -len(".sqlite")]
    if not name:
        raise ValueError("Workspace name cannot be empty.")
    if not WORKSPACE_PATTERN.match(name):
        raise ValueError(
            "Workspace names may only contain letters, numbers, dots, dashes, and underscores, "
            "and must start with a letter or number."
        )
    root = Path(data_root).resolve()
    path = (root / f"{name}.sqlite").resolve()
    if root != path.parent:
        raise ValueError("Workspace path escapes the local data directory.")
    return path


def get_engine(db_path: str | Path = DATA_ROOT / "outlier_runs.sqlite") -> Engine:
    """Create a SQLite engine for local persistence and migrate missing columns."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(URL.create("sqlite", database=str(path)), future=True)
    metadata.create_all(engine)
    _ensure_columns(engine)
    return engine


def _ensure_columns(engine: Engine) -> None:
    """Add columns introduced after a database file was first created."""

    inspector = inspect(engine)
    for table in (runs, outlier_rows):
        existing = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name not in existing:
                ddl = f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {column.type.compile(engine.dialect)}'
                with engine.begin() as connection:
                    connection.execute(text(ddl))


def save_run(
    engine: Engine,
    source_name: str,
    frame: pd.DataFrame,
    scores: pd.Series,
    is_outlier: pd.Series,
    architectures: list[str],
    backend: str,
    threshold: float,
    *,
    feature_count: int | None = None,
    params: dict | None = None,
    persist_mode: str = PERSIST_OUTLIERS,
) -> int:
    """Persist run metadata plus row decisions (flagged rows only unless opted in)."""

    if persist_mode not in PERSIST_MODES:
        raise ValueError(f"persist_mode must be one of {PERSIST_MODES}.")
    params = params or {}

    with engine.begin() as connection:
        result = connection.execute(
            insert(runs).values(
                created_at=datetime.now(timezone.utc),
                source_name=source_name,
                row_count=len(frame),
                feature_count=int(feature_count) if feature_count is not None else len(frame.columns),
                architectures=", ".join(architectures),
                backend=backend,
                threshold=float(threshold),
                k=params.get("k"),
                contamination=params.get("contamination"),
                epochs=params.get("epochs"),
                hidden_dim=params.get("hidden_dim"),
                seed=params.get("seed"),
                threshold_strategy=params.get("threshold_strategy"),
                persisted_rows=persist_mode,
            )
        )
        run_id = int(result.inserted_primary_key[0])

        payload = frame.copy()
        payload["outlier_score"] = np.asarray(scores, dtype=float)
        payload["is_outlier"] = np.asarray(is_outlier, dtype=bool)
        if persist_mode == PERSIST_OUTLIERS:
            payload = payload[payload["is_outlier"]]

        row_indices = payload.index.tolist()
        serialized = json.loads(payload.to_json(orient="records", date_format="iso"))
        records = [
            {
                "run_id": run_id,
                "row_index": int(index),
                "score": float(record["outlier_score"]),
                "is_outlier": bool(record["is_outlier"]),
                "payload_json": json.dumps(record),
                "verdict": None,
            }
            for index, record in zip(row_indices, serialized)
        ]
        if records:
            connection.execute(insert(outlier_rows), records)
    return run_id


def recent_runs(engine: Engine, limit: int = 10) -> pd.DataFrame:
    """Return recent detection runs from the local database."""

    query = select(runs).order_by(runs.c.created_at.desc(), runs.c.id.desc()).limit(int(limit))
    return pd.read_sql_query(query, engine)


def load_run_rows(engine: Engine, run_id: int) -> pd.DataFrame:
    """Return persisted row decisions and verdicts for one run."""

    query = (
        select(
            outlier_rows.c.row_index,
            outlier_rows.c.score,
            outlier_rows.c.is_outlier,
            outlier_rows.c.verdict,
        )
        .where(outlier_rows.c.run_id == int(run_id))
        .order_by(outlier_rows.c.row_index)
    )
    return pd.read_sql_query(query, engine)


def save_verdicts(engine: Engine, run_id: int, verdicts: dict[int, str]) -> int:
    """Persist analyst verdicts (confirmed / false positive) for flagged rows."""

    updated = 0
    with engine.begin() as connection:
        for row_index, verdict in verdicts.items():
            if verdict not in VERDICTS:
                raise ValueError(f"Verdict must be one of {VERDICTS}.")
            stored = None if verdict == VERDICT_UNREVIEWED else verdict
            result = connection.execute(
                update(outlier_rows)
                .where(outlier_rows.c.run_id == int(run_id), outlier_rows.c.row_index == int(row_index))
                .values(verdict=stored)
            )
            updated += result.rowcount
    return updated


def feedback_summary(engine: Engine) -> pd.DataFrame:
    """Per-run precision from analyst verdicts: confirmed / (confirmed + false positive)."""

    query = select(outlier_rows.c.run_id, outlier_rows.c.verdict).where(outlier_rows.c.verdict.is_not(None))
    rows = pd.read_sql_query(query, engine)
    if rows.empty:
        return pd.DataFrame(columns=["run_id", "confirmed", "false_positive", "precision"])
    summary = (
        rows.assign(
            confirmed=(rows["verdict"] == VERDICT_CONFIRMED).astype(int),
            false_positive=(rows["verdict"] == VERDICT_FALSE_POSITIVE).astype(int),
        )
        .groupby("run_id", as_index=False)[["confirmed", "false_positive"]]
        .sum()
    )
    reviewed = summary["confirmed"] + summary["false_positive"]
    summary["precision"] = (summary["confirmed"] / reviewed.replace(0, np.nan)).round(3)
    return summary


def compare_runs(engine: Engine, run_id_a: int, run_id_b: int) -> pd.DataFrame:
    """Row-level status diff between two runs (e.g., the same file scored twice)."""

    frame_a = load_run_rows(engine, run_id_a).rename(columns={"score": "score_a", "is_outlier": "is_outlier_a"})
    frame_b = load_run_rows(engine, run_id_b).rename(columns={"score": "score_b", "is_outlier": "is_outlier_b"})
    merged = frame_a.drop(columns=["verdict"]).merge(
        frame_b.drop(columns=["verdict"]), on="row_index", how="outer"
    )
    merged["is_outlier_a"] = merged["is_outlier_a"].fillna(False).astype(bool)
    merged["is_outlier_b"] = merged["is_outlier_b"].fillna(False).astype(bool)
    merged["changed"] = merged["is_outlier_a"] != merged["is_outlier_b"]
    return merged.sort_values(["changed", "row_index"], ascending=[False, True]).reset_index(drop=True)


def purge_runs(engine: Engine, keep_latest: int = 20) -> int:
    """Delete all but the most recent runs (and their rows). Returns runs removed."""

    keep_latest = max(0, int(keep_latest))
    with engine.begin() as connection:
        keep_query = select(runs.c.id).order_by(runs.c.created_at.desc(), runs.c.id.desc()).limit(keep_latest)
        keep_ids = [row[0] for row in connection.execute(keep_query)] if keep_latest else []
        row_filter = ~outlier_rows.c.run_id.in_(keep_ids) if keep_ids else outlier_rows.c.run_id.is_not(None)
        run_filter = ~runs.c.id.in_(keep_ids) if keep_ids else runs.c.id.is_not(None)
        connection.execute(delete(outlier_rows).where(row_filter))
        result = connection.execute(delete(runs).where(run_filter))
    return result.rowcount
