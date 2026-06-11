from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.gnn_outlier_app.database import (
    PERSIST_ALL,
    PERSIST_OUTLIERS,
    VERDICT_CONFIRMED,
    VERDICT_FALSE_POSITIVE,
    VERDICT_UNREVIEWED,
    compare_runs,
    feedback_summary,
    get_engine,
    load_run_rows,
    purge_runs,
    recent_runs,
    resolve_database_path,
    save_run,
    save_verdicts,
)


def _save_sample_run(engine, flags: list[bool], persist_mode: str = PERSIST_OUTLIERS) -> int:
    frame = pd.DataFrame({"x": range(len(flags))})
    scores = pd.Series(np.linspace(0, 1, len(flags)))
    return save_run(
        engine,
        "sample.csv",
        frame,
        scores,
        pd.Series(flags),
        ["GCN"],
        "test-backend",
        0.9,
        feature_count=1,
        params={"k": 8, "contamination": 0.05, "epochs": 10, "hidden_dim": 32, "seed": 42, "threshold_strategy": "mad"},
        persist_mode=persist_mode,
    )


def test_resolve_database_path_accepts_safe_names(tmp_path) -> None:
    path = resolve_database_path("my_runs.v2", data_root=tmp_path)
    assert path == (tmp_path / "my_runs.v2.sqlite").resolve()
    assert resolve_database_path("runs.sqlite", data_root=tmp_path).name == "runs.sqlite"


@pytest.mark.parametrize(
    "bad_name",
    ["", "   ", "../escape", "..", "nested/dir", "nested\\dir", "/etc/passwd", "~/.ssh/keys", ".hidden", "a?b"],
)
def test_resolve_database_path_rejects_unsafe_names(tmp_path, bad_name) -> None:
    with pytest.raises(ValueError):
        resolve_database_path(bad_name, data_root=tmp_path)


def test_save_run_outliers_only_by_default(tmp_path) -> None:
    engine = get_engine(tmp_path / "runs.sqlite")
    run_id = _save_sample_run(engine, [False, False, True, True])
    rows = load_run_rows(engine, run_id)
    assert len(rows) == 2
    assert rows["is_outlier"].all()

    history = recent_runs(engine)
    assert history.loc[0, "persisted_rows"] == PERSIST_OUTLIERS
    assert history.loc[0, "threshold_strategy"] == "mad"
    assert history.loc[0, "contamination"] == pytest.approx(0.05)


def test_save_run_persist_all_keeps_every_row(tmp_path) -> None:
    engine = get_engine(tmp_path / "runs.sqlite")
    run_id = _save_sample_run(engine, [False, True, False], persist_mode=PERSIST_ALL)
    rows = load_run_rows(engine, run_id)
    assert len(rows) == 3


def test_save_run_rejects_unknown_persist_mode(tmp_path) -> None:
    engine = get_engine(tmp_path / "runs.sqlite")
    with pytest.raises(ValueError):
        _save_sample_run(engine, [True], persist_mode="everything")


def test_verdict_roundtrip_and_feedback_summary(tmp_path) -> None:
    engine = get_engine(tmp_path / "runs.sqlite")
    run_id = _save_sample_run(engine, [True, True, True])

    updated = save_verdicts(engine, run_id, {0: VERDICT_CONFIRMED, 1: VERDICT_FALSE_POSITIVE, 2: VERDICT_UNREVIEWED})
    assert updated == 3

    rows = load_run_rows(engine, run_id).set_index("row_index")
    assert rows.loc[0, "verdict"] == VERDICT_CONFIRMED
    assert rows.loc[1, "verdict"] == VERDICT_FALSE_POSITIVE
    assert pd.isna(rows.loc[2, "verdict"])

    summary = feedback_summary(engine)
    assert summary.loc[0, "confirmed"] == 1
    assert summary.loc[0, "false_positive"] == 1
    assert summary.loc[0, "precision"] == pytest.approx(0.5)

    with pytest.raises(ValueError):
        save_verdicts(engine, run_id, {0: "Maybe"})


def test_compare_runs_reports_status_changes(tmp_path) -> None:
    engine = get_engine(tmp_path / "runs.sqlite")
    run_a = _save_sample_run(engine, [True, True, False], persist_mode=PERSIST_ALL)
    run_b = _save_sample_run(engine, [True, False, True], persist_mode=PERSIST_ALL)

    diff = compare_runs(engine, run_a, run_b)
    changed = diff[diff["changed"]]["row_index"].tolist()
    assert sorted(changed) == [1, 2]


def test_purge_runs_keeps_latest(tmp_path) -> None:
    engine = get_engine(tmp_path / "runs.sqlite")
    ids = [_save_sample_run(engine, [True]) for _ in range(5)]

    removed = purge_runs(engine, keep_latest=2)
    assert removed == 3
    remaining = recent_runs(engine, limit=10)["id"].tolist()
    assert sorted(remaining) == sorted(ids[-2:])
    # Orphaned row payloads are removed too.
    assert load_run_rows(engine, ids[0]).empty


def test_get_engine_migrates_missing_columns(tmp_path) -> None:
    db_file = tmp_path / "legacy.sqlite"
    legacy_engine = get_engine(db_file)
    with legacy_engine.begin() as connection:
        connection.execute(text("ALTER TABLE runs DROP COLUMN threshold_strategy"))
        connection.execute(text("ALTER TABLE outlier_rows DROP COLUMN verdict"))
    legacy_engine.dispose()

    migrated = get_engine(db_file)
    run_id = _save_sample_run(migrated, [True])
    assert recent_runs(migrated).loc[0, "id"] == run_id
