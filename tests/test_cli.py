from __future__ import annotations

import numpy as np
import pandas as pd

from src.gnn_outlier_app.cli import main
from src.gnn_outlier_app.database import get_engine, load_run_rows, recent_runs


def _write_sample_csv(path) -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"x": rng.normal(0, 1, 30), "y": rng.normal(0, 1, 30)})
    frame.loc[29] = [40.0, 40.0]
    frame.to_csv(path, index=False)


def test_cli_scores_file_and_writes_parquet(tmp_path, capsys) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "scored.parquet"
    _write_sample_csv(source)

    exit_code = main(["score", str(source), "--out", str(output), "--epochs", "1", "--contamination", "0.1"])

    assert exit_code == 0
    assert output.exists()
    scored = pd.read_parquet(output)
    assert {"outlier_score", "is_outlier", "top_outlier_features"} <= set(scored.columns)
    assert "backend:" in capsys.readouterr().out


def test_cli_outliers_only_and_persistence(tmp_path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "outliers.csv"
    db = tmp_path / "cli_runs.sqlite"
    _write_sample_csv(source)

    exit_code = main(
        [
            "score",
            str(source),
            "--out",
            str(output),
            "--epochs",
            "1",
            "--outliers-only",
            "--db",
            str(db),
        ]
    )

    assert exit_code == 0
    scored = pd.read_csv(output)
    assert scored["is_outlier"].all()

    engine = get_engine(db)
    history = recent_runs(engine)
    assert len(history) == 1
    assert not load_run_rows(engine, int(history.loc[0, "id"])).empty


def test_cli_rejects_unsupported_output(tmp_path, capsys) -> None:
    source = tmp_path / "input.csv"
    _write_sample_csv(source)
    exit_code = main(["score", str(source), "--out", str(tmp_path / "scored.json")])
    assert exit_code == 2
    assert "unsupported output format" in capsys.readouterr().err


def test_cli_rejects_tiny_dataset(tmp_path, capsys) -> None:
    source = tmp_path / "tiny.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).to_csv(source, index=False)
    exit_code = main(["score", str(source), "--out", str(tmp_path / "scored.csv")])
    assert exit_code == 2
    assert "error:" in capsys.readouterr().err
