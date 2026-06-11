# GraphGuard GNN Outlier Studio

GraphGuard is a professional, local-first Streamlit application (plus a headless CLI) for finding anomalous rows in tabular data with graph neural network autoencoders. It accepts `.txt`, `.csv`, `.xls`, `.xlsx`, and `.parquet` files, builds a k-nearest-neighbor graph where each row is a node, trains an ensemble of major GNN architectures, explains and visualizes the results, and exports the identified outliers.

## Highlights

- **Local data handling:** uploaded data stays in the local Streamlit process. By default only *flagged outlier rows* are persisted to the local SQLite workspace; storing all rows is an explicit opt-in.
- **Broad file support:** text, CSV, Excel, and Parquet ingestion, with cached loading and preprocessing so widget interactions don't re-parse the file.
- **Automatic preprocessing:** numeric-looking text, datetimes, and low-cardinality categories are converted into model-ready features; duplicate rows and identifier-like columns are reported.
- **Major GNN coverage:** GCN, GraphSAGE, GAT, GIN, ChebNet, and APPNP autoencoder variants, with GPU (CUDA/MPS) support, early stopping, and live training progress.
- **Explainable scores:** every row gets a `top_outlier_features` summary showing which features drive its anomaly score.
- **Two threshold strategies:** the classic contamination quantile (always flags ~the expected rate, even on clean data — the UI says so) or a robust MAD cutoff that can flag zero rows on clean data.
- **Rank-based ensembling:** detector scores are combined by average rank, so one extreme score cannot compress the rest.
- **Human-in-the-loop feedback:** mark flagged rows as Confirmed or False positive; verdicts persist per run and roll up into a precision summary.
- **Run history & comparison:** every run stores its full hyperparameters; compare any two runs row-by-row to see status changes (e.g., drift between monthly files).
- **Safe exports:** CSV/Excel downloads neutralize spreadsheet formula injection (`=`, `+`, `-`, `@` prefixes are escaped).
- **Honest backend behavior:** uses `torch-geometric` when available; otherwise a clearly labeled scikit-learn fallback (IsolationForest + LocalOutlierFactor) runs under its real names — never disguised as GNN output.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # lightweight core (scikit-learn fallback)
pip install -r requirements-gnn.txt      # optional: torch + torch-geometric (~2 GB)
streamlit run app.py
```

Dependencies in `requirements.txt` are pinned for reproducible, auditable installs. The GNN backend is optional by design; see the comments in `requirements-gnn.txt` for CPU-only torch installs.

## Headless CLI

The same pipeline runs without a browser (run from the repository root, or install the package to get the `gnn-outlier-studio` command):

```bash
python -m src.gnn_outlier_app score input.csv --out scored.parquet
python -m src.gnn_outlier_app score input.csv --out outliers.csv \
    --outliers-only --threshold-strategy mad --k 8 --epochs 80 \
    --db data/cron_runs.sqlite
```

Supported output formats: `.csv`, `.xlsx`, `.parquet`. Use `--help` for all options.

## How it works

1. Upload a supported tabular file (at least 5 rows).
2. The app automatically preprocesses and selects usable model features by coercing numeric-looking text, expanding datetime and categorical signals, imputing missing values, removing constants and identifier-like fields, and capping engineered features by min-max-scaled dispersion.
3. The app standardizes the prepared features and constructs a symmetric kNN row graph (approximate nearest neighbors via `pynndescent` are used automatically for ≥10k rows when installed).
4. Selected GNN autoencoders reconstruct node features (or the labeled scikit-learn fallback runs).
5. Per-detector scores are rank-normalized and ensembled by average rank.
6. The chosen threshold strategy flags outliers; per-row feature contributions explain why.
7. Results are visualized, exportable, persisted, and reviewable.

## Local workspace database

Run history lives in a SQLite file inside the local `data/` folder. The sidebar takes a *workspace name* (letters, numbers, dots, dashes, underscores), never a raw path — names that would escape `data/` are rejected. The database contains:

- `runs`: source file, row/feature counts, architecture choices, backend, threshold, and the full hyperparameters (`k`, `contamination`, `epochs`, `hidden_dim`, `seed`, `threshold_strategy`, persistence mode) so every run is reproducible.
- `outlier_rows`: row index, score, decision, analyst verdict, and a JSON payload — flagged rows only unless "Persist all scored rows" is enabled.

Workspace maintenance in the sidebar deletes old runs (keep-latest retention).

## Security notes

- **Run locally.** The app has no authentication. Keep Streamlit bound to localhost (the default); if you must share it, put it behind an authenticated reverse proxy.
- **Cap uploads** with `server.maxUploadSize` in `.streamlit/config.toml` if untrusted users can reach the app.
- **Database paths are sandboxed** to `data/` and the SQLAlchemy URL is built with `URL.create`, so UI input cannot create files elsewhere or smuggle URL parameters.
- **Exports are escaped** against CSV/Excel formula injection. Parquet preserves raw values (it is not opened as a spreadsheet).
- **SQL is parameterized** throughout via SQLAlchemy Core.
- **Data retention is opt-in:** only flagged rows are stored by default, and the retention tool removes old runs.

## Development

```bash
pytest -q          # 49 tests: pipeline, security, exports, CLI, database, explanations
```

CI (GitHub Actions) runs the suite twice: once on the pinned core dependencies (exercising the scikit-learn fallback) and once with the torch-geometric backend installed.
