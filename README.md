# GraphGuard GNN Outlier Studio

GraphGuard is a professional, local-first Streamlit application for finding anomalous rows in tabular data with graph neural network autoencoders. It accepts `.txt`, `.csv`, `.xls`, `.xlsx`, and `.parquet` files, builds a k-nearest-neighbor graph where each row is a node, trains an ensemble of major GNN architectures, visualizes model agreement, and exports the identified outliers.

## Highlights

- **Local data handling:** uploaded data stays in the local Streamlit process.
- **Broad file support:** text, CSV, Excel, and Parquet ingestion.
- **Major GNN coverage:** GCN, GraphSAGE, GAT, GIN, ChebNet, and APPNP autoencoder variants.
- **Professional visuals:** KPI cards, PCA embedding, score distribution, architecture agreement heatmap, and graph preview.
- **Export options:** save all scored rows or outliers only as CSV, Excel, or Parquet.
- **SQLite persistence:** every run and row-level decision is stored in a configurable local database.
- **Graceful backend behavior:** uses `torch-geometric` when available and a deterministic scikit-learn anomaly fallback for lightweight local environments.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. Upload a supported tabular file.
2. Select numeric columns to model.
3. The app standardizes features and constructs a symmetric kNN row graph.
4. Selected GNN autoencoders reconstruct node features.
5. Reconstruction error is normalized per architecture and ensembled into an outlier score.
6. Scores above the selected contamination quantile are flagged as outliers.
7. Results are visualized, exportable, and persisted to SQLite.

## Local database

The sidebar database path defaults to `data/outlier_runs.sqlite`. The database contains:

- `runs`: source file, architecture choices, threshold, backend, and run metadata.
- `outlier_rows`: row index, score, decision, and a JSON payload of the scored row.
