from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.gnn_outlier_app.data import load_table, numeric_columns
from src.gnn_outlier_app.database import get_engine, recent_runs, save_run
from src.gnn_outlier_app.export import attach_results, to_csv_bytes, to_excel_bytes, to_parquet_bytes
from src.gnn_outlier_app.graph import build_knn_graph
from src.gnn_outlier_app.models import ARCHITECTURE_DESCRIPTIONS, run_detection
from src.gnn_outlier_app.visualization import architecture_heatmap, embedding_scatter, graph_preview, score_distribution

st.set_page_config(page_title="GraphGuard GNN Outlier Studio", page_icon="🕸️", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 1.25rem 1.5rem; border-radius: 1.25rem; background: linear-gradient(120deg, #221C45, #102A43 55%, #102D2D); border: 1px solid rgba(255,255,255,.12);} 
    .metric-card {padding: 1rem; border-radius: 1rem; background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08);} 
    </style>
    <div class="hero">
      <h1>🕸️ GraphGuard GNN Outlier Studio</h1>
      <p>Local-first anomaly detection for tabular files using an ensemble of major Graph Neural Network architectures.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data & Model")
    upload = st.file_uploader("Upload .txt, .csv, .xls, .xlsx, or .parquet", type=["txt", "csv", "xls", "xlsx", "parquet"])
    db_path = st.text_input("SQLite database", value="data/outlier_runs.sqlite")
    k = st.slider("kNN graph neighbors", min_value=2, max_value=50, value=8)
    contamination = st.slider("Expected outlier rate", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
    epochs = st.slider("Training epochs", min_value=10, max_value=300, value=80, step=10)
    hidden_dim = st.select_slider("Hidden dimension", options=[16, 32, 64, 128], value=32)
    selected_architectures = st.multiselect(
        "GNN architectures",
        options=list(ARCHITECTURE_DESCRIPTIONS),
        default=list(ARCHITECTURE_DESCRIPTIONS),
    )

st.subheader("Architecture Coverage")
columns = st.columns(3)
for index, (name, description) in enumerate(ARCHITECTURE_DESCRIPTIONS.items()):
    with columns[index % 3]:
        st.markdown(f"<div class='metric-card'><b>{name}</b><br><small>{description}</small></div>", unsafe_allow_html=True)

if upload is None:
    st.info("Upload a supported tabular file to build a row graph, train the GNN ensemble, visualize outliers, and export results.")
    engine = get_engine(db_path)
    history = recent_runs(engine)
    if not history.empty:
        st.subheader("Recent Local Runs")
        st.dataframe(history, use_container_width=True)
    st.stop()

loaded = load_table(upload, upload.name)
frame = loaded.frame
num_cols = numeric_columns(frame)

st.subheader("Dataset Preview")
summary_cols = st.columns(4)
summary_cols[0].metric("Rows", f"{len(frame):,}")
summary_cols[1].metric("Columns", f"{len(frame.columns):,}")
summary_cols[2].metric("Numeric Features", f"{len(num_cols):,}")
summary_cols[3].metric("Source", loaded.source_name)
st.dataframe(frame.head(100), use_container_width=True)

if not num_cols:
    st.error("No numeric columns were found. Add numeric features before running graph outlier detection.")
    st.stop()

feature_columns = st.multiselect("Feature columns", options=num_cols, default=num_cols)
run_button = st.button("Train GNN Ensemble & Identify Outliers", type="primary")

if run_button:
    with st.spinner("Building graph and training selected architectures locally..."):
        graph = build_knn_graph(frame, feature_columns, k=k)
        result = run_detection(
            graph.features,
            graph.edge_index,
            selected_architectures,
            contamination=contamination,
            epochs=epochs,
            hidden_dim=hidden_dim,
        )
        scores = pd.Series(result.scores, name="outlier_score")
        flags = pd.Series(result.is_outlier, name="is_outlier")
        output = attach_results(frame, scores, flags)
        engine = get_engine(db_path)
        run_id = save_run(engine, loaded.source_name, frame, scores, flags, selected_architectures, result.used_backend, result.threshold)
        st.session_state["last_output"] = output
        st.session_state["last_result"] = result
        st.session_state["last_graph"] = graph
        st.session_state["last_run_id"] = run_id

if "last_output" in st.session_state:
    output = st.session_state["last_output"]
    result = st.session_state["last_result"]
    graph = st.session_state["last_graph"]
    run_id = st.session_state["last_run_id"]

    st.success(f"Run #{run_id} completed with backend: {result.used_backend}.")
    metrics = st.columns(4)
    metrics[0].metric("Outliers", int(result.is_outlier.sum()))
    metrics[1].metric("Threshold", f"{result.threshold:.4f}")
    metrics[2].metric("Edges", f"{graph.edge_index.shape[1]:,}")
    metrics[3].metric("Architectures", len(result.architecture_scores))

    tab_scores, tab_map, tab_graph, tab_table, tab_export = st.tabs(["Scores", "Embedding", "Graph", "Rows", "Export"])
    with tab_scores:
        st.plotly_chart(score_distribution(result.scores, result.threshold), use_container_width=True)
        st.plotly_chart(architecture_heatmap(result.architecture_scores), use_container_width=True)
    with tab_map:
        st.plotly_chart(embedding_scatter(graph.features, result.scores, result.is_outlier), use_container_width=True)
    with tab_graph:
        st.plotly_chart(graph_preview(graph.edge_index, result.scores), use_container_width=True)
    with tab_table:
        st.dataframe(output, use_container_width=True)
    with tab_export:
        outliers_only = output[output["is_outlier"]]
        export_choice = st.radio("Rows to export", ["All rows with scores", "Outliers only"], horizontal=True)
        export_frame = output if export_choice == "All rows with scores" else outliers_only
        base = Path(loaded.source_name).stem or "outliers"
        st.download_button("Download CSV", data=to_csv_bytes(export_frame), file_name=f"{base}_outliers.csv", mime="text/csv")
        st.download_button("Download Excel", data=to_excel_bytes(export_frame), file_name=f"{base}_outliers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Parquet", data=to_parquet_bytes(export_frame), file_name=f"{base}_outliers.parquet", mime="application/octet-stream")

st.subheader("Recent Local Runs")
st.dataframe(recent_runs(get_engine(db_path)), use_container_width=True)
