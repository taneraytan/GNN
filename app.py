from __future__ import annotations

import hashlib
import html
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.gnn_outlier_app.data import FeaturePreparation, LoadedTable, load_table, prepare_features
from src.gnn_outlier_app.database import (
    PERSIST_ALL,
    PERSIST_OUTLIERS,
    VERDICT_UNREVIEWED,
    VERDICTS,
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
from src.gnn_outlier_app.explain import top_feature_contributions
from src.gnn_outlier_app.export import attach_results, to_csv_bytes, to_excel_bytes, to_parquet_bytes
from src.gnn_outlier_app.graph import build_knn_graph
from src.gnn_outlier_app.models import ARCHITECTURE_DESCRIPTIONS, MIN_ROWS, run_detection
from src.gnn_outlier_app.visualization import detector_heatmap, embedding_scatter, graph_preview, score_distribution

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


@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_and_prepare(file_bytes: bytes, source_name: str) -> tuple[LoadedTable, FeaturePreparation]:
    loaded = load_table(BytesIO(file_bytes), source_name)
    return loaded, prepare_features(loaded.frame)


with st.sidebar:
    st.header("Data & Model")
    upload = st.file_uploader("Upload .txt, .csv, .xls, .xlsx, or .parquet", type=["txt", "csv", "xls", "xlsx", "parquet"])
    workspace = st.text_input(
        "Local workspace",
        value="outlier_runs",
        help="Run history is stored in a SQLite file inside the local data/ folder. "
        "Names may only use letters, numbers, dots, dashes, and underscores.",
    )
    try:
        db_path = resolve_database_path(workspace)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    k = st.slider("kNN graph neighbors", min_value=2, max_value=50, value=8)
    contamination = st.slider("Expected outlier rate", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
    threshold_choice = st.selectbox(
        "Threshold strategy",
        options=["Contamination quantile", "Robust MAD"],
        help="Contamination quantile always flags roughly the expected rate, even on clean data. "
        "Robust MAD flags only rows whose score is far above the typical score and may flag none.",
    )
    threshold_strategy = "contamination" if threshold_choice == "Contamination quantile" else "mad"
    if threshold_strategy == "contamination":
        st.caption("Heads up: this strategy flags the top ~{:.0%} of rows on any dataset, including clean ones.".format(contamination))
    epochs = st.slider("Training epochs", min_value=10, max_value=300, value=80, step=10)
    hidden_dim = st.select_slider("Hidden dimension", options=[16, 32, 64, 128], value=32)
    selected_architectures = st.multiselect(
        "GNN architectures",
        options=list(ARCHITECTURE_DESCRIPTIONS),
        default=list(ARCHITECTURE_DESCRIPTIONS),
    )
    st.divider()
    st.header("Privacy & Storage")
    persist_all = st.toggle(
        "Persist all scored rows locally",
        value=False,
        help="Off (default): only flagged outlier rows are stored in the workspace database. "
        "On: every scored row is stored. Data never leaves this machine either way.",
    )
    with st.expander("Workspace maintenance"):
        keep_latest = st.number_input("Runs to keep", min_value=0, max_value=500, value=20, step=1)
        if st.button("Delete older runs"):
            removed = purge_runs(get_engine(db_path), keep_latest=int(keep_latest))
            st.success(f"Removed {removed} old runs from this workspace.")
    st.caption(
        "Security note: run this app on localhost only. If you must share it, put it behind an "
        "authenticated reverse proxy and cap uploads via server.maxUploadSize. See the README."
    )

st.subheader("Architecture Coverage")
columns = st.columns(3)
for index, (name, description) in enumerate(ARCHITECTURE_DESCRIPTIONS.items()):
    with columns[index % 3]:
        st.markdown(
            f"<div class='metric-card'><b>{html.escape(name)}</b><br><small>{html.escape(description)}</small></div>",
            unsafe_allow_html=True,
        )

if upload is None:
    st.info("Upload a supported tabular file to build a row graph, train the GNN ensemble, visualize outliers, and export results.")
    engine = get_engine(db_path)
    history = recent_runs(engine)
    if not history.empty:
        st.subheader("Recent Local Runs")
        st.dataframe(history, use_container_width=True)
    st.stop()

file_bytes = upload.getvalue()
loaded, prepared = load_and_prepare(file_bytes, upload.name)
frame = loaded.frame

st.subheader("Dataset Preview")
summary_cols = st.columns(4)
summary_cols[0].metric("Rows", f"{len(frame):,}")
summary_cols[1].metric("Columns", f"{len(frame.columns):,}")
summary_cols[2].metric("Auto Features", f"{len(prepared.feature_columns):,}")
summary_cols[3].metric("Source", loaded.source_name)
st.dataframe(frame.head(100), use_container_width=True)

st.subheader("Automatic Preprocessing & Feature Selection")
if prepared.notes:
    for note in prepared.notes:
        st.caption(note)

prep_metrics = st.columns(3)
prep_metrics[0].metric("Source Columns Used", f"{len(prepared.source_columns):,}")
prep_metrics[1].metric("Model Features", f"{len(prepared.feature_columns):,}")
prep_metrics[2].metric("Columns Dropped", f"{len(prepared.dropped_columns):,}")

with st.expander("Review automatically selected features", expanded=False):
    st.write("Feature selection is automatic: numeric, numeric-looking text, datetimes, and low-cardinality categories are cleaned, encoded, imputed, de-duplicated, and filtered before graph construction.")
    st.dataframe(prepared.frame.head(100), use_container_width=True)
    if prepared.dropped_columns:
        dropped = pd.DataFrame(
            {"column": list(prepared.dropped_columns), "reason": list(prepared.dropped_columns.values())}
        )
        st.dataframe(dropped, use_container_width=True)

if not prepared.feature_columns:
    st.error("Automatic preprocessing could not find usable model features in this dataset.")
    st.stop()

if len(frame) < MIN_ROWS:
    st.error(f"At least {MIN_ROWS} rows are required for outlier detection; this file has {len(frame)}.")
    st.stop()

if not selected_architectures:
    st.warning("Select at least one GNN architecture in the sidebar to run detection.")
    st.stop()

run_signature = hashlib.sha256(
    repr(
        (
            hashlib.sha256(file_bytes).hexdigest(),
            upload.name,
            k,
            contamination,
            epochs,
            hidden_dim,
            tuple(sorted(selected_architectures)),
            threshold_strategy,
        )
    ).encode("utf-8")
).hexdigest()

run_button = st.button("Preprocess, Train GNN Ensemble & Identify Outliers", type="primary")

if run_button:
    progress = st.progress(0.0, text="Building graph...")

    def on_progress(label: str, completed: int, total: int) -> None:
        progress.progress(completed / total, text=f"Trained {label} ({completed}/{total})")

    graph = build_knn_graph(prepared.frame, prepared.feature_columns, k=k)
    result = run_detection(
        graph.features,
        graph.edge_index,
        selected_architectures,
        contamination=contamination,
        epochs=epochs,
        hidden_dim=hidden_dim,
        threshold_strategy=threshold_strategy,
        progress_callback=on_progress,
    )
    progress.empty()
    explanations = top_feature_contributions(result.feature_errors, prepared.feature_columns)
    scores = pd.Series(result.scores, name="outlier_score")
    flags = pd.Series(result.is_outlier, name="is_outlier")
    output = attach_results(frame, scores, flags, explanations)
    engine = get_engine(db_path)
    run_id = save_run(
        engine,
        loaded.source_name,
        frame,
        scores,
        flags,
        selected_architectures,
        result.used_backend,
        result.threshold,
        feature_count=len(prepared.feature_columns),
        params={
            "k": k,
            "contamination": contamination,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "seed": 42,
            "threshold_strategy": threshold_strategy,
        },
        persist_mode=PERSIST_ALL if persist_all else PERSIST_OUTLIERS,
    )
    st.session_state["last_run"] = {
        "signature": run_signature,
        "output": output,
        "result": result,
        "graph": graph,
        "feature_count": len(prepared.feature_columns),
        "run_id": run_id,
        "source_name": loaded.source_name,
    }

last_run = st.session_state.get("last_run")
if last_run is not None and last_run["signature"] != run_signature:
    st.info("The uploaded file or settings changed since the last run. Run detection again to refresh the results below.")
elif last_run is not None:
    output = last_run["output"]
    result = last_run["result"]
    graph = last_run["graph"]
    run_id = last_run["run_id"]

    st.success(f"Run #{run_id} completed with backend: {result.used_backend}.")
    if "fallback" in result.used_backend:
        st.warning(
            "torch-geometric is not installed, so honest non-GNN detectors (IsolationForest, LocalOutlierFactor) "
            "were used. Install the GNN backend with: pip install -r requirements-gnn.txt"
        )
    metrics = st.columns(4)
    metrics[0].metric("Outliers", int(result.is_outlier.sum()))
    metrics[1].metric("Threshold", f"{result.threshold:.4f}")
    metrics[2].metric("Edges", f"{graph.edge_index.shape[1]:,}")
    metrics[3].metric("Auto Features", last_run["feature_count"])
    if result.threshold_strategy == "contamination":
        st.caption(
            "Contamination-quantile thresholds always flag roughly the expected rate, even on clean data. "
            "Switch to Robust MAD in the sidebar for a score-driven cutoff."
        )

    tab_scores, tab_map, tab_graph, tab_table, tab_export = st.tabs(["Scores", "Embedding", "Graph", "Rows", "Export"])
    with tab_scores:
        st.plotly_chart(score_distribution(result.scores, result.threshold), use_container_width=True)
        st.plotly_chart(detector_heatmap(result.detector_scores), use_container_width=True)
    with tab_map:
        st.plotly_chart(embedding_scatter(graph.features, result.scores, result.is_outlier), use_container_width=True)
    with tab_graph:
        st.plotly_chart(graph_preview(graph.edge_index, result.scores), use_container_width=True)
    with tab_table:
        st.caption("The top_outlier_features column shows which features drive each row's anomaly score.")
        st.dataframe(output, use_container_width=True)

        flagged = output[output["is_outlier"]]
        if not flagged.empty:
            st.markdown("**Review flagged rows** — verdicts are saved to the local workspace and feed the precision summary below.")
            engine = get_engine(db_path)
            stored = load_run_rows(engine, run_id).set_index("row_index")
            review = pd.DataFrame(
                {
                    "row_index": flagged.index.astype(int),
                    "outlier_score": flagged["outlier_score"].to_numpy(),
                    "top_outlier_features": flagged["top_outlier_features"].to_numpy(),
                }
            )
            review["verdict"] = [
                stored["verdict"].get(row, None) or VERDICT_UNREVIEWED for row in review["row_index"]
            ]
            edited = st.data_editor(
                review,
                column_config={
                    "verdict": st.column_config.SelectboxColumn("verdict", options=list(VERDICTS), required=True),
                },
                disabled=["row_index", "outlier_score", "top_outlier_features"],
                hide_index=True,
                use_container_width=True,
                key=f"verdicts_{run_id}",
            )
            if st.button("Save verdicts"):
                verdicts = {int(row.row_index): row.verdict for row in edited.itertuples()}
                updated = save_verdicts(engine, run_id, verdicts)
                st.success(f"Saved verdicts for {updated} rows.")
    with tab_export:
        outliers_only = output[output["is_outlier"]]
        export_choice = st.radio("Rows to export", ["All rows with scores", "Outliers only"], horizontal=True)
        export_frame = output if export_choice == "All rows with scores" else outliers_only
        base = Path(last_run["source_name"]).stem or "outliers"
        st.caption("CSV and Excel exports neutralize spreadsheet formula injection by escaping cells that start with =, +, -, or @.")
        st.download_button("Download CSV", data=to_csv_bytes(export_frame), file_name=f"{base}_outliers.csv", mime="text/csv")
        st.download_button("Download Excel", data=to_excel_bytes(export_frame), file_name=f"{base}_outliers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Parquet", data=to_parquet_bytes(export_frame), file_name=f"{base}_outliers.parquet", mime="application/octet-stream")

st.subheader("Run History, Feedback & Comparison")
engine = get_engine(db_path)
history = recent_runs(engine, limit=20)
st.dataframe(history, use_container_width=True)

feedback = feedback_summary(engine)
if not feedback.empty:
    st.markdown("**Analyst feedback precision** (confirmed / reviewed) per run:")
    st.dataframe(feedback, use_container_width=True)

if len(history) >= 2:
    with st.expander("Compare two runs (row-level status changes)"):
        run_options = history["id"].tolist()
        compare_cols = st.columns(2)
        run_a = compare_cols[0].selectbox("Run A", options=run_options, index=1)
        run_b = compare_cols[1].selectbox("Run B", options=run_options, index=0)
        if run_a != run_b:
            diff = compare_runs(engine, int(run_a), int(run_b))
            changed = int(diff["changed"].sum())
            st.metric("Rows that changed outlier status", changed)
            st.dataframe(diff[diff["changed"]] if changed else diff.head(20), use_container_width=True)
        else:
            st.caption("Pick two different runs to compare.")
