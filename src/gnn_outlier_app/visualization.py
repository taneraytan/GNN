from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import express as px
from sklearn.decomposition import PCA


def score_distribution(scores: np.ndarray, threshold: float) -> go.Figure:
    fig = px.histogram(x=scores, nbins=40, labels={"x": "Outlier score", "y": "Rows"}, title="Outlier Score Distribution")
    fig.add_vline(x=threshold, line_dash="dash", line_color="#FF4B4B", annotation_text="Threshold")
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def embedding_scatter(features: np.ndarray, scores: np.ndarray, is_outlier: np.ndarray) -> go.Figure:
    if features.shape[1] >= 2:
        embedded = PCA(n_components=2, random_state=42).fit_transform(features)
    else:
        embedded = np.column_stack([features[:, 0], np.zeros(len(features))])
    plot_frame = pd.DataFrame(
        {
            "Component 1": embedded[:, 0],
            "Component 2": embedded[:, 1],
            "Outlier score": scores,
            "Decision": np.where(is_outlier, "Outlier", "Inlier"),
        }
    )
    fig = px.scatter(
        plot_frame,
        x="Component 1",
        y="Component 2",
        color="Decision",
        size="Outlier score",
        color_discrete_map={"Outlier": "#FF4B4B", "Inlier": "#41D3BD"},
        title="PCA Map of Tabular Graph Nodes",
        hover_data={"Outlier score": ":.4f"},
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def detector_heatmap(detector_scores: dict[str, np.ndarray]) -> go.Figure:
    """Heatmap of the detectors that actually ran (GNN architectures or fallback detectors)."""

    names = list(detector_scores)
    matrix = np.vstack([detector_scores[name] for name in names]) if names else np.empty((0, 0))
    fig = px.imshow(
        matrix,
        labels=dict(x="Row index", y="Detector", color="Score"),
        y=names,
        color_continuous_scale="Inferno",
        title="Detector Agreement Heatmap",
        aspect="auto",
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def graph_preview(edge_index: np.ndarray, scores: np.ndarray, max_nodes: int = 150, max_edges: int = 2000, seed: int = 42) -> go.Figure:
    """Spring-layout preview of a score-stratified sample (top outliers always included)."""

    scores = np.asarray(scores, dtype=float)
    total = len(scores)
    if total == 0:
        return go.Figure()

    if total <= max_nodes:
        selected = np.arange(total)
    else:
        top = np.argsort(-scores)[: max_nodes // 2]
        rest = np.setdiff1d(np.arange(total), top)
        rng = np.random.default_rng(seed)
        fill = rng.choice(rest, size=max_nodes - len(top), replace=False)
        selected = np.sort(np.concatenate([top, fill]))

    position_of = {int(node): position for position, node in enumerate(selected)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(selected)))
    if edge_index.size:
        selected_set = np.isin(edge_index[0], selected) & np.isin(edge_index[1], selected)
        sampled = edge_index[:, selected_set][:, :max_edges]
        graph.add_edges_from(
            (position_of[int(source)], position_of[int(target)])
            for source, target in sampled.T
            if source != target
        )

    layout = nx.spring_layout(graph, seed=seed)
    node_x = np.array([layout[node][0] for node in range(len(selected))])
    node_y = np.array([layout[node][1] for node in range(len(selected))])

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
        edge_x += [node_x[source], node_x[target], None]
        edge_y += [node_y[source], node_y[target], None]

    fig = go.Figure()
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="rgba(140,150,180,0.18)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    node_scores = scores[selected]
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=10 + 18 * node_scores, color=node_scores, colorscale="Turbo", showscale=True),
            text=[f"Row {int(node)}<br>Score {scores[int(node)]:.4f}" for node in selected],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"kNN Graph Preview ({len(selected)} of {total} rows, spring layout)",
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
