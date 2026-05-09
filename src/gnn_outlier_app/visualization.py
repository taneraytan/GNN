from __future__ import annotations

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


def architecture_heatmap(architecture_scores: dict[str, np.ndarray]) -> go.Figure:
    names = list(architecture_scores)
    matrix = np.vstack([architecture_scores[name] for name in names]) if names else np.empty((0, 0))
    fig = px.imshow(
        matrix,
        labels=dict(x="Row index", y="Architecture", color="Score"),
        y=names,
        color_continuous_scale="Inferno",
        title="Architecture Agreement Heatmap",
        aspect="auto",
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def graph_preview(edge_index: np.ndarray, scores: np.ndarray, max_nodes: int = 150) -> go.Figure:
    node_count = min(len(scores), max_nodes)
    if node_count == 0:
        return go.Figure()
    theta = np.linspace(0, 2 * np.pi, node_count, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    fig = go.Figure()
    if edge_index.size:
        sampled = edge_index[:, (edge_index[0] < node_count) & (edge_index[1] < node_count)]
        for source, target in sampled.T[:500]:
            fig.add_trace(
                go.Scatter(
                    x=[x[source], x[target]],
                    y=[y[source], y[target]],
                    mode="lines",
                    line=dict(color="rgba(140,150,180,0.18)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=10 + 18 * scores[:node_count], color=scores[:node_count], colorscale="Turbo", showscale=True),
            text=[f"Row {i}<br>Score {scores[i]:.4f}" for i in range(node_count)],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="kNN Graph Preview",
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
