from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

APPROX_NN_THRESHOLD = 10_000


@dataclass(frozen=True)
class GraphData:
    """Feature matrix plus edge list used by GNN models."""

    features: np.ndarray
    edge_index: np.ndarray
    scaled_frame: pd.DataFrame


def _knn_indices(features: np.ndarray, neighbors: int) -> np.ndarray:
    """Return (n, neighbors + 1) neighbor indices, approximate for large tables."""

    if len(features) >= APPROX_NN_THRESHOLD:
        try:
            from pynndescent import NNDescent
        except ImportError:
            pass
        else:
            index = NNDescent(features, n_neighbors=neighbors + 1, random_state=42)
            indices, _ = index.neighbor_graph
            return np.asarray(indices)
    model = NearestNeighbors(n_neighbors=neighbors + 1)
    model.fit(features)
    return model.kneighbors(features, return_distance=False)


def build_knn_graph(frame: pd.DataFrame, feature_columns: list[str], k: int = 8) -> GraphData:
    """Convert tabular rows into a symmetric k-nearest-neighbor graph."""

    if not feature_columns:
        raise ValueError("At least one numeric feature column is required.")

    clean = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    clean = clean.fillna(clean.median(numeric_only=True)).fillna(0)
    scaler = StandardScaler()
    features = scaler.fit_transform(clean).astype(np.float32)

    if len(features) <= 1:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        neighbors = max(1, min(k, len(features) - 1))
        indices = _knn_indices(features, neighbors)
        sources = np.repeat(np.arange(len(features), dtype=np.int64), indices.shape[1] - 1)
        targets = indices[:, 1:].astype(np.int64).ravel()
        mask = sources != targets
        sources, targets = sources[mask], targets[mask]
        if sources.size:
            forward = np.stack([sources, targets])
            backward = np.stack([targets, sources])
            edge_index = np.unique(np.concatenate([forward, backward], axis=1), axis=1)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)

    scaled_frame = pd.DataFrame(features, columns=feature_columns, index=frame.index)
    return GraphData(features=features, edge_index=edge_index, scaled_frame=scaled_frame)
