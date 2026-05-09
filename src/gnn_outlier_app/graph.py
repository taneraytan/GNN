from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GraphData:
    """Feature matrix plus edge list used by GNN models."""

    features: np.ndarray
    edge_index: np.ndarray
    scaled_frame: pd.DataFrame


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
        model = NearestNeighbors(n_neighbors=neighbors + 1)
        model.fit(features)
        indices = model.kneighbors(features, return_distance=False)
        edges: set[tuple[int, int]] = set()
        for source, row in enumerate(indices):
            for target in row[1:]:
                edges.add((source, int(target)))
                edges.add((int(target), source))
        edge_index = np.array(sorted(edges), dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)

    scaled_frame = pd.DataFrame(features, columns=feature_columns, index=frame.index)
    return GraphData(features=features, edge_index=edge_index, scaled_frame=scaled_frame)
