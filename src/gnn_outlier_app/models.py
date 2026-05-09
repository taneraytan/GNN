from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable

import numpy as np


ARCHITECTURE_DESCRIPTIONS = {
    "GCN": "Graph Convolutional Network baseline for homophilous row neighborhoods.",
    "GraphSAGE": "Inductive neighborhood aggregation for large and changing tables.",
    "GAT": "Graph Attention Network that learns neighbor importance weights.",
    "GIN": "Graph Isomorphism Network with expressive MLP aggregation.",
    "ChebNet": "Spectral Chebyshev graph convolution for localized filters.",
    "APPNP": "Personalized PageRank propagation for long-range smoothing.",
}


@dataclass(frozen=True)
class DetectionResult:
    scores: np.ndarray
    threshold: float
    is_outlier: np.ndarray
    architecture_scores: dict[str, np.ndarray]
    used_backend: str


def _torch_geometric_available() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


def run_detection(
    features: np.ndarray,
    edge_index: np.ndarray,
    architectures: list[str],
    contamination: float = 0.05,
    epochs: int = 80,
    hidden_dim: int = 32,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> DetectionResult:
    """Train one graph autoencoder per selected architecture and ensemble anomaly scores."""

    chosen = [name for name in architectures if name in ARCHITECTURE_DESCRIPTIONS]
    if not chosen:
        chosen = ["GCN", "GraphSAGE", "GAT", "GIN", "ChebNet", "APPNP"]

    if _torch_geometric_available() and len(features) > 1:
        return _run_torch_geometric(features, edge_index, chosen, contamination, epochs, hidden_dim, learning_rate, seed)
    return _run_sklearn_fallback(features, chosen, contamination)


def _run_sklearn_fallback(features: np.ndarray, architectures: list[str], contamination: float) -> DetectionResult:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    contamination = float(np.clip(contamination, 0.001, 0.5))
    architecture_scores: dict[str, np.ndarray] = {}

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(features)
    base_score = -iso.score_samples(features)

    lof = LocalOutlierFactor(n_neighbors=max(1, min(20, len(features) - 1)), contamination=contamination)
    lof.fit_predict(features)
    lof_score = -lof.negative_outlier_factor_

    for index, architecture in enumerate(architectures):
        blend = (0.7 + index * 0.03) * base_score + (0.3 - min(index * 0.02, 0.2)) * lof_score
        architecture_scores[architecture] = _minmax(blend)

    scores = _minmax(np.mean(np.vstack(list(architecture_scores.values())), axis=0))
    threshold = float(np.quantile(scores, 1 - contamination))
    return DetectionResult(scores=scores, threshold=threshold, is_outlier=scores >= threshold, architecture_scores=architecture_scores, used_backend="scikit-learn fallback")


def _run_torch_geometric(
    features: np.ndarray,
    edge_index: np.ndarray,
    architectures: list[str],
    contamination: float,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> DetectionResult:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import APPNP, GATConv, GCNConv, GINConv, SAGEConv, ChebConv

    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    edges = torch.tensor(edge_index, dtype=torch.long)
    if edges.numel() == 0:
        loops = torch.arange(x.shape[0], dtype=torch.long)
        edges = torch.stack([loops, loops], dim=0)

    class APPNPBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.propagation = APPNP(K=10, alpha=0.1)

        def forward(self, x, edge_index):
            return self.propagation(F.relu(self.linear(x)), edge_index)

    conv_factory: dict[str, Callable[[int, int], nn.Module]] = {
        "GCN": lambda in_dim, out_dim: GCNConv(in_dim, out_dim),
        "GraphSAGE": lambda in_dim, out_dim: SAGEConv(in_dim, out_dim),
        "GAT": lambda in_dim, out_dim: GATConv(in_dim, out_dim, heads=2, concat=False),
        "GIN": lambda in_dim, out_dim: GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))),
        "ChebNet": lambda in_dim, out_dim: ChebConv(in_dim, out_dim, K=3),
        "APPNP": lambda in_dim, out_dim: APPNPBlock(in_dim, out_dim),
    }

    architecture_scores: dict[str, np.ndarray] = {}
    for architecture in architectures:
        model = _build_graph_autoencoder(features.shape[1], hidden_dim, conv_factory[architecture])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        model.train()
        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            reconstruction = model(x, edges)
            loss = F.mse_loss(reconstruction, x)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            reconstruction = model(x, edges)
            error = torch.mean((reconstruction - x) ** 2, dim=1).cpu().numpy()
            architecture_scores[architecture] = _minmax(error)

    scores = _minmax(np.mean(np.vstack(list(architecture_scores.values())), axis=0))
    threshold = float(np.quantile(scores, 1 - float(np.clip(contamination, 0.001, 0.5))))
    return DetectionResult(scores=scores, threshold=threshold, is_outlier=scores >= threshold, architecture_scores=architecture_scores, used_backend="torch-geometric GNN")


def _build_graph_autoencoder(input_dim: int, hidden_dim: int, conv_builder: Callable[[int, int], object]):
    from torch import nn

    class GraphAutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = conv_builder(input_dim, hidden_dim)
            self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

        def forward(self, x, edge_index):
            from torch.nn import functional as F

            z = self.encoder(x, edge_index)
            z = F.relu(z)
            return self.decoder(z)

    return GraphAutoEncoder()


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = values.max() - values.min() if values.size else 0
    if spread <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - values.min()) / spread
