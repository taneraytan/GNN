from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import rankdata


ARCHITECTURE_DESCRIPTIONS = {
    "GCN": "Graph Convolutional Network baseline for homophilous row neighborhoods.",
    "GraphSAGE": "Inductive neighborhood aggregation for large and changing tables.",
    "GAT": "Graph Attention Network that learns neighbor importance weights.",
    "GIN": "Graph Isomorphism Network with expressive MLP aggregation.",
    "ChebNet": "Spectral Chebyshev graph convolution for localized filters.",
    "APPNP": "Personalized PageRank propagation for long-range smoothing.",
}

FALLBACK_DETECTOR_NAMES = ("IsolationForest", "LocalOutlierFactor")

THRESHOLD_STRATEGIES = ("contamination", "mad")

MIN_ROWS = 5

EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-5

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class DetectionResult:
    scores: np.ndarray
    threshold: float
    is_outlier: np.ndarray
    detector_scores: dict[str, np.ndarray]
    used_backend: str
    feature_errors: np.ndarray
    threshold_strategy: str


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
    threshold_strategy: str = "contamination",
    progress_callback: ProgressCallback | None = None,
) -> DetectionResult:
    """Train one detector per selection and ensemble rank-normalized anomaly scores."""

    chosen = [name for name in architectures if name in ARCHITECTURE_DESCRIPTIONS]
    if not chosen:
        raise ValueError("Select at least one valid GNN architecture.")
    if threshold_strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(f"Unknown threshold strategy '{threshold_strategy}'. Choose one of {THRESHOLD_STRATEGIES}.")
    if len(features) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} rows are required for outlier detection; got {len(features)}.")

    if _torch_geometric_available():
        return _run_torch_geometric(
            features, edge_index, chosen, contamination, epochs, hidden_dim, learning_rate, seed, threshold_strategy, progress_callback
        )
    return _run_sklearn_fallback(features, contamination, threshold_strategy, seed, progress_callback)


def _run_sklearn_fallback(
    features: np.ndarray,
    contamination: float,
    threshold_strategy: str,
    seed: int,
    progress_callback: ProgressCallback | None,
) -> DetectionResult:
    """Honest non-GNN fallback: real IsolationForest and LOF scores under their own names."""

    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    contamination = float(np.clip(contamination, 0.001, 0.5))
    raw_scores: dict[str, np.ndarray] = {}

    iso = IsolationForest(contamination=contamination, random_state=seed)
    iso.fit(features)
    raw_scores["IsolationForest"] = -iso.score_samples(features)
    if progress_callback:
        progress_callback("IsolationForest", 1, 2)

    lof = LocalOutlierFactor(n_neighbors=max(1, min(20, len(features) - 1)), contamination=contamination)
    lof.fit_predict(features)
    raw_scores["LocalOutlierFactor"] = -lof.negative_outlier_factor_
    if progress_callback:
        progress_callback("LocalOutlierFactor", 2, 2)

    detector_scores = {name: _rank_normalize(values) for name, values in raw_scores.items()}
    scores = np.mean(np.vstack(list(detector_scores.values())), axis=0)
    magnitudes = np.mean(np.vstack([_robust_z(values) for values in raw_scores.values()]), axis=0)
    threshold = _compute_threshold(scores, magnitudes, contamination, threshold_strategy)
    return DetectionResult(
        scores=scores,
        threshold=threshold,
        is_outlier=scores >= threshold,
        detector_scores=detector_scores,
        used_backend="scikit-learn fallback (IsolationForest + LocalOutlierFactor)",
        feature_errors=_robust_feature_deviations(features),
        threshold_strategy=threshold_strategy,
    )


def _run_torch_geometric(
    features: np.ndarray,
    edge_index: np.ndarray,
    architectures: list[str],
    contamination: float,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    threshold_strategy: str,
    progress_callback: ProgressCallback | None,
) -> DetectionResult:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import APPNP, GATConv, GCNConv, GINConv, SAGEConv, ChebConv

    torch.manual_seed(seed)
    device = _select_device(torch)
    x = torch.tensor(features, dtype=torch.float32, device=device)
    edges = torch.tensor(edge_index, dtype=torch.long, device=device)
    if edges.numel() == 0:
        loops = torch.arange(x.shape[0], dtype=torch.long, device=device)
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

    raw_scores: dict[str, np.ndarray] = {}
    feature_error_total = np.zeros_like(features, dtype=np.float64)
    for position, architecture in enumerate(architectures):
        model = _build_graph_autoencoder(features.shape[1], hidden_dim, conv_factory[architecture]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        model.train()
        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            reconstruction = model(x, edges)
            loss = F.mse_loss(reconstruction, x)
            loss.backward()
            optimizer.step()
            current = float(loss.item())
            if best_loss - current > EARLY_STOPPING_MIN_DELTA:
                best_loss = current
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= EARLY_STOPPING_PATIENCE:
                    break
        model.eval()
        with torch.no_grad():
            reconstruction = model(x, edges)
            error_matrix = ((reconstruction - x) ** 2).cpu().numpy()
            raw_scores[architecture] = error_matrix.mean(axis=1)
            feature_error_total += error_matrix
        if progress_callback:
            progress_callback(architecture, position + 1, len(architectures))

    detector_scores = {name: _rank_normalize(values) for name, values in raw_scores.items()}
    scores = np.mean(np.vstack(list(detector_scores.values())), axis=0)
    magnitudes = np.mean(np.vstack([_robust_z(values) for values in raw_scores.values()]), axis=0)
    contamination = float(np.clip(contamination, 0.001, 0.5))
    threshold = _compute_threshold(scores, magnitudes, contamination, threshold_strategy)
    return DetectionResult(
        scores=scores,
        threshold=threshold,
        is_outlier=scores >= threshold,
        detector_scores=detector_scores,
        used_backend=f"torch-geometric GNN ({device})",
        feature_errors=feature_error_total / max(1, len(architectures)),
        threshold_strategy=threshold_strategy,
    )


def _select_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


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


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] by average rank; robust to a single extreme score."""

    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return np.zeros_like(values, dtype=float)
    ranks = rankdata(values, method="average")
    return (ranks - 1.0) / (len(values) - 1.0)


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Center and scale by median/MAD so detector magnitudes are comparable."""

    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 1e-12 else (float(values.std()) or 1.0)
    return (values - median) / scale


def _compute_threshold(scores: np.ndarray, magnitudes: np.ndarray, contamination: float, strategy: str) -> float:
    """Translate the chosen strategy into a cutoff on the displayed rank-ensemble scores.

    'contamination' always flags ~contamination rows. 'mad' flags rows whose
    magnitude-preserving robust-z ensemble exceeds median + 3 * 1.4826 * MAD —
    which may flag none on clean data — then maps that decision back to the
    displayed score scale (ranks alone carry no magnitude, so the MAD rule
    cannot be applied to them directly).
    """

    if strategy == "mad":
        median = float(np.median(magnitudes))
        mad = float(np.median(np.abs(magnitudes - median)))
        if mad > 1e-12:
            flagged = magnitudes > median + 3.0 * 1.4826 * mad
            if flagged.any():
                return float(scores[flagged].min())
            return float(scores.max()) + 1e-6
    return float(np.quantile(scores, 1 - float(np.clip(contamination, 0.001, 0.5))))


def _robust_feature_deviations(features: np.ndarray) -> np.ndarray:
    """Per-feature robust z-scores used to explain fallback detections."""

    features = np.asarray(features, dtype=float)
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    scale = np.where(mad > 1e-12, 1.4826 * mad, 1.0)
    return np.abs(features - median) / scale
