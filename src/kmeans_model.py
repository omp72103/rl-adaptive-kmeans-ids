"""
"Enhanced K-Means" clustering module.

The source paper lists the RL-tunable parameters as: number of clusters K,
centroid initialization strategy, distance metric, maximum iterations, and
anomaly threshold. scikit-learn's KMeans only supports Euclidean distance,
so for the euclidean case we use the real, battle-tested sklearn
implementation directly; for the other distance metrics named in the
paper (manhattan / cosine) we run a from-scratch Lloyd's-algorithm
implementation (k-means++ seeding, real iterative centroid updates) so the
"distance metric" action is genuinely enacted rather than ignored.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans as SKKMeans

VALID_METRICS = ("euclidean", "manhattan", "cosine")
VALID_INITS = ("k-means++", "random")

# scipy's cdist spells Manhattan distance "cityblock"
_SCIPY_METRIC = {"euclidean": "euclidean", "manhattan": "cityblock", "cosine": "cosine"}


class EnhancedKMeans:
    """Unified interface: EnhancedKMeans(k, init, metric, max_iter).fit(X)."""

    def __init__(self, k: int, init: str = "k-means++", metric: str = "euclidean",
                 max_iter: int = 100, n_init: int = 5, random_state: int | None = None):
        assert metric in VALID_METRICS, f"metric must be one of {VALID_METRICS}"
        assert init in VALID_INITS, f"init must be one of {VALID_INITS}"
        self.k = k
        self.init = init
        self.metric = metric
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> "EnhancedKMeans":
        if self.metric == "euclidean":
            model = SKKMeans(
                n_clusters=self.k, init=self.init, max_iter=self.max_iter,
                n_init=self.n_init, random_state=self.random_state,
            )
            model.fit(X)
            self.cluster_centers_ = model.cluster_centers_
            self.labels_ = model.labels_
            self.inertia_ = float(model.inertia_)
            self.n_iter_ = model.n_iter_
            return self
        return self._fit_custom(X)

    def _fit_custom(self, X: np.ndarray) -> "EnhancedKMeans":
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        best_centers, best_labels, best_iters = None, None, 0

        for _ in range(self.n_init):
            centers = self._init_centers(X, rng)
            labels = np.zeros(len(X), dtype=int)
            for it in range(self.max_iter):
                dist = cdist(X, centers, metric=_SCIPY_METRIC[self.metric])
                new_labels = np.argmin(dist, axis=1)
                if it > 0 and np.array_equal(new_labels, labels):
                    labels = new_labels
                    break
                labels = new_labels
                for c in range(self.k):
                    members = X[labels == c]
                    if len(members) > 0:
                        centers[c] = members.mean(axis=0)
                    else:
                        centers[c] = X[rng.integers(0, len(X))]
            dist = cdist(X, centers, metric=_SCIPY_METRIC[self.metric])
            inertia = float(np.sum(np.min(dist, axis=1) ** 2))
            if inertia < best_inertia:
                best_inertia, best_centers, best_labels, best_iters = inertia, centers, labels, it + 1

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iters
        return self

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(X)
        if self.init == "random":
            idx = rng.choice(n, size=self.k, replace=False)
            return X[idx].copy()
        # k-means++ seeding, generalized to the chosen distance metric
        idx = [rng.integers(0, n)]
        for _ in range(1, self.k):
            d2 = cdist(X, X[idx], metric=_SCIPY_METRIC[self.metric]).min(axis=1) ** 2
            probs = d2 / (d2.sum() + 1e-12)
            idx.append(rng.choice(n, p=probs))
        return X[idx].copy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Distance of every sample to every centroid."""
        return cdist(X, self.cluster_centers_, metric=_SCIPY_METRIC[self.metric])

    def point_distances(self, X: np.ndarray) -> np.ndarray:
        """Distance of every sample to its *assigned* centroid -- the raw
        material for the anomaly score."""
        d = self.transform(X)
        return d[np.arange(len(X)), self.labels_]
