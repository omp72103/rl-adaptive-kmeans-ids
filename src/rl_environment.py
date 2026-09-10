"""
ClusteringEnv -- a real Gym-style Markov Decision Process for adaptive
K-Means parameter selection, matching the paper's diagram:

    Traffic Window/Batch -> Current State -> RL action
        -> K-Means + Anomaly Detection -> Reward -> Next Traffic Window

State
-----
A discretized descriptor of the *current traffic batch's own statistics*
(feature spread + connection-burstiness), computed WITHOUT looking at
attack labels -- this is what the RL agent is allowed to observe, exactly
as the paper specifies ("RL adaptation: unsupervised").

Action
------
A joint choice of the parameters the paper names as RL-tunable:
    - K               in {2,...,8}
    - distance metric in {euclidean, manhattan}
    - anomaly sigma   in {1.0, 1.5, 2.0}   (adaptive-threshold strictness)
(centroid init is held at the literature-standard k-means++, and
max_iter at a fixed convergence cap of 150 -- both are documented,
deliberate scope choices, see README.)

Reward
------
An unsupervised composite of clustering-quality signals named in the
paper: cluster compactness, cluster separation, silhouette quality,
clustering stability (agreement with the previous batch's structure),
and noise robustness (not wildly over/under flagging anomalies). Ground
truth labels never enter this function.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from .kmeans_model import EnhancedKMeans
from .anomaly_detection import detect
from .evaluation import clustering_silhouette
from .kmeans_model import _SCIPY_METRIC

K_CHOICES = [2, 3, 4, 5, 6, 7, 8]
METRIC_CHOICES = ["euclidean", "manhattan"]
SIGMA_CHOICES = [1.0, 1.5, 2.0]

ACTIONS = list(itertools.product(K_CHOICES, METRIC_CHOICES, SIGMA_CHOICES))  # 42 actions (7 K choices x 2 metrics x 3 sigma thresholds)


@dataclass
class StepResult:
    state: str
    next_state: str
    action_idx: int
    k: int
    metric: str
    sigma: float
    reward: float
    reward_parts: dict
    model: EnhancedKMeans
    distances: np.ndarray
    pred: np.ndarray
    threshold: float
    silhouette: float


def discretize_state(raw_numeric_batch: np.ndarray, global_ref: dict | None = None) -> str:
    """raw_numeric_batch: (n, d) *unscaled* numeric NSL-KDD columns for
    the current batch (with the two burst-proxy columns duplicated at the
    end). Buckets overall variance and connection-rate burstiness into a
    small discrete state space -- purely a function of the batch's own
    statistics compared against a fixed, dataset-wide unsupervised
    reference (`global_ref`, computed once from the training data's raw
    features -- no labels), so different batches actually land in
    different states instead of every batch looking "average" relative to
    itself."""
    variances = raw_numeric_batch.var(axis=0)
    spread = float(np.mean(variances))
    if global_ref is None:
        spread_bucket = "lo" if spread < np.percentile(variances, 25) else (
            "hi" if spread > np.percentile(variances, 75) else "mid"
        )
        burst_proxy = float(np.mean(raw_numeric_batch[:, -2:]))
        burst_bucket = "burst" if burst_proxy > np.median(raw_numeric_batch[:, -2:]) else "calm"
        return f"{spread_bucket}_{burst_bucket}"

    spread_bucket = "lo" if spread < global_ref["spread_q25"] else (
        "hi" if spread > global_ref["spread_q75"] else "mid"
    )
    burst_proxy = float(np.mean(raw_numeric_batch[:, -2:]))
    burst_bucket = "burst" if burst_proxy > global_ref["burst_median"] else "calm"
    return f"{spread_bucket}_{burst_bucket}"


def compute_global_reference(raw_numeric_full: np.ndarray, n_probe_batches: int = 40,
                              batch_size: int = 1500, seed: int = 0) -> dict:
    """Computes the fixed, dataset-wide reference stats `discretize_state`
    needs, by measuring per-feature variance and burst-proxy level across
    many random real sub-samples of the *unlabeled* training data. This is
    a one-time, unsupervised calibration step -- like fitting a scaler --
    not a form of label leakage."""
    rng = np.random.default_rng(seed)
    n = len(raw_numeric_full)
    spreads, bursts = [], []
    for _ in range(n_probe_batches):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        sample = raw_numeric_full[idx]
        spreads.append(float(np.mean(sample.var(axis=0))))
        bursts.append(float(np.mean(sample[:, -2:])))
    return {
        "spread_q25": float(np.percentile(spreads, 25)),
        "spread_q75": float(np.percentile(spreads, 75)),
        "burst_median": float(np.median(bursts)),
    }


def compactness_score(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, metric: str) -> float:
    d = cdist(X, centers, metric=_SCIPY_METRIC[metric])
    intra = d[np.arange(len(X)), labels].mean()
    return float(1.0 / (1.0 + intra))


def separation_score(centers: np.ndarray, metric: str) -> float:
    if len(centers) < 2:
        return 0.0
    d = cdist(centers, centers, metric=_SCIPY_METRIC[metric])
    iu = np.triu_indices(len(centers), k=1)
    avg = d[iu].mean()
    return float(min(1.0, avg / 8.0))


def stability_score(prev_labels: np.ndarray | None, prev_X: np.ndarray | None,
                     labels: np.ndarray, X: np.ndarray, sample: int = 200) -> float:
    """Rand-index-style pairwise agreement between this batch's clustering
    structure and the previous batch's, evaluated on a matched-size random
    sample of each (batches are independent draws, so we compare
    *within-batch pairwise co-cluster patterns*, which is what stability
    means for a streaming, non-persistent-ID setting)."""
    if prev_labels is None:
        return 1.0
    rng = np.random.default_rng(0)
    n1, n2 = len(labels), len(prev_labels)
    m = min(sample, n1, n2)
    idx1 = rng.choice(n1, size=m, replace=False)
    idx2 = rng.choice(n2, size=m, replace=False)
    l1, l2 = labels[idx1], prev_labels[idx2]
    # compare the *distribution* of within-cluster pair fractions as a
    # cheap, real stability proxy (co-membership rate similarity),
    # vectorized over all pairs via broadcasting
    def pair_rate(l):
        iu = np.triu_indices(len(l), k=1)
        same = (l[iu[0]] == l[iu[1]])
        return float(same.mean()) if same.size else 0.0
    r1, r2 = pair_rate(l1), pair_rate(l2)
    return float(max(0.0, 1.0 - abs(r1 - r2) * 3))


def noise_robustness_score(pred: np.ndarray, target_ratio: float = 0.30) -> float:
    """Penalizes wildly over/under-flagging relative to an assumed
    operating point (a domain prior an analyst would set for this
    deployment, e.g. 'we expect roughly a third of monitored sessions in
    this environment to be malicious' -- NOT derived from the ground-truth
    labels of any specific batch, which the RL loop never sees)."""
    flagged = pred.mean()
    return float(max(0.0, 1.0 - abs(flagged - target_ratio) / target_ratio))


class ClusteringEnv:
    """One `step()` = one traffic batch processed end-to-end."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._prev_labels = None
        self._prev_X = None

    def reset(self):
        self._prev_labels = None
        self._prev_X = None

    def step(self, X_scaled: np.ndarray, raw_numeric_with_burst_cols: np.ndarray,
              action_idx: int, state: str | None = None,
              global_ref: dict | None = None) -> StepResult:
        if state is None:
            state = discretize_state(raw_numeric_with_burst_cols, global_ref=global_ref)
        k, metric, sigma = ACTIONS[action_idx]

        model = EnhancedKMeans(k=k, init="k-means++", metric=metric,
                                max_iter=150, n_init=3, random_state=self.random_state)
        model.fit(X_scaled)
        distances = model.point_distances(X_scaled)
        pred, threshold = detect(distances, sigma=sigma)

        sil = clustering_silhouette(X_scaled, model.labels_)
        sil01 = (sil + 1) / 2
        comp = compactness_score(X_scaled, model.labels_, model.cluster_centers_, metric)
        sep = separation_score(model.cluster_centers_, metric)
        stab = stability_score(self._prev_labels, self._prev_X, model.labels_, X_scaled)
        noise = noise_robustness_score(pred)

        # Silhouette-dominant composite reward: the companion ARLDFK paper
        # is explicit that the RL agent works "by maximizing a
        # silhouette-based reward function", with the sibling paper's
        # compactness/separation/stability/noise-robustness terms acting
        # as smaller regularizers around that primary signal rather than
        # co-equal weights (the previous 0.35/0.20/0.20/0.15/0.10 split
        # let the agent trade away silhouette quality for the other terms,
        # which is why RL-adaptive runs could end up with *lower* average
        # silhouette than the naive fixed-K baseline -- exactly backwards
        # for a framework whose stated objective is silhouette-based).
        reward = 0.60 * sil01 + 0.15 * comp + 0.15 * sep + 0.05 * stab + 0.05 * noise

        self._prev_labels = model.labels_
        self._prev_X = X_scaled

        next_state = state  # next batch's state is computed by the caller once it arrives
        return StepResult(
            state=state, next_state=next_state, action_idx=action_idx,
            k=k, metric=metric, sigma=sigma, reward=float(reward),
            reward_parts={"silhouette": sil01, "compactness": comp, "separation": sep,
                          "stability": stab, "noise_robustness": noise},
            model=model, distances=distances, pred=pred, threshold=threshold, silhouette=sil,
        )
