"""
Distance-based anomaly scoring + adaptive thresholding.

Per the paper: K-Means groups traffic by similarity but does not directly
flag attacks. For every point we compute its distance to its assigned
centroid (the anomaly score); a large distance means the point is a poor
fit for the cluster it landed in and is therefore "potentially
anomalous". The anomaly score is compared against an adaptive threshold
(derived from the batch's own score distribution, not a fixed constant)
to turn the clustering into a detection decision.
"""
from __future__ import annotations

import numpy as np


def anomaly_scores(distances: np.ndarray) -> np.ndarray:
    """Anomaly score == raw distance-to-assigned-centroid."""
    return distances


def adaptive_threshold(scores: np.ndarray, sigma: float = 2.0) -> float:
    """mean + sigma * std of the batch's own score distribution -- adapts
    per batch instead of using one fixed global cutoff, matching the
    paper's 'adaptive thresholding' step."""
    mu = float(np.mean(scores))
    sd = float(np.std(scores)) or 1e-6
    return mu + sigma * sd


def detect(distances: np.ndarray, sigma: float = 2.0):
    """Returns (predicted_labels, threshold). predicted==1 means flagged
    as anomalous / potential attack."""
    scores = anomaly_scores(distances)
    threshold = adaptive_threshold(scores, sigma)
    pred = (scores > threshold).astype(int)
    return pred, threshold
