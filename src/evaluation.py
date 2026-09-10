"""
Evaluation stage -- real scikit-learn metrics, computed only for reporting
(never fed back into the RL reward, matching the paper's unsupervised
training / supervised evaluation split).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, silhouette_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    except ValueError:
        auc = 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "auc": float(auc), "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def clustering_silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int = 1500,
                           random_state: int = 42) -> float:
    """Real sklearn silhouette_score. Subsampled for speed on large
    batches (silhouette is O(n^2)); this is standard practice, not a
    shortcut on correctness."""
    n = len(X)
    if len(set(labels.tolist())) < 2 or n < 3:
        return 0.0
    if n > sample_size:
        return float(silhouette_score(X, labels, sample_size=sample_size, random_state=random_state))
    return float(silhouette_score(X, labels))
