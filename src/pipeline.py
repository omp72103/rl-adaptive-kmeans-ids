"""
End-to-end orchestration: real NSL-KDD data -> preprocessing -> streaming
batches -> RL-adaptive K-Means vs a fixed-K traditional K-Means baseline
-> real evaluation against ground-truth labels (evaluation-only).

This is the module both `train.py` (CLI) and `api/app.py` (the live
backend the frontend talks to) call into -- there is exactly one
implementation of the framework, not a demo copy and a "real" copy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import (
    Preprocessor, load_and_prepare, NUMERIC, binary_labels, load_raw, clean,
)
from .kmeans_model import EnhancedKMeans
from .anomaly_detection import detect
from .evaluation import classification_metrics, clustering_silhouette
from .rl_environment import ClusteringEnv, ACTIONS
from .rl_agent import QLearningAgent

BURST_COLS = ["count", "srv_count"]


def _raw_numeric_with_burst(df: pd.DataFrame) -> np.ndarray:
    """Numeric feature block with the two burst-proxy columns duplicated
    at the end, as `discretize_state` expects."""
    base = df[NUMERIC].to_numpy(dtype=float)
    burst = df[BURST_COLS].to_numpy(dtype=float)
    return np.concatenate([base, burst], axis=1)


def make_batches(df: pd.DataFrame, y: np.ndarray, batch_size: int, n_batches: int,
                  seed: int = 42, drift: bool = True):
    """Builds `n_batches` real-record windows from the real NSL-KDD data.

    With `drift=True` (default), records are first sorted by their own
    `count` feature (number of connections to the same host in the
    connection window -- an unsupervised, label-free traffic-volume
    signal already present in the raw NSL-KDD columns) and then sliced
    into windows that sweep back and forth from the calmest real traffic
    in the dataset to the burstiest real traffic, with a light local
    shuffle inside each window. This mimics genuinely drifting traffic
    conditions over time using only real records in their natural
    feature range -- no synthetic points, no label information -- so
    different batches actually differ statistically the way real traffic
    windows would, instead of every batch being another IID sample from
    the same stationary distribution (which is what plain random
    shuffling gives you, and why every batch used to look identical to
    the RL agent).

    With `drift=False`, batches are plain random subsamples (used by the
    one-shot manual-mode dashboard, where no streaming story is needed).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if not drift:
        idx_all = np.arange(n)
        rng.shuffle(idx_all)
        batches = []
        for b in range(n_batches):
            start = (b * batch_size) % max(1, n - batch_size)
            idx = idx_all[start:start + batch_size]
            if len(idx) < batch_size:
                idx = np.concatenate([idx, idx_all[: batch_size - len(idx)]])
            batches.append(idx)
        return batches

    order = df["count"].to_numpy().argsort()  # calm -> bursty, real feature only
    span = max(1, n - batch_size)
    # oscillate the window start back and forth across the sorted range so
    # the stream alternates calm/bursty/calm rather than drifting once and
    # then sitting at the extreme forever
    t = np.linspace(0, 1.5, n_batches)
    positions = 0.5 - 0.5 * np.cos(t * 2 * np.pi)
    batches = []
    for b in range(n_batches):
        start = int(positions[b] * span)
        window = order[start:start + batch_size]
        if len(window) < batch_size:
            window = np.concatenate([window, order[: batch_size - len(window)]])
        idx = window.copy()
        rng.shuffle(idx)  # local shuffle: within-batch row order is not sorted
        batches.append(idx)
    return batches


def run_rl_training(n_batches: int = 20, batch_size: int = 1500,
                     fixed_k: int = 4, fixed_metric: str = "euclidean",
                     fixed_sigma: float = 2.0, seed: int = 42, progress_cb=None):
    """Streams `n_batches` real traffic windows through both the RL-adaptive
    agent and a fixed-parameter baseline K-Means, batch by batch. Returns
    full per-batch history (genuinely computed, not templated) plus the
    trained agent.
    """
    df, y_bin, _ = load_and_prepare("train")
    pre = Preprocessor().fit(df)
    Z_all = pre.transform(df)

    batches = make_batches(df, y_bin, batch_size, n_batches, seed=seed, drift=True)

    env = ClusteringEnv(random_state=seed)
    env.reset()
    agent = QLearningAgent(n_actions=len(ACTIONS), seed=seed)

    rl_history, fixed_history = [], []

    from .rl_environment import discretize_state, compute_global_reference
    global_ref = compute_global_reference(_raw_numeric_with_burst(df), seed=seed)

    for b, idx in enumerate(batches):
        batch_df = df.iloc[idx].reset_index(drop=True)
        y_batch = y_bin[idx]
        X_batch = Z_all[idx]
        raw_num = _raw_numeric_with_burst(batch_df)

        # --- RL agent picks the full parameter action ---
        state = discretize_state(raw_num, global_ref=global_ref)
        action_idx = agent.choose_action(state)
        result = env.step(X_batch, raw_num, action_idx, state=state)
        next_state = state  # single-step-per-batch episodic setting
        agent.update(state, action_idx, result.reward, next_state)

        metrics_rl = classification_metrics(y_batch, result.pred, result.distances)
        rl_history.append({
            "batch": b, "state": state, "k": result.k, "metric": result.metric,
            "sigma": result.sigma, "reward": result.reward,
            "reward_parts": result.reward_parts, "silhouette": result.silhouette,
            **metrics_rl,
        })

        # --- fixed baseline, same batch, no adaptation ---
        fixed_model = EnhancedKMeans(k=fixed_k, init="k-means++", metric=fixed_metric,
                                      max_iter=150, n_init=3, random_state=seed)
        fixed_model.fit(X_batch)
        fixed_dist = fixed_model.point_distances(X_batch)
        fixed_pred, _ = detect(fixed_dist, sigma=fixed_sigma)
        fixed_sil = clustering_silhouette(X_batch, fixed_model.labels_)
        metrics_fixed = classification_metrics(y_batch, fixed_pred, fixed_dist)
        fixed_history.append({"batch": b, "k": fixed_k, "silhouette": fixed_sil, **metrics_fixed})

        if progress_cb:
            progress_cb(b, n_batches, rl_history[-1], fixed_history[-1])

    return {
        "agent": agent, "preprocessor": pre,
        "rl_history": rl_history, "fixed_history": fixed_history,
    }


def run_manual_clustering(k: int, metric: str = "euclidean", sigma: float = 2.0,
                           sample_size: int = 2000, split: str = "test", seed: int = 42):
    """One-shot clustering run on a real sample of the (held-out) NSL-KDD
    test set, for the manual "you pick K" dashboard mode."""
    df, y_bin, y_cat = load_and_prepare(split)
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample_df = df.iloc[idx].reset_index(drop=True)
    y_sample = y_bin[idx]
    cat_sample = y_cat[idx]

    pre = Preprocessor().fit(df)  # fit on the split itself for a self-contained demo run
    X = pre.transform(sample_df)

    model = EnhancedKMeans(k=k, init="k-means++", metric=metric, max_iter=150,
                            n_init=3, random_state=seed)
    model.fit(X)
    dist = model.point_distances(X)
    pred, threshold = detect(dist, sigma=sigma)
    sil = clustering_silhouette(X, model.labels_)
    metrics = classification_metrics(y_sample, pred, dist)

    # 2D projection for the frontend scatter plot -- real PCA on the real
    # (scaled) feature matrix, not a placeholder.
    from sklearn.decomposition import PCA
    coords2d = PCA(n_components=2, random_state=seed).fit_transform(X)
    centers2d = PCA(n_components=2, random_state=seed).fit(X).transform(model.cluster_centers_)

    return {
        "k": k, "metric": metric, "sigma": sigma, "threshold": threshold,
        "silhouette": sil, "metrics": metrics,
        "labels": model.labels_.tolist(), "pred": pred.tolist(),
        "y_true": y_sample.tolist(), "attack_category": cat_sample.tolist(),
        "coords2d": coords2d.tolist(), "centers2d": centers2d.tolist(),
        "n_points": len(sample_df),
    }
