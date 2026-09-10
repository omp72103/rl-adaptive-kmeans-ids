"""
Sanity tests -- run against the real NSL-KDD data (small samples, so this
stays fast). Run with:  pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.preprocessing import load_and_prepare, Preprocessor
from src.kmeans_model import EnhancedKMeans
from src.anomaly_detection import detect
from src.evaluation import classification_metrics, clustering_silhouette
from src.rl_agent import QLearningAgent
from src.rl_environment import ACTIONS
from src.pipeline import run_rl_training, run_manual_clustering


def test_data_loads():
    df, y_bin, y_cat = load_and_prepare("train")
    assert len(df) > 100000
    assert set(np.unique(y_bin)) <= {0, 1}
    assert 0.0 < y_bin.mean() < 1.0


def test_preprocessor_shapes():
    df, y_bin, _ = load_and_prepare("train")
    pre = Preprocessor().fit(df.iloc[:5000])
    Z = pre.transform(df.iloc[:5000])
    assert Z.shape[0] == 5000
    assert np.isfinite(Z).all()
    assert Z.max() <= 5.0 and Z.min() >= -5.0


def test_kmeans_euclidean_and_manhattan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    for metric in ["euclidean", "manhattan", "cosine"]:
        m = EnhancedKMeans(k=3, metric=metric, max_iter=50, n_init=2, random_state=0).fit(X)
        assert m.cluster_centers_.shape == (3, 6)
        assert len(set(m.labels_.tolist())) <= 3


def test_anomaly_detect_and_metrics():
    rng = np.random.default_rng(1)
    dist = np.abs(rng.normal(size=500))
    pred, thr = detect(dist, sigma=1.5)
    y_true = (dist > np.percentile(dist, 85)).astype(int)
    m = classification_metrics(y_true, pred, dist)
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["auc"] <= 1


def test_q_agent_learns_something():
    agent = QLearningAgent(n_actions=len(ACTIONS), seed=0)
    for _ in range(50):
        a = agent.choose_action("s0")
        agent.update("s0", a, reward=1.0 if a == 0 else 0.0, next_state="s0")
    assert agent.Q["s0"][0] > 0


def test_manual_clustering_end_to_end():
    out = run_manual_clustering(k=4, metric="euclidean", sigma=1.5, sample_size=500, split="test")
    assert out["n_points"] == 500
    assert len(out["labels"]) == 500
    assert -1 <= out["silhouette"] <= 1


def test_rl_training_smoke():
    out = run_rl_training(n_batches=3, batch_size=400, fixed_k=4, seed=0)
    assert len(out["rl_history"]) == 3
    assert len(out["fixed_history"]) == 3
    for h in out["rl_history"]:
        assert 0 <= h["reward"] <= 1
