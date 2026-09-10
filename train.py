#!/usr/bin/env python3
"""
CLI entry point: trains the RL-adaptive K-Means agent by streaming real
NSL-KDD traffic batches, compares it against a fixed-K traditional
K-Means baseline, and writes all results (metrics, plots, trained Q-table)
to results/.

Usage:
    python train.py --batches 200 --batch-size 1500 --fixed-k 4

Note on --batches: the RL agent's action space has 42 joint (K, distance
metric, anomaly-threshold) choices, and UCB1 exploration tries every
action at least once per traffic state before it starts exploiting the
best one it has found. With ~4 traffic states actually occurring in real
NSL-KDD data, that means roughly 4 x 42 = ~170 batches are needed just to
finish exploring -- fewer batches than that will show the coverage
warning in the summary and won't reflect a converged policy yet.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline import run_rl_training

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1500)
    p.add_argument("--fixed-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Streaming {args.batches} real NSL-KDD traffic batches "
          f"({args.batch_size} records/batch) through the RL agent "
          f"and a fixed-K={args.fixed_k} baseline...\n")

    t0 = time.time()

    def progress(b, n, rl, fx):
        print(f"[batch {b+1:>3}/{n}] state={rl['state']:<9} "
              f"RL -> K={rl['k']} metric={rl['metric']:<9} sigma={rl['sigma']} "
              f"reward={rl['reward']:.3f} F1={rl['f1']:.3f}  |  "
              f"fixed(K={fx['k']}) F1={fx['f1']:.3f}")

    out = run_rl_training(n_batches=args.batches, batch_size=args.batch_size,
                           fixed_k=args.fixed_k, seed=args.seed, progress_cb=progress)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")

    rl_hist, fx_hist = out["rl_history"], out["fixed_history"]

    def avg(hist, key):
        return float(np.mean([h[key] for h in hist]))

    metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "fpr", "silhouette"]
    summary = {
        "n_batches": args.batches, "batch_size": args.batch_size,
        "fixed_k": args.fixed_k, "elapsed_sec": elapsed,
        "rl": {k: avg(rl_hist, k) for k in metric_keys},
        "fixed": {k: avg(fx_hist, k) for k in metric_keys},
    }
    print("\n=== Averaged over all batches ===")
    print(f"{'metric':<12}{'fixed-K':>12}{'RL-adaptive':>14}")
    for k in summary["rl"]:
        print(f"{k:<12}{summary['fixed'][k]:>12.3f}{summary['rl'][k]:>14.3f}")

    # UCB1 needs to try every action at least once per state before it can
    # start exploiting the best one it found -- report coverage so a short
    # run's numbers aren't mistaken for a converged policy.
    n_actions = out["agent"].n_actions
    coverage = out["agent"].coverage_summary()
    print(f"\n=== UCB1 exploration coverage ({n_actions} actions/state) ===")
    for state, tried in coverage.items():
        flag = "" if tried >= n_actions else "  <- still exploring, not yet exploiting"
        print(f"  {state:<10} {tried:>3}/{n_actions} actions tried{flag}")
    summary["ucb_coverage"] = coverage
    summary["n_actions"] = n_actions

    # A fairer read on "did the agent actually learn": compare the first
    # half of the run (necessarily exploration-heavy under UCB1) against
    # the second half (mostly exploiting whatever it has learned by then).
    half = len(rl_hist) // 2
    if half >= 5:
        def avg_half(hist, key, part):
            sub = hist[:half] if part == "first" else hist[half:]
            return float(np.mean([h[key] for h in sub]))
        print(f"\n=== First half (batches 1-{half}, exploration-heavy) vs "
              f"second half (batches {half+1}-{len(rl_hist)}, mostly exploiting) ===")
        print(f"{'metric':<12}{'1st-half RL':>13}{'2nd-half RL':>13}{'2nd-half fixed':>16}")
        halves = {}
        for k in metric_keys:
            f1h, s1h, s1fx = avg_half(rl_hist, k, "first"), avg_half(rl_hist, k, "second"), avg_half(fx_hist, k, "second")
            print(f"{k:<12}{f1h:>13.3f}{s1h:>13.3f}{s1fx:>16.3f}")
            halves[k] = {"first_half_rl": f1h, "second_half_rl": s1h, "second_half_fixed": s1fx}
        summary["halves"] = halves

    # persist artifacts
    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump({"rl": rl_hist, "fixed": fx_hist, "summary": summary}, f, indent=2, default=str)
    out["agent"].save(os.path.join(RESULTS_DIR, "q_agent.json"))
    out["preprocessor"].save(os.path.join(RESULTS_DIR, "preprocessor.joblib"))
    print(f"\nSaved: results/history.json, results/q_agent.json, results/preprocessor.joblib")

    _plot_reward(rl_hist)
    _plot_k(rl_hist, args.fixed_k)
    _plot_comparison(summary)
    print("Saved: results/reward_curve.png, results/k_selection.png, results/comparison.png")


def _plot_reward(rl_hist):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [h["batch"] + 1 for h in rl_hist]
    ax.plot(xs, [h["reward"] for h in rl_hist], marker="o", color="#2a78d6")
    ax.set_xlabel("Traffic batch"); ax.set_ylabel("Composite reward")
    ax.set_title("RL learning curve (unsupervised reward per batch)")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "reward_curve.png"), dpi=140)
    plt.close(fig)


def _plot_k(rl_hist, fixed_k):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [h["batch"] + 1 for h in rl_hist]
    ax.plot(xs, [h["k"] for h in rl_hist], marker="o", color="#2a78d6", label="RL-selected K")
    ax.axhline(fixed_k, color="#eb6834", linestyle="--", label=f"Fixed baseline K={fixed_k}")
    ax.set_xlabel("Traffic batch"); ax.set_ylabel("K"); ax.set_yticks(range(1, 9))
    ax.set_title("K chosen by the RL agent vs. the fixed baseline")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "k_selection.png"), dpi=140)
    plt.close(fig)


def _plot_comparison(summary):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    fixed_vals = [summary["fixed"][m] for m in metrics]
    rl_vals = [summary["rl"][m] for m in metrics]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, fixed_vals, w, label="Fixed-K K-Means", color="#eb6834")
    ax.bar(x + w/2, rl_vals, w, label="RL-adaptive K-Means", color="#2a78d6")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Traditional (fixed K) vs. proposed RL-adaptive K-Means")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
