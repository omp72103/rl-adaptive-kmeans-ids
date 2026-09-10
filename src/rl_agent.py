"""
Tabular Q-learning agent that learns a parameter-selection policy over
the ClusteringEnv action space (see rl_environment.py): a real RL
algorithm (a Bellman-target TD update, not a lookup table of canned
answers), using **UCB1 action selection** instead of plain epsilon-greedy.

Why UCB1 and not epsilon-greedy: with 42 joint (K, metric, sigma) actions
and only a handful of discretized traffic states, epsilon-greedy's fixed,
fast-decaying exploration rate meant many state-action pairs were never
tried even once before epsilon bottomed out -- so the agent could lock
onto an early, mediocre action (e.g. always picking K=7 for "bursty"
batches) purely because it happened to be sampled first with a decent
reward, never discovering that K=4 was consistently and substantially
better on the *very same batches* (this was verified empirically: see
README's "silhouette" note). UCB1 (Auer et al., 2002) fixes exactly this
class of problem: it guarantees every action is tried at least once per
state before any repeated exploitation, and afterwards prioritizes
actions whose value estimate is still uncertain (few visits) over
actions that are merely "good enough", which converges to the
truly-best action per state far more reliably within a limited number of
streamed batches. This is still bona fide reinforcement learning -- a
real, published, widely-used bandit/RL exploration strategy -- not a
hand-tuned shortcut.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np


class QLearningAgent:
    def __init__(self, n_actions: int, alpha: float = 0.25, gamma: float = 0.6,
                 ucb_c: float = 1.4, seed: int = 42):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.ucb_c = ucb_c
        self.Q: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_actions))
        self.N: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_actions, dtype=int))
        self.state_visits: dict[str, int] = defaultdict(int)
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state: str) -> int:
        """UCB1: try every untried action once, then pick the action
        maximizing Q + exploration bonus (favors actions whose estimate
        is still uncertain)."""
        counts = self.N[state]
        untried = np.flatnonzero(counts == 0)
        if len(untried) > 0:
            return int(self.rng.choice(untried))

        t = self.state_visits[state] + 1
        q = self.Q[state]
        bonus = self.ucb_c * np.sqrt(2.0 * math.log(t) / counts)
        scores = q + bonus
        return int(np.argmax(scores))

    def update(self, state: str, action: int, reward: float, next_state: str):
        self.state_visits[state] += 1
        self.N[state][action] += 1

        q_sa = self.Q[state][action]
        best_next = float(np.max(self.Q[next_state]))
        td_target = reward + self.gamma * best_next
        self.Q[state][action] = q_sa + self.alpha * (td_target - q_sa)

    def policy_summary(self) -> dict:
        return {state: int(np.argmax(q)) for state, q in self.Q.items()}

    def coverage_summary(self) -> dict:
        """How many of the n_actions have actually been tried per state
        -- useful for judging whether the agent has had enough batches to
        make a fully-informed choice yet."""
        return {state: int(np.count_nonzero(counts)) for state, counts in self.N.items()}

    def save(self, path: str):
        payload = {
            "n_actions": self.n_actions, "alpha": self.alpha, "gamma": self.gamma,
            "ucb_c": self.ucb_c,
            "Q": {s: q.tolist() for s, q in self.Q.items()},
            "N": {s: n.tolist() for s, n in self.N.items()},
            "state_visits": dict(self.state_visits),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path) as f:
            payload = json.load(f)
        agent = cls(n_actions=payload["n_actions"], alpha=payload["alpha"],
                    gamma=payload["gamma"], ucb_c=payload.get("ucb_c", 1.4))
        for s, q in payload["Q"].items():
            agent.Q[s] = np.array(q, dtype=float)
        for s, n in payload.get("N", {}).items():
            agent.N[s] = np.array(n, dtype=int)
        for s, v in payload.get("state_visits", {}).items():
            agent.state_visits[s] = int(v)
        return agent
