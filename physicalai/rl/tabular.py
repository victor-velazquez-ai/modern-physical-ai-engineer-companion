"""Tabular Q-learning — reward-driven control in miniature (Ch 17).

A pure-Python Q-learner: no torch, no GPU. It is the toy analogue of the policy-optimization the
book runs with PPO in Isaac Lab (Ch 18). The point is the *mechanism* — learn a value from reward
by trial and error, with epsilon-greedy exploration — not the algorithm's scale.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

from physicalai.envs.gridworld import GW_ACTIONS, GridWorld


def train_q(
    make_env: Callable[[], GridWorld],
    episodes: int = 4000,
    alpha: float = 0.5,
    gamma: float = 0.97,
    epsilon: float = 0.2,
    seed: int = 0,
) -> tuple[dict, list[float]]:
    """Q-learning over `episodes`. Returns (Q-table, per-episode-return history). The return
    history rises as the policy learns — the toy version of an RL training curve."""
    rng = random.Random(seed)
    q: dict = defaultdict(lambda: [0.0] * len(GW_ACTIONS))
    returns: list[float] = []

    for ep in range(episodes):
        env = make_env()
        state = env.reset(seed=ep)
        done = False
        total = 0.0
        while not done:
            if rng.random() < epsilon:
                action = rng.choice(GW_ACTIONS)
            else:
                action = _argmax(q[state])
            nxt, reward, done, _info = env.step(action)
            total += reward
            best_next = 0.0 if done else max(q[nxt])
            q[state][action] += alpha * (reward + gamma * best_next - q[state][action])
            state = nxt
        returns.append(total)
    return dict(q), returns


def _argmax(values: list[float]) -> int:
    best = max(values)
    return values.index(best)


class GreedyQPolicy:
    """Acts greedily with respect to a learned Q-table (ties broken by action order)."""

    def __init__(self, q: dict) -> None:
        self._q = q

    def action(self, state) -> int:
        if state not in self._q:
            return GW_ACTIONS[0]
        return _argmax(self._q[state])
