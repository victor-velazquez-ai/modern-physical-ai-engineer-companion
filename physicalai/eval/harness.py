"""The evaluation harness — honest evaluation, the Chapter 37 / 43 discipline made code.

A result is *success rate over N trials*, never a single rollout, and the in-distribution and
out-of-distribution numbers are reported *separately* because a policy can be strong on one and
collapse on the other. The harness fixes the seeds up front (the denominator is decided before we
look), runs each episode deterministically, and returns a `Scorecard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from physicalai.envs.tabletop import TabletopEnv
from physicalai.policies.base import Policy


@dataclass
class Scorecard:
    label: str
    n_episodes: int
    successes: int
    mean_steps: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.n_episodes if self.n_episodes else 0.0

    def __str__(self) -> str:
        return (
            f"{self.label:<22} success {self.success_rate:6.1%}  "
            f"({self.successes}/{self.n_episodes})   mean steps {self.mean_steps:5.1f}"
        )


def evaluate(
    make_env: Callable[[], TabletopEnv],
    policy: Policy,
    seeds: list[int],
    exec_horizon: int = 4,
    label: str = "policy",
) -> Scorecard:
    """Run `policy` on a fresh `make_env()` for each seed; execute an `exec_horizon` prefix of
    each proposed action chunk, then re-query (receding horizon). Score success over the seeds."""
    successes = 0
    total_steps = 0
    for seed in seeds:
        env = make_env()
        obs = env.reset(seed)
        policy.reset()
        done = False
        steps = 0
        while not done:
            chunk = policy.select_action(obs)
            if not chunk:
                break  # policy declines / empty plan — end the episode
            for action in chunk[:exec_horizon]:
                obs, _reward, done, info = env.step(action)
                steps += 1
                if done:
                    break
        successes += 1 if env.is_success() else 0
        total_steps += steps
    mean_steps = total_steps / len(seeds) if seeds else 0.0
    return Scorecard(label=label, n_episodes=len(seeds), successes=successes, mean_steps=mean_steps)


def evaluate_split(
    policy: Policy,
    seeds: list[int],
    grid_size: int = 5,
    exec_horizon: int = 4,
    label: str = "policy",
) -> tuple[Scorecard, Scorecard]:
    """Evaluate the same policy on in-distribution and out-of-distribution layouts, returning
    both scorecards. The gap between them is the honest generalization signal (Ch 37)."""
    id_card = evaluate(
        lambda: TabletopEnv(grid_size=grid_size, ood=False),
        policy, seeds, exec_horizon, label=f"{label} [in-dist]",
    )
    ood_card = evaluate(
        lambda: TabletopEnv(grid_size=grid_size, ood=True),
        policy, seeds, exec_horizon, label=f"{label} [out-of-dist]",
    )
    return id_card, ood_card
