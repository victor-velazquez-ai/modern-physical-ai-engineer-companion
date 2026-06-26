"""The capstone system (Chapter 42): perception -> policy -> execution -> honest evaluation.

`RobotSystem` is the assembled end-to-end pipeline. It is deliberately thin: all the intelligence
lives in the pluggable policy (ScriptedPolicy, BCPolicy, or the VLMPolicy brain) and in the
TabletopEnv, so the capstone is the *assembly* — wiring perception to a policy to execution under
receding-horizon control, and scoring it honestly — exactly as the book frames it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from physicalai.envs.tabletop import ACTION_NAMES, TabletopEnv
from physicalai.eval.harness import Scorecard, evaluate
from physicalai.policies.base import Policy, ScriptedPolicy


@dataclass
class Episode:
    success: bool
    steps: int
    trajectory: list[str] = field(default_factory=list)


class RobotSystem:
    """Perception -> policy -> execution loop over TabletopEnv, with receding-horizon execution."""

    def __init__(self, policy: Policy | None = None, exec_horizon: int = 4) -> None:
        self.policy = policy or ScriptedPolicy()
        self.exec_horizon = exec_horizon

    def run_episode(
        self,
        env: TabletopEnv,
        seed: int = 0,
        goal_color: str | None = None,
        goal_target: str | None = None,
    ) -> Episode:
        obs = env.reset(seed=seed, goal_color=goal_color, goal_target=goal_target)
        self.policy.reset()
        traj: list[str] = []
        done = False
        while not done:
            chunk = self.policy.select_action(obs)        # perceive + decide
            if not chunk:
                break
            for action in chunk[: self.exec_horizon]:     # execute a prefix (receding horizon)
                obs, _r, done, info = env.step(action)
                traj.append(ACTION_NAMES[action])
                if done:
                    break
        return Episode(success=env.is_success(), steps=len(traj), trajectory=traj)

    def evaluate(self, make_env: Callable[[], TabletopEnv], seeds: list[int]) -> Scorecard:
        """Score the assembled system honestly: success over N seeds (the Ch 37/43 discipline)."""
        return evaluate(make_env, self.policy, seeds, self.exec_horizon, label="capstone")
