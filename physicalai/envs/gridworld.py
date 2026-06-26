"""GridWorld — a minimal reach task for the reinforcement-learning build (Ch 17-18), offline.

A point agent on an N x N grid must reach a fixed goal cell. Reward is +1 on reaching the goal
(episode ends) and a small negative step cost otherwise, so the optimal policy is the shortest
path. It is the toy analogue of a control task: a clear reward, deterministic dynamics, and a
policy that *reward optimization* (not demonstrations) discovers — the thing the massively-parallel
RL of Chapter 18 does at scale in Isaac Lab, here in pure Python on a 5 x 5 grid.

State is just the agent cell (ax, ay) — small enough for a tabular value function.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

NORTH, SOUTH, WEST, EAST = 0, 1, 2, 3
GW_ACTIONS = (NORTH, SOUTH, WEST, EAST)


@dataclass
class GridWorld:
    grid_size: int = 5
    goal: tuple[int, int] = (4, 4)
    step_cost: float = 0.01
    max_steps: int = 50
    _agent: tuple[int, int] = (0, 0)
    _steps: int = 0

    def reset(self, seed: int = 0) -> tuple[int, int]:
        rng = random.Random(seed)
        # Start anywhere that is not the goal — the policy must learn to reach it from all cells.
        while True:
            cell = (rng.randrange(self.grid_size), rng.randrange(self.grid_size))
            if cell != self.goal:
                break
        self._agent = cell
        self._steps = 0
        return self._agent

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, dict]:
        self._steps += 1
        c, r = self._agent
        if action == NORTH:
            r = max(0, r - 1)
        elif action == SOUTH:
            r = min(self.grid_size - 1, r + 1)
        elif action == WEST:
            c = max(0, c - 1)
        elif action == EAST:
            c = min(self.grid_size - 1, c + 1)
        self._agent = (c, r)

        if self._agent == self.goal:
            return self._agent, 1.0, True, {"success": True, "steps": self._steps}
        done = self._steps >= self.max_steps
        return self._agent, -self.step_cost, done, {"success": False, "steps": self._steps}
