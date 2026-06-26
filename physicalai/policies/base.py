"""Policy interfaces for the toy TabletopEnv.

A `Policy` maps an observation to an *action chunk* (a short list of future actions), mirroring
the action-chunking design of Chapter 13 and the receding-horizon execution of Chapter 14: the
policy proposes a chunk, the harness executes a prefix, then re-queries from a fresh observation.

Two reference policies ship here:
  * `ScriptedPolicy` — a hand-coded oracle that parses the instruction and solves the toy task.
    It is the offline stand-in for "a competent trained policy," so the harness and capstone have
    something that actually works without a GPU. A *real* learned policy (a fine-tuned VLA, Ch 36)
    is a drop-in replacement implementing the same `select_action` contract.
  * `RandomPolicy` — the baseline floor, for contrast in the eval scorecards.
"""

from __future__ import annotations

import random

from physicalai.envs.tabletop import EAST, NORTH, PICK, PLACE, SOUTH, WEST, ACTIONS, parse_instruction


class Policy:
    """The contract every policy implements: optionally reset internal state, then map an
    observation to a list of discrete actions (an action chunk)."""

    def reset(self) -> None:  # noqa: B027 - intentionally optional
        pass

    def select_action(self, obs: dict) -> list[int]:
        raise NotImplementedError


def _moves(a: tuple[int, int], b: tuple[int, int]) -> list[int]:
    """Manhattan move sequence from cell `a` to cell `b` (columns first, then rows)."""
    c, r = a
    tc, tr = b
    out: list[int] = []
    out += [EAST] * (tc - c) if tc > c else [WEST] * (c - tc)
    out += [SOUTH] * (tr - r) if tr > r else [NORTH] * (r - tr)
    return out


class ScriptedPolicy(Policy):
    """An oracle for the toy task: go to the goal block, pick it, carry it to the goal target,
    place it. It re-plans the full remaining chunk on every call, so it composes correctly with
    receding-horizon execution."""

    def select_action(self, obs: dict) -> list[int]:
        color, target = parse_instruction(obs["instruction"])
        gripper = obs["gripper"]
        target_cell = obs["targets"][target]

        if obs["holding"] == color:
            return _moves(gripper, target_cell) + [PLACE]

        block_cell = obs["blocks"].get(color)
        if block_cell is None:
            return []  # goal block neither held nor on the grid (already placed) — nothing to do
        return _moves(gripper, block_cell) + [PICK]


class RandomPolicy(Policy):
    """A uniformly random baseline — the floor an honest scorecard compares against."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def select_action(self, obs: dict) -> list[int]:
        return [self._rng.choice(ACTIONS)]
