"""TabletopEnv — a tiny, deterministic, language-conditioned manipulation environment.

This is the *offline core* of the companion: a pure-Python toy that runs with no MuJoCo, no
GPU, and no dependencies, so the library, the eval harness, and the capstone pipeline actually
execute and can be unit-tested anywhere. It is a teaching stand-in for the real MuJoCo/LeRobot
task of the book's builds (Ch 16, 36, 42) — same *shape* (perceive a scene + an instruction,
emit actions, execute, check success), a fraction of the machinery.

The task: a gripper on an N x N grid must "put the <color> block on the <target>". Actions are
discrete primitives (move N/S/E/W, pick, place). Success is the goal block resting on the goal
target. Everything is deterministic given a seed, which is what makes honest evaluation
(Ch 37/43) possible: the same seed is the same episode every time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Discrete action vocabulary (the toy analogue of a tokenized action space, Ch 12/24).
NORTH, SOUTH, WEST, EAST, PICK, PLACE = 0, 1, 2, 3, 4, 5
ACTIONS = (NORTH, SOUTH, WEST, EAST, PICK, PLACE)
ACTION_NAMES = {NORTH: "N", SOUTH: "S", WEST: "W", EAST: "E", PICK: "pick", PLACE: "place"}

# Colors the policy is "trained" on in-distribution; OOD introduces an unseen color + larger grid.
TRAIN_COLORS = ("red", "blue", "green")
OOD_COLORS = ("yellow", "purple")
TARGETS = ("bowl", "plate")


@dataclass
class TabletopState:
    grid_size: int
    gripper: tuple[int, int]
    holding: str | None
    blocks: dict[str, tuple[int, int]]
    targets: dict[str, tuple[int, int]]
    goal_color: str
    goal_target: str
    steps: int = 0
    ood: bool = False


@dataclass
class TabletopEnv:
    """A deterministic language-conditioned tabletop manipulation toy.

    Parameters mirror the choices a real task makes: grid size (workspace), max_steps (the
    episode horizon), and an `ood` flag that produces a harder, out-of-distribution layout
    (unseen color, larger grid, extra distractor) for the in/out-of-distribution eval split.
    """

    grid_size: int = 5
    max_steps: int = 60
    ood: bool = False
    _state: TabletopState | None = field(default=None, repr=False)

    # --- core gym-like API -------------------------------------------------------------
    def reset(self, seed: int = 0, goal_color: str | None = None,
              goal_target: str | None = None) -> dict:
        """Reset to a deterministic layout. Optionally force the goal (color, target) — used by the
        capstone to honor a user-supplied instruction; the requested block is guaranteed to exist."""
        rng = random.Random(seed)
        size = self.grid_size if not self.ood else self.grid_size + 2
        colors = list(TRAIN_COLORS if not self.ood else TRAIN_COLORS + (OOD_COLORS[0],))
        if goal_color and goal_color not in colors:
            colors.append(goal_color)
        n_blocks = len(colors) if self.ood else 2

        cells = [(c, r) for c in range(size) for r in range(size)]
        rng.shuffle(cells)
        pos = iter(cells)

        gripper = next(pos)
        chosen = colors[:n_blocks]
        if goal_color and goal_color not in chosen:
            chosen[-1] = goal_color  # ensure the requested block is on the table
        blocks = {color: next(pos) for color in chosen}
        targets = {name: next(pos) for name in TARGETS}

        if goal_target and goal_target not in TARGETS:
            raise ValueError(f"unknown target {goal_target!r}; choose from {TARGETS}")
        goal_color = goal_color or rng.choice(chosen)
        goal_target = goal_target or rng.choice(TARGETS)

        self._state = TabletopState(
            grid_size=size, gripper=gripper, holding=None, blocks=blocks,
            targets=targets, goal_color=goal_color, goal_target=goal_target, ood=self.ood,
        )
        return self._obs()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        s = self._require_state()
        s.steps += 1
        c, r = s.gripper

        if action == NORTH:
            s.gripper = (c, max(0, r - 1))
        elif action == SOUTH:
            s.gripper = (c, min(s.grid_size - 1, r + 1))
        elif action == WEST:
            s.gripper = (max(0, c - 1), r)
        elif action == EAST:
            s.gripper = (min(s.grid_size - 1, c + 1), r)
        elif action == PICK:
            if s.holding is None:
                for color, p in list(s.blocks.items()):
                    if p == s.gripper:
                        s.holding = color
                        del s.blocks[color]
                        break
        elif action == PLACE:
            if s.holding is not None:
                placed = s.holding
                s.blocks[placed] = s.gripper          # drop it here
                s.holding = None
                if placed == s.goal_color and s.gripper == s.targets[s.goal_target]:
                    return self._obs(), 1.0, True, {"success": True, "steps": s.steps}

        done = s.steps >= self.max_steps
        return self._obs(), 0.0, done, {"success": False, "steps": s.steps}

    # --- helpers -----------------------------------------------------------------------
    def is_success(self) -> bool:
        s = self._require_state()
        return s.blocks.get(s.goal_color) == s.targets[s.goal_target]

    @property
    def instruction(self) -> str:
        s = self._require_state()
        return f"put the {s.goal_color} block on the {s.goal_target}"

    def _obs(self) -> dict:
        s = self._require_state()
        return {
            "instruction": self.instruction,
            "grid_size": s.grid_size,
            "gripper": s.gripper,
            "holding": s.holding,
            "blocks": dict(s.blocks),
            "targets": dict(s.targets),
            # The parsed goal is exposed for convenience; a learned policy is expected to read
            # it from `instruction` (grounding language), not to lean on this shortcut.
            "goal": {"color": s.goal_color, "target": s.goal_target},
            "ood": s.ood,
        }

    def _require_state(self) -> TabletopState:
        if self._state is None:
            raise RuntimeError("call reset() before using the environment")
        return self._state


def parse_instruction(instruction: str) -> tuple[str, str]:
    """Parse 'put the <color> block on the <target>' -> (color, target). The toy analogue of
    the language grounding a real VLA must do."""
    toks = instruction.split()
    return toks[2], toks[-1]
