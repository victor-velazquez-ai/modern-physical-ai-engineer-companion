"""A tiny, pure-Python behavior-cloning policy for the toy TabletopEnv.

This is the offline analogue of the imitation learning of Chapters 11-13: collect expert
demonstrations (here, from the `ScriptedPolicy` oracle), reduce each observation to a small
feature, and learn a feature -> action map by majority vote. It is *behavior cloning* in
miniature — supervised learning on state-action pairs — and it exhibits the same pathology the
book warns about: it does well where its features match the training data and *drifts* where they
do not (covariate shift), which the in/out-of-distribution scorecards expose.

It deliberately uses no torch and no GPU; the point is the *mechanism*, not the model class.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from physicalai.envs.tabletop import ACTIONS, parse_instruction
from physicalai.policies.base import Policy, ScriptedPolicy


def featurize(obs: dict) -> tuple:
    """Reduce an observation to a feature for behavior cloning: the *raw* delta from the gripper to
    whatever currently matters — the goal block when not yet holding it, the target once holding —
    plus the holding flag.

    Two deliberate teaching choices. (1) The feature is *context-aware* (only the relevant delta),
    which matches the oracle's decision structure, so a handful of demos cover the in-distribution
    workspace and BC works there. (2) The deltas are *raw*, not signs: the oracle's action actually
    depends only on the signs, so a sign feature would generalize perfectly and hide the lesson.
    Raw deltas make BC *memorize* concrete positions, so it does well on the small grid it trained
    on and *drifts* on the larger out-of-distribution grid whose bigger deltas it never saw — the
    generalization gap that Chapter 11 calls covariate shift, made measurable by the ID/OOD split."""
    color, target = parse_instruction(obs["instruction"])
    gx, gy = obs["gripper"]

    if obs["holding"] == color:
        tx, ty = obs["targets"][target]
        return (1, 0, 0, tx - gx, ty - gy)

    block = obs["blocks"].get(color)
    if block is None:
        return (0, 0, 0, 0, 0)
    return (0, block[0] - gx, block[1] - gy, 0, 0)


class BCPolicy(Policy):
    """Behavior cloning by feature-majority-vote. `select_action` returns a length-1 chunk; the
    harness's `exec_horizon` therefore controls re-query frequency (receding horizon)."""

    def __init__(self) -> None:
        self._table: dict[tuple, int] = {}
        self._default = ACTIONS[0]

    def fit(self, demos: list[tuple[dict, int]]) -> "BCPolicy":
        """Train on (observation, expert action) pairs."""
        votes: dict[tuple, Counter] = defaultdict(Counter)
        for obs, action in demos:
            votes[featurize(obs)][action] += 1
        self._table = {feat: counter.most_common(1)[0][0] for feat, counter in votes.items()}
        if demos:
            overall = Counter(a for _, a in demos)
            self._default = overall.most_common(1)[0][0]
        return self

    def select_action(self, obs: dict) -> list[int]:
        return [self._table.get(featurize(obs), self._default)]


def collect_expert_demos(make_env, seeds: list[int], exec_horizon: int = 1) -> list[tuple[dict, int]]:
    """Roll out the `ScriptedPolicy` oracle and record (observation, action) pairs — the expert
    dataset a behavior-cloning policy learns from."""
    expert = ScriptedPolicy()
    demos: list[tuple[dict, int]] = []
    for seed in seeds:
        env = make_env()
        obs = env.reset(seed)
        done = False
        while not done:
            chunk = expert.select_action(obs)
            if not chunk:
                break
            for action in chunk[:exec_horizon]:
                demos.append((obs, action))
                obs, _r, done, _info = env.step(action)
                if done:
                    break
    return demos
