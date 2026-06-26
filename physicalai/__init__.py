"""physicalai — the companion library for *Modern Physical AI Engineer*.

The offline core (toy env, policy interfaces, eval harness) imports with no heavy dependencies
so the whole pedagogical surface runs and tests anywhere. The real MuJoCo / LeRobot / VLA paths
live behind the `[sim]`, `[vla]`, and `[api]` extras and are imported lazily where used.
"""

from physicalai.envs.tabletop import TabletopEnv, parse_instruction, ACTIONS, ACTION_NAMES
from physicalai.envs.gridworld import GridWorld, GW_ACTIONS
from physicalai.policies.base import Policy, ScriptedPolicy, RandomPolicy
from physicalai.policies.learned import BCPolicy, collect_expert_demos, featurize
from physicalai.eval.harness import Scorecard, evaluate, evaluate_split
from physicalai.rl.tabular import train_q, GreedyQPolicy

__all__ = [
    "TabletopEnv", "parse_instruction", "ACTIONS", "ACTION_NAMES",
    "GridWorld", "GW_ACTIONS",
    "Policy", "ScriptedPolicy", "RandomPolicy",
    "BCPolicy", "collect_expert_demos", "featurize",
    "Scorecard", "evaluate", "evaluate_split",
    "train_q", "GreedyQPolicy",
]
__version__ = "0.1.0"
