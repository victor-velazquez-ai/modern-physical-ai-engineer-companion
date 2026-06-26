"""physicalai — the companion library for *Modern Physical AI Engineer*.

The offline core (toy env, policy interfaces, eval harness) imports with no heavy dependencies
so the whole pedagogical surface runs and tests anywhere. The real MuJoCo / LeRobot / VLA paths
live behind the `[sim]`, `[vla]`, and `[api]` extras and are imported lazily where used.
"""

from physicalai.envs.tabletop import TabletopEnv, parse_instruction, ACTIONS, ACTION_NAMES
from physicalai.policies.base import Policy, ScriptedPolicy, RandomPolicy
from physicalai.eval.harness import Scorecard, evaluate, evaluate_split

__all__ = [
    "TabletopEnv", "parse_instruction", "ACTIONS", "ACTION_NAMES",
    "Policy", "ScriptedPolicy", "RandomPolicy",
    "Scorecard", "evaluate", "evaluate_split",
]
__version__ = "0.1.0"
