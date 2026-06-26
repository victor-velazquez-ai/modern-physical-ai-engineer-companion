"""Run the capstone on a single instruction, end to end, offline.

    python -m capstone.run "put the red block on the bowl"      (or: make capstone)

Prints the perceived scene, the executed trajectory, the outcome, and then a small honest
scorecard over several seeds. No GPU, no key (the default policy is the offline oracle; swap in
BCPolicy or VLMPolicy to see the learned / cloud-VLM brain).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from physicalai import TabletopEnv, parse_instruction  # noqa: E402
from physicalai.policies.vlm import describe  # noqa: E402
from capstone.pipeline import RobotSystem  # noqa: E402

DEFAULT = "put the red block on the bowl"


def main(argv: list[str]) -> None:
    instruction = argv[1] if len(argv) > 1 else DEFAULT
    try:
        color, target = parse_instruction(instruction)
    except Exception:
        print(f'Could not parse instruction; expected "put the <color> block on the <target>".')
        print(f'Using the default: "{DEFAULT}"')
        instruction = DEFAULT
        color, target = parse_instruction(instruction)

    system = RobotSystem()  # offline oracle by default
    env = TabletopEnv()

    # One narrated episode for the requested instruction.
    obs = env.reset(seed=0, goal_color=color, goal_target=target)
    print("\n=== CAPSTONE: a Physical-AI system in simulation ===\n")
    print("PERCEPTION (the scene the policy sees):")
    print("  " + describe(obs).replace("\n", "\n  "))
    episode = system.run_episode(env, seed=0, goal_color=color, goal_target=target)
    print("\nEXECUTION (the action chunk, executed under receding horizon):")
    print("  " + " -> ".join(episode.trajectory))
    print(f"\nOUTCOME: {'SUCCESS' if episode.success else 'failure'} in {episode.steps} steps.")

    # Honest evaluation over several seeds (never trust one rollout, Ch 37/43).
    card = system.evaluate(lambda: TabletopEnv(), list(range(50)))
    print(f"\nHONEST EVAL over 50 seeds:  {card}")
    print("\n(The default brain here is the offline oracle. Swap in BCPolicy to use the learned")
    print(" policy of Ch 16, or VLMPolicy with an API key to use a real cloud VLM as the brain.)\n")


if __name__ == "__main__":
    main(sys.argv)
