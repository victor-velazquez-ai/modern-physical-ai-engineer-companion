"""A 10-second offline demo: score the oracle and a random baseline on the toy tabletop task,
in- and out-of-distribution. No GPU, no API key.

    python tools/demo.py     (or: make demo)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from physicalai import RandomPolicy, ScriptedPolicy, evaluate_split  # noqa: E402

SEEDS = list(range(50))


def main() -> None:
    print(f"\nScoring policies on the toy language-conditioned tabletop task")
    print(f"(success rate over {len(SEEDS)} fixed-seed trials, the Ch 37/43 discipline):\n")
    for name, policy in [("Oracle (scripted)", ScriptedPolicy()), ("Random baseline", RandomPolicy())]:
        id_card, ood_card = evaluate_split(policy, SEEDS, label=name)
        print(f"  {id_card}")
        print(f"  {ood_card}")
        print()
    print("Read it honestly: the oracle *computes* the plan, so it solves both splits. The random")
    print("baseline's in- vs out-of-distribution gap is the kind of number that actually matters")
    print("for a *learned* policy - which is exactly what the harness is built to expose.\n")


if __name__ == "__main__":
    main()
