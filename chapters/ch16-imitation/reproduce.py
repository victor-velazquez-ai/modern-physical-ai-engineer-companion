"""Chapter 16 - Imitation in simulation: behavior cloning and covariate shift, on the toy env.

Offline, no GPU, no key. We collect expert demonstrations from the oracle, train a tiny
behavior-cloning policy, and evaluate it honestly (success over N held-out seeds, in- and
out-of-distribution). Two lessons appear as numbers:

  1. MORE DATA HELPS: BC trained on 40 demos beats BC trained on 5 (Ch 11/2 - imitation is
     data-hungry; it only knows what it was shown).
  2. COVARIATE SHIFT IS REAL: BC does far worse out-of-distribution than in-distribution, because
     it memorized concrete situations and the larger OOD grid presents ones it never saw (Ch 11).
     The oracle, which *computes* the plan, has no such gap - the gap is a property of *learned*
     policies, which is exactly why the harness reports the two splits separately.

    python chapters/ch16-imitation/reproduce.py     (or: make ch16)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from physicalai import (  # noqa: E402
    BCPolicy, RandomPolicy, ScriptedPolicy, TabletopEnv, collect_expert_demos, evaluate_split,
)

TRAIN_SMALL = list(range(5))
TRAIN_LARGE = list(range(40))
EVAL = list(range(200, 260))


GRID = 3  # train + in-distribution on a small grid; OOD evaluation uses a larger one (GRID + 2)


def main() -> None:
    def make_id_env():
        return TabletopEnv(grid_size=GRID)

    bc_small = BCPolicy().fit(collect_expert_demos(make_id_env, TRAIN_SMALL))
    bc_large = BCPolicy().fit(collect_expert_demos(make_id_env, TRAIN_LARGE))

    print(f"\nBehavior cloning on the toy tabletop task (success over {len(EVAL)} held-out seeds):\n")
    print("  policy                         in-distribution      out-of-distribution")
    print("  -----------------------------  -------------------  -------------------")
    rows = [
        ("Oracle (expert)", ScriptedPolicy()),
        (f"BC ({len(TRAIN_LARGE)} demos)", bc_large),
        (f"BC ({len(TRAIN_SMALL)} demos)", bc_small),
        ("Random baseline", RandomPolicy()),
    ]
    for name, policy in rows:
        id_card, ood_card = evaluate_split(policy, EVAL, grid_size=GRID, label=name)
        print(f"  {name:<29}  {id_card.success_rate:6.1%} ({id_card.successes:>2}/{id_card.n_episodes})"
              f"        {ood_card.success_rate:6.1%} ({ood_card.successes:>2}/{ood_card.n_episodes})")

    print("\nRead it:")
    print("  - BC(40) beats BC(5): imitation is data-hungry - it only knows what it was shown.")
    print("  - BC's in-dist >> out-of-dist: covariate shift (Ch 11). It memorized situations; the")
    print("    bigger OOD grid shows it new ones, and it drifts. The oracle has no such gap.")
    print("  - The harness reports BOTH splits because a single number would hide exactly this.\n")

    # The action-chunk / receding-horizon knob (Ch 13/14): the harness executes only `exec_horizon`
    # actions of each proposed chunk before re-querying. In this disturbance-free toy it does not
    # change success, but it is the knob that trades reactivity against horizon under disturbances.
    print("  (Action chunking: the harness re-queries every `exec_horizon` steps - the receding")
    print("   horizon of Ch 14. Disturbance-free here, so success is unchanged; it is the dial that")
    print("   matters once the world pushes back.)\n")


if __name__ == "__main__":
    main()
