"""Chapter 36 - Fine-tuning an open VLA (and a VLM as a high-level policy).

The REAL build fine-tunes an open VLA (OpenVLA / pi-0 via LeRobot) with LoRA on a simulated task
(the `[vla]` extra + a GPU). That is documented in the README. What runs HERE, offline, is the
adjacent idea of Chapter 36: a vision-language model used directly as a high-level *policy*. The
`VLMPolicy` describes the scene in text, asks a cloud VLM (Claude/GPT) for the next action, and -
with no API key - falls back to the oracle so this always runs.

    python chapters/ch36-vla-finetune/reproduce.py        (or: make ch36)

Set ANTHROPIC_API_KEY or OPENAI_API_KEY to drive the loop with a real model instead of the fallback.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from physicalai import TabletopEnv, VLMPolicy  # noqa: E402
from capstone.pipeline import RobotSystem  # noqa: E402


def main() -> None:
    policy = VLMPolicy(provider="anthropic")  # falls back to the oracle if no key / SDK
    system = RobotSystem(policy=policy)

    print("\nChapter 36 - a VLM as a high-level policy over the toy task (offline-safe):\n")
    successes = 0
    for seed in range(5):
        episode = system.run_episode(TabletopEnv(), seed=seed)
        successes += int(episode.success)
        print(f"  seed {seed}: {'SUCCESS' if episode.success else 'failure':<7} "
              f"in {episode.steps:>2} steps   ({' '.join(episode.trajectory)})")

    backend = "fell back to the offline oracle (no API key set)" if policy.used_fallback \
        else "drove the loop with a real cloud VLM"
    print(f"\n  Backend: {backend}.")
    print(f"  {successes}/5 episodes succeeded.\n")
    print("How this maps to the book:")
    print("  - The REAL Ch 36 build: fine-tune an open VLA (OpenVLA / pi-0) with LoRA - freeze the")
    print("    backbone, train small adapters, normalize the action space - on a LeRobot sim dataset")
    print("    (the [vla] extra + a GPU). See this folder's README.")
    print("  - The offline path here uses a cloud VLM AS the policy (no fine-tuning), which needs only")
    print("    an API key - and runs even without one, via the oracle fallback. Same Policy contract.\n")


if __name__ == "__main__":
    main()
