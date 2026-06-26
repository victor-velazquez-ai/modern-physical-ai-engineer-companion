# Chapter 36 — Build: Fine-Tuning an Open VLA (and a VLM as policy)

```bash
python chapters/ch36-vla-finetune/reproduce.py      # or: make ch36   (offline-safe)
```

This chapter has two halves: the **real** fine-tuning build (needs a GPU), and an **offline** path
(a cloud VLM as a high-level policy) that runs anywhere and falls back to the oracle with no key.

## The real build (the `[vla]` extra + a GPU)

Fine-tuning an open VLA is the "specialize" stage of the lifecycle (Chapters 28 and 36):

1. **Choose a checkpoint — and read its license first.** OpenVLA (MIT code, Llama-2-encumbered
   weights) or π0 via openpi (Apache-2.0) are the natural open choices. Avoid the checkpoints
   flagged non-commercial in the book (e.g. some GR00T weights) for a commercial project.
2. **Get a LeRobot-format dataset** for the simulated task (a few hundred demonstrations).
3. **LoRA fine-tune**: freeze the large pretrained backbone, train small low-rank adapters — what
   makes adapting a 7-billion-parameter model feasible on a single GPU — and set the action-space
   normalization to match the dataset.
4. **Evaluate in simulation**, honestly: success rate over N trials, in- vs out-of-distribution,
   and crucially *also* the zero-shot base checkpoint, so you measure what fine-tuning bought.

```python
# Illustrative (real path; needs torch + lerobot + a GPU)
from lerobot.common.policies.factory import make_policy
policy = make_policy("openvla", checkpoint="openvla/openvla-7b")
policy.add_adapter(lora_rank=32)            # freeze backbone, train adapters
policy.set_action_stats(ds.action_mean, ds.action_std)
# ... LoRA fine-tune loop on the LeRobot dataset, then evaluate with the harness ...
```

## The offline path (runs anywhere)

`reproduce.py` runs `VLMPolicy` — a cloud vision-language model used directly as a *high-level
policy* over the toy `TabletopEnv`. It describes the scene in text, asks the model for the next
primitive action, and executes it. With no API key it falls back to the oracle, so the script
always runs and stays green in CI.

```bash
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY — then it drives the loop with a real model
python chapters/ch36-vla-finetune/reproduce.py
```

## Honest about what runs where

The toy VLM path is the *idea* (a VLM as the brain, the `Policy` contract: observation in, action
out), not the real thing — it reasons over a text description, not pixels, and it does not
fine-tune anything. The real LoRA fine-tune of a 7B VLA on MuJoCo/LeRobot data needs the toolchain
of Appendix A and a GPU. Both teach the same contract; only one needs the hardware.
