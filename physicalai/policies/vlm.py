"""VLMPolicy — a cloud vision-language model acting as a high-level policy (Chapter 36).

This is the bring-your-own-key path: given the scene and the instruction, ask a real VLM (Claude
or GPT) to pick the next primitive action. It is the toy analogue of "a VLM is the brain" — here
the model reasons in text over a described scene rather than over pixels, but the contract is the
same one the book's real VLA fine-tune produces: observation in, action out.

It is built to ALWAYS run. The SDK import is lazy (inside the call), the key is read from the
environment, and on any miss — no key, no SDK, an unparseable reply — it falls back to the
`ScriptedPolicy` oracle. So the capstone and tests stay green offline, and the same code calls a
real model the moment a reader supplies a key.
"""

from __future__ import annotations

import os

from physicalai.envs.tabletop import (
    EAST, NORTH, PICK, PLACE, SOUTH, WEST, parse_instruction,
)
from physicalai.policies.base import Policy, ScriptedPolicy

_ACTION_FROM_NAME = {"N": NORTH, "S": SOUTH, "W": WEST, "E": EAST, "PICK": PICK, "PLACE": PLACE}

_SYSTEM = (
    "You control a gripper on a small grid. Choose ONE next action to accomplish the instruction. "
    "Coordinates are (col, row); col grows East, row grows South. "
    "Reply with EXACTLY one token from: N, S, W, E, PICK, PLACE. No other text."
)


def describe(obs: dict) -> str:
    """A compact text rendering of the scene — the toy stand-in for the visual observation a real
    VLA would encode from pixels."""
    color, target = parse_instruction(obs["instruction"])
    blocks = ", ".join(f"{c} block at {p}" for c, p in obs["blocks"].items())
    targets = ", ".join(f"{n} at {p}" for n, p in obs["targets"].items())
    holding = obs["holding"] or "nothing"
    return (
        f"Instruction: {obs['instruction']}\n"
        f"Gripper at {obs['gripper']}, holding {holding}.\n"
        f"Blocks: {blocks}.\n"
        f"Targets: {targets}.\n"
        f"Goal: place the {color} block on the {target}."
    )


class VLMPolicy(Policy):
    """A VLM-as-policy with a guaranteed offline fallback to the oracle.

    Parameters
    ----------
    provider : "anthropic" | "openai"
    model    : model id (defaults to a current Claude / GPT model)
    """

    def __init__(self, provider: str = "anthropic", model: str | None = None) -> None:
        self.provider = provider
        self.model = model or ("claude-opus-4-8" if provider == "anthropic" else "gpt-4o")
        self._fallback = ScriptedPolicy()
        self.used_fallback = False

    def select_action(self, obs: dict) -> list[int]:
        action = self._ask_vlm(describe(obs))
        if action is None:
            self.used_fallback = True
            return self._fallback.select_action(obs)[:1]
        return [action]

    def _ask_vlm(self, prompt: str) -> int | None:
        key_var = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        if not os.environ.get(key_var):
            return None
        try:
            reply = self._call(prompt)
        except Exception:
            return None
        return _ACTION_FROM_NAME.get(reply.strip().upper().split()[0] if reply.strip() else "")

    def _call(self, prompt: str) -> str:
        if self.provider == "anthropic":
            import anthropic  # lazy: only needed on the real path
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=self.model, max_tokens=5, system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        import openai  # lazy
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=self.model, max_tokens=5,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
