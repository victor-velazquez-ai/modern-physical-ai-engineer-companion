# Changelog — Modern Physical AI Engineer companion

## Scaffold (offline core)
- Repository established on the **"chapter is the front door"** principle: one learning axis
  (per-chapter folders) + one reuse axis (the `physicalai` library), with a clear **offline-core
  vs. GPU-toolchain** split so the pedagogical surface runs anywhere.
- Infra: `pyproject.toml` (`physicalai`, extras `[sim]`/`[vla]`/`[api]`/`[dev]`), `Makefile`
  (`make test` / `make demo` / `make capstone`), `.gitignore`, `.env.example`, `LICENSE` (MIT),
  `docs/SETUP.md`, master-map `README.md`.
- **`physicalai` library core (pure standard library, no GPU, no key):**
  - `physicalai.envs.tabletop.TabletopEnv` — a deterministic, language-conditioned toy
    manipulation task (grid + colored blocks + targets + a gripper; discrete move/pick/place
    actions; sparse success). Seeded → reproducible episodes; an `ood` flag for the in/out-of-
    distribution split. `parse_instruction` is the toy language-grounding step.
  - `physicalai.policies.base` — the `Policy` contract (`select_action(obs) -> action chunk`),
    a `ScriptedPolicy` oracle (the offline stand-in for a competent trained policy), and a
    `RandomPolicy` floor. Action chunks + receding-horizon execution mirror Ch 13/14.
  - `physicalai.eval.harness` — `evaluate` / `evaluate_split` returning a `Scorecard`
    (success rate over N fixed seeds, in/out-of-distribution reported separately): the honest-
    evaluation discipline of Ch 37/43, as code.
- **7 offline tests pass** (`make test`): env determinism, instruction parsing, manual + scripted
  success, oracle solves in- and out-of-distribution, random is a weak floor, scorecard readable.
- `tools/demo.py` (`make demo`) scores the oracle vs. random across both splits, end to end, offline.

### Coming next
- Cycle 2 — `chapters/ch16-imitation/` (behavior cloning + action chunking on the toy env),
  `chapters/ch20-rl/` (a tiny policy-gradient control demo).
- Cycle 3 — `chapters/ch36-vla-finetune/` (LoRA structure + a BYO-key VLM-as-policy path) and the
  `capstone/` (perception → policy → execution → eval, `make capstone`, offline end-to-end).
