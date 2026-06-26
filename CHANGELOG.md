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

## Cycle 2 — the imitation and RL builds (offline)
- **ch16 — imitation** (`chapters/ch16-imitation/`, `make ch16`): a pure-Python `BCPolicy`
  (behavior cloning by feature-majority-vote over expert demos) added to `physicalai.policies`.
  `reproduce.py` trains BC on 5 vs 40 demos and evaluates honestly (ID vs OOD over 60 held-out
  seeds), showing two lessons as numbers: **imitation is data-hungry** (BC(40) ~85% beats BC(5)
  ~43% in-distribution) and **covariate shift is real** (BC ~85% in-distribution drops to ~33%
  out-of-distribution; the oracle has no such gap). Ch 11/37 made measurable.
- **ch20 — reinforcement learning** (`chapters/ch20-rl/`, `make ch20`): a `GridWorld` reach env
  (`physicalai.envs`) + tabular Q-learning (`physicalai.rl`). `reproduce.py` prints a rising
  learning curve and a near-optimal greedy policy (100/100 reach the goal, ~7 steps) — reward
  optimization discovering a policy with no demonstrations (Ch 17/18), the toy analogue of
  PPO-in-Isaac-Lab.
- **13 offline tests pass** (`make test`): + BC learns and beats random in-distribution, BC shows
  the ID>OOD covariate-shift gap, more demos help, featurize is context-aware; GridWorld is
  deterministic and rewards the goal; Q-learning reaches the goal nearly always and its curve rises.

## Cycle 3 — the VLA path and the capstone (offline-safe)
- **`physicalai.policies.VLMPolicy`** — a cloud vision-language model used as a high-level policy
  (Ch 36): describes the scene in text, asks Claude/GPT for the next action (LAZY SDK import, key
  from env), parses the reply. On no key / no SDK / unparseable reply it **falls back to the
  oracle**, so it always runs and CI stays offline-green.
- **ch36 — VLA fine-tune** (`chapters/ch36-vla-finetune/`, `make ch36`): README documents the real
  LoRA-on-OpenVLA/π0 + LeRobot path (the `[vla]` extra, a GPU); `reproduce.py` runs the offline-safe
  `VLMPolicy` over a few seeds (oracle fallback without a key), honest about what runs where.
- **The capstone** (`capstone/`, `make capstone`): `RobotSystem` = perception → pluggable policy →
  execution (receding horizon) → honest eval. `python -m capstone.run "put the blue block on the
  plate"` narrates perception + the executed trajectory + the outcome, then scores 50 seeds.
  `TabletopEnv.reset` gained a goal override so a user instruction is honored.
- **18 offline tests pass** (`make test`): + VLMPolicy falls back and solves with no key, describe()
  renders the scene, the capstone runs end to end, honors a requested goal, and evaluates honestly.
- Full offline surface: `make test / demo / ch16 / ch20 / ch36 / capstone` — no GPU, no key.
