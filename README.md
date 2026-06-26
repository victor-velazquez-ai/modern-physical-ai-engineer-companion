# Modern Physical AI Engineer — Companion Repository

The runnable companion to the book *Modern Physical AI Engineer: The Theory and Technology of
Robot Foundation Models*. Everything the book describes as a **build** lives here as code you can
read and run.

## The principle: the chapter is the front door

This repo has exactly **two axes**, and no more — that is the whole navigation rule.

- **One learning axis — per-chapter folders** (`chapters/chNN-*/`). Finish a chapter, open its
  folder, run its `reproduce.py`. The book and the repo share one structure.
- **One reuse axis — the `physicalai` library.** The shared machinery (the environment, the
  policy interfaces, the evaluation harness) lives in one importable package so the per-chapter
  code stays short and the real logic is not copy-pasted.

## Offline core vs. the GPU toolchain (read this first)

Real Physical AI training needs MuJoCo, LeRobot, PyTorch, and a GPU (the toolchain of the book's
Appendix A). You **cannot** run that on a laptop with no card, so this repo is deliberately split:

- **The offline core runs anywhere, with zero heavy dependencies.** A tiny, deterministic,
  language-conditioned toy — `TabletopEnv` — plus the policy interfaces and the eval harness, all
  pure standard-library Python. The whole pedagogical *shape* (perceive a scene + an instruction →
  emit an action chunk → execute → score success over N trials) executes and is unit-tested with
  no GPU and no API key. This is where you learn the mechanisms.
- **The real-toolchain paths are implemented and clearly marked.** MuJoCo/LeRobot/VLA code is
  behind the `[sim]`, `[vla]`, and `[api]` extras with guarded imports, so the structure is real
  and inspectable and runs when you supply the toolchain and a GPU.

The toy is not a pretend result — it is an honest teaching stand-in that lets the library, the
harness, and the capstone actually run end to end. Where a real GPU/sim is required, the code says
so plainly.

## Master map

| Chapter | Folder | What it runs | `physicalai` module |
|---|---|---|---|
| Core | `physicalai/` | the toy env, policy interfaces, eval harness (offline) | `envs`, `policies`, `eval` |
| 16 — Imitation build | `chapters/ch16-imitation/` | behavior cloning + the covariate-shift gap (ID vs OOD) | `physicalai.policies` |
| 20 — RL build | `chapters/ch20-rl/` | tabular Q-learning control on GridWorld (a learning curve) | `physicalai.rl` |
| 36 — Fine-tune a VLA | `chapters/ch36-vla-finetune/` | LoRA structure + a BYO-key VLM-as-policy path *(cycle 3)* | `physicalai.policies` |
| 42 — Capstone | `capstone/` | perception → policy → execution → eval, end to end *(cycle 3)* | all |

*(Folders marked with a cycle are added in the noted build cycle; the core library, the imitation
and RL builds, and their tests are complete now — `make test` is green offline.)*

## Quickstart

```bash
pip install -e ".[dev]"      # the offline core needs nothing beyond the standard library
make test                    # run the offline test suite (no GPU, no key)
make demo                    # score the oracle vs. a random baseline, in- and out-of-distribution
```

To run the real simulation and VLA paths, install the extras and follow `docs/SETUP.md`:

```bash
pip install -e ".[all]"      # mujoco, lerobot, torch, anthropic, openai — GPU recommended
cp .env.example .env         # add ANTHROPIC_API_KEY / OPENAI_API_KEY for the BYO-key VLA path
```

## Honest evaluation, by construction

Every result the harness prints is a **success rate over N fixed-seed trials**, reported
**separately for in-distribution and out-of-distribution** layouts — the discipline of the book's
Chapters 37 and 43. There are no single-rollout demos here.

## License

MIT (see `LICENSE`). The book's research and figures are the book's; this code is yours to use.
