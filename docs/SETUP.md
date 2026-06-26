# Setup

## The offline core (no GPU, no key, no Docker)

```bash
git clone https://github.com/victor-velazquez-ai/modern-physical-ai-engineer-companion
cd modern-physical-ai-engineer-companion
pip install -e ".[dev]"
make test     # 7 tests, all offline
make demo     # score the oracle vs random, in/out-of-distribution
```

That is everything you need to read along with the book's builds and run the toy `TabletopEnv`,
the policy interfaces, the eval harness, and (from cycle 3) the capstone pipeline — all on the
standard library.

## The real toolchain (GPU recommended)

The simulation and VLA paths follow the book's Appendix A. Install the extras you need:

```bash
pip install -e ".[sim]"   # mujoco + gymnasium + numpy  — the real physics
pip install -e ".[vla]"   # torch + lerobot             — real imitation / VLA policies
pip install -e ".[api]"   # anthropic + openai          — a cloud VLM as a high-level policy
pip install -e ".[all]"   # everything
```

Notes:
- **MuJoCo** installs from PyPI (`pip install mujoco`) and runs on CPU for small scenes; a GPU
  matters for parallel training.
- **LeRobot** provides the dataset format and the ACT / Diffusion Policy / VLA implementations the
  book's builds use. See its own docs for current install steps; the version moves.
- **Isaac Lab** (for the massively-parallel RL of Chapter 18) is a separate, heavier install —
  follow NVIDIA's current guide. It is not required for anything in this repo's offline core.
- For the bring-your-own-key VLA path (Chapter 36), copy `.env.example` to `.env` and set
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. With no key, that path falls back to the offline oracle
  so the pipeline still runs.

## Why the split

Real robot learning is GPU-bound and toolchain-heavy, and a companion that only ran on a
well-equipped workstation would teach nothing to most readers. So the *mechanisms* — perceive,
ground language, emit an action chunk, execute under receding horizon, score honestly over N
trials — are taught on a toy that runs anywhere, and the *production* paths are real, inspectable,
and clearly gated behind the toolchain that actually needs a GPU.
