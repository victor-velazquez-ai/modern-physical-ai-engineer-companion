# Chapter 16 — Build: Imitation Policies in Simulation

Behavior cloning and covariate shift, made runnable on the toy `TabletopEnv`. No GPU, no key.

```bash
python chapters/ch16-imitation/reproduce.py      # or: make ch16
```

## What it does

1. Rolls out the **oracle** (`ScriptedPolicy`) to collect expert demonstrations — the imitation
   dataset of Chapter 11.
2. Trains a tiny pure-Python **behavior-cloning** policy (`BCPolicy`) that learns a
   feature → action map from those (observation, action) pairs.
3. Evaluates oracle, BC(40 demos), BC(5 demos), and a random baseline **honestly** — success rate
   over 60 held-out seeds, reported separately for in-distribution and out-of-distribution layouts
   (the Chapter 37/43 discipline).

## What you should see

```
  policy                         in-distribution      out-of-distribution
  Oracle (expert)                100.0%                100.0%
  BC (40 demos)                   ~85%                  ~33%
  BC (5 demos)                    ~43%                  ~10%
  Random baseline                 ~7%                    ~0%
```

Two lessons, as numbers:

- **Imitation is data-hungry.** BC(40) beats BC(5) — a behavior-cloning policy only knows what it
  was shown (Chapters 11 and 2).
- **Covariate shift is real.** BC does far worse out-of-distribution than in-distribution, because
  it *memorized* concrete situations and the larger OOD grid presents ones it never saw. The
  oracle, which *computes* the plan, has no such gap — so the gap is a property of the *learned*
  policy, which is exactly why the harness reports the two splits separately. A single headline
  number would hide it.

## How it maps to the book

- The `BCPolicy` is behavior cloning (Ch 11): supervised learning on state-action pairs.
- `select_action` returns an action **chunk**; the harness executes a prefix and re-queries — the
  receding-horizon execution of Chapters 13–14. (The toy is disturbance-free, so the chunk length
  does not change success here; it is the knob that matters once the world pushes back.)
- The real version of this build trains **ACT** and a **Diffusion Policy** in LeRobot + MuJoCo
  (the `[sim]`/`[vla]` extras, a GPU). The toy teaches the same mechanism — collect, clone,
  evaluate honestly — at a scale that runs anywhere.
