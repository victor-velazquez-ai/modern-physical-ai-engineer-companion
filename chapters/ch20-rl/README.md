# Chapter 20 — Build: a Reinforcement-Learning Control Policy

Reward-driven control in miniature, on a toy `GridWorld`. No GPU, no demonstrations.

```bash
python chapters/ch20-rl/reproduce.py      # or: make ch20
```

## What it does

Trains a **tabular Q-learner** to reach a goal cell on an 8×8 grid by *optimizing reward* — no
expert, no demonstrations, just trial and error with epsilon-greedy exploration. It prints the
learning curve (mean return over successive windows) and then evaluates the learned greedy policy.

## What you should see

The mean return **rises** as the agent learns — from wandering and timing out (a clearly negative
or low return) to reaching the goal directly:

```
  episodes        mean return
      0- 750       ~+0.47
   1500-2250       ~+0.57
   3750-4500       ~+0.63
  ...
  Learned greedy policy over 100 held-out starts: 100/100 reach the goal, mean ~6.9 steps.
```

Two things worth noticing:

- The **greedy** policy (≈ 7 steps, 100% success) is *better* than the training-return plateau,
  because the training number includes the 25% random exploratory moves. The policy you deploy is
  the greedy one; the exploration was the price of discovering it.
- No demonstrations were used. Reward optimization *discovered* a near-shortest-path policy — the
  lever reinforcement learning adds over imitation (Chapter 17): it can exceed any demonstrator, at
  the price of needing a reward and many trials.

## How it maps to the book

- This is the pure-Python analogue of the **PPO-in-Isaac-Lab** build of Chapters 17–18: same idea
  (reward is the specification; trial-and-error finds a policy), a few thousand episodes on 64
  states instead of billions of steps across thousands of parallel environments on a GPU.
- The reward shaping (the small step cost) is doing real work — it is what makes the *shortest*
  path optimal rather than merely *a* path. Mis-shape it and the agent games it (Chapter 17); that
  the step cost cleanly yields direct paths is the benign case.
- The real build trains a locomotion or reach policy in Isaac Lab with domain randomization
  (the `[sim]` extra, a GPU). The mechanism — and the honest evaluation — is the same.
