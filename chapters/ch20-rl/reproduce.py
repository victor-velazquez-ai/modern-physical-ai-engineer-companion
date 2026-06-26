"""Chapter 20 - Build: a reinforcement-learning control policy, on the toy GridWorld.

Offline, no GPU. We train a tabular Q-learner to reach a goal cell by *optimizing reward* (not by
copying a demonstrator), print the learning curve, and evaluate the learned greedy policy. It is
the pure-Python analogue of the PPO-in-Isaac-Lab build of Chapters 17-18: the same idea - reward
is the specification, and trial-and-error finds a policy - at a scale that runs anywhere.

    python chapters/ch20-rl/reproduce.py        (or: make ch20)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from physicalai import GridWorld, GreedyQPolicy, train_q  # noqa: E402

EPISODES = 6000


def make_env():
    # A bigger grid, a tight step budget, and a real step cost: an early wandering policy often
    # fails to reach the goal in time (a clearly negative return), while the learned shortest path
    # succeeds quickly — so the learning curve visibly rises from failure to competence.
    return GridWorld(grid_size=8, goal=(7, 7), max_steps=22, step_cost=0.04)


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    print(f"\nTraining tabular Q-learning on GridWorld 8x8 (reach the goal), {EPISODES} episodes:\n")
    q, returns = train_q(make_env, episodes=EPISODES, epsilon=0.25)

    # The learning curve: mean return over successive windows should rise toward the optimum.
    window = EPISODES // 8
    print("  episodes        mean return")
    print("  --------------  -----------")
    for w in range(8):
        chunk = returns[w * window:(w + 1) * window]
        print(f"  {w * window:>5}-{(w + 1) * window:<7}   {_avg(chunk):+6.3f}")

    # Evaluate the learned greedy policy: does it reach the goal, and how directly?
    policy = GreedyQPolicy(q)
    successes, steps_used = 0, []
    for seed in range(2000, 2100):
        env = make_env()
        state = env.reset(seed)
        done = False
        while not done:
            state, _r, done, info = env.step(policy.action(state))
        if info["success"]:
            successes += 1
            steps_used.append(info["steps"])

    print(f"\n  Learned greedy policy over 100 held-out starts: "
          f"{successes}/100 reach the goal, mean {_avg(steps_used):.1f} steps.\n")
    print("Read it: the return curve rises as the agent learns - reward optimization discovered a")
    print("near-shortest-path policy with no demonstrations. That is the lever RL adds over imitation")
    print("(Ch 17): it can exceed any demonstrator, at the price of needing a reward and many trials.\n")


if __name__ == "__main__":
    main()
