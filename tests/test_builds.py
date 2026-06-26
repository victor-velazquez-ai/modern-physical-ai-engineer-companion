"""Tests for the ch16 (imitation) and ch20 (RL) build pieces — all offline, no GPU."""

from physicalai import (
    BCPolicy, GreedyQPolicy, GridWorld, RandomPolicy, TabletopEnv,
    collect_expert_demos, evaluate, evaluate_split, featurize, train_q,
)
from physicalai.envs.gridworld import GW_ACTIONS

EVAL = list(range(200, 240))


# --- ch16: behavior cloning + covariate shift -----------------------------------------
def test_bc_learns_and_beats_random_in_distribution():
    demos = collect_expert_demos(lambda: TabletopEnv(grid_size=3), list(range(40)))
    bc = BCPolicy().fit(demos)
    bc_card = evaluate(lambda: TabletopEnv(grid_size=3), bc, EVAL, label="bc")
    rnd_card = evaluate(lambda: TabletopEnv(grid_size=3), RandomPolicy(), EVAL, label="rnd")
    assert bc_card.success_rate > 0.5            # BC works in-distribution
    assert bc_card.success_rate > rnd_card.success_rate


def test_bc_shows_covariate_shift_gap():
    demos = collect_expert_demos(lambda: TabletopEnv(grid_size=3), list(range(40)))
    bc = BCPolicy().fit(demos)
    id_card, ood_card = evaluate_split(bc, EVAL, grid_size=3, label="bc")
    # The whole lesson: BC generalizes worse out-of-distribution than in-distribution.
    assert id_card.success_rate > ood_card.success_rate


def test_more_demos_help():
    small = BCPolicy().fit(collect_expert_demos(lambda: TabletopEnv(grid_size=3), list(range(5))))
    large = BCPolicy().fit(collect_expert_demos(lambda: TabletopEnv(grid_size=3), list(range(40))))
    s = evaluate(lambda: TabletopEnv(grid_size=3), small, EVAL).success_rate
    l = evaluate(lambda: TabletopEnv(grid_size=3), large, EVAL).success_rate
    assert l >= s  # imitation is data-hungry


def test_featurize_is_context_aware():
    obs = TabletopEnv(grid_size=3).reset(seed=1)
    feat = featurize(obs)
    assert feat[0] == 0 and feat[3] == 0 and feat[4] == 0  # not holding -> only block-deltas matter


# --- ch20: gridworld + tabular Q-learning ---------------------------------------------
def test_gridworld_is_deterministic_and_rewards_the_goal():
    a = GridWorld().reset(seed=5)
    b = GridWorld().reset(seed=5)
    assert a == b
    env = GridWorld(grid_size=3, goal=(0, 0))
    env.reset(seed=0)
    env._agent = (1, 0)
    _state, reward, done, info = env.step(GW_ACTIONS[2])  # WEST -> reach (0,0)
    assert reward == 1.0 and done and info["success"]


def test_q_learning_learns_to_reach_the_goal():
    def make_env():
        return GridWorld(grid_size=6, goal=(5, 5), max_steps=40, step_cost=0.02)

    q, returns = train_q(make_env, episodes=2000, epsilon=0.2)
    policy = GreedyQPolicy(q)
    successes = 0
    for seed in range(500, 560):
        env = make_env()
        state = env.reset(seed)
        done = False
        while not done:
            state, _r, done, info = env.step(policy.action(state))
        successes += 1 if info["success"] else 0
    assert successes >= 57  # the learned greedy policy reaches the goal nearly always
    assert sum(returns[-200:]) / 200 > sum(returns[:200]) / 200  # learning curve rose
