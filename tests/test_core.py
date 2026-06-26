"""Offline tests for the physicalai core — env, policies, harness. No GPU, no key, no network."""

from physicalai import (
    RandomPolicy, ScriptedPolicy, TabletopEnv, evaluate, evaluate_split, parse_instruction,
)
from physicalai.envs.tabletop import PICK, PLACE


SEEDS = list(range(40))


def test_env_is_deterministic():
    a = TabletopEnv().reset(seed=7)
    b = TabletopEnv().reset(seed=7)
    assert a == b
    assert TabletopEnv().reset(seed=8) != a  # different seed -> different layout


def test_instruction_parses():
    obs = TabletopEnv().reset(seed=1)
    color, target = parse_instruction(obs["instruction"])
    assert obs["instruction"] == f"put the {color} block on the {target}"
    assert color == obs["goal"]["color"] and target == obs["goal"]["target"]


def test_manual_pick_and_place_succeeds():
    env = TabletopEnv()
    obs = env.reset(seed=3)
    color, target = parse_instruction(obs["instruction"])
    # Drive the gripper to the block, pick, to the target, place — by hand.
    pol = ScriptedPolicy()
    done = False
    while not done:
        for a in pol.select_action(env._obs()):
            obs, r, done, info = env.step(a)
            if done:
                break
    assert env.is_success()
    assert info["success"] is True


def test_scripted_policy_solves_in_distribution():
    card = evaluate(lambda: TabletopEnv(), ScriptedPolicy(), SEEDS, label="oracle")
    assert card.success_rate == 1.0  # the oracle solves every fixed-seed episode
    assert card.n_episodes == len(SEEDS)


def test_scripted_policy_solves_ood_too():
    # The oracle *computes* the plan, so it also solves the harder OOD layout — the ID/OOD gap
    # is a property of *learned* policies, which the harness is built to measure.
    id_card, ood_card = evaluate_split(ScriptedPolicy(), SEEDS, label="oracle")
    assert id_card.success_rate == 1.0
    assert ood_card.success_rate == 1.0


def test_random_policy_is_a_weak_floor():
    card = evaluate(lambda: TabletopEnv(), RandomPolicy(seed=0), SEEDS, label="random")
    assert card.success_rate < 0.5  # random rarely completes the task — the contrast baseline


def test_scorecard_str_is_readable():
    card = evaluate(lambda: TabletopEnv(), ScriptedPolicy(), SEEDS[:5], label="oracle")
    s = str(card)
    assert "success" in s and "100.0%" in s
