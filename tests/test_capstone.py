"""Tests for the ch36 VLM-as-policy path and the capstone — forced offline (no real API calls)."""

import pytest

from physicalai import TabletopEnv, VLMPolicy, evaluate
from physicalai.policies.vlm import describe
from capstone.pipeline import RobotSystem


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    # Force the offline path in every test here, regardless of the developer's environment, so the
    # suite never makes a network call or spends a token.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_describe_includes_the_instruction():
    obs = TabletopEnv().reset(seed=2)
    text = describe(obs)
    assert obs["instruction"] in text and "Gripper at" in text


def test_vlm_policy_falls_back_and_solves_offline():
    policy = VLMPolicy()
    card = evaluate(lambda: TabletopEnv(), policy, list(range(30)), label="vlm")
    assert policy.used_fallback is True       # no key -> fell back to the oracle
    assert card.success_rate == 1.0           # and still solved every episode


def test_capstone_runs_end_to_end_offline():
    system = RobotSystem()                     # default oracle brain
    episode = system.run_episode(TabletopEnv(), seed=0)
    assert episode.success is True
    assert len(episode.trajectory) > 0


def test_capstone_honors_a_requested_goal():
    system = RobotSystem()
    env = TabletopEnv()
    episode = system.run_episode(env, seed=0, goal_color="green", goal_target="bowl")
    assert episode.success is True
    # the env was actually set to the requested goal
    obs = env._obs()
    assert obs["goal"] == {"color": "green", "target": "bowl"}


def test_capstone_evaluate_is_honest_and_high():
    system = RobotSystem()
    card = system.evaluate(lambda: TabletopEnv(), list(range(40)))
    assert card.success_rate == 1.0 and card.n_episodes == 40
