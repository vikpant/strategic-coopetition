"""Reward-type routing: integrated / private / cooperative produce distinct streams."""

from __future__ import annotations

import os

import numpy as np

from extensions.slcd_2d import SLCDAppropriationEnv


def _roll(env, seed: int, n: int = 10) -> np.ndarray:
    env.reset(seed=seed)
    rewards = []
    for _ in range(n):
        action = np.array([50.0, 0.3, 50.0, 0.3], dtype=np.float32)
        _, r, term, trunc, _ = env.step(action)
        rewards.append(r.copy())
        if term or trunc:
            break
    return np.stack(rewards)


def test_private_differs_from_integrated():
    integrated = _roll(SLCDAppropriationEnv(reward_type="integrated"), seed=11)
    private = _roll(SLCDAppropriationEnv(reward_type="private"), seed=11)
    assert not np.allclose(integrated, private)


def test_cooperative_is_uniform_per_step():
    coop = _roll(SLCDAppropriationEnv(reward_type="cooperative"), seed=17)
    # For each step, all agents get the same (mean) reward
    for step_r in coop:
        assert np.allclose(step_r, step_r.mean())


def test_env_var_overrides_default():
    os.environ["COOPETITION_REWARD_TYPE"] = "private"
    try:
        env = SLCDAppropriationEnv()
        assert env.reward_type == "private"
    finally:
        del os.environ["COOPETITION_REWARD_TYPE"]


def test_explicit_reward_type_wins_over_env_var():
    os.environ["COOPETITION_REWARD_TYPE"] = "private"
    try:
        env = SLCDAppropriationEnv(reward_type="cooperative")
        assert env.reward_type == "cooperative"
    finally:
        del os.environ["COOPETITION_REWARD_TYPE"]
