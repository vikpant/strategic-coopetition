"""Action/observation shape and clipping tests."""

from __future__ import annotations

import numpy as np
import pytest

from extensions.slcd_2d import SLCDAppropriationEnv


def test_action_space_shape():
    env = SLCDAppropriationEnv()
    assert env.action_space.shape == (4,), env.action_space.shape
    assert np.allclose(env.action_space.low, [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(env.action_space.high, [100.0, 1.0, 100.0, 1.0])


def test_obs_space_matches_v1():
    env = SLCDAppropriationEnv()
    assert env.observation_space.shape == (15,)


def test_step_returns_float32_rewards():
    env = SLCDAppropriationEnv()
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.array([50.0, 0.3, 50.0, 0.3], dtype=np.float32))
    assert r.dtype == np.float32
    assert r.shape == (2,)
    assert "cooperation" in info and "appropriation" in info
    assert info["appropriation"].shape == (2,)


def test_action_is_clipped_to_bounds():
    env = SLCDAppropriationEnv()
    env.reset(seed=0)
    out_of_bounds = np.array([-10.0, 2.0, 200.0, -1.0], dtype=np.float32)
    _, _, _, _, info = env.step(out_of_bounds)
    assert 0.0 <= info["cooperation"][0] <= 100.0
    assert 0.0 <= info["cooperation"][1] <= 100.0
    assert 0.0 <= info["appropriation"][0] <= 1.0
    assert 0.0 <= info["appropriation"][1] <= 1.0


def test_invalid_action_shape_raises():
    env = SLCDAppropriationEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array([50.0, 0.3], dtype=np.float32))


def test_appropriation_changes_reward():
    env = SLCDAppropriationEnv()
    env.reset(seed=0)
    _, r_no_appr, _, _, _ = env.step(np.array([50.0, 0.0, 50.0, 0.0], dtype=np.float32))
    env.reset(seed=0)
    _, r_with_appr, _, _, _ = env.step(np.array([50.0, 0.3, 50.0, 0.3], dtype=np.float32))
    assert not np.allclose(r_no_appr, r_with_appr), (
        "Reward should change when p > 0"
    )
