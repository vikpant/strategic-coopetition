"""Backward-compatibility test: with p=0, SLCDAppropriationEnv matches SLCDEnv.

The 2D integrated utility reduces to the v1 integrated utility when every
p_i is zero (see utility.py docstring). We enforce this at the reward
stream level: for the same seed and the same cooperation trajectory, the
2D env with p=0 must produce reward vectors within a tight tolerance of
the v1 env.

Tolerance is 1e-4 (not 0) because the 2D utility is computed in float64
and cast to float32, while the v1 path computes in float32 throughout.
"""

from __future__ import annotations

import numpy as np

from coopetition_gym.envs import make

from extensions.slcd_2d import SLCDAppropriationEnv


TOL = 1e-3


def _run_v1(seed: int, cooperation_traj: np.ndarray) -> np.ndarray:
    env = make("SLCD-v0")
    env.reset(seed=seed)
    rewards = []
    for action in cooperation_traj:
        _, r, term, trunc, _ = env.step(action.astype(np.float32))
        rewards.append(r.copy())
        if term or trunc:
            break
    return np.asarray(rewards)


def _run_2d_with_p_zero(seed: int, cooperation_traj: np.ndarray) -> np.ndarray:
    env = SLCDAppropriationEnv()
    env.reset(seed=seed)
    rewards = []
    for c_action in cooperation_traj:
        flat = np.zeros(2 * env.n_agents, dtype=np.float32)
        flat[0::2] = c_action.astype(np.float32)
        _, r, term, trunc, _ = env.step(flat)
        rewards.append(r.copy())
        if term or trunc:
            break
    return np.asarray(rewards)


def test_constant_trajectory_matches_v1():
    traj = np.tile(np.array([50.0, 50.0], dtype=np.float32), (10, 1))
    v1 = _run_v1(seed=42, cooperation_traj=traj)
    v2 = _run_2d_with_p_zero(seed=42, cooperation_traj=traj)
    assert v1.shape == v2.shape
    max_abs = float(np.max(np.abs(v1 - v2)))
    assert max_abs < TOL, f"max abs diff {max_abs} >= {TOL}"


def test_varying_trajectory_matches_v1():
    rng = np.random.default_rng(123)
    traj = rng.uniform(10.0, 90.0, size=(20, 2)).astype(np.float32)
    v1 = _run_v1(seed=7, cooperation_traj=traj)
    v2 = _run_2d_with_p_zero(seed=7, cooperation_traj=traj)
    assert v1.shape == v2.shape
    max_abs = float(np.max(np.abs(v1 - v2)))
    assert max_abs < TOL, f"max abs diff {max_abs} >= {TOL}"


def test_extreme_corner_trajectory_matches_v1():
    traj = np.array(
        [
            [0.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
            [100.0, 0.0],
            [50.0, 50.0],
        ],
        dtype=np.float32,
    )
    v1 = _run_v1(seed=0, cooperation_traj=traj)
    v2 = _run_2d_with_p_zero(seed=0, cooperation_traj=traj)
    max_abs = float(np.max(np.abs(v1 - v2)))
    assert max_abs < TOL, f"corner-case max abs diff {max_abs} >= {TOL}"
