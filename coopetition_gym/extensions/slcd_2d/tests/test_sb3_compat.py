"""SB3 training-path compatibility: IPPO can actually train on the 2D env."""

from __future__ import annotations

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")


def test_ippo_trains_on_2d():
    import sys
    sys.path.insert(0, "/home/vik_p/projects/strategic-coopetition")
    from experiments.algorithms import IndependentPPO

    from extensions.slcd_2d import SLCDAppropriationEnv

    env = SLCDAppropriationEnv(max_steps=40)
    algo = IndependentPPO(env, device="cpu", seed=42, n_steps=64, batch_size=32)
    algo.train(total_timesteps=256)
    assert len(algo.training_returns) > 0, "No episodes completed — horizon/steps mismatch?"


def test_oracle_build_via_registry():
    from extensions.slcd_2d import SLCDAppropriationEnv
    from extensions.slcd_2d.algorithms import build_algorithm, list_algorithms

    assert "Oracle_Appropriation" in list_algorithms()
    assert "IPPO" in list_algorithms()
    assert "ISAC" in list_algorithms()

    env = SLCDAppropriationEnv(max_steps=40)
    algo = build_algorithm("Oracle_Appropriation", env, device="cpu", seed=0)
    action, _ = algo.predict(obs=None, deterministic=True)
    assert env.action_space.contains(action)
