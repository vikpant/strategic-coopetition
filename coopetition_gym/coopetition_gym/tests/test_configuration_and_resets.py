"""Regression coverage for configuration overrides and episode initialization."""

import numpy as np
import pytest

from coopetition_gym.envs import (
    CoopetitionEnv,
    EnvironmentConfig,
    list_environments,
    make,
    make_aec,
    make_parallel,
)


ENV_IDS = list_environments()
FACTORIES = [make, make_parallel, make_aec]
WRAPPERS = [make_parallel, make_aec]


def _base(env):
    return getattr(env, "base_env", env)


def _round(env, fraction=0.6):
    """Advance one complete joint-action round through each public API."""
    base = _base(env)
    actions = {
        agent: np.array([base.endowments[i] * fraction], dtype=np.float32)
        for i, agent in enumerate(base.possible_agents)
    }
    if not hasattr(env, "base_env"):
        _, rewards, terminated, truncated, _ = env.step(
            np.concatenate(list(actions.values()))
        )
        return rewards, terminated, truncated
    if hasattr(env, "agent_selection"):
        for _ in env.possible_agents:
            env.step(actions[env.agent_selection])
        rewards = env.rewards
        terminated = env.terminations
        truncated = env.truncations
    else:
        _, rewards, terminated, truncated, _ = env.step(actions)
    return (
        np.array([rewards[a] for a in env.possible_agents]),
        all(terminated.values()),
        all(truncated.values()),
    )


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("reward_type", ["integrated", "private", "cooperative"])
def test_factory_honors_reward_type(factory, env_id, reward_type):
    env = factory(env_id, reward_type=reward_type, max_steps=3)
    try:
        assert _base(env).reward_type == reward_type
        assert _base(env).config.reward_type == reward_type
        env.reset(seed=41)
        rewards, _, _ = _round(env)
        assert np.all(np.isfinite(rewards))
    finally:
        env.close()


@pytest.mark.parametrize("factory", FACTORIES)
def test_reward_modes_change_the_returned_objective(factory):
    rewards = {}
    for reward_type in ("integrated", "private", "cooperative"):
        env = factory("SLCD-v0", reward_type=reward_type)
        try:
            env.reset(seed=41)
            rewards[reward_type], _, _ = _round(env)
        finally:
            env.close()
    assert np.all(rewards["integrated"] > rewards["private"])
    np.testing.assert_allclose(
        rewards["cooperative"], np.mean(rewards["integrated"]), rtol=1e-6
    )


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_invalid_reward_type_is_rejected(env_id):
    with pytest.raises(ValueError, match="reward_type"):
        make(env_id, reward_type="unknown")


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_unknown_config_override_is_rejected(env_id):
    with pytest.raises(TypeError, match="rewrad_type"):
        make(env_id, rewrad_type="private")


def test_config_overrides_leave_the_supplied_config_unchanged():
    config = EnvironmentConfig()
    env = CoopetitionEnv(config=config, reward_type="private", reward_scale=2.0)
    try:
        assert env.reward_type == "private"
        assert env.reward_scale == 2.0
        assert config.reward_type == "integrated"
        assert config.reward_scale == 1.0
        assert env.config is not config
    finally:
        env.close()


@pytest.mark.parametrize("factory", WRAPPERS)
def test_coalition_reset_initializes_members_before_first_round(factory):
    env = factory("CoalitionFormation-v0", max_steps=5)
    try:
        for _ in range(2):
            env.reset(seed=41)
            assert env.base_env._coalition_members == list(range(env.base_env.n_agents))
            _, terminated, truncated = _round(env)
            assert not terminated
            assert not truncated
    finally:
        env.close()


@pytest.mark.parametrize("factory", WRAPPERS)
def test_loyalty_reset_clears_previous_episode(factory):
    env = factory("LoyaltyTeam-v0", max_steps=5)
    try:
        env.reset(seed=41)
        initial = env.base_env._loyalty_scores.copy()
        _round(env, fraction=1.0)
        assert np.all(env.base_env._loyalty_scores > initial)
        env.reset(seed=41)
        np.testing.assert_array_equal(env.base_env._loyalty_scores, initial)
        assert env.base_env._loyalty_history == []
        assert env.base_env._ca_state is not None
    finally:
        env.close()


@pytest.mark.parametrize("factory", WRAPPERS)
def test_wrapper_forwards_reputation_reset_option(factory):
    env = factory("DynamicPartnerSelection-v0")
    try:
        env.reset(seed=41)
        env.base_env._global_reputation.fill(0.8)
        env.reset(seed=41)
        np.testing.assert_allclose(env.base_env._global_reputation, 0.8)
        env.reset(seed=41, options={"reset_reputation": True})
        np.testing.assert_array_equal(env.base_env._global_reputation, 0.0)
    finally:
        env.close()


@pytest.mark.parametrize("factory", WRAPPERS)
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_repeated_seeded_wrapper_episodes_match(factory, env_id):
    env = factory(env_id, max_steps=3)

    def episode():
        env.reset(seed=41, options={"reset_reputation": True})
        initial_obs = np.concatenate([
            env.base_env.get_observation_for(a) for a in env.possible_agents
        ])
        trajectory = []
        for _ in range(3):
            rewards, terminated, truncated = _round(env)
            observations = np.concatenate([
                env.base_env.get_observation_for(a) for a in env.possible_agents
            ])
            trajectory.append(np.concatenate([
                observations, rewards, [terminated, truncated]
            ]))
            if terminated or truncated:
                break
        return initial_obs, np.stack(trajectory)

    try:
        first = episode()
        second = episode()
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
    finally:
        env.close()


@pytest.mark.parametrize("factory", FACTORIES)
def test_synergy_episode_gamma_follows_reset_seed(factory):
    env = factory("SynergySearch-v0")
    try:
        env.reset(seed=41)
        first = _base(env)._true_gamma
        env.reset(seed=42)
        second = _base(env)._true_gamma
        env.reset(seed=41)
        assert _base(env)._true_gamma == first
        assert second != first
    finally:
        env.close()
