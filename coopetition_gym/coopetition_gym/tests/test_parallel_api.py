"""
================================================================================
COOPETITION-GYM: Parallel API Tests
================================================================================

Tests for PettingZoo ParallelEnv compliance.

Authors: Vik Pant, Eric Yu
License: MIT
================================================================================
"""

import pytest
import numpy as np

# Note: pettingzoo.test may not be available in all environments
# These tests can be run when pettingzoo[testing] is installed


ENV_IDS = [
    "TrustDilemma-v0", 
    "PartnerHoldUp-v0", 
    "PlatformEcosystem-v0",
    "DynamicPartnerSelection-v0", 
    "RecoveryRace-v0", 
    "SynergySearch-v0",
    "SLCD-v0", 
    "RenaultNissan-v0", 
    "CooperativeNegotiation-v0", 
    "ReputationMarket-v0",
]


class TestParallelBasicFunctionality:
    """Test basic parallel environment functionality."""
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_parallel_reset(self, env_id):
        """Test that parallel reset returns correct format."""
        from coopetition_gym.envs import make_parallel
        
        env = make_parallel(env_id, max_steps=50)
        observations, infos = env.reset(seed=42)
        
        # Check that all agents have observations
        assert set(observations.keys()) == set(env.possible_agents)
        assert set(infos.keys()) == set(env.possible_agents)
        
        # Check observation shapes
        for agent in env.agents:
            obs = observations[agent]
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert env.observation_space(agent).contains(obs)
        
        env.close()
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_parallel_step(self, env_id):
        """Test that parallel step returns correct format."""
        from coopetition_gym.envs import make_parallel
        
        env = make_parallel(env_id, max_steps=50)
        observations, infos = env.reset(seed=42)
        
        # Take random actions
        actions = {
            agent: env.action_space(agent).sample() 
            for agent in env.agents
        }
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check return types
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)
        
        # Check all agents have entries
        for agent in env.possible_agents:
            assert agent in rewards
            assert agent in terminations
            assert agent in truncations
            
        # Check types
        for agent in env.possible_agents:
            assert isinstance(rewards[agent], (int, float))
            assert isinstance(terminations[agent], bool)
            assert isinstance(truncations[agent], bool)
        
        env.close()
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_parallel_episode(self, env_id):
        """Test running a complete episode."""
        from coopetition_gym.envs import make_parallel
        
        env = make_parallel(env_id, max_steps=20)
        observations, infos = env.reset(seed=42)
        
        total_rewards = {agent: 0.0 for agent in env.possible_agents}
        step_count = 0
        
        while env.agents:
            actions = {
                agent: env.action_space(agent).sample() 
                for agent in env.agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent in env.possible_agents:
                if agent in rewards:
                    total_rewards[agent] += rewards[agent]
            
            step_count += 1
            if step_count > 25:  # Safety limit
                break
        
        # Check that episode ran to max_steps
        assert step_count <= 20 + 5  # Allow small buffer
        
        env.close()
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_parallel_spaces(self, env_id):
        """Test that action and observation spaces are valid."""
        from coopetition_gym.envs import make_parallel
        from gymnasium import spaces
        
        env = make_parallel(env_id, max_steps=50)
        env.reset(seed=42)
        
        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            action_space = env.action_space(agent)
            
            assert isinstance(obs_space, spaces.Box)
            assert isinstance(action_space, spaces.Box)
            assert obs_space.dtype == np.float32
            assert action_space.dtype == np.float32
            
            # Check bounds
            assert np.all(action_space.low >= 0)
        
        env.close()


class TestParallelSeeding:
    """Test reproducibility with seeding."""
    
    @pytest.mark.parametrize("env_id", ["TrustDilemma-v0", "SLCD-v0"])
    def test_parallel_deterministic(self, env_id):
        """Test that seeding produces deterministic results."""
        from coopetition_gym.envs import make_parallel
        
        def run_episode(env, seed):
            obs1, _ = env.reset(seed=seed)
            actions = {agent: np.array([50.0]) for agent in env.agents}
            obs2, rewards, _, _, _ = env.step(actions)
            return obs1, obs2, rewards
        
        env1 = make_parallel(env_id, max_steps=50)
        env2 = make_parallel(env_id, max_steps=50)
        
        obs1_a, obs2_a, rewards_a = run_episode(env1, seed=123)
        obs1_b, obs2_b, rewards_b = run_episode(env2, seed=123)
        
        for agent in env1.possible_agents:
            assert np.allclose(obs1_a[agent], obs1_b[agent])
            assert np.allclose(obs2_a[agent], obs2_b[agent])
            assert np.isclose(rewards_a[agent], rewards_b[agent])
        
        env1.close()
        env2.close()


class TestParallelPettingZooCompliance:
    """Test PettingZoo API compliance (requires pettingzoo installed)."""
    
    @pytest.mark.parametrize("env_id", ENV_IDS[:3])  # Test subset
    def test_pettingzoo_api(self, env_id):
        """Run PettingZoo's official API test."""
        try:
            from pettingzoo.test import parallel_api_test
            from coopetition_gym.envs import make_parallel
            
            env = make_parallel(env_id, max_steps=50)
            parallel_api_test(env, num_cycles=20)
            env.close()
        except ImportError:
            pytest.skip("pettingzoo.test not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
