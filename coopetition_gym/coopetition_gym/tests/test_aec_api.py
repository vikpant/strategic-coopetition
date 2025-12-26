"""
================================================================================
COOPETITION-GYM: AEC API Tests
================================================================================

Tests for PettingZoo AECEnv compliance.

Authors: Vik Pant, Eric Yu
License: MIT
================================================================================
"""

import pytest
import numpy as np


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


class TestAECBasicFunctionality:
    """Test basic AEC environment functionality."""
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_aec_reset(self, env_id):
        """Test that AEC reset initializes correctly."""
        from coopetition_gym.envs import make_aec
        
        env = make_aec(env_id, max_steps=50)
        env.reset(seed=42)
        
        # Check agent selection
        assert env.agent_selection in env.possible_agents
        assert len(env.agents) == len(env.possible_agents)
        
        # Check initial state
        for agent in env.possible_agents:
            assert agent in env.rewards
            assert agent in env.terminations
            assert agent in env.truncations
        
        env.close()
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_aec_last(self, env_id):
        """Test that last() returns correct format."""
        from coopetition_gym.envs import make_aec
        
        env = make_aec(env_id, max_steps=50)
        env.reset(seed=42)
        
        observation, reward, termination, truncation, info = env.last()
        
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(termination, bool)
        assert isinstance(truncation, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_aec_step(self, env_id):
        """Test that AEC step processes actions correctly."""
        from coopetition_gym.envs import make_aec
        
        env = make_aec(env_id, max_steps=50)
        env.reset(seed=42)
        
        initial_agent = env.agent_selection
        action = env.action_space(initial_agent).sample()
        
        env.step(action)
        
        # Agent should have changed
        assert env.agent_selection is not None
        
        env.close()
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_aec_agent_iter(self, env_id):
        """Test agent iteration pattern."""
        from coopetition_gym.envs import make_aec
        
        env = make_aec(env_id, max_steps=10)
        env.reset(seed=42)
        
        step_count = 0
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample()
            
            env.step(action)
            step_count += 1
            
            if step_count > 100:  # Safety limit
                break
        
        env.close()
    
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_aec_observe(self, env_id):
        """Test observe() returns valid observations."""
        from coopetition_gym.envs import make_aec
        
        env = make_aec(env_id, max_steps=50)
        env.reset(seed=42)
        
        for agent in env.possible_agents:
            obs = env.observe(agent)
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert env.observation_space(agent).contains(obs)
        
        env.close()


class TestAECSequentialBehavior:
    """Test sequential move dynamics."""
    
    def test_revealed_actions(self):
        """Test that later movers see earlier movers' actions."""
        from coopetition_gym.envs import make_aec
        
        env = make_aec("TrustDilemma-v0", max_steps=50)
        env.reset(seed=42)
        
        # First agent acts
        first_agent = env.agent_selection
        first_action = np.array([75.0])
        
        obs_before = env.observe(env.possible_agents[1])
        env.step(first_action)
        
        # Second agent should see first agent's action in observation
        second_agent = env.agent_selection
        obs_after = env.observe(second_agent)
        
        # Observations should differ (revealed actions dimension)
        assert obs_before.shape == obs_after.shape
        # The revealed action component should show the action
        # (Last n_agents dimensions are revealed actions)
        assert obs_after[-2] == pytest.approx(75.0, rel=0.01)  # First agent's action revealed
        
        env.close()
    
    def test_full_round(self):
        """Test that a full round processes correctly."""
        from coopetition_gym.envs import make_aec
        
        env = make_aec("TrustDilemma-v0", max_steps=50)
        env.reset(seed=42)
        
        initial_rewards = {agent: env.rewards[agent] for agent in env.possible_agents}
        
        # Complete one round
        for _ in range(env.base_env.n_agents):
            agent = env.agent_selection
            action = env.action_space(agent).sample()
            env.step(action)
        
        # Rewards should have been updated after full round
        for agent in env.possible_agents:
            assert agent in env.rewards
        
        env.close()


class TestAECPettingZooCompliance:
    """Test PettingZoo AEC API compliance."""
    
    @pytest.mark.parametrize("env_id", ENV_IDS[:3])
    def test_pettingzoo_aec_api(self, env_id):
        """Run PettingZoo's official AEC API test."""
        try:
            from pettingzoo.test import api_test
            from coopetition_gym.envs import make_aec
            
            env = make_aec(env_id, max_steps=50)
            api_test(env, num_cycles=20)
            env.close()
        except ImportError:
            pytest.skip("pettingzoo.test not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
