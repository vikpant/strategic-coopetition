"""
================================================================================
COOPETITION-GYM: Comprehensive Test Suite
================================================================================

This test suite validates all components of the coopetition-gym library:
- Core mathematical modules (value functions, interdependence, trust, equilibrium)
- All 10 environments (API compliance, dynamics, edge cases)
- Utility functions

Run with: pytest tests/test_coopetition_gym.py -v

Authors: Vik Pant, Eric Yu
License: MIT
================================================================================
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import gymnasium as gym


# =============================================================================
# Test Core: Value Functions
# =============================================================================

class TestValueFunctions:
    """Test suite for value_functions.py module."""
    
    def test_power_value_basic(self):
        """Test power value function with standard inputs."""
        from coopetition_gym.core.value_functions import power_value
        
        # f(a) = a^0.75
        result = power_value(16.0, beta=0.75)
        expected = 16.0 ** 0.75  # 8.0
        assert_allclose(result, expected, rtol=1e-5)
    
    def test_power_value_zero(self):
        """Test power value handles zero input gracefully."""
        from coopetition_gym.core.value_functions import power_value
        
        result = power_value(0.0, beta=0.75)
        assert result >= 0  # Should be non-negative
        assert result < 0.01  # Should be close to zero
    
    def test_logarithmic_value_basic(self):
        """Test logarithmic value function."""
        from coopetition_gym.core.value_functions import logarithmic_value
        
        # f(a) = 20 * ln(1 + 50) = 20 * 3.9318... ≈ 78.55
        result = logarithmic_value(50.0, theta=20.0)
        expected = 20.0 * np.log(51.0)
        assert_allclose(result, expected, rtol=1e-5)
    
    def test_logarithmic_value_zero(self):
        """Test logarithmic value at zero."""
        from coopetition_gym.core.value_functions import logarithmic_value
        
        result = logarithmic_value(0.0, theta=20.0)
        assert result == 0.0  # ln(1) = 0
    
    def test_synergy_function_balanced(self):
        """Test synergy with equal contributions."""
        from coopetition_gym.core.value_functions import synergy_function
        
        actions = np.array([50.0, 50.0])
        result = synergy_function(actions)
        assert_allclose(result, 50.0, rtol=1e-5)  # Geometric mean
    
    def test_synergy_function_collapse(self):
        """Test synergy collapses when any agent defects."""
        from coopetition_gym.core.value_functions import synergy_function
        
        actions = np.array([100.0, 0.0])
        result = synergy_function(actions)
        assert result == 0.0  # Zero contribution kills synergy
    
    def test_total_value_composition(self):
        """Test total value = individual + synergy."""
        from coopetition_gym.core.value_functions import (
            total_value, individual_value, synergy_function,
            ValueFunctionParameters
        )
        
        params = ValueFunctionParameters(gamma=0.5)
        actions = np.array([40.0, 60.0])
        
        total = total_value(actions, params)
        indiv_sum = sum(individual_value(a, params) for a in actions)
        synergy = synergy_function(actions)
        
        expected = indiv_sum + params.gamma * synergy
        assert_allclose(total, expected, rtol=1e-5)
    
    def test_value_parameters_validation(self):
        """Test parameter validation catches invalid inputs."""
        from coopetition_gym.core.value_functions import ValueFunctionParameters
        
        with pytest.raises(ValueError):
            ValueFunctionParameters(beta=1.5)  # Must be <= 1
        
        with pytest.raises(ValueError):
            ValueFunctionParameters(gamma=-0.1)  # Must be >= 0


# =============================================================================
# Test Core: Interdependence
# =============================================================================

class TestInterdependence:
    """Test suite for interdependence.py module."""
    
    def test_slcd_interdependence_values(self):
        """Test S-LCD interdependence matches TR-1 validated values."""
        from coopetition_gym.core.interdependence import create_slcd_interdependence
        
        D = create_slcd_interdependence()
        
        # From TR-1 §8.2
        assert_allclose(D.get_dependency("Sony", "Samsung"), 0.86, rtol=0.01)
        assert_allclose(D.get_dependency("Samsung", "Sony"), 0.64, rtol=0.01)
        
        # Diagonal should be zero
        assert D.get_dependency("Samsung", "Samsung") == 0.0
    
    def test_symmetric_interdependence(self):
        """Test symmetric interdependence creation."""
        from coopetition_gym.core.interdependence import create_symmetric_interdependence
        
        D = create_symmetric_interdependence(n_agents=3, dependency_strength=0.5)
        
        assert D.n_agents == 3
        
        # All off-diagonal should be 0.5
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert D.matrix[i, j] == 0.5
                else:
                    assert D.matrix[i, j] == 0.0
    
    def test_asymmetry_calculation(self):
        """Test asymmetry calculation between agents."""
        from coopetition_gym.core.interdependence import create_asymmetric_interdependence
        
        D = create_asymmetric_interdependence(strong_dependency=0.85, weak_dependency=0.35)
        
        # Weak agent is more dependent on strong
        asymmetry = D.asymmetry(1, 0)  # Weak's perspective
        assert asymmetry == 0.85 - 0.35  # Positive: weak depends more
    
    def test_renault_nissan_phases(self):
        """Test Renault-Nissan has different configurations per phase."""
        from coopetition_gym.core.interdependence import create_renault_nissan_interdependence
        
        formation = create_renault_nissan_interdependence("formation")
        mature = create_renault_nissan_interdependence("mature")
        
        # Formation phase: Nissan highly dependent (near bankruptcy)
        # Mature phase: More balanced after recovery
        nissan_dep_formation = formation.get_dependency(0, 1)  # Nissan depends on Renault
        nissan_dep_mature = mature.get_dependency(0, 1)
        
        assert nissan_dep_formation > nissan_dep_mature  # Dependency decreased


# =============================================================================
# Test Core: Trust Dynamics
# =============================================================================

class TestTrustDynamics:
    """Test suite for trust_dynamics.py module."""
    
    def test_trust_parameters_defaults(self):
        """Test default parameters match TR-2 validated values."""
        from coopetition_gym.core.trust_dynamics import TrustParameters
        
        params = TrustParameters()
        
        # Negativity bias should be ~3x
        assert_allclose(params.negativity_ratio, 3.0, rtol=0.1)
    
    def test_cooperation_signal_positive(self):
        """Test positive cooperation signal when action > baseline."""
        from coopetition_gym.core.trust_dynamics import TrustDynamicsModel
        
        model = TrustDynamicsModel()
        signal = model.cooperation_signal(action=70.0, baseline=50.0)
        
        assert signal > 0  # Positive signal
        assert signal <= 1  # Bounded by tanh
    
    def test_cooperation_signal_negative(self):
        """Test negative cooperation signal when action < baseline."""
        from coopetition_gym.core.trust_dynamics import TrustDynamicsModel
        
        model = TrustDynamicsModel()
        signal = model.cooperation_signal(action=30.0, baseline=50.0)
        
        assert signal < 0  # Negative signal
        assert signal >= -1  # Bounded by tanh
    
    def test_trust_builds_with_cooperation(self):
        """Test trust increases when agent cooperates."""
        from coopetition_gym.core.trust_dynamics import TrustDynamicsModel, TrustState
        
        model = TrustDynamicsModel()
        state = model.create_initial_state(n_agents=2)
        
        initial_trust = state.get_trust(0, 1)
        
        # Cooperate above baseline
        actions = np.array([70.0, 70.0])
        baselines = np.array([50.0, 50.0])
        D = np.array([[0, 0.5], [0.5, 0]])
        
        new_state = model.update(state, actions, baselines, D)
        new_trust = new_state.get_trust(0, 1)
        
        assert new_trust > initial_trust
    
    def test_trust_erodes_with_defection(self):
        """Test trust decreases when agent defects."""
        from coopetition_gym.core.trust_dynamics import TrustDynamicsModel
        
        model = TrustDynamicsModel()
        state = model.create_initial_state(n_agents=2)
        
        initial_trust = state.get_trust(0, 1)
        
        # Defect below baseline
        actions = np.array([20.0, 20.0])
        baselines = np.array([50.0, 50.0])
        D = np.array([[0, 0.5], [0.5, 0]])
        
        new_state = model.update(state, actions, baselines, D)
        new_trust = new_state.get_trust(0, 1)
        
        assert new_trust < initial_trust
    
    def test_negativity_bias(self):
        """Test that trust erodes faster than it builds."""
        from coopetition_gym.core.trust_dynamics import TrustDynamicsModel
        
        model = TrustDynamicsModel()
        
        baselines = np.array([50.0, 50.0])
        D = np.array([[0, 0.5], [0.5, 0]])
        
        # Build trust
        state_build = model.create_initial_state(2)
        actions_cooperate = np.array([70.0, 70.0])
        state_after_build = model.update(state_build, actions_cooperate, baselines, D)
        trust_gain = state_after_build.get_trust(0, 1) - state_build.get_trust(0, 1)
        
        # Erode trust (symmetric deviation)
        state_erode = model.create_initial_state(2)
        actions_defect = np.array([30.0, 30.0])
        state_after_erode = model.update(state_erode, actions_defect, baselines, D)
        trust_loss = state_erode.get_trust(0, 1) - state_after_erode.get_trust(0, 1)
        
        # Loss should be greater than gain (negativity bias)
        assert trust_loss > trust_gain
    
    def test_reputation_damage_accumulates(self):
        """Test reputation damage from violations."""
        from coopetition_gym.core.trust_dynamics import TrustDynamicsModel
        
        model = TrustDynamicsModel()
        state = model.create_initial_state(2)
        
        initial_damage = state.get_reputation_damage(0, 1)
        
        # Defect
        actions = np.array([10.0, 10.0])
        baselines = np.array([50.0, 50.0])
        D = np.array([[0, 0.5], [0.5, 0]])
        
        new_state = model.update(state, actions, baselines, D)
        new_damage = new_state.get_reputation_damage(0, 1)
        
        assert new_damage > initial_damage


# =============================================================================
# Test Core: Equilibrium
# =============================================================================

class TestEquilibrium:
    """Test suite for equilibrium.py module."""
    
    def test_private_payoff_structure(self):
        """Test private payoff has correct components."""
        from coopetition_gym.core.equilibrium import (
            compute_private_payoff, PayoffParameters
        )
        from coopetition_gym.core.value_functions import ValueFunctionParameters
        from coopetition_gym.core.interdependence import create_symmetric_interdependence
        
        params = PayoffParameters(
            value_params=ValueFunctionParameters(gamma=0.5),
            endowments=np.array([100.0, 100.0]),
            alpha=np.array([0.5, 0.5]),
            interdependence=create_symmetric_interdependence(2, 0.5)
        )
        
        actions = np.array([50.0, 50.0])
        payoff = compute_private_payoff(0, actions, params)
        
        # Payoff should be positive for reasonable cooperation
        assert payoff > 0
    
    def test_integrated_utility_includes_others(self):
        """Test integrated utility weighs others' payoffs."""
        from coopetition_gym.core.equilibrium import (
            compute_private_payoff, compute_integrated_utility, PayoffParameters
        )
        from coopetition_gym.core.value_functions import ValueFunctionParameters
        from coopetition_gym.core.interdependence import create_symmetric_interdependence
        
        params = PayoffParameters(
            value_params=ValueFunctionParameters(gamma=0.5),
            endowments=np.array([100.0, 100.0]),
            alpha=np.array([0.5, 0.5]),
            interdependence=create_symmetric_interdependence(2, 0.5)
        )
        
        actions = np.array([50.0, 50.0])
        private = compute_private_payoff(0, actions, params)
        integrated = compute_integrated_utility(0, actions, params)
        
        # Integrated should include weighted other payoffs
        assert integrated > private
    
    def test_rewards_computation(self):
        """Test compute_rewards returns correct shape."""
        from coopetition_gym.core.equilibrium import (
            compute_rewards, PayoffParameters
        )
        from coopetition_gym.core.value_functions import ValueFunctionParameters
        from coopetition_gym.core.interdependence import create_symmetric_interdependence
        
        n_agents = 3
        params = PayoffParameters(
            value_params=ValueFunctionParameters(),
            endowments=np.full(n_agents, 100.0),
            alpha=np.full(n_agents, 1/n_agents),
            interdependence=create_symmetric_interdependence(n_agents, 0.4)
        )
        
        actions = np.array([40.0, 50.0, 60.0])
        rewards = compute_rewards(actions, params)
        
        assert len(rewards) == n_agents
        assert all(np.isfinite(rewards))


# =============================================================================
# Test Environments: Base
# =============================================================================

class TestBaseEnvironment:
    """Test suite for base CoopetitionEnv class."""
    
    def test_gymnasium_api_compliance(self):
        """Test environment follows Gymnasium API."""
        from coopetition_gym.envs import CoopetitionEnv
        
        env = CoopetitionEnv()
        
        # Check spaces exist
        assert env.observation_space is not None
        assert env.action_space is not None
        
        # Check reset returns correct format
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        
        # Check step returns correct format
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5  # obs, reward, terminated, truncated, info
        
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_observation_space_shape(self):
        """Test observation has correct dimensions."""
        from coopetition_gym.envs import CoopetitionEnv, EnvironmentConfig
        
        n_agents = 3
        config = EnvironmentConfig(n_agents=n_agents)
        env = CoopetitionEnv(config=config)
        
        obs, _ = env.reset()
        
        # Expected: actions(n) + trust(n²) + reputation(n²) + interdep(n²) + step(1)
        expected_dim = n_agents + 3 * (n_agents ** 2) + 1
        assert obs.shape == (expected_dim,)
    
    def test_action_clipping(self):
        """Test actions are clipped to valid range."""
        from coopetition_gym.envs import CoopetitionEnv
        
        env = CoopetitionEnv()
        env.reset(seed=42)
        
        # Try invalid action (negative and over endowment)
        invalid_action = np.array([-50.0, 200.0])
        obs, _, _, _, info = env.step(invalid_action)
        
        # Actions in info should be clipped
        actual_actions = info.get("actions", invalid_action)
        assert all(actual_actions >= 0)
        assert all(actual_actions <= env.endowments)
    
    def test_truncation_at_max_steps(self):
        """Test episode truncates at max_steps."""
        from coopetition_gym.envs import CoopetitionEnv, EnvironmentConfig
        
        max_steps = 10
        config = EnvironmentConfig(max_steps=max_steps)
        env = CoopetitionEnv(config=config)
        
        env.reset(seed=42)
        
        for i in range(max_steps + 5):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            
            if truncated or terminated:
                break
        
        assert truncated  # Should truncate at max_steps
        assert i == max_steps - 1  # 0-indexed
    
    def test_seeding_reproducibility(self):
        """Test same seed produces same trajectory."""
        from coopetition_gym.envs import CoopetitionEnv
        
        env1 = CoopetitionEnv()
        env2 = CoopetitionEnv()
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        assert_array_almost_equal(obs1, obs2)
        
        # Same random action with same seed - seed the action space, not np.random
        env1.action_space.seed(123)
        action1 = env1.action_space.sample()
        env2.action_space.seed(123)
        action2 = env2.action_space.sample()
        
        result1 = env1.step(action1)
        result2 = env2.step(action2)
        
        assert_array_almost_equal(result1[0], result2[0])  # Observations


# =============================================================================
# Test Environments: All 10 Environments
# =============================================================================

class TestAllEnvironments:
    """Test all 10 environments for basic functionality."""
    
    @pytest.fixture
    def env_ids(self):
        """List of all environment IDs."""
        return [
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
    
    def test_all_environments_instantiate(self, env_ids):
        """Test all environments can be created."""
        from coopetition_gym.envs import make
        
        for env_id in env_ids:
            env = make(env_id)
            assert env is not None
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
    
    def test_all_environments_reset(self, env_ids):
        """Test all environments reset properly."""
        from coopetition_gym.envs import make
        
        for env_id in env_ids:
            env = make(env_id)
            obs, info = env.reset(seed=42)
            
            assert isinstance(obs, np.ndarray)
            assert obs.shape == env.observation_space.shape
            assert isinstance(info, dict)
    
    def test_all_environments_step(self, env_ids):
        """Test all environments step properly."""
        from coopetition_gym.envs import make
        
        for env_id in env_ids:
            env = make(env_id)
            env.reset(seed=42)
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(obs, np.ndarray)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
    
    def test_all_environments_run_episode(self, env_ids):
        """Test all environments can run a full episode."""
        from coopetition_gym.envs import make
        
        for env_id in env_ids:
            env = make(env_id, max_steps=10)  # Short for testing
            env.reset(seed=42)
            
            done = False
            steps = 0
            while not done and steps < 20:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            
            assert steps > 0


class TestSpecificEnvironments:
    """Test specific environment behaviors."""
    
    def test_trust_dilemma_cooperation_effect(self):
        """Test TrustDilemma: cooperation builds trust."""
        from coopetition_gym.envs import TrustDilemmaEnv
        
        env = TrustDilemmaEnv()
        _, info = env.reset(seed=42)
        initial_trust = info["mean_trust"]
        
        # Cooperate above baseline for several steps
        for _ in range(10):
            _, _, _, _, info = env.step([70.0, 70.0])
        
        final_trust = info["mean_trust"]
        assert final_trust > initial_trust
    
    def test_partner_holdup_asymmetry(self):
        """Test PartnerHoldUp: asymmetric dependencies."""
        from coopetition_gym.envs import PartnerHoldUpEnv
        
        env = PartnerHoldUpEnv()
        _, info = env.reset(seed=42)
        
        # Check asymmetry exists
        assert info.get("power_asymmetry", 0) != 0
    
    def test_platform_ecosystem_scaling(self):
        """Test PlatformEcosystem: scales with developers."""
        from coopetition_gym.envs import PlatformEcosystemEnv
        
        env = PlatformEcosystemEnv(n_developers=5)
        
        assert env.n_agents == 6  # Platform + 5 developers
        
        env.reset(seed=42)
        action = np.full(6, 50.0)
        _, _, _, _, info = env.step(action)
        
        assert "platform_investment" in info
        assert "mean_developer_investment" in info
    
    def test_recovery_race_initial_conditions(self):
        """Test RecoveryRace: starts in crisis state."""
        from coopetition_gym.envs import RecoveryRaceEnv
        
        env = RecoveryRaceEnv()
        _, info = env.reset(seed=42)
        
        # Should start with low trust
        assert info["mean_trust"] < 0.5
        # Should start with reputation damage
        assert info["mean_reputation_damage"] > 0.2
    
    def test_synergy_search_hidden_gamma(self):
        """Test SynergySearch: gamma is hidden by default."""
        from coopetition_gym.envs import SynergySearchEnv
        
        env = SynergySearchEnv(reveal_gamma_in_obs=False)
        _, _ = env.reset(seed=42)
        
        _, _, _, _, info = env.step([50.0, 50.0])
        
        # Gamma should be None (hidden)
        assert info.get("true_gamma") is None
    
    def test_slcd_validated_params(self):
        """Test SLCD: uses TR-1 validated parameters."""
        from coopetition_gym.envs import SLCDEnv
        
        env = SLCDEnv()
        
        # Check validated gamma
        assert_allclose(env.value_params.gamma, 0.65, rtol=0.01)
        
        # Check agent names
        _, info = env.reset(seed=42)
        assert info["agent_names"] == ["Samsung", "Sony"]
    
    def test_renault_nissan_phases(self):
        """Test RenaultNissan: different phases have different dynamics."""
        from coopetition_gym.envs import RenaultNissanEnv
        
        env_formation = RenaultNissanEnv(phase="formation")
        env_crisis = RenaultNissanEnv(phase="crisis")
        
        _, info_f = env_formation.reset(seed=42)
        _, info_c = env_crisis.reset(seed=42)
        
        # Formation should have higher initial trust than crisis
        assert info_f["mean_trust"] > info_c["mean_trust"]


# =============================================================================
# Test Utilities
# =============================================================================

class TestUtilities:
    """Test suite for utility functions."""
    
    def test_run_episode(self):
        """Test episode runner."""
        from coopetition_gym.envs import TrustDilemmaEnv
        from coopetition_gym.utils import run_episode
        
        env = TrustDilemmaEnv(max_steps=20)
        result = run_episode(env, seed=42)
        
        assert result.total_steps <= 20
        assert len(result.actions_history) == result.total_steps
        assert len(result.rewards_history) == result.total_steps
    
    def test_constant_policy(self):
        """Test constant policy factory."""
        from coopetition_gym.utils import make_constant_policy
        
        policy = make_constant_policy(60.0)
        action = policy(np.zeros(10))  # Dummy observation
        
        assert_array_almost_equal(action, [60.0, 60.0])
    
    def test_proportional_policy(self):
        """Test proportional policy factory."""
        from coopetition_gym.utils import make_proportional_policy
        
        policy = make_proportional_policy(0.6)
        action = policy(np.zeros(10))
        
        assert_array_almost_equal(action, [60.0, 60.0])


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_training_loop_simulation(self):
        """Simulate a training loop structure."""
        from coopetition_gym.envs import make
        
        env = make("TrustDilemma-v0", max_steps=50)
        
        total_rewards = []
        
        for episode in range(3):
            obs, _ = env.reset(seed=episode)
            episode_reward = 0
            done = False
            
            while not done:
                # Simple policy: cooperate proportionally
                action = np.array([50.0, 50.0])
                obs, rewards, terminated, truncated, _ = env.step(action)
                episode_reward += np.sum(rewards)
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
        
        assert len(total_rewards) == 3
        assert all(r > 0 for r in total_rewards)
    
    def test_comparison_across_environments(self):
        """Compare metrics across different environments."""
        from coopetition_gym.envs import make
        from coopetition_gym.utils import run_episode, make_constant_policy
        
        policy = make_constant_policy(50.0)
        
        results = {}
        for env_id in ["TrustDilemma-v0", "SLCD-v0"]:
            env = make(env_id, max_steps=30)
            result = run_episode(env, policy, seed=42)
            results[env_id] = result.final_trust
        
        # Both should have reasonable trust levels with cooperation
        for trust in results.values():
            assert trust > 0.3


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
