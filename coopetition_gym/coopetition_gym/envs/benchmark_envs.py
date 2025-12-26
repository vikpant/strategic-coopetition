"""
================================================================================
COOPETITION-GYM: Research Benchmark Environments
================================================================================

This module implements environments specifically designed to test the limits of
RL algorithms against the specific phenomena validated in the TR-1 and TR-2
research papers. These serve as diagnostic tools for algorithmic development.

Environments:
-------------
1. RecoveryRace-v0: Post-crisis recovery coordination under trust constraints
2. SynergySearch-v0: Exploration vs exploitation with unknown complementarity

These environments present focused challenges that isolate specific aspects
of the coopetition framework for algorithmic analysis.

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.base import CoopetitionEnv, EnvironmentConfig
from core.value_functions import ValueFunctionParameters, ValueSpecification
from core.trust_dynamics import TrustParameters, TrustState, RecoveryModel
from core.interdependence import create_symmetric_interdependence


class RecoveryRaceEnv(CoopetitionEnv):
    """
    Recovery Race Environment (RecoveryRace-v0)
    
    The environment starts in a "Post-Crisis" state with low trust (T ≈ 0.25)
    and high reputation damage (R ≈ 0.50). Agents must coordinate to restore
    productivity while navigating the trust ceiling constraints.
    
    Challenge:
    ----------
    This is a pure policy optimization challenge: What is the optimal sequence
    of cooperative actions to maximize the rate of recovery given the mathematical
    constraints of the trust ceiling? This benchmarks planning algorithms and
    tests understanding of the TR-2 trust dynamics.
    
    Key Dynamics:
    -------------
    - Trust Ceiling (TR-2): Θ_ij = 1 - R_ij limits how high trust can grow
    - Slow Reputation Decay: R decays slowly, requiring patience
    - Recovery Coordination: Both agents must cooperate for mutual recovery
    - Optimal sequencing: The order of actions affects total recovery
    
    Scenario:
    ---------
    Think of two firms after a major breach of trust (e.g., contract violation,
    scandal) that are trying to rebuild their partnership. They must find the
    optimal recovery path given that:
    - Past damage creates lasting constraints
    - Recovery requires sustained mutual effort
    - Impatient defection can re-damage the relationship
    
    Benchmark Questions:
    --------------------
    1. How quickly can an optimal policy recover to 90% trust?
    2. What's the welfare loss from starting in crisis vs. normal?
    3. Can agents learn the patience required for slow reputation healing?
    
    Example:
    --------
    >>> env = RecoveryRaceEnv()
    >>> obs, info = env.reset(seed=42)
    >>> print(f"Initial trust: {info['mean_trust']:.3f}")  # ~0.25
    >>> print(f"Initial damage: {info['mean_reputation_damage']:.3f}")  # ~0.50
    >>> 
    >>> # Sustained cooperation strategy
    >>> for _ in range(100):
    ...     obs, rewards, done, truncated, info = env.step([70.0, 70.0])
    >>> print(f"Final trust: {info['mean_trust']:.3f}")  # Should improve
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "RecoveryRace-v0"
    }
    
    def __init__(
        self,
        initial_trust: float = 0.25,
        initial_reputation_damage: float = 0.50,
        recovery_target: float = 0.90,
        max_steps: int = 150,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Recovery Race environment.
        
        Args:
            initial_trust: Starting trust level (post-crisis, low)
            initial_reputation_damage: Starting reputation damage (high)
            recovery_target: Trust level to consider "recovered"
            max_steps: Maximum timesteps per episode
            render_mode: Rendering mode
        """
        # Trust parameters optimized for recovery dynamics
        trust_params = TrustParameters(
            lambda_plus=0.08,  # Slow recovery
            lambda_minus=0.35,  # Re-violation is costly
            mu_R=0.70,  # High damage from new violations
            delta_R=0.01,  # Very slow forgetting (patience required)
            xi=0.40,
            kappa=1.0,
            initial_trust=initial_trust,
            initial_reputation=initial_reputation_damage
        )
        
        # Standard value parameters
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=20.0,
            gamma=0.60
        )
        
        # Symmetric setup for clean benchmark
        D = create_symmetric_interdependence(2, 0.55).matrix
        
        config = EnvironmentConfig(
            n_agents=2,
            max_steps=max_steps,
            endowments=np.array([100.0, 100.0]),
            alpha=np.array([0.50, 0.50]),
            interdependence_matrix=D,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=np.array([35.0, 35.0]),
            reward_type="integrated",
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        
        self.initial_trust = initial_trust
        self.initial_reputation_damage = initial_reputation_damage
        self.recovery_target = recovery_target
        
        # Track recovery metrics
        self._peak_trust = initial_trust
        self._recovery_step = None
    
    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        """Reset to post-crisis state."""
        result = super().reset(**kwargs)
        
        # Override trust state to start in crisis
        if self._trust_state is not None:
            self._trust_state.trust_matrix.fill(self.initial_trust)
            np.fill_diagonal(self._trust_state.trust_matrix, 1.0)
            
            self._trust_state.reputation_matrix.fill(self.initial_reputation_damage)
            np.fill_diagonal(self._trust_state.reputation_matrix, 0.0)
            
            self._state["trust"] = self._trust_state.trust_matrix.copy()
            self._state["reputation"] = self._trust_state.reputation_matrix.copy()
        
        self._peak_trust = self.initial_trust
        self._recovery_step = None
        
        return result
    
    def step(self, action: NDArray) -> Tuple[NDArray, NDArray, bool, bool, Dict]:
        """Step with recovery tracking."""
        result = super().step(action)
        obs, rewards, terminated, truncated, info = result
        
        # Track recovery progress
        current_trust = info["mean_trust"]
        if current_trust > self._peak_trust:
            self._peak_trust = current_trust
        
        # Check if recovery target reached
        if current_trust >= self.recovery_target and self._recovery_step is None:
            self._recovery_step = self._step_count
        
        # Add recovery-specific info
        info["peak_trust"] = self._peak_trust
        info["recovery_step"] = self._recovery_step
        info["trust_ceiling"] = 1.0 - info["mean_reputation_damage"]
        info["recovery_progress"] = (current_trust - self.initial_trust) / (
            self.recovery_target - self.initial_trust
        )
        
        return obs, rewards, terminated, truncated, info
    
    def _check_terminated(self) -> bool:
        """Optionally terminate on successful recovery."""
        if self._trust_state is not None:
            if self._trust_state.mean_trust() >= self.recovery_target:
                return True  # Success!
        return False


class SynergySearchEnv(CoopetitionEnv):
    """
    Synergy Search Environment (SynergySearch-v0)
    
    Agents must invest resources without knowing the true Complementarity
    parameter (γ) of their partnership. They must infer whether they are
    in a "High Synergy" or "Low Synergy" partnership from observed rewards.
    
    Challenge:
    ----------
    This tests exploration vs. exploitation in a coopetitive setting:
    - High γ: Worth investing heavily for synergy gains
    - Low γ: Should play conservatively, focus on individual value
    
    Agents must probe the environment to discover the synergy potential
    and adjust their strategies accordingly. This requires balancing
    information-gathering actions with value-maximizing actions.
    
    Key Dynamics:
    -------------
    - Hidden γ parameter drawn at episode start
    - Rewards depend on γ but γ is not directly observable
    - Must infer γ from reward patterns across actions
    - Optimal policy depends on discovered γ value
    
    Scenario:
    ---------
    Two firms considering a joint venture don't know how complementary
    their capabilities truly are. They must experiment with cooperation
    levels to discover their synergy potential before committing to a
    long-term strategy.
    
    Example:
    --------
    >>> env = SynergySearchEnv(gamma_range=(0.3, 0.9))
    >>> obs, info = env.reset(seed=42)
    >>> print(f"True gamma (hidden): {env._true_gamma:.2f}")
    >>> 
    >>> # Exploration: try different cooperation levels
    >>> for coop in [30, 50, 70]:
    ...     obs, rewards, _, _, info = env.step([coop, coop])
    ...     print(f"Coop={coop}, Reward={rewards[0]:.2f}")
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "SynergySearch-v0"
    }
    
    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.20, 0.90),
        reveal_gamma_in_obs: bool = False,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Synergy Search environment.
        
        Args:
            gamma_range: (min, max) range for hidden complementarity parameter
            reveal_gamma_in_obs: If True, include γ in observation (easier mode)
            max_steps: Maximum timesteps per episode
            render_mode: Rendering mode
        """
        self.gamma_range = gamma_range
        self.reveal_gamma_in_obs = reveal_gamma_in_obs
        self._true_gamma = None  # Set in reset
        
        # Default value params (gamma will be overwritten each episode)
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=20.0,
            gamma=0.5  # Placeholder
        )
        
        # Standard trust dynamics
        trust_params = TrustParameters(
            lambda_plus=0.10,
            lambda_minus=0.30,
            initial_trust=0.55
        )
        
        # Symmetric setup
        D = create_symmetric_interdependence(2, 0.50).matrix
        
        config = EnvironmentConfig(
            n_agents=2,
            max_steps=max_steps,
            endowments=np.array([100.0, 100.0]),
            alpha=np.array([0.50, 0.50]),
            interdependence_matrix=D,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=np.array([40.0, 40.0]),
            reward_type="integrated",
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        
        # Track belief/exploration metrics
        self._reward_history = []
        self._action_history = []
        
        # Extend observation space if gamma is revealed
        if self.reveal_gamma_in_obs:
            base_dim = self.n_agents + 3 * (self.n_agents ** 2) + 1
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(base_dim + 1,), dtype=np.float32
            )
    
    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        """Reset with new random gamma."""
        # Sample new gamma
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        
        self._true_gamma = self._np_random.uniform(
            self.gamma_range[0], 
            self.gamma_range[1]
        )
        
        # Update value params with true gamma
        self.value_params = ValueFunctionParameters(
            specification=self.value_params.specification,
            theta=self.value_params.theta,
            gamma=self._true_gamma
        )
        self.payoff_params.value_params = self.value_params
        
        # Reset tracking
        self._reward_history = []
        self._action_history = []
        
        return super().reset(**kwargs)
    
    def step(self, action: NDArray) -> Tuple[NDArray, NDArray, bool, bool, Dict]:
        """Step with synergy tracking."""
        result = super().step(action)
        obs, rewards, terminated, truncated, info = result
        
        # Track history for analysis
        self._reward_history.append(rewards.copy())
        self._action_history.append(np.asarray(action).copy())
        
        # Add synergy-specific info
        info["true_gamma"] = self._true_gamma if self.reveal_gamma_in_obs else None
        info["gamma_type"] = "high" if self._true_gamma > 0.6 else "low"
        info["cumulative_rewards"] = np.sum(self._reward_history, axis=0).tolist()
        
        # Compute reward variance (useful for belief inference)
        if len(self._reward_history) > 3:
            info["reward_variance"] = float(np.var(self._reward_history))
        
        return obs, rewards, terminated, truncated, info
    
    def _get_legacy_observation(self) -> NDArray[np.float32]:
        """Optionally include gamma in observation."""
        base_obs = super()._get_legacy_observation()
        
        if self.reveal_gamma_in_obs and self._true_gamma is not None:
            obs = np.concatenate([base_obs, [self._true_gamma]])
            return obs.astype(np.float32)
        
        return base_obs
    
# =============================================================================
# Environment Registration Helpers
# =============================================================================

def make_recovery_race(**kwargs) -> RecoveryRaceEnv:
    """Factory function for RecoveryRace-v0."""
    return RecoveryRaceEnv(**kwargs)


def make_synergy_search(**kwargs) -> SynergySearchEnv:
    """Factory function for SynergySearch-v0."""
    return SynergySearchEnv(**kwargs)
