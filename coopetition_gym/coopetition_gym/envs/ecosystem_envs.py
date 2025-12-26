"""
================================================================================
COOPETITION-GYM: Ecosystem Environments (Macro Category)
================================================================================

This module implements the "macro" environments that scale to N agents, testing
how TR-1/TR-2 dynamics play out in groups. These environments appeal to
researchers studying emergent behavior, social dynamics, and mechanism design.

Environments:
-------------
1. PlatformEcosystem-v0: Platform agent with N developer agents
2. DynamicPartnerSelection-v0: Pool of agents forming dyads based on reputation

These environments introduce multi-agent complexity while maintaining
the core coopetitive dynamics from the research foundation.

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, Optional, Tuple, List
import gymnasium as gym
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.base import CoopetitionEnv, EnvironmentConfig
from core.value_functions import ValueFunctionParameters, ValueSpecification
from core.trust_dynamics import TrustParameters
from core.interdependence import create_hub_spoke_interdependence


class PlatformEcosystemEnv(CoopetitionEnv):
    """
    Platform Ecosystem Environment (PlatformEcosystem-v0)
    
    Models a platform economy where one central "Platform" agent sets
    infrastructure investment, and N "Developer" agents set app quality
    or contribution levels. This captures the dynamics of app stores,
    marketplace platforms, and developer ecosystems.
    
    Challenge:
    ----------
    The Platform agent faces a classic mechanism design problem: it must
    learn a policy that balances short-term revenue extraction (taking a
    large cut) against long-term ecosystem health (developer trust and
    participation). Over-extraction drives developers away; under-extraction
    leaves value on the table.
    
    Key Dynamics:
    -------------
    - Hub-spoke interdependence: Developers depend heavily on platform
    - Platform depends moderately on each developer (collectively important)
    - Platform takes a cut (α_platform) of total value created
    - Network effects through complementarity parameter γ
    - Developer trust in platform affects their investment decisions
    
    Scenario:
    ---------
    Agent 0: Platform (e.g., App Store)
    - Sets infrastructure investment affecting value multiplier
    - Takes α_platform share of synergistic surplus
    
    Agents 1-N: Developers
    - Set quality/effort investment
    - Depend on platform for distribution
    - Each receives (1-α_platform)/(N-1) share of surplus
    
    Example:
    --------
    >>> env = PlatformEcosystemEnv(n_developers=4)
    >>> obs, info = env.reset(seed=42)
    >>> 
    >>> # Platform invests heavily, developers respond
    >>> actions = np.array([80.0, 60.0, 55.0, 65.0, 50.0])  # [platform, dev1, dev2, dev3, dev4]
    >>> obs, rewards, done, truncated, info = env.step(actions)
    >>> print(f"Platform reward: {rewards[0]:.2f}")
    >>> print(f"Developer rewards: {rewards[1:].mean():.2f}")
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "PlatformEcosystem-v0"
    }
    
    def __init__(
        self,
        n_developers: int = 4,
        platform_share: float = 0.30,
        platform_dependency: float = 0.25,
        developer_dependency: float = 0.75,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Platform Ecosystem environment.
        
        Args:
            n_developers: Number of developer agents
            platform_share: α_platform - platform's share of synergistic surplus
            platform_dependency: How much platform depends on each developer
            developer_dependency: How much each developer depends on platform
            max_steps: Maximum timesteps per episode
            render_mode: Rendering mode
        """
        n_agents = 1 + n_developers  # Platform + developers
        
        # Hub-spoke interdependence
        interdep = create_hub_spoke_interdependence(
            n_spokes=n_developers,
            hub_dependency=platform_dependency,
            spoke_dependency=developer_dependency
        )
        
        # Trust parameters for platform dynamics
        trust_params = TrustParameters(
            lambda_plus=0.08,  # Slower trust building (institutional)
            lambda_minus=0.25,
            mu_R=0.45,  # Moderate reputation effects
            delta_R=0.02,
            xi=0.40,
            kappa=1.0,
            initial_trust=0.60  # Start with baseline platform trust
        )
        
        # Value parameters with network effects
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=25.0,  # Higher scale for platform economies
            gamma=0.75  # Strong complementarity (network effects)
        )
        
        # Platform has larger endowment (infrastructure budget)
        endowments = np.array(
            [150.0] + [80.0] * n_developers,
            dtype=np.float32
        )
        
        # Bargaining shares: platform takes its cut, rest split among developers
        developer_share = (1.0 - platform_share) / n_developers
        alpha = np.array(
            [platform_share] + [developer_share] * n_developers,
            dtype=np.float32
        )
        
        config = EnvironmentConfig(
            n_agents=n_agents,
            max_steps=max_steps,
            endowments=endowments,
            alpha=alpha,
            interdependence_matrix=interdep.matrix,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=endowments * 0.30,
            reward_type="integrated",
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        
        # Store ecosystem metrics
        self.n_developers = n_developers
        self.platform_share = platform_share
        self._developer_exits = 0
    
    def _get_legacy_info(self) -> Dict[str, Any]:
        """Add ecosystem-specific information."""
        info = super()._get_legacy_info()
        
        # Platform vs developer metrics
        if "actions" in self._state:
            actions = self._state["actions"]
            info["platform_investment"] = float(actions[0])
            info["mean_developer_investment"] = float(np.mean(actions[1:]))
            info["developer_investment_std"] = float(np.std(actions[1:]))
        
        # Trust from developers to platform
        if self._trust_state is not None:
            dev_trust_in_platform = np.mean([
                self._trust_state.get_trust(i, 0) 
                for i in range(1, self.n_agents)
            ])
            info["developer_trust_in_platform"] = float(dev_trust_in_platform)
        
        info["developer_exits"] = self._developer_exits
        
        return info
    
    def _check_terminated(self) -> bool:
        """
        Terminate if developer trust collapses (ecosystem death).
        """
        if self._trust_state is not None:
            # Average developer trust in platform
            dev_trust = np.mean([
                self._trust_state.get_trust(i, 0) 
                for i in range(1, self.n_agents)
            ])
            if dev_trust < 0.15:
                self._developer_exits = self.n_developers
                return True
        return False


class DynamicPartnerSelectionEnv(CoopetitionEnv):
    """
    Dynamic Partner Selection Environment (DynamicPartnerSelection-v0)
    
    A pool of N agents where, in each episode, agents observe the "Reputation
    Scores" of all peers and must engage in a collaborative task. This tests
    social learning - agents must learn to interpret reputation signals and
    maintain their own reputation to attract quality partners.
    
    Challenge:
    ----------
    Agents must solve two interrelated problems:
    1. Learn to interpret reputation signals to identify good partners
    2. Maintain their own reputation to be selected by quality partners
    
    This creates a reputation-driven ecosystem where past behavior affects
    future opportunities through partner selection dynamics.
    
    Key Dynamics:
    -------------
    - Global reputation tracking across episodes
    - Partner quality affects value creation potential
    - Reputation serves as a signal for partner selection
    - Social learning required to interpret reputation correctly
    
    Scenario:
    ---------
    Think of this as a marketplace where agents form temporary partnerships:
    - Freelancers selecting clients
    - Firms forming joint ventures
    - Researchers choosing collaborators
    
    Example:
    --------
    >>> env = DynamicPartnerSelectionEnv(n_agents=6)
    >>> obs, info = env.reset(seed=42)
    >>> 
    >>> # Agents decide on cooperation levels
    >>> actions = np.array([50.0, 45.0, 60.0, 40.0, 55.0, 48.0])
    >>> obs, rewards, done, truncated, info = env.step(actions)
    >>> print(f"Reputation scores: {info['reputation_scores']}")
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "DynamicPartnerSelection-v0"
    }
    
    def __init__(
        self,
        n_agents: int = 6,
        reputation_weight: float = 0.5,
        max_steps: int = 50,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Dynamic Partner Selection environment.
        
        Args:
            n_agents: Number of agents in the pool
            reputation_weight: How much reputation affects partner compatibility
            max_steps: Maximum timesteps per episode
            render_mode: Rendering mode
        """
        # All agents connected with moderate baseline dependency
        D = np.full((n_agents, n_agents), 0.40, dtype=np.float32)
        np.fill_diagonal(D, 0.0)
        
        # Trust parameters emphasizing reputation
        trust_params = TrustParameters(
            lambda_plus=0.12,
            lambda_minus=0.35,
            mu_R=0.60,  # Strong reputation effects
            delta_R=0.015,  # Very slow forgetting
            xi=0.45,
            kappa=1.2,
            initial_trust=0.50
        )
        
        # Value parameters
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=18.0,
            gamma=0.55  # Moderate complementarity
        )
        
        # Equal endowments and shares
        endowments = np.full(n_agents, 100.0, dtype=np.float32)
        alpha = np.full(n_agents, 1.0 / n_agents, dtype=np.float32)
        
        config = EnvironmentConfig(
            n_agents=n_agents,
            max_steps=max_steps,
            endowments=endowments,
            alpha=alpha,
            interdependence_matrix=D,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=endowments * 0.35,
            reward_type="integrated",
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        
        self.reputation_weight = reputation_weight
        self._global_reputation = np.zeros(n_agents, dtype=np.float32)
        self._interaction_history = []
        
        # Extend observation space to include reputation scores
        base_dim = self.n_agents + 3 * (self.n_agents ** 2) + 1
        extended_dim = base_dim + self.n_agents
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(extended_dim,), dtype=np.float32
        )
    
    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        """
        Reset environment, optionally preserving global reputation.
        
        The global reputation can persist across episodes to create
        long-term reputation dynamics.
        """
        result = super().reset(**kwargs)
        
        # Optionally reset global reputation
        options = kwargs.get("options", {})
        if options.get("reset_reputation", False):
            self._global_reputation = np.zeros(self.n_agents, dtype=np.float32)
        
        self._interaction_history = []
        
        return result
    
    def step(self, action: NDArray) -> Tuple[NDArray, NDArray, bool, bool, Dict]:
        """Step with reputation tracking."""
        result = super().step(action)
        obs, rewards, terminated, truncated, info = result
        
        actions = np.asarray(action, dtype=np.float32)
        
        # Update global reputation based on cooperation levels
        cooperation_scores = actions / self.endowments
        self._global_reputation = 0.9 * self._global_reputation + 0.1 * cooperation_scores
        
        # Track interactions
        self._interaction_history.append({
            "step": self._step_count,
            "actions": actions.copy(),
            "cooperation_scores": cooperation_scores.copy()
        })
        
        # Add reputation info
        info["reputation_scores"] = self._global_reputation.copy()
        info["reputation_ranking"] = np.argsort(-self._global_reputation).tolist()
        
        return obs, rewards, terminated, truncated, info
    
    def _get_legacy_observation(self) -> NDArray[np.float32]:
        """Include reputation scores in observation."""
        base_obs = super()._get_legacy_observation()
        
        # Append global reputation scores
        obs = np.concatenate([base_obs, self._global_reputation])
        
        return obs.astype(np.float32)
    
# =============================================================================
# Environment Registration Helpers
# =============================================================================

def make_platform_ecosystem(**kwargs) -> PlatformEcosystemEnv:
    """Factory function for PlatformEcosystem-v0."""
    return PlatformEcosystemEnv(**kwargs)


def make_dynamic_partner_selection(**kwargs) -> DynamicPartnerSelectionEnv:
    """Factory function for DynamicPartnerSelection-v0."""
    return DynamicPartnerSelectionEnv(**kwargs)
