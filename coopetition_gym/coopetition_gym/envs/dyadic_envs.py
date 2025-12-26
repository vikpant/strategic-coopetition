"""
================================================================================
COOPETITION-GYM: Dyadic Environments (Micro Category)
================================================================================

This module implements the "micro" environments focusing on fundamental mechanics
between two agents. These serve as unit tests for researchers to verify if their
algorithms can solve basic coopetitive dilemmas.

Environments:
-------------
1. TrustDilemma-v0: Continuous iterated Prisoner's Dilemma with trust dynamics
2. PartnerHoldUp-v0: Asymmetric vertical relationship with structural vulnerability

These environments are intentionally simpler than the ecosystem environments,
making them ideal for debugging algorithms and understanding core mechanics.

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
from core.trust_dynamics import TrustParameters, TrustDilemmaModel
from core.interdependence import (
    create_symmetric_interdependence,
    create_asymmetric_interdependence
)


class TrustDilemmaEnv(CoopetitionEnv):
    """
    Trust Dilemma Environment (TrustDilemma-v0)
    
    A continuous, iterated Prisoner's Dilemma where payoffs are non-stationary
    and evolve based on a hidden "Trust" state. This environment tests whether
    agents can learn long-horizon impulse control.
    
    Challenge:
    ----------
    A "greedy" RL agent will defect (low cooperation) to maximize immediate
    payoff, but this erodes trust and permanently ruins future rewards.
    Successful agents must learn to forego immediate payoff to maintain the
    "Trust Asset" that enables sustained mutual cooperation.
    
    Key Dynamics:
    -------------
    - Trust builds slowly with sustained cooperation (λ+ = 0.15)
    - Trust erodes quickly with any defection (λ- = 0.45)
    - Reputation damage creates permanent trust ceiling (hysteresis)
    - Payoffs are modulated by current trust level
    
    State Space:
    ------------
    Standard coopetition observation plus hidden trust state.
    
    Action Space:
    -------------
    Cooperation levels [0, 100] for each of 2 agents.
    
    Reward Structure:
    -----------------
    Integrated utility weighted by current trust level.
    High trust enables higher mutual payoffs; low trust caps potential gains.
    
    Example:
    --------
    >>> env = TrustDilemmaEnv()
    >>> obs, info = env.reset(seed=42)
    >>> 
    >>> # Cooperative strategy: invest above baseline
    >>> for _ in range(50):
    ...     obs, rewards, done, truncated, info = env.step([60.0, 60.0])
    >>> print(f"Final trust: {info['mean_trust']:.3f}")  # Should be high
    >>> 
    >>> # Defecting strategy: invest below baseline
    >>> obs, info = env.reset(seed=42)
    >>> for _ in range(50):
    ...     obs, rewards, done, truncated, info = env.step([20.0, 20.0])
    >>> print(f"Final trust: {info['mean_trust']:.3f}")  # Should be low
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "TrustDilemma-v0"
    }
    
    def __init__(
        self,
        max_steps: int = 100,
        trust_sensitivity: float = 1.5,
        trust_building_rate: float = 0.15,
        trust_erosion_rate: float = 0.45,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Trust Dilemma environment.
        
        Args:
            max_steps: Maximum timesteps per episode
            trust_sensitivity: κ parameter controlling signal sensitivity
            trust_building_rate: λ+ for trust growth rate
            trust_erosion_rate: λ- for trust erosion rate
            render_mode: Rendering mode
        """
        # Configure trust parameters for dilemma dynamics
        trust_params = TrustParameters(
            lambda_plus=trust_building_rate,
            lambda_minus=trust_erosion_rate,
            mu_R=0.50,  # Moderate reputation damage
            delta_R=0.02,  # Slow forgetting
            xi=0.60,  # High interdependence amplification
            kappa=trust_sensitivity,
            initial_trust=0.50  # Neutral starting point
        )
        
        # Symmetric value parameters
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=20.0,
            gamma=0.70  # High complementarity emphasizes cooperation
        )
        
        # Symmetric interdependence (equal mutual dependency)
        D = create_symmetric_interdependence(2, 0.60).matrix
        
        # Build configuration
        config = EnvironmentConfig(
            n_agents=2,
            max_steps=max_steps,
            endowments=np.array([100.0, 100.0]),
            alpha=np.array([0.50, 0.50]),
            interdependence_matrix=D,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=np.array([40.0, 40.0]),  # 40% of endowment as baseline
            reward_type="integrated",
            normalize_rewards=False,
            reward_scale=1.0,
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        
        # Track defection history for analysis
        self._defection_count = [0, 0]
        self._cooperation_count = [0, 0]
    
    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        """Reset with additional tracking initialization."""
        self._defection_count = [0, 0]
        self._cooperation_count = [0, 0]
        return super().reset(**kwargs)
    
    def step(self, action: NDArray) -> Tuple[NDArray, NDArray, bool, bool, Dict]:
        """Step with defection tracking."""
        actions = np.asarray(action, dtype=np.float32)
        
        # Track cooperation vs defection
        for i in range(2):
            if actions[i] >= self.baselines[i]:
                self._cooperation_count[i] += 1
            else:
                self._defection_count[i] += 1
        
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # Add dilemma-specific info
        info["defection_counts"] = self._defection_count.copy()
        info["cooperation_counts"] = self._cooperation_count.copy()
        info["cooperation_rates"] = [
            c / (c + d) if (c + d) > 0 else 0.0
            for c, d in zip(self._cooperation_count, self._defection_count)
        ]
        
        return obs, rewards, terminated, truncated, info
    
    def _check_terminated(self) -> bool:
        """
        Terminate if trust collapses completely.
        
        This provides a clear failure signal to the agent.
        """
        if self._trust_state is not None:
            mean_trust = self._trust_state.mean_trust()
            # Terminate if trust falls below threshold
            if mean_trust < 0.05:
                return True
        return False


class PartnerHoldUpEnv(CoopetitionEnv):
    """
    Partner Hold-Up Environment (PartnerHoldUp-v0)
    
    An asymmetric vertical relationship where one agent (the "weak" agent) is
    structurally vulnerable to the other (the "strong" agent). This models
    supplier-manufacturer, small-firm-large-firm, or other power-imbalanced
    relationships.
    
    Challenge:
    ----------
    The "weak" agent must learn defensive strategies like building trust
    carefully and maintaining reserves. The "strong" agent must learn not
    to exploit its partner to the point of collapse (killing the golden goose).
    
    Key Dynamics:
    -------------
    - Highly asymmetric interdependence (D[weak,strong] >> D[strong,weak])
    - Trust erosion amplified by interdependence factor ξ
    - The weak agent is more sensitive to violations
    - The strong agent has more structural power but risks partner exit
    
    Scenario:
    ---------
    Agent 0 (Strong): Large manufacturer with alternative suppliers
    Agent 1 (Weak): Small supplier heavily dependent on the manufacturer
    
    The weak supplier depends on the strong manufacturer for orders (D=0.85),
    while the manufacturer has only moderate dependency on this particular
    supplier (D=0.35).
    
    Example:
    --------
    >>> env = PartnerHoldUpEnv()
    >>> obs, info = env.reset(seed=42)
    >>> 
    >>> # Strong agent exploits: high cooperation from weak, low from strong
    >>> for _ in range(50):
    ...     obs, rewards, done, truncated, info = env.step([30.0, 70.0])
    >>> # Strong agent gets short-term gains but erodes relationship
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "PartnerHoldUp-v0"
    }
    
    def __init__(
        self,
        strong_dependency: float = 0.35,
        weak_dependency: float = 0.85,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Partner Hold-Up environment.
        
        Args:
            strong_dependency: How much the strong agent depends on weak (D[0,1])
            weak_dependency: How much the weak agent depends on strong (D[1,0])
            max_steps: Maximum timesteps per episode
            render_mode: Rendering mode
        """
        # Asymmetric interdependence matrix
        D = create_asymmetric_interdependence(
            strong_dependency=weak_dependency,  # Note: parameter naming is from weak's perspective
            weak_dependency=strong_dependency
        ).matrix
        
        # Trust parameters with interdependence amplification
        trust_params = TrustParameters(
            lambda_plus=0.10,
            lambda_minus=0.35,  # Faster erosion
            mu_R=0.55,
            delta_R=0.025,
            xi=0.70,  # High amplification magnifies asymmetry
            kappa=1.2,
            initial_trust=0.55  # Slightly positive starting relationship
        )
        
        # Value parameters favoring cooperation
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=20.0,
            gamma=0.60
        )
        
        # Asymmetric endowments (strong agent has more resources)
        endowments = np.array([120.0, 80.0])  # Strong has more
        
        # Asymmetric bargaining shares (strong agent captures more surplus)
        alpha = np.array([0.60, 0.40])
        
        config = EnvironmentConfig(
            n_agents=2,
            max_steps=max_steps,
            endowments=endowments,
            alpha=alpha,
            interdependence_matrix=D,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=endowments * 0.35,  # 35% baseline
            reward_type="integrated",
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        
        # Store asymmetry metrics
        self._power_asymmetry = weak_dependency - strong_dependency
    
    def _get_legacy_info(self) -> Dict[str, Any]:
        """Add asymmetry-specific information."""
        info = super()._get_legacy_info()
        
        # Compute power differential
        if self._trust_state is not None:
            T = self._trust_state.trust_matrix
            # Trust asymmetry: how much more weak trusts strong vs vice versa
            trust_asymmetry = T[1, 0] - T[0, 1]
            info["trust_asymmetry"] = float(trust_asymmetry)
        
        info["power_asymmetry"] = self._power_asymmetry
        info["strong_agent_payoff"] = info.get("actions", [0, 0])[0] if "actions" in info else 0
        info["weak_agent_payoff"] = info.get("actions", [0, 0])[1] if "actions" in info else 0
        
        return info
    
    def _check_terminated(self) -> bool:
        """
        Terminate if the weak agent's trust in strong collapses.
        
        This represents the weak agent "exiting" the relationship.
        """
        if self._trust_state is not None:
            # Check weak agent's trust in strong agent
            weak_trust_in_strong = self._trust_state.get_trust(1, 0)
            if weak_trust_in_strong < 0.10:
                return True
        return False


# =============================================================================
# Environment Registration Helpers
# =============================================================================

def make_trust_dilemma(**kwargs) -> TrustDilemmaEnv:
    """Factory function for TrustDilemma-v0."""
    return TrustDilemmaEnv(**kwargs)


def make_partner_holdup(**kwargs) -> PartnerHoldUpEnv:
    """Factory function for PartnerHoldUp-v0."""
    return PartnerHoldUpEnv(**kwargs)
