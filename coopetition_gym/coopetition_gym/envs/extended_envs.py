"""
================================================================================
COOPETITION-GYM: Extended Environments
================================================================================

This module implements additional environments that extend the core coopetition
framework with specialized mechanics for advanced research scenarios.

Environments:
-------------
1. CooperativeNegotiation-v0: Multi-round negotiation with commitment mechanics
2. ReputationMarket-v0: Market with observable reputation affecting outcomes

These environments complement the core suite by testing specific aspects of
strategic behavior in coopetitive settings.

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
from core.interdependence import create_symmetric_interdependence


class CooperativeNegotiationEnv(CoopetitionEnv):
    """
    Cooperative Negotiation Environment (CooperativeNegotiation-v0)
    
    A multi-round negotiation where agents make proposals and must reach
    agreement on cooperation levels. This tests communication, commitment,
    and the ability to build trust through negotiation dynamics.
    
    Challenge:
    ----------
    Unlike instant-action environments, agents here must:
    1. Make proposals that signal their intentions
    2. Interpret partner proposals to infer trustworthiness
    3. Build consensus through iterative refinement
    4. Honor commitments once agreement is reached
    
    Mechanics:
    ----------
    Each timestep has two phases:
    - Proposal Phase: Agents submit proposed cooperation levels
    - Execution Phase: If proposals are close enough, agents execute
    
    Agreement is reached when |proposal_i - proposal_j| < threshold.
    Once agreed, deviating from the agreement causes severe trust damage.
    
    This captures real-world negotiation dynamics where:
    - Communication precedes action
    - Agreements create binding expectations
    - Breach of agreement is punished more than mere non-cooperation
    
    Example:
    --------
    >>> env = CooperativeNegotiationEnv()
    >>> obs, info = env.reset(seed=42)
    >>> 
    >>> # First, agents make proposals
    >>> proposals = np.array([60.0, 55.0])  # Agent 0 proposes 60, Agent 1 proposes 55
    >>> obs, rewards, done, truncated, info = env.step(proposals)
    >>> print(f"Agreement reached: {info['agreement_reached']}")
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "CooperativeNegotiation-v0"
    }
    
    def __init__(
        self,
        agreement_threshold: float = 10.0,
        breach_penalty_multiplier: float = 3.0,
        negotiation_rounds: int = 5,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Cooperative Negotiation environment.
        
        Args:
            agreement_threshold: Maximum difference for proposals to be considered agreement
            breach_penalty_multiplier: How much worse breach is vs non-cooperation
            negotiation_rounds: Rounds of proposal before execution
            max_steps: Maximum timesteps per episode
            render_mode: Rendering mode
        """
        trust_params = TrustParameters(
            lambda_plus=0.12,
            lambda_minus=0.40,  # High penalty for violations
            mu_R=0.65,  # Strong reputation effects
            delta_R=0.02,
            xi=0.55,
            kappa=1.2,
            initial_trust=0.50
        )
        
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=20.0,
            gamma=0.60
        )
        
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
        
        self.agreement_threshold = agreement_threshold
        self.breach_penalty_multiplier = breach_penalty_multiplier
        self.negotiation_rounds = negotiation_rounds
        
        # Negotiation state
        self._current_agreement: Optional[NDArray] = None
        self._proposal_history: List[NDArray] = []
        self._agreements_reached = 0
        self._agreements_breached = 0
        
        # Extend observation space for agreement state
        base_dim = self.n_agents + 3 * (self.n_agents ** 2) + 1
        extended_dim = base_dim + self.n_agents + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(extended_dim,), dtype=np.float32
        )
    
    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        result = super().reset(**kwargs)
        self._current_agreement = None
        self._proposal_history = []
        self._agreements_reached = 0
        self._agreements_breached = 0
        return result
    
    def step(self, action: NDArray) -> Tuple[NDArray, NDArray, bool, bool, Dict]:
        actions = np.asarray(action, dtype=np.float32)
        actions = np.clip(actions, 0.0, self.endowments)
        
        # Track proposals
        self._proposal_history.append(actions.copy())
        
        # Check for agreement
        proposal_diff = abs(actions[0] - actions[1])
        agreement_reached = proposal_diff <= self.agreement_threshold
        
        # Check for breach of existing agreement
        breach_occurred = False
        if self._current_agreement is not None:
            deviation = np.abs(actions - self._current_agreement)
            if np.any(deviation > self.agreement_threshold):
                breach_occurred = True
                self._agreements_breached += 1
        
        # Execute with potential breach penalty
        if breach_occurred:
            # Amplify negative trust signal for breach
            modified_baselines = self.baselines + self._current_agreement * 0.5
        else:
            modified_baselines = self.baselines
        
        # Call parent step with potentially modified dynamics
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # Apply breach penalty to rewards
        if breach_occurred:
            rewards = rewards - self.breach_penalty_multiplier * np.abs(
                actions - self._current_agreement
            )
        
        # Update agreement state
        if agreement_reached:
            self._current_agreement = actions.copy()
            self._agreements_reached += 1
        
        # Add negotiation-specific info
        info["agreement_reached"] = agreement_reached
        info["current_agreement"] = self._current_agreement.tolist() if self._current_agreement is not None else None
        info["breach_occurred"] = breach_occurred
        info["total_agreements"] = self._agreements_reached
        info["total_breaches"] = self._agreements_breached
        info["proposal_convergence"] = 1.0 - proposal_diff / 100.0
        
        return obs, rewards, terminated, truncated, info
    
    def _get_legacy_observation(self) -> NDArray[np.float32]:
        base_obs = super()._get_legacy_observation()
        
        # Add agreement state to observation
        if self._current_agreement is not None:
            agreement_obs = self._current_agreement
        else:
            agreement_obs = np.zeros(self.n_agents, dtype=np.float32)
        
        has_agreement = np.array([1.0 if self._current_agreement is not None else 0.0])
        
        obs = np.concatenate([base_obs, agreement_obs, has_agreement])
        return obs.astype(np.float32)
    
class ReputationMarketEnv(CoopetitionEnv):
    """
    Reputation Market Environment (ReputationMarket-v0)
    
    An N-agent market where reputation scores are publicly observable and
    affect transaction outcomes. Agents must balance short-term gains from
    exploitation against long-term reputation that enables better partnerships.
    
    Challenge:
    ----------
    This environment tests whether agents can learn:
    1. The value of reputation as a strategic asset
    2. When to invest in reputation vs. exploit it
    3. How to select partners based on reputation signals
    4. The dynamics of reputation equilibria in markets
    
    Mechanics:
    ----------
    - All agents have public reputation scores visible to everyone
    - Higher reputation unlocks better partnership opportunities
    - Cooperation builds reputation, exploitation damages it
    - Market has "tiers" based on reputation thresholds
    
    The environment captures market dynamics where:
    - New entrants must build reputation to access premium tiers
    - Established players risk reputation to maximize short-term gains
    - Reputation serves as a coordination device
    
    Example:
    --------
    >>> env = ReputationMarketEnv(n_agents=5)
    >>> obs, info = env.reset(seed=42)
    >>> print(f"Initial reputations: {info['public_reputations']}")
    >>> 
    >>> # All agents act
    >>> actions = np.array([50.0, 60.0, 40.0, 55.0, 45.0])
    >>> obs, rewards, done, truncated, info = env.step(actions)
    >>> print(f"Updated reputations: {info['public_reputations']}")
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "ReputationMarket-v0"
    }
    
    # Reputation tier thresholds and bonuses
    REPUTATION_TIERS = {
        "premium": {"threshold": 0.80, "bonus": 1.30},
        "standard": {"threshold": 0.50, "bonus": 1.00},
        "probation": {"threshold": 0.25, "bonus": 0.70},
        "excluded": {"threshold": 0.00, "bonus": 0.40}
    }
    
    def __init__(
        self,
        n_agents: int = 5,
        reputation_visibility: float = 1.0,
        tier_enabled: bool = True,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Reputation Market environment.
        
        Args:
            n_agents: Number of agents in the market
            reputation_visibility: How accurately reputation is observed (1.0 = perfect)
            tier_enabled: Whether reputation tiers affect rewards
            max_steps: Maximum timesteps per episode
            render_mode: Rendering mode
        """
        # All agents connected in market
        D = np.full((n_agents, n_agents), 0.35, dtype=np.float32)
        np.fill_diagonal(D, 0.0)
        
        trust_params = TrustParameters(
            lambda_plus=0.10,
            lambda_minus=0.30,
            mu_R=0.55,
            delta_R=0.015,
            xi=0.45,
            kappa=1.0,
            initial_trust=0.50
        )
        
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=18.0,
            gamma=0.50
        )
        
        config = EnvironmentConfig(
            n_agents=n_agents,
            max_steps=max_steps,
            endowments=np.full(n_agents, 100.0),
            alpha=np.full(n_agents, 1.0 / n_agents),
            interdependence_matrix=D,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=np.full(n_agents, 35.0),
            reward_type="integrated",
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        
        self.reputation_visibility = reputation_visibility
        self.tier_enabled = tier_enabled
        
        # Public reputation scores (initialized in reset)
        self._public_reputations = np.full(n_agents, 0.50, dtype=np.float32)
        self._reputation_history = []
        
        # Extend observation space for reputation scores
        base_dim = n_agents + 3 * (n_agents ** 2) + 1
        extended_dim = base_dim + n_agents
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(extended_dim,), dtype=np.float32
        )
    
    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        result = super().reset(**kwargs)
        
        # Initialize public reputations
        self._public_reputations = np.full(self.n_agents, 0.50, dtype=np.float32)
        self._reputation_history = [self._public_reputations.copy()]
        
        return result
    
    def step(self, action: NDArray) -> Tuple[NDArray, NDArray, bool, bool, Dict]:
        result = super().step(action)
        obs, rewards, terminated, truncated, info = result
        
        actions = np.asarray(action, dtype=np.float32)
        
        # Update public reputations based on cooperation levels
        cooperation_scores = actions / self.endowments
        reputation_update = 0.1 * (cooperation_scores - 0.5)  # Centered at 50%
        self._public_reputations = np.clip(
            self._public_reputations + reputation_update,
            0.0, 1.0
        )
        
        self._reputation_history.append(self._public_reputations.copy())
        
        # Apply tier bonuses if enabled
        if self.tier_enabled:
            tier_bonuses = np.ones(self.n_agents)
            for i, rep in enumerate(self._public_reputations):
                for tier_name, tier_info in self.REPUTATION_TIERS.items():
                    if rep >= tier_info["threshold"]:
                        tier_bonuses[i] = tier_info["bonus"]
                        break
            rewards = rewards * tier_bonuses
        
        # Add market-specific info
        info["public_reputations"] = self._public_reputations.copy()
        info["reputation_ranking"] = np.argsort(-self._public_reputations).tolist()
        
        # Determine current tiers
        tiers = []
        for rep in self._public_reputations:
            for tier_name, tier_info in self.REPUTATION_TIERS.items():
                if rep >= tier_info["threshold"]:
                    tiers.append(tier_name)
                    break
        info["agent_tiers"] = tiers
        
        # Market statistics
        info["mean_reputation"] = float(np.mean(self._public_reputations))
        info["reputation_inequality"] = float(np.std(self._public_reputations))
        
        return obs, rewards, terminated, truncated, info
    
    def _get_legacy_observation(self) -> NDArray[np.float32]:
        base_obs = super()._get_legacy_observation()
        
        # Add noisy public reputations to observation
        noise = self._np_random.normal(
            0, 1 - self.reputation_visibility, 
            size=self.n_agents
        ) if self._np_random is not None else np.zeros(self.n_agents)
        
        observed_reputations = np.clip(
            self._public_reputations + noise.astype(np.float32),
            0.0, 1.0
        )
        
        obs = np.concatenate([base_obs, observed_reputations])
        return obs.astype(np.float32)
    
# =============================================================================
# Environment Registration Helpers
# =============================================================================

def make_cooperative_negotiation(**kwargs) -> CooperativeNegotiationEnv:
    """Factory function for CooperativeNegotiation-v0."""
    return CooperativeNegotiationEnv(**kwargs)


def make_reputation_market(**kwargs) -> ReputationMarketEnv:
    """Factory function for ReputationMarket-v0."""
    return ReputationMarketEnv(**kwargs)
