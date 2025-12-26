"""
================================================================================
COOPETITION-GYM: Equilibrium and Payoff Module
================================================================================

This module implements the payoff calculations and equilibrium solving from TR-1
(arXiv:2510.18802). It brings together value functions, interdependence, and
trust to compute agent utilities and solve for equilibrium action profiles.

Mathematical Foundation:
------------------------
From Pant & Yu (2025), the payoff structure consists of:

1. Private Payoff (Equation 11):
   π_i(a) = e_i - a_i + f_i(a_i) + α_i·S(a)
   
   Where:
   - e_i = endowment (initial resources)
   - a_i = cooperation level (investment/action)
   - f_i(a_i) = individual value created
   - α_i = share of synergistic surplus
   - S(a) = synergistic surplus available for distribution

2. Integrated Utility (Equation 13 - Coopetitive Equilibrium Objective):
   U_i(a) = π_i(a) + Σ_{j≠i} D_ij · π_j(a)
   
   The integrated utility adds other agents' payoffs weighted by
   interdependence coefficients. This captures the coopetitive nature
   where agents care about partners' outcomes.

3. Trust-Augmented Utility (TR-2 Extension):
   U_i^T(a) = π_i(a) + Σ_{j≠i} T_ij · D_ij · π_j(a)
   
   Trust modulates how much weight is given to partners' welfare.

Coopetitive Equilibrium:
------------------------
A Coopetitive Equilibrium extends Nash Equilibrium by having agents maximize
integrated utility rather than private payoff:

   a*_i = argmax_{a_i} U_i(a_i, a*_{-i})

This leads to higher cooperation levels than Nash (which maximizes π_i alone).

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging

from .value_functions import (
    ValueFunctionParameters, 
    individual_value, 
    synergistic_surplus,
    total_value,
    marginal_value,
    marginal_synergy
)
from .interdependence import InterdependenceMatrix
from .trust_dynamics import TrustState

logger = logging.getLogger(__name__)


@dataclass
class PayoffParameters:
    """
    Complete payoff configuration for coopetition environment.
    
    This dataclass bundles all parameters needed for payoff calculation,
    including value function parameters, agent-specific parameters, and
    interdependence structure.
    
    Attributes:
        value_params: Configuration for value function calculations
        endowments: Initial resource endowment for each agent
        alpha: Bargaining shares determining synergy distribution
        interdependence: Matrix capturing structural dependencies
    
    Constraints:
        - Σ α_i = 1 (shares sum to 1 for valid distribution)
        - e_i > 0 (positive endowments)
        - len(endowments) = len(alpha) = n_agents
    """
    value_params: ValueFunctionParameters
    endowments: NDArray[np.floating]
    alpha: NDArray[np.floating]
    interdependence: InterdependenceMatrix
    
    def __post_init__(self):
        """Validate parameter consistency."""
        n_agents = len(self.endowments)
        
        if len(self.alpha) != n_agents:
            raise ValueError(
                f"Alpha length ({len(self.alpha)}) must match "
                f"endowments length ({n_agents})"
            )
        
        if self.interdependence.n_agents != n_agents:
            raise ValueError(
                f"Interdependence matrix dimension ({self.interdependence.n_agents}) "
                f"must match number of agents ({n_agents})"
            )
        
        # Normalize alpha if needed (should sum to 1)
        alpha_sum = np.sum(self.alpha)
        if not np.isclose(alpha_sum, 1.0, rtol=1e-3):
            logger.warning(
                f"Alpha sums to {alpha_sum}, normalizing to 1.0"
            )
            self.alpha = self.alpha / alpha_sum
    
    @property
    def n_agents(self) -> int:
        """Number of agents in the system."""
        return len(self.endowments)


@dataclass
class PayoffResult:
    """
    Result of payoff computation for all agents.
    
    Contains both private payoffs and integrated utilities, along with
    component breakdowns for analysis and debugging.
    
    Attributes:
        private_payoffs: Array of π_i for each agent
        integrated_utilities: Array of U_i for each agent
        total_value: Total value created V(a|γ)
        synergistic_surplus: Surplus available for distribution S(a)
        individual_values: Array of f_i(a_i) for each agent
    """
    private_payoffs: NDArray[np.floating]
    integrated_utilities: NDArray[np.floating]
    total_value: float
    synergistic_surplus: float
    individual_values: NDArray[np.floating]
    
    @property
    def n_agents(self) -> int:
        return len(self.private_payoffs)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "private_payoffs": self.private_payoffs.tolist(),
            "integrated_utilities": self.integrated_utilities.tolist(),
            "total_value": self.total_value,
            "synergistic_surplus": self.synergistic_surplus,
            "individual_values": self.individual_values.tolist()
        }


def compute_private_payoff(
    agent_idx: int,
    actions: NDArray[np.floating],
    params: PayoffParameters
) -> float:
    """
    Compute the private payoff for a single agent.
    
    This implements TR-1 Equation 11:
    π_i(a) = e_i - a_i + f_i(a_i) + α_i·S(a)
    
    The payoff consists of:
    1. Kept endowment: (e_i - a_i) - what wasn't invested
    2. Individual value: f_i(a_i) - returns from own investment
    3. Synergy share: α_i·S(a) - share of collective surplus
    
    Args:
        agent_idx: Index of the agent
        actions: Array of cooperation levels from all agents
        params: Complete payoff parameters
    
    Returns:
        Private payoff π_i for the specified agent
    """
    # Extract agent-specific values
    e_i = params.endowments[agent_idx]
    a_i = actions[agent_idx]
    alpha_i = params.alpha[agent_idx]
    
    # Compute individual value from own contribution
    f_i = individual_value(a_i, params.value_params)
    
    # Compute synergistic surplus for distribution
    S = synergistic_surplus(actions, params.value_params)
    
    # Combine components
    # Note: e_i - a_i can be negative if agent invests more than endowment
    # This should be prevented by action space constraints
    payoff = (e_i - a_i) + f_i + alpha_i * S
    
    return float(payoff)


def compute_all_private_payoffs(
    actions: NDArray[np.floating],
    params: PayoffParameters
) -> NDArray[np.floating]:
    """
    Compute private payoffs for all agents.
    
    Args:
        actions: Array of cooperation levels
        params: Payoff parameters
    
    Returns:
        Array of private payoffs, one per agent
    """
    n_agents = params.n_agents
    payoffs = np.zeros(n_agents)
    
    for i in range(n_agents):
        payoffs[i] = compute_private_payoff(i, actions, params)
    
    return payoffs


def compute_integrated_utility(
    agent_idx: int,
    actions: NDArray[np.floating],
    params: PayoffParameters,
    trust_state: Optional[TrustState] = None
) -> float:
    """
    Compute the integrated utility for a single agent.
    
    This implements TR-1 Equation 13:
    U_i(a) = π_i(a) + Σ_{j≠i} D_ij · π_j(a)
    
    With optional trust augmentation (TR-2):
    U_i^T(a) = π_i(a) + Σ_{j≠i} T_ij · D_ij · π_j(a)
    
    The integrated utility captures the coopetitive nature of relationships:
    - Agents care about their own payoff (π_i)
    - They also care about partners' payoffs, weighted by interdependence
    - Trust modulates how much weight is given to partners' welfare
    
    Args:
        agent_idx: Index of the agent
        actions: Array of cooperation levels
        params: Payoff parameters
        trust_state: Optional trust state for trust-augmented utility
    
    Returns:
        Integrated utility U_i for the specified agent
    """
    n_agents = params.n_agents
    D = params.interdependence.matrix
    
    # Compute all private payoffs
    payoffs = compute_all_private_payoffs(actions, params)
    
    # Start with own payoff
    utility = payoffs[agent_idx]
    
    # Add weighted payoffs of other agents
    for j in range(n_agents):
        if j != agent_idx:
            D_ij = D[agent_idx, j]
            
            # Apply trust modulation if available
            if trust_state is not None:
                T_ij = trust_state.get_effective_trust(agent_idx, j)
                weight = T_ij * D_ij
            else:
                weight = D_ij
            
            utility += weight * payoffs[j]
    
    return float(utility)


def compute_all_integrated_utilities(
    actions: NDArray[np.floating],
    params: PayoffParameters,
    trust_state: Optional[TrustState] = None
) -> NDArray[np.floating]:
    """
    Compute integrated utilities for all agents.
    
    Args:
        actions: Array of cooperation levels
        params: Payoff parameters
        trust_state: Optional trust state for trust-augmented utility
    
    Returns:
        Array of integrated utilities, one per agent
    """
    n_agents = params.n_agents
    utilities = np.zeros(n_agents)
    
    for i in range(n_agents):
        utilities[i] = compute_integrated_utility(i, actions, params, trust_state)
    
    return utilities


def compute_complete_payoffs(
    actions: NDArray[np.floating],
    params: PayoffParameters,
    trust_state: Optional[TrustState] = None
) -> PayoffResult:
    """
    Compute comprehensive payoff results for analysis.
    
    Returns a PayoffResult with all components for debugging,
    visualization, and analysis.
    
    Args:
        actions: Array of cooperation levels
        params: Payoff parameters
        trust_state: Optional trust state
    
    Returns:
        PayoffResult with detailed breakdown
    """
    # Compute individual values
    individual_vals = np.array([
        individual_value(a, params.value_params) 
        for a in actions
    ])
    
    # Compute synergistic surplus
    S = synergistic_surplus(actions, params.value_params)
    
    # Compute total value
    V = total_value(actions, params.value_params)
    
    # Compute all payoffs
    private_payoffs = compute_all_private_payoffs(actions, params)
    integrated_utilities = compute_all_integrated_utilities(
        actions, params, trust_state
    )
    
    return PayoffResult(
        private_payoffs=private_payoffs,
        integrated_utilities=integrated_utilities,
        total_value=V,
        synergistic_surplus=S,
        individual_values=individual_vals
    )


# =============================================================================
# Equilibrium Solving
# =============================================================================

@dataclass
class EquilibriumResult:
    """
    Result of equilibrium computation.
    
    Attributes:
        actions: Equilibrium action profile a*
        utilities: Equilibrium utilities U_i(a*)
        payoffs: Equilibrium private payoffs π_i(a*)
        converged: Whether the solver converged
        iterations: Number of iterations to convergence
        equilibrium_type: "nash" or "coopetitive"
    """
    actions: NDArray[np.floating]
    utilities: NDArray[np.floating]
    payoffs: NDArray[np.floating]
    converged: bool
    iterations: int
    equilibrium_type: str
    
    @property
    def cooperation_rate(self) -> float:
        """Average cooperation as fraction of maximum."""
        return float(np.mean(self.actions))
    
    @property
    def total_welfare(self) -> float:
        """Sum of all utilities."""
        return float(np.sum(self.utilities))


def solve_best_response(
    agent_idx: int,
    other_actions: NDArray[np.floating],
    params: PayoffParameters,
    equilibrium_type: str = "coopetitive",
    trust_state: Optional[TrustState] = None
) -> float:
    """
    Solve for an agent's best response given others' actions.
    
    The best response is the action that maximizes the agent's objective
    (either private payoff or integrated utility) holding others' actions fixed.
    
    Args:
        agent_idx: Index of the agent finding best response
        other_actions: Actions of all other agents (agent_idx position ignored)
        params: Payoff parameters
        equilibrium_type: "nash" (maximize π_i) or "coopetitive" (maximize U_i)
        trust_state: Optional trust state for coopetitive equilibrium
    
    Returns:
        Optimal action a*_i
    """
    endowment = params.endowments[agent_idx]
    
    # Define objective function (negative for minimization)
    def objective(a_i: float) -> float:
        # Construct full action profile
        actions = other_actions.copy()
        actions[agent_idx] = a_i
        
        if equilibrium_type == "nash":
            return -compute_private_payoff(agent_idx, actions, params)
        else:  # coopetitive
            return -compute_integrated_utility(
                agent_idx, actions, params, trust_state
            )
    
    # Optimize over valid action range
    result = minimize_scalar(
        objective,
        bounds=(0.0, endowment),
        method='bounded'
    )
    
    return float(result.x)


def solve_equilibrium(
    params: PayoffParameters,
    equilibrium_type: str = "coopetitive",
    trust_state: Optional[TrustState] = None,
    initial_actions: Optional[NDArray[np.floating]] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> EquilibriumResult:
    """
    Solve for equilibrium using iterated best response.
    
    Starting from initial actions, iteratively compute each agent's best
    response until convergence (actions don't change) or max iterations.
    
    Args:
        params: Payoff parameters
        equilibrium_type: "nash" or "coopetitive"
        trust_state: Optional trust state for coopetitive equilibrium
        initial_actions: Starting action profile (defaults to 30% of endowments)
        max_iterations: Maximum iterations before declaring non-convergence
        tolerance: Convergence threshold for action changes
    
    Returns:
        EquilibriumResult with equilibrium profile and metadata
    """
    n_agents = params.n_agents
    
    # Initialize actions
    if initial_actions is None:
        actions = params.endowments * 0.3
    else:
        actions = initial_actions.copy()
    
    converged = False
    
    for iteration in range(max_iterations):
        old_actions = actions.copy()
        
        # Update each agent's action to best response
        for i in range(n_agents):
            actions[i] = solve_best_response(
                i, actions, params, equilibrium_type, trust_state
            )
        
        # Check convergence
        max_change = np.max(np.abs(actions - old_actions))
        if max_change < tolerance:
            converged = True
            break
    
    # Compute equilibrium payoffs and utilities
    payoffs = compute_all_private_payoffs(actions, params)
    utilities = compute_all_integrated_utilities(actions, params, trust_state)
    
    return EquilibriumResult(
        actions=actions,
        utilities=utilities,
        payoffs=payoffs,
        converged=converged,
        iterations=iteration + 1,
        equilibrium_type=equilibrium_type
    )


def compare_equilibria(
    params: PayoffParameters,
    trust_state: Optional[TrustState] = None
) -> Dict[str, EquilibriumResult]:
    """
    Compare Nash and Coopetitive equilibria.
    
    Computes both equilibrium types and returns them for comparison.
    The coopetitive equilibrium typically features higher cooperation
    and higher total welfare than the Nash equilibrium.
    
    Args:
        params: Payoff parameters
        trust_state: Optional trust state
    
    Returns:
        Dictionary with "nash" and "coopetitive" equilibrium results
    """
    nash = solve_equilibrium(
        params, equilibrium_type="nash", trust_state=trust_state
    )
    coopetitive = solve_equilibrium(
        params, equilibrium_type="coopetitive", trust_state=trust_state
    )
    
    logger.info(
        f"Nash cooperation: {nash.cooperation_rate:.2f}, "
        f"Coopetitive cooperation: {coopetitive.cooperation_rate:.2f}, "
        f"Increase: {(coopetitive.cooperation_rate - nash.cooperation_rate)*100:.1f}%"
    )
    
    return {"nash": nash, "coopetitive": coopetitive}


# =============================================================================
# Reward Computation for RL Environments
# =============================================================================

def compute_rewards(
    actions: NDArray[np.floating],
    params: PayoffParameters,
    trust_state: Optional[TrustState] = None,
    reward_type: str = "integrated",
    normalize: bool = False,
    reward_scale: float = 1.0
) -> NDArray[np.floating]:
    """
    Compute rewards for RL environment step.
    
    This is the main interface used by Gymnasium environments to compute
    agent rewards from action profiles.
    
    Args:
        actions: Array of agent actions
        params: Payoff parameters
        trust_state: Optional trust state for trust-augmented rewards
        reward_type: "integrated" (U_i), "private" (π_i), or "cooperative" (sum)
        normalize: Whether to normalize rewards to approximately [-1, 1]
        reward_scale: Scaling factor for rewards
    
    Returns:
        Array of rewards, one per agent
    """
    if reward_type == "private":
        rewards = compute_all_private_payoffs(actions, params)
    elif reward_type == "integrated":
        rewards = compute_all_integrated_utilities(actions, params, trust_state)
    elif reward_type == "cooperative":
        # Everyone gets total welfare (fully cooperative objective)
        utilities = compute_all_integrated_utilities(actions, params, trust_state)
        total = np.sum(utilities)
        rewards = np.full_like(utilities, total / params.n_agents)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")
    
    # Optional normalization
    if normalize:
        # Normalize by maximum possible reward (crude estimate)
        max_endowment = np.max(params.endowments)
        rewards = rewards / max_endowment
    
    rewards = rewards * reward_scale
    
    return rewards


def compute_social_welfare(
    actions: NDArray[np.floating],
    params: PayoffParameters
) -> float:
    """
    Compute social welfare (sum of private payoffs).
    
    Social welfare measures the total value captured by all agents.
    This is distinct from total value created, as some value may be
    "lost" to coordination failures.
    """
    payoffs = compute_all_private_payoffs(actions, params)
    return float(np.sum(payoffs))


def compute_cooperation_surplus(
    actions: NDArray[np.floating],
    params: PayoffParameters
) -> float:
    """
    Compute surplus from cooperation vs. autarky.
    
    Compares current welfare to what agents would get if they
    didn't cooperate at all (zero actions).
    """
    current_welfare = compute_social_welfare(actions, params)
    autarky_welfare = compute_social_welfare(
        np.zeros_like(actions), params
    )
    
    return current_welfare - autarky_welfare


# =============================================================================
# Factory Functions for Common Configurations
# =============================================================================

def create_slcd_payoff_params() -> PayoffParameters:
    """
    Create payoff parameters for Samsung-Sony S-LCD case study.
    
    Based on TR-1 Section 8 validation with 58/60 score.
    """
    from .interdependence import create_slcd_interdependence
    from .value_functions import create_slcd_parameters
    
    return PayoffParameters(
        value_params=create_slcd_parameters(),
        endowments=np.array([100.0, 100.0]),
        alpha=np.array([0.55, 0.45]),  # Samsung majority share
        interdependence=create_slcd_interdependence()
    )


def create_symmetric_payoff_params(
    n_agents: int = 2,
    endowment: float = 100.0,
    gamma: float = 0.5,
    dependency: float = 0.5
) -> PayoffParameters:
    """
    Create symmetric payoff parameters for baseline experiments.
    
    All agents have equal endowments, equal shares, and equal dependencies.
    """
    from .interdependence import create_symmetric_interdependence
    
    return PayoffParameters(
        value_params=ValueFunctionParameters(gamma=gamma),
        endowments=np.full(n_agents, endowment),
        alpha=np.full(n_agents, 1.0 / n_agents),
        interdependence=create_symmetric_interdependence(n_agents, dependency)
    )
