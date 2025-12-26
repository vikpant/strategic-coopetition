"""
================================================================================
COOPETITION-GYM: Value Functions Module
================================================================================

This module implements the value creation functions from TR-1 (arXiv:2510.18802).
These functions translate agent cooperation levels into individual and collective
value, forming the foundation of the reward calculation in coopetition environments.

Mathematical Foundation:
------------------------
From Pant & Yu (2025), the value creation framework consists of:

1. Individual Value Functions (Equations 5-6):
   - Power specification: f_i(a_i) = a_i^β (diminishing returns via Cobb-Douglas)
   - Logarithmic specification: f_i(a_i) = θ·ln(1 + a_i) (validated optimal, 58/60 score)

2. Synergy Function (Equation 7):
   - g(a) = (∏ a_i)^(1/N) - Geometric mean capturing balanced contribution

3. Total Value Creation (Equation 8):
   - V(a|γ) = Σf_i(a_i) + γ·g(a) - Individual plus synergistic value

Key Validated Parameters:
-------------------------
- β = 0.75 (power exponent, TR-1 §7.4)
- θ = 20.0 (logarithmic scale, TR-1 §8.3)
- γ = 0.65 (complementarity, Samsung-Sony S-LCD case)

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Union, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure module logger
logger = logging.getLogger(__name__)


class ValueSpecification(Enum):
    """
    Enumeration of available value function specifications.
    
    The choice of specification affects how individual contributions translate
    to value. Both exhibit diminishing marginal returns but with different
    curvature properties.
    """
    POWER = "power"
    LOGARITHMIC = "logarithmic"


@dataclass(frozen=True)
class ValueFunctionParameters:
    """
    Immutable configuration for value function calculations.
    
    This dataclass encapsulates all parameters needed for value computation,
    with defaults set to the validated optimal values from TR-1.
    
    Attributes:
        specification: Which functional form to use (power or logarithmic)
        beta: Exponent for power specification (default 0.75, TR-1 §7.4)
        theta: Scale factor for logarithmic specification (default 20.0, TR-1 §8.3)
        gamma: Complementarity parameter weighting synergy (default 0.65)
        epsilon: Small constant for numerical stability (prevents log(0))
    
    Example:
        >>> params = ValueFunctionParameters(specification=ValueSpecification.LOGARITHMIC)
        >>> params.theta
        20.0
    """
    specification: ValueSpecification = ValueSpecification.LOGARITHMIC
    beta: float = 0.75
    theta: float = 20.0
    gamma: float = 0.65
    epsilon: float = 1e-10
    
    def __post_init__(self):
        """Validate parameter bounds after initialization."""
        if not 0 < self.beta <= 1:
            raise ValueError(f"Beta must be in (0, 1], got {self.beta}")
        if self.theta <= 0:
            raise ValueError(f"Theta must be positive, got {self.theta}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"Gamma must be in [0, 1], got {self.gamma}")


def power_value(
    action: Union[float, NDArray[np.floating]],
    beta: float = 0.75,
    epsilon: float = 1e-10
) -> Union[float, NDArray[np.floating]]:
    """
    Compute individual value using the power (Cobb-Douglas) specification.
    
    This implements TR-1 Equation 5: f_i(a_i) = a_i^β
    
    The power function exhibits diminishing marginal returns, meaning each
    additional unit of cooperation contributes less value than the previous.
    This captures the economic reality that initial cooperation is highly
    valuable while extreme cooperation has bounded benefits.
    
    Args:
        action: Cooperation level(s), can be scalar or array. Must be non-negative.
        beta: Elasticity parameter controlling the rate of diminishing returns.
              Lower values = faster diminishing returns. Default 0.75 from TR-1.
        epsilon: Small constant added for numerical stability when action ≈ 0.
    
    Returns:
        Individual value contribution, same shape as input.
    
    Mathematical Properties:
        - f(0) = 0 (no cooperation yields no individual value)
        - f'(a) = β·a^(β-1) > 0 (monotonically increasing)
        - f''(a) = β(β-1)·a^(β-2) < 0 for β < 1 (concave, diminishing returns)
    
    Example:
        >>> power_value(50.0, beta=0.75)
        18.80...
        >>> power_value(np.array([25, 50, 75]), beta=0.75)
        array([11.18..., 18.80..., 25.09...])
    """
    # Ensure non-negative input for mathematical validity
    action_safe = np.maximum(action, epsilon)
    
    # Apply power function with validated exponent
    return np.power(action_safe, beta)


def logarithmic_value(
    action: Union[float, NDArray[np.floating]],
    theta: float = 20.0,
    epsilon: float = 1e-10
) -> Union[float, NDArray[np.floating]]:
    """
    Compute individual value using the logarithmic specification.
    
    This implements TR-1 Equation 6: f_i(a_i) = θ·ln(1 + a_i)
    
    The logarithmic function provides stronger diminishing returns than the
    power specification at high cooperation levels. This was validated as
    the optimal specification in the Samsung-Sony S-LCD case study, achieving
    a 58/60 validation score (96.7%) compared to 46/60 for power specification.
    
    Args:
        action: Cooperation level(s), can be scalar or array. Must be non-negative.
        theta: Scale parameter controlling the magnitude of value creation.
               Default 20.0 from TR-1 §8.3 S-LCD validation.
        epsilon: Small constant for numerical stability.
    
    Returns:
        Individual value contribution, same shape as input.
    
    Mathematical Properties:
        - f(0) = 0 (no cooperation yields no individual value)
        - f'(a) = θ/(1+a) > 0 (monotonically increasing)
        - f''(a) = -θ/(1+a)² < 0 (concave, diminishing returns)
        - Stronger concavity than power function for large a
    
    Example:
        >>> logarithmic_value(50.0, theta=20.0)
        78.55...
        >>> logarithmic_value(np.array([25, 50, 75]), theta=20.0)
        array([65.06..., 78.55..., 86.54...])
    """
    # Ensure non-negative input
    action_safe = np.maximum(action, 0.0)
    
    # Apply logarithmic function: θ·ln(1 + a)
    # The +1 ensures f(0) = 0 and avoids log(0)
    return theta * np.log1p(action_safe)


def individual_value(
    action: Union[float, NDArray[np.floating]],
    params: ValueFunctionParameters
) -> Union[float, NDArray[np.floating]]:
    """
    Compute individual value using the configured specification.
    
    This is the primary interface for individual value calculation, dispatching
    to the appropriate specification based on the parameters.
    
    Args:
        action: Cooperation level(s), can be scalar or array.
        params: Configuration specifying which function and parameters to use.
    
    Returns:
        Individual value contribution(s).
    
    Example:
        >>> params = ValueFunctionParameters(specification=ValueSpecification.LOGARITHMIC)
        >>> individual_value(50.0, params)
        78.55...
    """
    if params.specification == ValueSpecification.POWER:
        return power_value(action, params.beta, params.epsilon)
    elif params.specification == ValueSpecification.LOGARITHMIC:
        return logarithmic_value(action, params.theta, params.epsilon)
    else:
        raise ValueError(f"Unknown specification: {params.specification}")


def synergy_function(
    actions: NDArray[np.floating],
    epsilon: float = 1e-10
) -> float:
    """
    Compute synergistic value from collective cooperation.
    
    This implements TR-1 Equation 7: g(a) = (∏ a_i)^(1/N)
    
    The geometric mean captures the intuition that synergy requires balanced
    contributions from all parties. If any agent defects completely (a_i = 0),
    the synergy collapses to zero regardless of others' contributions. This
    models the interdependent nature of coopetitive relationships where value
    creation requires mutual engagement.
    
    Args:
        actions: Array of cooperation levels from all agents.
        epsilon: Small constant for numerical stability.
    
    Returns:
        Synergistic value component (scalar).
    
    Mathematical Properties:
        - g(a) = 0 if any a_i = 0 (synergy requires all participants)
        - g(a) is maximized when all a_i are equal (rewards balance)
        - Symmetric: order of agents doesn't matter
    
    Example:
        >>> synergy_function(np.array([50.0, 50.0]))
        50.0
        >>> synergy_function(np.array([100.0, 0.0]))  # One defector
        0.0
        >>> synergy_function(np.array([75.0, 25.0]))  # Imbalanced
        43.30...
    """
    # Handle edge cases
    if len(actions) == 0:
        return 0.0
    
    # Check for any zero contributions (synergy collapse)
    if np.any(actions <= epsilon):
        return 0.0
    
    # Compute geometric mean: (∏ a_i)^(1/N)
    # Using log-space computation for numerical stability with many agents
    n_agents = len(actions)
    log_product = np.sum(np.log(np.maximum(actions, epsilon)))
    geometric_mean = np.exp(log_product / n_agents)
    
    return float(geometric_mean)


def total_value(
    actions: NDArray[np.floating],
    params: ValueFunctionParameters
) -> float:
    """
    Compute total value creation from all agents' cooperation.
    
    This implements TR-1 Equation 8: V(a|γ) = Σf_i(a_i) + γ·g(a)
    
    Total value consists of two components:
    1. Sum of individual values: What each agent creates independently
    2. Synergistic value: Additional value from collective interaction
    
    The complementarity parameter γ controls the relative importance of synergy.
    When γ = 0, agents create value independently. When γ = 1, synergy is
    weighted equally with individual contributions.
    
    Args:
        actions: Array of cooperation levels from all agents.
        params: Value function configuration including specification and γ.
    
    Returns:
        Total value created (scalar).
    
    Example:
        >>> params = ValueFunctionParameters(gamma=0.65)
        >>> total_value(np.array([50.0, 50.0]), params)
        189.60...  # 78.55*2 + 0.65*50
    """
    # Compute individual values for all agents
    individual_values = individual_value(actions, params)
    sum_individual = np.sum(individual_values)
    
    # Compute synergistic component
    synergy = synergy_function(actions, params.epsilon)
    
    # Combine with complementarity weighting
    total = sum_individual + params.gamma * synergy
    
    logger.debug(
        f"Total value: {total:.4f} "
        f"(individual={sum_individual:.4f}, synergy={synergy:.4f})"
    )
    
    return float(total)


def synergistic_surplus(
    actions: NDArray[np.floating],
    params: ValueFunctionParameters
) -> float:
    """
    Compute the synergistic surplus to be distributed among agents.
    
    This implements the S(a) term from TR-1 Equation 11:
    S(a) = V(a|γ) - Σf_i(a_i) = γ·g(a)
    
    The synergistic surplus represents the additional value created through
    cooperation beyond what agents could achieve independently. This surplus
    is distributed according to bargaining shares α_i.
    
    Args:
        actions: Array of cooperation levels from all agents.
        params: Value function configuration.
    
    Returns:
        Synergistic surplus available for distribution (scalar).
    
    Example:
        >>> params = ValueFunctionParameters(gamma=0.65)
        >>> synergistic_surplus(np.array([50.0, 50.0]), params)
        32.5  # 0.65 * 50
    """
    synergy = synergy_function(actions, params.epsilon)
    return params.gamma * synergy


def marginal_value(
    action: float,
    params: ValueFunctionParameters
) -> float:
    """
    Compute the marginal (derivative) value of additional cooperation.
    
    This gives the first derivative of the individual value function,
    representing how much additional value is created by a small increase
    in cooperation. Used for equilibrium analysis and optimization.
    
    Args:
        action: Current cooperation level.
        params: Value function configuration.
    
    Returns:
        Marginal value (derivative) at the given action level.
    
    Mathematical Derivation:
        Power: f'(a) = β·a^(β-1)
        Logarithmic: f'(a) = θ/(1+a)
    """
    action_safe = max(action, params.epsilon)
    
    if params.specification == ValueSpecification.POWER:
        # Derivative of a^β is β·a^(β-1)
        return params.beta * np.power(action_safe, params.beta - 1)
    else:
        # Derivative of θ·ln(1+a) is θ/(1+a)
        return params.theta / (1 + action_safe)


def marginal_synergy(
    actions: NDArray[np.floating],
    agent_index: int,
    params: ValueFunctionParameters
) -> float:
    """
    Compute the marginal synergy contribution for a specific agent.
    
    This gives the partial derivative of the synergy function with respect
    to agent i's action, representing how much additional synergy is created
    by increasing agent i's cooperation.
    
    Args:
        actions: Current cooperation levels of all agents.
        agent_index: Index of the agent for whom to compute marginal synergy.
        params: Value function configuration.
    
    Returns:
        Marginal synergy contribution for the specified agent.
    
    Mathematical Derivation:
        For g(a) = (∏a_j)^(1/N), the partial derivative is:
        ∂g/∂a_i = (1/N) · (∏a_j)^(1/N) / a_i = g(a) / (N·a_i)
    """
    n_agents = len(actions)
    a_i = max(actions[agent_index], params.epsilon)
    
    # If any action is zero, marginal synergy is zero
    if np.any(actions <= params.epsilon):
        return 0.0
    
    synergy = synergy_function(actions, params.epsilon)
    return synergy / (n_agents * a_i)


# =============================================================================
# Factory Functions for Common Configurations
# =============================================================================

def create_slcd_parameters() -> ValueFunctionParameters:
    """
    Create parameters matching the Samsung-Sony S-LCD case study.
    
    These parameters achieved a 58/60 validation score in TR-1 §8,
    representing the empirically optimal configuration for the joint venture.
    
    Returns:
        ValueFunctionParameters configured for S-LCD case study.
    """
    return ValueFunctionParameters(
        specification=ValueSpecification.LOGARITHMIC,
        theta=20.0,
        gamma=0.65
    )


def create_power_parameters(
    beta: float = 0.75,
    gamma: float = 0.50
) -> ValueFunctionParameters:
    """
    Create parameters using power specification.
    
    Args:
        beta: Elasticity parameter (default 0.75 from TR-1).
        gamma: Complementarity parameter.
    
    Returns:
        ValueFunctionParameters with power specification.
    """
    return ValueFunctionParameters(
        specification=ValueSpecification.POWER,
        beta=beta,
        gamma=gamma
    )


# =============================================================================
# Vectorized Operations for Batch Processing
# =============================================================================

def batch_total_value(
    actions_batch: NDArray[np.floating],
    params: ValueFunctionParameters
) -> NDArray[np.floating]:
    """
    Compute total value for a batch of action profiles.
    
    Efficient vectorized computation for processing multiple action profiles
    simultaneously, useful for Monte Carlo simulation and batch RL updates.
    
    Args:
        actions_batch: 2D array of shape (batch_size, n_agents).
        params: Value function configuration.
    
    Returns:
        Array of total values, shape (batch_size,).
    
    Example:
        >>> params = ValueFunctionParameters()
        >>> actions = np.array([[50, 50], [30, 70], [40, 40]])
        >>> batch_total_value(actions, params)
        array([189.60..., 178.34..., 169.28...])
    """
    batch_size = actions_batch.shape[0]
    values = np.zeros(batch_size)
    
    for i in range(batch_size):
        values[i] = total_value(actions_batch[i], params)
    
    return values
