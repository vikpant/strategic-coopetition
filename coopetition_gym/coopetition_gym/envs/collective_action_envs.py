"""
================================================================================
COOPETITION-GYM: Collective Action Environments (TR-3 Category)
================================================================================

Technical Report: TR-3 (arXiv:2601.16237)
Title: Computational Foundations for Strategic Coopetition:
       Formalizing Collective Action and Loyalty

This module implements environments for studying collective action problems
and loyalty dynamics in team production contexts. These environments test
whether agents can overcome free-rider temptations and sustain cooperation.

Environments:
-------------
1. TeamProduction-v0: Core team production with free-rider dynamics
2. LoyaltyTeam-v0: Team production with TR-3 loyalty mechanisms
3. CoalitionFormation-v0: Dynamic coalition with entry/exit dynamics
4. ApacheProject-v0: Validated Apache HTTP Server case study (52/60)
5. PublicGoods-v0: Classic public goods with collective action modifiers

Mathematical Framework (TR-3):
------------------------------
- Team production: Q(a) = ω · (Σa_i)^β
- Base payoff: π_i^team = (1/n) · Q(a) - c · a_i
- Loyalty modifier: L_i = θ_i · [φ_B · π̄_{-i} + φ_C · c · a_i]
- Free-riding equilibrium: a* = (ωβ / nc)^(1/(1-β))
- Team cohesion: C = Σ(D_{T,i} · θ_i) / Σ(D_{T,i})

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
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.base import CoopetitionEnv, EnvironmentConfig
from core.value_functions import ValueFunctionParameters, ValueSpecification
from core.trust_dynamics import TrustParameters
from core.interdependence import create_symmetric_interdependence
from core.collective_action import (
    CollectiveActionParameters,
    CollectiveActionState,
    CollectiveActionModel
)


# =============================================================================
# TR-3 MATHEMATICAL CORE
# =============================================================================

@dataclass
class TR3Parameters:
    """
    Parameters for TR-3 team production model.

    From TR-3 Equations:
    - Q(a) = ω · (Σa_i)^β                      [Eq. 1]
    - π_i = (1/n) · Q(a) - c · a_i             [Eq. 2]
    - L_i = θ_i · [φ_B · π̄_{-i} + φ_C · c · a_i]  [Eq. 3]
    """
    omega: float = 25.0      # Productivity factor
    beta: float = 0.7        # Returns to scale (< 1 for diminishing)
    c: float = 1.0           # Effort cost coefficient
    phi_B: float = 0.8       # Loyalty benefit strength
    phi_C: float = 0.3       # Cost tolerance strength
    a_max: float = 50.0      # Maximum effort bound


def team_production(actions: np.ndarray, omega: float, beta: float) -> float:
    """
    Compute team production output.

    Q(a) = ω · (Σa_i)^β

    Args:
        actions: Array of agent efforts
        omega: Productivity factor
        beta: Returns to scale

    Returns:
        Total team output Q
    """
    total_effort = np.sum(actions)
    return omega * (total_effort ** beta)


def base_team_payoff(
    agent_idx: int,
    actions: np.ndarray,
    omega: float,
    beta: float,
    c: float
) -> float:
    """
    Compute base team payoff for agent i.

    π_i^team = (1/n) · Q(a) - c · a_i

    Args:
        agent_idx: Index of agent
        actions: Array of all agent efforts
        omega: Productivity factor
        beta: Returns to scale
        c: Effort cost coefficient

    Returns:
        Base payoff for agent i
    """
    n = len(actions)
    Q = team_production(actions, omega, beta)
    share = Q / n
    cost = c * actions[agent_idx]
    return share - cost


def teammates_payoff(
    agent_idx: int,
    actions: np.ndarray,
    omega: float,
    beta: float,
    c: float
) -> float:
    """
    Compute average payoff of teammates (excluding agent i).

    π̄_{-i} = (1/(n-1)) · Σ_{j≠i} π_j

    Args:
        agent_idx: Index of agent
        actions: Array of all agent efforts
        omega: Productivity factor
        beta: Returns to scale
        c: Effort cost coefficient

    Returns:
        Average payoff of teammates
    """
    n = len(actions)
    if n <= 1:
        return 0.0

    total_payoff = 0.0
    for j in range(n):
        if j != agent_idx:
            total_payoff += base_team_payoff(j, actions, omega, beta, c)

    return total_payoff / (n - 1)


def loyalty_modifier(
    agent_idx: int,
    actions: np.ndarray,
    loyalty: float,
    omega: float,
    beta: float,
    c: float,
    phi_B: float,
    phi_C: float
) -> float:
    """
    Compute loyalty modifier for agent i.

    L_i = θ_i · [φ_B · π̄_{-i} + φ_C · c · a_i]

    Args:
        agent_idx: Index of agent
        actions: Array of all agent efforts
        loyalty: Agent's loyalty score θ_i ∈ [0, 1]
        omega, beta, c: Team production parameters
        phi_B: Loyalty benefit strength
        phi_C: Cost tolerance strength

    Returns:
        Loyalty modifier L_i
    """
    teammates_avg = teammates_payoff(agent_idx, actions, omega, beta, c)
    cost_term = c * actions[agent_idx]

    return loyalty * (phi_B * teammates_avg + phi_C * cost_term)


def loyalty_augmented_utility(
    agent_idx: int,
    actions: np.ndarray,
    loyalty: float,
    params: TR3Parameters
) -> float:
    """
    Compute loyalty-augmented utility for agent i.

    U_i = π_i^team + L_i

    Args:
        agent_idx: Index of agent
        actions: Array of all agent efforts
        loyalty: Agent's loyalty score θ_i ∈ [0, 1]
        params: TR3 parameters

    Returns:
        Total utility for agent i
    """
    base = base_team_payoff(
        agent_idx, actions,
        params.omega, params.beta, params.c
    )
    modifier = loyalty_modifier(
        agent_idx, actions, loyalty,
        params.omega, params.beta, params.c,
        params.phi_B, params.phi_C
    )
    return base + modifier


def free_riding_equilibrium(params: TR3Parameters, n: int) -> float:
    """
    Compute the Nash equilibrium effort level (free-riding equilibrium).

    a* = (ωβ / nc)^(1/(1-β))

    This is the effort level each agent would choose if acting selfishly,
    ignoring positive externalities to teammates.

    Note: When β=1.0 (linear returns), the formula has no interior solution.
    The equilibrium is a corner solution: 0 if ω/n < c, else a_max.

    Args:
        params: TR3 parameters
        n: Number of agents

    Returns:
        Nash equilibrium effort level
    """
    # Handle linear returns (beta=1.0) - corner solution
    if abs(params.beta - 1.0) < 1e-9:
        marginal_benefit = params.omega / n
        if marginal_benefit < params.c:
            return 0.0  # Free-riding dominates
        else:
            return params.a_max  # Contributing dominates

    numerator = params.omega * params.beta
    denominator = n * params.c
    exponent = 1.0 / (1.0 - params.beta)

    return (numerator / denominator) ** exponent


def social_optimum_effort(params: TR3Parameters, n: int) -> float:
    """
    Compute the socially optimal effort level.

    a^opt = (ωβ / c)^(1/(1-β))

    This is the effort level that maximizes total team welfare.

    Note: When β=1.0 (linear returns), the formula has no interior solution.
    The social optimum is a_max if ω > c, else 0.

    Args:
        params: TR3 parameters
        n: Number of agents

    Returns:
        Socially optimal effort level
    """
    # Handle linear returns (beta=1.0) - corner solution
    if abs(params.beta - 1.0) < 1e-9:
        # With linear returns, social optimum is max effort if multiplier > cost
        if params.omega > params.c:
            return params.a_max
        else:
            return 0.0

    numerator = params.omega * params.beta
    denominator = params.c
    exponent = 1.0 / (1.0 - params.beta)

    return (numerator / denominator) ** exponent


def team_cohesion(
    loyalty_scores: np.ndarray,
    interdependence_weights: np.ndarray
) -> float:
    """
    Compute team cohesion metric.

    C = Σ(D_{T,i} · θ_i) / Σ(D_{T,i})

    Args:
        loyalty_scores: Array of loyalty scores θ_i
        interdependence_weights: Array of team interdependence weights D_{T,i}

    Returns:
        Team cohesion score ∈ [0, 1]
    """
    weighted_sum = np.sum(interdependence_weights * loyalty_scores)
    total_weight = np.sum(interdependence_weights)

    if total_weight < 1e-8:
        return 0.5  # Neutral cohesion

    return float(weighted_sum / total_weight)


# =============================================================================
# BASE TR-3 ENVIRONMENT
# =============================================================================

class BaseTR3Env(CoopetitionEnv):
    """
    Base class for TR-3 collective action environments.

    Provides common functionality for team production and loyalty dynamics.
    """

    def __init__(
        self,
        n_agents: int = 4,
        tr3_params: Optional[TR3Parameters] = None,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize TR-3 base environment.

        Args:
            n_agents: Number of team members
            tr3_params: TR-3 specific parameters
            max_steps: Maximum episode length
            render_mode: Rendering mode
        """
        self.tr3_params = tr3_params or TR3Parameters()
        self._n_agents_tr3 = n_agents

        # Initialize loyalty scores
        self._loyalty_scores = np.full(n_agents, 0.5, dtype=np.float32)
        self._loyalty_history: List[np.ndarray] = []

        # Collective action model
        ca_params = CollectiveActionParameters(
            free_rider_threshold=0.3,
            free_rider_penalty=0.5,
            loyalty_horizon=10,
            loyalty_bonus_rate=0.15,
            loyalty_threshold=0.6,
            coordination_bonus=0.2,
            coordination_threshold=0.5
        )
        self._ca_model = CollectiveActionModel(ca_params)
        self._ca_state: Optional[CollectiveActionState] = None

        # Standard coopetition config
        trust_params = TrustParameters(
            lambda_plus=0.10,
            lambda_minus=0.25,
            mu_R=0.40,
            delta_R=0.02,
            xi=0.50,
            kappa=1.0,
            initial_trust=0.50
        )

        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=20.0,
            gamma=0.65
        )

        D = create_symmetric_interdependence(n_agents, 0.55).matrix
        endowments = np.full(n_agents, self.tr3_params.a_max, dtype=np.float32)

        config = EnvironmentConfig(
            n_agents=n_agents,
            max_steps=max_steps,
            endowments=endowments,
            alpha=np.full(n_agents, 1.0 / n_agents),
            interdependence_matrix=D,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=endowments * 0.4,
            reward_type="integrated",
            normalize_rewards=False,
            reward_scale=1.0,
            render_mode=render_mode
        )

        super().__init__(config=config, **kwargs)

    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        """Reset with TR-3 specific initialization."""
        self._loyalty_scores = np.full(self._n_agents_tr3, 0.5, dtype=np.float32)
        self._loyalty_history = []
        self._ca_state = CollectiveActionState.create_initial(self._n_agents_tr3)
        return super().reset(**kwargs)

    def _update_loyalty(self, actions: np.ndarray) -> None:
        """Update loyalty scores based on actions."""
        cooperation_rates = actions / self.endowments

        # Loyalty increases with cooperation, decreases with free-riding
        for i in range(self._n_agents_tr3):
            if cooperation_rates[i] >= 0.5:
                # Cooperating: loyalty builds slowly
                self._loyalty_scores[i] = min(
                    1.0,
                    self._loyalty_scores[i] + 0.02 * cooperation_rates[i]
                )
            else:
                # Free-riding: loyalty erodes
                self._loyalty_scores[i] = max(
                    0.0,
                    self._loyalty_scores[i] - 0.05 * (0.5 - cooperation_rates[i])
                )

        self._loyalty_history.append(self._loyalty_scores.copy())

    def _compute_tr3_rewards(self, actions: np.ndarray) -> np.ndarray:
        """
        Compute TR-3 team production rewards.

        Uses loyalty-augmented utility from TR-3.
        """
        rewards = np.zeros(self._n_agents_tr3, dtype=np.float32)

        for i in range(self._n_agents_tr3):
            rewards[i] = loyalty_augmented_utility(
                i, actions, self._loyalty_scores[i], self.tr3_params
            )

        return rewards

    def _compute_collective_action_modifier(self, actions: np.ndarray) -> np.ndarray:
        """
        Override to implement TR-3 collective action effects.
        """
        # Update loyalty first
        self._update_loyalty(actions)

        # Update collective action state
        if self._ca_state is not None:
            self._ca_state = self._ca_model.update_state(
                self._ca_state, actions, self.endowments, self._action_history
            )

        # Compute modifiers using CA model
        return self._ca_model.compute_reward_modifiers(
            actions, self.endowments, self._action_history
        )

    def _get_legacy_info(self) -> Dict[str, Any]:
        """Add TR-3 specific info."""
        info = super()._get_legacy_info()

        # TR-3 metrics
        info["team_output"] = float(team_production(
            self._state["actions"], self.tr3_params.omega, self.tr3_params.beta
        ))
        info["nash_equilibrium"] = float(free_riding_equilibrium(
            self.tr3_params, self._n_agents_tr3
        ))
        info["social_optimum"] = float(social_optimum_effort(
            self.tr3_params, self._n_agents_tr3
        ))
        info["mean_loyalty"] = float(np.mean(self._loyalty_scores))
        info["loyalty_scores"] = self._loyalty_scores.copy().tolist()
        info["team_cohesion"] = float(team_cohesion(
            self._loyalty_scores,
            np.mean(self.D, axis=1)  # Use row means as weights
        ))

        # Free-rider detection
        free_riders = self.detect_free_riders(threshold=0.3)
        info["free_rider_count"] = len(free_riders)
        info["free_rider_indices"] = free_riders

        # Efficiency ratio (actual vs optimal)
        actual_effort = np.mean(self._state["actions"])
        optimal_effort = info["social_optimum"]
        if optimal_effort > 0:
            info["efficiency_ratio"] = float(actual_effort / optimal_effort)
        else:
            info["efficiency_ratio"] = 0.0

        return info


# =============================================================================
# ENVIRONMENT 1: TeamProduction-v0
# =============================================================================

class TeamProductionEnv(BaseTR3Env):
    """
    Team Production Environment (TeamProduction-v0)

    Core team production environment demonstrating free-rider dynamics
    without loyalty mechanisms. This serves as the baseline for TR-3
    environments.

    Challenge:
    ----------
    Agents must decide how much effort to contribute to team production.
    The Nash equilibrium (free-riding) leads to suboptimal team output,
    while the social optimum requires trust and coordination.

    Key Dynamics:
    -------------
    - Team production: Q(a) = ω · (Σa_i)^β
    - Each agent receives 1/n share minus their effort cost
    - Diminishing returns (β < 1) create free-rider incentive
    - No loyalty mechanisms (baseline comparison)

    Example:
    --------
    >>> env = TeamProductionEnv(n_agents=4)
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # Nash equilibrium effort
    >>> nash_effort = info["nash_equilibrium"]
    >>> actions = [nash_effort] * 4
    >>> obs, rewards, done, truncated, info = env.step(actions)
    >>> print(f"Team output at Nash: {info['team_output']:.2f}")
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "TeamProduction-v0"
    }

    def __init__(
        self,
        n_agents: int = 4,
        omega: float = 25.0,
        beta: float = 0.7,
        c: float = 1.0,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Team Production environment.

        Args:
            n_agents: Number of team members
            omega: Productivity factor
            beta: Returns to scale (< 1 for diminishing returns)
            c: Effort cost coefficient
            max_steps: Maximum episode length
            render_mode: Rendering mode
        """
        tr3_params = TR3Parameters(
            omega=omega,
            beta=beta,
            c=c,
            phi_B=0.0,  # No loyalty benefit (baseline)
            phi_C=0.0,  # No cost tolerance (baseline)
            a_max=50.0
        )

        super().__init__(
            n_agents=n_agents,
            tr3_params=tr3_params,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs
        )

    def _compute_collective_action_modifier(self, actions: np.ndarray) -> np.ndarray:
        """No loyalty modifiers in baseline environment."""
        # Still track free-riders but don't apply loyalty bonuses
        free_riders = self.detect_free_riders(threshold=0.3)
        modifiers = np.ones(self._n_agents_tr3, dtype=np.float32)

        # Light penalty for severe free-riding (below 20%)
        for i in free_riders:
            rate = actions[i] / self.endowments[i]
            if rate < 0.2:
                modifiers[i] *= 0.9

        return modifiers


# =============================================================================
# ENVIRONMENT 2: LoyaltyTeam-v0
# =============================================================================

class LoyaltyTeamEnv(BaseTR3Env):
    """
    Loyalty Team Environment (LoyaltyTeam-v0)

    Team production with full TR-3 loyalty mechanisms. Tests whether
    loyalty dynamics can sustain cooperation above Nash equilibrium.

    Challenge:
    ----------
    Agents with higher loyalty receive bonuses proportional to teammate
    welfare. This creates positive-sum dynamics where investing in
    team success is intrinsically rewarding for loyal agents.

    Key Dynamics:
    -------------
    - Full loyalty modifier: L_i = θ_i · [φ_B · π̄_{-i} + φ_C · c · a_i]
    - Loyalty builds with sustained cooperation
    - Loyal agents get bonus from teammates' success
    - High-loyalty teams can sustain cooperation above Nash

    Example:
    --------
    >>> env = LoyaltyTeamEnv(n_agents=4, phi_B=0.8, phi_C=0.3)
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # Sustained cooperation builds loyalty
    >>> for _ in range(50):
    ...     obs, rewards, done, truncated, info = env.step([40.0] * 4)
    >>> print(f"Mean loyalty after cooperation: {info['mean_loyalty']:.2f}")
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "LoyaltyTeam-v0"
    }

    def __init__(
        self,
        n_agents: int = 4,
        omega: float = 25.0,
        beta: float = 0.7,
        c: float = 1.0,
        phi_B: float = 0.8,
        phi_C: float = 0.3,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Loyalty Team environment.

        Args:
            n_agents: Number of team members
            omega: Productivity factor
            beta: Returns to scale
            c: Effort cost coefficient
            phi_B: Loyalty benefit strength (bonus from teammate welfare)
            phi_C: Cost tolerance strength (bonus for own effort)
            max_steps: Maximum episode length
            render_mode: Rendering mode
        """
        tr3_params = TR3Parameters(
            omega=omega,
            beta=beta,
            c=c,
            phi_B=phi_B,
            phi_C=phi_C,
            a_max=50.0
        )

        super().__init__(
            n_agents=n_agents,
            tr3_params=tr3_params,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs
        )

    def _get_legacy_info(self) -> Dict[str, Any]:
        """Add loyalty-specific metrics."""
        info = super()._get_legacy_info()

        # Loyalty lift: improvement over no-loyalty baseline
        if len(self._action_history) > 0:
            actual_output = info["team_output"]
            # What would output be at Nash (no loyalty)?
            nash_actions = np.full(self._n_agents_tr3, info["nash_equilibrium"])
            nash_output = team_production(
                nash_actions, self.tr3_params.omega, self.tr3_params.beta
            )
            if nash_output > 0:
                info["loyalty_lift"] = float(actual_output / nash_output)
            else:
                info["loyalty_lift"] = 1.0
        else:
            info["loyalty_lift"] = 1.0

        return info


# =============================================================================
# ENVIRONMENT 3: CoalitionFormation-v0
# =============================================================================

class CoalitionFormationEnv(BaseTR3Env):
    """
    Coalition Formation Environment (CoalitionFormation-v0)

    Dynamic coalition with entry/exit mechanics. Agents can join or leave
    the active coalition, and only coalition members share team output.

    Challenge:
    ----------
    Agents must decide whether to stay in the coalition (sharing output)
    or exit (keeping own effort). Persistent free-riders risk exclusion,
    while reliable contributors attract coalition membership.

    Key Dynamics:
    -------------
    - Only coalition members share team production
    - Agents with very low loyalty may be excluded
    - Excluded agents work independently (lower payoff)
    - Coalition stability requires maintaining critical mass

    Example:
    --------
    >>> env = CoalitionFormationEnv(n_agents=6, min_coalition_size=3)
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # Some agents free-ride
    >>> actions = [40, 40, 40, 5, 5, 5]  # 3 cooperators, 3 free-riders
    >>> for _ in range(20):
    ...     obs, rewards, done, truncated, info = env.step(actions)
    >>> print(f"Coalition size: {info['coalition_size']}")
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "CoalitionFormation-v0"
    }

    def __init__(
        self,
        n_agents: int = 6,
        min_coalition_size: int = 3,
        exit_threshold: float = 0.15,
        reentry_cooldown: int = 5,
        omega: float = 25.0,
        beta: float = 0.7,
        max_steps: int = 150,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Coalition Formation environment.

        Args:
            n_agents: Number of potential team members
            min_coalition_size: Minimum viable coalition
            exit_threshold: Loyalty below which agent risks exit
            reentry_cooldown: Steps before excluded agent can rejoin
            omega: Productivity factor
            beta: Returns to scale
            max_steps: Maximum episode length
            render_mode: Rendering mode
        """
        self._min_coalition_size = min_coalition_size
        self._exit_threshold = exit_threshold
        self._reentry_cooldown = reentry_cooldown

        # Track coalition membership
        self._coalition_members: List[int] = []
        self._excluded_agents: List[int] = []
        self._exclusion_timers: Dict[int, int] = {}

        tr3_params = TR3Parameters(
            omega=omega,
            beta=beta,
            c=1.0,
            phi_B=0.7,
            phi_C=0.3,
            a_max=50.0
        )

        super().__init__(
            n_agents=n_agents,
            tr3_params=tr3_params,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs
        )

    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        """Reset with coalition initialization."""
        self._coalition_members = list(range(self._n_agents_tr3))
        self._excluded_agents = []
        self._exclusion_timers = {}
        return super().reset(**kwargs)

    def _update_coalition(self, actions: np.ndarray) -> None:
        """Update coalition membership based on loyalty."""
        # Check for exclusions
        for i in list(self._coalition_members):
            if self._loyalty_scores[i] < self._exit_threshold:
                # Risk of exclusion - simple threshold check
                if len(self._coalition_members) > self._min_coalition_size:
                    self._coalition_members.remove(i)
                    self._excluded_agents.append(i)
                    self._exclusion_timers[i] = self._reentry_cooldown

        # Check for reentry
        for i in list(self._excluded_agents):
            if i in self._exclusion_timers:
                self._exclusion_timers[i] -= 1
                if self._exclusion_timers[i] <= 0:
                    # Can rejoin if loyalty improved
                    if self._loyalty_scores[i] >= 0.4:
                        self._excluded_agents.remove(i)
                        self._coalition_members.append(i)
                        del self._exclusion_timers[i]

    def _compute_collective_action_modifier(self, actions: np.ndarray) -> np.ndarray:
        """Apply coalition-based modifiers."""
        # First apply base TR-3 modifiers
        modifiers = super()._compute_collective_action_modifier(actions)

        # Update coalition membership
        self._update_coalition(actions)

        # Excluded agents get reduced rewards (work alone)
        for i in self._excluded_agents:
            modifiers[i] *= 0.5  # Independent work is less valuable

        # Coalition bonus if stable
        if len(self._coalition_members) >= self._min_coalition_size:
            for i in self._coalition_members:
                modifiers[i] *= 1.1  # Coalition stability bonus

        return modifiers

    def _get_legacy_info(self) -> Dict[str, Any]:
        """Add coalition-specific metrics."""
        info = super()._get_legacy_info()

        info["coalition_size"] = len(self._coalition_members)
        info["coalition_members"] = self._coalition_members.copy()
        info["excluded_agents"] = self._excluded_agents.copy()
        info["coalition_stability"] = float(
            len(self._coalition_members) / self._n_agents_tr3
        )

        return info

    def _check_terminated(self) -> bool:
        """Terminate if coalition collapses below minimum."""
        if len(self._coalition_members) < self._min_coalition_size:
            return True
        return False


# =============================================================================
# ENVIRONMENT 4: ApacheProject-v0
# =============================================================================

class ApacheProjectEnv(BaseTR3Env):
    """
    Apache Project Environment (ApacheProject-v0)

    Validated case study based on Apache HTTP Server project (1995-2023).
    This environment reproduces the 52/60 validation score from TR-3.

    Challenge:
    ----------
    Model the evolution of contributor dynamics across four project phases,
    with varying team sizes and loyalty multipliers calibrated to historical
    data.

    Phases:
    -------
    1. Emergence (1995-1999): Small core team, high individual impact
    2. Growth (2000-2005): Rapid expansion, moderate loyalty
    3. Maturity (2006-2015): Large stable team, established norms
    4. Evolution (2016-2023): Gradual decline, legacy maintenance

    Example:
    --------
    >>> env = ApacheProjectEnv(phase="emergence")
    >>> obs, info = env.reset(seed=42)
    >>> print(f"Core team size: {info['n_agents']}")
    >>> print(f"Phase loyalty multiplier: {info['phase_loyalty']}")
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "ApacheProject-v0"
    }

    # Phase configurations from TR-3 validation
    PHASE_CONFIG = {
        "emergence": {
            "n_agents": 8,
            "loyalty_multiplier": 1.0,
            "omega": 20.0,
            "expected_effort": 35.0,
            "years": "1995-1999"
        },
        "growth": {
            "n_agents": 15,
            "loyalty_multiplier": 0.85,
            "omega": 25.0,
            "expected_effort": 30.0,
            "years": "2000-2005"
        },
        "maturity": {
            "n_agents": 40,
            "loyalty_multiplier": 0.70,
            "omega": 30.0,
            "expected_effort": 22.0,
            "years": "2006-2015"
        },
        "evolution": {
            "n_agents": 35,
            "loyalty_multiplier": 0.60,
            "omega": 28.0,
            "expected_effort": 18.0,
            "years": "2016-2023"
        }
    }

    def __init__(
        self,
        phase: str = "maturity",
        max_steps: int = 60,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Apache Project environment.

        Args:
            phase: Project phase ("emergence", "growth", "maturity", "evolution")
            max_steps: Maximum episode length (60 for monthly resolution)
            render_mode: Rendering mode
        """
        if phase not in self.PHASE_CONFIG:
            raise ValueError(f"Unknown phase: {phase}. Available: {list(self.PHASE_CONFIG.keys())}")

        self._phase = phase
        config = self.PHASE_CONFIG[phase]

        # Phase-specific loyalty multiplier affects all loyalty scores
        self._phase_loyalty_multiplier = config["loyalty_multiplier"]

        tr3_params = TR3Parameters(
            omega=config["omega"],
            beta=0.70,
            c=1.0,
            phi_B=0.75,
            phi_C=0.25,
            a_max=50.0
        )

        super().__init__(
            n_agents=config["n_agents"],
            tr3_params=tr3_params,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs
        )

        self._expected_effort = config["expected_effort"]

    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        """Reset with phase-appropriate loyalty initialization."""
        result = super().reset(**kwargs)

        # Initialize loyalty with phase-appropriate levels
        self._loyalty_scores *= self._phase_loyalty_multiplier

        return result

    def _update_loyalty(self, actions: np.ndarray) -> None:
        """Update loyalty with phase-specific dynamics."""
        super()._update_loyalty(actions)

        # Apply phase multiplier to cap maximum loyalty
        self._loyalty_scores = np.clip(
            self._loyalty_scores,
            0.0,
            self._phase_loyalty_multiplier
        )

    def _get_legacy_info(self) -> Dict[str, Any]:
        """Add Apache-specific metrics."""
        info = super()._get_legacy_info()

        info["phase"] = self._phase
        info["phase_years"] = self.PHASE_CONFIG[self._phase]["years"]
        info["phase_loyalty"] = self._phase_loyalty_multiplier
        info["expected_effort"] = self._expected_effort

        # Validation accuracy: how close are actions to expected?
        if len(self._action_history) > 0:
            mean_effort = np.mean(self._state["actions"])
            deviation = abs(mean_effort - self._expected_effort) / self._expected_effort
            info["effort_deviation"] = float(deviation)
            info["validation_accuracy"] = float(max(0, 1 - deviation))
        else:
            info["effort_deviation"] = 0.0
            info["validation_accuracy"] = 1.0

        return info


# =============================================================================
# ENVIRONMENT 5: PublicGoods-v0
# =============================================================================

class PublicGoodsEnv(BaseTR3Env):
    """
    Public Goods Environment (PublicGoods-v0)

    Classic public goods game with TR-3 collective action modifiers.
    Agents contribute to a public good that benefits everyone equally.

    Challenge:
    ----------
    The public good is non-excludable: everyone benefits regardless of
    contribution. This creates a strong free-rider incentive that can
    only be overcome through loyalty dynamics and social pressure.

    Key Dynamics:
    -------------
    - Contributions are multiplied and shared equally
    - Multiplier > 1 makes cooperation socially beneficial
    - Individual incentive is to contribute zero (free-ride)
    - Loyalty mechanisms can sustain contributions

    Example:
    --------
    >>> env = PublicGoodsEnv(n_agents=5, multiplier=2.0)
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # Full cooperation
    >>> actions = [50.0] * 5
    >>> obs, rewards, done, truncated, info = env.step(actions)
    >>> print(f"Total public good: {info['public_good']:.2f}")
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "PublicGoods-v0"
    }

    def __init__(
        self,
        n_agents: int = 5,
        multiplier: float = 2.0,
        endowment: float = 50.0,
        phi_B: float = 0.6,
        phi_C: float = 0.2,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Public Goods environment.

        Args:
            n_agents: Number of players
            multiplier: How much contributions are multiplied before sharing
            endowment: Each agent's starting endowment
            phi_B: Loyalty benefit strength
            phi_C: Cost tolerance strength
            max_steps: Maximum episode length
            render_mode: Rendering mode
        """
        self._multiplier = multiplier
        self._endowment = endowment

        # Map to TR-3 parameters
        # omega scales the production function
        # beta = 1.0 for linear returns in public goods
        tr3_params = TR3Parameters(
            omega=multiplier,
            beta=1.0,  # Linear returns in classic public goods
            c=1.0,
            phi_B=phi_B,
            phi_C=phi_C,
            a_max=endowment
        )

        super().__init__(
            n_agents=n_agents,
            tr3_params=tr3_params,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs
        )

    def _compute_tr3_rewards(self, actions: np.ndarray) -> np.ndarray:
        """Compute public goods rewards."""
        n = len(actions)

        # Total contribution
        total_contribution = np.sum(actions)

        # Public good value (multiplied and shared)
        public_good = self._multiplier * total_contribution
        share = public_good / n

        # Each agent gets: kept_endowment + share_of_public_good
        rewards = np.zeros(n, dtype=np.float32)
        for i in range(n):
            kept = self._endowment - actions[i]  # What they kept
            rewards[i] = kept + share

            # Add loyalty bonus
            if self.tr3_params.phi_B > 0 or self.tr3_params.phi_C > 0:
                loyalty_bonus = loyalty_modifier(
                    i, actions, self._loyalty_scores[i],
                    self.tr3_params.omega, 1.0, self.tr3_params.c,
                    self.tr3_params.phi_B, self.tr3_params.phi_C
                )
                rewards[i] += loyalty_bonus

        return rewards

    def _get_legacy_info(self) -> Dict[str, Any]:
        """Add public goods specific metrics."""
        info = super()._get_legacy_info()

        total_contribution = np.sum(self._state["actions"])
        info["total_contribution"] = float(total_contribution)
        info["public_good"] = float(self._multiplier * total_contribution)
        info["contribution_rate"] = float(
            total_contribution / (self._endowment * self._n_agents_tr3)
        )

        # Social efficiency: achieved welfare vs maximum possible
        actual_welfare = np.sum(self._compute_tr3_rewards(self._state["actions"]))
        max_actions = np.full(self._n_agents_tr3, self._endowment)
        max_welfare = np.sum(self._compute_tr3_rewards(max_actions))
        if max_welfare > 0:
            info["social_efficiency"] = float(actual_welfare / max_welfare)
        else:
            info["social_efficiency"] = 0.0

        return info


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def make_team_production(**kwargs) -> TeamProductionEnv:
    """Factory function for TeamProduction-v0."""
    return TeamProductionEnv(**kwargs)


def make_loyalty_team(**kwargs) -> LoyaltyTeamEnv:
    """Factory function for LoyaltyTeam-v0."""
    return LoyaltyTeamEnv(**kwargs)


def make_coalition_formation(**kwargs) -> CoalitionFormationEnv:
    """Factory function for CoalitionFormation-v0."""
    return CoalitionFormationEnv(**kwargs)


def make_apache_project(**kwargs) -> ApacheProjectEnv:
    """Factory function for ApacheProject-v0."""
    return ApacheProjectEnv(**kwargs)


def make_public_goods(**kwargs) -> PublicGoodsEnv:
    """Factory function for PublicGoods-v0."""
    return PublicGoodsEnv(**kwargs)
