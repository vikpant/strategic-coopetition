"""
================================================================================
COOPETITION-GYM: Trust Dynamics Module
================================================================================

This module implements the trust and reputation dynamics from TR-2 
(arXiv:2510.24909). Trust captures how agents update their beliefs about
partners' future behavior based on observed actions.

Mathematical Foundation:
------------------------
From Pant & Yu (2025), trust evolves through a two-layer system:

1. Immediate Trust T_ij: Responds to current behavior
   - Cooperation signal (Eq. 6): s_ij = tanh(κ·(a_j - baseline))
   - Trust building (Eq. 7): ΔT = λ+·s·(1-T)·ceiling (when s > 0)
   - Trust erosion (Eq. 8): ΔT = λ-·s·T·(1+ξ·D_ij) (when s < 0)

2. Reputation R_ij: Tracks violation history, creates trust ceiling
   - Damage accumulation (Eq. 9): ΔR = -μ_R·s·(1-R) when s < 0
   - Gradual decay: ΔR = -δ_R·R when s ≥ 0

Key Behavioral Properties:
--------------------------
- Asymmetric updating: Trust erodes ~3× faster than it builds (negativity bias)
- Hysteresis: Reputation damage creates persistent trust ceiling
- Interdependence amplification: Higher D_ij increases trust sensitivity to violations

Validated Parameters (TR-2 §8):
-------------------------------
- λ+ = 0.10 (trust building rate)
- λ- = 0.30 (trust erosion rate, giving 3× negativity ratio)
- μ_R = 0.60 (reputation damage severity)
- δ_R = 0.03 (reputation decay/forgetting rate)
- ξ = 0.50 (interdependence amplification factor)
- κ = 1.0 (signal sensitivity)

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass, field
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrustParameters:
    """
    Configuration parameters for trust dynamics.
    
    These parameters control how trust evolves in response to observed behavior.
    Defaults are set to the validated values from TR-2 §8 (Renault-Nissan case).
    
    Attributes:
        lambda_plus: Trust building rate when cooperation is observed.
                     Controls how quickly trust grows after positive signals.
        lambda_minus: Trust erosion rate when violation is observed.
                      Higher than lambda_plus to capture negativity bias.
        mu_R: Reputation damage severity. Controls how much a single
              violation impacts long-term reputation.
        delta_R: Reputation decay rate. Controls how quickly past violations
                 are "forgotten" in the absence of new violations.
        xi: Interdependence amplification factor. Higher interdependence
            makes trust more sensitive to violations.
        kappa: Signal sensitivity. Controls the steepness of the tanh
               function that maps actions to cooperation signals.
        initial_trust: Starting trust level for new relationships.
        initial_reputation: Starting reputation damage (0 = clean slate).
    
    Design Rationale:
        The 3× ratio between lambda_minus and lambda_plus captures the
        well-documented negativity bias in trust research - negative
        events are weighted more heavily than positive events of equal
        magnitude. The reputation system creates hysteresis where past
        violations continue to constrain trust even after many periods
        of cooperation.
    """
    lambda_plus: float = 0.10
    lambda_minus: float = 0.30
    mu_R: float = 0.60
    delta_R: float = 0.03
    xi: float = 0.50
    kappa: float = 1.0
    initial_trust: float = 0.50
    initial_reputation: float = 0.0
    
    def __post_init__(self):
        """Validate parameter bounds and relationships."""
        if not 0 < self.lambda_plus <= 1:
            raise ValueError(f"lambda_plus must be in (0,1], got {self.lambda_plus}")
        if not 0 < self.lambda_minus <= 1:
            raise ValueError(f"lambda_minus must be in (0,1], got {self.lambda_minus}")
        if not 0 <= self.mu_R <= 1:
            raise ValueError(f"mu_R must be in [0,1], got {self.mu_R}")
        if not 0 <= self.delta_R <= 1:
            raise ValueError(f"delta_R must be in [0,1], got {self.delta_R}")
        if self.xi < 0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if not 0 <= self.initial_trust <= 1:
            raise ValueError(f"initial_trust must be in [0,1], got {self.initial_trust}")
        if not 0 <= self.initial_reputation <= 1:
            raise ValueError(f"initial_reputation must be in [0,1], got {self.initial_reputation}")
    
    @property
    def negativity_ratio(self) -> float:
        """Compute the negativity bias ratio λ-/λ+."""
        return self.lambda_minus / self.lambda_plus
    
    def with_updates(self, **kwargs) -> "TrustParameters":
        """Create a new TrustParameters with specified updates."""
        params = {
            "lambda_plus": self.lambda_plus,
            "lambda_minus": self.lambda_minus,
            "mu_R": self.mu_R,
            "delta_R": self.delta_R,
            "xi": self.xi,
            "kappa": self.kappa,
            "initial_trust": self.initial_trust,
            "initial_reputation": self.initial_reputation,
        }
        params.update(kwargs)
        return TrustParameters(**params)


@dataclass
class TrustState:
    """
    Complete trust state between all pairs of agents.
    
    Maintains both the immediate trust matrix T and the reputation damage
    matrix R. The effective trust ceiling for agent i trusting agent j is
    constrained by: ceiling_ij = 1 - R_ij
    
    Attributes:
        trust_matrix: NxN matrix where T[i,j] is i's trust in j
        reputation_matrix: NxN matrix where R[i,j] is i's damage record of j
        n_agents: Number of agents in the system
    """
    trust_matrix: NDArray[np.floating]
    reputation_matrix: NDArray[np.floating]
    
    def __post_init__(self):
        """Validate state consistency."""
        if self.trust_matrix.shape != self.reputation_matrix.shape:
            raise ValueError("Trust and reputation matrices must have same shape")
        if self.trust_matrix.ndim != 2:
            raise ValueError("Matrices must be 2D")
        
        # Clip values to valid ranges
        self.trust_matrix = np.clip(self.trust_matrix, 0.0, 1.0)
        self.reputation_matrix = np.clip(self.reputation_matrix, 0.0, 1.0)
        
        # Diagonal should be 1 (self-trust) and 0 (self-reputation)
        np.fill_diagonal(self.trust_matrix, 1.0)
        np.fill_diagonal(self.reputation_matrix, 0.0)
    
    @property
    def n_agents(self) -> int:
        """Number of agents."""
        return self.trust_matrix.shape[0]
    
    @property
    def trust_ceiling(self) -> NDArray[np.floating]:
        """
        Compute the trust ceiling matrix.
        
        The ceiling constrains how high trust can grow, based on
        accumulated reputation damage: ceiling_ij = 1 - R_ij
        """
        return 1.0 - self.reputation_matrix
    
    def get_trust(self, i: int, j: int) -> float:
        """Get trust of agent i in agent j."""
        return float(self.trust_matrix[i, j])
    
    def get_reputation_damage(self, i: int, j: int) -> float:
        """Get reputation damage that agent i has recorded for agent j."""
        return float(self.reputation_matrix[i, j])
    
    def get_effective_trust(self, i: int, j: int) -> float:
        """
        Get effective trust, constrained by reputation ceiling.
        
        Effective trust is the minimum of current trust and the ceiling
        imposed by reputation damage.
        """
        ceiling = 1.0 - self.reputation_matrix[i, j]
        return float(min(self.trust_matrix[i, j], ceiling))
    
    def mean_trust(self) -> float:
        """Compute mean trust across all agent pairs (excluding self)."""
        n = self.n_agents
        if n <= 1:
            return 1.0
        # Exclude diagonal (self-trust)
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(self.trust_matrix[mask]))
    
    def mean_reputation_damage(self) -> float:
        """Compute mean reputation damage across all pairs."""
        n = self.n_agents
        if n <= 1:
            return 0.0
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(self.reputation_matrix[mask]))
    
    def copy(self) -> "TrustState":
        """Create a deep copy of the trust state."""
        return TrustState(
            trust_matrix=self.trust_matrix.copy(),
            reputation_matrix=self.reputation_matrix.copy()
        )
    
    def to_flat_array(self) -> NDArray[np.floating]:
        """Flatten state to 1D array for observation space."""
        return np.concatenate([
            self.trust_matrix.flatten(),
            self.reputation_matrix.flatten()
        ])
    
    @classmethod
    def from_flat_array(cls, arr: NDArray[np.floating], n_agents: int) -> "TrustState":
        """Reconstruct state from flattened array."""
        n_elements = n_agents * n_agents
        trust = arr[:n_elements].reshape(n_agents, n_agents)
        reputation = arr[n_elements:2*n_elements].reshape(n_agents, n_agents)
        return cls(trust_matrix=trust, reputation_matrix=reputation)


class TrustDynamicsModel:
    """
    Implements the trust evolution dynamics from TR-2.
    
    This is the core model that updates trust and reputation based on
    observed agent behaviors. It captures the key empirical findings:
    - Trust builds slowly through sustained cooperation
    - Trust erodes quickly when violations occur (negativity bias)
    - Reputation damage creates persistent constraints (hysteresis)
    - Interdependence amplifies trust sensitivity to violations
    
    The model is stateless - it computes updates but doesn't maintain state.
    State is stored externally (typically in the environment).
    
    Example:
        >>> params = TrustParameters()
        >>> model = TrustDynamicsModel(params)
        >>> state = model.create_initial_state(n_agents=2)
        >>> 
        >>> # Agent 1 cooperates above baseline
        >>> actions = np.array([50.0, 70.0])
        >>> baselines = np.array([50.0, 50.0])
        >>> D = np.array([[0, 0.6], [0.8, 0]])
        >>> 
        >>> new_state = model.update(state, actions, baselines, D)
        >>> new_state.get_trust(0, 1)  # Trust of 0 in 1 increased
        0.52...
    """
    
    def __init__(self, params: Optional[TrustParameters] = None):
        """
        Initialize the trust dynamics model.
        
        Args:
            params: Trust dynamics parameters. Uses defaults if not provided.
        """
        self.params = params or TrustParameters()
    
    def create_initial_state(self, n_agents: int) -> TrustState:
        """
        Create initial trust state for a new episode.
        
        All agents start with the initial trust level (default 0.5) and
        no reputation damage (clean slate).
        
        Args:
            n_agents: Number of agents in the environment
        
        Returns:
            Fresh TrustState with initial values
        """
        trust = np.full(
            (n_agents, n_agents), 
            self.params.initial_trust,
            dtype=np.float64
        )
        reputation = np.full(
            (n_agents, n_agents),
            self.params.initial_reputation,
            dtype=np.float64
        )
        
        return TrustState(trust_matrix=trust, reputation_matrix=reputation)
    
    def cooperation_signal(
        self,
        action: float,
        baseline: float
    ) -> float:
        """
        Compute the cooperation signal from an observed action.
        
        This implements TR-2 Equation 6: s = tanh(κ·(a - baseline))
        
        The signal is bounded in [-1, 1]:
        - Positive signal indicates cooperation above expectations
        - Negative signal indicates violation (below expectations)
        - Signal of 0 indicates exactly meeting baseline expectations
        
        The tanh function ensures bounded signals while being sensitive
        to deviations near the baseline.
        
        Args:
            action: Observed action from the agent
            baseline: Expected/baseline action level
        
        Returns:
            Cooperation signal in [-1, 1]
        """
        deviation = action - baseline
        signal = np.tanh(self.params.kappa * deviation)
        return float(signal)
    
    def compute_trust_update(
        self,
        current_trust: float,
        signal: float,
        ceiling: float,
        interdependence: float
    ) -> float:
        """
        Compute the trust update delta.
        
        This implements TR-2 Equations 7-8 with asymmetric dynamics:
        - When s > 0: ΔT = λ+·s·(1-T)·ceiling (trust building)
        - When s < 0: ΔT = λ-·s·T·(1+ξ·D) (trust erosion)
        
        Trust building is gradual and constrained by both the current
        trust level (harder to build when already high) and the ceiling
        imposed by reputation damage.
        
        Trust erosion is faster and amplified by interdependence - agents
        who depend more on a partner are more sensitive to violations.
        
        Args:
            current_trust: Current trust level T_ij
            signal: Cooperation signal s ∈ [-1, 1]
            ceiling: Trust ceiling (1 - R_ij)
            interdependence: Dependency coefficient D_ij
        
        Returns:
            Trust update delta ΔT
        """
        if signal > 0:
            # Trust building: gradual, constrained by ceiling
            # Growth slows as trust approaches ceiling (logistic-like)
            room_to_grow = min(1 - current_trust, ceiling - current_trust)
            room_to_grow = max(0, room_to_grow)
            delta = self.params.lambda_plus * signal * room_to_grow
        elif signal < 0:
            # Trust erosion: faster, amplified by interdependence
            # Higher interdependence → more sensitive to violations
            amplification = 1 + self.params.xi * interdependence
            delta = self.params.lambda_minus * signal * current_trust * amplification
        else:
            delta = 0.0
        
        return delta
    
    def compute_reputation_update(
        self,
        current_reputation: float,
        signal: float
    ) -> float:
        """
        Compute the reputation damage update.
        
        This implements TR-2 Equation 9:
        - When s < 0: ΔR = -μ_R·s·(1-R) (damage accumulation)
        - When s ≥ 0: ΔR = -δ_R·R (gradual decay)
        
        Reputation damage accumulates with violations and decays slowly
        over time when no violations occur. This creates hysteresis where
        past violations continue to affect trust capacity.
        
        Args:
            current_reputation: Current reputation damage R_ij ∈ [0,1]
            signal: Cooperation signal s ∈ [-1,1]
        
        Returns:
            Reputation update delta ΔR
        """
        if signal < 0:
            # Violation: damage accumulates
            # -signal is positive when signal is negative
            room_for_damage = 1 - current_reputation
            delta = self.params.mu_R * (-signal) * room_for_damage
        else:
            # No violation: gradual decay (forgetting)
            delta = -self.params.delta_R * current_reputation
        
        return delta
    
    def update(
        self,
        state: TrustState,
        actions: NDArray[np.floating],
        baselines: NDArray[np.floating],
        interdependence: NDArray[np.floating]
    ) -> TrustState:
        """
        Update trust state based on observed actions.
        
        This is the main update function that processes all pairwise
        trust relationships based on the observed action profile.
        
        Args:
            state: Current trust state
            actions: Array of actions taken by each agent
            baselines: Expected baseline actions for each agent
            interdependence: NxN interdependence matrix D
        
        Returns:
            New TrustState after updates
        
        Note:
            The returned state is a new object; the input is not modified.
        """
        new_state = state.copy()
        n_agents = state.n_agents
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue  # Skip self-trust
                
                # Compute cooperation signal for agent j
                signal = self.cooperation_signal(actions[j], baselines[j])
                
                # Get current values
                current_trust = state.trust_matrix[i, j]
                current_reputation = state.reputation_matrix[i, j]
                ceiling = 1.0 - current_reputation
                D_ij = interdependence[i, j]
                
                # Compute updates
                trust_delta = self.compute_trust_update(
                    current_trust, signal, ceiling, D_ij
                )
                reputation_delta = self.compute_reputation_update(
                    current_reputation, signal
                )
                
                # Apply updates with clipping
                new_state.trust_matrix[i, j] = np.clip(
                    current_trust + trust_delta, 0.0, ceiling
                )
                new_state.reputation_matrix[i, j] = np.clip(
                    current_reputation + reputation_delta, 0.0, 1.0
                )
        
        return new_state
    
    def simulate_trajectory(
        self,
        initial_state: TrustState,
        action_sequence: List[NDArray[np.floating]],
        baselines: NDArray[np.floating],
        interdependence: NDArray[np.floating]
    ) -> List[TrustState]:
        """
        Simulate a sequence of trust state updates.
        
        Useful for analysis and visualization of trust dynamics over time.
        
        Args:
            initial_state: Starting trust state
            action_sequence: List of action profiles over time
            baselines: Baseline expectations (assumed constant)
            interdependence: Interdependence matrix (assumed constant)
        
        Returns:
            List of trust states including initial state
        """
        trajectory = [initial_state]
        current_state = initial_state
        
        for actions in action_sequence:
            current_state = self.update(
                current_state, actions, baselines, interdependence
            )
            trajectory.append(current_state)
        
        return trajectory


# =============================================================================
# Specialized Trust Models
# =============================================================================

class TrustDilemmaModel(TrustDynamicsModel):
    """
    Trust dynamics optimized for the TrustDilemma environment.
    
    Uses slightly more sensitive parameters to create interesting
    dynamics in short episodes.
    """
    
    def __init__(self):
        super().__init__(TrustParameters(
            lambda_plus=0.15,
            lambda_minus=0.45,
            mu_R=0.50,
            delta_R=0.02,
            xi=0.60,
            kappa=1.5,
            initial_trust=0.50
        ))


class RecoveryModel(TrustDynamicsModel):
    """
    Trust dynamics for post-crisis recovery scenarios.
    
    Starts with low trust and high reputation damage, with slower
    recovery dynamics to test patience and coordination.
    """
    
    def __init__(self):
        super().__init__(TrustParameters(
            lambda_plus=0.08,  # Slower recovery
            lambda_minus=0.35,
            mu_R=0.70,
            delta_R=0.01,  # Very slow forgetting
            xi=0.40,
            kappa=1.0,
            initial_trust=0.25,  # Start with low trust
            initial_reputation=0.50  # Start with damage
        ))


class AutomotiveAllianceModel(TrustDynamicsModel):
    """
    Trust dynamics calibrated for automotive alliance case studies.
    
    Based on the Renault-Nissan validation in TR-2 §9, these parameters
    capture the longer time horizons and institutional nature of
    automotive partnerships.
    """
    
    def __init__(self):
        super().__init__(TrustParameters(
            lambda_plus=0.08,
            lambda_minus=0.25,
            mu_R=0.55,
            delta_R=0.02,
            xi=0.50,
            kappa=1.0,
            initial_trust=0.60  # Start with baseline institutional trust
        ))


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_negativity_bias(params: TrustParameters) -> Dict[str, float]:
    """
    Analyze the negativity bias in trust dynamics.
    
    Returns metrics about how much faster trust erodes vs builds.
    """
    return {
        "negativity_ratio": params.negativity_ratio,
        "recovery_periods": estimate_recovery_periods(params, damage=0.3),
        "full_erosion_periods": estimate_erosion_periods(params, start_trust=0.8)
    }


def estimate_recovery_periods(
    params: TrustParameters,
    damage: float,
    threshold: float = 0.9
) -> int:
    """
    Estimate periods needed to recover from reputation damage.
    
    Simulates cooperative behavior to estimate how long it takes
    to recover from a given level of reputation damage.
    """
    reputation = damage
    periods = 0
    max_periods = 1000
    
    while reputation > (1 - threshold) * damage and periods < max_periods:
        reputation -= params.delta_R * reputation
        periods += 1
    
    return periods


def estimate_erosion_periods(
    params: TrustParameters,
    start_trust: float,
    target_trust: float = 0.2
) -> int:
    """
    Estimate periods of consistent violation to erode trust.
    
    Simulates defecting behavior to see how quickly trust collapses.
    """
    trust = start_trust
    periods = 0
    max_periods = 100
    
    # Assume strong violation signal
    signal = -0.8
    
    while trust > target_trust and periods < max_periods:
        delta = params.lambda_minus * signal * trust
        trust = max(0, trust + delta)
        periods += 1
    
    return periods


def compute_trust_equilibrium(
    params: TrustParameters,
    consistent_signal: float,
    interdependence: float = 0.5
) -> float:
    """
    Compute the equilibrium trust level for a consistent signal.
    
    If agents consistently cooperate or defect at a fixed level,
    trust converges to an equilibrium value.
    """
    if consistent_signal >= 0:
        # Trust grows until ΔT = 0
        # λ+·s·(1-T) = 0 → T = 1 (assuming no reputation damage)
        return 1.0
    else:
        # Trust erodes until ΔT = 0
        # This is more complex due to reputation accumulation
        # Approximate by finding where erosion balances
        amplification = 1 + params.xi * interdependence
        # At equilibrium: λ-·|s|·T = 0 → T = 0
        return 0.0
