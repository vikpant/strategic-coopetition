"""
================================================================================
COOPETITION-GYM: Reciprocity Environments (TR-4 Category)
================================================================================

AUTHORITATIVE TR-4 implementation. The full paper formalism
(cooperation signal, memory-windowed baseline, bounded response,
reciprocity sensitivity, trust-gated reciprocity modifier, and
complete utility) is implemented in this file via the TR4Parameters
dataclass and its helper functions. Auxiliary state-tracking utilities
(ReciprocityParameters, ReciprocityState, ReciprocityModel) live in
core/reciprocity.py.

Technical Report: TR-4 (arXiv:2604.01240)
Title: Computational Foundations for Strategic Coopetition:
       Formalizing Sequential Interaction and Reciprocity

This module implements environments for studying sequential interaction and
reciprocity dynamics in coopetitive settings. These environments test whether
agents can learn conditional cooperation strategies based on observed partner
behavior over bounded memory windows.

Environments:
-------------
1. ReciprocalDilemma-v0: Continuous iterated PD with direct reciprocity
2. GiftExchange-v0: Asymmetric employer-worker gift exchange
3. IndirectReciprocity-v0: Population-level reputation and image scoring
4. GraduatedSanction-v0: Common-pool resource with graduated sanctions
5. AppleAppStore-v0: Validated Apple iOS App Store case study (2008-2024)

Mathematical Framework (TR-4):
------------------------------
- Cooperation signal: s_ij = a_j - ā_j                           [Eq 19]
- Memory average: ā_j = (1/min(k,t-1)) Σ a_j^τ                  [Eq 20]
- Bounded response: φ(x) = tanh(κx)                              [Eq 21]
- Reciprocity sensitivity: ρ_ij = ρ_0 · D_ij^η                   [Eq 23]
- Reciprocity modifier: U_recip = λ_R Σ T_ij·(1+ωD_ij)·ρ_ij·φ(s_ij) [Eq 44]
- Complete utility: U_i = π_base + U_interdep + U_trust + U_recip [Eq 45]

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
from core.interdependence import (
    create_symmetric_interdependence,
    create_asymmetric_interdependence
)


# =============================================================================
# TR-4 MATHEMATICAL CORE
# =============================================================================

@dataclass
class TR4Parameters:
    """
    Parameters for TR-4 reciprocity model.

    From TR-4 Equations 19-25, 44:
    - s_ij = a_j - ā_j                                [Eq 19]
    - ā_j = (1/min(k,t-1)) Σ a_j^τ                   [Eq 20]
    - φ(x) = tanh(κx)                                 [Eq 21]
    - ρ_ij = ρ_0 · D_ij^η                             [Eq 23]
    - U_recip = λ_R Σ T_ij·(1+ωD_ij)·ρ_ij·φ(s_ij)   [Eq 44]
    """
    rho_0: float = 1.0      # Base reciprocity strength (Eq 23)
    eta: float = 1.0         # Dependency elasticity (Eq 23)
    kappa: float = 1.0       # Response sensitivity (Eq 21)
    k: int = 5               # Memory window length (Eq 20)
    lambda_R: float = 1.0    # Reciprocity weight (Eq 44)
    omega: float = 0.6       # Dependency amplification (Eq 44)


def cooperation_signal(
    current_action: float,
    memory_avg: float
) -> float:
    """
    Compute cooperation signal (Eq 19).

    s_ij = a_j - ā_j

    Positive signal: cooperating above recent norm.
    Negative signal: defecting below recent norm.

    Args:
        current_action: Agent j's current action a_j^t
        memory_avg: Moving average ā_j of j's recent actions

    Returns:
        Raw cooperation signal (unbounded)
    """
    return current_action - memory_avg


def memory_average(
    action_history: List[np.ndarray],
    agent_idx: int,
    k: int,
    t: int
) -> float:
    """
    Compute moving average of agent's actions over memory window (Eq 20).

    ā_j = (1/min(k, t-1)) Σ_{τ=max(1,t-k)}^{t-1} a_j^τ

    Args:
        action_history: List of past action arrays (each array has all agents)
        agent_idx: Index of agent j
        k: Memory window length
        t: Current time step (1-indexed; len(action_history) before current)

    Returns:
        Moving average of agent's actions
    """
    if t <= 1 or len(action_history) == 0:
        return 0.0

    window_size = min(k, t - 1)
    recent = action_history[-window_size:]
    return float(np.mean([h[agent_idx] for h in recent]))


def bounded_response(signal: float, kappa: float) -> float:
    """
    Bounded response function (Eq 21).

    φ(x) = tanh(κx)

    Maps unbounded cooperation signals to bounded reciprocity responses in (-1, 1).

    Args:
        signal: Raw cooperation signal s_ij
        kappa: Response sensitivity parameter

    Returns:
        Bounded response in (-1, 1)
    """
    return float(np.tanh(kappa * signal))


def reciprocity_sensitivity(
    rho_0: float,
    D_ij: float,
    eta: float
) -> float:
    """
    Compute reciprocity sensitivity from structural dependency (Eq 23).

    ρ_ij = ρ_0 · D_ij^η

    Args:
        rho_0: Base reciprocity tendency
        D_ij: Structural dependency from interdependence matrix
        eta: Dependency elasticity parameter

    Returns:
        Reciprocity sensitivity ρ_ij >= 0
    """
    return rho_0 * (D_ij ** eta)


def trust_gated_reciprocity(
    T_ij: float,
    rho_ij: float,
    phi_s: float,
    lambda_R: float,
    omega: float,
    D_ij: float
) -> float:
    """
    Compute trust-gated reciprocity effect for one partner pair (Eq 44 inner term).

    λ_R · T_ij · (1 + ω·D_ij) · ρ_ij · φ(s_ij)

    Args:
        T_ij: Trust from agent i toward agent j
        rho_ij: Reciprocity sensitivity
        phi_s: Bounded response φ(s_ij)
        lambda_R: Reciprocity weight
        omega: Dependency amplification weight
        D_ij: Structural dependency

    Returns:
        Reciprocity contribution from this partner pair
    """
    return lambda_R * T_ij * (1.0 + omega * D_ij) * rho_ij * phi_s


# =============================================================================
# BASE TR-4 ENVIRONMENT
# =============================================================================

class BaseTR4Env(CoopetitionEnv):
    """
    Base class for TR-4 reciprocity environments.

    Provides common functionality for sequential interaction and reciprocity
    dynamics. Overrides _compute_reciprocity_modifier() to implement Eq 44.
    """

    def __init__(
        self,
        n_agents: int = 2,
        tr4_params: Optional[TR4Parameters] = None,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        trust_params: Optional[TrustParameters] = None,
        value_params: Optional[ValueFunctionParameters] = None,
        endowments: Optional[np.ndarray] = None,
        interdependence_matrix: Optional[np.ndarray] = None,
        baselines: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Initialize TR-4 base environment.

        Args:
            n_agents: Number of agents
            tr4_params: TR-4 specific reciprocity parameters
            max_steps: Maximum episode length
            render_mode: Rendering mode
            trust_params: Trust dynamics parameters (TR-2)
            value_params: Value function parameters (TR-1)
            endowments: Per-agent resource endowments
            interdependence_matrix: NxN dependency structure
            baselines: Baseline cooperation expectations
        """
        self.tr4_params = tr4_params or TR4Parameters()
        self._n_agents_tr4 = n_agents

        # Default trust parameters (canonical TR-2 values)
        if trust_params is None:
            trust_params = TrustParameters(
                lambda_plus=0.10,
                lambda_minus=0.30,
                mu_R=0.60,
                delta_R=0.03,
                xi=0.50,
                kappa=1.0,
                initial_trust=0.50
            )

        # Default value parameters
        if value_params is None:
            value_params = ValueFunctionParameters(
                specification=ValueSpecification.LOGARITHMIC,
                theta=20.0,
                gamma=0.65
            )

        # Default endowments
        if endowments is None:
            endowments = np.full(n_agents, 100.0, dtype=np.float32)

        # Default interdependence matrix
        if interdependence_matrix is None:
            interdependence_matrix = create_symmetric_interdependence(
                n_agents, 0.5
            ).matrix

        # Default baselines (30% of endowments)
        if baselines is None:
            baselines = endowments * 0.3

        config = EnvironmentConfig(
            n_agents=n_agents,
            max_steps=max_steps,
            endowments=endowments,
            alpha=np.full(n_agents, 1.0 / n_agents),
            interdependence_matrix=interdependence_matrix,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=baselines,
            reward_type="integrated",
            normalize_rewards=False,
            reward_scale=1.0,
            render_mode=render_mode
        )

        super().__init__(config=config, **kwargs)

    def reset(self, **kwargs) -> Tuple[NDArray, Dict]:
        """Reset with TR-4 specific initialization."""
        return super().reset(**kwargs)

    def _compute_reciprocity_modifier(self, agent_idx: int) -> float:
        """
        Compute reciprocity effect on agent's reward (Eq 44).

        U_recip_i = λ_R Σ_{j≠i} T_ij · (1+ω·D_ij) · ρ_ij · φ(s_ij)

        The additive U_recip is converted to a multiplicative modifier by
        normalizing against the mean endowment as a stable scale reference.

        Returns:
            Reward multiplier (floored at 0.01 to prevent sign flip)
        """
        # Need at least 1 step of history to compute cooperation signals
        if len(self._action_history) < 1:
            return 1.0

        p = self.tr4_params
        t = len(self._action_history)  # Number of completed steps
        current_actions = self._state["actions"]
        trust_matrix = self._state["trust"]

        u_recip = 0.0
        for j in range(self.n_agents):
            if j == agent_idx:
                continue

            # Eq 20: Memory average of j's actions
            avg_j = memory_average(self._action_history, j, p.k, t)

            # Eq 19: Cooperation signal
            s_ij = cooperation_signal(current_actions[j], avg_j)

            # Eq 21: Bounded response
            phi_s = bounded_response(s_ij, p.kappa)

            # Eq 23: Reciprocity sensitivity
            D_ij = float(self.D[agent_idx, j])
            rho_ij = reciprocity_sensitivity(p.rho_0, D_ij, p.eta)

            # Eq 44: Trust-gated reciprocity
            T_ij = float(trust_matrix[agent_idx, j])
            u_recip += trust_gated_reciprocity(
                T_ij, rho_ij, phi_s, p.lambda_R, p.omega, D_ij
            )

        # Convert additive reciprocity to multiplicative modifier
        # Normalize against mean endowment for stable scale
        base_scale = float(np.mean(self.endowments))
        modifier = 1.0 + u_recip / base_scale

        return max(0.01, modifier)

    def _get_legacy_info(self) -> Dict[str, Any]:
        """Add TR-4 specific info to step output."""
        info = super()._get_legacy_info()

        p = self.tr4_params
        t = len(self._action_history)
        current_actions = self._state["actions"]
        trust_matrix = self._state["trust"]

        # Compute per-pair cooperation signals and reciprocity effects
        signals = {}
        reciprocity_effects = {}
        memory_averages = {}

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                pair_key = f"{i}->{j}"

                avg_j = memory_average(self._action_history, j, p.k, t) if t > 0 else 0.0
                memory_averages[pair_key] = avg_j

                if t > 0:
                    s_ij = cooperation_signal(current_actions[j], avg_j)
                    signals[pair_key] = s_ij

                    phi_s = bounded_response(s_ij, p.kappa)
                    D_ij = float(self.D[i, j])
                    rho_ij = reciprocity_sensitivity(p.rho_0, D_ij, p.eta)
                    T_ij = float(trust_matrix[i, j])
                    eff = trust_gated_reciprocity(
                        T_ij, rho_ij, phi_s, p.lambda_R, p.omega, D_ij
                    )
                    reciprocity_effects[pair_key] = eff
                else:
                    signals[pair_key] = 0.0
                    reciprocity_effects[pair_key] = 0.0

        info["cooperation_signals"] = signals
        info["reciprocity_effects"] = reciprocity_effects
        info["memory_averages"] = memory_averages
        info["tr4_memory_window"] = p.k

        return info


# =============================================================================
# ENVIRONMENT 1: ReciprocalDilemma-v0
# =============================================================================

class ReciprocalDilemmaEnv(BaseTR4Env):
    """
    Reciprocal Dilemma Environment (ReciprocalDilemma-v0)

    Continuous iterated Prisoner's Dilemma with TR-4 reciprocity dynamics.
    Two symmetric firms decide cooperation level in a shared project.

    Game-Theoretic Foundation:
    --------------------------
    - Axelrod (1984) "The Evolution of Cooperation"
    - Killingback & Doebeli (2002) "The Continuous Prisoner's Dilemma"

    Challenge:
    ----------
    Agents must learn conditional cooperation. The reciprocity modifier
    enables tit-for-tat-like strategies: cooperation above baseline generates
    positive signals that encourage partner reciprocation, while defection
    generates negative signals that trigger proportional retaliation.

    Distinction from TrustDilemma-v0:
    ---------------------------------
    TrustDilemma uses TR-2 trust dynamics alone (slow erosion/building).
    ReciprocalDilemma adds fast TR-4 behavioral reciprocity — agents respond
    to cooperation signals within 1-5 steps via memory window, enabling
    tit-for-tat dynamics that TrustDilemma cannot express.

    Example:
    --------
    >>> env = ReciprocalDilemmaEnv()
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # Sustained cooperation builds positive reciprocity
    >>> for _ in range(20):
    ...     obs, rewards, done, truncated, info = env.step([70.0, 70.0])
    >>> print(f"Signals: {info['cooperation_signals']}")
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "ReciprocalDilemma-v0",
        "source": "TR-4, Axelrod (1984), Killingback & Doebeli (2002)"
    }

    def __init__(
        self,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        tr4_params = TR4Parameters(
            rho_0=1.0,
            eta=1.0,
            kappa=1.0,
            k=5,
            lambda_R=1.0,
            omega=0.6
        )

        D = create_symmetric_interdependence(2, 0.5).matrix

        super().__init__(
            n_agents=2,
            tr4_params=tr4_params,
            max_steps=max_steps,
            render_mode=render_mode,
            endowments=np.array([100.0, 100.0], dtype=np.float32),
            interdependence_matrix=D,
            baselines=np.array([30.0, 30.0], dtype=np.float32),
            **kwargs
        )


# =============================================================================
# ENVIRONMENT 2: GiftExchange-v0
# =============================================================================

class GiftExchangeEnv(BaseTR4Env):
    """
    Gift Exchange Environment (GiftExchange-v0)

    Asymmetric employer-worker exchange with TR-4 reciprocity dynamics.
    The employer (Agent 0) sets wage-cooperation and the worker (Agent 1)
    responds with effort-cooperation.

    Game-Theoretic Foundation:
    --------------------------
    - Fehr, Kirchsteiger & Riedl (1993) "Does Fairness Prevent Market Clearing?"
    - Akerlof (1982) "Labor Markets as Partial Gift Exchange"

    Challenge:
    ----------
    The asymmetric dependency (D_21=0.7 >> D_12=0.4) means the worker
    reciprocates more strongly than the employer. The employer must learn
    that setting wages above baseline triggers reciprocal effort that more
    than compensates. The worker must learn fair reciprocation rather than
    exploiting generous wages.

    Distinction from PartnerHoldUp-v0:
    ----------------------------------
    PartnerHoldUp models structural lock-in and sunk costs (TR-1/TR-2).
    GiftExchange models ongoing voluntary reciprocity with no lock-in.
    Worker's reciprocity sensitivity ρ_21 = 1.2·0.7^1.5 ≈ 0.703 vs
    employer's ρ_12 = 1.2·0.4^1.5 ≈ 0.304.

    Example:
    --------
    >>> env = GiftExchangeEnv()
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # Employer offers fair wage, worker reciprocates
    >>> for _ in range(20):
    ...     obs, rewards, done, truncated, info = env.step([60.0, 50.0])
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "GiftExchange-v0",
        "source": "TR-4, Fehr et al (1993), Akerlof (1982)"
    }

    def __init__(
        self,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        tr4_params = TR4Parameters(
            rho_0=1.2,
            eta=1.5,
            kappa=1.0,
            k=3,
            lambda_R=1.2,
            omega=0.8
        )

        # Asymmetric dependency: worker more dependent on employer
        D = np.array([
            [0.0, 0.4],
            [0.7, 0.0]
        ], dtype=np.float32)

        super().__init__(
            n_agents=2,
            tr4_params=tr4_params,
            max_steps=max_steps,
            render_mode=render_mode,
            endowments=np.array([100.0, 80.0], dtype=np.float32),
            interdependence_matrix=D,
            baselines=np.array([30.0, 24.0], dtype=np.float32),
            **kwargs
        )


# =============================================================================
# ENVIRONMENT 3: IndirectReciprocity-v0
# =============================================================================

class IndirectReciprocityEnv(BaseTR4Env):
    """
    Indirect Reciprocity Environment (IndirectReciprocity-v0)

    Four-agent population with reputation-mediated cooperation. Cooperation
    with any partner is observed by all members, enabling indirect reciprocity:
    "I cooperate with you because you cooperated with someone else."

    Game-Theoretic Foundation:
    --------------------------
    - Nowak & Sigmund (1998) "Evolution of Indirect Reciprocity by Image Scoring"
    - Nowak & Sigmund (2005) "Evolution of Indirect Reciprocity"
    - Panchanathan & Boyd (2004) "Indirect Reciprocity Can Stabilize Cooperation"

    Challenge:
    ----------
    Agents must learn that their cooperation with any single partner is
    observed by all others. Defecting with one partner damages reputation
    globally, reducing cooperation from all partners. The longer memory
    window (k=7) means negative signals persist.

    Distinction from PlatformEcosystem-v0:
    --------------------------------------
    PlatformEcosystem models market competition. IndirectReciprocity models
    reputation-mediated cooperation. Indirect reciprocity emerges from
    Eq 44's multi-agent summation: agent i's signal toward j incorporates
    j's actions visible to all.

    Example:
    --------
    >>> env = IndirectReciprocityEnv()
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # All cooperate — builds mutual reputation
    >>> for _ in range(20):
    ...     obs, rewards, done, truncated, info = env.step([60.0]*4)
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "IndirectReciprocity-v0",
        "source": "TR-4, Nowak & Sigmund (1998, 2005)"
    }

    def __init__(
        self,
        max_steps: int = 150,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        tr4_params = TR4Parameters(
            rho_0=0.8,
            eta=1.0,
            kappa=1.0,
            k=7,
            lambda_R=1.5,
            omega=0.5
        )

        D = create_symmetric_interdependence(4, 0.4).matrix

        super().__init__(
            n_agents=4,
            tr4_params=tr4_params,
            max_steps=max_steps,
            render_mode=render_mode,
            endowments=np.full(4, 100.0, dtype=np.float32),
            interdependence_matrix=D,
            baselines=np.full(4, 30.0, dtype=np.float32),
            **kwargs
        )


# =============================================================================
# ENVIRONMENT 4: GraduatedSanction-v0
# =============================================================================

class GraduatedSanctionEnv(BaseTR4Env):
    """
    Graduated Sanction Environment (GraduatedSanction-v0)

    Six-agent common-pool resource with graduated reciprocity sanctions.
    Agents decide how much to contribute to a shared resource. Reciprocity
    manifests as graduated sanctions: mild response to first defection,
    escalating with repeated violations.

    Game-Theoretic Foundation:
    --------------------------
    - Ostrom (1990) "Governing the Commons"
    - Ostrom, Walker & Gardner (1992) "Covenants With and Without a Sword"
    - Fehr & Gächter (2000) "Cooperation and Punishment in Public Goods"

    Challenge:
    ----------
    With 6 agents, each receives signals from 5 partners. The graduated
    nature emerges from lower κ=0.8 (gradual tanh response) and long k=10
    memory. Free-riding is tempting but triggers graduated retaliation from
    5 partners simultaneously.

    Distinction from PublicGoods-v0 (TR-3):
    ----------------------------------------
    PublicGoods uses static TR-3 collective action modifiers (free-rider
    penalties, loyalty bonuses). GraduatedSanction uses adaptive TR-4
    history-dependent reciprocity. TR-3 applies rule-based modifiers;
    TR-4 applies adaptive memory-based responses.

    Example:
    --------
    >>> env = GraduatedSanctionEnv()
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # All contribute above baseline
    >>> for _ in range(20):
    ...     obs, rewards, done, truncated, info = env.step([60.0]*6)
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "GraduatedSanction-v0",
        "source": "TR-4, Ostrom (1990), Fehr & Gächter (2000)"
    }

    def __init__(
        self,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        tr4_params = TR4Parameters(
            rho_0=0.6,
            eta=1.5,
            kappa=0.8,
            k=10,
            lambda_R=1.8,
            omega=1.0
        )

        D = create_symmetric_interdependence(6, 0.35).matrix

        super().__init__(
            n_agents=6,
            tr4_params=tr4_params,
            max_steps=max_steps,
            render_mode=render_mode,
            endowments=np.full(6, 100.0, dtype=np.float32),
            interdependence_matrix=D,
            baselines=np.full(6, 40.0, dtype=np.float32),
            **kwargs
        )


# =============================================================================
# ENVIRONMENT 5: AppleAppStore-v0
# =============================================================================

class AppleAppStoreEnv(BaseTR4Env):
    """
    Apple App Store Environment (AppleAppStore-v0)

    Validated case study from TR-4 Section 8: Apple iOS App Store ecosystem
    (2008-2024). Three agents with asymmetric dependencies model platform
    power dynamics across 66 quarters in five phases.

    Actors:
    -------
    - Agent 0: Apple (platform provider)
    - Agent 1: Major Developers (Epic, Spotify, Netflix)
    - Agent 2: Small Developers (aggregated)

    Game-Theoretic Foundation:
    --------------------------
    - TR-4 Section 8: Empirical Validation (48/55 = 87.3%)
    - Parker, Van Alstyne & Choudary (2016) "Platform Revolution"
    - Rochet & Tirole (2003) "Platform Competition in Two-Sided Markets"

    Challenge:
    ----------
    Asymmetric dependencies create asymmetric reciprocity. Developers respond
    strongly to Apple's policy changes (ρ_21=0.770, ρ_31=0.820) while Apple
    is relatively insensitive to developer actions (ρ_12=0.249, ρ_13=0.157).
    The 66-step episode maps to 66 historical quarters.

    Historical Phases:
    ------------------
    1. Symbiosis (Q1-Q16): High mutual cooperation
    2. Maturation (Q17-Q36): Stable cooperation
    3. Tension (Q37-Q48): Declining reciprocity
    4. Crisis (Q49-Q54): Reciprocal defection
    5. Adjustment (Q55-Q66): Partial restoration

    Distinction from SLCD-v0 / RenaultNissan-v0:
    ---------------------------------------------
    SLCD models TR-1 value creation, RenaultNissan models TR-2 trust dynamics.
    AppleAppStore models TR-4 reciprocity — cooperation conditional on observed
    partner behavior over a memory window.

    Example:
    --------
    >>> env = AppleAppStoreEnv()
    >>> obs, info = env.reset(seed=42)
    >>>
    >>> # Simulate symbiosis phase — all cooperate
    >>> for _ in range(16):
    ...     obs, rewards, done, truncated, info = env.step([70.0, 55.0, 40.0])
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "AppleAppStore-v0",
        "validation_score": "48/55 (87.3%)",
        "source": "TR-4 Section 8"
    }

    def __init__(
        self,
        max_steps: int = 66,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        tr4_params = TR4Parameters(
            rho_0=1.0,
            eta=1.2,
            kappa=0.8,
            k=4,
            lambda_R=1.0,
            omega=0.8
        )

        # Asymmetric dependency matrix from TR-4 Section 8
        # Apple (0), Major Devs (1), Small Devs (2)
        D = np.array([
            [0.0,  0.3,  0.2 ],   # Apple depends on devs moderately
            [0.8,  0.0,  0.15],   # Major devs highly dependent on Apple
            [0.85, 0.1,  0.0 ]    # Small devs very highly dependent on Apple
        ], dtype=np.float32)

        # Trust parameters tuned for platform dynamics
        trust_params = TrustParameters(
            lambda_plus=0.10,
            lambda_minus=0.30,
            mu_R=0.60,
            delta_R=0.03,
            xi=0.50,
            kappa=1.0,
            initial_trust=0.60  # Start with moderate-high trust (symbiosis)
        )

        super().__init__(
            n_agents=3,
            tr4_params=tr4_params,
            max_steps=max_steps,
            render_mode=render_mode,
            trust_params=trust_params,
            endowments=np.array([100.0, 80.0, 60.0], dtype=np.float32),
            interdependence_matrix=D,
            baselines=np.array([30.0, 24.0, 18.0], dtype=np.float32),
            **kwargs
        )
