"""
================================================================================
COOPETITION-GYM: Core Mathematical Module
================================================================================

This module provides the mathematical foundations for coopetition environments,
implementing the computational frameworks from:

- TR-1 (arXiv:2510.18802): Interdependence & Complementarity
- TR-2 (arXiv:2510.24909): Trust Dynamics
- TR-3: Collective Action & Loyalty (skeleton for v0.2.x)
- TR-4: Reciprocity & Conditionality (skeleton for v0.3.x)

Submodules:
-----------
- value_functions: Individual and synergistic value calculations
- interdependence: Structural dependency analysis and matrices
- trust_dynamics: Trust evolution and reputation tracking
- equilibrium: Payoff computation and equilibrium solving
- collective_action: TR-3 collective action and loyalty (skeleton)
- reciprocity: TR-4 reciprocity and conditionality (skeleton)

Example Usage:
--------------
>>> from coopetition_gym.core import (
...     ValueFunctionParameters,
...     TrustParameters,
...     TrustDynamicsModel,
...     create_slcd_interdependence,
...     compute_rewards,
...     CollectiveActionModel,  # TR-3
...     ReciprocityModel,       # TR-4
... )

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

# Value Functions (TR-1)
from .value_functions import (
    # Enums and Config
    ValueSpecification,
    ValueFunctionParameters,
    # Core Functions
    power_value,
    logarithmic_value,
    individual_value,
    synergy_function,
    total_value,
    synergistic_surplus,
    marginal_value,
    marginal_synergy,
    # Factory Functions
    create_slcd_parameters,
    create_power_parameters,
    # Batch Operations
    batch_total_value,
)

# Interdependence (TR-1)
from .interdependence import (
    # Data Classes
    DependencyType,
    Dependency,
    InterdependenceMatrix,
    # Core Functions
    compute_interdependence_coefficient,
    build_interdependence_matrix,
    # Predefined Matrices
    create_slcd_interdependence,
    create_renault_nissan_interdependence,
    create_symmetric_interdependence,
    create_asymmetric_interdependence,
    create_hub_spoke_interdependence,
    # Analysis
    compute_power_index,
    compute_vulnerability_index,
)

# Trust Dynamics (TR-2)
from .trust_dynamics import (
    # Data Classes
    TrustParameters,
    TrustState,
    # Core Model
    TrustDynamicsModel,
    # Specialized Models
    TrustDilemmaModel,
    RecoveryModel,
    AutomotiveAllianceModel,
    # Analysis Functions
    analyze_negativity_bias,
    estimate_recovery_periods,
    estimate_erosion_periods,
    compute_trust_equilibrium,
)

# Equilibrium and Payoffs
from .equilibrium import (
    # Data Classes
    PayoffParameters,
    PayoffResult,
    EquilibriumResult,
    # Payoff Computation
    compute_private_payoff,
    compute_all_private_payoffs,
    compute_integrated_utility,
    compute_all_integrated_utilities,
    compute_complete_payoffs,
    # Equilibrium Solving
    solve_best_response,
    solve_equilibrium,
    compare_equilibria,
    # RL Interface
    compute_rewards,
    compute_social_welfare,
    compute_cooperation_surplus,
    # Factory Functions
    create_slcd_payoff_params,
    create_symmetric_payoff_params,
)

# Collective Action - TR-3 (skeleton)
from .collective_action import (
    CollectiveActionParameters,
    CollectiveActionState,
    CollectiveActionModel,
)

# Reciprocity - TR-4 (skeleton)
from .reciprocity import (
    ReciprocityParameters,
    ReciprocityState,
    ReciprocityModel,
)

__all__ = [
    # Value Functions
    "ValueSpecification",
    "ValueFunctionParameters",
    "power_value",
    "logarithmic_value",
    "individual_value",
    "synergy_function",
    "total_value",
    "synergistic_surplus",
    "marginal_value",
    "marginal_synergy",
    "create_slcd_parameters",
    "create_power_parameters",
    "batch_total_value",
    # Interdependence
    "DependencyType",
    "Dependency",
    "InterdependenceMatrix",
    "compute_interdependence_coefficient",
    "build_interdependence_matrix",
    "create_slcd_interdependence",
    "create_renault_nissan_interdependence",
    "create_symmetric_interdependence",
    "create_asymmetric_interdependence",
    "create_hub_spoke_interdependence",
    "compute_power_index",
    "compute_vulnerability_index",
    # Trust Dynamics
    "TrustParameters",
    "TrustState",
    "TrustDynamicsModel",
    "TrustDilemmaModel",
    "RecoveryModel",
    "AutomotiveAllianceModel",
    "analyze_negativity_bias",
    "estimate_recovery_periods",
    "estimate_erosion_periods",
    "compute_trust_equilibrium",
    # Equilibrium
    "PayoffParameters",
    "PayoffResult",
    "EquilibriumResult",
    "compute_private_payoff",
    "compute_all_private_payoffs",
    "compute_integrated_utility",
    "compute_all_integrated_utilities",
    "compute_complete_payoffs",
    "solve_best_response",
    "solve_equilibrium",
    "compare_equilibria",
    "compute_rewards",
    "compute_social_welfare",
    "compute_cooperation_surplus",
    "create_slcd_payoff_params",
    "create_symmetric_payoff_params",
    # Collective Action - TR-3
    "CollectiveActionParameters",
    "CollectiveActionState",
    "CollectiveActionModel",
    # Reciprocity - TR-4
    "ReciprocityParameters",
    "ReciprocityState",
    "ReciprocityModel",
]
