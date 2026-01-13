# Equilibrium Module

`coopetition_gym.core.equilibrium`

Payoff computation, equilibrium solving, and reward calculation.
Implements TR-1 Equations 10-13.

*Module version: 0.2.0*

---

## Overview

The equilibrium module provides:
- **Payoff computation**: Private payoffs and integrated utilities
- **Equilibrium solving**: Nash, Stackelberg, and Coopetitive equilibria
- **Reward interface**: RL-compatible reward computation

---

## Quick Start

```python
from coopetition_gym.core.equilibrium import (
    PayoffParameters,
    compute_complete_payoffs,
    solve_equilibrium,
    compute_rewards,
    create_slcd_payoff_params,
)
import numpy as np

# Setup
params = create_slcd_payoff_params()
actions = np.array([60.0, 55.0])

# Compute payoffs
result = compute_complete_payoffs(actions, params)
print(f"Private payoffs: {result.private_payoffs}")
print(f"Integrated utilities: {result.integrated_utilities}")
print(f"Total value: {result.total_value:.2f}")

# Solve equilibrium
eq = solve_equilibrium(params, method="coopetitive")
print(f"Equilibrium actions: {eq.actions}")
print(f"Total welfare: {eq.total_welfare:.2f}")
```

---

## Classes

### PayoffParameters

```python
@dataclass
class PayoffParameters:
    """Complete payoff configuration for N agents."""

    value_params: ValueFunctionParameters
    endowments: NDArray[np.floating]
    alpha: NDArray[np.floating]          # Bargaining shares
    interdependence: InterdependenceMatrix
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `value_params` | `ValueFunctionParameters` | Value function configuration |
| `endowments` | `NDArray` | Agent endowments [e_1, ..., e_N] |
| `alpha` | `NDArray` | Bargaining shares [α_1, ..., α_N], sum = 1 |
| `interdependence` | `InterdependenceMatrix` | Structural dependencies |

---

### PayoffResult

```python
@dataclass
class PayoffResult:
    """Result of payoff computation."""

    private_payoffs: NDArray[np.floating]
    integrated_utilities: NDArray[np.floating]
    total_value: float
    synergistic_surplus: float
    individual_values: NDArray[np.floating]
```

---

### EquilibriumResult

```python
@dataclass
class EquilibriumResult:
    """Equilibrium solution results."""

    actions: NDArray[np.floating]
    payoffs: PayoffResult
    cooperation_rate: float
    total_welfare: float
```

---

## Functions

### compute_private_payoff

```python
def compute_private_payoff(
    action_i: float,
    actions: NDArray[np.floating],
    params: PayoffParameters,
    agent_idx: int
) -> float:
    """
    Compute private payoff for agent i.

    π_i = (e_i - a_i) + f(a_i) + α_i · g(a)
    """
```

---

### compute_integrated_utility

```python
def compute_integrated_utility(
    agent_idx: int,
    actions: NDArray[np.floating],
    params: PayoffParameters
) -> float:
    """
    Compute integrated utility accounting for interdependence.

    U_i = π_i + Σ_j D[i,j] · π_j
    """
```

---

### compute_rewards

```python
def compute_rewards(
    actions: NDArray[np.floating],
    params: PayoffParameters,
    trust_state: Optional[TrustState] = None,
    reward_type: str = "integrated"
) -> NDArray[np.floating]:
    """
    Compute rewards for RL training.

    Args:
        actions: Agent actions
        params: Payoff parameters
        trust_state: Optional trust state for modulation
        reward_type: 'private', 'integrated', or 'cooperative'

    Returns:
        Reward array [r_1, ..., r_N]
    """
```

---

### solve_equilibrium

```python
def solve_equilibrium(
    params: PayoffParameters,
    method: str = "coopetitive"
) -> EquilibriumResult:
    """
    Solve for equilibrium action profile.

    Args:
        params: Payoff parameters
        method: 'nash', 'stackelberg', or 'coopetitive'

    Returns:
        EquilibriumResult with optimal actions and payoffs
    """
```

---

## Factory Functions

### create_slcd_payoff_params

```python
def create_slcd_payoff_params() -> PayoffParameters:
    """Create payoff params for S-LCD case study."""
```

### create_symmetric_payoff_params

```python
def create_symmetric_payoff_params(
    n_agents: int,
    endowment: float = 100.0,
    interdependence: float = 0.50,
    gamma: float = 0.65
) -> PayoffParameters:
    """Create symmetric payoff params for N homogeneous agents."""
```

---

## Mathematical Background

### Private Payoff (TR-1 Eq. 10)

$$\pi_i(\mathbf{a}) = (e_i - a_i) + f_i(a_i) + \alpha_i \cdot g(\mathbf{a})$$

| Component | Meaning |
|-----------|---------|
| $e_i - a_i$ | Resources retained |
| $f_i(a_i)$ | Individual value created |
| $\alpha_i \cdot g(\mathbf{a})$ | Share of synergistic surplus |

### Integrated Utility (TR-1 Eq. 13)

$$U_i(\mathbf{a}) = \pi_i(\mathbf{a}) + \sum_{j \neq i} D_{ij} \cdot \pi_j(\mathbf{a})$$

Agents internalize partner payoffs proportional to structural dependency.

---

## See Also

- [Value Functions](value_functions.md) - Value computation
- [Interdependence](interdependence.md) - Dependency matrices
- [Trust Dynamics](trust_dynamics.md) - Trust-modulated payoffs
