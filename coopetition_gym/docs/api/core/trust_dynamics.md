# Trust Dynamics Module

`coopetition_gym.core.trust_dynamics`

Mathematical models for trust evolution with negativity bias and reputation hysteresis.
Implements TR-2 Equations 6-9.

*Module version: 0.2.0*

---

## Overview

The trust dynamics module implements a **two-layer architecture** for modeling relationship evolution:

| Layer | Symbol | Update Frequency | What It Captures |
|-------|--------|------------------|------------------|
| **Immediate Trust** | $T_{{ij}} \in [0,1]$ | Every interaction | Current confidence |
| **Reputation Damage** | $R_{{ij}} \in [0,1]$ | On violations | Historical memory |

**Key Properties:**

- **Negativity Bias**: Trust erodes ~3× faster than it builds ($\lambda^-/\lambda^+ \approx 3.0$)
- **Trust Ceiling**: $\Theta = 1 - R$ (reputation damage limits recovery)
- **Interdependence Amplification**: High-dependency relationships experience faster erosion

---

## Quick Start

```python
from coopetition_gym.core.trust_dynamics import (
    TrustParameters,
    TrustState,
    TrustDynamicsModel,
)
import numpy as np

# Initialize
params = TrustParameters()
state = TrustState.create_initial(n_agents=2, params=params)
model = TrustDynamicsModel(params)

# Cooperation signal (above baseline = positive)
actions = np.array([60.0, 55.0])
baselines = np.array([35.0, 35.0])
D = np.array([[0.0, 0.5], [0.5, 0.0]])  # Symmetric interdependence

# Update trust
new_state = model.update(state, actions, baselines, D)
print(f"Trust: {{state.trust_matrix[0,1]:.3f}} → {{new_state.trust_matrix[0,1]:.3f}}")
```

---

## Classes

### TrustParameters

```python
@dataclass
class TrustParameters:
    """
    Configuration parameters for trust dynamics.

    Defaults are set to validated values from TR-2 §8 (Renault-Nissan case).
    """

    lambda_plus: float = 0.10      # Trust building rate
    lambda_minus: float = 0.30    # Trust erosion rate
    mu_R: float = 0.60            # Reputation damage severity
    delta_R: float = 0.03         # Reputation decay rate
    xi: float = 0.50              # Interdependence amplification
    kappa: float = 1.0            # Signal sensitivity
    initial_trust: float = 0.50
    initial_reputation: float = 0.0
```

**Attributes:**

| Name | Symbol | Default | Range | Description |
|------|--------|---------|-------|-------------|
| `lambda_plus` | $\lambda^+$ | `0.10` | (0, 1) | Rate of trust increase |
| `lambda_minus` | $\lambda^-$ | `0.30` | (0, 1) | Rate of trust decrease |
| `mu_R` | $\mu_R$ | `0.60` | (0, 1) | Reputation damage severity |
| `delta_R` | $\delta_R$ | `0.03` | (0, 0.1) | Reputation decay rate |
| `xi` | $\xi$ | `0.50` | (0, 1) | Interdependence amplification |
| `kappa` | $\kappa$ | `1.0` | (0, 2) | Signal sensitivity |
| `initial_trust` | $\tau_0$ | `0.50` | [0, 1] | Starting trust level |
| `initial_reputation` | $R_0$ | `0.0` | [0, 1] | Starting reputation damage |

**Properties:**

```python
@property
def negativity_ratio(self) -> float:
    """Return λ⁻/λ⁺ ratio (typically ~3.0)."""
    return self.lambda_minus / self.lambda_plus
```

**Methods:**

```python
def with_updates(self, **kwargs) -> "TrustParameters":
    """Create new instance with updated values."""
```

**Example:**

```python
from coopetition_gym.core.trust_dynamics import TrustParameters

# Default (validated)
params = TrustParameters()
print(f"Negativity ratio: {{params.negativity_ratio}}")  # 3.0

# Custom for crisis scenario
crisis_params = params.with_updates(
    lambda_minus=0.45,  # Faster erosion
    mu_R=0.70           # More severe damage
)
```

---

### TrustState

```python
@dataclass
class TrustState:
    """
    Complete trust state between all agents.

    Maintains both immediate trust and reputation damage matrices.
    """

    trust_matrix: NDArray[np.floating]       # T[i,j] = i's trust in j
    reputation_matrix: NDArray[np.floating]  # R[i,j] = i's damage record of j
    n_agents: int
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `trust_matrix` | `NDArray` | Shape (n, n), T[i,j] = agent i's trust in agent j |
| `reputation_matrix` | `NDArray` | Shape (n, n), R[i,j] = damage i has recorded for j |
| `n_agents` | `int` | Number of agents |

**Class Methods:**

```python
@classmethod
def create_initial(
    cls,
    n_agents: int,
    params: TrustParameters
) -> "TrustState":
    """Create initial state with uniform trust/reputation."""
```

**Properties:**

```python
@property
def trust_ceiling(self) -> NDArray[np.floating]:
    """Return Θ = 1 - R (maximum achievable trust)."""
    return 1.0 - self.reputation_matrix

@property
def mean_trust(self) -> float:
    """Return mean off-diagonal trust."""

@property
def mean_reputation_damage(self) -> float:
    """Return mean off-diagonal reputation damage."""
```

**Example:**

```python
from coopetition_gym.core.trust_dynamics import TrustState, TrustParameters

params = TrustParameters(initial_trust=0.55)
state = TrustState.create_initial(n_agents=2, params=params)

print(f"Trust matrix:\n{{state.trust_matrix}}")
print(f"Mean trust: {{state.mean_trust:.3f}}")
print(f"Trust ceiling: {{state.trust_ceiling}}")
```

---

### TrustDynamicsModel

```python
class TrustDynamicsModel:
    """
    Core trust evolution model implementing TR-2 dynamics.

    Handles:
    - Cooperation signal computation
    - Asymmetric trust updating (building vs erosion)
    - Reputation damage accumulation
    - Trust ceiling enforcement
    - Interdependence amplification
    """

    def __init__(self, params: TrustParameters):
        self.params = params
```

**Methods:**

```python
def update(
    self,
    state: TrustState,
    actions: NDArray[np.floating],
    baselines: NDArray[np.floating],
    D: NDArray[np.floating]
) -> TrustState:
    """
    Perform one trust update step.

    Args:
        state: Current trust state
        actions: Agent cooperation levels
        baselines: Expected cooperation baselines
        D: Interdependence matrix

    Returns:
        New TrustState with updated trust and reputation
    """

def compute_cooperation_signal(
    self,
    actions: NDArray[np.floating],
    baselines: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute cooperation signals for all agent pairs.

    Returns:
        Signal matrix S[i,j] = tanh(κ · (a_j - b_j))
        Positive if j cooperated above baseline, negative if below
    """
```

**Example:**

```python
from coopetition_gym.core.trust_dynamics import (
    TrustParameters,
    TrustState,
    TrustDynamicsModel
)
import numpy as np

# Setup
params = TrustParameters()
state = TrustState.create_initial(n_agents=2, params=params)
model = TrustDynamicsModel(params)

# Define scenario
actions = np.array([60.0, 30.0])  # Agent 0 cooperates, Agent 1 defects
baselines = np.array([35.0, 35.0])
D = np.array([[0.0, 0.6], [0.6, 0.0]])

# Compute signals
signals = model.compute_cooperation_signal(actions, baselines)
print(f"Cooperation signals:\n{{signals}}")
# Agent 0 sent positive signal, Agent 1 sent negative signal

# Update trust
for step in range(10):
    state = model.update(state, actions, baselines, D)

print(f"Final trust: {{state.trust_matrix}}")
# Agent 0's trust in Agent 1 decreased (defection)
# Agent 1's trust in Agent 0 increased (cooperation)
```

---

## Specialized Models

### TrustDilemmaModel

```python
class TrustDilemmaModel(TrustDynamicsModel):
    """
    Trust dynamics optimized for TrustDilemma-v0.

    Uses default parameters with moderate negativity bias.
    """
```

### RecoveryModel

```python
class RecoveryModel(TrustDynamicsModel):
    """
    Trust dynamics for RecoveryRace-v0.

    Parameters optimized for post-crisis recovery:
    - λ⁺ = 0.08 (slower building)
    - λ⁻ = 0.35 (faster re-erosion)
    - δ_R = 0.01 (very slow reputation decay)
    """
```

### AutomotiveAllianceModel

```python
class AutomotiveAllianceModel(TrustDynamicsModel):
    """
    Trust dynamics validated for RenaultNissan-v0.

    Parameters calibrated to historical alliance phases.
    """
```

---

## Analysis Functions

### analyze_negativity_bias

```python
def analyze_negativity_bias(params: TrustParameters) -> Dict:
    """
    Analyze the negativity bias in trust dynamics.

    Returns:
        Dictionary with:
        - 'ratio': λ⁻/λ⁺
        - 'build_time_estimate': Steps to build from 0.5 to 0.8
        - 'erosion_time_estimate': Steps to erode from 0.8 to 0.5
        - 'asymmetry_factor': How much faster erosion is
    """
```

**Example:**

```python
from coopetition_gym.core.trust_dynamics import (
    TrustParameters,
    analyze_negativity_bias
)

params = TrustParameters()
analysis = analyze_negativity_bias(params)

print(f"Negativity ratio: {{analysis['ratio']:.2f}}")
print(f"Build time: ~{{analysis['build_time_estimate']}} steps")
print(f"Erosion time: ~{{analysis['erosion_time_estimate']}} steps")
```

---

### estimate_recovery_periods

```python
def estimate_recovery_periods(params: TrustParameters) -> Dict:
    """
    Estimate time to recover trust after violations.

    Returns:
        Dictionary with recovery times for various damage levels.
    """
```

---

### compute_trust_equilibrium

```python
def compute_trust_equilibrium(
    D: NDArray[np.floating],
    params: TrustParameters
) -> Dict:
    """
    Compute steady-state trust given consistent behavior.

    Args:
        D: Interdependence matrix
        params: Trust parameters

    Returns:
        Dictionary with equilibrium trust levels for cooperation/defection
    """
```

---

## Mathematical Background

### Trust Update Equations (TR-2 §6)

**Building (positive signal):**

$$\Delta T_{{ij}} = \lambda^+ \cdot s_{{ij}} \cdot (\Theta_{{ij}} - T_{{ij}})$$

**Erosion (negative signal):**

$$\Delta T_{{ij}} = \lambda^- \cdot |s_{{ij}}| \cdot T_{{ij}} \cdot (1 + \xi \cdot D_{{ij}})$$

where:
- $s_{{ij}} = \tanh(\kappa \cdot (a_j - b_j))$ is the cooperation signal
- $\Theta_{{ij}} = 1 - R_{{ij}}$ is the trust ceiling
- $\xi \cdot D_{{ij}}$ amplifies erosion for high-dependency relationships

### Negativity Bias

The 3:1 ratio ($\lambda^-/\lambda^+ \approx 3.0$) is grounded in behavioral economics research showing that negative events have disproportionate impact on trust judgments.

**Implications:**
- A single major violation can destroy months of trust-building
- Consistent cooperation is essential for sustainable partnerships
- Recovery from betrayal requires sustained effort

### Trust Ceiling (Hysteresis)

Reputation damage $R_{{ij}}$ creates a ceiling on achievable trust:

$$T_{{ij}} \leq \Theta_{{ij}} = 1 - R_{{ij}}$$

Even with perfect cooperation, damaged reputation prevents full trust recovery. This captures the intuition that "you never fully trust someone who betrayed you."

---

## See Also

- [Value Functions](value_functions.md) - Value creation
- [Equilibrium](equilibrium.md) - Payoff computation with trust
- [TR-2 Theory](../../theory/trust_dynamics.md) - Mathematical foundations
- [RecoveryRace-v0](../../environments/recovery_race.md) - Trust recovery benchmark
