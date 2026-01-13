# Value Functions Module

`coopetition_gym.core.value_functions`

Mathematical functions for computing individual value contributions and synergistic surplus.
Implements TR-1 Equations 5-8.

*Module version: 0.2.0*

---

## Overview

The value functions module provides two validated specifications for computing value creation in coopetitive settings:

| Specification | Formula | Best For | Validation Score |
|--------------|---------|----------|------------------|
| **Logarithmic** (default) | $f(a) = \theta \cdot \ln(1 + a)$ | Manufacturing JVs | 58/60 |
| **Power** | $f(a) = a^\beta$ | General scenarios | 46/60 |

Both specifications exhibit diminishing marginal returns—each additional unit of cooperation yields progressively smaller value gains.

---

## Quick Start

```python
from coopetition_gym.core.value_functions import (
    logarithmic_value,
    power_value,
    synergy_function,
    total_value,
    create_slcd_parameters,
)
import numpy as np

# Individual value
v = logarithmic_value(50.0, theta=20.0)  # ~78.2

# Joint synergy
actions = np.array([60.0, 55.0])
synergy = synergy_function(actions)  # ~57.4

# Total value creation
params = create_slcd_parameters()
total = total_value(actions, params)  # ~193.5
```

---

## Classes

### ValueSpecification

```python
class ValueSpecification(Enum):
    """Enumeration of available value function specifications."""

    POWER = "power"
    LOGARITHMIC = "logarithmic"
```

**Members:**

| Value | Description | TR Reference |
|-------|-------------|--------------|
| `POWER` | Cobb-Douglas specification: $f(a) = a^\beta$ | TR-1 Eq. 5 |
| `LOGARITHMIC` | Logarithmic specification: $f(a) = \theta \cdot \ln(1+a)$ | TR-1 Eq. 6 |

---

### ValueFunctionParameters

```python
@dataclass(frozen=True)
class ValueFunctionParameters:
    """
    Immutable configuration for value function computation.

    All parameters are validated on construction. Invalid values raise ValueError.
    """

    specification: ValueSpecification = ValueSpecification.LOGARITHMIC
    beta: float = 0.75
    theta: float = 20.0
    gamma: float = 0.65
    epsilon: float = 1e-10
```

**Attributes:**

| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `specification` | `ValueSpecification` | `LOGARITHMIC` | - | Which value function to use |
| `beta` | `float` | `0.75` | (0, 1) | Power function exponent |
| `theta` | `float` | `20.0` | > 0 | Logarithmic scale factor |
| `gamma` | `float` | `0.65` | [0, 1] | Complementarity coefficient |
| `epsilon` | `float` | `1e-10` | > 0 | Numerical stability constant |

**Example:**

```python
from coopetition_gym.core.value_functions import (
    ValueFunctionParameters,
    ValueSpecification
)

# Default (validated S-LCD parameters)
params = ValueFunctionParameters()

# Custom parameters
params = ValueFunctionParameters(
    specification=ValueSpecification.POWER,
    beta=0.80,
    gamma=0.70
)

# Access derived properties
print(f"Using {{params.specification.value}} specification")
```

**Validation:**

- `beta` must be in (0, 1) for diminishing returns
- `theta` must be positive
- `gamma` must be in [0, 1]
- `epsilon` must be positive

---

## Functions

### power_value

```python
def power_value(
    action: Union[float, NDArray[np.floating]],
    beta: float = 0.75,
    epsilon: float = 1e-10
) -> Union[float, NDArray[np.floating]]:
    """
    Compute individual value using the power (Cobb-Douglas) specification.

    Implements TR-1 Equation 5: f_i(a_i) = a_i^β
    """
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `action` | `float \| NDArray` | *required* | Cooperation level(s), must be ≥ 0 |
| `beta` | `float` | `0.75` | Elasticity parameter ∈ (0, 1) |
| `epsilon` | `float` | `1e-10` | Stability constant for action ≈ 0 |

**Returns:**

| Type | Description |
|------|-------------|
| `float \| NDArray` | Individual value, same shape as input |

**Mathematical Properties:**

- $f(0) = 0$ (no cooperation yields no value)
- $f'(a) = \beta \cdot a^{{\beta-1}} > 0$ (monotonically increasing)
- $f''(a) = \beta(\beta-1) \cdot a^{{\beta-2}} < 0$ for $\beta < 1$ (concave)

**Example:**

```python
from coopetition_gym.core.value_functions import power_value
import numpy as np

# Scalar input
v = power_value(50.0, beta=0.75)
print(f"Value: {{v:.2f}}")  # ~18.80

# Vector input (batch processing)
actions = np.array([25.0, 50.0, 75.0, 100.0])
values = power_value(actions, beta=0.75)
print(f"Values: {{values}}")  # [11.18, 18.80, 25.09, 31.62]

# Demonstrate diminishing returns
increments = np.diff(values)
print(f"Marginal values: {{increments}}")  # Decreasing
```

---

### logarithmic_value

```python
def logarithmic_value(
    action: Union[float, NDArray[np.floating]],
    theta: float = 20.0,
    epsilon: float = 1e-10
) -> Union[float, NDArray[np.floating]]:
    """
    Compute individual value using the logarithmic specification.

    Implements TR-1 Equation 6: f_i(a_i) = θ · ln(1 + a_i)

    This is the validated default for manufacturing joint ventures,
    achieving 58/60 accuracy on the Samsung-Sony S-LCD case study.
    """
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `action` | `float \| NDArray` | *required* | Cooperation level(s) |
| `theta` | `float` | `20.0` | Scale factor (validated) |
| `epsilon` | `float` | `1e-10` | Numerical stability |

**Returns:**

| Type | Description |
|------|-------------|
| `float \| NDArray` | Individual value |

**Example:**

```python
from coopetition_gym.core.value_functions import logarithmic_value
import numpy as np

# Single value
v = logarithmic_value(50.0, theta=20.0)
print(f"Value: {{v:.2f}}")  # ~78.24

# Compare specifications
from coopetition_gym.core.value_functions import power_value

action = 50.0
log_v = logarithmic_value(action, theta=20.0)
pow_v = power_value(action, beta=0.75)
print(f"Logarithmic: {{log_v:.2f}}, Power: {{pow_v:.2f}}")
# Logarithmic produces higher values at moderate cooperation
```

---

### individual_value

```python
def individual_value(
    action: Union[float, NDArray[np.floating]],
    params: ValueFunctionParameters
) -> Union[float, NDArray[np.floating]]:
    """
    Compute individual value using configured specification.

    Dispatches to power_value() or logarithmic_value() based on
    params.specification.
    """
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `action` | `float \| NDArray` | Cooperation level(s) |
| `params` | `ValueFunctionParameters` | Configuration object |

**Example:**

```python
from coopetition_gym.core.value_functions import (
    individual_value,
    ValueFunctionParameters,
    ValueSpecification
)

# Using logarithmic (default)
params_log = ValueFunctionParameters()
v_log = individual_value(50.0, params_log)

# Using power
params_pow = ValueFunctionParameters(specification=ValueSpecification.POWER)
v_pow = individual_value(50.0, params_pow)
```

---

### synergy_function

```python
def synergy_function(
    actions: NDArray[np.floating]
) -> float:
    """
    Compute synergistic value from joint cooperation.

    Implements TR-1 Equation 7: g(a) = (∏_{i=1}^N a_i)^{1/N}

    The geometric mean captures true complementarity—value requires
    contributions from all parties.
    """
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `actions` | `NDArray` | Array of cooperation levels [a_1, ..., a_N] |

**Returns:**

| Type | Description |
|------|-------------|
| `float` | Synergistic value (geometric mean) |

**Mathematical Properties:**

- $g(\mathbf{{a}}) = 0$ if any $a_i = 0$ (requires all to contribute)
- Symmetric: order of agents doesn't matter
- Bounded: $g(\mathbf{{a}}) \leq \max_i a_i$

**Example:**

```python
from coopetition_gym.core.value_functions import synergy_function
import numpy as np

# Balanced cooperation
actions = np.array([50.0, 50.0])
s = synergy_function(actions)
print(f"Synergy: {{s:.2f}}")  # 50.0

# Imbalanced cooperation (synergy suffers)
actions = np.array([90.0, 10.0])
s = synergy_function(actions)
print(f"Synergy: {{s:.2f}}")  # 30.0 (geometric mean < arithmetic mean)

# One defector kills synergy
actions = np.array([100.0, 0.0])
s = synergy_function(actions)
print(f"Synergy: {{s:.2f}}")  # 0.0
```

---

### total_value

```python
def total_value(
    actions: NDArray[np.floating],
    params: ValueFunctionParameters
) -> float:
    """
    Compute total value creation from all contributions.

    Implements TR-1 Equation 8:
    V(a|γ) = Σ_{i=1}^N f_i(a_i) + γ · g(a_1, ..., a_N)
    """
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `actions` | `NDArray` | Array of cooperation levels |
| `params` | `ValueFunctionParameters` | Value function configuration |

**Returns:**

| Type | Description |
|------|-------------|
| `float` | Total value created |

**Example:**

```python
from coopetition_gym.core.value_functions import (
    total_value,
    create_slcd_parameters
)
import numpy as np

params = create_slcd_parameters()
actions = np.array([60.0, 55.0])

# Total value with complementarity
v = total_value(actions, params)
print(f"Total value: {{v:.2f}}")

# Decomposition
from coopetition_gym.core.value_functions import (
    logarithmic_value,
    synergy_function
)

individual = sum(logarithmic_value(a) for a in actions)
synergy = params.gamma * synergy_function(actions)
print(f"Individual: {{individual:.2f}}, Synergy: {{synergy:.2f}}")
print(f"Sum: {{individual + synergy:.2f}}")  # Matches total
```

---

### batch_total_value

```python
def batch_total_value(
    actions_batch: NDArray,
    params: ValueFunctionParameters
) -> NDArray:
    """
    Compute total value for a batch of action profiles.

    Efficient vectorized computation for multiple scenarios.
    """
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `actions_batch` | `NDArray` | Shape (batch_size, n_agents) |
| `params` | `ValueFunctionParameters` | Configuration |

**Returns:**

| Type | Description |
|------|-------------|
| `NDArray` | Shape (batch_size,) of total values |

**Example:**

```python
from coopetition_gym.core.value_functions import (
    batch_total_value,
    create_slcd_parameters
)
import numpy as np

params = create_slcd_parameters()

# Multiple scenarios
scenarios = np.array([
    [50.0, 50.0],  # Equal moderate
    [80.0, 80.0],  # Equal high
    [90.0, 10.0],  # Imbalanced
    [0.0, 0.0],    # No cooperation
])

values = batch_total_value(scenarios, params)
print(f"Values: {{values}}")
```

---

## Factory Functions

### create_slcd_parameters

```python
def create_slcd_parameters() -> ValueFunctionParameters:
    """
    Create parameters validated against Samsung-Sony S-LCD case study.

    These parameters achieved 58/60 validation score (96.7% accuracy).

    Returns:
        ValueFunctionParameters with:
        - specification: LOGARITHMIC
        - theta: 20.0
        - gamma: 0.65
    """
```

**Example:**

```python
params = create_slcd_parameters()
print(f"Specification: {{params.specification.value}}")
print(f"Theta: {{params.theta}}")
print(f"Gamma: {{params.gamma}}")
```

---

### create_power_parameters

```python
def create_power_parameters(**kwargs) -> ValueFunctionParameters:
    """
    Create parameters using power specification.

    Keyword arguments override defaults:
    - beta: 0.75
    - gamma: 0.50
    """
```

---

## Mathematical Background

### Value Creation Model (TR-1 §5-6)

The total value $V(\mathbf{{a}} | \gamma)$ created by a coopetitive relationship:

$$V(\mathbf{{a}} | \gamma) = \sum_{{i=1}}^N f_i(a_i) + \gamma \cdot g(a_1, \ldots, a_N)$$

| Component | Formula | Captures |
|-----------|---------|----------|
| Individual value | $\sum f_i(a_i)$ | What each agent creates independently |
| Synergistic surplus | $\gamma \cdot g(\mathbf{{a}})$ | Additional value from complementarity |

### Complementarity Parameter γ

The complementarity coefficient γ ∈ [0, 1] controls the balance:

| γ Value | Interpretation | Example Domain |
|---------|---------------|----------------|
| γ ≈ 0 | Independent value creation | Arms-length transactions |
| γ ≈ 0.5 | Balanced individual/joint | General partnerships |
| γ ≈ 0.65 | Validated for JVs | Manufacturing alliances |
| γ ≈ 1.0 | Highly complementary | R&D collaborations |

### Diminishing Returns

Both specifications exhibit concavity (diminishing marginal value):

```
Value
│
│        ─────────────────  (saturation)
│      ──
│    ──
│  ──
│──
└─────────────────────────── Action
```

This reflects economic reality: initial investments yield high returns, but eventually additional cooperation has diminishing impact.

---

## See Also

- [Interdependence Module](interdependence.md) - Structural coupling
- [Equilibrium Module](equilibrium.md) - Full payoff computation
- [TR-1 Theory](../../theory/value_creation.md) - Mathematical foundations
- [SLCD-v0 Environment](../../environments/slcd.md) - Validated case study
