# Interdependence Module

`coopetition_gym.core.interdependence`

Structural dependency matrices and power analysis.
Implements TR-1 Equations 1-4.

*Module version: 0.2.0*

---

## Overview

The interdependence module captures **structural coupling** between agents—why actors must consider partner outcomes even while competing.

**Key Concept:** When Actor A depends on Actor B for critical resources, A's success structurally requires B's success, creating *instrumental* concern for B's welfare.

$$D_{{ij}} = \frac{{\sum_{{d}} w_d \cdot \text{{Dep}}(i,j,d) \cdot \text{{crit}}(i,j,d)}}{{\sum_d w_d}}$$

---

## Quick Start

```python
from coopetition_gym.core.interdependence import (
    InterdependenceMatrix,
    create_slcd_interdependence,
    compute_power_index,
    compute_vulnerability_index,
)

# Pre-configured for Samsung-Sony case
D = create_slcd_interdependence()
print(f"Sony's dependency on Samsung: {{D.get_dependency('Sony', 'Samsung'):.2f}}")
print(f"Samsung's dependency on Sony: {{D.get_dependency('Samsung', 'Sony'):.2f}}")

# Power analysis
power = compute_power_index(D)
vulnerability = compute_vulnerability_index(D)
print(f"Power indices: {{power}}")
print(f"Vulnerability indices: {{vulnerability}}")
```

---

## Classes

### DependencyType

```python
class DependencyType(Enum):
    """Types of dependency relationships from i* framework."""

    GOAL = "goal"           # Depends on partner for goal achievement
    TASK = "task"           # Depends on partner for task completion
    RESOURCE = "resource"   # Depends on partner for resource provision
    SOFTGOAL = "softgoal"   # Depends on partner for quality attribute
```

---

### Dependency

```python
@dataclass
class Dependency:
    """
    Single dependency relationship between two actors.
    """

    depender: str              # Who depends (agent name)
    dependee: str              # On whom (agent name)
    dependum: str              # For what (goal/task/resource name)
    dep_type: DependencyType = DependencyType.RESOURCE
    importance: float = 1.0    # Weight of this dependency
    criticality: float = 1.0   # 1.0 = sole provider, 0.0 = many alternatives
```

**Attributes:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `depender` | `str` | *required* | Agent who depends |
| `dependee` | `str` | *required* | Agent depended upon |
| `dependum` | `str` | *required* | What the dependency is for |
| `dep_type` | `DependencyType` | `RESOURCE` | Type of dependency |
| `importance` | `float` | `1.0` | Weight ∈ [0, 1] |
| `criticality` | `float` | `1.0` | Criticality ∈ [0, 1] |

**Example:**

```python
from coopetition_gym.core.interdependence import Dependency, DependencyType

# Sony depends on Samsung for panel supply
dep = Dependency(
    depender="Sony",
    dependee="Samsung",
    dependum="LCD_panels",
    dep_type=DependencyType.RESOURCE,
    importance=0.9,
    criticality=0.95  # Few alternative suppliers
)
```

---

### InterdependenceMatrix

```python
@dataclass
class InterdependenceMatrix:
    """
    N×N interdependence matrix capturing structural dependencies.

    D[i,j] represents how much agent i depends on agent j.
    Note: D[i,j] ≠ D[j,i] in general (asymmetric).
    """

    matrix: NDArray[np.floating]
    agent_names: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)
```

**Properties:**

```python
@property
def n_agents(self) -> int:
    """Number of agents."""
```

**Methods:**

```python
def get_dependency(self, i: Union[int, str], j: Union[int, str]) -> float:
    """Get D[i,j] - how much i depends on j."""

def total_dependency(self, agent: Union[int, str]) -> float:
    """Sum of agent's dependencies on all others."""

def total_dependability(self, agent: Union[int, str]) -> float:
    """Sum of how much all others depend on agent."""

def asymmetry(self, i: Union[int, str], j: Union[int, str]) -> float:
    """Compute D[i,j] - D[j,i] (positive if i depends more on j)."""

def to_dict(self) -> Dict:
    """Serialize to dictionary."""

@classmethod
def from_dict(cls, data: Dict) -> "InterdependenceMatrix":
    """Deserialize from dictionary."""
```

**Example:**

```python
from coopetition_gym.core.interdependence import InterdependenceMatrix
import numpy as np

# Create asymmetric interdependence
matrix = np.array([
    [0.0, 0.35],   # Strong depends moderately on Weak
    [0.85, 0.0]    # Weak depends heavily on Strong
])

D = InterdependenceMatrix(
    matrix=matrix,
    agent_names=["Strong", "Weak"]
)

# Analysis
print(f"Strong's total dependency: {{D.total_dependency('Strong'):.2f}}")
print(f"Weak's total dependency: {{D.total_dependency('Weak'):.2f}}")
print(f"Power asymmetry: {{D.asymmetry('Weak', 'Strong'):.2f}}")
# Positive = Weak depends more on Strong than vice versa
```

---

## Factory Functions

### create_slcd_interdependence

```python
def create_slcd_interdependence() -> InterdependenceMatrix:
    """
    Create interdependence matrix for Samsung-Sony S-LCD case.

    Validated parameters from TR-1 §8.2:
    - Sony depends on Samsung: 0.86
    - Samsung depends on Sony: 0.64

    Returns:
        InterdependenceMatrix with agent names ['Samsung', 'Sony']
    """
```

---

### create_renault_nissan_interdependence

```python
def create_renault_nissan_interdependence(
    phase: str = "mature"
) -> InterdependenceMatrix:
    """
    Create interdependence matrix for Renault-Nissan alliance.

    Phase-specific parameters from TR-2:

    Args:
        phase: One of 'formation', 'mature', 'crisis', 'strained'

    Returns:
        InterdependenceMatrix with phase-appropriate values
    """
```

---

### create_symmetric_interdependence

```python
def create_symmetric_interdependence(
    n_agents: int,
    value: float
) -> InterdependenceMatrix:
    """
    Create symmetric N×N interdependence matrix.

    All off-diagonal entries set to `value`.
    Useful for homogeneous environments.
    """
```

---

### create_asymmetric_interdependence

```python
def create_asymmetric_interdependence(
    n_agents: int,
    strong_idx: int = 0,
    strong_dependency: float = 0.35,
    weak_dependency: float = 0.85
) -> InterdependenceMatrix:
    """
    Create asymmetric interdependence with one powerful agent.

    Agent at strong_idx has low dependency on others (powerful),
    others have high dependency on strong agent (vulnerable).
    """
```

---

### create_hub_spoke_interdependence

```python
def create_hub_spoke_interdependence(
    n_agents: int,
    hub_idx: int = 0,
    spoke_to_hub: float = 0.80,
    hub_to_spoke: float = 0.20,
    spoke_to_spoke: float = 0.10
) -> InterdependenceMatrix:
    """
    Create hub-and-spoke structure (e.g., platform ecosystems).

    Hub (typically platform) has moderate dependency on spokes.
    Spokes (typically developers) heavily depend on hub.
    Spokes have minimal direct dependency on each other.
    """
```

---

## Analysis Functions

### compute_power_index

```python
def compute_power_index(D: InterdependenceMatrix) -> NDArray[np.floating]:
    """
    Compute power index for each agent.

    Power = Total dependability / Total dependency
    High power = others depend on you more than you depend on them.

    Returns:
        Array of power indices (> 1 indicates power advantage)
    """
```

**Example:**

```python
from coopetition_gym.core.interdependence import (
    create_slcd_interdependence,
    compute_power_index
)

D = create_slcd_interdependence()
power = compute_power_index(D)

print(f"Samsung power index: {{power[0]:.2f}}")  # > 1 (more powerful)
print(f"Sony power index: {{power[1]:.2f}}")     # < 1 (more dependent)
```

---

### compute_vulnerability_index

```python
def compute_vulnerability_index(D: InterdependenceMatrix) -> NDArray[np.floating]:
    """
    Compute vulnerability index for each agent.

    Vulnerability = Total dependency (sum of D[i,j] for all j≠i)
    High vulnerability = agent depends heavily on partners.

    Returns:
        Array of vulnerability indices
    """
```

---

## Mathematical Background

### Interdependence Equation (TR-1 §3)

$$D_{{ij}} = \frac{{\sum_{{d \in \mathcal{{D}}_i}} w_d \cdot \text{{Dep}}(i,j,d) \cdot \text{{crit}}(i,j,d)}}{{\sum_{{d \in \mathcal{{D}}_i}} w_d}}$$

| Component | Meaning |
|-----------|---------|
| $w_d$ | Importance weight of goal/task d |
| $\text{{Dep}}(i,j,d)$ | Binary: does i depend on j for d? |
| $\text{{crit}}(i,j,d)$ | Criticality: 1 = sole provider, 0 = many alternatives |

### Asymmetry and Power

**Key insight:** $D_{{ij}} \neq D_{{ji}}$ in general.

Power asymmetries create strategic leverage:
- A startup may critically depend on a platform ($D_{{\text{{startup,platform}}}} \approx 0.8$)
- The platform barely notices any single startup ($D_{{\text{{platform,startup}}}} \approx 0.01$)

### Integration with Utility

Interdependence enters the utility function:

$$U_i(\mathbf{{a}}) = \pi_i(\mathbf{{a}}) + \sum_{{j \neq i}} D_{{ij}} \cdot \pi_j(\mathbf{{a}})$$

High $D_{{ij}}$ means agent i rationally cares about agent j's payoff—not from altruism but from structural coupling.

---

## See Also

- [Value Functions](value_functions.md) - Value creation
- [Equilibrium](equilibrium.md) - Integrated utility computation
- [TR-1 Theory](../../theory/interdependence.md) - Mathematical foundations
- [PartnerHoldUp-v0](../../environments/partner_holdup.md) - Power asymmetry environment
