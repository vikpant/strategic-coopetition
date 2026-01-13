# Quick Reference

Coopetition-Gym v0.2.0 quick reference card.

---

## Environment Creation

```python
import coopetition_gym

# Gymnasium API (default)
env = coopetition_gym.make("TrustDilemma-v0")

# PettingZoo Parallel API (simultaneous moves)
env = coopetition_gym.make_parallel("TrustDilemma-v0")

# PettingZoo AEC API (sequential moves)
env = coopetition_gym.make_aec("TrustDilemma-v0")

# List all environments
coopetition_gym.list_environments()
```

---

## Available Environments

| ID | Type | Agents | Challenge |
|----|------|--------|-----------|
| `TrustDilemma-v0` | Dyadic | 2 | Long-horizon trust |
| `PartnerHoldUp-v0` | Dyadic | 2 | Power asymmetry |
| `PlatformEcosystem-v0` | Ecosystem | N+1 | Ecosystem health |
| `DynamicPartnerSelection-v0` | Ecosystem | N | Reputation signals |
| `RecoveryRace-v0` | Benchmark | 2 | Trust recovery |
| `SynergySearch-v0` | Benchmark | 2 | Hidden parameter |
| `SLCD-v0` | Case Study | 2 | Validated (58/60) |
| `RenaultNissan-v0` | Case Study | 2 | Multi-phase |
| `CooperativeNegotiation-v0` | Extended | 2 | Commitments |
| `ReputationMarket-v0` | Extended | N | Tiered rewards |

---

## Standard Loop (Gymnasium)

```python
import numpy as np

env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

for _ in range(100):
    actions = np.array([50.0, 50.0])
    obs, rewards, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break

env.close()
```

---

## Standard Loop (PettingZoo Parallel)

```python
env = coopetition_gym.make_parallel("TrustDilemma-v0")
observations, infos = env.reset(seed=42)

for _ in range(100):
    actions = {agent: 50.0 for agent in env.agents}
    observations, rewards, terms, truncs, infos = env.step(actions)
    if all(terms.values()) or all(truncs.values()):
        break

env.close()
```

---

## Standard Loop (PettingZoo AEC)

```python
env = coopetition_gym.make_aec("TrustDilemma-v0")
env.reset(seed=42)

for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    action = None if term or trunc else 50.0
    env.step(action)

env.close()
```

---

## Key Parameters

### Trust Dynamics

| Symbol | Default | Range | Meaning |
|--------|---------|-------|---------|
| λ⁺ | 0.10 | (0, 1) | Trust building rate |
| λ⁻ | 0.30 | (0, 1) | Trust erosion rate |
| μ_R | 0.60 | (0, 1) | Reputation damage |
| δ_R | 0.03 | (0, 0.1) | Reputation decay |
| ξ | 0.50 | (0, 1) | Dependency amplification |

### Value Function

| Symbol | Default | Range | Meaning |
|--------|---------|-------|---------|
| θ | 20.0 | > 0 | Logarithmic scale |
| γ | 0.65 | [0, 1] | Complementarity |
| β | 0.75 | (0, 1) | Power exponent |

---

## Info Dictionary Keys

| Key | Type | Environments |
|-----|------|--------------|
| `step` | int | All |
| `mean_trust` | float | All |
| `mean_reputation_damage` | float | All |
| `total_value` | float | All |
| `cooperation_rate` | float | All |
| `trust_matrix` | NDArray | All |
| `true_gamma` | float | SynergySearch |
| `weak_trust_in_strong` | float | PartnerHoldUp |
| `phase` | str | RenaultNissan |

---

## Imports

```python
# Main API
from coopetition_gym import make, make_parallel, make_aec, list_environments

# Configuration
from coopetition_gym import ObservationConfig

# Core modules (advanced)
from coopetition_gym.core.value_functions import (
    ValueFunctionParameters,
    logarithmic_value,
    synergy_function,
    total_value,
)

from coopetition_gym.core.trust_dynamics import (
    TrustParameters,
    TrustState,
    TrustDynamicsModel,
)

from coopetition_gym.core.interdependence import (
    InterdependenceMatrix,
    create_slcd_interdependence,
)

from coopetition_gym.core.equilibrium import (
    PayoffParameters,
    compute_rewards,
    solve_equilibrium,
)
```

---

## Links

- [Full API Reference](index.md)
- [Environment Documentation](../environments/index.md)
- [Theoretical Foundations](../theory/index.md)
- [Tutorials](../tutorials/index.md)
