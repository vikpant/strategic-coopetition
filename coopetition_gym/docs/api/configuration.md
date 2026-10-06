# Configuration Classes

Dataclass configurations for environments and components.

**Source candidate: 1.0.8 (unreleased).**

---

## EnvironmentConfig

```python
@dataclass
class EnvironmentConfig:
    """
    Complete configuration for a coopetition environment.

    Used internally by environment classes. Most users should use
    factory functions (make, make_parallel, make_aec) instead.
    """

    n_agents: int = 2
    max_steps: int = 100
    endowments: Optional[NDArray[np.floating]] = None
    alpha: Optional[NDArray[np.floating]] = None
    interdependence_matrix: Optional[NDArray[np.floating]] = None
    value_params: Optional[ValueFunctionParameters] = None
    trust_params: Optional[TrustParameters] = None
    trust_enabled: bool = True
    baselines: Optional[NDArray[np.floating]] = None
    reward_type: str = "integrated"
    normalize_rewards: bool = False
    reward_scale: float = 1.0
    render_mode: Optional[str] = None
```

**Attributes:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_agents` | `int` | 2 | Number of agents |
| `max_steps` | `int` | 100 | Maximum episode length |
| `endowments` | `NDArray` | None | Agent endowments (auto-initialized if None) |
| `alpha` | `NDArray` | None | Bargaining shares (auto-initialized if None) |
| `interdependence_matrix` | `NDArray` | None | D matrix (auto-initialized if None) |
| `value_params` | `ValueFunctionParameters` | None | Value function config |
| `trust_params` | `TrustParameters` | None | Trust dynamics config |
| `trust_enabled` | `bool` | True | Enable trust dynamics |
| `baselines` | `NDArray` | None | Cooperation baselines |
| `reward_type` | `str` | "integrated" | 'private', 'integrated', or 'cooperative' |
| `normalize_rewards` | `bool` | False | Divide base reward by the largest endowment before scaling; no strict range guarantee |
| `reward_scale` | `float` | 1.0 | Reward scaling factor |
| `render_mode` | `str` | None | Rendering mode |

## Constructor overrides

```python
from coopetition_gym import make, CoopetitionEnv, EnvironmentConfig

env = make("SLCD-v0", reward_type="private", reward_scale=0.5)
env.close()

config = EnvironmentConfig()
env = CoopetitionEnv(config=config, reward_type="cooperative")
assert config.reward_type == "integrated"  # The caller's config is preserved.
env.close()
```

All 20 factories honor common configuration overrides in this candidate. An invalid `reward_type` raises `ValueError`; unknown field names raise `TypeError`. The valid objectives are `private`, `integrated` and `cooperative`. Cooperative base reward is the mean integrated utility shared among agents, before environment-specific modifiers. Normalization and reward scaling also precede those modifiers.

---

## TrustParameters

See [Trust Dynamics Module](core/trust_dynamics.md#trustparameters).

---

## ValueFunctionParameters

See [Value Functions Module](core/value_functions.md#valuefunctionparameters).

---

## ObservationConfig

See [Wrappers Module](wrappers.md#observationconfig).

---

## PayoffParameters

See [Equilibrium Module](core/equilibrium.md#payoffparameters).

---

## InterdependenceMatrix

See [Interdependence Module](core/interdependence.md#interdependencematrix).

---

## See Also

- [API Index](index.md) - Main API reference
- [Parameter Reference](../theory/parameters.md) - Validated parameter values
