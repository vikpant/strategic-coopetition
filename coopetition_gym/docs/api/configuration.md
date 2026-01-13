# Configuration Classes

Dataclass configurations for environments and components.

*Module version: 0.2.0*

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
| `normalize_rewards` | `bool` | False | Normalize rewards to [0, 1] |
| `reward_scale` | `float` | 1.0 | Reward scaling factor |
| `render_mode` | `str` | None | Rendering mode |

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
