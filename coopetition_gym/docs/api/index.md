# API Reference

Complete API documentation for **Coopetition-Gym v0.2.0**.

*Generated: 2026-01-13*

---

## Quick Navigation

| Module | Description |
|--------|-------------|
| [Factory Functions](#factory-functions) | Environment creation |
| [Core: Value Functions](core/value_functions.md) | TR-1 value creation |
| [Core: Interdependence](core/interdependence.md) | TR-1 structural coupling |
| [Core: Trust Dynamics](core/trust_dynamics.md) | TR-2 trust evolution |
| [Core: Equilibrium](core/equilibrium.md) | Payoff computation |
| [Environments](environments.md) | Environment classes |
| [Wrappers](wrappers.md) | PettingZoo adapters |
| [Configuration](configuration.md) | Dataclass configs |

---

## Package Overview

```python
import coopetition_gym

# Version and metadata
coopetition_gym.__version__  # '0.2.0'
coopetition_gym.__author__   # 'Vik Pant, Eric Yu'

# List available environments
coopetition_gym.list_environments()
# ['TrustDilemma-v0', 'PartnerHoldUp-v0', ...]
```

---

## Factory Functions

### make

```python
coopetition_gym.make(
    env_id: str,
    **kwargs
) -> gymnasium.Env
```

Create a Gymnasium-compatible coopetition environment.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `env_id` | `str` | Environment identifier (see [`list_environments()`](#list_environments)) |
| `**kwargs` | | Environment-specific configuration parameters |

**Returns:**

| Type | Description |
|------|-------------|
| `gymnasium.Env` | Gymnasium-compatible environment instance |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Unknown environment ID |
| `TypeError` | Invalid configuration parameter |

**Example:**

```python
import coopetition_gym
import numpy as np

# Basic usage
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

# With custom parameters
env = coopetition_gym.make(
    "PlatformEcosystem-v0",
    n_developers=8,
    max_steps=200
)

# Step through environment
actions = np.array([50.0, 50.0])
obs, rewards, terminated, truncated, info = env.step(actions)
```

**See Also:**

- [`make_parallel()`](#make_parallel) - PettingZoo Parallel API
- [`make_aec()`](#make_aec) - PettingZoo AEC API
- [Environment Reference](../environments/index.md) - Full environment documentation

---

### make_parallel

```python
coopetition_gym.make_parallel(
    env_id: str,
    obs_config: Optional[ObservationConfig] = None,
    render_mode: Optional[str] = None,
    **kwargs
) -> CoopetitionParallelEnv
```

Create a PettingZoo Parallel API environment for simultaneous agent moves.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `env_id` | `str` | *required* | Environment identifier |
| `obs_config` | `ObservationConfig` | `None` | Observation configuration (see [ObservationConfig](wrappers.md#observationconfig)) |
| `render_mode` | `str` | `None` | Rendering mode (`None`, `"ansi"`, `"rgb_array"`) |
| `**kwargs` | | | Environment-specific parameters |

**Returns:**

| Type | Description |
|------|-------------|
| `CoopetitionParallelEnv` | PettingZoo-compatible parallel environment |

**Example:**

```python
import coopetition_gym

# Basic parallel environment
env = coopetition_gym.make_parallel("TrustDilemma-v0")
observations, infos = env.reset(seed=42)

# Actions are dictionaries keyed by agent name
actions = {{agent: 50.0 for agent in env.agents}}
observations, rewards, terminations, truncations, infos = env.step(actions)

# With realistic observation asymmetry (agents can't see others' trust toward them)
from coopetition_gym import ObservationConfig

env = coopetition_gym.make_parallel(
    "TrustDilemma-v0",
    obs_config=ObservationConfig.realistic_asymmetry()
)
```

**Notes:**

- All agents act simultaneously each step
- Observations and actions are dictionaries keyed by agent ID
- Agent IDs follow pattern `"agent_0"`, `"agent_1"`, etc.

**See Also:**

- [`make_aec()`](#make_aec) - Sequential moves
- [ObservationConfig](wrappers.md#observationconfig) - Observation configuration

---

### make_aec

```python
coopetition_gym.make_aec(
    env_id: str,
    obs_config: Optional[ObservationConfig] = None,
    render_mode: Optional[str] = None,
    **kwargs
) -> CoopetitionAECEnv
```

Create a PettingZoo AEC (Agent Environment Cycle) environment for sequential moves.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `env_id` | `str` | *required* | Environment identifier |
| `obs_config` | `ObservationConfig` | `None` | Observation configuration |
| `render_mode` | `str` | `None` | Rendering mode |
| `**kwargs` | | | Environment-specific parameters |

**Returns:**

| Type | Description |
|------|-------------|
| `CoopetitionAECEnv` | PettingZoo AEC environment |

**Example:**

```python
import coopetition_gym

env = coopetition_gym.make_aec("TrustDilemma-v0")
env.reset(seed=42)

# Iterate through agents sequentially
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = 50.0  # Your policy here

    env.step(action)
```

**Notes:**

- Agents take turns acting in sequence
- Use `agent_iter()` for standard iteration pattern
- Use `last()` to get current agent's observation

---

### list_environments

```python
coopetition_gym.list_environments() -> List[str]
```

Return list of all available environment identifiers.

**Returns:**

| Type | Description |
|------|-------------|
| `List[str]` | Sorted list of environment IDs |

**Example:**

```python
import coopetition_gym

envs = coopetition_gym.list_environments()
print(envs)
# ['CooperativeNegotiation-v0', 'DynamicPartnerSelection-v0',
#  'PartnerHoldUp-v0', 'PlatformEcosystem-v0', 'RecoveryRace-v0',
#  'RenaultNissan-v0', 'ReputationMarket-v0', 'SLCD-v0',
#  'SynergySearch-v0', 'TrustDilemma-v0']
```

---

### version

```python
coopetition_gym.version() -> str
```

Return the package version string.

**Returns:**

| Type | Description |
|------|-------------|
| `str` | Version in semver format (e.g., `"0.2.0"`) |

---

### info

```python
coopetition_gym.info() -> None
```

Print package information including version, authors, and available environments.

**Example:**

```python
import coopetition_gym
coopetition_gym.info()
# Coopetition-Gym v0.2.0
# Authors: Vik Pant, Eric Yu
# Faculty of Information, University of Toronto
# ...
```

---

## Type Aliases

Common type aliases used throughout the API:

```python
from numpy.typing import NDArray
import numpy as np

# Array types
FloatArray = NDArray[np.floating]  # General floating-point array
IntArray = NDArray[np.integer]     # Integer array

# Common function signatures
ActionType = Union[float, NDArray[np.floating]]
ObservationType = NDArray[np.floating]
RewardType = NDArray[np.floating]
```

---

## Module Index

### Core Mathematical Modules

| Module | Description | Technical Report |
|--------|-------------|------------------|
| [`core.value_functions`](core/value_functions.md) | Individual and synergistic value computation | TR-1 §5-6 |
| [`core.interdependence`](core/interdependence.md) | Structural dependency matrices | TR-1 §3-4 |
| [`core.trust_dynamics`](core/trust_dynamics.md) | Trust and reputation evolution | TR-2 §4-6 |
| [`core.equilibrium`](core/equilibrium.md) | Payoff computation and equilibrium solving | TR-1 §7 |
| [`core.collective_action`](core/collective_action.md) | Collective action mechanics (skeleton) | TR-3 |
| [`core.reciprocity`](core/reciprocity.md) | Reciprocity dynamics (skeleton) | TR-4 |

### Environment Modules

| Module | Description |
|--------|-------------|
| [`envs.base`](environments.md#base-classes) | Abstract environment classes |
| [`envs.dyadic_envs`](environments.md#dyadic-environments) | 2-agent environments |
| [`envs.ecosystem_envs`](environments.md#ecosystem-environments) | N-agent environments |
| [`envs.benchmark_envs`](environments.md#benchmark-environments) | Research benchmarks |
| [`envs.case_study_envs`](environments.md#case-study-environments) | Validated case studies |
| [`envs.extended_envs`](environments.md#extended-environments) | Extended mechanics |

### Wrapper Modules

| Module | Description |
|--------|-------------|
| [`envs.wrappers.observation_config`](wrappers.md#observationconfig) | Observation configuration |
| [`envs.wrappers.parallel_wrapper`](wrappers.md#coopetitionparallelenv) | PettingZoo Parallel adapter |
| [`envs.wrappers.aec_wrapper`](wrappers.md#coopetitionaecenv) | PettingZoo AEC adapter |

---

## Changelog

### v0.2.0 (Current)

- Added `ObservationConfig` for configurable information asymmetry
- Added `make_parallel()` and `make_aec()` factory functions
- Added PettingZoo wrapper classes
- Enhanced type annotations throughout

### v0.1.0

- Initial release
- 10 environments implemented
- Core mathematical framework complete

---

## See Also

- [Getting Started](../tutorials/quickstart.md) - Tutorial introduction
- [Environment Reference](../environments/index.md) - Detailed environment docs
- [Theoretical Foundations](../theory/index.md) - Mathematical background
