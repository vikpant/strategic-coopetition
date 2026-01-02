# API Reference

Complete API documentation for Coopetition-Gym.

---

## Factory Functions

### coopetition_gym.make

```python
coopetition_gym.make(env_id: str, **kwargs) -> gymnasium.Env
```

Create a Gymnasium-compatible environment.

**Parameters:**
- `env_id` (str): Environment identifier (e.g., "TrustDilemma-v0")
- `**kwargs`: Environment-specific configuration

**Returns:**
- `gymnasium.Env`: Gymnasium-compatible environment instance

**Example:**
```python
import coopetition_gym

# Basic usage
env = coopetition_gym.make("TrustDilemma-v0")

# With parameters
env = coopetition_gym.make("PlatformEcosystem-v0", n_developers=6)
```

---

### coopetition_gym.make_parallel

```python
coopetition_gym.make_parallel(env_id: str, **kwargs) -> pettingzoo.ParallelEnv
```

Create a PettingZoo Parallel API environment for simultaneous moves.

**Parameters:**
- `env_id` (str): Environment identifier
- `**kwargs`: Environment-specific configuration

**Returns:**
- `pettingzoo.ParallelEnv`: PettingZoo parallel environment instance

**Example:**
```python
import coopetition_gym

env = coopetition_gym.make_parallel("TrustDilemma-v0")
observations, infos = env.reset(seed=42)

# Actions are dictionaries
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

---

### coopetition_gym.make_aec

```python
coopetition_gym.make_aec(env_id: str, **kwargs) -> pettingzoo.AECEnv
```

Create a PettingZoo AEC (Agent Environment Cycle) environment for sequential moves.

**Parameters:**
- `env_id` (str): Environment identifier
- `**kwargs`: Environment-specific configuration

**Returns:**
- `pettingzoo.AECEnv`: PettingZoo AEC environment instance

**Example:**
```python
import coopetition_gym

env = coopetition_gym.make_aec("TrustDilemma-v0")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = None if termination or truncation else env.action_space(agent).sample()
    env.step(action)
```

---

### coopetition_gym.list_environments

```python
coopetition_gym.list_environments() -> List[str]
```

List all available environment identifiers.

**Returns:**
- `List[str]`: List of environment IDs

**Example:**
```python
import coopetition_gym

envs = coopetition_gym.list_environments()
print(envs)
# ['TrustDilemma-v0', 'PartnerHoldUp-v0', 'PlatformEcosystem-v0', ...]
```

---

## Environment Interface

All environments implement the Gymnasium interface.

### reset

```python
env.reset(seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]
```

Reset the environment to initial state.

**Parameters:**
- `seed` (int, optional): Random seed for reproducibility
- `options` (dict, optional): Additional reset options

**Returns:**
- `observation` (np.ndarray): Initial observation
- `info` (dict): Initial info dictionary

**Example:**
```python
obs, info = env.reset(seed=42)
print(f"Initial trust: {info['mean_trust']}")
```

---

### step

```python
env.step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]
```

Execute one timestep.

**Parameters:**
- `action` (np.ndarray): Joint action array [a_0, a_1, ..., a_n]

**Returns:**
- `observation` (np.ndarray): Next observation
- `rewards` (np.ndarray): Rewards for each agent
- `terminated` (bool): Whether episode ended due to terminal state
- `truncated` (bool): Whether episode ended due to time limit
- `info` (dict): Additional information

**Example:**
```python
import numpy as np

actions = np.array([60.0, 55.0])  # Cooperation levels
obs, rewards, terminated, truncated, info = env.step(actions)
```

---

### render

```python
env.render() -> Optional[Union[np.ndarray, str]]
```

Render the environment.

**Returns:**
- Depends on `render_mode`:
  - `None`: No return (renders to display)
  - `"ansi"`: String representation
  - `"rgb_array"`: NumPy array of pixels

---

### close

```python
env.close() -> None
```

Clean up environment resources.

---

## Environment Properties

### Common Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_agents` | int | Number of agents |
| `observation_space` | gymnasium.Space | Observation space |
| `action_space` | gymnasium.Space | Action space |
| `endowments` | np.ndarray | Agent endowments |
| `baselines` | np.ndarray | Agent baselines |
| `alphas` | np.ndarray | Bargaining shares |

**Example:**
```python
env = coopetition_gym.make("TrustDilemma-v0")
print(f"Agents: {env.n_agents}")
print(f"Observation shape: {env.observation_space.shape}")
print(f"Action bounds: [{env.action_space.low}, {env.action_space.high}]")
print(f"Endowments: {env.endowments}")
```

---

## Observation Structure

### Dyadic Environments (2 Agents)

Observation dimension: 17

| Index | Component | Shape | Description |
|-------|-----------|-------|-------------|
| 0-1 | Actions | (2,) | Previous cooperation levels |
| 2-5 | Trust Matrix | (4,) | Flattened 2×2 trust τ_ij |
| 6-9 | Reputation Matrix | (4,) | Flattened 2×2 reputation R_ij |
| 10-13 | Interdependence | (4,) | Flattened 2×2 dependency D_ij |
| 14-16 | Metadata | (3,) | Timestep, auxiliary info |

### N-Agent Environments

Observation dimension: (N+1) + 3(N+1)² + 1 + extras

**PlatformEcosystem-v0** (N developers):

| n_developers | Total Agents | State Dim |
|--------------|--------------|-----------|
| 4 | 5 | 81 |
| 8 | 9 | 253 |
| 16 | 17 | 885 |

---

## Action Space

### Continuous Actions

All environments use continuous action spaces:

```python
Box(low=0.0, high=endowment_i, shape=(n_agents,), dtype=float32)
```

**Interpretation:**
- `0.0`: Full defection (no cooperation)
- `baseline_i`: Minimum expected cooperation (~35% of endowment)
- `endowment_i`: Full cooperation (maximum investment)

### Agent-Specific Bounds

| Environment | Agent | Action Range |
|-------------|-------|--------------|
| TrustDilemma-v0 | All | [0, 100] |
| PartnerHoldUp-v0 | Strong | [0, 120] |
| PartnerHoldUp-v0 | Weak | [0, 80] |
| PlatformEcosystem-v0 | Platform | [0, 150] |
| PlatformEcosystem-v0 | Developers | [0, 80] |

---

## Info Dictionary

### Common Keys

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `mean_trust` | float | Mean pairwise trust |
| `mean_reputation_damage` | float | Mean reputation damage |
| `total_value` | float | Total value created |
| `mean_cooperation` | float | Mean cooperation level |
| `cooperation_rate` | float | Cooperation as % of endowments |
| `trust_matrix` | np.ndarray | Full trust matrix |
| `reputation_matrix` | np.ndarray | Full reputation matrix |

### Environment-Specific Keys

**PartnerHoldUp-v0:**
| Key | Type | Description |
|-----|------|-------------|
| `weak_trust_in_strong` | float | Critical exit metric |
| `power_asymmetry` | float | Dependency imbalance |

**PlatformEcosystem-v0:**
| Key | Type | Description |
|-----|------|-------------|
| `developer_trust_in_platform` | float | Ecosystem health |
| `platform_investment` | float | Platform's action |
| `mean_developer_investment` | float | Average developer action |

**SynergySearch-v0:**
| Key | Type | Description |
|-----|------|-------------|
| `true_gamma` | float | Hidden complementarity |
| `gamma_type` | str | "high_synergy" or "low_synergy" |

**SLCD-v0:**
| Key | Type | Description |
|-----|------|-------------|
| `samsung_investment` | float | Samsung's action |
| `sony_investment` | float | Sony's action |
| `samsung_payoff` | float | Samsung's reward |
| `sony_payoff` | float | Sony's reward |

---

## Configuration Parameters

### Trust Parameters

Available in most environments:

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| `lambda_plus` | λ⁺ | 0.08-0.15 | Trust building rate |
| `lambda_minus` | λ⁻ | 0.25-0.45 | Trust erosion rate |
| `mu_r` | μ_R | 0.45-0.70 | Reputation damage coefficient |
| `delta_r` | δ_R | 0.01-0.03 | Reputation decay rate |
| `xi` | ξ | 0.40-0.70 | Interdependence amplification |
| `kappa` | κ | 1.0-1.5 | Signal sensitivity |
| `initial_trust` | τ₀ | 0.50-0.65 | Starting trust level |

### Value Function Parameters

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| `theta` | θ | 18-25 | Logarithmic scale factor |
| `gamma` | γ | 0.50-0.75 | Complementarity coefficient |

---

## PettingZoo Parallel API

### Agent Naming

```python
env = coopetition_gym.make_parallel("TrustDilemma-v0")
print(env.possible_agents)  # ['agent_0', 'agent_1']
print(env.agents)           # Active agents (same initially)
```

### Action/Observation Dictionaries

```python
# Observations are dictionaries
observations, infos = env.reset(seed=42)
print(observations['agent_0'].shape)  # (17,)

# Actions must be dictionaries
actions = {'agent_0': 60.0, 'agent_1': 55.0}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

### Agent Methods

```python
# Per-agent spaces
obs_space = env.observation_space('agent_0')
act_space = env.action_space('agent_0')
```

---

## PettingZoo AEC API

### Agent Iteration

```python
env = coopetition_gym.make_aec("TrustDilemma-v0")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = policy(observation)

    env.step(action)
```

### Current Agent

```python
current_agent = env.agent_selection
```

---

## Integration Examples

### Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import coopetition_gym

# Wrap for SB3
env = DummyVecEnv([lambda: coopetition_gym.make("TrustDilemma-v0")])

# Train
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

### RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import coopetition_gym

# Register environment
def env_creator(config):
    return coopetition_gym.make_parallel("TrustDilemma-v0")

register_env("trust_dilemma", env_creator)

# Configure
config = PPOConfig().environment("trust_dilemma")
```

---

## Error Handling

### Common Exceptions

| Exception | Cause | Solution |
|-----------|-------|----------|
| `ValueError` | Invalid env_id | Check `list_environments()` |
| `TypeError` | Wrong action type | Use numpy array |
| `AssertionError` | Action out of bounds | Clip to action space |

### Action Clipping

```python
import numpy as np

# Safe action execution
action = np.clip(action, env.action_space.low, env.action_space.high)
obs, reward, terminated, truncated, info = env.step(action)
```

---

## See Also

- [Environment Reference](../environments/index.md) - Detailed environment documentation
- [Tutorials](../tutorials/index.md) - Usage examples
- [Installation](../installation.md) - Setup guide
