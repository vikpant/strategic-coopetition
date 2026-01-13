# Wrappers Module

PettingZoo adapters and observation configuration.

*Module version: 0.2.0*

---

## ObservationConfig

```python
@dataclass
class ObservationConfig:
    """
    Configure agent-specific observation structure.

    Controls what each agent can observe, enabling realistic
    information asymmetry in multi-agent settings.
    """

    own_actions_visible: bool = True
    others_actions_visible: bool = True
    action_history_depth: int = 1
    own_trust_row_visible: bool = True
    others_trust_toward_self_visible: bool = False  # KEY: False by default
    full_trust_matrix_visible: bool = False
    own_reputation_visible: bool = True
    public_reputation_visible: bool = True
    interdependence_visible: bool = True
    step_count_visible: bool = True
    private_info_keys: List[str] = field(default_factory=list)
```

**Attributes:**

| Name | Default | Description |
|------|---------|-------------|
| `own_actions_visible` | `True` | Agent sees own previous actions |
| `others_actions_visible` | `True` | Agent sees others' previous actions |
| `own_trust_row_visible` | `True` | Agent sees own trust toward others |
| `others_trust_toward_self_visible` | `False` | Agent sees others' trust toward them |
| `full_trust_matrix_visible` | `False` | Agent sees complete trust matrix |

**Class Methods:**

```python
@classmethod
def full_observability(cls) -> "ObservationConfig":
    """Legacy mode: all information visible (v0.1.0 behavior)."""

@classmethod
def realistic_asymmetry(cls) -> "ObservationConfig":
    """
    Recommended: agents can't see others' trust toward them.

    This models realistic information asymmetry where you know
    how much you trust others, but not how much they trust you.
    """

@classmethod
def minimal(cls) -> "ObservationConfig":
    """Minimal observation: only own actions and trust row."""
```

**Example:**

```python
from coopetition_gym import make_parallel, ObservationConfig

# Default (full observability for backward compatibility)
env = make_parallel("TrustDilemma-v0")

# Realistic asymmetry (recommended for research)
config = ObservationConfig.realistic_asymmetry()
env = make_parallel("TrustDilemma-v0", obs_config=config)

# Custom configuration
config = ObservationConfig(
    own_trust_row_visible=True,
    others_trust_toward_self_visible=False,
    full_trust_matrix_visible=False,
    others_actions_visible=True
)
env = make_parallel("TrustDilemma-v0", obs_config=config)
```

---

## CoopetitionParallelEnv

```python
class CoopetitionParallelEnv:
    """
    PettingZoo ParallelEnv wrapper for simultaneous moves.

    All agents act at the same time. Actions and observations
    are dictionaries keyed by agent ID.
    """

    def __init__(
        self,
        base_env: CoopetitionEnv,
        obs_config: Optional[ObservationConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Wrap a CoopetitionEnv for PettingZoo Parallel API.

        Args:
            base_env: Underlying Gymnasium environment
            obs_config: Observation configuration
            render_mode: Rendering mode
        """
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `agents` | `List[str]` | Currently active agent IDs |
| `possible_agents` | `List[str]` | All possible agent IDs |
| `num_agents` | `int` | Number of active agents |

**Methods:**

```python
def reset(
    self,
    seed: Optional[int] = None,
    options: Optional[dict] = None
) -> Tuple[Dict[str, NDArray], Dict[str, dict]]:
    """Reset environment. Returns (observations, infos)."""

def step(
    self,
    actions: Dict[str, float]
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Execute simultaneous actions.

    Args:
        actions: Dict mapping agent_id -> action

    Returns:
        (observations, rewards, terminations, truncations, infos)
        All are dicts keyed by agent_id.
    """

def observation_space(self, agent: str) -> gym.Space:
    """Get observation space for specific agent."""

def action_space(self, agent: str) -> gym.Space:
    """Get action space for specific agent."""

def render(self) -> Optional[str]:
    """Render environment."""

def close(self):
    """Clean up resources."""
```

**Example:**

```python
import coopetition_gym

env = coopetition_gym.make_parallel("TrustDilemma-v0")
observations, infos = env.reset(seed=42)

print(f"Agents: {env.agents}")  # ['agent_0', 'agent_1']
print(f"Observation shape: {observations['agent_0'].shape}")

# Step with all agents
actions = {'agent_0': 60.0, 'agent_1': 55.0}
observations, rewards, terms, truncs, infos = env.step(actions)

print(f"Rewards: {rewards}")
```

---

## CoopetitionAECEnv

```python
class CoopetitionAECEnv:
    """
    PettingZoo AECEnv wrapper for sequential moves.

    Agents take turns acting. Use agent_iter() for standard loop.
    """

    def __init__(
        self,
        base_env: CoopetitionEnv,
        obs_config: Optional[ObservationConfig] = None,
        render_mode: Optional[str] = None
    ):
        """Wrap a CoopetitionEnv for PettingZoo AEC API."""
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `agent_selection` | `str` | Current agent's turn |
| `agents` | `List[str]` | Active agents |
| `rewards` | `Dict[str, float]` | Current rewards |
| `terminations` | `Dict[str, bool]` | Termination status |
| `truncations` | `Dict[str, bool]` | Truncation status |
| `infos` | `Dict[str, dict]` | Info dictionaries |

**Methods:**

```python
def reset(
    self,
    seed: Optional[int] = None,
    options: Optional[dict] = None
):
    """Reset environment. Does not return values."""

def step(self, action: Optional[float]):
    """
    Execute action for current agent.

    Args:
        action: Action for current agent, or None if terminated
    """

def last(self) -> Tuple[NDArray, float, bool, bool, dict]:
    """Get current agent's observation, reward, term, trunc, info."""

def agent_iter(self, max_iter: int = 2**63) -> Iterator[str]:
    """Iterate through agents until episode ends."""

def observe(self, agent: str) -> NDArray:
    """Get observation for specific agent."""
```

**Example:**

```python
import coopetition_gym

env = coopetition_gym.make_aec("TrustDilemma-v0")
env.reset(seed=42)

# Standard AEC loop
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = 50.0  # Your policy here

    env.step(action)
```

---

## Integration with MARL Frameworks

### Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import coopetition_gym

# Use Gymnasium API
env = DummyVecEnv([lambda: coopetition_gym.make("TrustDilemma-v0")])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

### RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import coopetition_gym

# Use PettingZoo Parallel API
def env_creator(config):
    return coopetition_gym.make_parallel("TrustDilemma-v0")

register_env("trust_dilemma", env_creator)
config = PPOConfig().environment("trust_dilemma")
```

### SuperSuit Wrappers

```python
import supersuit as ss
import coopetition_gym

# Create parallel env and apply supersuit wrappers
env = coopetition_gym.make_parallel("TrustDilemma-v0")
env = ss.normalize_obs_v0(env)
env = ss.pad_action_space_v0(env)
```

---

## See Also

- [Factory Functions](index.md#make_parallel) - Creating wrapped environments
- [Environments](environments.md) - Base environment classes
- [PettingZoo Documentation](https://pettingzoo.farama.org/) - Framework reference
