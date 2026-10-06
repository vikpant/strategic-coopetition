# API Reference

This reference describes **Coopetition-Gym 1.0.8, an unreleased source candidate**.

## Factory functions

| Function | Input | Output |
|---|---|---|
| `make(env_id, **kwargs)` | Environment ID and constructor/configuration arguments | Joint-action Gymnasium-style environment |
| `make_parallel(env_id, obs_config=None, render_mode=None, **kwargs)` | ID and optional observation configuration | PettingZoo Parallel environment |
| `make_aec(env_id, obs_config=None, render_mode=None, **kwargs)` | ID and optional observation configuration | PettingZoo AEC environment |
| `list_environments()` | None | 20 IDs in registry order |
| `version()` | None | Source version string |
| `info()` | None | Prints source version, environment count and APIs |

Unknown IDs and invalid reward types raise `ValueError`; unknown configuration keywords raise `TypeError`. Configuration overrides are applied without mutating a supplied `EnvironmentConfig`.

```python
import coopetition_gym as cg
import numpy as np

env = cg.make("PlatformEcosystem-v0", n_developers=4,
              reward_type="private", max_steps=20)
obs, info = env.reset(seed=42)
actions = np.minimum(env.endowments, 50.0).astype(np.float32)
obs, rewards, terminated, truncated, info = env.step(actions)
assert rewards.shape == (env.n_agents,)
env.close()
```

## Gymnasium registration

```python
import gymnasium as gym

env = gym.make("coopetition_gym:TrustDilemma-v0")
obs, info = env.reset(seed=42)
obs, rewards, terminated, truncated, info = env.step([50.0, 50.0])
env.close()
```

Importing `coopetition_gym` registers all base IDs. `_register_gymnasium_envs()` is also the distribution entry-point target and can be called repeatedly without replacing existing registrations.

The reward remains a NumPy vector with one entry per agent. Registration disables Gymnasium's scalar passive reward checker for this API; it does not convert rewards into a scalar or guarantee compatibility with arbitrary single-agent learners. See the [scalar-reward adapter](wrappers.md#scalar-reward-adapter).

## Parallel API

```python
import coopetition_gym as cg
from coopetition_gym import ObservationConfig

env = cg.make_parallel(
    "TrustDilemma-v0", max_steps=20,
    obs_config=ObservationConfig.realistic_asymmetry(),
)
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

## AEC API

```python
import coopetition_gym as cg

env = cg.make_aec("TrustDilemma-v0", max_steps=20)
env.reset(seed=42)
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    action = None if terminated or truncated else env.action_space(agent).sample()
    env.step(action)
env.close()
```

## Configuration and environment reference

- [Configuration](configuration.md): reward objectives and dataclass fields.
- [Wrappers](wrappers.md): observations, reset behavior and scalar aggregation.
- [Environment classes](environments.md): actual defaults and implemented methods.
- [Quick reference](quick_reference.md): all 20 IDs, agent counts and horizons.

Base environments accept one cooperation value per agent; rewards and dynamics include structural interdependence and the relevant trust, collective-action or reciprocity mechanisms. The optional `slcd_2d` package adds an appropriation dimension and has a separate environment registry.

## Core modules

| Module | Reference |
|---|---|
| `coopetition_gym.core.value_functions` | [Value functions](core/value_functions.md) |
| `coopetition_gym.core.interdependence` | [Interdependence](core/interdependence.md) |
| `coopetition_gym.core.trust_dynamics` | [Trust dynamics](core/trust_dynamics.md) |
| `coopetition_gym.core.equilibrium` | [Payoffs and equilibrium](core/equilibrium.md) |
| `coopetition_gym.core.collective_action` | [Source](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/core/collective_action.py) |
| `coopetition_gym.core.reciprocity` | [Source](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/core/reciprocity.py) |

```python
from coopetition_gym.core import create_slcd_payoff_params, solve_equilibrium

params = create_slcd_payoff_params()
result = solve_equilibrium(params, equilibrium_type="coopetitive")
print(result.converged, result.actions, result.total_welfare)
```

This solves the configured model. It is not an empirical validation score. See [score provenance](../benchmarks/score_provenance.md) and the [repository reproduction guide](https://github.com/vikpant/strategic-coopetition/blob/master/REPRODUCE.md) before interpreting historical benchmark results.
