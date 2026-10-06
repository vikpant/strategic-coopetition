# Wrappers and Observation Configuration

The **unreleased 1.0.8 candidate** exposes joint-action, Parallel and AEC APIs. The joint-action API returns a reward vector; PettingZoo APIs return one scalar reward per named agent.

## ObservationConfig

`ObservationConfig` is exported by `coopetition_gym`. Its implemented constructors are `full_observability()`, `realistic_asymmetry()` and `minimal()`.

```python
from coopetition_gym import make_parallel, ObservationConfig

env = make_parallel(
    "TrustDilemma-v0",
    obs_config=ObservationConfig.realistic_asymmetry(),
)
observations, infos = env.reset(seed=42)
print(env.observation_space("agent_0"), observations["agent_0"])
env.close()
```

| Field | Dataclass default |
|---|---|
| `own_actions_visible`, `others_actions_visible` | `True` |
| `action_history_depth` | `1` |
| `own_trust_row_visible` | `True` |
| `others_trust_toward_self_visible`, `full_trust_matrix_visible` | `False` |
| `own_reputation_visible`, `public_reputation_visible` | `True` |
| `interdependence_visible`, `step_count_visible` | `True` |
| `private_info_keys` | Empty list |

Factory-created base environments use full observability for compatibility unless `obs_config` is supplied to a PettingZoo factory. Configure visibility explicitly for experiments requiring information asymmetry.

## CoopetitionParallelEnv

Direct constructor: `CoopetitionParallelEnv(base_env, render_mode=None)`. Prefer `make_parallel(env_id, obs_config=..., **kwargs)` when selecting observations.

- `reset(seed=None, options=None)` returns `(observations, infos)` dictionaries.
- `step(actions)` returns `(observations, rewards, terminations, truncations, infos)` dictionaries.
- `action_space(agent)` and `observation_space(agent)` describe each agent's arrays.
- `agents` lists active agents; `possible_agents` lists all potential agents.
- `state()` provides a concatenation of per-agent observations for centralized consumers.

```python
import coopetition_gym as cg

env = cg.make_parallel("CoalitionFormation-v0", max_steps=5)
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

## CoopetitionAECEnv

Direct constructor: `CoopetitionAECEnv(base_env, render_mode=None)`. Prefer `make_aec` to configure observations.

```python
import coopetition_gym as cg

env = cg.make_aec("TrustDilemma-v0", max_steps=5)
env.reset(seed=42)
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    action = None if terminated or truncated else env.action_space(agent).sample()
    env.step(action)
env.close()
```

`reset` and `step` do not return transition tuples in AEC mode. Read `last()` for the selected agent and `observe(agent)` for an individual observation. A full action round advances the shared dynamics; earlier moves are exposed to later movers.

Both wrappers invoke the underlying environment's specialized reset with seed and options. This initializes coalitions and resets loyalty. DynamicPartnerSelection can preserve reputation by design; use `options={"reset_reputation": True}` to clear it.

## Scalar-reward adapter

Stable-Baselines3's ordinary single-agent learners expect scalar rewards. The packaged experiment adapter explicitly **sums** the multi-agent rewards and exposes a single controller over the joint action:

```python
import coopetition_gym as cg
from experiments.algorithms import MultiAgentToSingleAgentWrapper

base_env = cg.make("TrustDilemma-v0", reward_type="private")
env = MultiAgentToSingleAgentWrapper(base_env)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step([50.0, 50.0])
assert isinstance(reward, float)
env.close()
```

The historical `IndependentPPO`, `IndependentSAC` and `IndependentA2C` classes use this joint-controller adapter; their names alone do not establish decentralized independent learning. Record the actual controller and aggregation when interpreting results. Do not pass an unadapted reward-vector environment to `DummyVecEnv` or a scalar-reward learner and assume the objectives are unchanged.

Other frameworks may require their own PettingZoo or multi-agent adapters. This reference does not claim that registering a raw Parallel environment supplies those conversions.

See [factory functions](index.md#factory-functions), [environment classes](environments.md), and [PettingZoo's documentation](https://pettingzoo.farama.org/).
