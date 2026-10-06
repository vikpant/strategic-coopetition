# Quick Start Tutorial

These examples target the **unreleased 1.0.8 source candidate**. Complete the [installation guide](../installation.md) first. The examples run episodes without training a policy.

## One joint-action episode

```python
import numpy as np
import coopetition_gym as cg

env = cg.make("TrustDilemma-v0", reward_type="integrated", max_steps=20)
obs, info = env.reset(seed=42)
returns = np.zeros(env.n_agents)

while True:
    actions = np.array([60.0, 55.0], dtype=np.float32)
    obs, rewards, terminated, truncated, info = env.step(actions)
    returns += rewards
    if terminated or truncated:
        break

print(returns, info["mean_trust"])
env.close()
```

Each action is cooperation/investment within the corresponding agent's endowment. Zero means no contribution. The base environments model competitive incentives through payoffs and interdependence. The separate SLCD prototype adds an appropriation action.

`TrustDilemma-v0` has two agents, a `(2,)` joint action and `(15,)` observation. Its base observation contains:

| Slice | Contents |
|---|---|
| `obs[:2]` | Previous cooperation actions |
| `obs[2:6]` | Trust matrix, flattened |
| `obs[6:10]` | Reputation-damage matrix, flattened |
| `obs[10:14]` | Interdependence matrix, flattened |
| `obs[14]` | Step count |

Other environments can append observations. Inspect `env.observation_space` instead of assuming this shape everywhere.

## Choose the reward objective

```python
import coopetition_gym as cg

for mode in ("private", "integrated", "cooperative"):
    env = cg.make("SLCD-v0", reward_type=mode)
    env.reset(seed=42)
    _, rewards, _, _, _ = env.step([50.0, 50.0])
    print(mode, rewards)
    env.close()
```

`private` starts from each agent's payoff; `integrated` includes weighted partner payoffs; `cooperative` assigns the mean integrated utility as the shared base reward. Environment-specific modifiers still apply. Keep objective, horizon, seeds, and aggregation consistent when comparing runs.

The candidate validates `reward_type` and rejects unknown configuration arguments. See [configuration](../api/configuration.md).

## Gymnasium registration

```python
import gymnasium as gym

env = gym.make("coopetition_gym:TrustDilemma-v0", max_steps=20)
obs, info = env.reset(seed=42)
obs, rewards, terminated, truncated, info = env.step([50.0, 50.0])
env.close()
```

This API returns a reward vector, as does `cg.make`. It does not supply the scalar reward expected by ordinary single-agent Stable-Baselines3 learners. Use the [explicit scalar adapter](../api/wrappers.md#scalar-reward-adapter) when that is the intended objective.

## PettingZoo Parallel API

```python
import coopetition_gym as cg

env = cg.make_parallel("TrustDilemma-v0", max_steps=20)
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

Actions and rewards are dictionaries keyed by agent. This wrapper collects simultaneous actions.

## PettingZoo AEC API

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

AEC collects one action per turn and processes a full round after all active agents act. Later movers observe earlier current-round actions. Pass `None` when the selected agent has finished.

Both wrappers dispatch environment-specific resets. For environments with state intentionally preserved between episodes, use their reset options; for example, `DynamicPartnerSelection-v0` accepts `options={"reset_reputation": True}`. `reset(seed=...)` seeds environment dynamics; seed action spaces separately when reproducing random-action policies.

## Explore the catalog

```python
import coopetition_gym as cg

for env_id in cg.list_environments():
    env = cg.make(env_id)
    print(env_id, env.n_agents, env.max_steps, env.observation_space.shape)
    env.close()
```

There are 20 base environments. See [default agent counts and horizons](../api/quick_reference.md#available-environments), the [environment reference](../api/environments.md), and [score provenance](../benchmarks/score_provenance.md) for the boundaries between implementation facts and case-study validation claims.
