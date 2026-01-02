# Quick Start Tutorial

This tutorial introduces Coopetition-Gym through hands-on examples. You'll learn to create environments, interact with them, and understand the core concepts.

**Time:** 15 minutes
**Prerequisites:** Python basics, NumPy familiarity
**Before starting:** Complete [installation](../installation.md)

---

## Your First Environment

Let's create and interact with the TrustDilemma-v0 environment:

```python
import coopetition_gym
import numpy as np

# Create the environment
env = coopetition_gym.make("TrustDilemma-v0")

# Reset with a seed for reproducibility
obs, info = env.reset(seed=42)

print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print(f"Number of agents: {env.n_agents}")
```

**Expected output:**
```
Observation shape: (17,)
Action space: Box(0.0, 100.0, (2,), float32)
Number of agents: 2
```

---

## Understanding the Observation

The observation contains the current state of the environment:

```python
# Reset and examine observation
obs, info = env.reset(seed=42)

# The observation is a flat array with structure:
# [actions(2), trust_matrix(4), reputation_matrix(4), interdependence(4), metadata(3)]
print(f"Previous actions: {obs[0:2]}")
print(f"Trust matrix (flattened): {obs[2:6]}")
print(f"Reputation damage: {obs[6:10]}")
print(f"Interdependence: {obs[10:14]}")
print(f"Metadata: {obs[14:17]}")
```

**Key components:**
- **Actions**: What each agent did last step
- **Trust matrix**: Pairwise trust levels τ_ij ∈ [0, 1]
- **Reputation damage**: Accumulated damage R_ij ∈ [0, 1]
- **Interdependence**: How much each agent depends on others D_ij

---

## Taking Actions

Actions represent **cooperation levels** - how much each agent invests in the partnership:

```python
# Both agents choose cooperation levels
# Agent 0: 60 out of 100 endowment (60% cooperation)
# Agent 1: 55 out of 100 endowment (55% cooperation)
actions = np.array([60.0, 55.0])

# Take a step
obs, rewards, terminated, truncated, info = env.step(actions)

print(f"Rewards: Agent 0 = {rewards[0]:.2f}, Agent 1 = {rewards[1]:.2f}")
print(f"Current trust: {info['mean_trust']:.3f}")
print(f"Terminated: {terminated}, Truncated: {truncated}")
```

**Action interpretation:**
- **High action (60-100)**: Cooperation - investing in joint value
- **Medium action (35-60)**: Cautious - balanced approach
- **Low action (0-35)**: Defection - prioritizing self-interest

---

## Running a Complete Episode

Let's run a full episode with a simple strategy:

```python
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

total_rewards = np.zeros(2)
trust_history = []
step_count = 0

while True:
    # Simple strategy: both agents cooperate moderately
    actions = np.array([55.0, 55.0])

    obs, rewards, terminated, truncated, info = env.step(actions)
    total_rewards += rewards
    trust_history.append(info['mean_trust'])
    step_count += 1

    if terminated or truncated:
        break

print(f"Episode finished after {step_count} steps")
print(f"Total rewards: Agent 0 = {total_rewards[0]:.1f}, Agent 1 = {total_rewards[1]:.1f}")
print(f"Final trust: {trust_history[-1]:.3f}")
print(f"Trust range: [{min(trust_history):.3f}, {max(trust_history):.3f}]")
```

---

## Understanding Rewards

Rewards come from **integrated utility** - a combination of individual value and partner outcomes:

```python
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

# Compare different action profiles
scenarios = [
    ("Mutual cooperation", np.array([70.0, 70.0])),
    ("Mutual defection", np.array([20.0, 20.0])),
    ("Agent 0 defects", np.array([20.0, 70.0])),
    ("Agent 1 defects", np.array([70.0, 20.0])),
]

for name, actions in scenarios:
    obs, info = env.reset(seed=42)  # Reset to same state
    obs, rewards, _, _, info = env.step(actions)
    print(f"{name}: R0={rewards[0]:.1f}, R1={rewards[1]:.1f}, Trust={info['mean_trust']:.3f}")
```

**Key insight:** High mutual cooperation yields better total rewards, but defecting while the partner cooperates can be individually tempting (the dilemma!).

---

## Trust Dynamics

Trust is the key state variable that evolves based on actions:

```python
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

print("Demonstrating trust dynamics:\n")

# Phase 1: Build trust through cooperation
print("Phase 1: Building trust (cooperation)")
for _ in range(10):
    obs, rewards, _, _, info = env.step(np.array([70.0, 70.0]))
print(f"  Trust after cooperation: {info['mean_trust']:.3f}")

# Phase 2: Erode trust through defection
print("\nPhase 2: Eroding trust (defection)")
for _ in range(5):
    obs, rewards, _, _, info = env.step(np.array([20.0, 20.0]))
print(f"  Trust after defection: {info['mean_trust']:.3f}")

# Phase 3: Attempt recovery
print("\nPhase 3: Attempting recovery")
for _ in range(10):
    obs, rewards, _, _, info = env.step(np.array([70.0, 70.0]))
print(f"  Trust after recovery attempt: {info['mean_trust']:.3f}")
```

**Key properties:**
- Trust builds slowly (λ⁺ ≈ 0.10-0.15)
- Trust erodes quickly (λ⁻ ≈ 0.30-0.45)
- **3:1 negativity bias**: Trust erodes 3× faster than it builds

---

## Using Different APIs

Coopetition-Gym supports three APIs:

### Gymnasium API (Default)

```python
# Standard Gymnasium interface
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)
actions = np.array([50.0, 50.0])  # Joint action array
obs, rewards, terminated, truncated, info = env.step(actions)
```

### PettingZoo Parallel API

```python
# For simultaneous-move multi-agent settings
env = coopetition_gym.make_parallel("TrustDilemma-v0")
observations, infos = env.reset(seed=42)

# Actions are dictionaries keyed by agent name
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)

print(f"Agents: {env.agents}")
print(f"Rewards: {rewards}")
```

### PettingZoo AEC API

```python
# For turn-based or sequential settings
env = coopetition_gym.make_aec("TrustDilemma-v0")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)
```

---

## Exploring Available Environments

Coopetition-Gym provides 10 environments:

```python
# List all available environments
envs = coopetition_gym.list_environments()
print("Available environments:")
for env_id in envs:
    print(f"  - {env_id}")
```

### Environment Categories

| Category | Environments | Description |
|----------|--------------|-------------|
| Dyadic | TrustDilemma-v0, PartnerHoldUp-v0 | 2-agent partnerships |
| Ecosystem | PlatformEcosystem-v0, DynamicPartnerSelection-v0 | N-agent markets |
| Benchmark | RecoveryRace-v0, SynergySearch-v0 | Algorithm evaluation |
| Case Study | SLCD-v0, RenaultNissan-v0 | Validated real scenarios |
| Extended | CooperativeNegotiation-v0, ReputationMarket-v0 | Advanced mechanics |

### Trying Different Environments

```python
# Try the SLCD (Samsung-Sony) environment
env = coopetition_gym.make("SLCD-v0")
obs, info = env.reset(seed=42)
print(f"SLCD observation shape: {obs.shape}")

# Try an N-agent environment
env = coopetition_gym.make("PlatformEcosystem-v0", n_developers=4)
obs, info = env.reset(seed=42)
print(f"PlatformEcosystem observation shape: {obs.shape}")
print(f"Number of agents: {env.n_agents}")
```

---

## Basic Strategy Implementation

Let's implement a simple tit-for-tat strategy:

```python
def tit_for_tat_episode(env, initial_action=60.0, num_steps=100):
    """Run episode with tit-for-tat strategy."""
    obs, info = env.reset(seed=42)

    # Start with initial cooperation
    my_action = initial_action
    partner_action = initial_action

    total_rewards = np.zeros(2)

    for step in range(num_steps):
        # Agent 0: Tit-for-tat (copy partner's last action)
        # Agent 1: Fixed moderate cooperation
        actions = np.array([my_action, 55.0])

        obs, rewards, terminated, truncated, info = env.step(actions)
        total_rewards += rewards

        # Update my_action to match partner's last action
        my_action = obs[1]  # Partner's last action

        if terminated or truncated:
            break

    return total_rewards, info

env = coopetition_gym.make("TrustDilemma-v0")
rewards, final_info = tit_for_tat_episode(env)
print(f"Tit-for-Tat results:")
print(f"  Total rewards: {rewards}")
print(f"  Final trust: {final_info['mean_trust']:.3f}")
```

---

## Experiment: Cooperation vs. Defection

Run a comparison experiment:

```python
def run_strategy(env_name, strategy_fn, episodes=5, seed=42):
    """Run multiple episodes with a strategy and return average rewards."""
    all_rewards = []

    for ep in range(episodes):
        env = coopetition_gym.make(env_name)
        obs, info = env.reset(seed=seed + ep)

        ep_rewards = np.zeros(2)
        for _ in range(100):
            actions = strategy_fn(obs, info)
            obs, rewards, terminated, truncated, info = env.step(actions)
            ep_rewards += rewards
            if terminated or truncated:
                break

        all_rewards.append(ep_rewards)
        env.close()

    return np.mean(all_rewards, axis=0)

# Define strategies
def always_cooperate(obs, info):
    return np.array([80.0, 80.0])

def always_defect(obs, info):
    return np.array([20.0, 20.0])

def mixed_strategy(obs, info):
    return np.array([50.0, 50.0])

# Compare strategies
print("Strategy Comparison on TrustDilemma-v0:")
print("-" * 45)

for name, fn in [("Always Cooperate", always_cooperate),
                  ("Always Defect", always_defect),
                  ("Mixed (50/50)", mixed_strategy)]:
    rewards = run_strategy("TrustDilemma-v0", fn)
    print(f"{name:20s}: Agent0={rewards[0]:7.1f}, Agent1={rewards[1]:7.1f}")
```

---

## Summary

You've learned:

1. **Creating environments** with `coopetition_gym.make()`
2. **Understanding observations** - trust, reputation, interdependence
3. **Taking actions** - cooperation levels from 0 to endowment
4. **Understanding rewards** - integrated utility framework
5. **Trust dynamics** - the 3:1 negativity bias
6. **Using different APIs** - Gymnasium, PettingZoo Parallel, AEC

---

## Next Steps

- **[Environment Reference](../environments/index.md)** - Explore all 10 environments
- **[SLCD-v0](../environments/slcd.md)** - Try the validated Samsung-Sony case study
- **Training Tutorial** - Train RL agents with Stable-Baselines3

---

## Troubleshooting

**ImportError: No module named 'coopetition_gym'**
- Verify installation with `pip show coopetition-gym`
- Ensure you're in the correct virtual environment

**Observation shape doesn't match expected**
- Different environments have different observation dimensions
- Check the specific environment documentation

**Episode terminates early**
- Trust may have collapsed below threshold
- Check `info['mean_trust']` to monitor trust levels
