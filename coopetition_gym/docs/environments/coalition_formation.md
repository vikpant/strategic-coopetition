# CoalitionFormation-v0

**Category:** Collective Action Environment (TR-3)
**Agents:** 6 (configurable)
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/collective_action_envs.py`

---

## Overview

CoalitionFormation-v0 implements **dynamic coalition mechanics** with entry/exit dynamics. Agents can be excluded from the coalition if their loyalty falls too low, and excluded agents work independently with reduced payoffs.

This environment tests whether agents can maintain coalition stability while managing free-rider threats through exclusion mechanisms.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | N-player Markov Game with coalition structure |
| **Cooperation Structure** | Coalition-based with exclusion |
| **Observability** | Full |
| **Communication** | Implicit (through actions) |
| **Agent Symmetry** | Symmetric initially, dynamic membership |
| **Reward Structure** | Coalition share or independent work |
| **Action Space** | Continuous: $A_i = [0, 50]$ |
| **State Dynamics** | Stochastic coalition transitions |
| **Horizon** | Finite, T = 150 steps |
| **Early Termination** | Coalition collapse (< min_size) |
| **Canonical Comparison** | Coalition formation games; Greenberg (1994) |

---

## Formal Specification

### Coalition Dynamics

**Coalition Membership:**
- Agents with loyalty below `exit_threshold` risk exclusion
- Excluded agents cannot share team production
- Reentry possible after `reentry_cooldown` steps if loyalty recovers

**Reward Structure:**

For coalition members:
$$r_i = \frac{1}{|C|} \cdot Q(\mathbf{a}_C) - c \cdot a_i + \text{stability bonus}$$

For excluded agents:
$$r_i = 0.5 \times (\text{independent payoff})$$

### Coalition Stability

**Minimum Coalition Size:** Coalition must maintain at least `min_coalition_size` members.

**Exclusion Mechanism:**
```python
if loyalty[i] < exit_threshold:
    if len(coalition) > min_coalition_size:
        exclude(agent_i)
        exclusion_timer[i] = reentry_cooldown
```

**Reentry Mechanism:**
```python
if i in excluded and timer[i] <= 0:
    if loyalty[i] >= 0.4:
        readmit(agent_i)
```

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("CoalitionFormation-v0")

obs, info = env.reset(seed=42)

# Mixed behavior: some cooperate, some free-ride
for step in range(50):
    # 3 cooperators, 3 free-riders
    actions = np.array([40, 40, 40, 5, 5, 5])
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated:
        print("Coalition collapsed!")
        break

print(f"Coalition size: {info['coalition_size']}")
print(f"Excluded agents: {info['excluded_agents']}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 6 | Number of potential members |
| `min_coalition_size` | 3 | Minimum viable coalition |
| `exit_threshold` | 0.15 | Loyalty below which exclusion occurs |
| `reentry_cooldown` | 5 | Steps before excluded agent can rejoin |
| `omega` | 25.0 | Productivity factor |
| `beta` | 0.7 | Returns to scale |
| `max_steps` | 150 | Maximum timesteps |
| `render_mode` | None | Rendering mode |

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Includes actions, trust, reputation, interdependence, loyalty scores, coalition membership indicators.

### Action Space

**Type:** `Box`
**Shape:** `(n_agents,)`
**Dtype:** `float32`
**Range:** `[0.0, 50.0]` for each agent

---

## Metrics and Info

The `info` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `coalition_size` | int | Current coalition members |
| `coalition_members` | list | Indices of active members |
| `excluded_agents` | list | Indices of excluded agents |
| `coalition_stability` | float | Coalition size / total agents |
| `team_output` | float | Production from coalition |
| `mean_loyalty` | float | Average loyalty score |
| `free_rider_count` | int | Low-contribution agents |

---

## Key Dynamics

### Exclusion Effects

| Condition | Effect |
|-----------|--------|
| Loyalty < 0.15 | Risk of exclusion |
| Coalition > min_size | Exclusion occurs |
| Coalition = min_size | Free-riders tolerated |
| Excluded agent | 50% reward penalty |

### Coalition Stability Bonus

Coalition members receive a 10% bonus when coalition is stable:
```python
if len(coalition) >= min_coalition_size:
    modifiers[coalition_members] *= 1.1
```

### Termination Condition

Episode terminates if coalition collapses below minimum:
```python
if len(coalition_members) < min_coalition_size:
    terminated = True
```

---

## Strategic Considerations

### For Coalition Members
- Maintain cooperation to preserve membership
- Coalition stability provides bonus
- Too few members triggers termination

### For Potential Free-Riders
- Exclusion risk if loyalty falls
- Independent work yields 50% less
- Reentry requires loyalty recovery

### Emergent Dynamics
- Core-periphery structure may emerge
- Stable coalitions exclude chronic free-riders
- Minimum viable coalition creates threshold effects

---

## Example: Coalition Dynamics Tracking

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("CoalitionFormation-v0", n_agents=6)
obs, info = env.reset(seed=42)

coalition_history = []

for step in range(150):
    # Heterogeneous efforts
    actions = np.array([45, 40, 35, 30, 15, 10])
    obs, rewards, terminated, truncated, info = env.step(actions)

    coalition_history.append(info['coalition_size'])

    if terminated:
        print(f"Terminated at step {step}: coalition collapsed")
        break
    if truncated:
        break

print(f"Initial coalition: 6")
print(f"Final coalition: {coalition_history[-1]}")
print(f"Excluded agents: {info['excluded_agents']}")
```

---

## Research Applications

CoalitionFormation-v0 is suitable for studying:

- **Coalition Stability**: Conditions for sustainable coalitions
- **Exclusion Mechanisms**: Effectiveness of exclusion threats
- **Core-Periphery Dynamics**: Emergence of stable cores
- **Threshold Effects**: Minimum viable coalition behavior
- **Reentry Dynamics**: Recovery and rehabilitation

---

## Related Environments

- [TeamProduction-v0](team_production.md): No coalition dynamics
- [LoyaltyTeam-v0](loyalty_team.md): Fixed membership
- [ApacheProject-v0](apache_project.md): Phase-based dynamics

---

## References

1. Pant, V. & Yu, E. (2026). [Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty](https://arxiv.org/abs/2601.16237). arXiv:2601.16237
2. Ray, D. (2007). A Game-Theoretic Perspective on Coalition Formation. Oxford University Press.
3. Greenberg, J. (1994). Coalition Structures. Handbook of Game Theory, Vol. 2.
