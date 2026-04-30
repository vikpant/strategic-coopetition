# TeamProduction-v0

**Category:** Collective Action Environment (TR-3)
**Agents:** 4 (configurable)
**Difficulty:** Intermediate
**Source:** `coopetition_gym/envs/collective_action_envs.py`

---

## Overview

TeamProduction-v0 implements a **team production game** where agents contribute effort to a shared production function. This is the baseline TR-3 environment demonstrating free-rider dynamics without loyalty mechanisms.

The environment tests whether reinforcement learning agents can overcome the **free-rider temptation**,contributing less effort while benefiting from teammates' contributions.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | N-player Markov Game (general-sum) |
| **Cooperation Structure** | Mixed-Motive (team production vs. individual cost) |
| **Observability** | Full (all state variables observable) |
| **Communication** | Implicit (through actions only) |
| **Agent Symmetry** | Symmetric (identical capabilities) |
| **Reward Structure** | Team share minus individual cost |
| **Action Space** | Continuous, bounded: $A_i = [0, 50]$ |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T = 100 steps |
| **Canonical Comparison** | Team production games; Holmström (1982); Alchian & Demsetz (1972) |

---

## Formal Specification

### Mathematical Framework (TR-3)

**Team Production Function:**
$$Q(\mathbf{a}) = \omega \cdot \left(\sum_{i=1}^{n} a_i\right)^\beta$$

Where:
- $\omega = 25.0$ is the productivity factor
- $\beta = 0.7$ captures diminishing returns to scale
- $a_i$ is agent $i$'s effort contribution

**Base Payoff:**
$$\pi_i^{team} = \frac{1}{n} \cdot Q(\mathbf{a}) - c \cdot a_i$$

Where:
- Each agent receives equal share of output ($1/n$)
- $c = 1.0$ is the effort cost coefficient

**Nash Equilibrium (Free-Riding):**
$$a^* = \left(\frac{\omega\beta}{nc}\right)^{\frac{1}{1-\beta}}$$

**Social Optimum:**
$$a^{opt} = \left(\frac{\omega\beta}{c}\right)^{\frac{1}{1-\beta}}$$

### State Space

**S** ⊆ ℝ^d with components:

| Component | Symbol | Description |
|-----------|--------|-------------|
| Actions | a | Previous effort levels |
| Trust Matrix | τ | Pairwise trust (from TR-2) |
| Reputation | R | Accumulated reputation damage |
| Interdependence | D | Structural dependencies |
| Loyalty Scores | θ | Per-agent loyalty (initialized neutral) |

### Action Space

For each agent $i$:
$$A_i = [0, a_{max}] = [0, 50] \subset \mathbb{R}$$

Actions represent **effort contribution** to team production.

> **Uniaxial Treatment**: This environment uses the single-dimension action space characteristic of Coopetition-Gym v1.x. The free-rider dynamic emerges through divergence between individual and collective incentives rather than explicit competitive actions.

### Reward Function

In the baseline TeamProduction-v0, rewards are pure team payoffs without loyalty:

$$r_i = \frac{1}{n} \cdot Q(\mathbf{a}) - c \cdot a_i$$

With light penalty for severe free-riding (cooperation rate < 20%).

---

## Game-Theoretic Background

### The Free-Rider Problem

Team production creates a classic collective action problem: 1. **Individual incentive**: Contribute less effort (save cost) while sharing output
2. **Collective outcome**: If all free-ride, team output is minimal
3. **Nash equilibrium**: Suboptimal effort level $a^* < a^{opt}$

### Strategic Implications

**Free-Rider Equilibrium:**
- Each agent contributes $a^* \approx 6.8$ (with n=4, default params)
- Team output is significantly below potential

**Social Optimum:**
- Each agent contributes $a^{opt} \approx 18.4$
- Requires coordination or loyalty mechanisms

**Price of Anarchy:**
$$PoA = \frac{\text{Social Welfare at Optimum}}{\text{Social Welfare at Nash}} \approx 2.5$$

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("TeamProduction-v0")

# Reset
obs, info = env.reset(seed=42)

# Check Nash equilibrium
print(f"Nash effort: {info['nash_equilibrium']:.2f}")
print(f"Social optimum: {info['social_optimum']:.2f}")

# Run episode with Nash equilibrium strategy
for step in range(100): nash_effort = info['nash_equilibrium']
    actions = np.array([nash_effort] * 4)
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated: break

print(f"Team output at Nash: {info['team_output']:.2f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 4 | Number of team members |
| `omega` | 25.0 | Productivity factor |
| `beta` | 0.7 | Returns to scale |
| `c` | 1.0 | Effort cost coefficient |
| `max_steps` | 100 | Maximum timesteps |
| `render_mode` | None | Rendering mode |

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Includes actions, trust matrix, reputation, interdependence, and step info.

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
| `team_output` | float | Q(a) = ω·(Σaᵢ)^β |
| `nash_equilibrium` | float | Theoretical Nash effort |
| `social_optimum` | float | Theoretical optimal effort |
| `mean_loyalty` | float | Average loyalty score |
| `free_rider_count` | int | Agents below threshold |
| `efficiency_ratio` | float | Actual/optimal effort ratio |
| `mean_trust` | float | Average trust level |

---

## Key Dynamics

### Free-Rider Detection

Agents with cooperation rate below 30% are flagged as free-riders:

```python
free_riders = [i for i in range(n) if actions[i]/endowments[i] < 0.3]
```

### Baseline Behavior

Without loyalty mechanisms:
- Rational agents converge toward Nash equilibrium
- Team output is suboptimal
- No intrinsic motivation to cooperate beyond self-interest

This establishes the baseline for comparing with LoyaltyTeam-v0.

---

## Research Applications

TeamProduction-v0 is suitable for studying:

- **Free-Rider Problem**: Classic collective action dynamics
- **Nash Equilibrium Convergence**: Do agents learn the equilibrium?
- **Baseline Comparison**: Reference point for loyalty mechanisms
- **Team Incentive Design**: Effects of different production functions

---

## Related Environments

- [LoyaltyTeam-v0](loyalty_team.md): Adds TR-3 loyalty mechanisms
- [CoalitionFormation-v0](coalition_formation.md): Dynamic membership
- [PublicGoods-v0](public_goods.md): Classic public goods variant

---

## References

1. Pant, V. & Yu, E. (2026). [Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty](https://arxiv.org/abs/2601.16237). arXiv:2601.16237
2. Holmström, B. (1982). Moral Hazard in Teams. Bell Journal of Economics.
3. Alchian, A. & Demsetz, H. (1972). Production, Information Costs, and Economic Organization. American Economic Review.
