# LoyaltyTeam-v0

**Category:** Collective Action Environment (TR-3)
**Agents:** 4 (configurable)
**Difficulty:** Intermediate-Advanced
**Source:** `coopetition_gym/envs/collective_action_envs.py`

---

## Overview

LoyaltyTeam-v0 implements team production with **full TR-3 loyalty mechanisms**. Unlike the baseline TeamProduction-v0, agents with high loyalty receive bonuses proportional to teammate welfare, creating positive-sum dynamics that can sustain cooperation above the Nash equilibrium.

This environment tests whether loyalty dynamics can overcome the free-rider problem and sustain cooperation in team settings.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | N-player Markov Game (general-sum) |
| **Cooperation Structure** | Mixed-Motive with loyalty amplification |
| **Observability** | Full |
| **Communication** | Implicit (through actions) |
| **Agent Symmetry** | Symmetric (capabilities), asymmetric (loyalty states) |
| **Reward Structure** | Team share + loyalty modifier |
| **Action Space** | Continuous: $A_i = [0, 50]$ |
| **State Dynamics** | Deterministic with loyalty evolution |
| **Horizon** | Finite, T = 100 steps |
| **Canonical Comparison** | Behavioral team theory; TR-3 loyalty model |

---

## Formal Specification

### Mathematical Framework (TR-3)

**Team Production Function:**
$$Q(\mathbf{a}) = \omega \cdot \left(\sum_{i=1}^{n} a_i\right)^\beta$$

**Loyalty Modifier:**
$$L_i = \theta_i \cdot \left[\phi_B \cdot \bar{\pi}_{-i} + \phi_C \cdot c \cdot a_i\right]$$

Where:
- $\theta_i \in [0,1]$ is agent $i$'s loyalty score
- $\phi_B = 0.8$ is the loyalty benefit strength (care for teammates)
- $\phi_C = 0.3$ is the cost tolerance strength (reduced burden from own effort)
- $\bar{\pi}_{-i}$ is the average payoff of teammates

**Loyalty-Augmented Utility:**
$$U_i = \pi_i^{team} + L_i$$

### Loyalty Evolution

Loyalty scores evolve based on cooperation:

```python
if cooperation_rate >= 0.5:
    loyalty += 0.02 * cooperation_rate  # Build slowly
else:
    loyalty -= 0.05 * (0.5 - cooperation_rate)  # Erode faster
```

### Key Insight

High-loyalty agents ($\theta_i \approx 1$) receive substantial bonuses when:
1. Teammates are doing well ($\phi_B \cdot \bar{\pi}_{-i}$)
2. They contribute high effort themselves ($\phi_C \cdot c \cdot a_i$)

This creates intrinsic motivation for sustained cooperation.

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment with default loyalty parameters
env = coopetition_gym.make("LoyaltyTeam-v0")

# Or customize loyalty strengths
env = coopetition_gym.make(
    "LoyaltyTeam-v0",
    phi_B=0.9,  # Strong teammate-welfare bonus
    phi_C=0.4,  # Moderate cost tolerance
)

obs, info = env.reset(seed=42)

# Sustained cooperation builds loyalty
for step in range(50):
    # High cooperation
    actions = np.array([40.0, 40.0, 40.0, 40.0])
    obs, rewards, terminated, truncated, info = env.step(actions)

print(f"Mean loyalty after 50 steps: {info['mean_loyalty']:.2f}")
print(f"Loyalty lift over Nash: {info['loyalty_lift']:.2f}x")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 4 | Number of team members |
| `omega` | 25.0 | Productivity factor |
| `beta` | 0.7 | Returns to scale |
| `c` | 1.0 | Effort cost coefficient |
| `phi_B` | 0.8 | Loyalty benefit strength |
| `phi_C` | 0.3 | Cost tolerance strength |
| `max_steps` | 100 | Maximum timesteps |
| `render_mode` | None | Rendering mode |

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Includes actions, trust matrix, reputation, interdependence, loyalty scores, and step info.

### Action Space

**Type:** `Box`
**Shape:** `(n_agents,)`
**Dtype:** `float32`
**Range:** `[0.0, 50.0]` for each agent

> **Uniaxial Treatment**: This environment uses the single-dimension action space characteristic of Coopetition-Gym v1.x. Competition emerges through the free-rider dynamic modulated by loyalty mechanisms rather than explicit competitive actions.

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
| `loyalty_scores` | list | Per-agent loyalty [0,1] |
| `loyalty_lift` | float | Output ratio vs. Nash baseline |
| `team_cohesion` | float | Weighted loyalty metric |
| `free_rider_count` | int | Agents below threshold |
| `efficiency_ratio` | float | Actual/optimal effort ratio |

---

## Key Dynamics

### Loyalty Building

Sustained cooperation builds loyalty over time:

| Behavior | Loyalty Change |
|----------|----------------|
| High cooperation (>50%) | +0.02 × rate |
| Low cooperation (<50%) | -0.05 × deficit |
| Sustained high | Approaches 1.0 |
| Sustained low | Approaches 0.0 |

### Loyalty Lift

The key metric measuring TR-3 effectiveness:

$$\text{Loyalty Lift} = \frac{\text{Actual Team Output}}{\text{Nash Equilibrium Output}}$$

With effective loyalty mechanisms, teams can achieve lift > 2.0.

### Phase Dynamics

1. **Early phase (steps 1-20)**: Loyalty neutral (~0.5), behavior determines trajectory
2. **Middle phase (steps 20-60)**: Loyalty diverges based on cooperation patterns
3. **Late phase (steps 60-100)**: Loyalty stable, determines final outcomes

---

## Comparison with TeamProduction-v0

| Aspect | TeamProduction-v0 | LoyaltyTeam-v0 |
|--------|-------------------|----------------|
| Loyalty mechanisms | None | Full TR-3 |
| Expected equilibrium | Nash (low) | Above Nash possible |
| Free-rider penalty | Light | Moderate + loyalty erosion |
| Cooperation incentive | External only | Intrinsic (loyalty bonus) |
| Typical output | ~baseline | 1.5-2.5× baseline |

---

## Research Applications

LoyaltyTeam-v0 is suitable for studying:

- **Loyalty Dynamics**: How loyalty builds and erodes
- **Cooperation Sustainability**: Can loyalty sustain above-Nash cooperation?
- **Mechanism Design**: Optimal φ_B and φ_C settings
- **Team Formation**: Which agents develop loyalty?
- **Free-Rider Response**: Do loyal agents punish free-riders?

---

## Example: Loyalty Evolution Tracking

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("LoyaltyTeam-v0")
obs, info = env.reset(seed=42)

loyalty_history = []

for step in range(100):
    # Cooperative strategy
    actions = np.array([35.0, 35.0, 35.0, 35.0])
    obs, rewards, terminated, truncated, info = env.step(actions)

    loyalty_history.append(info['mean_loyalty'])

    if terminated or truncated:
        break

# Loyalty should increase over time with cooperation
print(f"Initial loyalty: {loyalty_history[0]:.2f}")
print(f"Final loyalty: {loyalty_history[-1]:.2f}")
print(f"Loyalty lift: {info['loyalty_lift']:.2f}x")
```

---

## Related Environments

- [TeamProduction-v0](team_production.md): Baseline without loyalty
- [CoalitionFormation-v0](coalition_formation.md): Dynamic membership
- [ApacheProject-v0](apache_project.md): Validated case study

---

## References

1. Pant, V. & Yu, E. (2026). [Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty](https://arxiv.org/abs/2601.16237). arXiv:2601.16237
2. Akerlof, G. & Kranton, R. (2010). Identity Economics. Princeton University Press.
3. Kandel, E. & Lazear, E. (1992). Peer Pressure and Partnerships. Journal of Political Economy.
