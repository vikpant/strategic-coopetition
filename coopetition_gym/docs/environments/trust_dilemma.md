# TrustDilemma-v0

**Category:** Dyadic Environment
**Agents:** 2
**Difficulty:** Intermediate
**Source:** `coopetition_gym/envs/dyadic_envs.py`

---

## Overview

TrustDilemma-v0 implements a **continuous iterated Prisoner's Dilemma** where payoffs evolve based on a hidden trust state. Unlike the classic discrete Prisoner's Dilemma, agents choose continuous cooperation levels, and the reward structure is dynamically modulated by the current trust between agents.

This environment tests whether reinforcement learning agents can learn **long-horizon impulse control**—resisting the temptation of short-term defection gains to maintain the trust that enables higher long-term payoffs.

---

## Game-Theoretic Background

### The Trust Dilemma

In many real-world partnerships, the classic Prisoner's Dilemma structure is complicated by:

1. **Continuous choices**: Partners don't simply "cooperate" or "defect"—they choose how much effort, investment, or commitment to contribute
2. **Dynamic payoffs**: The value of cooperation depends on the relationship's current state
3. **Trust sensitivity**: Past behavior affects future opportunities through trust accumulation

TrustDilemma-v0 captures these dynamics by:
- Using continuous action spaces (cooperation levels from 0% to 100% of endowment)
- Modulating rewards by trust level (higher trust amplifies cooperative gains)
- Implementing asymmetric trust dynamics (trust builds slowly but erodes quickly)

### Strategic Implications

**Short-term incentive**: Defecting (low cooperation) captures immediate gains while free-riding on partner's contributions.

**Long-term incentive**: Maintaining cooperation preserves trust, which:
- Amplifies future payoffs from joint value creation
- Prevents the trust ceiling from limiting recovery options
- Sustains access to synergistic benefits

The core challenge is learning that **today's defection constrains tomorrow's possibilities**.

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("TrustDilemma-v0")

# Reset with seed for reproducibility
obs, info = env.reset(seed=42)

# Run episode
done = False
total_rewards = np.zeros(2)

while not done:
    # Both agents choose cooperation levels
    actions = np.array([60.0, 55.0])  # 60% and 55% of endowments
    obs, rewards, terminated, truncated, info = env.step(actions)
    total_rewards += rewards
    done = terminated or truncated

print(f"Episode rewards: {total_rewards}")
print(f"Final trust: {info['mean_trust']:.3f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 100 | Maximum timesteps per episode |
| `render_mode` | None | Rendering mode ("human", "ansi", or None) |

---

## Spaces

### Observation Space

**Type:** `Box`
**Shape:** `(17,)` for 2 agents
**Dtype:** `float32`

| Index | Component | Shape | Description |
|-------|-----------|-------|-------------|
| 0-1 | Actions | (2,) | Last cooperation levels |
| 2-5 | Trust Matrix | (2,2) | Pairwise trust levels [0,1] |
| 6-9 | Reputation Matrix | (2,2) | Pairwise reputation damage [0,1] |
| 10-13 | Interdependence | (2,2) | Structural dependencies |
| 14-16 | Step Info | (3,) | Normalized timestep and metadata |

### Action Space

**Type:** `Box`
**Shape:** `(2,)` for joint actions
**Dtype:** `float32`
**Range:** `[0.0, 100.0]` for each agent

Actions represent the **cooperation level** (investment amount) for each agent. Higher values indicate more cooperation.

---

## Reward Structure

### Integrated Utility

Rewards are computed using the **integrated utility** framework (TR-1):

```
U_i = private_payoff_i + Σ_j (D_ij × private_payoff_j)
```

Where:
- `private_payoff_i = (endowment - action) + f(action) + α_i × synergy`
- `D_ij` is the interdependence weight (how much agent i values agent j's outcomes)
- `synergy` is the collaborative surplus from joint investment

### Trust Modulation

Payoffs are amplified by the current trust level:

```
effective_payoff = base_payoff × (1 + κ × trust_level)
```

This means:
- **High trust (0.8+)**: Cooperation yields amplified returns
- **Low trust (0.2-)**: Even cooperative actions yield diminished returns
- **Trust collapse (<0.05)**: Episode terminates

---

## Trust Dynamics

### Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Trust Building Rate | λ⁺ | 0.15 | Rate of trust increase from cooperation |
| Trust Erosion Rate | λ⁻ | 0.45 | Rate of trust decrease from defection |
| Reputation Damage | μ_R | 0.50 | Damage coefficient from violations |
| Reputation Decay | δ_R | 0.02 | Rate of reputation forgiveness |
| Interdependence Amp. | ξ | 0.60 | Amplification from dependencies |
| Signal Sensitivity | κ | 1.5 | Action-to-signal conversion |
| Initial Trust | τ₀ | 0.50 | Starting trust level |

### Update Mechanism

Trust updates follow TR-2 dynamics:

```python
# Compute signal from actions
signal = (action - baseline) / baseline  # Positive = cooperative

# Asymmetric update
if signal > 0:
    delta_trust = λ⁺ × signal × (1 - trust)  # Bounded by ceiling
else:
    delta_trust = λ⁻ × signal × trust  # Faster erosion

# Apply with reputation ceiling
trust = min(trust + delta_trust, 1 - reputation_damage)
```

### Key Properties

1. **Negativity Bias (3:1)**: Trust erodes 3× faster than it builds
2. **Trust Ceiling**: Reputation damage creates a permanent ceiling on trust recovery
3. **Hysteresis**: Once trust is damaged, full recovery becomes impossible

---

## Value Function

### Logarithmic Specification

TrustDilemma-v0 uses logarithmic value creation:

```
V(a₁, a₂) = θ × ln(a₁ + a₂) × (1 + γ × complementarity)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| θ | 20.0 | Logarithmic scale factor |
| γ | 0.70 | Complementarity coefficient |

### Complementarity Effect

When both agents cooperate highly, the complementarity bonus amplifies returns:

```
complementarity = min(a₁/e₁, a₂/e₂)  # Bottleneck by lower cooperator
bonus = γ × complementarity
```

This creates strong incentives for **mutual high cooperation**.

---

## Episode Dynamics

### Termination Conditions

The episode ends when:

1. **Truncation**: Maximum steps (100) reached
2. **Termination**: Mean trust falls below 0.05 (trust collapse)

### Typical Trajectories

**Mutual Cooperation:**
- Trust rises toward ceiling (~0.8-0.9)
- Rewards increase over time
- Episode completes at max steps

**Mutual Defection:**
- Trust declines rapidly
- Rewards remain low and flat
- May terminate early from trust collapse

**Mixed Strategy:**
- Trust oscillates based on action patterns
- Rewards fluctuate with trust level
- Vulnerable to defection spirals

---

## Metrics and Info

The `info` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `mean_trust` | float | Average trust level |
| `mean_reputation_damage` | float | Average reputation damage |
| `total_value` | float | Total value created this step |
| `mean_cooperation` | float | Mean cooperation level |
| `cooperation_rate` | float | Cooperation as % of endowments |
| `trust_matrix` | ndarray | Full trust matrix (2×2) |
| `reputation_matrix` | ndarray | Full reputation matrix (2×2) |

---

## Example: Tit-for-Tat Strategy

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

# Tit-for-Tat: Start cooperative, then mirror partner
my_action = 60.0  # Start with 60% cooperation
partner_last_action = 60.0

for step in range(100):
    # Tit-for-Tat: match partner's last action
    actions = np.array([my_action, partner_last_action])
    obs, rewards, terminated, truncated, info = env.step(actions)

    # Update for next round (mirror partner)
    partner_last_action = obs[1]  # Partner's last action
    my_action = partner_last_action

    if terminated or truncated:
        break

print(f"Final trust: {info['mean_trust']:.3f}")
```

---

## Research Applications

TrustDilemma-v0 is suitable for studying:

- **Cooperation Emergence**: How agents learn to sustain cooperation
- **Trust Building**: Strategies for trust recovery after defection
- **Impulse Control**: Resisting short-term temptation for long-term gain
- **Reciprocity**: Tit-for-tat and related strategies in continuous settings
- **MARL Algorithms**: Comparing PPO, MAPPO, and other algorithms

---

## Related Environments

- [PartnerHoldUp-v0](partner_holdup.md): Adds asymmetric power dynamics
- [RecoveryRace-v0](recovery_race.md): Focuses on trust recovery
- [SLCD-v0](slcd.md): Validated real-world case study

---

## References

1. Pant, V. & Yu, E. (2025). Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity. arXiv:2510.18802
2. Pant, V. & Yu, E. (2025). Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics. arXiv:2510.24909
3. Axelrod, R. (1984). The Evolution of Cooperation. Basic Books.
