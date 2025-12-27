# ReputationMarket-v0

**Category:** Extended Environment
**Agents:** N (configurable)
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/extended_envs.py`

---

## Overview

ReputationMarket-v0 models an **N-agent competitive market** where public reputation scores directly affect agent rewards through a tiered bonus system. High-reputation agents receive premium bonuses while low-reputation agents face penalties.

This environment tests:
1. **Reputation as strategic asset**: Long-term investment in standing
2. **Market tier dynamics**: Stratification effects
3. **Reputation competition**: Positional goods
4. **Equilibrium in reputation games**: Stable market states

---

## Game-Theoretic Background

### Reputation Markets

Real-world examples:
- **Freelance platforms**: Star ratings affect job access
- **Credit markets**: Credit scores determine rates
- **Professional services**: Reputation affects pricing power
- **Academic markets**: Citations affect opportunities

### The Tier System

Markets often feature discrete tiers:
- **Premium tier**: Best opportunities, highest margins
- **Standard tier**: Normal market conditions
- **Probation tier**: Limited access, lower returns
- **Excluded tier**: Severely restricted

### Strategic Implications

Agents must balance:
- **Short-term returns**: Defection yields immediate gains
- **Reputation investment**: Cooperation builds standing
- **Tier thresholds**: Incentives concentrate near boundaries

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment with 5 agents
env = coopetition_gym.make("ReputationMarket-v0", n_agents=5)

obs, info = env.reset(seed=42)

for step in range(100):
    # All agents choose cooperation levels
    actions = np.random.uniform(40, 70, size=5)
    obs, rewards, terminated, truncated, info = env.step(actions)

print(f"Reputation ranking: {info['reputation_ranking']}")
print(f"Agent tiers: {info['agent_tiers']}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 5 | Number of market participants |
| `max_steps` | 100 | Maximum timesteps |
| `reputation_visibility` | 1.0 | Observation noise (1.0 = perfect) |
| `tier_enabled` | True | Whether tiers affect rewards |
| `render_mode` | None | Rendering mode |

---

## Reputation Tier System

### Tier Definitions

| Tier | Reputation Threshold | Reward Multiplier |
|------|---------------------|-------------------|
| Premium | ≥ 0.80 | 1.30× (30% bonus) |
| Standard | ≥ 0.50 | 1.00× (no change) |
| Probation | ≥ 0.25 | 0.70× (30% penalty) |
| Excluded | < 0.25 | 0.40× (60% penalty) |

### Tier Transitions

```python
# Reputation update
coop_score = action / endowment  # [0, 1]
reputation = reputation + 0.1 * (coop_score - 0.5)
reputation = np.clip(reputation, 0, 1)
```

Moving up/down tiers:
- Sustained cooperation → reputation rises → tier promotion
- Sustained defection → reputation falls → tier demotion

### Tier Bonuses Applied

```python
base_reward = compute_integrated_utility(...)
tier = get_tier(reputation)
multiplier = tier_multipliers[tier]
final_reward = base_reward * multiplier
```

---

## Observation Space

### Extended Observation

| Component | Shape | Description |
|-----------|-------|-------------|
| Standard | N + 3N² + 1 | Actions, trust, rep, interdep, step |
| Public Reputations | N | Visible reputation scores |

**Total dimension**: Base + N

### Observation Noise

If `reputation_visibility < 1.0`:

```python
noise_std = 1 - reputation_visibility
observed_rep = true_rep + np.random.normal(0, noise_std)
```

This models imperfect reputation observation.

---

## Agent Configuration

### Endowments

All agents have equal endowment:
- **Endowment**: 100.0 for each agent

### Interdependence

Fully connected market:
```
D[i,j] = 0.35 for all i ≠ j
D[i,i] = 0.00
```

---

## Trust Dynamics

### Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Trust Building Rate | λ⁺ | 0.10 | Standard building |
| Trust Erosion Rate | λ⁻ | 0.30 | Standard erosion |
| Reputation Damage | μ_R | 0.55 | Moderate damage |
| Reputation Decay | δ_R | 0.015 | Slow forgetting |
| Interdependence Amp. | ξ | 0.45 | Moderate amplification |
| Signal Sensitivity | κ | 1.0 | Standard sensitivity |
| Initial Trust | τ₀ | 0.50 | Neutral start |

---

## Value Function

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| θ | 18.0 | Moderate logarithmic scale |
| γ | 0.50 | Moderate complementarity |

---

## Metrics and Info

The `info` dictionary includes:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `public_reputations` | ndarray | All agents' reputations |
| `reputation_ranking` | list | Agents sorted by reputation |
| `agent_tiers` | dict | Tier assignment per agent |
| `mean_reputation` | float | Market average |
| `reputation_inequality` | float | Standard deviation |
| `tier_distribution` | dict | Count per tier |

---

## Strategic Analysis

### Tier Threshold Effects

Agents near tier boundaries face strong incentives:

**Near Premium threshold (0.80):**
- Small reputation gain → 30% bonus
- Worth investing heavily in cooperation

**Near Probation threshold (0.25):**
- Small reputation loss → move from 70% to 40%
- Strong incentive to avoid this boundary

### Equilibrium Dynamics

**High-reputation equilibrium:**
- All agents cooperate highly
- All stay in Premium tier
- Stable if no one defects

**Low-reputation equilibrium:**
- All agents defect
- All in Excluded tier
- Stable but suboptimal

**Stratified equilibrium:**
- Some Premium, some Standard, some lower
- Competition for limited Premium slots
- Persistent inequality

### Competition Effects

With limited Premium slots (reputation is relative):
- Agent A improving → may push Agent B down
- Zero-sum dynamics near tier boundaries
- Positional competition

---

## Example: Tier-Aware Strategy

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("ReputationMarket-v0", n_agents=5)
obs, info = env.reset(seed=42)

# I am agent 0
my_reputation_history = []
my_tier_history = []

for step in range(100):
    my_rep = info['public_reputations'][0]
    my_tier = info['agent_tiers'].get(0, 'Standard')

    # Tier-aware strategy
    if my_rep < 0.30:
        # Near Excluded: desperate cooperation
        my_action = 90.0
    elif my_rep < 0.55:
        # Near Probation/Standard boundary
        my_action = 70.0
    elif my_rep < 0.82:
        # Near Standard/Premium boundary
        my_action = 75.0
    else:
        # In Premium: maintain with moderate cooperation
        my_action = 60.0

    # Other agents: random
    other_actions = np.random.uniform(40, 60, size=4)

    actions = np.concatenate([[my_action], other_actions])
    obs, rewards, terminated, truncated, info = env.step(actions)

    my_reputation_history.append(info['public_reputations'][0])
    my_tier_history.append(info['agent_tiers'].get(0, 'Standard'))

# Summary
print(f"Final reputation: {my_reputation_history[-1]:.3f}")
print(f"Final tier: {my_tier_history[-1]}")
print(f"Time in Premium: {my_tier_history.count('Premium')} steps")
```

---

## Research Applications

ReputationMarket-v0 is suitable for studying:

- **Reputation Systems**: Design and incentive effects
- **Market Design**: Tier structures and stratification
- **Positional Competition**: Relative standing games
- **Credit Markets**: Rating-based dynamics
- **Multi-Agent RL**: Learning in competitive environments

---

## Scaling Considerations

### Agent Count

| n_agents | Observation Dim | Tier Competition |
|----------|-----------------|------------------|
| 5 | ~105 | Moderate |
| 10 | ~310 | High |
| 20 | ~1210 | Very High |

### Tier Dynamics with Scale

With more agents:
- More competition for Premium tier
- Steeper reputation gradients
- More stratification

---

## Related Environments

- [DynamicPartnerSelection-v0](dynamic_partner_selection.md): Reputation without tiers
- [PlatformEcosystem-v0](platform_ecosystem.md): Platform-mediated market
- [CooperativeNegotiation-v0](cooperative_negotiation.md): Explicit contracts

---

## References

1. Shapiro, C. (1983). Premiums for High Quality Products as Returns to Reputations. Quarterly Journal of Economics.
2. Tadelis, S. (1999). What's in a Name? Reputation as a Tradeable Asset. American Economic Review.
3. Pant, V. & Yu, E. (2025). Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics. arXiv:2510.24909
