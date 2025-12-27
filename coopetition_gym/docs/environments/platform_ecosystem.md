# PlatformEcosystem-v0

**Category:** Ecosystem Environment
**Agents:** 1 + N (Platform + Developers)
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/ecosystem_envs.py`

---

## Overview

PlatformEcosystem-v0 models a **platform economy** with one central platform (Agent 0) and N developer agents (Agents 1-N). This environment captures the dynamics of app stores, online marketplaces, cloud platforms, and other multi-sided markets.

The platform must balance **short-term revenue extraction** against **long-term ecosystem health**. Developers must decide how much to invest in the platform given its policies and the behavior of other developers.

---

## Game-Theoretic Background

### Platform Economics

Multi-sided platforms create value by:
1. **Network Effects**: More developers attract more users, increasing value
2. **Complementarity**: Developer contributions complement the platform's infrastructure
3. **Ecosystem Health**: Trust between platform and developers enables investment

### The Platform's Dilemma

**Short-term incentive**: Extract maximum value (high fees, restrictive policies)

**Long-term incentive**: Maintain developer trust and participation to:
- Sustain network effects
- Encourage quality investments
- Prevent developer exodus

### Developer Dynamics

Developers face:
- **Platform dependency**: High switching costs once invested
- **Collective action problem**: Individual defection may not trigger platform response
- **Trust fragility**: Platform abuse can trigger coordinated exit

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment with 4 developers (default)
env = coopetition_gym.make("PlatformEcosystem-v0")

# Or customize number of developers
env = coopetition_gym.make("PlatformEcosystem-v0", n_developers=6)

obs, info = env.reset(seed=42)

# Run episode
for step in range(100):
    # Platform invests 90 out of 150 (60%)
    platform_action = 90.0

    # Developers each invest 50 out of 80 (62.5%)
    developer_actions = [50.0] * env.n_agents - 1

    actions = np.array([platform_action] + developer_actions)
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated:
        print(f"Ecosystem collapsed at step {step}")
        break

print(f"Platform reward: {rewards[0]:.1f}")
print(f"Mean developer reward: {np.mean(rewards[1:]):.1f}")
print(f"Developer trust in platform: {info['developer_trust_in_platform']:.2f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_developers` | 4 | Number of developer agents |
| `max_steps` | 100 | Maximum timesteps per episode |
| `render_mode` | None | Rendering mode |

---

## Agent Configuration

### Endowments

| Agent | Role | Endowment | Description |
|-------|------|-----------|-------------|
| 0 | Platform | 150.0 | Large infrastructure budget |
| 1-N | Developers | 80.0 each | Individual development capacity |

### Bargaining Shares (Alpha)

| Agent | Alpha | Description |
|-------|-------|-------------|
| Platform | 0.30 | Platform captures 30% of ecosystem surplus |
| Each Developer | 0.70/N | Remaining 70% split among developers |

**Example with 4 developers:**
- Platform: 30%
- Each developer: 17.5% (70% / 4)

---

## Interdependence Structure

### Hub-Spoke Topology

The platform acts as a **hub** with developers as **spokes**:

```
              Developer 1
                   |
Developer 4 ---- Platform ---- Developer 2
                   |
              Developer 3
```

### Dependency Matrix

Created using `create_hub_spoke_interdependence()`:

```
D = [[ 0.00,  0.25,  0.25,  0.25,  0.25 ],   # Platform's row
     [ 0.75,  0.00,  0.00,  0.00,  0.00 ],   # Dev 1's row
     [ 0.75,  0.00,  0.00,  0.00,  0.00 ],   # Dev 2's row
     [ 0.75,  0.00,  0.00,  0.00,  0.00 ],   # Dev 3's row
     [ 0.75,  0.00,  0.00,  0.00,  0.00 ]]   # Dev 4's row
```

**Interpretation:**
- **D[0,j] = 0.25**: Platform depends moderately on each developer
- **D[i,0] = 0.75**: Developers depend heavily on platform
- **D[i,j] = 0.00** (i,j > 0): Developers don't directly depend on each other

---

## Trust Dynamics

### Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Trust Building Rate | λ⁺ | 0.08 | Slower institutional trust building |
| Trust Erosion Rate | λ⁻ | 0.25 | Moderate erosion |
| Reputation Damage | μ_R | 0.45 | Moderate damage |
| Reputation Decay | δ_R | 0.02 | Standard decay |
| Interdependence Amp. | ξ | 0.40 | Lower than dyadic (more actors) |
| Signal Sensitivity | κ | 1.0 | Standard sensitivity |
| Initial Trust | τ₀ | 0.60 | Baseline platform trust |

### Critical Trust Metric

The most important trust metric is **average developer trust in platform**:

```python
developer_trust = mean(trust_matrix[1:, 0])  # All developers' trust in platform
```

If this falls below 0.15, the ecosystem "dies" (episode terminates).

---

## Termination Conditions

### Normal Truncation

Episode ends at `max_steps` (100) if ecosystem persists.

### Ecosystem Death

**Critical condition:** If average developer trust in platform falls below 0.15:

```python
if mean(trust_matrix[1:, 0]) < 0.15:
    terminated = True
    # Ecosystem collapse - developers abandon platform
```

This represents:
- Mass developer exodus
- Platform becoming unviable
- Network effects reversing

---

## Value Function

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| θ | 25.0 | Higher scale for platform (larger transactions) |
| γ | 0.75 | Strong complementarity (network effects) |

### Network Effects

The high complementarity (γ = 0.75) means:
- Value grows superlinearly with total investment
- Platform and developers benefit from mutual cooperation
- Defection by any party reduces ecosystem value

---

## Reward Structure

### Platform Rewards

```
Platform_payoff = kept_resources + f(platform_action) + 0.30 × synergy + Σ(0.25 × developer_payoffs)
```

The platform benefits from:
1. Its own investment returns
2. 30% of ecosystem synergy
3. Moderate weight on developer success

### Developer Rewards

```
Developer_i_payoff = kept_resources + f(action_i) + (0.70/N) × synergy + 0.75 × platform_payoff
```

Developers benefit from:
1. Their own investment returns
2. Share of ecosystem synergy
3. Strong weight on platform success (75%)

---

## Metrics and Info

The `info` dictionary includes:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `platform_investment` | float | Platform's action |
| `mean_developer_investment` | float | Average developer action |
| `developer_investment_std` | float | Variation among developers |
| `developer_trust_in_platform` | float | Critical ecosystem health metric |
| `platform_trust_in_developers` | float | Platform's view of developers |
| `total_ecosystem_value` | float | Total value created |

---

## Strategic Analysis

### Platform Strategies

**Extractive Strategy:**
- Invest minimally in infrastructure
- Capture maximum surplus
- Risk: Developer exit if trust falls

**Growth Strategy:**
- Invest heavily in infrastructure
- Build developer trust
- Sacrifice short-term for ecosystem growth

**Balanced Strategy:**
- Moderate investment matching developers
- Maintain sustainable trust levels
- Optimize for long-term value

### Developer Strategies

**High Investment:**
- Commit heavily to platform
- Benefit from network effects
- Risk: Platform exploitation

**Defensive Investment:**
- Invest minimally above threshold
- Reduce platform dependency
- Maintain exit option

**Coordinated Response:**
- Mirror platform's behavior
- Punish exploitation collectively
- Requires implicit coordination

---

## Example: Ecosystem Dynamics

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("PlatformEcosystem-v0", n_developers=4)
obs, info = env.reset(seed=42)

# Track ecosystem health
trust_history = []
value_history = []

for step in range(100):
    # Platform: Responsive to developer trust
    dev_trust = info.get('developer_trust_in_platform', 0.6)

    # High trust -> high investment; Low trust -> defensive
    platform_action = 150.0 * min(0.8, dev_trust + 0.2)

    # Developers: Tit-for-tat with platform
    platform_last = obs[0] if step > 0 else 75.0
    dev_action = 80.0 * (platform_last / 150.0)

    actions = np.array([platform_action] + [dev_action] * 4)
    obs, rewards, terminated, truncated, info = env.step(actions)

    trust_history.append(info['developer_trust_in_platform'])
    value_history.append(info['total_ecosystem_value'])

    if terminated:
        print(f"Ecosystem collapsed at step {step}")
        break

print(f"Final developer trust: {trust_history[-1]:.3f}")
print(f"Average ecosystem value: {np.mean(value_history):.1f}")
```

---

## Research Applications

PlatformEcosystem-v0 is suitable for studying:

- **Platform Economics**: Multi-sided market dynamics
- **Mechanism Design**: Incentive structures for ecosystems
- **Network Effects**: Value creation in platform economies
- **Credit Assignment**: MARL challenge with shared outcomes
- **Collective Action**: Coordinated responses to platform behavior

---

## Scaling Considerations

### Agent Count

| n_developers | Total Agents | Observation Dim | Complexity |
|--------------|--------------|-----------------|------------|
| 4 | 5 | 81 | Moderate |
| 8 | 9 | 253 | High |
| 16 | 17 | 885 | Very High |

### Recommendations

- **Algorithm testing**: Use n_developers=4 (default)
- **Scalability studies**: Increase progressively
- **Production**: Consider attention-based policies for large N

---

## Related Environments

- [DynamicPartnerSelection-v0](dynamic_partner_selection.md): Peer-to-peer dynamics
- [ReputationMarket-v0](reputation_market.md): Market with reputation tiers
- [PartnerHoldUp-v0](partner_holdup.md): Dyadic asymmetric power

---

## References

1. Rochet, J.C. & Tirole, J. (2003). Platform Competition in Two-Sided Markets. JEEA.
2. Parker, G.G. & Van Alstyne, M.W. (2005). Two-Sided Network Effects. Management Science.
3. Pant, V. & Yu, E. (2025). Strategic Value Creation in Coopetitive Partnerships. arXiv:2510.18802
