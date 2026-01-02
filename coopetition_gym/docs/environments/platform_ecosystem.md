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

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | Markov Game (N+1 players, general-sum); Mean-Field approximation applicable for large N |
| **Cooperation Structure** | Mixed-Motive with hub-spoke topology (platform vs developers, no inter-developer competition) |
| **Observability** | Full (all agents observe complete state) |
| **Communication** | Implicit (through actions only) |
| **Agent Symmetry** | Heterogeneous (1 platform + N homogeneous developers) |
| **Reward Structure** | Mixed with hub-spoke interdependence (developers: D=0.75 on platform; platform: D=0.25 per developer) |
| **Action Space** | Continuous: A_platform=[0,150], A_dev=[0,80] |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T=100 (early termination on ecosystem collapse) |
| **Canonical Comparison** | Multi-agent platform games; cf. Mogul (ICML 2020), Multi-Principal Multi-Agent problems |

---

## Formal Specification

This environment is formalized as an (N+1)-player Markov Game with hub-spoke structure.

### Agents
**N** = {Platform} ∪ {Dev_1, ..., Dev_N} where N = n_developers (default 4)

| Role | Count | Endowment | Baseline | Bargaining α |
|------|-------|-----------|----------|--------------|
| Platform | 1 | 150.0 | 52.5 (35%) | 0.30 |
| Developer | N | 80.0 each | 28.0 (35%) | 0.70/N each |

### State Space
**S** ⊆ ℝ^d where d = (N+1) + 3(N+1)² + 1

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Actions | N+1 | Previous cooperation levels |
| Trust Matrix | (N+1)² | Pairwise trust τ_ij |
| Reputation Matrix | (N+1)² | Reputation damage R_ij |
| Interdependence | (N+1)² | Hub-spoke dependencies D_ij |
| Timestep | 1 | Normalized t/T |

**Dimension formula**: d = (N+1) + 3(N+1)² + 1

| n_developers | Total Agents | State Dim |
|--------------|--------------|-----------|
| 4 | 5 | 81 |
| 8 | 9 | 253 |
| 16 | 17 | 885 |

### Action Space
- **Platform**: **A**_0 = [0, 150] ⊂ ℝ
- **Each Developer**: **A**_i = [0, 80] ⊂ ℝ for i ∈ {1, ..., N}

### Interdependence Matrix (Hub-Spoke Topology)

```
D = | 0.00   0.25   0.25   ...  0.25  |   ← Platform row (depends equally on all devs)
    | 0.75   0.00   0.00   ...  0.00  |   ← Dev 1 (depends heavily on platform)
    | 0.75   0.00   0.00   ...  0.00  |   ← Dev 2
    |  ⋮      ⋮      ⋮     ⋱    ⋮    |
    | 0.75   0.00   0.00   ...  0.00  |   ← Dev N
```

Key properties:
- **Platform→Developers**: D[0,j] = 0.25 for all j>0 (moderate, distributed dependency)
- **Developer→Platform**: D[i,0] = 0.75 for all i>0 (high, concentrated dependency)
- **Developer→Developer**: D[i,j] = 0.00 for i,j>0 (no direct dependencies)

### Transition Dynamics

Trust dynamics follow TR-2 with ecosystem-specific parameters:

**Trust Update**:
```
τ_ij(t+1) = clip(τ_ij(t) + Δτ_ij, 0, Θ_ij)
```

**Critical Ecosystem Metric**:
```
avg_dev_trust = (1/N) · Σᵢ τ[i,0]   (developers' trust in platform)
```

If avg_dev_trust < 0.15, ecosystem collapses (termination).

### Reward Function

**Platform reward**:
```
r_platform = π_platform + 0.25 · Σⱼ π_dev_j
```

**Developer i reward**:
```
r_dev_i = π_dev_i + 0.75 · π_platform
```

where private payoffs use θ = 25.0 and γ = 0.75 (strong network effects).

### Episode Structure

- **Horizon**: T = 100 steps
- **Truncation**: t ≥ T
- **Termination**: avg(τ[1:,0]) < 0.15 (ecosystem death)
- **Discount**: γ = 1.0

### Initial State
- τ_ij(0) = 0.60 (baseline ecosystem trust)
- R_ij(0) = 0.00
- D fixed as hub-spoke matrix above

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
3. Pant, V. & Yu, E. (2025). [Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity](https://arxiv.org/abs/2510.18802). arXiv:2510.18802
