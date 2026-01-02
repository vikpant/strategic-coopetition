# PartnerHoldUp-v0

**Category:** Dyadic Environment
**Agents:** 2 (Strong + Weak)
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/dyadic_envs.py`

---

## Overview

PartnerHoldUp-v0 models an **asymmetric vertical relationship** where one agent (the "strong" partner) holds structural power over another (the "weak" partner). This environment captures the dynamics of supplier-manufacturer relationships, small-firm-large-firm partnerships, and other scenarios where one party has greater bargaining power or outside options.

The key challenge is understanding how power asymmetry affects cooperation dynamics, and whether the weak partner can develop effective defensive strategies against potential exploitation.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | Markov Game (2-player, general-sum, asymmetric) |
| **Cooperation Structure** | Mixed-Motive with power asymmetry (exploitation risk) |
| **Observability** | Full (all state variables observable to both agents) |
| **Communication** | Implicit (through actions only) |
| **Agent Symmetry** | Asymmetric (different endowments: 120 vs 80, different dependencies) |
| **Reward Structure** | Mixed with asymmetric interdependence (D_strong=0.35, D_weak=0.85) |
| **Action Space** | Continuous, agent-specific: A_0=[0,120], A_1=[0,80] |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T=100 (early termination if weak partner's trust < 0.10) |
| **Canonical Comparison** | Asymmetric Prisoner's Dilemma; cf. principal-agent models, Stackelberg games with endogenous commitment |

---

## Formal Specification

This environment is formalized as a 2-player asymmetric Markov Game **M** = (**N**, **S**, {**A**_i}, **P**, {**R**_i}, T).

### Agents
**N** = {Strong, Weak} with heterogeneous capabilities:

| Agent | Index | Endowment e_i | Baseline b_i | Bargaining α_i |
|-------|-------|---------------|--------------|----------------|
| Strong | 0 | 120.0 | 42.0 (35%) | 0.60 |
| Weak | 1 | 80.0 | 28.0 (35%) | 0.40 |

### State Space
**S** ⊆ ℝ¹⁷ with identical structure to TrustDilemma-v0:

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Actions a | 2 | Previous cooperation levels |
| Trust τ | 4 | Pairwise trust matrix (flattened) |
| Reputation R | 4 | Reputation damage matrix |
| Interdependence D | 4 | Dependency matrix (fixed) |
| Metadata | 3 | Timestep, auxiliary |

### Action Space
Agent-specific continuous action spaces:

- **Strong**: **A**_0 = [0, 120] ⊂ ℝ
- **Weak**: **A**_1 = [0, 80] ⊂ ℝ

### Interdependence Matrix (Critical Asymmetry)

```
D = | 0.00  0.35 |   (Strong depends moderately on Weak)
    | 0.85  0.00 |   (Weak depends heavily on Strong)
```

This creates **vulnerability asymmetry**: Weak's utility is strongly coupled to Strong's behavior, but not vice versa.

### Transition Dynamics

Trust dynamics follow TR-2 with amplified asymmetry:

**Trust Update**:
```
τ_ij(t+1) = clip(τ_ij(t) + Δτ_ij · (1 + ξ · D_ij), 0, Θ_ij)
```

where ξ = 0.70 amplifies trust sensitivity based on dependency.

**Critical Threshold**: τ_weak→strong < 0.10 triggers termination (weak partner exits).

### Reward Function

Integrated utility with asymmetric weights:

```
r_i(s, a) = π_i(a) + Σ_j D_ij · π_j(a)
```

**Strong's reward**: Includes only 35% of Weak's payoff
**Weak's reward**: Includes 85% of Strong's payoff (highly coupled)

Private payoffs:
```
π_Strong = (120 - a_0) + 20·ln(1+a_0) + 0.60·G(a)
π_Weak   = (80 - a_1) + 20·ln(1+a_1) + 0.40·G(a)
```

### Episode Structure

- **Horizon**: T = 100 steps
- **Truncation**: t ≥ T
- **Termination**: τ[1,0] < 0.10 (weak partner exits due to exploitation)
- **Discount**: γ = 1.0

### Initial State
- τ_ij(0) = 0.55 (moderate initial trust)
- R_ij(0) = 0.00
- D fixed as above

---

## Game-Theoretic Background

### The Hold-Up Problem

The classic hold-up problem arises when:

1. **Relationship-specific investments**: One party makes investments that have less value outside the relationship
2. **Incomplete contracts**: Not all contingencies can be specified ex ante
3. **Asymmetric switching costs**: One party can exit more easily than the other

This creates a tension between:
- **Ex ante efficiency**: Both parties benefit from investment
- **Ex post opportunism**: The stronger party may exploit the weaker party's lock-in

### Power Dynamics in PartnerHoldUp-v0

**Agent 0 (Strong Partner):**
- Larger endowment (120 vs. 80)
- Higher bargaining share (60% vs. 40%)
- Low dependency on partner (D = 0.35)
- More outside options (can replace weak partner)

**Agent 1 (Weak Partner):**
- Smaller endowment
- Lower bargaining share
- High dependency on partner (D = 0.85)
- Few alternatives (locked into relationship)

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("PartnerHoldUp-v0")

# Reset
obs, info = env.reset(seed=42)

# Run episode
for step in range(100):
    # Strong partner: 70 out of 120 (58%)
    # Weak partner: 50 out of 80 (62%)
    actions = np.array([70.0, 50.0])
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated:
        print(f"Relationship ended at step {step}")
        print(f"Weak partner exited due to exploitation")
        break

print(f"Strong reward: {rewards[0]:.1f}, Weak reward: {rewards[1]:.1f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 100 | Maximum timesteps per episode |
| `render_mode` | None | Rendering mode |

---

## Agent Configuration

### Endowments

| Agent | Role | Endowment | Description |
|-------|------|-----------|-------------|
| 0 | Strong | 120.0 | Large manufacturer with more resources |
| 1 | Weak | 80.0 | Small supplier with limited capacity |

### Bargaining Shares (Alpha)

| Agent | Alpha | Description |
|-------|-------|-------------|
| 0 | 0.60 | Strong captures 60% of surplus |
| 1 | 0.40 | Weak captures 40% of surplus |

### Baselines (Cooperation Threshold)

| Agent | Baseline | As % of Endowment |
|-------|----------|-------------------|
| 0 | 42.0 | 35% |
| 1 | 28.0 | 35% |

Actions below baseline are considered "defection" and erode trust.

---

## Interdependence Structure

The key asymmetry is in the **dependency matrix**:

```
D = [[ 0.00,  0.35 ],    # Strong's row
     [ 0.85,  0.00 ]]    # Weak's row
```

**Interpretation:**
- **D[0,1] = 0.35**: Strong depends moderately on weak (has alternatives)
- **D[1,0] = 0.85**: Weak depends heavily on strong (locked in)

### Implications

1. **Weak values strong's payoff highly**: Included in weak's integrated utility
2. **Strong values weak's payoff less**: Can afford to exploit
3. **Asymmetric trust dynamics**: Weak's trust in strong is critical
4. **Exit threshold matters**: If weak's trust falls too low, relationship ends

---

## Trust Dynamics

### Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Trust Building Rate | λ⁺ | 0.10 | Slow trust building |
| Trust Erosion Rate | λ⁻ | 0.35 | Faster erosion than TrustDilemma |
| Reputation Damage | μ_R | 0.55 | Moderate-high damage |
| Reputation Decay | δ_R | 0.025 | Slow forgetting |
| Interdependence Amp. | ξ | 0.70 | High amplification (magnifies asymmetry) |
| Signal Sensitivity | κ | 1.2 | Moderate sensitivity |
| Initial Trust | τ₀ | 0.55 | Moderate starting trust |

### Asymmetric Trust Effects

The high interdependence amplification (ξ = 0.70) means:

- **For weak partner**: Trust erosion is amplified due to high dependency
- **For strong partner**: Trust erosion has less impact (lower dependency)

This creates a **vulnerability asymmetry** where the weak partner is more affected by trust dynamics.

---

## Termination Conditions

### Normal Truncation

Episode ends at `max_steps` (100) if relationship persists.

### Relationship Exit

**Critical condition:** If weak agent's trust in strong falls below 0.10, the episode terminates.

```python
if trust_matrix[1, 0] < 0.10:  # Weak's trust in strong
    terminated = True
    # Weak partner "exits" the relationship
```

This represents the weak partner:
- Finding an alternative supplier
- Exiting the market
- Seeking legal recourse
- Refusing further cooperation

---

## Reward Structure

### Integrated Utility with Asymmetry

Rewards follow the standard integrated utility formula, but the asymmetry manifests through:

1. **Bargaining shares**: Strong gets 60% of synergy, weak gets 40%
2. **Interdependence**: Weak's utility includes 85% weight on strong's payoff
3. **Trust modulation**: Both payoffs are affected by trust levels

### Value Function

| Parameter | Value | Description |
|-----------|-------|-------------|
| θ | 20.0 | Logarithmic scale |
| γ | 0.60 | Moderate complementarity |

---

## Strategic Analysis

### Strong Partner's Dilemma

The strong partner faces a choice:

**Exploitation Strategy:**
- Contribute less than fair share
- Capture most of the surplus
- Risk triggering weak partner's exit

**Sustainable Strategy:**
- Maintain moderate cooperation
- Keep weak partner's trust above exit threshold
- Accept lower per-period returns for longevity

### Weak Partner's Options

The weak partner can:

**Defensive Strategy:**
- Maintain minimal viable cooperation
- Reduce relationship-specific investments
- Keep exit option credible

**Trust-Building Strategy:**
- Over-cooperate to build trust cushion
- Invest in relationship to increase strong's dependency
- Signal long-term commitment

### Equilibrium Considerations

In repeated games:
- Strong partner may restrain exploitation to avoid triggering exit
- Weak partner may invest if strong signals long-term orientation
- "Shadow of the future" moderates short-term opportunism

---

## Metrics and Info

The `info` dictionary includes:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `mean_trust` | float | Average trust level |
| `weak_trust_in_strong` | float | Critical metric for exit |
| `strong_trust_in_weak` | float | Less critical |
| `power_asymmetry` | float | Measure of dependency imbalance |
| `cooperation_rate` | float | Overall cooperation level |

---

## Example: Defensive Weak Partner Strategy

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("PartnerHoldUp-v0")
obs, info = env.reset(seed=42)

episode_rewards = np.zeros(2)

for step in range(100):
    # Strong: Moderate exploitation (50% of 120 = 60, below fair share)
    strong_action = 60.0

    # Weak: Defensive (slightly above baseline to maintain trust)
    # Baseline is 28, so 35 is modest cooperation
    weak_action = 35.0

    actions = np.array([strong_action, weak_action])
    obs, rewards, terminated, truncated, info = env.step(actions)
    episode_rewards += rewards

    if terminated:
        print(f"Relationship ended at step {step}")
        break

print(f"Strong total: {episode_rewards[0]:.1f}")
print(f"Weak total: {episode_rewards[1]:.1f}")
print(f"Final weak trust in strong: {info.get('weak_trust_in_strong', 'N/A')}")
```

---

## Research Applications

PartnerHoldUp-v0 is suitable for studying:

- **Power Dynamics**: How asymmetry affects cooperation
- **Contract Theory**: Hold-up problems and incomplete contracts
- **Supply Chain Management**: Buyer-supplier relationships
- **Strategic Management**: Alliance design with power imbalances
- **Fair Division**: Bargaining with unequal positions

---

## Related Environments

- [TrustDilemma-v0](trust_dilemma.md): Symmetric version
- [PlatformEcosystem-v0](platform_ecosystem.md): Multi-agent power asymmetry
- [CooperativeNegotiation-v0](cooperative_negotiation.md): Explicit contracts

---

## References

1. Williamson, O.E. (1985). The Economic Institutions of Capitalism. Free Press.
2. Hart, O. & Moore, J. (1990). Property Rights and the Nature of the Firm. Journal of Political Economy.
3. Pant, V. & Yu, E. (2025). [Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics](https://arxiv.org/abs/2510.24909). arXiv:2510.24909
