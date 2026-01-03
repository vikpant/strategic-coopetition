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

---

## Theoretical Foundations

### Relationship to Classical Game Theory

PartnerHoldUp-v0 extends the classical hold-up problem literature by incorporating:

1. **Continuous investments**: Rather than binary invest/not-invest decisions, agents choose continuous contribution levels
2. **Dynamic power relations**: Trust and reputation evolve over time, affecting the effective power balance
3. **Endogenous exit**: The weak partner's exit threshold creates a credible threat mechanism
4. **Repeated interaction**: Unlike one-shot hold-up models, the environment captures ongoing relationship dynamics

### Key Theoretical Results

**Stage-Game Analysis**:

In the single-shot version (ignoring trust dynamics and exit):

- **Strong agent's dominant strategy**: Contribute near baseline (a₀* ≈ 42)
  - Captures maximum surplus given weak's limited response
- **Weak agent's best response**: Match effort level proportionally (a₁* ≈ 28-35)
  - Cannot profitably exceed strong's proportional contribution
- **Nash equilibrium**: (a₀*, a₁*) ≈ (45, 30) - Mutual low cooperation
- **Pareto frontier**: Achievable only with binding commitments

**Asymmetric Bargaining Analysis**:

With bargaining shares α = (0.60, 0.40):

- **Nash Bargaining Solution** (with disagreement point at baseline):
  - Strong receives: 60% × (total surplus - disagreement payoffs)
  - Weak receives: 40% × (total surplus - disagreement payoffs)
- **Outside option effect**: Strong's lower dependency (D = 0.35) improves BATNA

**Repeated Game Equilibria**:

- **Exploitation equilibrium**: Strong exploits maximally until exit threshold approached
- **Sustainable equilibrium**: Strong moderates exploitation to maintain relationship
- **Trigger strategy by weak**: Credible exit threat at τ < 0.10 disciplines strong

The critical insight is that the exit threshold τ* = 0.10 creates a **credible commitment device** for the weak partner, partially offsetting the power asymmetry.

### Connections to Prior Work

| Concept | PartnerHoldUp-v0 | Classical Reference |
|---------|------------------|---------------------|
| Relationship-specific investment | Continuous a_i ∈ [0, e_i] | Binary in Williamson (1985) |
| Incomplete contracts | Trust dynamics as implicit contract | Explicit contracts in Hart & Moore (1990) |
| Exit option | Endogenous τ < 0.10 termination | Exogenous outside option in Nash bargaining |
| Power asymmetry | D_ij captures structural dependency | Bargaining power in Rubinstein (1982) |
| Dynamic adjustment | Trust evolution | Static in classical models |

### Literature Connections

**Williamson (1985)**: Transaction cost economics and the hold-up problem. PartnerHoldUp-v0 operationalizes:
- Asset specificity → High D_weak = 0.85
- Opportunism risk → Trust erosion from defection
- Governance mechanisms → Trust dynamics as relational contract

**Hart & Moore (1990)**: Property rights theory of the firm. The environment captures:
- Residual control rights → Bargaining share α
- Investment incentives → Continuous action choice
- Renegotiation → Ongoing trust-mediated interaction

**Klein, Crawford & Alchian (1978)**: The appropriable quasi-rent. Represented by:
- Weak partner's relationship-specific value captured by high interdependence D = 0.85
- Strong partner's ability to extract this rent via exploitation

**Rubinstein (1982)**: Alternating offers bargaining. The environment extends this with:
- Continuous-time (each step) rather than discrete rounds
- State-dependent patience (trust affects future value)
- Endogenous breakdown (exit threshold)

### Power Balance Mechanisms

The environment embeds several mechanisms that affect the power balance:

1. **Exit credibility**: τ < 0.10 termination gives weak partner leverage
2. **Reputation effects**: Exploitation damages strong's ability to sustain relationships
3. **Trust ceiling**: Θ = 1 - R limits recovery from exploitation
4. **Interdependence amplification**: High ξ = 0.70 magnifies trust effects for weak partner

---

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

## Equilibrium Analysis

### Asymmetric Stage-Game Equilibrium

Unlike TrustDilemma-v0, PartnerHoldUp-v0 has asymmetric equilibria due to differing capabilities.

**Best Response Functions**:

For strong agent (i=0) with e_0 = 120, α_0 = 0.60, D_01 = 0.35:
```
BR_strong(a_weak) = argmax_{a_0} [(120-a_0) + θ·ln(1+a_0) + 0.60·G(a_0,a_1) + 0.35·π_weak]
```

For weak agent (i=1) with e_1 = 80, α_1 = 0.40, D_10 = 0.85:
```
BR_weak(a_strong) = argmax_{a_1} [(80-a_1) + θ·ln(1+a_1) + 0.40·G(a_0,a_1) + 0.85·π_strong]
```

**Nash Equilibrium (Myopic)**:

| Agent | Equilibrium Action | % of Endowment |
|-------|-------------------|----------------|
| Strong | a_0* ≈ 45 | 37.5% |
| Weak | a_1* ≈ 30 | 37.5% |

Both contribute near their baselines (35%), reflecting mutual caution.

**Equilibrium Payoffs**:

| Agent | Myopic NE Payoff | Full Cooperation Payoff |
|-------|------------------|------------------------|
| Strong | ~95 | ~145 |
| Weak | ~68 | ~110 |

### Pareto Frontier with Asymmetry

The Pareto frontier is asymmetric:

```
Strong prefers: High (a_s, a_w) with more surplus capture
Weak prefers: Moderate cooperation with trust protection
```

**Pareto-Optimal Profiles**:

| Profile | Strong Utility | Weak Utility | Total | Sustainable? |
|---------|---------------|--------------|-------|--------------|
| (45, 30) NE | 95 | 68 | 163 | Yes (marginally) |
| (70, 50) | 125 | 92 | 217 | Yes |
| (90, 65) | 138 | 105 | 243 | Yes |
| (120, 80) Max | 145 | 112 | 257 | Fragile |

**Price of Anarchy**: PoA = 257/163 ≈ 1.58

### Stackelberg Equilibrium

Given strong's power advantage, a Stackelberg analysis is relevant:

**Strong as Leader**:

If strong commits first, the optimal strategy is:
1. Choose a_0 anticipating weak's best response BR_weak(a_0)
2. Maximize own payoff subject to weak's participation constraint

**Stackelberg Outcome**:
- Strong: a_0* ≈ 55-60 (moderate exploitation)
- Weak: BR_weak ≈ 35-40 (defensive response)
- Strong extracts more surplus than simultaneous-move NE

**Weak's Participation Constraint**:

Weak will participate (not exit) iff:
```
U_weak(a_0, BR_weak(a_0)) ≥ Outside_Option
```

The exit threshold τ < 0.10 operationalizes this constraint dynamically.

### Repeated Game with Exit Threat

The exit threshold creates a credible commitment mechanism:

**Exit as Punishment**:

Weak's threat to exit if τ < 0.10 is credible because:
- Weak loses from staying in exploitative relationship
- Exit terminates strong's long-term gains
- Creates dynamic incentive for strong to moderate

**Modified Folk Theorem**:

With exit threats, a range of equilibria becomes feasible:
- **Lower bound**: Strong exploits until exit threshold approached
- **Upper bound**: Full cooperation if credible punishment exists

**Equilibrium Exploitation Level**:

Strong's optimal exploitation satisfies:
```
∂π_strong/∂(exploitation) = λ⁻ × (value_of_relationship) × ∂τ/∂a
```

Balancing marginal gain from extraction against marginal loss from trust erosion.

### Trust Ceiling Effects

Reputation damage creates strategic constraints:

**Dynamic Constraints**:
```
τ_max(t) = 1 - R(t)
```

Once strong exploits and causes reputation damage R:
- Trust ceiling drops permanently
- Future cooperation limited
- Weak's defensive options reduced

**Irreversibility Premium**:

Strong must weigh:
- Short-term exploitation gains
- Permanent ceiling reduction
- Reduced future surplus to extract

### MARL Implications

**Learning Challenges**:

| Challenge | Source | MARL Implication |
|-----------|--------|------------------|
| Power asymmetry | Different BRs | Agents need different policies |
| Exit threshold | Termination risk | Long-horizon credit assignment |
| Trust ceiling | Irreversibility | Exploration/exploitation tension |
| Stackelberg dynamics | Sequential structure | Leader-follower learning |

**Expected Algorithm Performance**:

| Algorithm | Strong's Outcome | Weak's Outcome | Notes |
|-----------|-----------------|----------------|-------|
| Independent PPO | Exploitation | Defensive | Power asymmetry amplified |
| MAPPO | Moderate | Moderate | Centralization helps coordination |
| Hierarchical | Optimal extraction | Constrained | Matches Stackelberg |

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

## Baseline Results

Benchmark results following the [Evaluation Protocol](../evaluation_protocol.md).

### Evaluation Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 100 |
| Seeds | 0-99 |
| Horizon | 100 steps |
| Training seeds | 100-104 (5 runs) |

### Performance Comparison

| Algorithm | Strong Return | Weak Return | Final Trust | Exploitation Rate |
|-----------|---------------|-------------|-------------|-------------------|
| Random | 78.2 | 45.6 | 0.24 | 0.52 |
| Constant(Fair) | 95.4 | 62.8 | 0.48 | 0.35 |
| Strong Exploits | 108.3 | 38.2 | 0.22 | 0.68 |
| Weak Defensive | 86.5 | 58.4 | 0.36 | 0.42 |
| IPPO | 102.6 | 52.4 | 0.38 | 0.55 |
| MAPPO | 98.2 | 64.2 | 0.52 | 0.40 |

*Exploitation Rate = (Strong action/Strong endowment) - (Weak action/Weak endowment). Higher indicates strong exploiting power asymmetry.*

### Key Observations

- **Power amplification**: IPPO tends to amplify power asymmetry (strong learns exploitation)
- **Centralized training helps**: MAPPO achieves more balanced outcomes
- **Trust-fairness tradeoff**: Higher exploitation yields higher strong returns but lower trust
- **Exit threshold matters**: Weak trust < 0.15 often leads to early termination

### Recommended Hyperparameters

```yaml
# PPO configuration for PartnerHoldUp-v0
algorithm: PPO
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
gamma: 0.99
# Higher entropy helps explore fair strategies
ent_coef: 0.02
network:
  hidden_layers: [128, 128]
```

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
