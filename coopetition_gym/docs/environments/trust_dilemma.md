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

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | Markov Game (2-player, general-sum) |
| **Cooperation Structure** | Mixed-Motive (cooperation creates value, competition captures it) |
| **Observability** | Full (all state variables observable to both agents) |
| **Communication** | Implicit (through actions only) |
| **Agent Symmetry** | Symmetric (identical endowments, baselines, capabilities) |
| **Reward Structure** | Mixed (individual + interdependence-weighted partner rewards) |
| **Action Space** | Continuous, bounded: $A_i = [0, 100]$ |
| **State Dynamics** | Deterministic (given actions, next state is deterministic) |
| **Horizon** | Finite, T = 100 steps (or early termination on trust collapse) |
| **Canonical Comparison** | Continuous-action Iterated Prisoner's Dilemma with state-dependent payoffs; cf. Lerer & Peysakhovich (2017) "Maintaining Cooperation in Complex Social Dilemmas" |

---

## Formal Specification

This environment is formalized as a 2-player Markov Game **M** = (**N**, **S**, {**A**_i}, **P**, {**R**_i}, T).

### Agents
**N** = {1, 2} (symmetric dyad)

### State Space
**S** ⊆ ℝ¹⁷ with components:

| Component | Symbol | Dimension | Range | Description |
|-----------|--------|-----------|-------|-------------|
| Actions | a | 2 | [0, 100] | Previous cooperation levels |
| Trust Matrix | $\tau$ | 4 | [0, 1] | Pairwise trust $\tau_{ij}$ |
| Reputation Damage | $R$ | 4 | [0, 1] | Accumulated damage $R_{ij}$ |
| Interdependence | D | 4 | [0, 1] | Structural dependencies |
| Metadata | m | 3 | varies | Timestep, auxiliary info |

**Total dimension**: d = 17

### Action Space
For each agent $i \in \{1, 2\}$:

$$A_i = [0, e_i] = [0, 100] \subset \mathbb{R}$$

where $e_i = 100$ is the endowment. Actions represent **cooperation level** (investment in joint value creation).

### Transition Dynamics

**Trust Update** (TR-2 dynamics):

```
τ_ij(t+1) = clip(τ_ij(t) + Δτ_ij, 0, Θ_ij)
```

where the trust ceiling $\Theta_{ij} = 1 - R_{ij}$ and the update is:

```
Δτ_ij = λ⁺ · max(0, σ_j) · (1 - τ_ij) - λ⁻ · max(0, -σ_j) · τ_ij
```

with cooperation signal:
```
σ_j = κ · (a_j - b_j) / b_j
```

**Reputation Update**:
```
R_ij(t+1) = R_ij(t) · (1 - $\delta_R$) + $\mu_R$ · 𝟙[σ_j < -threshold]
```

### Reward Function

Agent i receives integrated utility:

```
r_i(s, a) = U_i(a) = π_i(a) + Σ_j D_ij · π_j(a)
```

where private payoff π_i is:

```
π_i(a) = (e_i - a_i) + f(a_i) + α_i · G(a)
```

with:
- **Retained resources**: e_i - a_i
- **Individual value**: f(a_i) = θ · ln(1 + a_i), θ = 20.0
- **Synergy share**: $\alpha_i \cdot G(a)$ where $G(a) = (a_1 \cdot a_2)^{1/2} \cdot (1 + \gamma \cdot C(a))$
- **Complementarity**: C(a) = min(a_1/e_1, a_2/e_2), γ = 0.70

### Episode Structure

- **Horizon**: T = 100 steps
- **Truncation**: t ≥ T
- **Termination**: mean(τ) < 0.05 (trust collapse)
- **Discount**: γ = 1.0 (undiscounted finite horizon)

### Initial State
- $\tau_{ij}(0) = 0.50$ for all $i \neq j$
- $R_{ij}(0) = 0.00$ for all $i, j$
- $a(0) = (0, 0)$

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

## Theoretical Foundations

### Relationship to Classical Game Theory

TrustDilemma-v0 extends the classical Iterated Prisoner's Dilemma (IPD) by incorporating:

1. **Continuous action spaces**: Rather than discrete {Cooperate, Defect}, agents choose cooperation intensity $a_i \in [0, 100]$
2. **State-dependent payoffs**: Rewards are modulated by endogenous trust, creating a Markov Game rather than a repeated normal-form game
3. **Asymmetric dynamics**: The 3:1 negativity bias (λ⁻/λ⁺) captures empirically-observed trust asymmetry (Slovic, 1993)
4. **Reputation hysteresis**: Cumulative damage creates irreversibility absent from classical models

### Key Theoretical Results

**Stage-Game Analysis**:

In the single-shot version (ignoring trust dynamics), the environment resembles a continuous public goods game:

- **Nash equilibrium (myopic)**: $a^* \approx 35$ (baseline contribution)
  - At this level, marginal cost of contribution equals marginal private benefit
- **Pareto-optimal outcome**: $a^* = 100$ (full cooperation)
  - Social welfare maximized when both agents fully invest
- **Price of Anarchy**: PoA ≈ 2.3
  - Ratio of optimal social welfare to welfare at Nash equilibrium

**Repeated Game Considerations**:

With T = 100 repetitions and trust dynamics, the Folk Theorem applies conditionally:

- **Folk Theorem applicability**: Partial
  - Finite horizon limits exact Folk Theorem results
  - However, trust dynamics create state-dependent continuation values that support cooperation
- **Subgame-perfect equilibria**:
  - Mutual defection (a = 35) is always an SPE
  - Cooperative equilibria (a > 35) sustainable when trust amplification exceeds defection temptation
  - Critical threshold: τ* ≈ 0.45 for cooperation to be self-enforcing

**Trust-Mediated Cooperation**:

The trust dynamics create a novel mechanism for cooperation:

```
Cooperation sustainable iff: ∂U/∂τ × ∂τ/∂a > temptation_gain
```

Where the left side captures the long-term value of trust investment.

### Connections to Prior Work

| Concept | TrustDilemma-v0 | Classical Reference |
|---------|-----------------|---------------------|
| Continuous cooperation | $a_i \in [0, e_i]$ | Discrete {C, D} in Axelrod (1984) |
| State-dependent payoffs | Trust modulation $\tau$ | Stateless in classical IPD |
| Asymmetric dynamics | $\lambda^- = 3 \times \lambda^+$ | Symmetric in standard models |
| Reputation effects | Ceiling $\Theta = 1 - R$ | No reputation in basic IPD |
| Complementarity | $\gamma$-weighted synergy | Not present in classical |

### Literature Connections

**Axelrod (1984)**: The foundational work on IPD tournaments. TrustDilemma-v0 extends this by:
- Continuous actions (allows graduated responses)
- Trust state (enables history-dependent but Markovian strategies)
- Complementarity (rewards coordinated high cooperation)

**Lerer & Peysakhovich (2017)**: "Maintaining Cooperation in Complex Social Dilemmas" studies deep RL in social dilemmas. TrustDilemma-v0 provides:
- Similar mixed-motive structure
- Richer state dynamics (trust, reputation)
- Continuous action complexity

**Leibo et al. (2017)**: Sequential social dilemmas in grid-worlds. TrustDilemma-v0 differs by:
- Direct strategic interaction (not spatially mediated)
- Explicit trust dynamics (not emergent from grid mechanics)
- Continuous rather than discrete action timing

---

## Equilibrium Analysis

### Stage-Game Nash Equilibrium

In the single-shot game (ignoring trust dynamics), we analyze best responses:

**Best Response Functions**:

For agent $i$ with utility $U_i = (e_i - a_i) + \theta \cdot \ln(1 + a_i) + \alpha_i \cdot G(\mathbf{a}) + D_{ij} \cdot \pi_j$:

```
∂U_i/∂a_i = -1 + θ/(1 + a_i) + α_i·∂G/∂a_i = 0
```

Solving for interior solutions:
```
a_i* ≈ θ - 1 + α_i·(∂G/∂a_i)
```

**Nash Equilibrium (Symmetric Case)**:

With $\theta = 20$, $\alpha = 0.50$, and moderate complementarity:
- **Myopic NE**: $a^* \approx 35$ (baseline level)
- Both agents contribute at the minimum expected level
- Neither has unilateral incentive to deviate

**Interpretation**: The stage-game NE represents mutual caution—each agent invests just enough to avoid being seen as defecting.

### Pareto Frontier

The set of Pareto-optimal action profiles satisfies:

```
max_{a_1, a_2} W = U_1 + U_2
subject to: a_i ∈ [0, 100]
```

**Pareto-optimal outcomes**:
- **Full cooperation**: (a_1*, a_2*) = (100, 100)
  - Total welfare: W* ≈ 285
- **Pareto frontier**: All symmetric profiles (a, a) where a > 35

**Social Welfare Comparison**:

| Profile | Agent 0 Utility | Agent 1 Utility | Total | Trust Δ |
|---------|-----------------|-----------------|-------|---------|
| (35, 35) | 72.3 | 72.3 | 144.6 | -0.02 |
| (50, 50) | 85.1 | 85.1 | 170.2 | +0.01 |
| (70, 70) | 98.2 | 98.2 | 196.4 | +0.04 |
| (100, 100) | 112.5 | 112.5 | 225.0 | +0.06 |

**Price of Anarchy**: PoA = W*/W_NE ≈ 225/145 ≈ 1.55

### Repeated Game Equilibria

With T = 100 repetitions and trust dynamics, richer equilibria emerge:

**Grim Trigger Strategy**:
```
Play a_high if no defection observed
Play a_low forever after any defection
```

Grim trigger supports cooperation when:
```
δ·V_coop/(1-δ) > V_defect + δ·V_punishment/(1-δ)
```

Where δ captures effective discount rate accounting for trust dynamics.

**Trust-Augmented Trigger**:

The trust dynamics provide a natural "soft" trigger:
- Defection erodes trust (λ⁻ = 0.45)
- Eroded trust reduces future payoffs
- Creates self-enforcing cooperation without explicit punishment

**Cooperative Equilibrium Conditions**:

Cooperation (a > baseline) is sustainable in equilibrium when:
```
τ > τ* where τ* ≈ 0.45
```

Below τ*, the trust-mediated payoff amplification is insufficient to deter defection.

### Trust-Mediated Equilibrium Dynamics

The environment creates a novel equilibrium structure:

**Trust Threshold Effects**:

1. **High trust regime** (τ > 0.70):
   - Cooperation strongly reinforced
   - High payoffs sustain investment
   - Robust to small deviations

2. **Medium trust regime** (0.30 < τ < 0.70):
   - Multiple equilibria possible
   - Coordination challenge
   - History-dependent outcomes

3. **Low trust regime** (τ < 0.30):
   - Defection dominates
   - Recovery difficult (3:1 negativity bias)
   - Approaching termination threshold

**Basin of Attraction**:

Starting from τ₀ = 0.50:
- Sustained cooperation → converges to high-trust equilibrium
- Early defection → converges to low-trust/collapse

The **separating trajectory** depends on initial cooperation and response to early deviations.

### Equilibrium Selection in MARL

For RL agents learning in this environment:

**Expected Learning Dynamics**:

1. **Self-play with exploration**: May converge to either equilibrium
   - High exploration → samples cooperative outcomes → possible convergence to cooperation
   - Greedy exploitation → myopic defection → convergence to low equilibrium

2. **Curriculum considerations**:
   - Starting with high initial trust helps discover cooperative equilibrium
   - Training with trust "warm-starts" improves convergence

**Algorithm Implications**:

| Algorithm | Expected Equilibrium | Notes |
|-----------|---------------------|-------|
| Independent PPO | Mixed/Low | Coordination challenge |
| MAPPO (shared) | High possible | Shared critic helps |
| MADDPG | Medium | Centralized training helps |
| LOLA | High likely | Models opponent adaptation |

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
| Reputation Damage | $\mu_R$ | 0.50 | Damage coefficient from violations |
| Reputation Decay | $\delta_R$ | 0.02 | Rate of reputation forgiveness |
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

| Algorithm | Mean Return | Std | Final Trust | Coop Rate | Training Steps |
|-----------|-------------|-----|-------------|-----------|----------------|
| Random | 82.4 | 14.2 | 0.28 | 0.49 | - |
| Constant(0.35) | 98.6 | 6.3 | 0.42 | 0.35 | - |
| Constant(0.50) | 112.8 | 7.1 | 0.56 | 0.50 | - |
| Constant(0.75) | 134.2 | 8.4 | 0.68 | 0.75 | - |
| Tit-for-Tat | 128.5 | 10.8 | 0.64 | 0.58 | - |
| IPPO | 145.2 | 11.3 | 0.72 | 0.62 | 500K |
| MAPPO | 162.8 | 9.2 | 0.78 | 0.68 | 500K |

*Results averaged over 5 training seeds. Mean Return is sum of both agents' episode returns.*

### Learning Curve Characteristics

- **Random**: Baseline lower bound; trust decays due to inconsistent behavior
- **Constant policies**: Stable but suboptimal; no adaptation
- **Tit-for-Tat**: Strong initial performance; sensitive to early defection
- **IPPO**: Converges around 200K steps; coordination challenge leads to variance
- **MAPPO**: Faster convergence (~150K steps); shared critic aids coordination

### Recommended Hyperparameters

```yaml
# PPO configuration for TrustDilemma-v0
algorithm: PPO
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
ent_coef: 0.01
network:
  hidden_layers: [128, 128]
```

---

## Related Environments

- [PartnerHoldUp-v0](partner_holdup.md): Adds asymmetric power dynamics
- [RecoveryRace-v0](recovery_race.md): Focuses on trust recovery
- [SLCD-v0](slcd.md): Validated real-world case study

---

## References

1. Pant, V. & Yu, E. (2025). [Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity](https://arxiv.org/abs/2510.18802). arXiv:2510.18802
2. Pant, V. & Yu, E. (2025). [Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics](https://arxiv.org/abs/2510.24909). arXiv:2510.24909
3. Axelrod, R. (1984). The Evolution of Cooperation. Basic Books.
