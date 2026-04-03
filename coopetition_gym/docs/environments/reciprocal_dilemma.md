# ReciprocalDilemma-v0

**Category:** Reciprocity Environment (TR-4)
**Agents:** 2
**Difficulty:** Intermediate
**Source:** `coopetition_gym/envs/reciprocity_envs.py`

---

## Overview

ReciprocalDilemma-v0 implements a **continuous iterated Prisoner's Dilemma** with TR-4 reciprocity dynamics. Two symmetric firms decide cooperation levels in a shared project, where reciprocity enables tit-for-tat-like conditional cooperation through bounded memory windows.

The environment tests whether reinforcement learning agents can learn **conditional cooperation**—responding to partner behavior over recent history rather than relying solely on slow-moving trust dynamics.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | 2-player Markov Game (general-sum) |
| **Cooperation Structure** | Mixed-Motive (cooperation vs. exploitation) |
| **Observability** | Full (all state variables observable) |
| **Communication** | Implicit (through actions only) |
| **Agent Symmetry** | Symmetric (identical capabilities) |
| **Reward Structure** | Integrated utility with reciprocity modifier |
| **Action Space** | Continuous, bounded: $A_i = [0, 100]$ |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T = 100 steps |
| **Canonical Comparison** | Iterated PD; Axelrod (1984); Killingback & Doebeli (2002) |

---

## Formal Specification

### Mathematical Framework (TR-4)

**Cooperation Signal** (Eq 19):
$$s_{ij} = a_j - \bar{a}_j$$

Where $\bar{a}_j$ is the memory average of agent $j$'s recent actions.

**Memory Average** (Eq 20):
$$\bar{a}_j = \frac{1}{\min(k, t-1)} \sum_{\tau=\max(1,t-k)}^{t-1} a_j^\tau$$

Where $k = 5$ is the memory window length.

**Bounded Response** (Eq 21):
$$\varphi(x) = \tanh(\kappa \cdot x)$$

Where $\kappa = 1.0$ controls response sensitivity.

**Reciprocity Sensitivity** (Eq 23):
$$\rho_{ij} = \rho_0 \cdot D_{ij}^\eta$$

Where $\rho_0 = 1.0$ and $\eta = 1.0$.

**Reciprocity Modifier** (Eq 44):
$$U_{\text{recip},i} = \lambda_R \sum_{j \neq i} T_{ij} \cdot (1 + \omega D_{ij}) \cdot \rho_{ij} \cdot \varphi(s_{ij})$$

Where $\lambda_R = 1.0$ and $\omega = 0.6$.

### State Space

**S** ⊆ ℝ^d with components:

| Component | Symbol | Description |
|-----------|--------|-------------|
| Actions | a | Previous cooperation levels |
| Trust Matrix | T | Pairwise trust (from TR-2) |
| Reputation | R | Accumulated reputation damage |
| Interdependence | D | Structural dependencies |
| Memory | ā | Recent action averages |

### Action Space

For each agent $i$:
$$A_i = [0, e_i] = [0, 100] \subset \mathbb{R}$$

Actions represent **cooperation level** in the shared project.

> **Uniaxial Treatment**: This environment uses the single-dimension action space characteristic of Coopetition-Gym v1.x. Competitive dynamics emerge through the PD payoff structure and reciprocity responses rather than explicit competitive actions.

### Reward Function

Rewards combine integrated utility (TR-1/TR-2) with reciprocity modifier (TR-4):

$$r_i = \pi_i^{\text{base}} \cdot m_{\text{recip},i}$$

Where $m_{\text{recip},i}$ is the multiplicative reciprocity modifier derived from Eq 44.

---

## Distinction from TrustDilemma-v0

| Aspect | TrustDilemma-v0 | ReciprocalDilemma-v0 |
|--------|-----------------|----------------------|
| **Mechanism** | TR-2 trust dynamics (slow erosion/building) | TR-4 behavioral reciprocity (fast, 1-5 step response) |
| **Response Time** | Trust changes over 10-20+ steps | Memory window of $k=5$ steps |
| **Adaptation** | Gradual trust adjustment | Immediate reciprocal response |
| **Key Equation** | Trust update (Eqs 8-9) | Reciprocity modifier (Eq 44) |
| **Strategy** | Long-horizon impulse control | Conditional cooperation (tit-for-tat) |

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("ReciprocalDilemma-v0")

# Reset
obs, info = env.reset(seed=42)

# Run episode with cooperative strategy
for step in range(100):
    actions = np.array([60.0, 60.0])
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated:
        break

print(f"Mean trust: {info['mean_trust']:.3f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 100 | Maximum timesteps |
| `render_mode` | None | Rendering mode |

### TR-4 Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Base reciprocity | $\rho_0$ | 1.0 | Reciprocity strength |
| Dependency elasticity | $\eta$ | 1.0 | How dependency scales reciprocity |
| Response sensitivity | $\kappa$ | 1.0 | Steepness of bounded response |
| Memory window | $k$ | 5 | Steps of recent history considered |
| Reciprocity weight | $\lambda_R$ | 1.0 | Overall reciprocity scaling |
| Dependency amplification | $\omega$ | 0.6 | Dependency boost in trust gating |

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Includes actions, trust matrix, reputation, interdependence, and step info.

### Action Space

**Type:** `Box`
**Shape:** `(2,)`
**Dtype:** `float32`
**Range:** `[0.0, 100.0]` for each agent

---

## Metrics and Info

The `info` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `mean_trust` | float | Average trust level |
| `cooperation_signals` | dict | Per-pair $s_{ij}$ values |
| `reciprocity_effects` | dict | Per-pair reciprocity contributions |
| `memory_averages` | dict | Per-pair memory averages $\bar{a}_j$ |
| `tr4_memory_window` | int | Memory window $k$ |

---

## Key Dynamics

### Reciprocity-Driven Cooperation

With symmetric dependencies ($D_{12} = D_{21} = 0.5$):
- $\rho_{12} = \rho_{21} = 1.0 \cdot 0.5^{1.0} = 0.5$
- Both agents respond equally to partner's cooperation signals
- Sustained cooperation builds positive feedback loop

### Defection Response

When one agent defects:
1. Cooperation signal $s_{ij}$ becomes negative (action below memory average)
2. Bounded response $\varphi(s)$ maps to negative value in $(-1, 0)$
3. Reciprocity modifier reduces defector's reward multiplier
4. Fast response within $k = 5$ steps (unlike slow trust erosion)

---

## Research Applications

ReciprocalDilemma-v0 is suitable for studying:

- **Conditional Cooperation**: Can agents learn tit-for-tat-like strategies?
- **Memory Effects**: How does memory window $k$ affect cooperation stability?
- **Reciprocity vs. Trust**: Comparing fast reciprocity (TR-4) with slow trust (TR-2)
- **Cooperation Emergence**: Conditions for sustained cooperation in continuous PD

---

## Related Environments

- [TrustDilemma-v0](trust_dilemma.md): TR-2 trust-based variant (slower dynamics)
- [GiftExchange-v0](gift_exchange.md): Asymmetric TR-4 reciprocity
- [GraduatedSanction-v0](graduated_sanction.md): Multi-agent reciprocity with sanctions

---

## References

1. Pant, V. & Yu, E. (2026). Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity. *arXiv:2604.01240*. [Link](https://arxiv.org/abs/2604.01240)
2. Axelrod, R. (1984). The Evolution of Cooperation. Basic Books.
3. Killingback, T. & Doebeli, M. (2002). The Continuous Prisoner's Dilemma and the Evolution of Cooperation through Reciprocal Altruism with Variable Investment. American Naturalist.
