# GiftExchange-v0

**Category:** Reciprocity Environment (TR-4)
**Agents:** 2
**Difficulty:** Intermediate
**Source:** `coopetition_gym/envs/reciprocity_envs.py`

---

## Overview

GiftExchange-v0 implements an **asymmetric employer-worker gift exchange game** with TR-4 reciprocity dynamics. The employer (Agent 0) sets a wage-cooperation level, and the worker (Agent 1) responds with effort-cooperation. Fair wages elicit reciprocal effort; unfair wages trigger shirking.

The environment tests whether agents can learn **asymmetric reciprocity**,the worker reciprocates more strongly than the employer due to higher structural dependency.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | 2-player Markov Game (general-sum) |
| **Cooperation Structure** | Mixed-Motive (wage-effort exchange) |
| **Observability** | Full |
| **Communication** | Implicit |
| **Agent Symmetry** | Asymmetric (different endowments, dependencies) |
| **Reward Structure** | Integrated utility with asymmetric reciprocity |
| **Action Space** | Continuous: $A_0 = [0, 100]$, $A_1 = [0, 80]$ |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T = 100 steps |
| **Canonical Comparison** | Gift exchange game; Fehr, Kirchsteiger & Riedl (1993); Akerlof (1982) |

---

## Formal Specification

### Asymmetric Dependency Structure

The interdependence matrix captures power asymmetry:

$$\mathbf{D} = \begin{pmatrix} 0 & 0.4 \\ 0.7 & 0 \end{pmatrix}$$

The worker depends more on the employer ($D_{21} = 0.7$) than the employer depends on the worker ($D_{12} = 0.4$).

### Asymmetric Reciprocity Sensitivity (Eq 23)

With $\rho_0 = 1.2$ and $\eta = 1.5$:

- **Worker's sensitivity**: $\rho_{21} = 1.2 \cdot 0.7^{1.5} \approx 0.703$
- **Employer's sensitivity**: $\rho_{12} = 1.2 \cdot 0.4^{1.5} \approx 0.304$

The worker reciprocates **2.3× more strongly** than the employer, capturing the empirical finding that dependent parties show stronger reciprocal responses.

### TR-4 Equations

All equations follow the standard TR-4 framework:

| Equation | Paper Ref | Description |
|----------|-----------|-------------|
| $s_{ij} = a_j - \bar{a}_j$ | Eq 19 | Cooperation signal |
| $\bar{a}_j = \frac{1}{\min(k,t-1)} \sum a_j^\tau$ | Eq 20 | Memory average ($k = 3$) |
| $\varphi(x) = \tanh(\kappa x)$ | Eq 21 | Bounded response ($\kappa = 1.0$) |
| $\rho_{ij} = \rho_0 \cdot D_{ij}^\eta$ | Eq 23 | Reciprocity sensitivity |
| $U_{\text{recip}} = \lambda_R \sum T_{ij} \cdot (1+\omega D_{ij}) \cdot \rho_{ij} \cdot \varphi(s_{ij})$ | Eq 44 | Reciprocity modifier |

### State Space

**S** ⊆ ℝ^d with components:

| Component | Symbol | Description |
|-----------|--------|-------------|
| Actions | a | Previous cooperation levels |
| Trust Matrix | T | Pairwise trust (from TR-2) |
| Reputation | R | Accumulated reputation damage |
| Interdependence | D | Asymmetric dependencies |
| Memory | ā | Recent action averages |

### Action Space

| Agent | Role | Endowment | Action Range |
|-------|------|-----------|-------------|
| 0 | Employer | 100 | $[0, 100]$ |
| 1 | Worker | 80 | $[0, 80]$ |

> **Uniaxial Treatment**: This environment uses the single-dimension action space characteristic of Coopetition-Gym v1.x. The employer-worker power asymmetry emerges through asymmetric endowments, dependencies, and reciprocity sensitivities.

---

## Distinction from PartnerHoldUp-v0

| Aspect | PartnerHoldUp-v0 | GiftExchange-v0 |
|--------|-------------------|-----------------|
| **Mechanism** | Structural lock-in (TR-1/TR-2) | Voluntary reciprocity (TR-4) |
| **Asymmetry Source** | Endowment and dependency | Reciprocity sensitivity |
| **Exit Option** | Weak partner can exit | No exit, ongoing exchange |
| **Key Dynamic** | Hold-up exploitation | Gift-giving and reciprocation |
| **Strategy** | Defensive vs. exploitative | Fair wages trigger effort |

---

## Game-Theoretic Background

### The Gift Exchange Paradigm

Akerlof's (1982) efficiency wage theory and Fehr et al.'s (1993) experimental findings establish that: 1. **Standard prediction**: Workers exert minimum effort regardless of wage
2. **Observed behavior**: Higher wages elicit higher effort (positive reciprocity)
3. **Unfair wages**: Below-baseline wages trigger effort reduction (negative reciprocity)
4. **Asymmetric response**: Workers reciprocate more strongly when dependent

### Strategic Implications

**Employer's Dilemma:**
- Low wages save cost but trigger shirking via negative reciprocity
- Fair wages cost more but elicit above-minimum effort
- Optimal wage depends on worker's reciprocity sensitivity

**Worker's Response:**
- Short memory ($k = 3$) enables rapid response to wage changes
- Higher dependency ($D_{21} = 0.7$) amplifies reciprocal reaction
- Trust gating means reciprocity requires baseline trust

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("GiftExchange-v0")

obs, info = env.reset(seed=42)

# Employer offers fair wage, worker reciprocates
for step in range(100): actions = np.array([65.0, 55.0])  # Employer: 65%, Worker: 55%
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated: break

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
| Base reciprocity | $\rho_0$ | 1.2 | Higher base reciprocity |
| Dependency elasticity | $\eta$ | 1.5 | Superlinear dependency effect |
| Response sensitivity | $\kappa$ | 1.0 | Bounded response steepness |
| Memory window | $k$ | 3 | Short memory (fast response) |
| Reciprocity weight | $\lambda_R$ | 1.2 | Stronger reciprocity scaling |
| Dependency amplification | $\omega$ | 0.8 | High dependency boost |

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
**Range:** `[0.0, 100.0]` for Agent 0, `[0.0, 80.0]` for Agent 1

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

### Wage-Effort Reciprocity

1. Employer increases wage above baseline → positive $s_{21}$
2. Worker detects above-average cooperation → bounded response $\varphi > 0$
3. Worker's high reciprocity sensitivity ($\rho_{21} \approx 0.703$) amplifies response
4. Worker increases effort → positive feedback cycle

### Wage Cut Response

1. Employer reduces wage below memory average → negative $s_{21}$
2. Worker's short memory ($k = 3$) detects change quickly
3. Negative reciprocity reduces worker's reward modifier
4. Persistent low wages erode trust (TR-2) compounding the effect

---

## Research Applications

GiftExchange-v0 is suitable for studying:

- **Efficiency Wages**: Does paying above-market wages increase effort?
- **Asymmetric Reciprocity**: How does dependency asymmetry affect exchange?
- **Fair Wage Hypothesis**: Akerlof's model of gift exchange in labor markets
- **Power Dynamics**: How power asymmetry interacts with reciprocity

---

## Related Environments

- [PartnerHoldUp-v0](partner_holdup.md): Structural lock-in variant (TR-1/TR-2)
- [ReciprocalDilemma-v0](reciprocal_dilemma.md): Symmetric reciprocity baseline
- [AppleAppStore-v0](apple_app_store.md): Multi-agent asymmetric case study

---

## References

1. Pant, V. & Yu, E. (2026). Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity. *arXiv:2604.01240*. [Link](https://arxiv.org/abs/2604.01240)
2. Fehr, E., Kirchsteiger, G. & Riedl, A. (1993). Does Fairness Prevent Market Clearing? An Experimental Investigation. Quarterly Journal of Economics.
3. Akerlof, G. A. (1982). Labor Contracts as Partial Gift Exchange. Quarterly Journal of Economics.
