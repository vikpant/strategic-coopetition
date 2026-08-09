# AppleAppStore-v0

**Category:** Validated Case Study (TR-4)
**Agents:** 3
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/reciprocity_envs.py`

---

## Overview

AppleAppStore-v0 is a **validated case study** based on the Apple iOS App Store ecosystem (2008-2024). This environment reproduces the **43/51 validation score** produced by the released TR-4 validation suite, modeling platform-developer reciprocity dynamics across 66 quarterly time steps in five historical phases.

The environment tests whether agents can learn phase-appropriate reciprocity strategies that match historically observed cooperation and defection patterns.

---

## Validation Score

**TR-4 Validation: 48/55**

This environment's parameters are calibrated against real Apple App Store data (2008-2024), making it a gold standard for TR-4 research.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | 3-player Markov Game (general-sum) |
| **Cooperation Structure** | Platform coopetition with power asymmetry |
| **Observability** | Full |
| **Communication** | Implicit |
| **Agent Symmetry** | Asymmetric (platform vs. developers) |
| **Reward Structure** | Integrated utility with asymmetric reciprocity |
| **Action Space** | Continuous: $A_0 = [0,100]$, $A_1 = [0,80]$, $A_2 = [0,60]$ |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T = 66 steps (quarterly, 2008-2024) |
| **Canonical Comparison** | Platform markets; Parker, Van Alstyne & Choudary (2016); Rochet & Tirole (2003) |

---

## Agent Configuration

### Three Actors

| Agent | Role | Endowment | Baseline |
|-------|------|-----------|----------|
| 0 | **Apple** (platform provider) | 100 | 30 |
| 1 | **Major Developers** (Epic, Spotify, Netflix) | 80 | 24 |
| 2 | **Small Developers** (aggregated) | 60 | 18 |

### Asymmetric Dependency Structure

$$\mathbf{D} = \begin{pmatrix} 0 & 0.3 & 0.2 \\ 0.8 & 0 & 0.15 \\ 0.85 & 0.1 & 0 \end{pmatrix}$$

| Dependency | Value | Interpretation |
|-----------|-------|----------------|
| $D_{10}$ (MajorDevs → Apple) | 0.80 | Major developers highly depend on platform |
| $D_{20}$ (SmallDevs → Apple) | 0.85 | Small developers critically depend on platform |
| $D_{01}$ (Apple → MajorDevs) | 0.30 | Apple moderately depends on major developers |
| $D_{02}$ (Apple → SmallDevs) | 0.20 | Apple weakly depends on small developers |
| $D_{12}$ (MajorDevs → SmallDevs) | 0.15 | Minimal cross-developer dependency |
| $D_{21}$ (SmallDevs → MajorDevs) | 0.10 | Minimal cross-developer dependency |

### Asymmetric Reciprocity Sensitivity (Eq 23)

With $\rho_0 = 1.0$ and $\eta = 1.2$:

| Pair | $\rho_{ij}$ | Interpretation |
|------|-----------|----------------|
| MajorDevs → Apple | $1.0 \cdot 0.8^{1.2} = 0.770$ | Developers strongly reciprocate Apple's actions |
| SmallDevs → Apple | $1.0 \cdot 0.85^{1.2} = 0.820$ | Small developers reciprocate even more strongly |
| Apple → MajorDevs | $1.0 \cdot 0.3^{1.2} = 0.249$ | Apple weakly reciprocates developer actions |
| Apple → SmallDevs | $1.0 \cdot 0.2^{1.2} = 0.157$ | Apple barely reciprocates small developer actions |

**Power asymmetry captured precisely**: Developers respond to Apple 3-5× more strongly than Apple responds to developers.

---

## Historical Phases

The 66-step episode maps to 66 quarters of Apple App Store history:

| Phase | Quarters | Years | Cooperation Pattern |
|-------|----------|-------|-------------------|
| **Symbiosis** | Q1-Q16 | 2008-2012 | High mutual cooperation; generous revenue sharing |
| **Maturation** | Q17-Q36 | 2012-2017 | Stable cooperation; platform rules tighten gradually |
| **Tension** | Q37-Q48 | 2017-2020 | Declining reciprocity; developer complaints mount |
| **Crisis** | Q49-Q54 | 2020-2021 | Reciprocal defection; Epic lawsuit, antitrust hearings |
| **Adjustment** | Q55-Q66 | 2022-2024 | Partial restoration; regulatory-driven concessions |

---

## Formal Specification

### TR-4 Equations

All standard TR-4 equations apply:

| Equation | Paper Ref | Description |
|----------|-----------|-------------|
| $s_{ij} = a_j - \bar{a}_j$ | Eq 19 | Cooperation signal |
| $\bar{a}_j = \frac{1}{\min(k,t-1)} \sum a_j^\tau$ | Eq 20 | Memory average ($k = 4$) |
| $\varphi(x) = \tanh(\kappa x)$ | Eq 21 | Bounded response ($\kappa = 0.8$) |
| $\rho_{ij} = \rho_0 \cdot D_{ij}^\eta$ | Eq 23 | Reciprocity sensitivity |
| $U_{\text{recip}} = \lambda_R \sum T_{ij} \cdot (1+\omega D_{ij}) \cdot \rho_{ij} \cdot \varphi(s_{ij})$ | Eq 44 | Reciprocity modifier |

### Trust Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\lambda^+$ | 0.10 | Trust building rate |
| $\lambda^-$ | 0.30 | Trust erosion rate (3:1 negativity bias) |
| $\mu_R$ | 0.60 | Reputation damage severity |
| $\delta_R$ | 0.03 | Reputation decay rate |
| $\xi$ | 0.50 | Interdependence amplification |
| Initial trust | 0.60 | Moderate-high (symbiosis phase start) |

---

## Distinction from SLCD-v0 and RenaultNissan-v0

| Aspect | SLCD-v0 | RenaultNissan-v0 | AppleAppStore-v0 |
|--------|---------|------------------|------------------|
| **TR Focus** | TR-1 value creation | TR-2 trust dynamics | TR-4 reciprocity |
| **Agents** | 2 (Samsung, Sony) | 2 (Renault, Nissan) | 3 (Apple, MajorDevs, SmallDevs) |
| **Key Dynamic** | Complementarity | Trust erosion and recovery | Conditional reciprocity |
| **Validation** | 58/60 | 49/60 | 48/55 |
| **Time Scale** | 2004-2011 | 1999-2025 | 2008-2024 |
| **Industry** | Manufacturing JV | Automotive alliance | Platform ecosystem |

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("AppleAppStore-v0")

obs, info = env.reset(seed=42)

# Run 66-quarter episode
for quarter in range(66):
    # Apple cooperates moderately, developers follow
    actions = np.array([60.0, 50.0, 40.0])
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated: break

print(f"Mean trust: {info['mean_trust']:.3f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 66 | Quarterly timesteps (2008-2024) |
| `render_mode` | None | Rendering mode |

### TR-4 Parameters

| Parameter | Symbol | Value | Rationale |
|-----------|--------|-------|-----------|
| Base reciprocity | $\rho_0$ | 1.0 | Standard base |
| Dependency elasticity | $\eta$ | 1.2 | Slightly superlinear |
| Response sensitivity | $\kappa$ | 0.8 | Gradual corporate response |
| Memory window | $k$ | 4 | Quarterly evaluation cycle |
| Reciprocity weight | $\lambda_R$ | 1.0 | Standard weight |
| Dependency amplification | $\omega$ | 0.8 | Strong dependency boost |

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Includes actions, trust matrix (3×3), reputation (3×3), interdependence (3×3), and step info.

### Action Space

**Type:** `Box`
**Shape:** `(3,)`
**Dtype:** `float32`
**Range:** `[0.0, 100.0]` for Apple, `[0.0, 80.0]` for Major Developers, `[0.0, 60.0]` for Small Developers

---

## Metrics and Info

The `info` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current quarter (0-65) |
| `mean_trust` | float | Average trust level |
| `cooperation_signals` | dict | Per-pair $s_{ij}$ values (6 pairs) |
| `reciprocity_effects` | dict | Per-pair reciprocity contributions |
| `memory_averages` | dict | Per-pair memory averages |
| `tr4_memory_window` | int | Memory window $k = 4$ |

---

## Key Dynamics

### Platform Power Asymmetry

Apple's reciprocity insensitivity ($\rho_{01} = 0.249$, $\rho_{02} = 0.157$) versus developers' high sensitivity ($\rho_{10} = 0.770$, $\rho_{20} = 0.820$) creates a power dynamic where:

- Apple can reduce cooperation with limited reciprocal consequences
- Developers must cooperate to maintain positive reciprocity from Apple
- Small developers are the most vulnerable (highest $D_{20}$, lowest endowment)

### Phase Transitions

**Symbiosis → Tension**: As Apple tightens rules (lower cooperation), developers' high reciprocity sensitivity amplifies the negative signal, creating tension

**Crisis**: Mutual defection, Apple's actions trigger strong negative reciprocity from developers, but Apple is relatively insensitive to developer defection

**Adjustment**: Apple partially restores cooperation; developers' strong reciprocity sensitivity drives rapid positive response

---

## Empirical Validation

### TR-4 Validation Framework (55 points)

| Category | Points | Description |
|----------|--------|-------------|
| Phase transition timing | 15 | Matches historical quarter transitions |
| Reciprocity asymmetry | 15 | Developer sensitivity ratio validated |
| Cooperation trajectory | 15 | Action levels match empirical trends |
| Trust dynamics | 10 | Trust evolution matches partnership history |

**Apple Score: 48/55**

---

## Research Applications

AppleAppStore-v0 is suitable for studying:

- **Platform Governance**: How platform policy affects developer cooperation
- **Power Asymmetry**: Reciprocity under extreme dependency imbalance
- **Policy Analysis**: Effects of regulatory intervention on cooperation
- **Phase Transitions**: Historical pattern reproduction and prediction
- **Model Validation**: Comparing simulated vs historical platform dynamics

---

## Example: Phase Analysis

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("AppleAppStore-v0")
obs, info = env.reset(seed=42)

phase_data = {"symbiosis": [], "maturation": [], "tension": [],
              "crisis": [], "adjustment": []}

for q in range(66):
    # Strategy: Apple gradually reduces, developers respond
    apple_coop = max(30.0, 80.0 - q * 0.5)
    actions = np.array([apple_coop, 50.0, 35.0])
    obs, rewards, terminated, truncated, info = env.step(actions)

    # Classify phase
    if q < 16: phase = "symbiosis"
    elif q < 36: phase = "maturation"
    elif q < 48: phase = "tension"
    elif q < 54: phase = "crisis"
    else: phase = "adjustment"

    phase_data[phase].append(info['mean_trust'])

    if terminated or truncated: break

for phase, trusts in phase_data.items(): if trusts: print(f"{phase}: mean_trust={np.mean(trusts):.3f}")
```

---

## Related Environments

- [SLCD-v0](slcd.md): Samsung-Sony case study (TR-1 value creation)
- [RenaultNissan-v0](renault_nissan.md): Renault-Nissan case study (TR-2 trust dynamics)
- [ApacheProject-v0](apache_project.md): Apache case study (TR-3 collective action)
- [GiftExchange-v0](gift_exchange.md): Asymmetric reciprocity in dyadic setting
- [PlatformEcosystem-v0](platform_ecosystem.md): Platform dynamics (TR-1/TR-2)

---

## References

1. Pant, V. & Yu, E. (2026). Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity. *arXiv:2604.01240*. [Link](https://arxiv.org/abs/2604.01240)
2. Parker, G. G., Van Alstyne, M. W. & Choudary, S. P. (2016). Platform Revolution. W. W. Norton.
3. Rochet, J.-C. & Tirole, J. (2003). Platform Competition in Two-Sided Markets. Journal of the European Economic Association.
