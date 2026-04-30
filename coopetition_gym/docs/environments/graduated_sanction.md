# GraduatedSanction-v0

**Category:** Reciprocity Environment (TR-4)
**Agents:** 6
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/reciprocity_envs.py`

---

## Overview

GraduatedSanction-v0 implements a **six-agent common-pool resource game** with TR-4 graduated reciprocity sanctions. Agents share a common resource and decide how much to contribute. Reciprocity manifests as graduated sanctions: mild response to first defection, escalating with repeated violations.

The environment captures Ostrom's (1990) insight about proportional punishment, effective governance relies on graduated rather than draconian responses to rule violations.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | 6-player Markov Game (general-sum) |
| **Cooperation Structure** | Common-pool resource dilemma |
| **Observability** | Full |
| **Communication** | Implicit |
| **Agent Symmetry** | Symmetric (identical capabilities) |
| **Reward Structure** | Integrated utility with graduated reciprocity |
| **Action Space** | Continuous: $A_i = [0, 100]$ |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T = 200 steps |
| **Canonical Comparison** | Common-pool resource; Ostrom (1990); Fehr & Gächter (2000) |

---

## Formal Specification

### Common-Pool Resource Structure

Six symmetric agents with uniform interdependence:

$$D_{ij} = 0.35 \quad \text{for all } i \neq j$$

Higher baselines ($b_i = 40$) reflect the social expectation in a commons setting.

### Graduated Sanction Mechanism

Graduated sanctions emerge from the interaction of TR-4 parameters: 1. **Lower $\kappa = 0.8$**: The bounded response $\varphi(x) = \tanh(0.8x)$ has a **gentler slope**, producing proportional (not binary) reactions to defection
2. **Long memory $k = 10$**: Extended memory window tracks behavioral patterns over time, enabling escalation based on repeated violations
3. **High $\lambda_R = 1.8$**: Strong reciprocity weight amplifies the aggregate sanction effect across 5 partners
4. **High $\omega = 1.0$**: Maximum dependency amplification in trust gating

### Reciprocity Sensitivity

With $\rho_0 = 0.6$, $\eta = 1.5$, and $D_{ij} = 0.35$:

$$\rho_{ij} = 0.6 \cdot 0.35^{1.5} \approx 0.124 \quad \text{per pair}$$

Low per-pair sensitivity, but summed over 5 partners with $\lambda_R = 1.8$:

$$\text{Maximum aggregate} = 5 \times 1.8 \times T_{ij} \times (1 + 1.0 \times 0.35) \times 0.124 \times 1.0 \approx 1.50$$

Substantial aggregate effect when all 5 partners sanction simultaneously.

---

## Distinction from PublicGoods-v0 (TR-3)

| Aspect | PublicGoods-v0 | GraduatedSanction-v0 |
|--------|----------------|----------------------|
| **Mechanism** | Static TR-3 collective action modifiers | Adaptive TR-4 history-dependent reciprocity |
| **Sanctions** | Fixed free-rider penalties | Graduated proportional sanctions |
| **Adaptation** | Loyalty score adjusts slowly | Memory window ($k=10$) enables rapid response |
| **Escalation** | No escalation, penalty is constant | Repeated defection compounds via memory |
| **Key Equation** | Loyalty modifier (TR-3 Eq 5) | Reciprocity modifier (TR-4 Eq 44) |
| **Agents** | 5 (default) | 6 |

---

## Game-Theoretic Background

### Ostrom's Design Principles

Ostrom (1990) identified **graduated sanctions** as a key institutional design principle for sustainable commons governance: 1. **Proportional monitoring**: All agents observe each other's contributions
2. **Graduated sanctions**: First offenses receive mild punishment
3. **Escalation**: Repeated violations trigger increasingly severe responses
4. **Low-cost enforcement**: Reciprocity provides decentralized sanctions

### Strategic Implications

**Free-Riding Temptation:**
- Each agent faces incentive to under-contribute to the commons
- With 6 agents, each free-rider benefits from 5 others' contributions

**Reciprocity as Governance:**
- 5 partners simultaneously detect and respond to free-riding
- Graduated response ($\kappa = 0.8$) avoids over-punishment of small deviations
- Long memory ($k = 10$) ensures persistent free-riders face escalating sanctions

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("GraduatedSanction-v0")

obs, info = env.reset(seed=42)

# All agents contribute at baseline
for step in range(200): actions = np.full(6, 50.0)
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated: break

print(f"Mean trust: {info['mean_trust']:.3f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 200 | Extended horizon for graduated dynamics |
| `render_mode` | None | Rendering mode |

### TR-4 Parameters

| Parameter | Symbol | Value | Rationale |
|-----------|--------|-------|-----------|
| Base reciprocity | $\rho_0$ | 0.6 | Lower per-pair (but 5 pairs) |
| Dependency elasticity | $\eta$ | 1.5 | Superlinear dependency effect |
| Response sensitivity | $\kappa$ | 0.8 | Gradual response (graduated) |
| Memory window | $k$ | 10 | Long memory for escalation |
| Reciprocity weight | $\lambda_R$ | 1.8 | Strong aggregate reciprocity |
| Dependency amplification | $\omega$ | 1.0 | Maximum dependency boost |

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Includes actions, trust matrix (6×6), reputation (6×6), interdependence (6×6), and step info.

### Action Space

**Type:** `Box`
**Shape:** `(6,)`
**Dtype:** `float32`
**Range:** `[0.0, 100.0]` for each agent

---

## Metrics and Info

The `info` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `mean_trust` | float | Average trust level |
| `cooperation_signals` | dict | Per-pair $s_{ij}$ values (30 pairs) |
| `reciprocity_effects` | dict | Per-pair reciprocity contributions |
| `memory_averages` | dict | Per-pair memory averages |
| `tr4_memory_window` | int | Memory window $k = 10$ |

---

## Key Dynamics

### Graduated Response Profile

The $\kappa = 0.8$ parameter creates a proportional response:

| Defection Magnitude | $\varphi(s)$ | Response Level |
|---------------------|-------------|----------------|
| Small ($s \approx -5$) | $\approx -0.97$ | Near-maximum |
| Moderate ($s \approx -2$) | $\approx -0.92$ | Strong |
| Minor ($s \approx -0.5$) | $\approx -0.38$ | Mild |
| Negligible ($s \approx -0.1$) | $\approx -0.08$ | Minimal |

With standard $\kappa = 1.0$, these responses would be sharper. The lower $\kappa = 0.8$ provides the graduated quality.

### Escalation Through Memory

1. **First defection**: Memory average barely changes → mild sanction
2. **Repeated defection**: Memory average drops → cooperation signal becomes more negative
3. **Persistent defection**: Full memory window contaminated → maximum sanction from all 5 partners

---

## Research Applications

GraduatedSanction-v0 is suitable for studying:

- **Commons Governance**: Can decentralized reciprocity sustain cooperation?
- **Graduated Sanctions**: Does proportional punishment outperform binary?
- **Scalability**: How does 6-agent reciprocity compare to 2-agent?
- **Institutional Design**: Ostrom's principles in computational settings
- **Free-Rider Detection**: Multi-agent monitoring and enforcement

---

## Related Environments

- [PublicGoods-v0](public_goods.md): Static TR-3 collective action variant
- [CoalitionFormation-v0](coalition_formation.md): Dynamic exclusion mechanism
- [IndirectReciprocity-v0](indirect_reciprocity.md): 4-agent reputation dynamics

---

## References

1. Pant, V. & Yu, E. (2026). Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity. *arXiv:2604.01240*. [Link](https://arxiv.org/abs/2604.01240)
2. Ostrom, E. (1990). Governing the Commons: The Evolution of Institutions for Collective Action. Cambridge University Press.
3. Ostrom, E., Walker, J. & Gardner, R. (1992). Covenants With and Without a Sword: Self-Governance Is Possible. American Political Science Review.
4. Fehr, E. & Gächter, S. (2000). Cooperation and Punishment in Public Goods Experiments. American Economic Review.
