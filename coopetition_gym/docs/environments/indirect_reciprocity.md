# IndirectReciprocity-v0

**Category:** Reciprocity Environment (TR-4)
**Agents:** 4
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/reciprocity_envs.py`

---

## Overview

IndirectReciprocity-v0 implements a **four-agent population** with reputation-mediated cooperation and TR-4 reciprocity dynamics. Cooperation with any partner is observed by all members, enabling indirect reciprocity—"I cooperate with you because you cooperated with someone else."

The environment tests whether agents can learn that **reputation matters**: cooperation with any single partner builds reputation that encourages cooperation from all others.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | 4-player Markov Game (general-sum) |
| **Cooperation Structure** | Mixed-Motive with reputation externalities |
| **Observability** | Full (all actions visible to all agents) |
| **Communication** | Implicit (through observed actions) |
| **Agent Symmetry** | Symmetric (identical capabilities) |
| **Reward Structure** | Integrated utility with multi-partner reciprocity |
| **Action Space** | Continuous: $A_i = [0, 100]$ |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T = 150 steps |
| **Canonical Comparison** | Indirect reciprocity; Nowak & Sigmund (1998, 2005); Panchanathan & Boyd (2004) |

---

## Formal Specification

### Population Structure

Four symmetric agents with uniform interdependence:

$$\mathbf{D} = \begin{pmatrix} 0 & 0.4 & 0.4 & 0.4 \\ 0.4 & 0 & 0.4 & 0.4 \\ 0.4 & 0.4 & 0 & 0.4 \\ 0.4 & 0.4 & 0.4 & 0 \end{pmatrix}$$

### Indirect Reciprocity Mechanism

Indirect reciprocity emerges from Eq 44's multi-agent summation. Agent $i$'s reciprocity modifier aggregates signals from **all** partners:

$$U_{\text{recip},i} = \lambda_R \sum_{j \neq i} T_{ij} \cdot (1 + \omega D_{ij}) \cdot \rho_{ij} \cdot \varphi(s_{ij})$$

When agent $j$ cooperates with agent $k$ (not $i$), agent $i$ observes $j$'s high action level. This creates a positive memory average for $j$, generating positive cooperation signals when $i$ evaluates $j$—even without direct interaction history.

### Reciprocity Sensitivity

With $\rho_0 = 0.8$, $\eta = 1.0$, and $D_{ij} = 0.4$:

$$\rho_{ij} = 0.8 \cdot 0.4^{1.0} = 0.32 \quad \text{for all pairs}$$

Lower individual sensitivity, but with 3 partners contributing to each agent's modifier, the **aggregate effect** is substantial.

### TR-4 Parameters

| Parameter | Symbol | Value | Rationale |
|-----------|--------|-------|-----------|
| Base reciprocity | $\rho_0$ | 0.8 | Lower per-pair (but 3 pairs sum) |
| Dependency elasticity | $\eta$ | 1.0 | Linear dependency effect |
| Response sensitivity | $\kappa$ | 1.0 | Standard bounded response |
| Memory window | $k$ | 7 | Longer memory for reputation tracking |
| Reciprocity weight | $\lambda_R$ | 1.5 | Higher weight (reputation matters more) |
| Dependency amplification | $\omega$ | 0.5 | Moderate dependency boost |

---

## Distinction from PlatformEcosystem-v0

| Aspect | PlatformEcosystem-v0 | IndirectReciprocity-v0 |
|--------|----------------------|------------------------|
| **Mechanism** | Market competition (TR-1/TR-2) | Reputation-mediated reciprocity (TR-4) |
| **Topology** | Hub-spoke (platform + developers) | Fully connected peer network |
| **Agent Roles** | Heterogeneous (platform vs. developer) | Homogeneous (all symmetric) |
| **Key Dynamic** | Ecosystem health management | Reputation building across population |
| **Cooperation Driver** | Structural dependency on platform | Indirect reciprocity via observed actions |

---

## Game-Theoretic Background

### Indirect Reciprocity Theory

Nowak & Sigmund (1998, 2005) established that cooperation can be sustained in populations through **image scoring**:

1. **Direct reciprocity**: "I help you because you helped me" (TFT)
2. **Indirect reciprocity**: "I help you because you helped someone" (reputation)

### Strategic Implications

**Reputation as Public Good:**
- Each cooperative act is observed by all 3 other agents
- Cooperation builds reputation, encouraging future cooperation from all partners
- Defection damages reputation with all observers simultaneously

**The Scoring Problem:**
- With $k = 7$ memory window, recent defection is remembered for 7 steps
- Longer memory enables more nuanced reputation assessment
- Higher $\lambda_R = 1.5$ amplifies reputation effects

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("IndirectReciprocity-v0")

obs, info = env.reset(seed=42)

# All agents cooperate
for step in range(150):
    actions = np.array([60.0, 60.0, 60.0, 60.0])
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated:
        break

print(f"Mean trust: {info['mean_trust']:.3f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 150 | Extended horizon for reputation dynamics |
| `render_mode` | None | Rendering mode |

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Includes actions, trust matrix (4×4), reputation (4×4), interdependence (4×4), and step info.

### Action Space

**Type:** `Box`
**Shape:** `(4,)`
**Dtype:** `float32`
**Range:** `[0.0, 100.0]` for each agent

---

## Metrics and Info

The `info` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `mean_trust` | float | Average trust level |
| `cooperation_signals` | dict | Per-pair $s_{ij}$ values (12 pairs) |
| `reciprocity_effects` | dict | Per-pair reciprocity contributions |
| `memory_averages` | dict | Per-pair memory averages |
| `tr4_memory_window` | int | Memory window $k = 7$ |

---

## Key Dynamics

### Reputation Cascade

1. Agent $j$ cooperates with agent $k$ at high level
2. Agents $i$ and $l$ observe $j$'s high action in memory
3. When evaluating $j$, both $i$ and $l$ compute positive $s_{ij}$ and $s_{lj}$
4. Both $i$ and $l$ receive positive reciprocity modifier when cooperating with $j$
5. This encourages all agents to cooperate with $j$—indirect reciprocity

### Defection Contagion

A single defection creates negative signals with **all 3 partners simultaneously**, triggering faster punishment than in 2-agent settings.

---

## Research Applications

IndirectReciprocity-v0 is suitable for studying:

- **Indirect Reciprocity**: Can agents learn reputation-based cooperation?
- **Image Scoring**: Do agents develop consistent cooperation patterns?
- **Population Dynamics**: How does group size affect cooperation stability?
- **Reputation Cascades**: How does one agent's behavior propagate through the network?

---

## Related Environments

- [DynamicPartnerSelection-v0](dynamic_partner_selection.md): Reputation-based matching (TR-1/TR-2)
- [ReciprocalDilemma-v0](reciprocal_dilemma.md): Direct reciprocity baseline
- [GraduatedSanction-v0](graduated_sanction.md): Larger population with graduated sanctions

---

## References

1. Pant, V. & Yu, E. (2026). Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity. *arXiv:2604.01240*. [Link](https://arxiv.org/abs/2604.01240)
2. Nowak, M. A. & Sigmund, K. (1998). Evolution of Indirect Reciprocity by Image Scoring. Nature.
3. Nowak, M. A. & Sigmund, K. (2005). Evolution of Indirect Reciprocity. Nature.
4. Panchanathan, K. & Boyd, R. (2004). Indirect Reciprocity Can Stabilize Cooperation Without the Second-Order Free Rider Problem. Nature.
