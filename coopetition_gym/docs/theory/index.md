# Theoretical Foundations

**Mathematical Framework for Computational Coopetition**

This section provides comprehensive documentation of the theoretical foundations underlying Coopetition-Gym, derived from the published technical reports on computational foundations for strategic coopetition.

---

## Overview

Coopetition-Gym implements a mathematically rigorous framework bridging two traditions:

1. **Conceptual Modeling** (*i*\* Framework): Rich qualitative representations of strategic dependencies and actor relationships
2. **Game Theory**: Precise quantitative analysis of strategic interactions and equilibrium behavior

The synthesis produces environments where:
- Structural dependencies from organizational analysis inform utility functions
- Trust dynamics evolve based on observed behavior
- Equilibrium analysis incorporates interdependence and complementarity
- Parameters are validated against real business partnerships

### Research Program Architecture

![Research Program Architecture](../assets/images/manim/architecture_diagram.gif)

*Animated four-pillar research program architecture. Pillars 1 (Interdependence & Complementarity) and 2 (Trust & Reputation) are implemented (green); Pillars 3 (Team Production) and 4 (Sequential Reciprocity) are planned (gray). The Coopetitive Equilibrium Framework (blue bar) integrates all pillars.*

---

## Research Program Structure

The theoretical foundations are organized into four pillars, of which **Pillars 1 and 2** are currently implemented:

### Implemented Pillars

| Pillar | Technical Report | Focus | Status |
|--------|-----------------|-------|--------|
| **1** | [TR-2025-01](https://arxiv.org/abs/2510.18802) | Interdependence & Complementarity | ✓ Implemented |
| **2** | [TR-2025-02](https://arxiv.org/abs/2510.24909) | Trust & Reputation Dynamics | ✓ Implemented |

### Planned Pillars

| Pillar | Technical Report | Focus | Status |
|--------|-----------------|-------|--------|
| **3** | TR-2025-03 (Draft) | Team Production & Loyalty | Planned |
| **4** | TR-2025-04 (Draft) | Sequential Interaction & Reciprocity | Planned |

See [Implementation Roadmap](../roadmap.md) for development timeline and planned features.

---

## Theory Documentation

### Core Mathematical Framework

| Document | Content | Audience |
|----------|---------|----------|
| [Interdependence Framework](interdependence.md) | Structural dependencies, *i*\* translation, D matrix | Researchers, Advanced Users |
| [Value Creation & Complementarity](value_creation.md) | Value functions, synergy, superadditivity | Researchers, Economists |
| [Trust Dynamics](trust_dynamics.md) | Two-layer trust, asymmetric updating, hysteresis | Researchers, Behavioral Scientists |
| [Parameter Reference](parameters.md) | Validated values, calibration guidance | All Users |

### Quick Reference

**For Practitioners** seeking to use the environments:
- Start with [Parameter Reference](parameters.md) for recommended values
- Review [Quick Start](../tutorials/quickstart.md) for implementation patterns

**For Researchers** seeking to extend the framework:
- Study [Interdependence Framework](interdependence.md) for Pillar 1 foundations
- Study [Trust Dynamics](trust_dynamics.md) for Pillar 2 foundations
- Review original technical reports for complete proofs and derivations

---

## Mathematical Notation

The following notation is used consistently throughout the documentation:

### Indices and Sets

| Symbol | Definition |
|--------|------------|
| $N$ | Number of agents |
| $i, j \in \{1, \ldots, N\}$ | Agent indices |
| $t \in \{0, 1, 2, \ldots\}$ | Time period |
| $d \in \mathcal{D}_i$ | Dependum (goal, task, resource) in agent $i$'s goal set |

### Actions and Payoffs

| Symbol | Definition | Range |
|--------|------------|-------|
| $a_i$ | Agent $i$'s action (cooperation/investment level) | $[0, e_i]$ |
| $\mathbf{a} = (a_1, \ldots, a_N)$ | Action profile (all agents) | $\prod_i [0, e_i]$ |
| $e_i$ | Agent $i$'s endowment | $\mathbb{R}^+$ |
| $\pi_i(\mathbf{a})$ | Agent $i$'s private payoff | $\mathbb{R}$ |
| $U_i(\mathbf{a})$ | Agent $i$'s integrated utility | $\mathbb{R}$ |

### Interdependence (Pillar 1)

| Symbol | Definition | Range |
|--------|------------|-------|
| $D_{ij}$ | Interdependence coefficient ($i$ depends on $j$) | $[0, 1]$ |
| $w_d$ | Importance weight of dependum $d$ | $\mathbb{R}^+$ |
| $\text{Dep}(i,j,d)$ | Dependency indicator (binary) | $\{0, 1\}$ |
| $\text{crit}(i,j,d)$ | Criticality factor | $[0, 1]$ |
| $\mathbf{D}$ | Interdependence matrix | $[0,1]^{N \times N}$ |

### Value Functions (Pillar 1)

| Symbol | Definition | Range |
|--------|------------|-------|
| $V(\mathbf{a} \mid \gamma)$ | Total value created | $\mathbb{R}^+$ |
| $f_i(a_i)$ | Individual value contribution | $\mathbb{R}^+$ |
| $g(a_1, \ldots, a_N)$ | Synergy function | $\mathbb{R}^+$ |
| $\gamma$ | Complementarity parameter | $[0, 1]$ |
| $\theta$ | Logarithmic scale parameter | $\mathbb{R}^+$ |
| $\beta$ | Power function exponent | $(0, 1)$ |
| $\alpha_i$ | Agent $i$'s share of synergistic value | $[0, 1]$ |

### Trust Dynamics (Pillar 2)

| Symbol | Definition | Range |
|--------|------------|-------|
| $T_{ij}^t$ | Immediate trust ($i$ toward $j$ at time $t$) | $[0, 1]$ |
| $R_{ij}^t$ | Reputation damage ($j$'s violations from $i$'s view) | $[0, 1]$ |
| $\Theta_{ij}^t$ | Trust ceiling | $[0, 1]$ |
| $s_{ij}^t$ | Cooperation signal | $(-1, 1)$ |
| $\lambda^+$ | Trust building rate | $(0, 1)$ |
| $\lambda^-$ | Trust erosion rate | $(0, 1)$ |
| $\mu_R$ | Reputation damage severity | $(0, 1)$ |
| $\delta_R$ | Reputation decay rate | $(0, 1)$ |
| $\xi$ | Interdependence amplification factor | $[0, 1]$ |
| $\kappa$ | Signal sensitivity | $\mathbb{R}^+$ |

---

## Core Equations Summary

### Pillar 1: Interdependence & Complementarity

**Interdependence Matrix** (Equation 1, TR-1):

$$\Large D_{ij} = \frac{\sum_{d \in \mathcal{D}_i} w_d \cdot \text{Dep}(i,j,d) \cdot \text{crit}(i,j,d)}{\sum_{d \in \mathcal{D}_i} w_d}$$

**Value Creation** (Equation 2, TR-1):

$$\Large V(\mathbf{a} \mid \gamma) = \sum_{i=1}^{N} f_i(a_i) + \gamma \cdot g(a_1, \ldots, a_N)$$

**Logarithmic Individual Value** (Equation 6, TR-1):

$$\Large f_i(a_i) = \theta \cdot \ln(1 + a_i) \quad \text{where } \theta = 20.0$$

**Power Individual Value** (Equation 3, TR-1):

$$\Large f_i(a_i) = a_i^{\beta} \quad \text{where } \beta = 0.75$$

**Geometric Mean Synergy** (Equation 4, TR-1):

$$\Large g(a_1, \ldots, a_N) = \left(\prod_{i=1}^{N} a_i\right)^{1/N}$$

**Private Payoff** (Equation 11, TR-1):

$$\Large \pi_i(\mathbf{a}) = e_i - a_i + f_i(a_i) + \alpha_i \left[V(\mathbf{a}) - \sum_{j=1}^{N} f_j(a_j)\right]$$

**Integrated Utility** (Equation 13, TR-1):

$$\Large U_i(\mathbf{a}) = \pi_i(\mathbf{a}) + \sum_{j \neq i} D_{ij} \cdot \pi_j(\mathbf{a})$$

### Pillar 2: Trust Dynamics

**Cooperation Signal** (Equation 4, TR-2):

$$\Large s_{ij}^t = \tanh\left(\kappa \cdot (a_j^t - a_j^{\text{baseline}})\right)$$

**Trust Evolution** (Equation 5, TR-2):

$$\Large T_{ij}^{t+1} = T_{ij}^t + \Delta T_{ij}^t$$

where:

$$\Large \Delta T_{ij}^t = \begin{cases} \lambda^+ \cdot s \cdot (\Theta - T) & \text{if } s > 0 \text{ (building)} \\ -\lambda^- \cdot |s| \cdot T \cdot (1 + \xi \cdot D_{ij}) & \text{if } s \leq 0 \text{ (erosion)} \end{cases}$$

**Trust Ceiling** (Equation 7, TR-2):

$$\Large \Theta_{ij}^t = 1 - R_{ij}^t$$

**Reputation Evolution** (Equation 8, TR-2):

$$\Large R_{ij}^{t+1} = R_{ij}^t + \Delta R_{ij}^t - \delta_R \cdot R_{ij}^t$$

where:

$$\Large \Delta R_{ij}^t = \begin{cases} \mu_R \cdot |s| \cdot (1 - R) & \text{if } s < 0 \text{ (damage)} \\ 0 & \text{if } s \geq 0 \text{ (no damage)} \end{cases}$$

---

## Validation Methodology

The framework employs **dual-track validation**:

### Track 1: Experimental Robustness

Systematic parameter sweeps ensure phenomena emerge robustly:

| Validation Set | Configurations | Purpose |
|----------------|----------------|---------|
| TR-1 Validation | 22,000+ trials | Value function robustness |
| TR-2 Validation | 78,125 configs | Trust dynamics robustness |
| Benchmark Suite | 760 experiments | Algorithm performance |

### Track 2: Empirical Case Studies

Real-world validation against documented business partnerships:

| Case Study | Period | Validation Score | Dynamics Validated |
|------------|--------|------------------|-------------------|
| Samsung-Sony S-LCD | 2004-2011 | 58/60 (96.7%) | Interdependence, complementarity |
| Renault-Nissan Alliance | 1999-2025 | 49/60 (81.7%) | Trust evolution, crisis, recovery |

### Statistical Significance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| p-value | < 0.001 | Highly significant |
| Cohen's d | 9.87 | Very large effect size |
| Negativity Ratio | 3.0 median | Consistent with behavioral economics |

---

## Theoretical Assumptions

The framework makes the following key assumptions:

### Rationality Assumptions

1. **Bounded Rationality**: Agents optimize utility but with limited information
2. **Forward-Looking**: Agents consider future consequences (discount factor β ≈ 0.95)
3. **Observable Actions**: Cooperation levels are observable (no hidden actions)

### Structural Assumptions

1. **Asymmetric Dependencies**: $D_{ij} \neq D_{ji}$ in general
2. **Stable Structure**: Interdependence matrix $\mathbf{D}$ is fixed within episodes
3. **Continuous Actions**: Cooperation levels are continuous, not discrete

### Trust Assumptions

1. **Negativity Bias**: Trust erodes faster than it builds ($\lambda^- > \lambda^+$)
2. **Path Dependence**: Historical violations constrain future trust (hysteresis)
3. **Bilateral Trust**: $T_{ij} \neq T_{ji}$ (trust is not automatically symmetric)

### Limitations

1. **No Communication**: Agents cannot explicitly signal intentions
2. **No Contracting**: No binding commitment mechanisms (Pillar 4 addresses this)
3. **Homogeneous Agents**: Within-agent-type homogeneity assumed
4. **Western Business Context**: Validated primarily on Western partnerships

---

## Citation

If you use the theoretical framework in your research, please cite:

```bibtex
@article{pant2025tr1,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Interdependence and Complementarity},
  author={Pant, Vik and Yu, Eric},
  journal={arXiv preprint arXiv:2510.18802},
  year={2025}
}

@article{pant2025tr2,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Trust and Reputation Dynamics},
  author={Pant, Vik and Yu, Eric},
  journal={arXiv preprint arXiv:2510.24909},
  year={2025}
}
```

---

## Navigation

### Theory Documents
- [Interdependence Framework](interdependence.md)
- [Value Creation & Complementarity](value_creation.md)
- [Trust Dynamics](trust_dynamics.md)
- [Parameter Reference](parameters.md)

### Related Documentation
- [Documentation Home](../index.md)
- [Implementation Roadmap](../roadmap.md)
- [Environment Reference](../environments/index.md)
- [Benchmark Results](../benchmarks/index.md)
