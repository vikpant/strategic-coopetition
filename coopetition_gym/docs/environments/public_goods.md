# PublicGoods-v0

**Category:** Collective Action Environment (TR-3)
**Agents:** 5 (configurable)
**Difficulty:** Intermediate
**Source:** `coopetition_gym/envs/collective_action_envs.py`

---

## Overview

PublicGoods-v0 implements a **classic public goods game** with TR-3 collective action modifiers. Agents contribute to a public good that benefits everyone equally, regardless of individual contribution levels.

This environment provides a well-studied baseline for testing collective action mechanisms against classic game theory predictions.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | N-player Markov Game (symmetric) |
| **Cooperation Structure** | Pure public goods with non-excludability |
| **Observability** | Full |
| **Communication** | Implicit |
| **Agent Symmetry** | Symmetric |
| **Reward Structure** | Kept endowment + share of public good |
| **Action Space** | Continuous: $A_i = [0, \text{endowment}]$ |
| **Horizon** | Finite, T = 100 steps |
| **Canonical Comparison** | Linear public goods game; Isaac & Walker (1988) |

---

## Formal Specification

### Public Goods Mechanism

**Contribution:**
Each agent has endowment $e$ and contributes $a_i \in [0, e]$ to the public good.

**Public Good Value:**
$$G = m \cdot \sum_{i=1}^{n} a_i$$

Where $m > 1$ is the multiplier (making cooperation socially beneficial).

**Payoff:**
$$\pi_i = (e - a_i) + \frac{G}{n} = (e - a_i) + \frac{m \cdot \sum_j a_j}{n}$$

### Game-Theoretic Analysis

**Nash Equilibrium:**
With $m < n$ (always true for meaningful games), the dominant strategy is:
$$a_i^* = 0 \quad \text{(contribute nothing)}$$

**Social Optimum:**
Full contribution maximizes total welfare:
$$a_i^{opt} = e$$

**Price of Anarchy:**
$$PoA = \frac{n \cdot e \cdot m}{n \cdot e} = m$$

### TR-3 Loyalty Enhancement

With loyalty mechanisms enabled:
$$U_i = \pi_i + L_i$$

Where $L_i = \theta_i \cdot [\phi_B \cdot \bar{\pi}_{-i} + \phi_C \cdot a_i]$

This can sustain positive contributions when loyalty is high.

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("PublicGoods-v0", n_agents=5, multiplier=2.0)

obs, info = env.reset(seed=42)

# Full cooperation
for step in range(100):
    # Everyone contributes fully
    actions = np.array([50.0] * 5)
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated:
        break

print(f"Total contribution: {info['total_contribution']:.1f}")
print(f"Public good value: {info['public_good']:.1f}")
print(f"Contribution rate: {info['contribution_rate']:.1%}")
print(f"Social efficiency: {info['social_efficiency']:.1%}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 5 | Number of players |
| `multiplier` | 2.0 | Public good multiplier (m) |
| `endowment` | 50.0 | Each agent's starting endowment |
| `phi_B` | 0.6 | Loyalty benefit strength |
| `phi_C` | 0.2 | Cost tolerance strength |
| `max_steps` | 100 | Maximum timesteps |
| `render_mode` | None | Rendering mode |

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Includes actions, trust matrix, reputation, interdependence, loyalty scores, and step info.

### Action Space

**Type:** `Box`
**Shape:** `(n_agents,)`
**Dtype:** `float32`
**Range:** `[0.0, endowment]` for each agent

---

## Metrics and Info

The `info` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `total_contribution` | float | Sum of all contributions |
| `public_good` | float | m × total_contribution |
| `contribution_rate` | float | Total / (n × endowment) |
| `social_efficiency` | float | Actual / maximum welfare |
| `mean_loyalty` | float | Average loyalty score |
| `team_cohesion` | float | Weighted loyalty metric |
| `free_rider_count` | int | Agents contributing below threshold |

---

## Key Dynamics

### The Public Goods Dilemma

**Individual incentive:**
- Keep endowment: payoff = $e$
- Contribute $a$: payoff = $(e - a) + \frac{m \cdot a}{n}$
- Since $\frac{m}{n} < 1$ (when $m < n$), contributing is costly

**Collective benefit:**
- Total welfare from contribution $a$: $n \cdot \frac{m \cdot a}{n} = m \cdot a$
- Since $m > 1$, contributions create net social value
- But individual capture is only $\frac{m}{n}$ of value created

### Loyalty-Mediated Cooperation

TR-3 loyalty mechanisms can overcome the dilemma:

1. **High loyalty ($\theta \approx 1$):**
   - $\phi_B$ bonus from others' payoffs
   - $\phi_C$ bonus from own contribution
   - Sustains positive contributions

2. **Low loyalty ($\theta \approx 0$):**
   - Reverts to standard Nash (zero contribution)
   - No intrinsic cooperation motivation

---

## Example: Varying Multiplier Effects

```python
import coopetition_gym
import numpy as np

multipliers = [1.5, 2.0, 3.0, 4.0]
results = {}

for m in multipliers:
    env = coopetition_gym.make("PublicGoods-v0", multiplier=m)
    obs, info = env.reset(seed=42)

    # Mixed contribution strategy
    for step in range(50):
        actions = np.array([40.0] * 5)  # 80% contribution
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    results[m] = {
        "public_good": info['public_good'],
        "efficiency": info['social_efficiency'],
        "mean_loyalty": info['mean_loyalty'],
    }
    env.close()

for m, data in results.items():
    print(f"Multiplier {m}: PG={data['public_good']:.1f}, Eff={data['efficiency']:.1%}")
```

---

## Research Applications

PublicGoods-v0 is suitable for studying:

- **Classic Public Goods**: Standard game theory benchmarks
- **Mechanism Design**: Effects of multiplier and group size
- **Loyalty Effectiveness**: Can TR-3 sustain contributions?
- **Social Efficiency**: How close to optimum?
- **Free-Rider Detection**: Identifying non-contributors

---

## Literature Connection

### Experimental Economics

The public goods game is one of the most studied experimental economics paradigms:

| Finding | Reference | PublicGoods-v0 Relevance |
|---------|-----------|-------------------------|
| ~50% contribution | Isaac & Walker (1988) | Baseline comparison |
| Decay over time | Ledyard (1995) | Loyalty dynamics |
| Punishment helps | Fehr & Gächter (2000) | Exclusion mechanics |
| Communication helps | Ostrom (2000) | Trust mechanisms |

### Connection to TR-3

PublicGoods-v0 bridges experimental findings with TR-3 theory:
- Loyalty mechanisms formalize "warm-glow" giving
- Cost tolerance captures intrinsic contribution value
- Free-rider detection enables punishment strategies

---

## Related Environments

- [TeamProduction-v0](team_production.md): Production-based variant
- [LoyaltyTeam-v0](loyalty_team.md): Full TR-3 mechanisms
- [CoalitionFormation-v0](coalition_formation.md): Exclusion dynamics

---

## References

1. Pant, V. & Yu, E. (2026). [Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty](https://arxiv.org/abs/2601.16237). arXiv:2601.16237
2. Isaac, R. M. & Walker, J. M. (1988). Group Size Effects in Public Goods Provision: The Voluntary Contributions Mechanism. Quarterly Journal of Economics.
3. Ledyard, J. (1995). Public Goods: A Survey of Experimental Research. Handbook of Experimental Economics.
4. Fehr, E. & Gächter, S. (2000). Cooperation and Punishment in Public Goods Experiments. American Economic Review.
