# SLCD-v0

**Category:** Validated Case Study
**Agents:** 2 (Samsung, Sony)
**Difficulty:** Intermediate
**Source:** `coopetition_gym/envs/case_study_envs.py`

---

## Overview

SLCD-v0 implements the **Samsung-Sony S-LCD Joint Venture** as described in TR-1 (arXiv:2510.18802). This is the **gold standard benchmark** for Coopetition-Gym, with parameters validated against real business data.

The environment models the 2004-2012 joint venture where Samsung and Sony collaborated on LCD panel manufacturing while competing in the consumer electronics market—a quintessential coopetitive relationship.

**Validation Score:** 58/60 against historical data

---

## Historical Background

### The S-LCD Joint Venture (2004-2012)

**Partners:**
- **Samsung Electronics**: World's largest TV manufacturer, leading display technology
- **Sony Corporation**: Premium brand, strong in high-end TV market

**Structure:**
- 50-50 ownership (later adjusted)
- Samsung provided manufacturing technology
- Sony provided brand and market access
- Both competed in final TV products

**Outcome:**
- Initially successful collaboration
- Growing tensions as Samsung's TV business grew
- Sony eventually sold stake (2012)
- Complex coopetitive dynamics throughout

### Why This Case Study?

The S-LCD venture is ideal because:
1. **Well-documented**: Public financial and strategic data available
2. **Clear coopetition**: Cooperation (panels) + competition (TVs)
3. **Dynamic trust**: Trust evolved significantly over time
4. **Quantifiable outcomes**: Investment, market share, and exit data

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("SLCD-v0")

obs, info = env.reset(seed=42)

# Agents are named for clarity
print(f"Agents: {env.possible_agents}")  # ['Samsung', 'Sony']

# Run episode
for step in range(100):
    # Cooperation levels for Samsung and Sony
    samsung_action = 55.0  # 55% of endowment
    sony_action = 45.0     # 45% of endowment

    actions = np.array([samsung_action, sony_action])
    obs, rewards, terminated, truncated, info = env.step(actions)

print(f"Samsung total: {sum(info.get('samsung_rewards', [0]))}")
print(f"Sony total: {sum(info.get('sony_rewards', [0]))}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 100 | Maximum timesteps per episode |
| `trust_enabled` | True | Enable trust dynamics |
| `render_mode` | None | Rendering mode |

---

## Agent Configuration (TR-1 §8.3)

### Endowments

| Agent | Role | Endowment | Interpretation |
|-------|------|-----------|----------------|
| Samsung (0) | Technology Provider | 100.0 | Manufacturing capacity |
| Sony (1) | Market Partner | 100.0 | Market access/brand |

### Bargaining Shares

| Agent | Alpha | Interpretation |
|-------|-------|----------------|
| Samsung | 0.55 | Slightly higher due to technology control |
| Sony | 0.45 | Brand premium but less operational control |

Samsung's slight advantage reflects its control over manufacturing technology.

### Baselines

| Agent | Baseline | Interpretation |
|-------|----------|----------------|
| Samsung | 30.0 | 30% minimum viable cooperation |
| Sony | 30.0 | 30% minimum viable cooperation |

Actions below baseline are considered defection.

---

## Interdependence Structure

### S-LCD Specific Dependencies

Created using `create_slcd_interdependence()`:

```
D = [[ 0.00,  D_Samsung_Sony ],
     [ D_Sony_Samsung,  0.00 ]]
```

**Validated Values:**
- Samsung's dependency on Sony's market access
- Sony's dependency on Samsung's technology
- Asymmetric but both parties are invested

---

## Trust Dynamics

### Parameters (TR-1 Validated)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Trust Building Rate | λ⁺ | 0.08 | TR-1 §8.3 |
| Trust Erosion Rate | λ⁻ | 0.28 | TR-1 §8.3 |
| Reputation Damage | μ_R | 0.50 | TR-1 §8.3 |
| Reputation Decay | δ_R | 0.02 | TR-1 §8.3 |
| Interdependence Amp. | ξ | 0.45 | TR-1 §8.3 |
| Signal Sensitivity | κ | 1.0 | TR-1 §8.3 |
| Initial Trust | τ₀ | 0.65 | Historical: Strong initial relationship |

### Historical Trust Evolution

The real S-LCD venture showed:
- **2004-2006**: High trust, successful collaboration
- **2007-2009**: Growing tensions as Samsung's TV share grew
- **2010-2012**: Trust decline leading to Sony's exit

---

## Value Function (TR-1)

### Logarithmic Specification

| Parameter | Value | Source |
|-----------|-------|--------|
| θ | 20.0 | TR-1 §8.3 |
| γ | 0.65 | Validated complementarity |

### Value Creation

```
V(a_S, a_Y) = θ × ln(a_S + a_Y) × (1 + γ × complementarity)
```

Where a_S = Samsung's action, a_Y = Sony's action.

---

## Validation Details

### 58/60 Score

The environment was validated against 60 historical data points:
- Investment levels
- Market share evolution
- Trust indicators (public statements, contract terms)
- Exit timing

**58 data points matched** the model predictions within tolerance.

### Deviations

2 data points showed deviation due to:
- External market shocks (2008 financial crisis)
- Unmodeled strategic events

---

## Metrics and Info

The `info` dictionary includes:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `mean_trust` | float | Average trust level |
| `samsung_investment` | float | Samsung's current action |
| `sony_investment` | float | Sony's current action |
| `total_value` | float | Total value created |
| `samsung_payoff` | float | Samsung's current payoff |
| `sony_payoff` | float | Sony's current payoff |

---

## Reproducing Historical Dynamics

### Early Phase (High Cooperation)

```python
# 2004-2006: Both partners invest heavily
samsung_action = 70.0  # 70% cooperation
sony_action = 65.0     # 65% cooperation
```

Expected outcome:
- Trust rises from 0.65 toward 0.80
- High value creation
- Mutual benefit

### Middle Phase (Growing Tension)

```python
# 2007-2009: Samsung invests more, Sony becomes cautious
samsung_action = 60.0  # Samsung maintains
sony_action = 45.0     # Sony pulls back
```

Expected outcome:
- Trust begins to erode
- Asymmetric payoffs emerge
- Sony's caution reflects competitive concerns

### Late Phase (Pre-Exit)

```python
# 2010-2012: Declining cooperation
samsung_action = 40.0  # Samsung reduces
sony_action = 35.0     # Sony minimizes
```

Expected outcome:
- Trust falls significantly
- Value creation declines
- Relationship approaches termination

---

## Example: Historical Simulation

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("SLCD-v0")
obs, info = env.reset(seed=42)

# Simulate historical phases
phases = [
    ("Early (2004-2006)", 30, (70.0, 65.0)),
    ("Middle (2007-2009)", 35, (60.0, 45.0)),
    ("Late (2010-2012)", 35, (40.0, 35.0))
]

total_samsung = 0
total_sony = 0

for phase_name, steps, (samsung, sony) in phases:
    phase_samsung = 0
    phase_sony = 0

    for step in range(steps):
        actions = np.array([samsung, sony])
        obs, rewards, terminated, truncated, info = env.step(actions)
        phase_samsung += rewards[0]
        phase_sony += rewards[1]

    print(f"{phase_name}:")
    print(f"  Trust: {info['mean_trust']:.3f}")
    print(f"  Samsung: {phase_samsung:.1f}, Sony: {phase_sony:.1f}")

    total_samsung += phase_samsung
    total_sony += phase_sony

print(f"\nTotal: Samsung={total_samsung:.1f}, Sony={total_sony:.1f}")
```

---

## Research Applications

SLCD-v0 is the recommended environment for:

- **Model Validation**: Testing theoretical predictions
- **Benchmark Comparisons**: Standardized evaluation
- **Case Study Analysis**: Studying real coopetition dynamics
- **Policy Analysis**: Evaluating intervention strategies

---

## Comparison with Other Environments

| Feature | SLCD-v0 | TrustDilemma-v0 |
|---------|---------|-----------------|
| Validation | 58/60 historical | Theoretical |
| Agents | Named (Samsung, Sony) | Generic (0, 1) |
| Parameters | Fixed (validated) | Default (configurable) |
| Trust Dynamics | Moderate | High |
| Recommended For | Benchmarking | Algorithm development |

---

## Related Environments

- [RenaultNissan-v0](renault_nissan.md): Another validated case study
- [TrustDilemma-v0](trust_dilemma.md): Simpler theoretical version
- [RecoveryRace-v0](recovery_race.md): Crisis recovery dynamics

---

## References

1. Pant, V. & Yu, E. (2025). [Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity](https://arxiv.org/abs/2510.18802). arXiv:2510.18802
2. Samsung Electronics & Sony Corporation. (2004). S-LCD Joint Venture Announcement.
3. Ritala, P. & Hurmelinna-Laukkanen, P. (2009). What's in it for me? Creating and appropriating value in innovation-related coopetition. Technovation.
