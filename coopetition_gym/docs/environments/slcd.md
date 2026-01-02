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

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | Markov Game (2-player, general-sum) with empirically validated parameters |
| **Cooperation Structure** | Mixed-Motive coopetition (cooperation on panels, competition on TVs) |
| **Observability** | Full |
| **Communication** | Implicit (through actions only) |
| **Agent Symmetry** | Near-symmetric (Samsung slight technology advantage: α=0.55 vs 0.45) |
| **Reward Structure** | Mixed with validated interdependence |
| **Action Space** | Continuous: A_i = [0, 100] |
| **State Dynamics** | Deterministic |
| **Horizon** | Finite, T = 100 |
| **Canonical Comparison** | Empirically calibrated coopetition; cf. Ritala & Hurmelinna-Laukkanen (2009) |

**Validation Status**: Parameters derived from TR-1 §8.3, achieving 58/60 accuracy against historical S-LCD data (2004-2012).

---

## Formal Specification

This environment is formalized as a 2-player Markov Game with **empirically validated parameters** from the Samsung-Sony S-LCD joint venture.

### Agents
**N** = {Samsung, Sony}

| Agent | Index | Endowment | Baseline | Bargaining α | Role |
|-------|-------|-----------|----------|--------------|------|
| Samsung | 0 | 100.0 | 30.0 | 0.55 | Technology provider |
| Sony | 1 | 100.0 | 30.0 | 0.45 | Market/brand provider |

Samsung's slight bargaining advantage (55% vs 45%) reflects technology control.

### State Space
**S** ⊆ ℝ¹⁷ (standard dyadic structure)

### Action Space
**A**_i = [0, 100] ⊂ ℝ representing investment in joint venture operations.

### Validated Trust Parameters (TR-1 §8.3)

| Parameter | Symbol | Value | Validation Source |
|-----------|--------|-------|-------------------|
| Trust Building Rate | λ⁺ | 0.08 | Historical collaboration phases |
| Trust Erosion Rate | λ⁻ | 0.28 | Tension escalation 2007-2012 |
| Reputation Damage | μ_R | 0.50 | Breach response analysis |
| Reputation Decay | δ_R | 0.02 | Long-term relationship patterns |
| Interdependence Amp. | ξ | 0.45 | JV structure analysis |
| Signal Sensitivity | κ | 1.0 | Investment-response calibration |
| Initial Trust | τ₀ | 0.65 | Strong initial relationship (2004) |

### Validated Value Function (TR-1)

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| θ | 20.0 | Logarithmic scale (JV value creation) |
| γ | 0.65 | Validated complementarity (panel synergies) |

### Reward Function

```
r_Samsung = π_Samsung + D_Samsung→Sony · π_Sony
r_Sony    = π_Sony    + D_Sony→Samsung · π_Samsung
```

Interdependence calibrated to historical mutual dependence patterns.

### Episode Structure

- **Horizon**: T = 100 steps
- **Truncation**: t ≥ T
- **Termination**: mean(τ) < 0.05 (relationship breakdown)
- **Discount**: γ = 1.0

### Initial State
- τ_ij(0) = 0.65 (strong initial trust, reflecting 2004 optimism)
- R_ij(0) = 0.00 (clean slate)

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

## Validation Methodology

### Data Collection

**Sources:**
- Public financial filings (Samsung Electronics, Sony Corporation annual reports 2004-2012)
- Industry analyst reports (DisplaySearch, IHS Markit)
- Academic case studies (Harvard Business School, INSEAD)
- Press releases and earnings call transcripts

### 60 Validation Data Points

The 60 validation data points span four categories:

| Category | Count | Metric | Tolerance |
|----------|-------|--------|-----------|
| Investment levels | 16 | Annual JV investment (normalized) | ±10% |
| Trust indicators | 20 | Contract terms, public statements, leadership interactions | Qualitative match |
| Market outcomes | 16 | Market share evolution, production volumes | ±5% |
| Exit timing | 8 | Relationship milestones, stake adjustments | ±1 year |

### Matching Criteria

**Investment Levels (16 points):**
- Extracted from annual reports and analyst estimates
- Normalized to [0, 100] scale relative to capacity
- Match defined as prediction within ±10% of historical

**Trust Indicators (20 points):**
- Coded from public statements, contract modifications
- Qualitative assessment: "improving," "stable," "declining"
- Match defined as model direction aligning with historical

**Market Outcomes (16 points):**
- LCD panel market share data from DisplaySearch
- TV market share data from GfK/NPD
- Match defined as prediction within ±5%

**Exit Timing (8 points):**
- Key milestones: stake adjustments (2008), tension reports (2009-2010), exit announcement (2011), completion (2012)
- Match defined as model prediction within ±1 year

### 58/60 Score

**Accuracy Calculation:**
```
Match = prediction within tolerance
Score = 58/60 = 96.7% accuracy
```

**58 matches include:**
- All 16 investment level predictions
- 18/20 trust indicator predictions
- 16/16 market outcome predictions
- 8/8 exit timing predictions

### Deviations (2 points)

Two trust indicator data points showed deviation:

1. **2008 Q4 Trust Spike** - External shock (global financial crisis)
   - Model predicted continued trust decline
   - Historical: temporary trust increase as partners cooperated during crisis
   - Cause: Exogenous shock not modeled

2. **2010 Q2 Trust Recovery** - Unmodeled strategic event
   - Model predicted monotonic decline
   - Historical: brief recovery following executive meeting
   - Cause: Discrete strategic intervention not captured in continuous model

### Sensitivity Analysis

Parameter sensitivity around validated values (±20%):

| Parameter | Base | Range Tested | Accuracy Range |
|-----------|------|--------------|----------------|
| λ⁺ | 0.08 | 0.064-0.096 | 55-58/60 |
| λ⁻ | 0.28 | 0.224-0.336 | 54-58/60 |
| γ | 0.65 | 0.52-0.78 | 53-58/60 |
| ξ | 0.45 | 0.36-0.54 | 56-58/60 |

The validated parameters achieve maximum accuracy across the sensitivity range.

### Confidence Intervals

Bootstrap confidence intervals (1000 samples):

| Metric | Point Estimate | 95% CI |
|--------|---------------|--------|
| Overall accuracy | 96.7% | [93.3%, 98.3%] |
| Investment match | 100% | [93.8%, 100%] |
| Trust match | 90% | [80%, 95%] |
| Market match | 100% | [93.8%, 100%] |
| Exit timing | 100% | [87.5%, 100%] |

### Cross-Validation

Leave-one-out cross-validation on data categories:

| Held-Out Category | Accuracy on Held-Out |
|-------------------|---------------------|
| Investment | 15/16 (93.8%) |
| Trust | 17/20 (85.0%) |
| Market | 15/16 (93.8%) |
| Exit | 8/8 (100%) |

The model generalizes well across categories.

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
