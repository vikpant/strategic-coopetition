# ApacheProject-v0

**Category:** Validated Case Study (TR-3)
**Agents:** 8-40 (phase-dependent)
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/collective_action_envs.py`

---

## Overview

ApacheProject-v0 is a **validated case study** based on the Apache HTTP Server project (1995-2023). This environment reproduces the **52/60 validation score** from TR-3, modeling contributor dynamics across four project phases.

The environment tests whether agents can learn phase-appropriate contribution strategies that match historically observed patterns.

---

## Validation Score

**TR-3 Validation: 52/60 (86.7%)**

This environment's parameters are calibrated against real Apache project data, making it a gold standard for TR-3 research.

---

## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | N-player Markov Game (phase-structured) |
| **Cooperation Structure** | Open source contribution dynamics |
| **Observability** | Full |
| **Communication** | Implicit |
| **Agent Symmetry** | Symmetric within phase |
| **Reward Structure** | Team production with phase-specific loyalty |
| **Action Space** | Continuous: $A_i = [0, 50]$ |
| **Horizon** | Finite, T = 60 steps (monthly resolution) |
| **Canonical Comparison** | Open source dynamics; Lerner & Tirole (2002) |

---

## Project Phases

### Phase Configuration

| Phase | Years | Core Team Size | Loyalty Multiplier | Expected Effort |
|-------|-------|----------------|-------------------|-----------------|
| **Emergence** | 1995-1999 | 8 | 1.00 | 35.0 |
| **Growth** | 2000-2005 | 15 | 0.85 | 30.0 |
| **Maturity** | 2006-2015 | 40 | 0.70 | 22.0 |
| **Evolution** | 2016-2023 | 35 | 0.60 | 18.0 |

### Phase Dynamics

**Emergence (1995-1999):**
- Small core team of highly committed founders
- High individual impact on project success
- Maximum loyalty potential (1.0)

**Growth (2000-2005):**
- Rapid contributor expansion
- Dilution of founding culture
- Moderate loyalty (0.85 cap)

**Maturity (2006-2015):**
- Large stable contributor base
- Established norms and processes
- Lower but stable loyalty (0.70 cap)

**Evolution (2016-2023):**
- Gradual transition to maintenance
- Legacy codebase dynamics
- Reduced loyalty ceiling (0.60)

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment for specific phase
env = coopetition_gym.make("ApacheProject-v0", phase="maturity")

obs, info = env.reset(seed=42)

print(f"Phase: {info['phase']}")
print(f"Years: {info['phase_years']}")
print(f"Core team size: {env.n_agents}")
print(f"Expected effort: {info['expected_effort']:.1f}")

# Run episode
for step in range(60):
    # Contribute at expected level
    expected = info['expected_effort']
    actions = np.full(env.n_agents, expected)
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated:
        break

print(f"Validation accuracy: {info['validation_accuracy']:.1%}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phase` | "maturity" | Project phase to simulate |
| `max_steps` | 60 | Monthly timesteps |
| `render_mode` | None | Rendering mode |

### Phase Selection

```python
# Emergence phase (small, high-loyalty team)
env = coopetition_gym.make("ApacheProject-v0", phase="emergence")

# Growth phase (expanding team)
env = coopetition_gym.make("ApacheProject-v0", phase="growth")

# Maturity phase (large stable team)
env = coopetition_gym.make("ApacheProject-v0", phase="maturity")

# Evolution phase (maintenance mode)
env = coopetition_gym.make("ApacheProject-v0", phase="evolution")
```

---

## Spaces

### Observation Space

**Type:** `Box`
**Dtype:** `float32`

Dimension varies by phase (based on team size).

### Action Space

**Type:** `Box`
**Shape:** `(n_agents,)` where n_agents is phase-specific
**Dtype:** `float32`
**Range:** `[0.0, 50.0]` for each agent

---

## Metrics and Info

The `info` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `phase` | str | Current phase name |
| `phase_years` | str | Historical year range |
| `phase_loyalty` | float | Maximum loyalty for phase |
| `expected_effort` | float | Historically calibrated effort |
| `effort_deviation` | float | Actual vs expected deviation |
| `validation_accuracy` | float | 1 - deviation (higher is better) |
| `team_output` | float | Production Q(a) |
| `mean_loyalty` | float | Average loyalty score |

---

## Key Dynamics

### Phase-Specific Loyalty

Loyalty is capped by the phase multiplier:
```python
loyalty_scores = np.clip(loyalty_scores, 0.0, phase_loyalty_multiplier)
```

This captures the empirical observation that project maturity affects contributor commitment levels.

### Validation Metric

The environment tracks how closely agent behavior matches historical patterns:
```python
deviation = |mean_effort - expected_effort| / expected_effort
validation_accuracy = max(0, 1 - deviation)
```

High validation accuracy indicates realistic simulation.

---

## Research Applications

ApacheProject-v0 is suitable for studying:

- **Open Source Dynamics**: Contributor behavior patterns
- **Phase Transitions**: How projects evolve
- **Loyalty Evolution**: Long-term commitment dynamics
- **Model Validation**: Comparing simulated vs historical data
- **Organizational Lifecycle**: Growth and maturity effects

---

## Empirical Validation

### TR-3 Validation Framework (60 points)

| Category | Points | Description |
|----------|--------|-------------|
| Free-riding baseline | 10 | Matches theoretical equilibrium |
| Loyalty dynamics | 15 | Evolution patterns match data |
| Effort differentiation | 15 | High vs low loyalty difference |
| Team size effects | 10 | Scaling behavior validated |
| Mechanism synergy | 10 | Combined effects accurate |

**Apache Score: 52/60 (86.7%)**

### Key Validated Behaviors

1. **Effort differentiation (4.12× median)**: High-loyalty contributors give ~4× more effort
2. **Phase-appropriate loyalty**: Loyalty ceilings match historical observations
3. **Team size effects**: Larger teams show expected per-capita reduction
4. **Free-riding baseline**: Matches theoretical predictions (99.7% accuracy)

---

## Example: Cross-Phase Comparison

```python
import coopetition_gym
import numpy as np

phases = ["emergence", "growth", "maturity", "evolution"]
results = {}

for phase in phases:
    env = coopetition_gym.make("ApacheProject-v0", phase=phase)
    obs, info = env.reset(seed=42)

    # Cooperative strategy
    for step in range(60):
        actions = np.full(env.n_agents, 30.0)
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    results[phase] = {
        "team_size": env.n_agents,
        "max_loyalty": info['phase_loyalty'],
        "mean_loyalty": info['mean_loyalty'],
        "team_output": info['team_output'],
    }
    env.close()

for phase, data in results.items():
    print(f"{phase}: {data}")
```

---

## Related Environments

- [TeamProduction-v0](team_production.md): Simplified baseline
- [LoyaltyTeam-v0](loyalty_team.md): Configurable loyalty
- [CoalitionFormation-v0](coalition_formation.md): Dynamic membership

---

## References

1. Pant, V. & Yu, E. (2026). [Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty](https://arxiv.org/abs/2601.16237). arXiv:2601.16237
2. Mockus, A., Fielding, R., & Herbsleb, J. (2002). Two Case Studies of Open Source Software Development: Apache and Mozilla. ACM TOSEM.
3. Lerner, J. & Tirole, J. (2002). Some Simple Economics of Open Source. Journal of Industrial Economics.
