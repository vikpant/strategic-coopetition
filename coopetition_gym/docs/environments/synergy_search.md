# SynergySearch-v0

**Category:** Benchmark Environment
**Agents:** 2
**Difficulty:** Hard
**Source:** `coopetition_gym/envs/benchmark_envs.py`

---

## Overview

SynergySearch-v0 presents an **exploration vs. exploitation challenge** where the complementarity parameter (γ) is hidden from agents. Agents must discover whether they have high or low synergy potential from reward signals alone, then adapt their strategy accordingly.

This environment tests agents' ability to:
1. **Extract information** from reward patterns
2. **Estimate hidden parameters** through interaction
3. **Adapt strategy** based on discovered synergy level

---

## Game-Theoretic Background

### The Synergy Discovery Problem

Real-world parallels:
- **Joint venture exploration**: Firms don't know partnership potential before trying
- **Research collaboration**: Complementarity of skills is discovered through work
- **Merger evaluation**: Synergy is uncertain until integration begins

### Information Economics

The hidden γ parameter creates an **information asymmetry**:
- The environment knows the true γ
- Agents must infer γ from reward signals
- Optimal policy depends on the true γ value

### Strategic Implications

**If γ is high (>0.6):**
- Heavy mutual investment is optimal
- Synergy bonus justifies high cooperation
- "High synergy" equilibrium

**If γ is low (≤0.6):**
- Conservative investment is optimal
- Limited synergy doesn't justify high cooperation
- "Low synergy" equilibrium

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment (gamma is hidden by default)
env = coopetition_gym.make("SynergySearch-v0")

obs, info = env.reset(seed=42)

# True gamma is hidden but revealed in info for analysis
print(f"True gamma: {info['true_gamma']:.3f}")  # For debugging only

# Run episode
for step in range(100):
    actions = np.array([50.0, 50.0])
    obs, rewards, terminated, truncated, info = env.step(actions)

# Check gamma type
print(f"Gamma type: {info['gamma_type']}")  # "high_synergy" or "low_synergy"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 100 | Maximum timesteps per episode |
| `gamma_range` | (0.20, 0.90) | Range for random gamma sampling |
| `reveal_gamma_in_obs` | False | Include gamma in observations |
| `render_mode` | None | Rendering mode |

---

## Hidden Gamma Mechanism

### Random Sampling

Each episode, γ is sampled uniformly:

```python
gamma = np.random.uniform(gamma_range[0], gamma_range[1])
# gamma ∈ [0.20, 0.90]
```

### Gamma Classification

| Range | Classification | Optimal Strategy |
|-------|---------------|------------------|
| γ > 0.60 | High Synergy | Heavy investment |
| γ ≤ 0.60 | Low Synergy | Conservative investment |

### Optional Revelation

For supervised learning or testing:

```python
# Make gamma observable
env = coopetition_gym.make("SynergySearch-v0", reveal_gamma_in_obs=True)

# Gamma is now in the observation vector
obs, info = env.reset()
gamma_in_obs = obs[-1]  # Last element is gamma
```

---

## Observation Space

### Standard Mode (gamma hidden)

| Component | Shape | Description |
|-----------|-------|-------------|
| Actions | (2,) | Last cooperation levels |
| Trust Matrix | (2,2) | Pairwise trust levels |
| Reputation Matrix | (2,2) | Pairwise reputation damage |
| Interdependence | (2,2) | Structural dependencies |
| Step Info | (1,) | Normalized timestep |

**Total dimension**: 17

### Extended Mode (gamma revealed)

Additional component:
| Component | Shape | Description |
|-----------|-------|-------------|
| Gamma | (1,) | True complementarity value |

**Total dimension**: 18

---

## Reward Structure

### Gamma-Dependent Value

Value creation uses the hidden gamma:

```
V(a₁, a₂) = θ × ln(a₁ + a₂) × (1 + γ × complementarity)
```

Where complementarity = min(a₁/e₁, a₂/e₂).

### Reward Variance

Key insight for inference:
- **High gamma**: Larger variance in rewards across cooperation levels
- **Low gamma**: Smaller variance (flatter reward landscape)

Agents can estimate gamma from:
- Absolute reward levels
- Reward changes with action changes
- Reward variance across episodes

---

## Inference Challenge

### Bayesian Perspective

Agents should ideally:
1. Maintain belief distribution over γ
2. Update beliefs based on rewards
3. Select actions that:
   - Are optimal given current beliefs
   - Provide information to refine beliefs

### Practical Approaches

**Probing Strategy:**
- Try high cooperation (e.g., 80%)
- Observe rewards
- Compare to expected rewards under different γ hypotheses

**Gradient Estimation:**
- Try varying cooperation levels
- Estimate ∂reward/∂action
- High gradient suggests high γ

---

## Trust Dynamics

### Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Trust Building Rate | λ⁺ | 0.10 | Standard building |
| Trust Erosion Rate | λ⁻ | 0.30 | Standard erosion |
| Initial Trust | τ₀ | 0.55 | Moderate start |

Trust dynamics are standard, not the main challenge in this environment.

---

## Interdependence Structure

### Symmetric Dependencies

```
D = [[ 0.00,  0.50 ],
     [ 0.50,  0.00 ]]
```

Moderate mutual dependency creates incentive for coordination.

---

## Metrics and Info

The `info` dictionary includes:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `true_gamma` | float | The hidden gamma value |
| `gamma_type` | str | "high_synergy" or "low_synergy" |
| `cumulative_rewards` | list | Reward history for inference |
| `reward_variance` | float | Variance in recent rewards |
| `total_value` | float | Total value created |

---

## Optimal Strategy by Gamma

### High Synergy (γ > 0.6)

```python
# High cooperation pays off
if gamma_type == "high_synergy":
    optimal_coop = 0.75  # 75% of endowment
```

Expected dynamics:
- Rewards increase significantly with cooperation
- Trust builds easily
- Mutual high investment is stable

### Low Synergy (γ ≤ 0.6)

```python
# Conservative investment is optimal
if gamma_type == "low_synergy":
    optimal_coop = 0.45  # 45% of endowment
```

Expected dynamics:
- Rewards are relatively flat
- High investment has low ROI
- Moderate cooperation is stable

---

## Example: Inference-Based Strategy

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("SynergySearch-v0")
obs, info = env.reset(seed=42)

# Probe phase: estimate gamma
probe_rewards = []
probe_actions = [30.0, 50.0, 70.0]

for probe_action in probe_actions:
    actions = np.array([probe_action, probe_action])
    obs, rewards, _, _, info = env.step(actions)
    probe_rewards.append(sum(rewards))

# Infer gamma from reward gradient
gradient = (probe_rewards[2] - probe_rewards[0]) / 40.0  # Per unit action

# High gradient suggests high synergy
estimated_high_synergy = gradient > 0.5

if estimated_high_synergy:
    exploit_action = 75.0
    print("Inferred: HIGH synergy - using heavy investment")
else:
    exploit_action = 45.0
    print("Inferred: LOW synergy - using conservative investment")

# Exploitation phase
for step in range(97):  # Remaining steps
    actions = np.array([exploit_action, exploit_action])
    obs, rewards, terminated, truncated, info = env.step(actions)

print(f"True gamma: {info['true_gamma']:.3f} ({info['gamma_type']})")
print(f"Our inference was: {'CORRECT' if (info['gamma_type'] == 'high_synergy') == estimated_high_synergy else 'WRONG'}")
```

---

## Research Applications

SynergySearch-v0 is suitable for studying:

- **Information Economics**: Learning under uncertainty
- **Bayesian RL**: Belief-based decision making
- **Exploration-Exploitation**: When and how to probe
- **Meta-Learning**: Adapting to new γ values
- **Partner Assessment**: Evaluating collaboration potential

---

## Related Environments

- [TrustDilemma-v0](trust_dilemma.md): Known parameters
- [RecoveryRace-v0](recovery_race.md): Another benchmark challenge
- [SLCD-v0](slcd.md): Fixed validated parameters

---

## References

1. Ghavamzadeh, M. & Engel, Y. (2007). Bayesian Policy Gradient Algorithms. NeurIPS.
2. Duff, M.O. (2002). Optimal Learning: Computational Procedures for Bayes-Adaptive MDPs. UMass Dissertation.
3. Pant, V. & Yu, E. (2025). Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity. arXiv:2510.18802
