# RecoveryRace-v0

**Category:** Benchmark Environment
**Agents:** 2
**Difficulty:** Hard
**Source:** `coopetition_gym/envs/benchmark_envs.py`

---

## Overview

RecoveryRace-v0 models **post-crisis trust recovery** between two agents after a major breach. Starting from a damaged state (low trust, high reputation damage), agents must find the optimal sequence of cooperative actions to rebuild the relationship.

This environment is specifically designed to test understanding of **TR-2 trust dynamics** and the mathematical constraints on recovery. The key insight is that reputation damage creates a **trust ceiling** that limits how much trust can be recovered.

---

## Game-Theoretic Background

### Post-Crisis Dynamics

Real-world examples:
- Business partners after contract violation
- Firms after a major scandal
- Countries after diplomatic crisis

### The Recovery Challenge

After a crisis:
1. **Trust is low**: Recent violations have eroded confidence
2. **Reputation is damaged**: Past behavior creates a ceiling on recovery
3. **Recovery is slow**: Trust builds slowly but can erode quickly
4. **Patience required**: Reputation damage decays very slowly

### Mathematical Constraints

The trust ceiling is:
```
Θ_ij = 1 - R_ij
```

Where R_ij is reputation damage. If R = 0.50, trust cannot exceed 0.50 until reputation heals.

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("RecoveryRace-v0")

obs, info = env.reset(seed=42)

print(f"Starting trust: {info['mean_trust']:.2f}")
print(f"Starting reputation damage: {info['mean_reputation_damage']:.2f}")
print(f"Trust ceiling: {1 - info['mean_reputation_damage']:.2f}")
print(f"Recovery target: 0.90")

# Run recovery attempt
for step in range(150):
    # Consistent high cooperation for recovery
    actions = np.array([80.0, 80.0])  # 80% cooperation
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated:
        print(f"Recovery achieved at step {step}!")
        break

if not terminated:
    print(f"Final trust: {info['mean_trust']:.3f}")
    print(f"Recovery target not reached")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 150 | Extended horizon for recovery |
| `initial_trust` | 0.25 | Post-crisis low trust |
| `initial_reputation_damage` | 0.50 | High starting damage |
| `recovery_target` | 0.90 | Trust level to reach |
| `render_mode` | None | Rendering mode |

---

## Initial State

### Crisis Starting Point

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Initial Trust | 0.25 | Very low (crisis aftermath) |
| Reputation Damage | 0.50 | High (serious past violations) |
| Trust Ceiling | 0.50 | Cannot exceed until rep heals |

### The Ceiling Problem

With initial reputation damage of 0.50:
- Trust ceiling = 1 - 0.50 = 0.50
- Target of 0.90 is IMPOSSIBLE initially
- Must wait for reputation to decay

---

## Trust Dynamics

### Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Trust Building Rate | λ⁺ | 0.08 | Slow recovery |
| Trust Erosion Rate | λ⁻ | 0.35 | Re-violation is costly |
| Reputation Damage | μ_R | 0.70 | New violations add significant damage |
| Reputation Decay | δ_R | 0.01 | Very slow healing |
| Interdependence Amp. | ξ | 0.40 | Moderate amplification |
| Signal Sensitivity | κ | 1.0 | Standard sensitivity |

### Key Insight: Patience

With δ_R = 0.01, reputation heals by ~1% per step.

To reduce reputation from 0.50 to 0.10 (ceiling = 0.90):
- Need: 0.50 → 0.10 = reduction of 0.40
- At 1% per step: ~40 steps minimum
- Plus trust building time

**Recovery is mathematically constrained to be slow.**

---

## Termination Conditions

### Success (Termination)

Episode terminates successfully when:

```python
if mean_trust >= recovery_target:  # 0.90
    terminated = True
    # Recovery achieved!
```

### Time Limit (Truncation)

Episode truncates at `max_steps` (150) if target not reached.

### Re-Violation (Termination)

If agents defect and trust collapses below 0.05:

```python
if mean_trust < 0.05:
    terminated = True
    # Recovery failed - relationship ended
```

---

## Interdependence Structure

### Symmetric Dependencies

```
D = [[ 0.00,  0.55 ],
     [ 0.55,  0.00 ]]
```

Both agents are moderately dependent on each other, creating mutual incentive for recovery.

---

## Value Function

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| θ | 20.0 | Standard logarithmic scale |
| γ | 0.60 | Moderate complementarity |

---

## Metrics and Info

The `info` dictionary includes:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `mean_trust` | float | Average trust level |
| `mean_reputation_damage` | float | Average reputation damage |
| `trust_ceiling` | float | 1 - mean_reputation_damage |
| `recovery_progress` | float | trust / recovery_target |
| `peak_trust` | float | Highest trust achieved |
| `recovery_step` | int | Step when target first reached (or None) |

---

## Optimal Recovery Strategy

### Phase 1: Wait for Ceiling (Steps 0-30)

- Cooperate moderately (60-70%)
- Don't waste resources on impossible trust gains
- Let reputation decay raise the ceiling

### Phase 2: Aggressive Building (Steps 30-80)

- Cooperate highly (80-90%)
- Trust can now grow toward ceiling
- Each cooperative step builds incrementally

### Phase 3: Sustain (Steps 80+)

- Maintain moderate cooperation
- Prevent any trust erosion
- Coast to target as reputation heals

---

## Example: Optimal Recovery Strategy

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("RecoveryRace-v0")
obs, info = env.reset(seed=42)

recovery_achieved = False

for step in range(150):
    ceiling = 1 - info['mean_reputation_damage']
    current_trust = info['mean_trust']

    # Phase-based strategy
    if current_trust >= ceiling - 0.05:
        # Near ceiling: moderate cooperation (wait for ceiling to rise)
        coop_level = 0.6
    elif current_trust < ceiling - 0.1:
        # Below ceiling: aggressive building
        coop_level = 0.85
    else:
        # Approaching ceiling: high cooperation
        coop_level = 0.75

    actions = np.array([100.0 * coop_level, 100.0 * coop_level])
    obs, rewards, terminated, truncated, info = env.step(actions)

    if step % 20 == 0:
        print(f"Step {step}: Trust={info['mean_trust']:.3f}, "
              f"Ceiling={ceiling:.3f}, Progress={info['recovery_progress']:.1%}")

    if terminated and info['mean_trust'] >= 0.90:
        print(f"\nRecovery achieved at step {step}!")
        recovery_achieved = True
        break

if not recovery_achieved:
    print(f"\nRecovery not achieved. Final trust: {info['mean_trust']:.3f}")
```

---

## Research Applications

RecoveryRace-v0 is suitable for studying:

- **Trust Repair**: Strategies after violations
- **Constrained Optimization**: Planning under mathematical constraints
- **Long-horizon Planning**: Patience and delayed gratification
- **TR-2 Dynamics**: Understanding trust ceiling mechanics
- **Crisis Management**: Post-crisis relationship repair

---

## Related Environments

- [TrustDilemma-v0](trust_dilemma.md): Normal trust dynamics
- [SynergySearch-v0](synergy_search.md): Another benchmark challenge
- [RenaultNissan-v0](renault_nissan.md): Real-world crisis recovery (crisis phase)

---

## References

1. Kim, P.H., Ferrin, D.L., Cooper, C.D., & Dirks, K.T. (2004). Removing the Shadow of Suspicion. Journal of Applied Psychology.
2. Lewicki, R.J. & Bunker, B.B. (1996). Developing and Maintaining Trust in Work Relationships. Trust in Organizations.
3. Pant, V. & Yu, E. (2025). Trust Dynamics in Strategic Alliances. arXiv:2510.24909
