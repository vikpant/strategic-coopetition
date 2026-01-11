# Parameter Reference

**Validated Parameters for Coopetition-Gym Environments**

This document provides a comprehensive reference for all parameters used in Coopetition-Gym, including validated values, acceptable ranges, and calibration guidance.

---

## Quick Reference Card

For practitioners who need parameter values quickly:

### Recommended Default Values

```python
# Value Function Parameters (TR-1)
THETA = 20.0          # Logarithmic scale
BETA = 0.75           # Power exponent
GAMMA = 0.65          # Complementarity

# Trust Dynamics Parameters (TR-2)
LAMBDA_PLUS = 0.10    # Trust building rate
LAMBDA_MINUS = 0.30   # Trust erosion rate
MU_R = 0.60           # Reputation damage severity
DELTA_R = 0.03        # Reputation decay rate
XI = 0.50             # Interdependence amplification
KAPPA = 1.0           # Signal sensitivity

# Initial Conditions
T_INIT = 0.50         # Initial trust level
R_INIT = 0.00         # Initial reputation damage
```

---

## Pillar 1: Interdependence & Complementarity Parameters

### Value Function Parameters

#### Logarithmic Specification (Recommended)

| Parameter | Symbol | Validated Value | Range | Source |
|-----------|--------|-----------------|-------|--------|
| Scale Factor | θ | **20.0** | [15, 30] | TR-1 §7.3, S-LCD validation |
| Complementarity | γ | **0.65** | [0.50, 0.80] | TR-1 §7.2, multi-criteria |

**Usage**:
```python
f_i(a_i) = θ × ln(1 + a_i)
V(a|γ) = Σ f_i(a_i) + γ × (∏a_i)^(1/N)
```

**Validation Performance**: 58/60 (96.7%) on Samsung-Sony S-LCD case study

**When to Use**: Manufacturing joint ventures, technology partnerships, scenarios where initial capabilities are highly valuable but incremental improvements have declining impact.

#### Power Specification (Alternative)

| Parameter | Symbol | Validated Value | Range | Source |
|-----------|--------|-----------------|-------|--------|
| Exponent | β | **0.75** | [0.65, 0.85] | TR-1 §7.1, 22,000+ trials |
| Complementarity | γ | **0.65** | [0.50, 0.80] | TR-1 §7.2 |

**Usage**:
```python
f_i(a_i) = a_i^β
V(a|γ) = Σ f_i(a_i) + γ × (∏a_i)^(1/N)
```

**Validation Performance**: 46/60 (76.7%) on Samsung-Sony S-LCD case study

**When to Use**: General scenarios, platform ecosystems, when cooperation magnitudes may be larger.

#### Comparison

| Criterion | Logarithmic (θ=20) | Power (β=0.75) | Winner |
|-----------|-------------------|----------------|--------|
| S-LCD Validation | 58/60 | 46/60 | Logarithmic |
| Cooperation Prediction | 41% increase | 166% increase | Logarithmic (realistic) |
| Bounded Returns | Yes | No | Logarithmic |
| Mathematical Tractability | Moderate | High | Power |

### Interdependence Parameters

| Parameter | Symbol | Typical Values | Range | Description |
|-----------|--------|----------------|-------|-------------|
| Dependency Weight | w_d | Context-specific | [0, 1] | Goal importance (normalized) |
| Criticality Factor | crit | Calculated | [0, 1] | 1/n for n alternatives |
| Bargaining Share | α_i | 0.50 (symmetric) | [0, 1] | Must sum to 1.0 |

**Interdependence Matrix Guidance**:

| Relationship Type | D_ij Range | Example |
|-------------------|------------|---------|
| No dependency | 0.00 | Competitors in separate markets |
| Weak dependency | 0.10 - 0.30 | Multiple alternative suppliers |
| Moderate dependency | 0.30 - 0.60 | Preferred but substitutable partner |
| Strong dependency | 0.60 - 0.85 | Critical supplier, few alternatives |
| Complete dependency | 0.85 - 1.00 | Sole provider of essential resource |

---

## Pillar 2: Trust Dynamics Parameters

### Trust Evolution Parameters

| Parameter | Symbol | Validated Value | Range | Source |
|-----------|--------|-----------------|-------|--------|
| Trust Building Rate | λ⁺ | **0.10** | [0.05, 0.15] | TR-2 §7.2 |
| Trust Erosion Rate | λ⁻ | **0.30** | [0.20, 0.45] | TR-2 §7.2 |
| Negativity Ratio | λ⁻/λ⁺ | **3.0** | [2.5, 4.0] | Behavioral economics |

**The 3:1 Ratio**: Empirically grounded in behavioral economics research showing trust erodes approximately 3× faster than it builds. This captures:
- Negativity bias in human judgment
- Asymmetric impact of violations vs. cooperation
- Evolutionary caution toward potential threats

**Calibration Guidance**:

| Context | λ⁺ | λ⁻ | Ratio | Rationale |
|---------|-----|-----|-------|-----------|
| High-trust culture | 0.12 | 0.30 | 2.5 | Faster trust building |
| Standard business | 0.10 | 0.30 | 3.0 | Default validated values |
| Low-trust environment | 0.08 | 0.35 | 4.4 | Slower building, faster erosion |
| Post-crisis recovery | 0.06 | 0.25 | 4.2 | Difficult trust rebuilding |

### Reputation Parameters

| Parameter | Symbol | Validated Value | Range | Source |
|-----------|--------|-----------------|-------|--------|
| Damage Severity | μ_R | **0.60** | [0.45, 0.75] | TR-2 §7.3 |
| Decay Rate | δ_R | **0.03** | [0.01, 0.05] | TR-2 §7.3 |

**Interpretation**:
- **μ_R = 0.60**: A full violation (s = -1) causes 60% of available reputation space to be damaged
- **δ_R = 0.03**: Approximately 33 periods of no violations to decay reputation by 63%

**Hysteresis Effect**:
```
Trust Ceiling: Θ = 1 - R

Example trajectory:
- Initial: T=0.50, R=0.00, Θ=1.00 (full recovery possible)
- After violation: T=0.35, R=0.40, Θ=0.60 (ceiling at 60%)
- After recovery: T→0.58 max (cannot exceed ceiling)
```

### Amplification Parameters

| Parameter | Symbol | Validated Value | Range | Source |
|-----------|--------|-----------------|-------|--------|
| Interdep. Amplification | ξ | **0.50** | [0.30, 0.70] | TR-2 §7.4 |
| Signal Sensitivity | κ | **1.0** | [0.5, 2.0] | TR-2 §6.1 |

**Interdependence Amplification Effect**:
```
Erosion = λ⁻ × |signal| × T × (1 + ξ × D_ij)

Example:
- Low dependency (D=0.2):  Erosion factor = 1.10
- High dependency (D=0.8): Erosion factor = 1.40

Result: 27% faster trust erosion in high-dependency relationships
```

### Initial Conditions

| Parameter | Symbol | Recommended | Range | Description |
|-----------|--------|-------------|-------|-------------|
| Initial Trust | T⁰_ij | **0.50** | [0.0, 1.0] | Starting trust level |
| Initial Reputation | R⁰_ij | **0.00** | [0.0, 1.0] | Starting reputation damage |
| Baseline Action | a^baseline | Context | [0, e] | Expected cooperation level |

**Initial Trust Guidance**:

| Relationship History | T⁰ | R⁰ | Scenario |
|----------------------|-----|-----|----------|
| First interaction | 0.50 | 0.00 | Neutral starting point |
| Positive reputation | 0.70 | 0.00 | Known reliable partner |
| Prior relationship | 0.60-0.80 | 0.00 | Successful past collaboration |
| Recovery scenario | 0.30 | 0.40 | Post-violation situation |
| Hostile history | 0.20 | 0.60 | Prior conflicts |

---

## Environment-Specific Parameters

### TrustDilemma-v0

```python
default_params = {
    'n_agents': 2,
    'max_steps': 100,
    'endowment': 100.0,
    'theta': 20.0,
    'gamma': 0.65,
    'lambda_plus': 0.10,
    'lambda_minus': 0.30,
    'mu_R': 0.60,
    'delta_R': 0.03,
    'xi': 0.50,
    'kappa': 1.0,
    'T_init': 0.50,
    'R_init': 0.00,
}
```

### PlatformEcosystem-v0

```python
default_params = {
    'n_agents': 5,  # 1 platform + 4 developers
    'max_steps': 100,
    'endowment': [200.0, 100.0, 100.0, 100.0, 100.0],  # Platform has more
    'theta': 20.0,
    'gamma': 0.70,  # Higher complementarity in ecosystems
    'lambda_plus': 0.10,
    'lambda_minus': 0.30,
    'mu_R': 0.55,   # Slightly lower damage (more forgiving)
    'delta_R': 0.04, # Slightly faster decay
    'xi': 0.40,     # Lower amplification
    'kappa': 1.0,
}
```

### SLCD-v0 (Validated Case Study)

```python
# Parameters calibrated to Samsung-Sony S-LCD (2004-2011)
validated_params = {
    'n_agents': 2,
    'max_steps': 8,  # 8 years
    'endowment': [100.0, 100.0],
    'theta': 20.0,           # Validated
    'gamma': 0.65,           # Validated
    'alpha': [0.50, 0.50],   # Equal bargaining power
    'D_matrix': [[0.0, 0.45],
                 [0.40, 0.0]],  # Moderate mutual dependency
    'lambda_plus': 0.10,
    'lambda_minus': 0.30,
    'T_init': 0.50,
    'R_init': 0.00,
}
```

### RenaultNissan-v0 (Validated Case Study)

```python
# Parameters calibrated to Renault-Nissan Alliance (1999-2025)
validated_params = {
    'n_agents': 2,
    'max_steps': 26,  # 26 years
    'phases': 5,      # Crisis, Recovery, Growth, Ghosn, Post-Ghosn
    'theta': 20.0,
    'gamma': 0.60,
    'lambda_plus': 0.08,   # Slower trust building (cross-cultural)
    'lambda_minus': 0.32,  # Standard erosion
    'mu_R': 0.65,          # Higher damage (visible scandals)
    'delta_R': 0.02,       # Slower forgetting
    'T_init': 0.30,        # Started in crisis
    'R_init': 0.20,        # Some initial reputation damage
}
```

---

## Parameter Sensitivity Analysis

### High-Impact Parameters

Parameters with largest effect on environment dynamics:

| Parameter | Sensitivity | Impact |
|-----------|-------------|--------|
| λ⁻/λ⁺ ratio | **Very High** | Determines trust recovery feasibility |
| γ (complementarity) | **High** | Controls cooperative incentive strength |
| D_ij (interdependence) | **High** | Shapes utility landscape |
| μ_R (damage severity) | **Medium-High** | Determines hysteresis strength |

### Low-Impact Parameters

Parameters that can be approximated without significant effect:

| Parameter | Sensitivity | Notes |
|-----------|-------------|-------|
| κ (signal sensitivity) | Low | 1.0 works for most scenarios |
| δ_R (reputation decay) | Low | Affects long-horizon only |
| θ vs β (specification) | Low within spec | Both work, θ slightly better validated |

### Sensitivity Recommendations

```python
# For robustness testing, vary these parameters:
sensitivity_ranges = {
    'lambda_ratio': [2.5, 3.0, 3.5, 4.0],  # Primary sensitivity
    'gamma': [0.55, 0.65, 0.75],            # Secondary sensitivity
    'mu_R': [0.50, 0.60, 0.70],             # Tertiary sensitivity
}

# Keep these fixed at validated values:
fixed_params = {
    'theta': 20.0,
    'kappa': 1.0,
    'delta_R': 0.03,
}
```

---

## Calibration Workflow

### For New Scenarios

1. **Start with defaults**: Use recommended values from this document
2. **Identify analogous case**: Find most similar validated environment
3. **Adjust key parameters**: Focus on high-impact parameters
4. **Validate qualitatively**: Check behavior matches domain expectations
5. **Run sensitivity analysis**: Test robustness across parameter ranges

### For Research Extensions

1. **Define hypothesis**: What parameter relationship are you testing?
2. **Design sweep**: Systematic variation of parameters
3. **Measure outcomes**: Return, trust, cooperation patterns
4. **Report ranges**: Document which parameter ranges support findings

### Example: Custom Manufacturing Partnership

```python
# Step 1: Start with SLCD as base (validated manufacturing JV)
params = slcd_params.copy()

# Step 2: Adjust for specific context
params['D_matrix'] = [[0.0, 0.60],   # Higher mutual dependency
                      [0.55, 0.0]]
params['gamma'] = 0.70               # Higher complementarity (specialized assets)
params['T_init'] = 0.65              # Prior positive relationship

# Step 3: Run and validate
env = coopetition_gym.make("TrustDilemma-v0", **params)
# ... run episodes and verify reasonable dynamics
```

---

## Benchmark-Derived Insights

From 760 experiments across 20 algorithms:

### Parameter Combinations That Work Well

| Configuration | Result | Insight |
|---------------|--------|---------|
| λ⁻/λ⁺ = 3.0, T_init = 0.50 | Trust-Return r=0.552 | Standard validated setup |
| High γ + High D | Cooperation emergence | Strong incentives align behavior |
| Moderate all parameters | Robust performance | Avoids edge case instabilities |

### Parameter Combinations That Fail

| Configuration | Result | Insight |
|---------------|--------|---------|
| λ⁻/λ⁺ > 5.0 | Trust collapse | Recovery becomes impossible |
| γ < 0.3 | Defection dominance | Insufficient cooperative incentive |
| D_ij → 0 for all pairs | Pure competition | No structural incentive for cooperation |

---

## Citation

For parameter validation methodology:

```bibtex
@article{pant2025tr1,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Interdependence and Complementarity},
  author={Pant, Vik and Yu, Eric},
  journal={arXiv preprint arXiv:2510.18802},
  year={2025},
  note={Section 7: Validation methodology; 22,000+ trials}
}

@article{pant2025tr2,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Trust and Reputation Dynamics},
  author={Pant, Vik and Yu, Eric},
  journal={arXiv preprint arXiv:2510.24909},
  year={2025},
  note={Section 7: Parameter validation; 78,125 configurations}
}
```

---

## Navigation

- [Theoretical Foundations](index.md)
- [Interdependence Framework](interdependence.md)
- [Value Creation & Complementarity](value_creation.md)
- [Trust Dynamics](trust_dynamics.md)
- [Environment Reference](../environments/index.md)
