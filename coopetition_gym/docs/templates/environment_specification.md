# Environment Documentation Templates

This document provides standardized templates for documenting Coopetition-Gym environments according to modern MARL research conventions.

---

## MARL Classification Block

Insert this block after the Overview section in each environment document.

```markdown
## MARL Classification

| Property | Value |
|----------|-------|
| **Game Type** | Markov Game / Dec-POMDP / Mean-Field Game |
| **Cooperation Structure** | Competitive / Cooperative / Mixed-Motive |
| **Observability** | Full / Partial (specify hidden components) |
| **Communication** | None / Implicit (actions) / Explicit (messages) |
| **Agent Symmetry** | Symmetric / Asymmetric (specify differences) |
| **Reward Structure** | Individual / Team / Mixed (interdependence) |
| **Action Space** | Continuous / Discrete (specify bounds) |
| **State Dynamics** | Deterministic / Stochastic |
| **Horizon** | Finite (T=N) / Infinite (discounted) |
| **Canonical Comparison** | Reference to similar benchmarks |
```

---

## Formal Specification Block

Insert this block after the MARL Classification section.

```markdown
## Formal Specification

This environment is formalized as an N-player Markov Game M = (N, S, {A_i}, P, {R_i}, γ, T).

### Agents
N = {1, ..., n} with n = [number of agents]

### State Space
S ⊆ ℝ^d where d = [dimension formula]

Components:
- **Actions**: a ∈ ℝ^N (previous cooperation levels)
- **Trust Matrix**: τ ∈ [0,1]^(N×N) (pairwise trust)
- **Reputation Matrix**: R ∈ [0,1]^(N×N) (reputation damage)
- **Interdependence**: D ∈ [0,1]^(N×N) (structural dependencies)
- **Time**: t ∈ [0,1] (normalized timestep)

### Action Space
A_i = [0, e_i] ⊂ ℝ for agent i with endowment e_i

### Transition Dynamics
**Trust Update**:
τ_ij(t+1) = clip(τ_ij(t) + Δτ_ij, 0, Θ_ij)

where:
- Δτ_ij = λ⁺ · max(0, σ_ij) · (1 - τ_ij) - λ⁻ · max(0, -σ_ij) · τ_ij
- σ_ij = κ · (a_j - b_j) / b_j (cooperation signal)
- Θ_ij = 1 - R_ij (trust ceiling from reputation)

**Reputation Update**:
R_ij(t+1) = clip(R_ij(t) · (1 - $\delta_R$) + $\mu_R$ · 𝟙[σ_ij < 0], 0, 1)

### Reward Function
r_i(s, a) = U_i(a) where integrated utility is: U_i = (e_i - a_i) + f(a_i) + α_i · G(a) + Σ_j D_ij · π_j

with:
- f(a_i) = θ · ln(1 + a_i) (individual value)
- G(a) = (∏_i a_i)^(1/N) · (1 + γ · C(a)) (synergy)
- C(a) = min_i(a_i / e_i) (complementarity)

### Episode Termination
- **Truncation**: t ≥ T (max_steps reached)
- **Termination**: [environment-specific conditions]

### Discount Factor
γ = 1.0 (finite horizon, undiscounted episodic)
```

---

## Parameter Table Template

```markdown
### Environment Parameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Max Steps | T | 100 | [50, 500] | Episode horizon |
| Trust Building Rate | λ⁺ | 0.10 | [0.05, 0.20] | Cooperation → trust |
| Trust Erosion Rate | λ⁻ | 0.30 | [0.20, 0.50] | Defection → trust loss |
| ... | ... | ... | ... | ... |
```

---

## Observation Space Table Template

```markdown
### Observation Space Details

| Component | Indices | Shape | Range | Description |
|-----------|---------|-------|-------|-------------|
| Actions | [0:N] | (N,) | [0, e_max] | Previous cooperation levels |
| Trust Matrix | [N:N+N²] | (N,N) | [0, 1] | Pairwise trust τ_ij |
| Reputation Matrix | [N+N²:N+2N²] | (N,N) | [0, 1] | Reputation damage R_ij |
| Interdependence | [N+2N²:N+3N²] | (N,N) | [0, 1] | Structural dependencies D_ij |
| Timestep | [N+3N²] | (1,) | [0, 1] | Normalized t/T |

**Total Dimension**: d = N + 3N² + 1
```
