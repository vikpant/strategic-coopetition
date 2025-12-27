# RenaultNissan-v0

**Category:** Validated Case Study
**Agents:** 2 (Nissan, Renault)
**Difficulty:** Advanced
**Source:** `coopetition_gym/envs/case_study_envs.py`

---

## Overview

RenaultNissan-v0 models the **Renault-Nissan Alliance** across four distinct phases, as described in TR-2 (arXiv:2510.24909). This environment captures the 25+ year evolution of one of the most complex and enduring automotive alliances.

Unlike SLCD-v0 which models a single joint venture, RenaultNissan-v0 supports **multi-phase simulation** with different initial conditions for each era of the alliance.

---

## Historical Background

### The Renault-Nissan Alliance (1999-Present)

**Partners:**
- **Nissan Motor Corporation**: Japanese automaker, near-bankruptcy in 1999
- **Renault SA**: French automaker, seeking global expansion

**Key Events:**
- **1999**: Renault rescues Nissan with $5.4B investment
- **2002-2018**: Successful collaboration, shared platforms
- **2018**: Carlos Ghosn arrest, governance crisis
- **2020+**: Alliance restructuring, strained relations

### Why Multi-Phase?

The alliance demonstrates:
1. **Trust Recovery**: Nissan's trust in Renault rebuilt after rescue
2. **Trust Erosion**: Growing tensions over governance
3. **Crisis and Recovery**: Post-Ghosn restructuring
4. **Ongoing Dynamics**: Current strained but continuing relationship

---

## Environment Specification

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment with specific phase
env = coopetition_gym.make("RenaultNissan-v0", phase="formation")

obs, info = env.reset(seed=42)

print(f"Phase: {info['phase']}")
print(f"Period: {info['period']}")
print(f"Initial trust: {info['mean_trust']:.2f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phase` | "formation" | Alliance phase to simulate |
| `max_steps` | 100 | Maximum timesteps per episode |
| `render_mode` | None | Rendering mode |

---

## Alliance Phases

### Phase Configuration

| Phase | Period | Initial Trust | Initial Rep. Damage | Interpretation |
|-------|--------|---------------|---------------------|----------------|
| `"formation"` | 1999-2002 | 0.45 | 0.05 | Early uncertain partnership |
| `"mature"` | 2002-2018 | 0.70 | 0.02 | Successful collaboration |
| `"crisis"` | 2018-2020 | 0.30 | 0.45 | Post-Ghosn governance crisis |
| `"strained"` | 2020-2025 | 0.40 | 0.35 | Restructured but tense |

### Selecting a Phase

```python
# Formation phase (early alliance)
env_formation = coopetition_gym.make("RenaultNissan-v0", phase="formation")

# Mature phase (successful collaboration)
env_mature = coopetition_gym.make("RenaultNissan-v0", phase="mature")

# Crisis phase (governance crisis)
env_crisis = coopetition_gym.make("RenaultNissan-v0", phase="crisis")

# Strained phase (current era)
env_strained = coopetition_gym.make("RenaultNissan-v0", phase="strained")
```

---

## Agent Configuration

### Endowments

| Agent | Role | Endowment | Interpretation |
|-------|------|-----------|----------------|
| Nissan (0) | Japanese Partner | 90.0 | Slightly smaller (historical) |
| Renault (1) | French Partner | 100.0 | Larger (controlling stake) |

### Bargaining Shares

| Agent | Alpha | Interpretation |
|-------|-------|----------------|
| Nissan | 0.48 | Slight disadvantage (rescued party) |
| Renault | 0.52 | Slight advantage (rescuer) |

### Baselines

| Agent | Baseline | As % of Endowment |
|-------|----------|-------------------|
| Nissan | 27.0 | 30% |
| Renault | 30.0 | 30% |

---

## Interdependence Structure

### Phase-Dependent Dependencies

Created using `create_renault_nissan_interdependence(phase)`:

The interdependence matrix varies by phase:
- **Formation**: Nissan highly dependent on Renault (rescue)
- **Mature**: More balanced mutual dependency
- **Crisis**: Reduced dependency as alliance frays
- **Strained**: Attempted rebalancing

---

## Trust Dynamics

### Parameters (All Phases)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Trust Building Rate | λ⁺ | 0.08 | TR-2 validated |
| Trust Erosion Rate | λ⁻ | 0.25 | TR-2 validated |
| Reputation Damage | μ_R | 0.55 | TR-2 validated |
| Reputation Decay | δ_R | 0.02 | TR-2 validated |
| Interdependence Amp. | ξ | 0.50 | TR-2 validated |
| Signal Sensitivity | κ | 1.0 | Standard |

### Phase-Specific Initial Conditions

**Formation (1999-2002):**
- Trust = 0.45: Uncertain but hopeful
- Rep. Damage = 0.05: Clean slate

**Mature (2002-2018):**
- Trust = 0.70: Strong collaborative relationship
- Rep. Damage = 0.02: Minor accumulated issues

**Crisis (2018-2020):**
- Trust = 0.30: Severely damaged by governance scandal
- Rep. Damage = 0.45: Significant institutional damage

**Strained (2020-2025):**
- Trust = 0.40: Partial recovery
- Rep. Damage = 0.35: Lingering damage

---

## Value Function

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| θ | 22.0 | Slightly higher scale (larger industry) |
| γ | 0.58 | Moderate complementarity |

---

## Metrics and Info

The `info` dictionary includes:

| Key | Type | Description |
|-----|------|-------------|
| `step` | int | Current timestep |
| `phase` | str | Current alliance phase |
| `period` | str | Historical period string |
| `mean_trust` | float | Average trust level |
| `nissan_investment` | float | Nissan's action |
| `renault_investment` | float | Renault's action |
| `trust_matrix` | ndarray | Full trust matrix |

---

## Phase-Specific Analysis

### Formation Phase Strategy

**Historical context:** Renault rescues near-bankrupt Nissan. Trust must be built.

```python
env = coopetition_gym.make("RenaultNissan-v0", phase="formation")
obs, info = env.reset()

# Renault should invest heavily (demonstrate commitment)
# Nissan should respond cautiously (rebuilding)
renault_action = 70.0  # High investment
nissan_action = 50.0   # Moderate response
```

**Expected dynamics:**
- Trust rises from 0.45 toward 0.60-0.70
- Nissan gains confidence in Renault's commitment
- Value creation increases over time

### Mature Phase Strategy

**Historical context:** Successful partnership with shared platforms.

```python
env = coopetition_gym.make("RenaultNissan-v0", phase="mature")
obs, info = env.reset()

# Both partners maintain high cooperation
renault_action = 65.0
nissan_action = 60.0
```

**Expected dynamics:**
- Trust stable around 0.70-0.75
- Consistent high value creation
- Minor fluctuations from market conditions

### Crisis Phase Strategy

**Historical context:** Post-Ghosn arrest, governance scandal.

```python
env = coopetition_gym.make("RenaultNissan-v0", phase="crisis")
obs, info = env.reset()

# Both partners are cautious
# Must avoid further trust erosion
renault_action = 45.0  # Defensive
nissan_action = 40.0   # Defensive
```

**Expected dynamics:**
- Trust at risk of further decline
- High reputation damage limits recovery
- Careful cooperation needed to stabilize

### Strained Phase Strategy

**Historical context:** Restructured alliance, ongoing tensions.

```python
env = coopetition_gym.make("RenaultNissan-v0", phase="strained")
obs, info = env.reset()

# Gradual trust rebuilding
renault_action = 55.0
nissan_action = 50.0
```

**Expected dynamics:**
- Trust can slowly recover
- Reputation damage gradually heals
- Long-term view required

---

## Example: Multi-Phase Simulation

```python
import coopetition_gym
import numpy as np

phases = ["formation", "mature", "crisis", "strained"]

for phase in phases:
    env = coopetition_gym.make("RenaultNissan-v0", phase=phase)
    obs, info = env.reset(seed=42)

    print(f"\n{phase.upper()} Phase ({info['period']})")
    print(f"  Initial trust: {info['mean_trust']:.2f}")

    # Run 50 steps with moderate cooperation
    for step in range(50):
        actions = np.array([50.0, 55.0])  # Nissan, Renault
        obs, rewards, terminated, truncated, info = env.step(actions)

    print(f"  Final trust: {info['mean_trust']:.2f}")
    print(f"  Trust change: {info['mean_trust'] - 0.5:.+.2f}")
```

---

## Research Applications

RenaultNissan-v0 is suitable for studying:

- **Alliance Evolution**: How partnerships change over time
- **Crisis Management**: Recovery from governance failures
- **Cross-Cultural Dynamics**: Japanese-French partnership
- **Long-Term Coopetition**: Decades-long relationships
- **Phase Transitions**: How alliances move between states

---

## Comparison with SLCD-v0

| Feature | RenaultNissan-v0 | SLCD-v0 |
|---------|------------------|---------|
| Phases | 4 configurable | 1 fixed |
| Duration | 25+ years modeled | 8 years modeled |
| Complexity | Multi-phase dynamics | Single trajectory |
| Best For | Phase analysis | Benchmark testing |

---

## Related Environments

- [SLCD-v0](slcd.md): Simpler validated case study
- [RecoveryRace-v0](recovery_race.md): Crisis recovery focus
- [PartnerHoldUp-v0](partner_holdup.md): Power asymmetry dynamics

---

## References

1. Pant, V. & Yu, E. (2025). Trust Dynamics in Strategic Alliances. arXiv:2510.24909
2. Segrestin, B. (2005). Partnering to Explore: The Renault–Nissan Alliance. Research Policy.
3. Freyssenet, M. (2009). The Second Automobile Revolution. Palgrave Macmillan.
