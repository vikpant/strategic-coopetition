# Environment Classes

API documentation for all coopetition environment classes.

*Module version: 0.2.0*

---

## Base Classes

### AbstractCoopetitionEnv

```python
class AbstractCoopetitionEnv(ABC):
    """
    API-agnostic base class containing core game logic.

    This class implements all coopetition mechanics independent of
    the external API (Gymnasium vs PettingZoo).
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        obs_config: Optional[ObservationConfig] = None
    ):
        """
        Initialize environment with configuration.

        Args:
            config: Environment configuration dataclass
            obs_config: Optional observation configuration
        """
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `process_actions(actions)` | Validate and process agent actions |
| `update_trust()` | Perform trust dynamics update |
| `compute_rewards()` | Calculate rewards based on current state |
| `get_observation()` | Construct observation array/dict |
| `get_info()` | Build info dictionary |

---

### CoopetitionEnv

```python
class CoopetitionEnv(AbstractCoopetitionEnv, gymnasium.Env):
    """
    Gymnasium-compatible wrapper for coopetition environments.

    Provides standard Gymnasium interface:
    - reset(seed, options) -> (obs, info)
    - step(action) -> (obs, reward, terminated, truncated, info)
    - render() -> Optional[str]
    - close()
    """
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `observation_space` | `gym.Space` | Observation space specification |
| `action_space` | `gym.Space` | Action space specification |
| `n_agents` | `int` | Number of agents |
| `endowments` | `NDArray` | Agent endowments |
| `baselines` | `NDArray` | Cooperation baselines |

**Example:**

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("TrustDilemma-v0")

# Standard Gymnasium loop
obs, info = env.reset(seed=42)
for _ in range(100):
    actions = np.array([50.0, 50.0])
    obs, rewards, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break
env.close()
```

---

## Dyadic Environments

### TrustDilemmaEnv

```python
class TrustDilemmaEnv(CoopetitionEnv):
    """
    TrustDilemma-v0: Continuous iterated Prisoner's Dilemma with trust dynamics.

    Two symmetric agents choose cooperation levels. Trust evolves based
    on observed behavior with 3:1 negativity bias.
    """
```

**Environment ID:** `TrustDilemma-v0`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 100 | Episode length |
| `initial_trust` | 0.50 | Starting trust level |

**Observation Shape:** (17,)

**Action Space:** Box(0, 100, shape=(2,))

---

### PartnerHoldUpEnv

```python
class PartnerHoldUpEnv(CoopetitionEnv):
    """
    PartnerHoldUp-v0: Asymmetric power relationship.

    Strong partner (larger endowment, lower dependency) can exploit
    weak partner. Exit threshold creates credible commitment.
    """
```

**Environment ID:** `PartnerHoldUp-v0`

**Agent Configuration:**

| Agent | Endowment | Dependency | Bargaining |
|-------|-----------|------------|------------|
| Strong | 120 | 0.35 | 0.60 |
| Weak | 80 | 0.85 | 0.40 |

**Termination:** Episode ends if weak partner's trust < 0.10

---

## Ecosystem Environments

### PlatformEcosystemEnv

```python
class PlatformEcosystemEnv(CoopetitionEnv):
    """
    PlatformEcosystem-v0: Platform with N developers.

    Hub-and-spoke interdependence structure. Platform success
    depends on developer contributions; developers depend on platform.
    """
```

**Environment ID:** `PlatformEcosystem-v0`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_developers` | 4 | Number of developer agents |
| `platform_endowment` | 150 | Platform's resource pool |
| `developer_endowment` | 80 | Each developer's resources |

---

### DynamicPartnerSelectionEnv

```python
class DynamicPartnerSelectionEnv(CoopetitionEnv):
    """
    DynamicPartnerSelection-v0: Reputation-based marketplace.

    N agents with public reputation scores. Cooperation builds reputation;
    reputation affects partner quality.
    """
```

**Environment ID:** `DynamicPartnerSelection-v0`

---

## Benchmark Environments

### RecoveryRaceEnv

```python
class RecoveryRaceEnv(CoopetitionEnv):
    """
    RecoveryRace-v0: Post-crisis trust recovery.

    Agents start with low trust (0.25) and high reputation damage (0.50).
    Goal: reach trust ≥ 0.90 before time limit.
    """
```

**Environment ID:** `RecoveryRace-v0`

**Initial State:**

| Variable | Value | Interpretation |
|----------|-------|----------------|
| Trust | 0.25 | Very low |
| Rep. Damage | 0.50 | High (ceiling = 0.50) |
| Target | 0.90 | Success threshold |
| Horizon | 150 | Extended for recovery |

---

### SynergySearchEnv

```python
class SynergySearchEnv(CoopetitionEnv):
    """
    SynergySearch-v0: Hidden complementarity discovery.

    Complementarity γ is sampled from [0.20, 0.90] at episode start
    and hidden from agents. Must infer γ from rewards.
    """
```

**Environment ID:** `SynergySearch-v0`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma_range` | (0.20, 0.90) | Range for γ sampling |
| `reveal_gamma_in_obs` | False | Include γ in observation |

---

## Case Study Environments

### SLCDEnv

```python
class SLCDEnv(CoopetitionEnv):
    """
    SLCD-v0: Samsung-Sony LCD Joint Venture (2004-2011).

    Validated parameters achieving 58/60 accuracy against historical data.
    """
```

**Environment ID:** `SLCD-v0`

**Agent Configuration:**

| Agent | Endowment | Dependency | Bargaining |
|-------|-----------|------------|------------|
| Samsung | 100 | 0.64 | 0.55 |
| Sony | 100 | 0.86 | 0.45 |

---

### RenaultNissanEnv

```python
class RenaultNissanEnv(CoopetitionEnv):
    """
    RenaultNissan-v0: Renault-Nissan Alliance (1999-2025).

    Multi-phase simulation with configurable initial conditions.
    """
```

**Environment ID:** `RenaultNissan-v0`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phase` | "formation" | Alliance phase |

**Phases:**

| Phase | Period | Initial Trust | Initial Damage |
|-------|--------|---------------|----------------|
| formation | 1999-2002 | 0.45 | 0.05 |
| mature | 2002-2018 | 0.70 | 0.02 |
| crisis | 2018-2020 | 0.30 | 0.45 |
| strained | 2020-2025 | 0.40 | 0.35 |

---

## Extended Environments

### CooperativeNegotiationEnv

```python
class CooperativeNegotiationEnv(CoopetitionEnv):
    """
    CooperativeNegotiation-v0: Multi-round negotiation with commitments.

    Agents submit proposals; aligned proposals form binding agreements.
    Breach penalties apply for violations.
    """
```

**Environment ID:** `CooperativeNegotiation-v0`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agreement_threshold` | 10.0 | Max proposal difference for agreement |
| `breach_penalty_multiplier` | 3.0 | Penalty for violating agreement |

---

### ReputationMarketEnv

```python
class ReputationMarketEnv(CoopetitionEnv):
    """
    ReputationMarket-v0: N-agent market with tiered reputation bonuses.

    Reputation determines reward multiplier:
    - Premium (≥0.80): 1.30×
    - Standard (≥0.50): 1.00×
    - Probation (≥0.25): 0.70×
    - Excluded (<0.25): 0.40×
    """
```

**Environment ID:** `ReputationMarket-v0`

---

## See Also

- [Factory Functions](index.md#factory-functions) - Creating environments
- [Wrappers](wrappers.md) - PettingZoo adapters
- [Environment Reference](../environments/index.md) - Detailed documentation
