# Environment Classes

API documentation for all coopetition environment classes.

*Module version: 0.3.0*

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

        Args: config: Environment configuration dataclass
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
for _ in range(100): actions = np.array([50.0, 50.0])
    obs, rewards, terminated, truncated, info = env.step(actions)
    if terminated or truncated: break
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

## Reciprocity Environments (TR-4)

### ReciprocalDilemmaEnv

```python
class ReciprocalDilemmaEnv(BaseTR4Env):
    """
    ReciprocalDilemma-v0: Continuous iterated PD with TR-4 reciprocity.

    Two symmetric agents with direct reciprocity via bounded memory window.
    Enables tit-for-tat-like conditional cooperation strategies.
    """
```

**Environment ID:** `ReciprocalDilemma-v0`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 100 | Episode length |

**Agent Configuration:** 2 symmetric agents, endowment 100 each, D=0.5

**TR-4 Parameters:** ρ₀=1.0, η=1.0, κ=1.0, k=5, λ_R=1.0, ω=0.6

---

### GiftExchangeEnv

```python
class GiftExchangeEnv(BaseTR4Env):
    """
    GiftExchange-v0: Asymmetric employer-worker gift exchange.

    Employer sets wage-cooperation, worker responds with effort-cooperation.
    Asymmetric dependency amplifies worker's reciprocity (ρ_21 ≈ 0.70 vs ρ_12 ≈ 0.30).
    """
```

**Environment ID:** `GiftExchange-v0`

**Agent Configuration:**

| Agent | Role | Endowment | Dependency |
|-------|------|-----------|------------|
| 0 | Employer | 100 | D₀₁=0.4 |
| 1 | Worker | 80 | D₁₀=0.7 |

**TR-4 Parameters:** ρ₀=1.2, η=1.5, κ=1.0, k=3, λ_R=1.2, ω=0.8

---

### IndirectReciprocityEnv

```python
class IndirectReciprocityEnv(BaseTR4Env):
    """
    IndirectReciprocity-v0: 4-agent population with reputation-mediated cooperation.

    Cooperation with any partner is observed by all, enabling indirect reciprocity.
    Longer memory (k=7) and higher reciprocity weight (λ_R=1.5) amplify reputation effects.
    """
```

**Environment ID:** `IndirectReciprocity-v0`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 150 | Extended for reputation dynamics |

**Agent Configuration:** 4 symmetric agents, endowment 100 each, D=0.4

**TR-4 Parameters:** ρ₀=0.8, η=1.0, κ=1.0, k=7, λ_R=1.5, ω=0.5

---

### GraduatedSanctionEnv

```python
class GraduatedSanctionEnv(BaseTR4Env):
    """
    GraduatedSanction-v0: 6-agent commons with graduated reciprocity sanctions.

    Lower κ=0.8 creates proportional (graduated) response. Long memory k=10
    enables escalation for repeated violations. Captures Ostrom's design principles.
    """
```

**Environment ID:** `GraduatedSanction-v0`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 200 | Extended for graduated dynamics |

**Agent Configuration:** 6 symmetric agents, endowment 100 each, D=0.35

**TR-4 Parameters:** ρ₀=0.6, η=1.5, κ=0.8, k=10, λ_R=1.8, ω=1.0

---

### AppleAppStoreEnv

```python
class AppleAppStoreEnv(BaseTR4Env):
    """
    AppleAppStore-v0: Validated Apple iOS App Store case study (2008-2024).

    3 agents (Apple, Major Developers, Small Developers) with asymmetric
    dependencies. 66-step episodes map to 66 historical quarters.
    Validation: 48/55 (87.3%).
    """
```

**Environment ID:** `AppleAppStore-v0`

**Agent Configuration:**

| Agent | Role | Endowment | Key Dependencies |
|-------|------|-----------|-----------------|
| 0 | Apple | 100 | D₀₁=0.3, D₀₂=0.2 |
| 1 | Major Devs | 80 | D₁₀=0.8 |
| 2 | Small Devs | 60 | D₂₀=0.85 |

**TR-4 Parameters:** ρ₀=1.0, η=1.2, κ=0.8, k=4, λ_R=1.0, ω=0.8

---

## See Also

- [Factory Functions](index.md#factory-functions) - Creating environments
- [Wrappers](wrappers.md) - PettingZoo adapters
- [Environment Reference](../environments/index.md) - Detailed documentation


## Technical Reports

- TR-1: [Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity](https://arxiv.org/pdf/2510.18802) (arXiv:2510.18802)
- TR-2: [Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics](https://arxiv.org/pdf/2510.24909) (arXiv:2510.24909)
- TR-3: [Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty](https://arxiv.org/pdf/2601.16237) (arXiv:2601.16237)
- TR-4: [Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity](https://arxiv.org/pdf/2604.01240) (arXiv:2604.01240)
