# Environment Reference

This section provides detailed documentation for all 10 environments in Coopetition-Gym.

---

## Environment Overview

| Environment | Agents | Category | Key Challenge |
|-------------|--------|----------|---------------|
| [TrustDilemma-v0](trust_dilemma.md) | 2 | Dyadic | Long-horizon impulse control |
| [PartnerHoldUp-v0](partner_holdup.md) | 2 | Dyadic | Power dynamics and exploitation |
| [PlatformEcosystem-v0](platform_ecosystem.md) | 1+N | Ecosystem | Ecosystem health management |
| [DynamicPartnerSelection-v0](dynamic_partner_selection.md) | N | Ecosystem | Social learning and signaling |
| [RecoveryRace-v0](recovery_race.md) | 2 | Benchmark | Planning under trust constraints |
| [SynergySearch-v0](synergy_search.md) | 2 | Benchmark | Exploration vs. exploitation |
| [SLCD-v0](slcd.md) | 2 | Case Study | Validated Samsung-Sony model |
| [RenaultNissan-v0](renault_nissan.md) | 2 | Case Study | Multi-phase alliance dynamics |
| [CooperativeNegotiation-v0](cooperative_negotiation.md) | 2 | Extended | Commitment and breach penalties |
| [ReputationMarket-v0](reputation_market.md) | N | Extended | Reputation as strategic asset |

---

## Categories

### Dyadic Environments (2-Agent)

Micro-level scenarios modeling direct partnerships between two agents. Ideal for:
- Studying bilateral negotiation dynamics
- Understanding trust evolution in partnerships
- Testing basic MARL algorithms

**Environments:**
- [TrustDilemma-v0](trust_dilemma.md) - Continuous Prisoner's Dilemma with trust
- [PartnerHoldUp-v0](partner_holdup.md) - Asymmetric power relationships

### Ecosystem Environments (N-Agent)

Macro-level scenarios with multiple interacting agents. Ideal for:
- Studying network effects and platform dynamics
- Understanding reputation-based coordination
- Testing scalable MARL algorithms

**Environments:**
- [PlatformEcosystem-v0](platform_ecosystem.md) - Hub-spoke platform model
- [DynamicPartnerSelection-v0](dynamic_partner_selection.md) - Reputation-based matching

### Benchmark Environments

Research-focused environments designed for algorithm evaluation. Ideal for:
- Comparing algorithm performance
- Studying specific learning challenges
- Reproducible research

**Environments:**
- [RecoveryRace-v0](recovery_race.md) - Post-crisis trust recovery
- [SynergySearch-v0](synergy_search.md) - Hidden parameter discovery

### Validated Case Studies

Environments with parameters validated against real business data. Ideal for:
- Realistic simulation studies
- Validating theoretical models
- Policy analysis

**Environments:**
- [SLCD-v0](slcd.md) - Samsung-Sony Joint Venture (58/60 validation)
- [RenaultNissan-v0](renault_nissan.md) - Multi-phase alliance model

### Extended Environments

Advanced scenarios with additional game mechanics. Ideal for:
- Studying communication and commitment
- Understanding market dynamics
- Testing sophisticated strategies

**Environments:**
- [CooperativeNegotiation-v0](cooperative_negotiation.md) - Negotiation with contracts
- [ReputationMarket-v0](reputation_market.md) - Tiered reputation market

---

## Common Interface

All environments share a common interface:

```python
import coopetition_gym

# Create environment
env = coopetition_gym.make("EnvironmentName-v0")

# Reset
obs, info = env.reset(seed=42)

# Step
obs, rewards, terminated, truncated, info = env.step(actions)

# Access spaces
obs_space = env.observation_space
act_space = env.action_space

# Get info
n_agents = env.n_agents
endowments = env.endowments
```

---

## Choosing an Environment

### By Learning Challenge

| Challenge | Recommended Environment |
|-----------|------------------------|
| Basic MARL | TrustDilemma-v0 |
| Credit assignment | PlatformEcosystem-v0 |
| Partner selection | DynamicPartnerSelection-v0 |
| Hidden states | SynergySearch-v0 |
| Long-term planning | RecoveryRace-v0 |
| Power asymmetry | PartnerHoldUp-v0 |
| Communication | CooperativeNegotiation-v0 |
| Market dynamics | ReputationMarket-v0 |

### By Research Area

| Research Area | Recommended Environments |
|---------------|-------------------------|
| Game Theory | TrustDilemma-v0, SynergySearch-v0 |
| Platform Economics | PlatformEcosystem-v0, ReputationMarket-v0 |
| Alliance Management | SLCD-v0, RenaultNissan-v0 |
| Trust & Reputation | RecoveryRace-v0, DynamicPartnerSelection-v0 |
| Negotiation | CooperativeNegotiation-v0, PartnerHoldUp-v0 |

---

## Environment Comparison

### Trust Dynamics Intensity

| Environment | Trust Sensitivity | Reputation Effects |
|-------------|-------------------|-------------------|
| TrustDilemma-v0 | High | Moderate |
| PartnerHoldUp-v0 | Very High | High |
| RecoveryRace-v0 | Extreme | Very High |
| PlatformEcosystem-v0 | Moderate | Moderate |
| CooperativeNegotiation-v0 | High | High |

### Scalability

| Environment | Fixed Agents | Configurable | Max Tested |
|-------------|--------------|--------------|------------|
| TrustDilemma-v0 | 2 | No | 2 |
| PlatformEcosystem-v0 | 1+N | Yes | 20 |
| DynamicPartnerSelection-v0 | N | Yes | 20 |
| ReputationMarket-v0 | N | Yes | 20 |

---

## Next Steps

- Read individual environment documentation for detailed parameters
- Check [Tutorials](../tutorials/index.md) for usage examples
- See [API Reference](../api/index.md) for complete method documentation
