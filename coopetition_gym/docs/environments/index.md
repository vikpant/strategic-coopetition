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

## MARL Classification Summary

Quick reference for environment selection based on game-theoretic and MARL properties.

### Game Type and Structure

| Environment | Game Type | Cooperation | Observability | Symmetry |
|-------------|-----------|-------------|---------------|----------|
| TrustDilemma-v0 | Markov Game | Mixed-Motive | Full | Symmetric |
| PartnerHoldUp-v0 | Markov Game | Mixed-Motive (power asymmetry) | Full | Asymmetric |
| PlatformEcosystem-v0 | Markov Game / Mean-Field | Hub-spoke topology | Full | Heterogeneous |
| DynamicPartnerSelection-v0 | Markov Game / Mean-Field | Reputation externalities | Full | Symmetric |
| RecoveryRace-v0 | Markov Game | Cooperative goal | Full | Symmetric |
| SynergySearch-v0 | Bayes-Adaptive MG | Unknown synergy | **Partial** (hidden γ) | Symmetric |
| SLCD-v0 | Markov Game | Mixed-Motive coopetition | Full | Near-symmetric |
| RenaultNissan-v0 | Markov Game | Phase-dependent | Full | Asymmetric |
| CooperativeNegotiation-v0 | Markov Game + Contracts | Enforceable agreements | Full | Symmetric |
| ReputationMarket-v0 | Markov Game + Tiers | Reputation competition | Full | Symmetric |

### Action and State Spaces

| Environment | Action Space | State Dim | Horizon | Early Termination |
|-------------|--------------|-----------|---------|-------------------|
| TrustDilemma-v0 | Continuous [0,100]² | 17 | T=100 | Trust collapse |
| PartnerHoldUp-v0 | Continuous [0,120]×[0,80] | 17 | T=100 | Weak partner exit |
| PlatformEcosystem-v0 | Continuous [0,150]×[0,80]ᴺ | (N+1)+3(N+1)²+1 | T=100 | Ecosystem death |
| DynamicPartnerSelection-v0 | Continuous [0,100]ᴺ | N+3N²+1+N | T=50 | None |
| RecoveryRace-v0 | Continuous [0,100]² | 17 | T=150 | Success/Collapse |
| SynergySearch-v0 | Continuous [0,100]² | 17 (or 18) | T=100 | Trust collapse |
| SLCD-v0 | Continuous [0,100]² | 17 | T=100 | Relationship breakdown |
| RenaultNissan-v0 | Continuous [0,90]×[0,100] | 17 | T=100 | Alliance dissolution |
| CooperativeNegotiation-v0 | Continuous [0,100]² | 18 | T=100 | Trust collapse |
| ReputationMarket-v0 | Continuous [0,100]ᴺ | N+3N²+1+N | T=100 | None |

### Canonical Literature Comparisons

| Environment | Related Benchmarks & Literature |
|-------------|-------------------------------|
| TrustDilemma-v0 | Continuous IPD; Lerer & Peysakhovich (2017) |
| PartnerHoldUp-v0 | Principal-Agent; Stackelberg games |
| PlatformEcosystem-v0 | Mogul (ICML 2020); Multi-Principal Multi-Agent |
| DynamicPartnerSelection-v0 | Resnick & Zeckhauser (2002); Rating systems |
| RecoveryRace-v0 | Kim et al. (2004) trust repair |
| SynergySearch-v0 | Bayes-Adaptive MDP; Duff (2002) |
| SLCD-v0 | Ritala & Hurmelinna-Laukkanen (2009) |
| RenaultNissan-v0 | Segrestin (2005) "Partnering to Explore" |
| CooperativeNegotiation-v0 | Crawford & Sobel (1982); Raiffa (1982) |
| ReputationMarket-v0 | Shapiro (1983); Tadelis (1999) |

### Special Features

| Environment | Distinguishing Mechanism |
|-------------|------------------------|
| PartnerHoldUp-v0 | Asymmetric interdependence (D=0.35 vs D=0.85) |
| PlatformEcosystem-v0 | Hub-spoke topology, ecosystem collapse |
| DynamicPartnerSelection-v0 | Public reputation signals |
| RecoveryRace-v0 | Trust ceiling constraint (Θ = 1 - R) |
| SynergySearch-v0 | Hidden complementarity parameter γ |
| SLCD-v0 | Empirically validated (58/60 accuracy) |
| RenaultNissan-v0 | Four configurable historical phases |
| CooperativeNegotiation-v0 | Endogenous agreement formation, breach penalties |
| ReputationMarket-v0 | Four-tier reward multipliers (0.40× to 1.30×) |

### Equilibrium Summary

| Environment | Stage-Game NE | Pareto Optimal | Price of Anarchy | Notes |
|-------------|---------------|----------------|------------------|-------|
| TrustDilemma-v0 | $a^* \approx 35$ | $a^* = 100$ | ~1.55 | Trust-mediated cooperation |
| PartnerHoldUp-v0 | (45, 30) | (120, 80) | ~1.58 | Asymmetric Stackelberg |
| PlatformEcosystem-v0 | (55, 35) | (120, 65) | ~1.50 | Collective action threshold |
| SynergySearch-v0 | Conditional on $\gamma$ | Conditional | ~1.10 | Bayesian exploration needed |
| SLCD-v0 | Validated | Validated | N/A | 58/60 historical accuracy |
| RecoveryRace-v0 | Trust-constrained | Recovery-dependent | N/A | Ceiling $\Theta = 1 - R$ |
| CooperativeNegotiation-v0 | Pre-agreement | Post-agreement | ~1.40 | Breach penalty enforces |
| ReputationMarket-v0 | Tier-dependent | Premium tier | ~1.35 | Reputation competition |

**Key Insights:**
- All environments exhibit **cooperation deficit** in myopic equilibrium
- Trust dynamics create **multiple equilibria** (high-trust cooperative, low-trust defection)
- Power asymmetry in PartnerHoldUp-v0 and PlatformEcosystem-v0 creates **exploitation risk**
- SynergySearch-v0 requires **exploration** to discover optimal equilibrium

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
