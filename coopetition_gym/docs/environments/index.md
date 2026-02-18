# Environment Reference

This section provides detailed documentation for all 20 environments in Coopetition-Gym.

> **Not sure which environment to use?** Try the [Environment Finder](../finder/index.html) - an interactive tool that matches your research questions to the most relevant environments.

---

## Action Space Model

All Coopetition-Gym v1.x environments implement the **uniaxial treatment** of coopetition, where agents choose cooperation levels along a `[0, endowment]` continuum. This design follows the social dilemma tradition in coopetition research (Bengtsson & Kock, 2000; Lado et al., 1997).

**Key characteristics:**
- **Action interpretation**: Higher values represent greater cooperation/investment in joint value creation
- **Competition modeling**: Competitive dynamics emerge through structural parameters—interdependence matrices, bargaining shares, and trust evolution—rather than explicit competitive actions
- **Theoretical grounding**: The uniaxial and biaxial treatments represent complementary paradigms in the coopetition literature, each capturing different aspects of strategic interaction

Coopetition-Gym v2.x will introduce **biaxial treatment** with independent cooperation and competition dimensions, enabling agents to simultaneously vary investment in value creation and value capture. See [Scope and Strategic Roadmap](../scope_roadmap.md) for theoretical rationale and extension plans.

---

## Environment Overview

### TR-1: Interdependence and Complementarity (5 Environments)

| Environment | Agents | Category | Key Challenge |
|-------------|--------|----------|---------------|
| [PartnerHoldUp-v0](partner_holdup.md) | 2 | Dyadic | Power dynamics and exploitation |
| [PlatformEcosystem-v0](platform_ecosystem.md) | 1+N | Ecosystem | Ecosystem health management |
| [DynamicPartnerSelection-v0](dynamic_partner_selection.md) | N | Ecosystem | Social learning and signaling |
| [SynergySearch-v0](synergy_search.md) | 2 | Benchmark | Exploration vs. exploitation |
| [RenaultNissan-v0](renault_nissan.md) | 2 | Case Study | Multi-phase alliance dynamics |

### TR-2: Trust and Reputation Dynamics (5 Environments)

| Environment | Agents | Category | Key Challenge |
|-------------|--------|----------|---------------|
| [TrustDilemma-v0](trust_dilemma.md) | 2 | Dyadic | Long-horizon impulse control |
| [RecoveryRace-v0](recovery_race.md) | 2 | Benchmark | Planning under trust constraints |
| [SLCD-v0](slcd.md) | 2 | Case Study | Validated Samsung-Sony model |
| [CooperativeNegotiation-v0](cooperative_negotiation.md) | 2 | Extended | Commitment and breach penalties |
| [ReputationMarket-v0](reputation_market.md) | N | Extended | Reputation as strategic asset |

### TR-3: Collective Action and Loyalty (5 Environments)

| Environment | Agents | Category | Key Challenge |
|-------------|--------|----------|---------------|
| [TeamProduction-v0](team_production.md) | 4 | Collective Action | Free-rider dynamics |
| [LoyaltyTeam-v0](loyalty_team.md) | 4 | Collective Action | Loyalty-sustained cooperation |
| [CoalitionFormation-v0](coalition_formation.md) | 6 | Collective Action | Coalition stability with exclusion |
| [ApacheProject-v0](apache_project.md) | 8-40 | Validated (TR-3) | Phase-dependent contributor dynamics |
| [PublicGoods-v0](public_goods.md) | 5 | Collective Action | Classic public goods contribution |

### TR-4: Sequential Interaction and Reciprocity (5 Environments)

| Environment | Agents | Category | Key Challenge |
|-------------|--------|----------|---------------|
| [ReciprocalDilemma-v0](reciprocal_dilemma.md) | 2 | Reciprocity | Conditional cooperation via memory |
| [GiftExchange-v0](gift_exchange.md) | 2 | Reciprocity | Asymmetric reciprocity sensitivity |
| [IndirectReciprocity-v0](indirect_reciprocity.md) | 4 | Reciprocity | Reputation-mediated cooperation |
| [GraduatedSanction-v0](graduated_sanction.md) | 6 | Reciprocity | Graduated proportional sanctions |
| [AppleAppStore-v0](apple_app_store.md) | 3 | Validated (TR-4) | Platform power and reciprocity dynamics |

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
| TeamProduction-v0 | N-player Markov Game | Mixed-Motive (team vs. individual) | Full | Symmetric |
| LoyaltyTeam-v0 | N-player Markov Game | Mixed-Motive + loyalty amplification | Full | Symmetric |
| CoalitionFormation-v0 | N-player Markov Game | Coalition-based with exclusion | Full | Dynamic membership |
| ApacheProject-v0 | N-player Markov Game | Open source contribution | Full | Phase-symmetric |
| PublicGoods-v0 | N-player Markov Game | Classic public goods | Full | Symmetric |
| ReciprocalDilemma-v0 | 2-player Markov Game | Mixed-Motive + reciprocity | Full | Symmetric |
| GiftExchange-v0 | 2-player Markov Game | Asymmetric gift exchange | Full | Asymmetric |
| IndirectReciprocity-v0 | 4-player Markov Game | Reputation externalities | Full | Symmetric |
| GraduatedSanction-v0 | 6-player Markov Game | Common-pool resource | Full | Symmetric |
| AppleAppStore-v0 | 3-player Markov Game | Platform coopetition | Full | Asymmetric |

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
| TeamProduction-v0 | Continuous [0,50]ᴺ | Phase-dependent | T=100 | None |
| LoyaltyTeam-v0 | Continuous [0,50]ᴺ | Phase-dependent | T=100 | None |
| CoalitionFormation-v0 | Continuous [0,50]ᴺ | Phase-dependent | T=150 | Coalition collapse |
| ApacheProject-v0 | Continuous [0,50]ᴺ | Phase-dependent | T=60 | None |
| PublicGoods-v0 | Continuous [0,endowment]ᴺ | Phase-dependent | T=100 | None |
| ReciprocalDilemma-v0 | Continuous [0,100]² | 17+ | T=100 | None |
| GiftExchange-v0 | Continuous [0,100]×[0,80] | 17+ | T=100 | None |
| IndirectReciprocity-v0 | Continuous [0,100]⁴ | Phase-dependent | T=150 | None |
| GraduatedSanction-v0 | Continuous [0,100]⁶ | Phase-dependent | T=200 | None |
| AppleAppStore-v0 | Continuous [0,100]×[0,80]×[0,60] | Phase-dependent | T=66 | None |

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
| TeamProduction-v0 | Holmström (1982); Alchian & Demsetz (1972) |
| LoyaltyTeam-v0 | Akerlof & Kranton (2010); Kandel & Lazear (1992) |
| CoalitionFormation-v0 | Ray (2007); Greenberg (1994) |
| ApacheProject-v0 | Mockus et al. (2002); Lerner & Tirole (2002) |
| PublicGoods-v0 | Fehr & Gächter (2000); Ledyard (1995) |
| ReciprocalDilemma-v0 | Axelrod (1984); Killingback & Doebeli (2002) |
| GiftExchange-v0 | Fehr, Kirchsteiger & Riedl (1993); Akerlof (1982) |
| IndirectReciprocity-v0 | Nowak & Sigmund (1998, 2005); Panchanathan & Boyd (2004) |
| GraduatedSanction-v0 | Ostrom (1990); Fehr & Gächter (2000) |
| AppleAppStore-v0 | Parker, Van Alstyne & Choudary (2016); Rochet & Tirole (2003) |

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
| TeamProduction-v0 | Nash equilibrium baseline, free-rider dynamics |
| LoyaltyTeam-v0 | TR-3 loyalty modifiers (φ_B=0.8, φ_C=0.3) |
| CoalitionFormation-v0 | Dynamic exclusion/reentry, minimum coalition |
| ApacheProject-v0 | Empirically validated (52/60), four project phases |
| PublicGoods-v0 | Configurable multiplier, punishment mechanism |
| ReciprocalDilemma-v0 | TR-4 reciprocity modifier, k=5 memory window |
| GiftExchange-v0 | Asymmetric reciprocity (ρ worker 2.3× employer) |
| IndirectReciprocity-v0 | 4-agent reputation cascades, k=7 memory |
| GraduatedSanction-v0 | Graduated sanctions (κ=0.8), k=10 escalation |
| AppleAppStore-v0 | Empirically validated (48/55), 66-quarter history |

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
| TeamProduction-v0 | $a^* \approx 6.8$ | $a^{opt} \approx 18.4$ | ~2.5 | Free-rider equilibrium |
| LoyaltyTeam-v0 | Above Nash | Social optimum | ~1.2 | Loyalty sustains cooperation |
| CoalitionFormation-v0 | Coalition-stable | Full coalition | ~1.4 | Exclusion threat maintains |
| ApacheProject-v0 | Phase-specific | Validated | N/A | 52/60 historical accuracy |
| PublicGoods-v0 | Zero contribution | Full contribution | ~2.0 | Classic public goods |
| ReciprocalDilemma-v0 | ~35 (similar to TD) | 100 | ~1.55 | Reciprocity enables TFT |
| GiftExchange-v0 | Low effort from worker | Fair exchange | ~1.6 | Asymmetric reciprocity |
| IndirectReciprocity-v0 | Free-riding | Full cooperation | ~2.0 | Reputation sustains cooperation |
| GraduatedSanction-v0 | Under-contribution | Full contribution | ~2.0 | Graduated sanctions deter |
| AppleAppStore-v0 | Phase-specific | Validated | N/A | 48/55 historical accuracy |

**Key Insights:**
- All environments exhibit **cooperation deficit** in myopic equilibrium
- Trust dynamics create **multiple equilibria** (high-trust cooperative, low-trust defection)
- Power asymmetry in PartnerHoldUp-v0 and PlatformEcosystem-v0 creates **exploitation risk**
- SynergySearch-v0 requires **exploration** to discover optimal equilibrium
- TR-3 environments demonstrate **loyalty mechanisms** can sustain above-Nash cooperation
- ApacheProject-v0 achieves **52/60 empirical validation** against real open source data
- TR-4 environments demonstrate **reciprocity mechanisms** enable conditional cooperation
- AppleAppStore-v0 achieves **48/55 empirical validation** against real platform data

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

### Collective Action Environments (TR-3)

Team production and collective action scenarios with loyalty dynamics. Ideal for:
- Studying free-rider problems and their solutions
- Understanding loyalty mechanisms in teams
- Testing coalition stability and dynamics
- Validating against real open source project data

**Environments:**
- [TeamProduction-v0](team_production.md) - Baseline team production with free-rider dynamics
- [LoyaltyTeam-v0](loyalty_team.md) - Team production with TR-3 loyalty mechanisms
- [CoalitionFormation-v0](coalition_formation.md) - Dynamic coalition with entry/exit
- [ApacheProject-v0](apache_project.md) - Validated Apache HTTP Server case study (52/60)
- [PublicGoods-v0](public_goods.md) - Classic public goods with collective action modifiers

### Reciprocity Environments (TR-4)

Sequential interaction and reciprocity scenarios with bounded memory. Ideal for:
- Studying conditional cooperation and tit-for-tat strategies
- Understanding how memory and reputation sustain cooperation
- Testing reciprocity in asymmetric power structures
- Validating against real platform ecosystem data

**Environments:**
- [ReciprocalDilemma-v0](reciprocal_dilemma.md) - Continuous PD with direct reciprocity
- [GiftExchange-v0](gift_exchange.md) - Asymmetric employer-worker gift exchange
- [IndirectReciprocity-v0](indirect_reciprocity.md) - Population-level reputation and image scoring
- [GraduatedSanction-v0](graduated_sanction.md) - Common-pool resource with graduated sanctions
- [AppleAppStore-v0](apple_app_store.md) - Validated Apple iOS case study (48/55)

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
| Free-rider problems | TeamProduction-v0 |
| Loyalty dynamics | LoyaltyTeam-v0 |
| Coalition stability | CoalitionFormation-v0 |
| Empirical validation | ApacheProject-v0, SLCD-v0, AppleAppStore-v0 |
| Public goods | PublicGoods-v0 |
| Conditional cooperation | ReciprocalDilemma-v0 |
| Asymmetric reciprocity | GiftExchange-v0 |
| Reputation dynamics | IndirectReciprocity-v0 |
| Commons governance | GraduatedSanction-v0 |
| Platform reciprocity | AppleAppStore-v0 |

### By Research Area

| Research Area | Recommended Environments |
|---------------|-------------------------|
| Game Theory | TrustDilemma-v0, SynergySearch-v0, TeamProduction-v0 |
| Platform Economics | PlatformEcosystem-v0, ReputationMarket-v0 |
| Alliance Management | SLCD-v0, RenaultNissan-v0 |
| Trust & Reputation | RecoveryRace-v0, DynamicPartnerSelection-v0 |
| Negotiation | CooperativeNegotiation-v0, PartnerHoldUp-v0 |
| Collective Action | TeamProduction-v0, LoyaltyTeam-v0, PublicGoods-v0 |
| Coalition Theory | CoalitionFormation-v0 |
| Open Source Dynamics | ApacheProject-v0 |
| Mechanism Design | LoyaltyTeam-v0, PublicGoods-v0 |
| Reciprocity & Memory | ReciprocalDilemma-v0, GiftExchange-v0 |
| Reputation Systems | IndirectReciprocity-v0, DynamicPartnerSelection-v0 |
| Commons & Sanctions | GraduatedSanction-v0, PublicGoods-v0 |
| Platform Ecosystems | AppleAppStore-v0, PlatformEcosystem-v0 |

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

### Loyalty Dynamics Intensity (TR-3)

| Environment | Loyalty Sensitivity | Free-Rider Pressure |
|-------------|---------------------|---------------------|
| TeamProduction-v0 | None (baseline) | Very High |
| LoyaltyTeam-v0 | High | Moderate (mitigated) |
| CoalitionFormation-v0 | High | High (exclusion threat) |
| ApacheProject-v0 | Phase-dependent | Phase-dependent |
| PublicGoods-v0 | Moderate | High |

### Reciprocity Dynamics Intensity (TR-4)

| Environment | Reciprocity Sensitivity | Memory Length | Agents |
|-------------|------------------------|---------------|--------|
| ReciprocalDilemma-v0 | Moderate (ρ=0.5) | k=5 | 2 |
| GiftExchange-v0 | High (asymmetric, ρ up to 0.7) | k=3 | 2 |
| IndirectReciprocity-v0 | Moderate (ρ=0.32, but 3 partners) | k=7 | 4 |
| GraduatedSanction-v0 | Graduated (ρ=0.12, but 5 partners) | k=10 | 6 |
| AppleAppStore-v0 | Very High asymmetry (0.16-0.82) | k=4 | 3 |

### Scalability

| Environment | Fixed Agents | Configurable | Max Tested |
|-------------|--------------|--------------|------------|
| TrustDilemma-v0 | 2 | No | 2 |
| PlatformEcosystem-v0 | 1+N | Yes | 20 |
| DynamicPartnerSelection-v0 | N | Yes | 20 |
| ReputationMarket-v0 | N | Yes | 20 |
| TeamProduction-v0 | N | Yes | 20 |
| LoyaltyTeam-v0 | N | Yes | 20 |
| CoalitionFormation-v0 | N | Yes | 20 |
| ApacheProject-v0 | Phase-specific | Yes (phase) | 40 |
| PublicGoods-v0 | N | Yes | 20 |
| ReciprocalDilemma-v0 | 2 | No | 2 |
| GiftExchange-v0 | 2 | No | 2 |
| IndirectReciprocity-v0 | 4 | No | 4 |
| GraduatedSanction-v0 | 6 | No | 6 |
| AppleAppStore-v0 | 3 | No | 3 |

---

## Next Steps

- Read individual environment documentation for detailed parameters
- Check [Tutorials](../tutorials/index.md) for usage examples
- See [API Reference](../api/index.md) for complete method documentation
