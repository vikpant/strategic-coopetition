# Scope and Strategic Roadmap

**Computational Foundations for Strategic Coopetition: Research Program Architecture**

This document provides the theoretical rationale, scope justification, and strategic roadmap for the ongoing research program on computational modeling of strategic coopetition. It serves as the authoritative reference for guiding and directing future extensions and advancements.

---

## Document Purpose

This scope roadmap addresses fundamental questions about the research program's design:

1. **Why does coopetition-gym use a [0, endowment] action space?**
2. **How is this grounded in game theory and coopetition literature?**
3. **What is the relationship between cooperation and competition in the model?**
4. **How will the research program evolve to address broader treatments?**

The document establishes that the current implementation represents **Phase 1 (Foundations)** of a multi-phase research program, with **Phase 2 (Extensions)** planned to introduce biaxial treatments of cooperation and competition.

---

## Part I: The Coopetition Modeling Debate

### 1.1 Defining Coopetition

**Coopetition** is a portmanteau of "cooperation" and "competition" describing the phenomenon of simultaneous cooperation and competition between economic actors. The central maxim of coopetition is:

> **"Cooperate to grow the pie, compete to split the pie."**
> — Brandenburger & Nalebuff (1996)

This implies that cooperation and competition may operate as distinct strategic dimensions rather than opposite poles of a single continuum.

### 1.2 The Uniaxial vs. Biaxial Debate

The coopetition literature contains a substantive debate about whether cooperation and competition should be modeled as:

#### Uniaxial (Continuum) Treatment

| Authors | Work | Position |
|---------|------|----------|
| Bengtsson & Kock (2000) | "Coopetition in Business Networks" | Cooperation-competition as ends of a spectrum; firms position along continuum |
| Lado, Boyd & Hanlon (1997) | "Competition, Cooperation, and the Search for Economic Rents" | "Syncretic rent-seeking" behavior on single dimension |
| Quintana-García & Benavides-Velasco (2004) | "Cooperation, Competition, and Innovative Capability" | Coopetition intensity as single variable measure |
| Padula & Dagnino (2007) | "Untangling the Rise of Coopetition" | Coopetition as intermediate position between pure cooperation and pure competition |

**Key argument**: Firms must allocate limited resources between cooperative and competitive activities; more of one implies less of the other. The choice is fundamentally about positioning on a cooperation-competition spectrum.

#### Biaxial (Orthogonal) Treatment

| Authors | Work | Position |
|---------|------|----------|
| Brandenburger & Nalebuff (1996) | "Co-opetition" | Value creation (cooperation) vs. value capture (competition) as distinct activities |
| Luo (2007) | "Coopetition in International Business" | Two dimensions: "coopetition intensity" × "coopetition scope" |
| Gnyawali & Park (2011) | "Co-opetition between Giants" | Simultaneous high cooperation AND high competition possible and observed |
| Bengtsson, Kock & Johansson (2014) | "A Systematic Review of Coopetition" | Recognizes both unidimensional and multidimensional conceptualizations |
| Bouncken et al. (2015) | "Coopetition: A Systematic Review" | Identifies tension between "balance" (uniaxial) and "simultaneity" (biaxial) views |

**Key argument**: Firms can simultaneously invest heavily in both cooperation (joint R&D, standard-setting) and competition (pricing, market positioning). These are orthogonal strategic choices.

### 1.3 Implications for Computational Modeling

| Aspect | Uniaxial Treatment | Biaxial Treatment |
|--------|-------------------|-------------------|
| **Action space** | 1D: [0, max] or [-max, +max] | 2D: [0, max]_coop × [0, max]_comp |
| **Strategic tension** | Resource allocation tradeoff | Independent optimization |
| **Nash equilibrium** | Single dimension | Multi-dimensional |
| **Computational tractability** | Higher (1D optimization) | Lower (2D optimization) |
| **MARL compatibility** | Direct (standard continuous) | Requires adaptation |
| **Interpretability** | Clear single metric | Richer but complex |

Both treatments are legitimate modeling choices with different strengths. The debate is not resolved in the literature, and different research questions may warrant different treatments.

---

## Part II: Foundations Series Justification (TR-1 through TR-4)

### 2.1 Scope of the Foundations Series

The "Computational Foundations for Strategic Coopetition" technical reports (TR-1 through TR-4) adopt the **uniaxial treatment** following the social dilemma tradition. This is a deliberate and defensible modeling choice.

| Technical Report | Focus | Action Interpretation |
|------------------|-------|----------------------|
| TR-1: Interdependence & Complementarity | Value creation through joint investment | a_i = contribution to synergy |
| TR-2: Trust & Reputation Dynamics | Trust evolution from observed cooperation | a_i = cooperation signal |
| TR-3: Collective Action & Loyalty | Team production and loyalty | a_i = effort contribution |
| TR-4: Sequential Interaction & Reciprocity | History-dependent cooperation | a_i = reciprocal response |

### 2.2 Theoretical Justification

The Foundations series models the **cooperation dimension** of coopetition, with competition captured through **structural parameters**:

#### Competition via Structural Parameters

| Parameter | Symbol | Competitive Interpretation |
|-----------|--------|---------------------------|
| Bargaining shares | α_i | Value capture allocation |
| Interdependence matrix | D_ij | Structural power asymmetries |
| Trust dynamics | T_ij, R_ij | Reputation-mediated future access |
| Endowment asymmetries | e_i | Resource-based competitive advantage |

This approach reflects the insight that competition often manifests through **structural position** rather than explicit competitive actions. An agent with high α captures more value regardless of action choice; an agent with asymmetric D_ij (high dependency on them, low dependency on others) holds structural power.

#### Grounding in Game Theory Literature

The [0, endowment] action space is standard in established game-theoretic models:

| Game Class | Action Space | "Zero" Meaning | Literature |
|------------|--------------|----------------|------------|
| Public Goods | [0, E] | Contribute nothing | Ledyard (1995), Isaac et al. (1988) |
| Trust Game | [0, E] | Send nothing | Berg, Dickhaut & McCabe (1995) |
| Continuous PD | [0, 1] | Full defection | Killingback & Doebeli (2002) |
| Social Dilemmas | [0, max] | Selfish action | Leibo et al. (2017), MARL literature |
| Common Pool Resources | [0, harvest_max] | Restrain extraction | Ostrom et al. (1992) |

The Foundations series extends this tradition to model **cooperation with structural competition**, not "cooperation vs. no action."

### 2.3 What the Current Model Captures

| Phenomenon | How Modeled | Where |
|------------|-------------|-------|
| Value creation | Synergy function g(a) | TR-1, all environments |
| Value capture | Bargaining shares α | All environments |
| Power asymmetry | Interdependence matrix D | PartnerHoldUp-v0, PlatformEcosystem-v0 |
| Trust dynamics | T, R evolution | TR-2, TrustDilemma-v0 |
| Reputation effects | Reputation damage ceiling | TR-2, RecoveryRace-v0 |
| Free-riding | Nash equilibrium analysis | TR-3, TeamProduction-v0 |
| Loyalty | Welfare internalization | TR-3, LoyaltyTeam-v0 |
| Coalition stability | Entry/exit dynamics | TR-3, CoalitionFormation-v0 |
| Reciprocity | History-dependent response | TR-4, ReciprocalDilemma-v0, GiftExchange-v0, IndirectReciprocity-v0, GraduatedSanction-v0, AppleAppStore-v0 |

### 2.4 What the Current Model Does NOT Capture

The uniaxial treatment does not model **active competition** as a strategic choice:

| Competitive Mechanism | Example | Status in Foundations |
|-----------------------|---------|----------------------|
| Price competition | Bertrand undercutting | Not modeled |
| Quantity competition | Cournot flooding | Not modeled |
| Sabotage | Actively harming rival | Not modeled |
| Market capture | Aggressive positioning | Not modeled |
| Rent-seeking | Contest for fixed prize | Not modeled |

These mechanisms require agents to take actions that impose **direct costs on rivals**—something the [0, endowment] action space cannot express. This is the domain of the planned Extensions series.

### 2.5 Terminology Clarification

To avoid confusion, the following terminology mapping is adopted:

| Common Term | Foundations Meaning | More Precise Term |
|-------------|---------------------|-------------------|
| "Defection" | Non-contribution to joint value | Retention, Non-investment |
| "Full cooperation" | Maximum contribution to synergy | Full investment |
| "Cooperation level" | Contribution to value creation | Investment level |
| "Competitive dynamics" | Structural position effects | Structural competition |

The term "defection" is borrowed from Prisoner's Dilemma literature and should be understood as "selfish retention" rather than "active harm."

---

## Part III: Extensions Series Roadmap (TR-5 onwards)

### 3.1 Planned Transition to Biaxial Treatment

The Extensions series will introduce **biaxial action spaces** where cooperation and competition are independent strategic dimensions.

```
FOUNDATIONS (TR-1 to TR-4)          EXTENSIONS (TR-5 onwards)
         │                                    │
    Uniaxial                              Biaxial
         │                                    │
    a ∈ [0, E]                    a = (a_coop, a_comp)
    Cooperation                   a_coop ∈ [0, E_coop]
    dimension                     a_comp ∈ [0, E_comp]
    only                          Both dimensions
```

### 3.2 Planned Technical Reports

| Report | Title | Focus | Key Innovation |
|--------|-------|-------|----------------|
| **TR-5** | Value Capture & Market Competition | Active competition dimension | Biaxial action space |
| **TR-6** | Sabotage & Defensive Strategies | Negative externality actions | Harm modeling |
| **TR-7** | Multi-Market Contact & Spillovers | Cross-market dynamics | Portfolio competition |
| **TR-8** | Coalition Competition & Rivalry | Inter-coalition competition | Nested game structure |

### 3.3 TR-5: Value Capture & Market Competition (Planned)

**Focus**: Introduce explicit competition dimension alongside cooperation.

#### Biaxial Action Space Design

```python
# Proposed action space for TR-5 environments
action_space = spaces.Dict({
    "cooperation": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
    "competition": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
})

# Alternative: Single Box with two dimensions
action_space = spaces.Box(
    low=np.array([0.0, 0.0]),
    high=np.array([100.0, 100.0]),
    shape=(2,),
    dtype=np.float32
)
```

#### Reward Structure

```python
# Biaxial reward function (conceptual)
def biaxial_reward(actions, params):
    # Cooperation creates value
    total_value = synergy_function(actions["cooperation"])

    # Competition allocates value
    shares = competition_function(actions["competition"])

    # Individual payoff
    payoff_i = shares[i] * total_value

    # Competition has direct costs
    competition_cost = cost_function(actions["competition"][i])

    return payoff_i - competition_cost
```

#### Planned Environments

| Environment | Description | Key Feature |
|-------------|-------------|-------------|
| ValueCapture-v0 | Basic biaxial environment | Independent coop/comp dimensions |
| PriceCompetition-v0 | Bertrand-style pricing | Price undercutting dynamics |
| MarketShare-v0 | Quantity competition | Market flooding effects |
| R&D-Race-v0 | Innovation competition | Technology capture |

### 3.4 TR-6: Sabotage & Defensive Strategies (Planned)

**Focus**: Actions that actively harm rivals (negative externalities).

#### Tripolar Action Space

```python
# Tripolar action space: cooperate, compete, or sabotage
action_space = spaces.Box(
    low=np.array([0.0, 0.0, 0.0]),
    high=np.array([100.0, 100.0, 100.0]),
    shape=(3,),
    dtype=np.float32
)
# Dimensions: [cooperation, competition, sabotage]
```

#### Sabotage Mechanics

| Action Type | Effect on Self | Effect on Rival |
|-------------|----------------|-----------------|
| Cooperation (a_coop) | + Synergy share | + Synergy share |
| Competition (a_comp) | + Larger share | - Smaller share |
| Sabotage (a_sab) | - Direct cost | - Output reduction |

#### Planned Environments

| Environment | Description | Key Feature |
|-------------|-------------|-------------|
| Sabotage-v0 | Basic sabotage mechanics | Negative externality actions |
| DefensiveInvestment-v0 | Protection against sabotage | Defense allocation |
| IndustrialEspionage-v0 | Information sabotage | Knowledge appropriation |

### 3.5 TR-7 and TR-8 (Conceptual)

| Report | Focus | Novel Contribution |
|--------|-------|-------------------|
| TR-7 | Multi-market contact | Cross-market retaliation and forbearance |
| TR-8 | Coalition competition | Intra-coalition cooperation, inter-coalition competition |

---

## Part IV: Environment Architecture Evolution

### 4.1 Current Environment Categories (v1.x)

| Category | Environments | Treatment |
|----------|--------------|-----------|
| Dyadic | TrustDilemma-v0, PartnerHoldUp-v0 | Uniaxial |
| Ecosystem | PlatformEcosystem-v0, DynamicPartnerSelection-v0 | Uniaxial |
| Benchmark | RecoveryRace-v0, SynergySearch-v0 | Uniaxial |
| Case Study | SLCD-v0, RenaultNissan-v0 | Uniaxial |
| Extended | CooperativeNegotiation-v0, ReputationMarket-v0 | Uniaxial |
| TR-3 | TeamProduction-v0, LoyaltyTeam-v0, CoalitionFormation-v0, ApacheProject-v0, PublicGoods-v0 | Uniaxial |

### 4.2 Planned Environment Categories (v2.x)

| Category | Environments | Treatment |
|----------|--------------|-----------|
| Biaxial Basic | ValueCapture-v0, BiaxialDilemma-v0 | Biaxial |
| Market Competition | PriceCompetition-v0, MarketShare-v0 | Biaxial |
| Technology | R&D-Race-v0, PatentCompetition-v0 | Biaxial |
| Sabotage | Sabotage-v0, DefensiveInvestment-v0 | Tripolar |
| Multi-Market | CrossMarket-v0, PortfolioCompetition-v0 | Biaxial |
| Coalition Rivalry | CoalitionRivalry-v0, AllianceWar-v0 | Nested biaxial |

### 4.3 Backward Compatibility

All v1.x environments will be preserved in v2.x releases:

```python
# v2.x usage - both paradigms available
import coopetition_gym

# Uniaxial environments (Foundations)
env_uniaxial = coopetition_gym.make("TrustDilemma-v0")
# action_space: Box([0, 100], shape=(2,))

# Biaxial environments (Extensions)
env_biaxial = coopetition_gym.make("ValueCapture-v0")
# action_space: Box([[0,0], [100,100]], shape=(2, 2))
```

---

## References

### Coopetition Literature (Uniaxial Tradition)

- Bengtsson, M., & Kock, S. (2000). "Coopetition" in business networks—to cooperate and compete simultaneously. *Industrial Marketing Management*, 29(5), 411-426.

- Lado, A. A., Boyd, N. G., & Hanlon, S. C. (1997). Competition, cooperation, and the search for economic rents: A syncretic model. *Academy of Management Review*, 22(1), 110-141.

- Quintana-García, C., & Benavides-Velasco, C. A. (2004). Cooperation, competition, and innovative capability: A panel data of European dedicated biotechnology firms. *Technovation*, 24(12), 927-938.

- Padula, G., & Dagnino, G. B. (2007). Untangling the rise of coopetition: The intrusion of competition in a cooperative game structure. *International Studies of Management & Organization*, 37(2), 32-52.

### Coopetition Literature (Biaxial Tradition)

- Brandenburger, A. M., & Nalebuff, B. J. (1996). *Co-opetition*. Currency Doubleday.

- Luo, Y. (2007). A coopetition perspective of global competition. *Journal of World Business*, 42(2), 129-144.

- Gnyawali, D. R., & Park, B. J. R. (2011). Co-opetition between giants: Collaboration with competitors for technological innovation. *Research Policy*, 40(5), 650-663.

- Bengtsson, M., Kock, S., & Johansson, M. (2014). A systematic review of coopetition. *Proceedings of the 2014 Annual Conference of the International Association for Business and Society*.

- Bouncken, R. B., Gast, J., Kraus, S., & Bogers, M. (2015). Coopetition: A systematic review, synthesis, and future research directions. *Review of Managerial Science*, 9(3), 577-601.

### Game Theory Foundations

- Ledyard, J. O. (1995). Public goods: A survey of experimental research. In J. H. Kagel & A. E. Roth (Eds.), *The Handbook of Experimental Economics* (pp. 111-194). Princeton University Press.

- Berg, J., Dickhaut, J., & McCabe, K. (1995). Trust, reciprocity, and social history. *Games and Economic Behavior*, 10(1), 122-142.

- Killingback, T., & Doebeli, M. (2002). The continuous prisoner's dilemma and the evolution of cooperation through reciprocal altruism with variable investment. *The American Naturalist*, 160(4), 421-438.

- Ostrom, E., Walker, J., & Gardner, R. (1992). Covenants with and without a sword: Self-governance is possible. *American Political Science Review*, 86(2), 404-417.

### Multi-Agent Reinforcement Learning

- Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). Multi-agent reinforcement learning in sequential social dilemmas. *Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems* (pp. 464-473).

### Technical Reports (This Research Program)

1. Pant, V., & Yu, E. (2025). Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity. *arXiv:2510.18802*.

2. Pant, V., & Yu, E. (2025). Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics. *arXiv:2510.24909*.

3. Pant, V., & Yu, E. (2025). Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty. *arXiv:2601.16237*.

4. Pant, V., & Yu, E. (2026). Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity. *Forthcoming*.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-09 | Initial scope roadmap documenting Foundations justification and Extensions roadmap |

---

## Navigation

- [Documentation Home](index.md)
- [Implementation Roadmap](roadmap.md)
- [Theoretical Foundations](theory/index.md)
- [Environment Reference](environments/index.md)
- [Benchmark Results](benchmarks/index.md)
- [API Documentation](api/index.md)
