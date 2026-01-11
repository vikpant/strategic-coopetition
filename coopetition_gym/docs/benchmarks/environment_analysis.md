# Environment Analysis

**Performance Patterns Across Coopetition-Gym Environments**

This page provides detailed analysis of algorithm performance across the 10 Coopetition-Gym environments, organized by environment category.

---

## Environment Overview

### Performance by Environment

| Environment | Mean Return | Std Dev | Best Algorithm | Best Return |
|-------------|-------------|---------|----------------|-------------|
| PlatformEcosystem-v0 | 77,508 | 53,526 | Constant_035 | 146,967 |
| ReputationMarket-v0 | 67,751 | 42,015 | Constant_075 | 163,761 |
| DynamicPartnerSelection-v0 | 61,373 | 19,690 | Constant_050 | 107,314 |
| RecoveryRace-v0 | 42,952 | 8,017 | Constant_050 | 59,508 |
| SLCD-v0 | 31,924 | 8,236 | Constant_035 | 50,946 |
| RenaultNissan-v0 | 31,560 | 7,090 | Constant_035 | 46,406 |
| SynergySearch-v0 | 29,934 | 5,712 | Constant_050 | 41,994 |
| CooperativeNegotiation-v0 | 29,216 | 10,679 | Constant_050 | 43,948 |
| TrustDilemma-v0 | 21,978 | 18,949 | Constant_050 | 46,267 |
| PartnerHoldUp-v0 | 20,420 | 19,247 | Constant_050 | 45,320 |

### Category Aggregates

| Category | Environments | Mean Return | Cooperation | Trust |
|----------|--------------|-------------|-------------|-------|
| Ecosystem | PlatformEcosystem, DynamicPartnerSelection | 69,440 | 59.8% | 62.3% |
| Extended | CooperativeNegotiation, ReputationMarket | 48,484 | 48.5% | 48.7% |
| Benchmark | RecoveryRace, SynergySearch | 36,443 | 42.4% | 40.5% |
| Validated | SLCD, RenaultNissan | 31,742 | 40.6% | 42.9% |
| Dyadic | TrustDilemma, PartnerHoldUp | 21,199 | 52.9% | 52.7% |

---

## Dyadic Environments (2-Agent)

### TrustDilemma-v0

**Description:** Continuous Prisoner's Dilemma with trust dynamics over 100 timesteps.

**Key Challenge:** Long-horizon impulse control—agents must resist short-term defection temptation to build trust.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_050 | 46,267 | 55.0% | 98.7% |
| 2 | ISAC | 44,660 | 80.4% | 98.6% |
| 3 | Constant_075 | 43,648 | 78.7% | 98.7% |
| 4 | Constant_035 | 36,612 | 38.5% | 55.4% |
| 5 | MADDPG | 29,478 | 75.8% | 73.5% |
| ... | ... | ... | ... | ... |
| 20 | IPPO | 2,004 | 11.7% | 6.5% |

#### Insights

1. **Moderate cooperation wins**: Constant_050 outperforms Constant_075 despite lower cooperation—50% cooperation maximizes long-term trust-weighted returns

2. **ISAC discovers cooperation**: Among learning algorithms, only ISAC achieves near-optimal trust (98.6%)

3. **IPPO catastrophe**: IPPO achieves only 2,004 return (4.3% of optimal), demonstrating complete failure in basic coopetition

#### Optimal Strategy Pattern
```
Optimal: Consistent 50% cooperation → Builds trust → Unlocks synergy bonus
Failure: Defection → Trust erosion → Locked out of cooperative surplus
```

---

### PartnerHoldUp-v0

**Description:** Asymmetric power relationship where one partner makes relationship-specific investments.

**Key Challenge:** Managing power dynamics when one party has more leverage.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_050 | 45,320 | 55.0% | 98.7% |
| 2 | ISAC | 43,237 | 80.5% | 98.6% |
| 3 | Constant_075 | 41,833 | 78.7% | 98.7% |
| 4 | Constant_035 | 37,219 | 38.5% | 55.4% |
| 5 | M3DDPG | 30,406 | 75.8% | 73.6% |

#### Insights

1. **Power asymmetry doesn't change optimal strategy**: Same algorithms win as TrustDilemma

2. **Exploitation is suboptimal**: Algorithms that exploit power advantage (low cooperation) underperform

3. **Trust as protection**: High trust (>90%) protects against hold-up attempts

---

## Ecosystem Environments (N-Agent)

### PlatformEcosystem-v0

**Description:** Platform with 5 developers competing for platform resources while depending on platform health.

**Key Challenge:** Balancing individual contribution vs. free-riding on others' contributions.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_035 | 146,967 | 38.5% | 55.4% |
| 2 | Constant_050 | 138,286 | 55.0% | 98.7% |
| 3 | Constant_075 | 127,605 | 78.7% | 98.7% |
| 4 | VDN | 123,704 | 49.6% | 61.9% |
| 5 | ISAC | 118,889 | 80.0% | 98.6% |

#### Insights

1. **Lower cooperation wins**: Constant_035 outperforms Constant_050—unique among all environments

2. **Free-rider advantage**: In 5-agent setting, moderate contribution while others cooperate maximizes individual returns

3. **VDN beats continuous CTDE**: Value decomposition handles multi-agent credit assignment better

4. **MeanFieldAC applicable**: Only ecosystem environment where mean-field approximation is valid
   - MeanFieldAC achieves 92,022 (competitive with learning algorithms)

#### Why Constant_035 Wins

In multi-agent ecosystems:
- Platform health depends on total contributions
- Individual return depends on individual contribution efficiency
- At 35% contribution, agents capture maximum share while platform remains healthy
- Higher contribution (50-75%) "wastes" resources on platform health that others enjoy

---

### DynamicPartnerSelection-v0

**Description:** 4 agents with reputation-based partner matching each round.

**Key Challenge:** Building reputation to attract better partners while extracting value from current partnerships.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_050 | 107,314 | 55.0% | 98.7% |
| 2 | Constant_075 | 94,021 | 78.7% | 98.7% |
| 3 | ISAC | 81,620 | 79.8% | 98.6% |
| 4 | VDN | 72,192 | 49.7% | 62.1% |
| 5 | MADDPG | 64,254 | 75.8% | 73.5% |

#### Insights

1. **Reputation as strategic asset**: High cooperation builds reputation → attracts better partners → higher returns

2. **Partner selection amplifies trust**: Unlike fixed-partner environments, poor reputation leads to partner rejection

3. **Constant_050 optimal**: 50% cooperation balances reputation building with value extraction

---

## Benchmark Environments

### RecoveryRace-v0

**Description:** Post-crisis scenario where agents must rebuild trust after a shock.

**Key Challenge:** Coordinating recovery when both agents are in low-trust state.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_050 | 59,508 | 55.0% | 98.7% |
| 2 | ISAC | 57,348 | 80.4% | 98.6% |
| 3 | Constant_075 | 55,563 | 78.7% | 98.7% |
| 4 | Constant_035 | 47,694 | 38.5% | 55.4% |
| 5 | MADDPG | 42,850 | 75.8% | 73.5% |

#### Insights

1. **Recovery requires consistency**: Constant cooperation signals commitment to recovery

2. **ISAC learns recovery**: Near-optimal performance suggests SAC can discover trust-rebuilding strategies

3. **TitForTat fails**: Reactive strategies cannot escape low-trust equilibrium (TitForTat ranks #13)

---

### SynergySearch-v0

**Description:** Agents search for hidden complementarities through exploration.

**Key Challenge:** Exploration vs. exploitation when synergy benefits are unknown.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_050 | 41,994 | 55.0% | 98.7% |
| 2 | ISAC | 40,439 | 80.5% | 98.6% |
| 3 | Constant_075 | 38,975 | 78.7% | 98.7% |
| 4 | M3DDPG | 35,062 | 75.8% | 73.5% |
| 5 | MATD3 | 35,058 | 75.8% | 73.5% |

#### Insights

1. **Exploration via entropy**: ISAC's maximum entropy objective encourages synergy discovery

2. **CTDE clustering**: M3DDPG, MATD3, MASAC, MADDPG achieve identical results (35,058-35,062)

3. **Constant cooperation enables exploration**: Stable cooperation allows focus on synergy search

---

## Validated Case Studies

### SLCD-v0 (Samsung-Sony LCD Joint Venture)

**Description:** Parameters calibrated to Samsung-Sony S-LCD partnership (2004-2012).

**Validation:** 58/60 accuracy against historical outcomes.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_035 | 50,946 | 38.5% | 55.5% |
| 2 | Constant_050 | 49,870 | 55.0% | 98.7% |
| 3 | Constant_075 | 46,827 | 78.7% | 98.7% |
| 4 | ISAC | 45,555 | 80.3% | 98.6% |
| 5 | VDN | 43,789 | 49.5% | 61.8% |

#### Historical Alignment

The actual Samsung-Sony partnership exhibited:
- Moderate cooperation (technology sharing but separate branding)
- Eventual dissolution (2012) after market shifts

Constant_035 and Constant_050 best match this historical pattern of "cautious cooperation."

---

### RenaultNissan-v0

**Description:** Multi-phase Renault-Nissan Alliance dynamics.

**Key Challenge:** Adapting cooperation levels across alliance phases.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_035 | 46,406 | 38.5% | 55.3% |
| 2 | Constant_050 | 45,472 | 55.0% | 98.7% |
| 3 | ISAC | 43,033 | 80.5% | 98.6% |
| 4 | Constant_075 | 42,694 | 78.7% | 98.7% |
| 5 | MADDPG | 37,107 | 75.8% | 73.5% |

#### Insights

1. **Lower cooperation optimal**: Like SLCD, validated cases favor moderate (35-50%) cooperation

2. **Alliance complexity**: Multi-phase structure rewards adaptable strategies

3. **ISAC generalizes**: Strong performance across both validated environments

---

## Extended Environments

### CooperativeNegotiation-v0

**Description:** Multi-round negotiation with commitment and breach penalties.

**Key Challenge:** Balancing aggressive negotiation with relationship maintenance.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_050 | 43,948 | 55.0% | 98.7% |
| 2 | Constant_075 | 41,035 | 78.7% | 98.7% |
| 3 | ISAC | 38,321 | 80.3% | 98.6% |
| 4 | Constant_035 | 33,908 | 38.5% | 55.4% |
| 5 | MADDPG | 32,469 | 75.8% | 73.5% |

#### Insights

1. **Commitment value**: Breach penalties make trust especially valuable

2. **Constant_050 dominates**: Predictable cooperation avoids breach situations

3. **Lower variance in returns**: Negotiation structure reduces outcome uncertainty

---

### ReputationMarket-v0

**Description:** Market with reputation tiers determining access to opportunities.

**Key Challenge:** Managing reputation as a strategic asset while extracting market value.

#### Performance Rankings

| Rank | Algorithm | Mean Return | Cooperation | Trust |
|------|-----------|-------------|-------------|-------|
| 1 | Constant_075 | 163,761 | 78.7% | 98.7% |
| 2 | Constant_050 | 146,966 | 55.0% | 98.7% |
| 3 | ISAC | 144,730 | 80.3% | 98.6% |
| 4 | M3DDPG | 120,877 | 75.8% | 73.5% |
| 5 | MATD3 | 120,873 | 75.8% | 73.5% |

#### Insights

1. **High cooperation wins**: Only environment where Constant_075 beats Constant_050

2. **Reputation premium**: Higher reputation tiers provide multiplicative benefits

3. **Highest absolute returns**: ReputationMarket generates the highest returns among all environments (163,761)

#### Why Constant_075 Wins Here

ReputationMarket mechanics:
- Reputation tiers unlock progressively better opportunities
- High cooperation → High reputation → Premium tier access
- The reputation multiplier outweighs the cost of higher cooperation

---

## Cross-Environment Patterns

### Optimal Cooperation Level by Environment Type

| Environment Type | Optimal Cooperation | Rationale |
|------------------|---------------------|-----------|
| Dyadic (fixed partner) | 50% | Balance value creation/capture |
| Ecosystem (N-agent) | 35% | Free-rider optimization |
| Reputation-based | 75% | Reputation multiplier |
| Validated (real cases) | 35-50% | Matches historical patterns |
| Benchmark (general) | 50% | Robust default |

### Algorithm Robustness

**Most Robust (low variance across environments):**
1. ISAC: Consistently #1-3 learning algorithm
2. Constant_050: Never below #3 overall
3. MADDPG: Stable mid-tier performance

**Least Robust (high variance):**
1. MeanFieldAC: Only applicable to ecosystem environments
2. MAPPO: Performance varies 10x across environments
3. IPPO: Consistently poor but with high variance

### Environment Difficulty

**Easiest (highest mean returns):**
1. ReputationMarket-v0 (67,751)
2. PlatformEcosystem-v0 (77,508)
3. DynamicPartnerSelection-v0 (61,373)

**Hardest (lowest mean returns):**
1. PartnerHoldUp-v0 (20,420)
2. TrustDilemma-v0 (21,978)
3. CooperativeNegotiation-v0 (29,216)

**Interpretation:** Dyadic environments are harder because there's no opportunity to free-ride or leverage reputation with multiple partners.

---

## Environment Selection Guide

### For Algorithm Evaluation

| Goal | Recommended Environment | Rationale |
|------|------------------------|-----------|
| Basic coopetition | TrustDilemma-v0 | Simplest dynamics |
| Multi-agent scaling | PlatformEcosystem-v0 | N-agent complexity |
| Trust recovery | RecoveryRace-v0 | Tests trust-building |
| Real-world validity | SLCD-v0 | Historically validated |
| Reputation dynamics | ReputationMarket-v0 | Reputation mechanics |

### For Research Questions

| Question | Environment(s) |
|----------|---------------|
| Does trust affect performance? | TrustDilemma-v0, PartnerHoldUp-v0 |
| How do algorithms scale? | PlatformEcosystem-v0, DynamicPartnerSelection-v0 |
| Can algorithms recover from crisis? | RecoveryRace-v0 |
| Do results match reality? | SLCD-v0, RenaultNissan-v0 |
| Is reputation strategic? | ReputationMarket-v0 |

---

## Navigation

- [Benchmark Overview](index.md)
- [Algorithm Comparison](algorithm_comparison.md)
- [Trust Dynamics](trust_dynamics.md)
- [Research Insights](research_insights.md)
