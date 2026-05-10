# Environment Analysis

This page enumerates the twenty environments in Coopetition-Gym v1, organized by the four mechanism classes corresponding to the four foundational technical reports. Each subsection identifies the environments, the agent count, and the focal phenomenon each environment is designed to expose.

> **Source for environment lists:** `coopetition_gym/envs/__init__.py` and the per-tier environment modules. **Source for returns:** `aggregates/returns_summary_v2.csv` in the released analysis pipeline.

---

## TR-1: Interdependence and Complementarity (5 environments)

These environments expose joint value creation through complementary investment, with structural competition arising from bargaining shares and asymmetric interdependence.

| Environment | Agents | Focal phenomenon |
|-------------|:------:|------------------|
| `TrustDilemma-v0` | 2 | Continuous prisoner's-dilemma-style trust game |
| `PartnerHoldUp-v0` | 2 | Asymmetric power and hold-up risk |
| `PlatformEcosystem-v0` | 4 | Platform owner with three complementors |
| `DynamicPartnerSelection-v0` | 5 | Endogenous partnership formation |
| `SLCD-v0` | 2 | Calibrated Samsung-Sony LCD joint venture (96.7% case validation) |

**Best learning algorithm under integrated reward (TR-1 tier):** QMIX at 69,630 mean episodic return. ISAC follows at 65,551 (rank 4). The CTDE leader holds across integrated and cooperative configurations; under private reward, ISAC takes the top spot at 38,761.

**Mechanism-class characteristic:** TR-1's relational state (the interdependence matrix `D_ij`) is **static** — agents do not modify it through their actions. Centralized critics retain their conventional advantage on this tier.

---

## TR-2: Trust and Reputation Dynamics (5 environments)

These environments expose trust as a dynamic state variable that evolves through observed cooperation signals, with the validated 3:1 negativity-bias asymmetry between trust gain (λ⁺=0.10) and trust loss (λ⁻=0.30).

| Environment | Agents | Focal phenomenon |
|-------------|:------:|------------------|
| `RecoveryRace-v0` | 2 | Trust recovery dynamics after a defection event |
| `SynergySearch-v0` | 2 | Building trust during exploratory partnership |
| `RenaultNissan-v0` | 2 | Calibrated Renault-Nissan Alliance (81.7% case validation) |
| `CooperativeNegotiation-v0` | 2 | Multi-round bargaining under reputation pressure |
| `ReputationMarket-v0` | 4 | Reputation effects with multiple potential partners |

**Best learning algorithm under integrated reward (TR-2 tier):** ISAC at 65,368 mean episodic return. COMA follows at 60,792 (rank 2). The independent-learning leader holds across all three reward configurations.

**Mechanism-class characteristic:** TR-2's relational state is **action-mutable** — agents modify trust and reputation through their cooperation actions. Independent learners outperform centralized critics on this tier; the local reward gradient remains stationary along the training trajectory in a way that the centrally-computed value cannot.

---

## TR-3: Collective Action and Loyalty (5 environments)

These environments expose team production with free-riding incentives and loyalty-modified payoffs. Agents face the classic public-goods problem with the loyalty mechanism specified in TR-3.

| Environment | Agents | Focal phenomenon |
|-------------|:------:|------------------|
| `TeamProduction-v0` | 3 | Symmetric team production with free-riding incentives |
| `LoyaltyTeam-v0` | 4 | Loyalty-modified team production |
| `CoalitionFormation-v0` | 5 | Endogenous coalition stability |
| `PublicGoods-v0` | 5 | Voluntary contribution mechanism |
| `ApacheProject-v0` | 6 | Calibrated Apache HTTP Server community (86.7% case validation) |

**Best learning algorithm under integrated reward (TR-3 tier):** ISAC at 1,272,467 mean episodic return. MASAC follows at 1,138,192 (rank 2). Independent learning dominates this tier under integrated and cooperative reward; under private reward, ISAC also leads at 105,293.

**Notable result on TR-3:** ISAC exceeds the highest mean episodic return achievable by any constant-action policy on **all five environments** — see [Case Study Validation](case_study_validation.md) for the per-environment exceedance values. The exceedance is positive on every seed-environment pair.

**Notable failure on TR-3:** MADDPG, MATD3, and M3DDPG produce 100% NaN returns on `ApacheProject-v0` under integrated and cooperative reward — see [Reward-Type Ablation](reward_type_ablation.md) for the full failure pattern.

**Mechanism-class characteristic:** TR-3's loyalty modifier `θ_i` is action-mutable through accumulation of contribution history. As with TR-2, independent learning outperforms centralized critics on this tier under integrated reward.

---

## TR-4: Sequential Interaction and Reciprocity (5 environments)

These environments expose memory-bounded reciprocity with the bounded response function `φ(x) = tanh(κx)`, capturing how agents condition current actions on observed past behavior.

| Environment | Agents | Focal phenomenon |
|-------------|:------:|------------------|
| `ReciprocalDilemma-v0` | 2 | Symmetric repeated dilemma with bounded reciprocity |
| `GiftExchange-v0` | 2 | Sequential gift-exchange with reciprocity expectations |
| `IndirectReciprocity-v0` | 4 | Reputation-mediated indirect reciprocity |
| `GraduatedSanction-v0` | 6 | Proportional response to graduated defection |
| `AppleAppStore-v0` | 4 | Calibrated Apple iOS App Store ecosystem (87.3% case validation) |

**Best learning algorithm under integrated reward (TR-4 tier):** COMA at 125,819 mean episodic return. ISAC follows at 122,208 (rank 2). The CTDE leader holds across integrated and cooperative reward; under private reward, MeanFieldAC takes the top spot at 74,063.

**Notable crossover on AppleAppStore-v0:** Under private reward (`D_ij=0`), ISAC outperforms COMA by +7.2%; under integrated reward, COMA outperforms ISAC by 1.8%; under cooperative reward, COMA outperforms ISAC by 4.2%. See [Reward-Type Ablation](reward_type_ablation.md) for the analysis.

**Mechanism-class characteristic:** TR-4's relational state is **history-encoded** — the bounded-memory window summarizes past actions. Centralized critics regain their conventional advantage on this tier under integrated reward, distinguishing TR-4 from TR-2 and TR-3.

---

## Cross-tier patterns

### Two algorithms appear in every tier's top five (integrated reward)

- **ISAC** — rank 4, 1, 1, 2 on TR-1, TR-2, TR-3, TR-4 respectively.
- **COMA** — rank 2, 2, 3, 1 on TR-1, TR-2, TR-3, TR-4 respectively.

### Mean-Field actor-critic is restricted

The mean-field approximation that MeanFieldAC relies on degenerates at N=2; this algorithm is therefore evaluated only on environments with N≥3. It does not appear in TR-1 dyadic results (`TrustDilemma-v0`, `PartnerHoldUp-v0`, `SLCD-v0`) or TR-2 dyadic results (`RecoveryRace-v0`, `SynergySearch-v0`, `RenaultNissan-v0`, `CooperativeNegotiation-v0`).

### On-policy PPO algorithms underperform on coopetitive tasks

Across all four tiers under integrated reward, IPPO, SelfPlay_PPO, and IA2C consistently rank in the bottom three. Their median `D_ij` contributions are below 20%, indicating that their learned policies are largely insensitive to whether reward incorporates partner payoffs. The pattern is consistent with a capacity-ceiling interpretation: these algorithms converge to policies that derive return primarily from solitary play rather than from coupled action.

### Reward-mode sensitivity varies across tiers

Per-tier median `D_ij` contribution is approximately 30% on TR-1, 24% on TR-2, 59% on TR-3, and 29% on TR-4. TR-3 algorithms derive a substantially larger fraction of their return from partner-payoff coupling than algorithms on the other three tiers, and TR-2 is the only tier where some algorithm-environment pairs show negative contributions. See [Reward-Type Ablation](reward_type_ablation.md).

---

## Deterministic-policy multi-agent algorithm failure on N=6

The `ApacheProject-v0` environment is the largest in the benchmark at 6 agents. Three deterministic-policy multi-agent algorithms (MADDPG, MATD3, M3DDPG) produce 100% NaN returns on this environment under integrated and cooperative reward, while predominantly converging under private reward. The contrast with stochastic-policy multi-agent algorithms (MASAC: converges under all three modes) and with independent learners (LOLA: converges under all three modes) localizes the failure to a specific intersection of algorithm class, environment, and reward mode rather than to any single factor.

The phenomenon is documented further in [Reward-Type Ablation](reward_type_ablation.md) and traced cell-by-cell in `aggregates/all_nan_cells_v2.txt`.