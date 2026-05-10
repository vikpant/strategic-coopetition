# Algorithm Comparison

This page reports per-tier algorithm rankings for the 16 training algorithms, the 2 heuristic baselines, and the three reward configurations. All values are mean episodic return across cells with defined returns at the n=10+ fold (baseline seeds 99-105 plus extension seeds 106-111).

> **Source:** `aggregates/tier_summary_v2.txt` and `aggregates/ctde_vs_ind_boundary_v2.txt` in the released analysis pipeline. Symbols in the Class column: **I** = independent learner, **C** = centralized training with decentralized execution, **H** = heuristic baseline.

---

## Algorithm portfolio

### Independent learners (5)

| Name | Description |
|------|-------------|
| ISAC | Independent Soft Actor-Critic |
| IPPO | Independent Proximal Policy Optimization |
| IA2C | Independent Advantage Actor-Critic |
| IndependentREINFORCE | Per-agent REINFORCE |
| LOLA | Learning with Opponent-Learning Awareness |

### Population-based independent learners (2)

| Name | Description |
|------|-------------|
| SelfPlay_PPO | Self-play with PPO |
| FCP | Fictitious Co-Play |

### Centralized training, decentralized execution (CTDE) — continuous (4)

| Name | Description |
|------|-------------|
| MADDPG | Multi-Agent Deep Deterministic Policy Gradient |
| MATD3 | Multi-Agent Twin Delayed DDPG |
| M3DDPG | Minimax Multi-Agent DDPG |
| MASAC | Multi-Agent Soft Actor-Critic |

### CTDE — value decomposition (3)

| Name | Description |
|------|-------------|
| QMIX | Q-value mixing network |
| VDN | Value decomposition network |
| COMA | Counterfactual multi-agent policy gradient |

### CTDE — on-policy and mean-field (2)

| Name | Description |
|------|-------------|
| MAPPO | Multi-Agent PPO |
| MeanFieldAC | Mean-field actor-critic (evaluated only on N≥3 environments because the mean-field approximation degenerates at N=2) |

### Heuristic baselines (2)

| Name | Description |
|------|-------------|
| Random | Uniform random over the action space |
| TitForTat | Reciprocates the partner's previous action; cooperates on round one |

---

## Best CTDE versus best independent learner, by mechanism class

The clearest summary of cross-paradigm behavior is the per-tier comparison of the strongest representative of each paradigm. The **integrated** reward column reflects the calibrated `D_ij` configuration; the **private** column reflects `D_ij = 0`; the **cooperative** column reflects shared reward.

| Tier | Reward | Best CTDE | CTDE return | Best IND | IND return | Gap (%) | Winner |
|------|--------|-----------|-------------:|----------|-----------:|--------:|--------|
| TR-1 | private | QMIX | 37,183 | ISAC | 38,761 | +4.2 | IND |
| TR-1 | integrated | QMIX | 69,630 | ISAC | 65,551 | -5.9 | **CTDE** |
| TR-1 | cooperative | QMIX | 68,550 | ISAC | 65,779 | -4.0 | **CTDE** |
| TR-2 | private | COMA | 35,610 | ISAC | 39,727 | +11.6 | **IND** |
| TR-2 | integrated | COMA | 60,792 | ISAC | 65,368 | +7.5 | **IND** |
| TR-2 | cooperative | COMA | 61,110 | ISAC | 65,550 | +7.3 | **IND** |
| TR-3 | private | MADDPG | 101,680 | ISAC | 105,293 | +3.6 | IND |
| TR-3 | integrated | MASAC | 1,138,192 | ISAC | 1,272,467 | +11.8 | **IND** |
| TR-3 | cooperative | MASAC | 1,135,133 | ISAC | 1,272,373 | +12.1 | **IND** |
| TR-4 | private | MeanFieldAC | 74,063 | ISAC | 66,647 | -10.0 | **CTDE** |
| TR-4 | integrated | COMA | 125,819 | ISAC | 122,208 | -2.9 | **CTDE** |
| TR-4 | cooperative | COMA | 127,062 | ISAC | 122,217 | -3.8 | **CTDE** |

**Reading the table.** Under integrated reward, CTDE wins on TR-1 and TR-4; independent learning wins on TR-2 and TR-3. This is a 2-2 split across the four mechanism classes, not a uniform paradigm advantage. The split is preserved under cooperative reward but altered under private reward (TR-1 flips to IND under private).

---

## Full per-tier rankings (integrated reward)

The full ranking includes all 18 algorithms (16 training + 2 heuristic). Reading the columns: **Class** is the paradigm category, **Return** is mean episodic return at the n=10+ fold.

### TR-1 — Interdependence and Complementarity

| Rank | Algorithm | Class | Return |
|-----:|-----------|:-----:|-------:|
| 1 | QMIX | C | 69,630 |
| 2 | COMA | C | 67,208 |
| 3 | MeanFieldAC | C | 66,199 |
| 4 | ISAC | I | 65,551 |
| 5 | VDN | C | 63,556 |
| 6 | FCP | I | 48,869 |
| 7 | MADDPG | C | 45,083 |
| 8 | MATD3 | C | 44,578 |
| 9 | MAPPO | C | 43,142 |
| 10 | MASAC | C | 41,943 |
| 11 | M3DDPG | C | 34,938 |
| 12 | TitForTat | H | 34,182 |
| 13 | IndependentREINFORCE | I | 33,304 |
| 14 | LOLA | I | 29,828 |
| 15 | Random | H | 22,762 |
| 16 | IA2C | I | 20,172 |
| 17 | SelfPlay_PPO | I | 18,744 |
| 18 | IPPO | I | 18,485 |

### TR-2 — Trust and Reputation Dynamics

| Rank | Algorithm | Class | Return |
|-----:|-----------|:-----:|-------:|
| 1 | ISAC | I | 65,368 |
| 2 | COMA | C | 60,792 |
| 3 | QMIX | C | 56,680 |
| 4 | MASAC | C | 53,949 |
| 5 | VDN | C | 48,607 |
| 6 | TitForTat | H | 43,646 |
| 7 | MATD3 | C | 42,949 |
| 8 | MADDPG | C | 41,738 |
| 9 | LOLA | I | 38,900 |
| 10 | IndependentREINFORCE | I | 38,794 |
| 11 | FCP | I | 38,564 |
| 12 | M3DDPG | C | 37,649 |
| 13 | Random | H | 27,250 |
| 14 | MAPPO | C | 25,707 |
| 15 | IA2C | I | 23,360 |
| 16 | SelfPlay_PPO | I | 23,281 |
| 17 | IPPO | I | 23,171 |

### TR-3 — Collective Action and Loyalty

| Rank | Algorithm | Class | Return |
|-----:|-----------|:-----:|----------:|
| 1 | ISAC | I | 1,272,467 |
| 2 | MASAC | C | 1,138,192 |
| 3 | COMA | C | 785,314 |
| 4 | FCP | I | 769,129 |
| 5 | VDN | C | 697,824 |
| 6 | MeanFieldAC | C | 651,348 |
| 7 | QMIX | C | 613,577 |
| 8 | TitForTat | H | 610,835 |
| 9 | IndependentREINFORCE | I | 460,801 |
| 10 | LOLA | I | 448,761 |
| 11 | IA2C | I | 245,515 |
| 12 | MAPPO | C | 203,067 |
| 13 | Random | H | 144,218 |
| 14 | MATD3 | C | 141,518 |
| 15 | MADDPG | C | 131,427 |
| 16 | M3DDPG | C | 90,727 |
| 17 | SelfPlay_PPO | I | 38,429 |
| 18 | IPPO | I | 38,322 |

> **Note on TR-3 ranks 14-16:** MADDPG, MATD3, and M3DDPG produce 100% NaN returns on `ApacheProject-v0` under integrated reward; the ranking aggregates returns over the four TR-3 environments where these algorithms produce defined returns. See [Reward-Type Ablation](reward_type_ablation.md) for the failure pattern.

### TR-4 — Sequential Interaction and Reciprocity

| Rank | Algorithm | Class | Return |
|-----:|-----------|:-----:|-------:|
| 1 | COMA | C | 125,819 |
| 2 | ISAC | I | 122,208 |
| 3 | QMIX | C | 118,998 |
| 4 | TitForTat | H | 110,098 |
| 5 | MeanFieldAC | C | 106,745 |
| 6 | VDN | C | 105,968 |
| 7 | FCP | I | 98,201 |
| 8 | MADDPG | C | 95,433 |
| 9 | MATD3 | C | 95,232 |
| 10 | M3DDPG | C | 90,024 |
| 11 | MASAC | C | 82,011 |
| 12 | MAPPO | C | 79,485 |
| 13 | IndependentREINFORCE | I | 75,943 |
| 14 | LOLA | I | 75,878 |
| 15 | IA2C | I | 65,940 |
| 16 | Random | H | 65,103 |
| 17 | SelfPlay_PPO | I | 60,280 |
| 18 | IPPO | I | 60,221 |

---

## Algorithm consistency across tiers

Two algorithms appear in the top five of all four mechanism classes under integrated reward:

- **ISAC** — top five on TR-1, TR-2, TR-3, TR-4 (rank 4, 1, 1, 2 respectively)
- **COMA** — top five on TR-1, TR-2, TR-3, TR-4 (rank 2, 2, 3, 1 respectively)

No other algorithm achieves top-five status on every tier. The pattern suggests that ISAC and COMA are the most consistent representatives of their respective paradigm classes for benchmark comparison purposes.

> **Source for top-five claim:** Computed from the four ranking tables above. Both ISAC (max rank 4 across tiers) and COMA (max rank 3 across tiers) satisfy "top five on every tier". No other algorithm has this property under integrated reward.

---

## Game-theoretic oracle baselines

Seven oracle baselines provide game-theoretic reference points. Each oracle plays a closed-form policy derived from analysis of the underlying game, not a learned policy. Oracles are used as references against which learned algorithms are compared.

| Oracle | TR | Reference role |
|--------|----|----|
| Oracle_Equilibrium | TR-1 | Nash equilibrium for the static interdependence game |
| Oracle_TrustAware | TR-2 | Equilibrium with full trust-state observation |
| Oracle_Nash | TR-3 | Nash equilibrium for the team-production game (free-riding lower bound) |
| Oracle_Loyalty | TR-3 | Loyalty-augmented optimum (constant-action upper bound for loyalty-modified payoffs) |
| Oracle_SocialOptimum | TR-3 | Joint-welfare maximum |
| Oracle_ReciprocityEquilibrium | TR-4 | Reciprocity-aware equilibrium |
| Oracle_BoundedReciprocity | TR-4 | Bounded-rationality reciprocity reference |

Selected oracle results:

- **ISAC exceeds `Oracle_Loyalty` on all 5 TR-3 environments** under integrated reward at the n=10+ fold. Per-environment exceedance ranges +0.62% to +1.29%; mean +0.88%. Source: `aggregates/oracle_exceedance_v2.txt`. See the [Case Study Validation](case_study_validation.md) page for details and interpretation.
- **No algorithm exceeds `Oracle_BoundedReciprocity` on TR-4** environments.
- TR-1 algorithms achieve returns far above `Oracle_Equilibrium` because the equilibrium is a Nash lower bound rather than a performance ceiling.

---

## Constant-action policies

The 101 constant-action policies (`Constant_00` through `Constant_100`, in 1% increments of cooperation level) span the full cooperation continuum. They serve two purposes:

1. **Static cooperation surface.** For any environment under any reward configuration, the 101 constants generate the complete payoff curve as a function of uniform cooperation level.
2. **Best-fixed-action upper bound.** The maximum return across the 101 constants is the best return achievable by any policy that plays the same action at every step. Learned policies that exceed this bound (such as ISAC on TR-3) are doing something the constants cannot: adaptive sequencing.

Constant-action policies are deterministic given a cooperation level and produce zero standard deviation across seeds.