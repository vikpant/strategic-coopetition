# Reward-Type Ablation

Coopetition-Gym v1 separates **payoff** (the game-theoretic outcome of agents' joint actions) from **reward** (the scalar signal an algorithm trains on). The same payoff structure can be exposed to a learning algorithm through three reward configurations:

| Mode | Reward an agent receives | Interpretation |
|------|--------------------------|----------------|
| `private` | The agent's own payoff only (`D_ij = 0`) | No incorporation of partner outcomes |
| `integrated` | Own payoff plus `D_ij`-weighted partner payoffs | Calibrated coupling from the case study or synthetic design |
| `cooperative` | Joint payoff shared equally | Full reward mutuality |

Switching modes does not change the environment's dynamics, action space, or observation space; it changes only the scalar reward an algorithm sees. Running the same algorithm under all three modes, with everything else held constant, surfaces structure that single-mode benchmark evaluation cannot.

> **Source:** `aggregates/tier_summary_v2.txt`, `aggregates/all_nan_cells_v2.txt`, `aggregates/dij_contribution_summary_v2.txt`, and `aggregates/oracle_exceedance_v2.txt` in the released analysis pipeline.

---

## Why it matters

A benchmark that reports a single ranking per environment implicitly assumes that the reward-mutuality choice is fixed. A reader applying the benchmark to a deployment setting whose mutuality structure differs from the calibrated one is using the published ranking outside its support. Reward-type ablation is the corrective: by reporting how rankings change as reward mutuality varies, a benchmark gives a reader a more complete picture of how an algorithm's standing depends on the conditions of evaluation.

---

## The crossover finding

On the `AppleAppStore-v0` case study environment (TR-4 reciprocity, 87.3% case validation), the leader changes between paradigm classes as reward mode varies:

| Reward configuration | Best CTDE | Best IND | Gap (IND-CTDE)/CTDE | Leader |
|----------------------|----------:|---------:|--------------------:|--------|
| Private (`D_ij = 0`) | 23,191 (COMA) | 24,854 (ISAC) | +7.2% | Independent |
| Integrated | 40,670 (COMA) | 39,953 (ISAC) | -1.8% | CTDE |
| Cooperative | 40,673 (COMA) | 38,979 (ISAC) | -4.2% | CTDE |

The sign change between private and integrated modes isolates the crossover to the introduction of partner-payoff coupling. If reward magnitude or reward variance were driving the ranking shift, we would expect a monotone change across the three modes; the observed pattern is consistent with the interpretation that the `D_ij` coupling itself is the structural feature selecting between paradigms — when present, centralized critics are advantaged; when absent, independent learners are advantaged.

> **Source:** `aggregates/returns_summary_v2.csv`, rows for AppleAppStore-v0 across the three reward modes. Best-CTDE and best-IND identified within each mode by maximum mean return.

The crossover is not unique to AppleAppStore-v0. Aggregating across the full benchmark, paradigm-class leaders change in either direction between private and integrated modes on more than half of the twenty environments, and the `D_ij`-introduces-CTDE-advantage pattern observed on AppleAppStore-v0 is the modal direction across those flips.

---

## The mechanism-class split

Under integrated reward at the n=10+ fold, the four mechanism classes do not all favor the same paradigm. The split is **2-2**:

- **TR-1 (interdependence) and TR-4 (reciprocity)** favor centralized training (CTDE).
- **TR-2 (trust) and TR-3 (collective action)** favor independent learning.

| Tier | Best CTDE return | Best IND return | Winner |
|------|----------------:|----------------:|--------|
| TR-1 | 69,630 (QMIX) | 65,551 (ISAC) | CTDE |
| TR-2 | 60,792 (COMA) | 65,368 (ISAC) | IND |
| TR-3 | 1,138,192 (MASAC) | 1,272,467 (ISAC) | IND |
| TR-4 | 125,819 (COMA) | 122,208 (ISAC) | CTDE |

> **Source:** `aggregates/ctde_vs_ind_boundary_v2.txt`, integrated-reward rows. Per-tier returns are means across constituent environments at the n=10+ fold.

The pattern suggests that centralized critics are disadvantaged on mechanism classes with **action-mutable relational state** — that is, mechanisms where agents themselves modify the relational state through their actions (trust-building under TR-2, loyalty accumulation under TR-3). On mechanism classes with static relational state (TR-1's interdependence matrix) or history-encoded relational state (TR-4's bounded reciprocity memory), centralized critics retain their conventional advantage.

---

## The interdependence-coefficient contribution

For each algorithm-environment pair with defined returns under both integrated and private reward, one can compute the fraction of the algorithm's return attributable to the `D_ij` coupling:

```
Contrib(D_ij) = (R_integrated - R_private) / R_integrated
```

The contribution is neither uniform nor monotonic across the benchmark. Per-tier summary statistics:

| Tier | Median | IQR | 5th-95th percentile | Negative pairs |
|------|------:|----|---|---:|
| TR-1 | 30.10% | 16.47% to 42.98% | 0.67% to 60.88% | 0 of 77 |
| TR-2 | 23.51% | 6.33% to 34.21% | -1.63% to 52.20% | 5 of 75 |
| TR-3 | 59.23% | 48.93% to 69.49% | 1.82% to 94.18% | 0 of 77 |
| TR-4 | 28.58% | 14.27% to 41.84% | 0.32% to 51.80% | 0 of 78 |

> **Source:** `aggregates/dij_contribution_summary_v2.txt`. The five negative pairs on TR-2 indicate algorithm-environment pairs for which incorporating partner payoffs measurably worsens the learned policy. The most extreme negative is MAPPO on `TrustDilemma-v0` at -605.45% (verified in `aggregates/dij_contribution_v2.csv`).

**Reading the table.** TR-3 (collective action) is by far the most `D_ij`-dependent tier — algorithms typically derive a majority of their return from partner-payoff coupling on these environments. TR-2 (trust) is the least dependent and the only tier where some pairs see negative contribution, meaning that incorporating partner payoffs actively harms training rather than helping. Reward mutuality therefore does not uniformly improve performance: whether it helps is mechanism-class-dependent.

---

## Reward-induced failure modes

Reward-type ablation surfaces qualitatively distinct failure modes — the same algorithm on the same environment fails differently depending on reward configuration. The clearest example is the deterministic-policy multi-agent algorithm family (MADDPG, MATD3, M3DDPG) on the 6-agent `ApacheProject-v0` environment.

### DDPG-family training divergence on `ApacheProject-v0`

At the n=13 fold (baseline seeds 99-105 plus extension seeds 106-111), per (algorithm, reward mode):

| Algorithm | Private | Integrated | Cooperative |
|-----------|---------|------------|-------------|
| MADDPG | 13/13 converge | 13/13 NaN | 13/13 NaN |
| MATD3 | 12/13 converge (seed 106 NaN) | 13/13 NaN | 13/13 NaN |
| M3DDPG | 10/13 converge (seeds 106-108 NaN) | 13/13 NaN | 13/13 NaN |

> **Source:** `aggregates/all_nan_cells_v2.txt`, rows tagged `ApacheProject` for MADDPG, MATD3, and M3DDPG.

A benchmark that evaluated these algorithms only under integrated reward would report them as failed implementations on `ApacheProject-v0`. A benchmark that evaluated them only under private reward would report them as predominantly stable. The reward-type ablation reveals that both single-mode conclusions are incomplete: the failure is reward-mode-conditional.

The clean stochastic-vs-deterministic and independent-vs-centralized contrasts further scope the failure: MASAC (same environment, stochastic policy with entropy regularization) converges under all three reward modes, and LOLA (same environment, meta-gradient independent learner) also converges. The failure is therefore specific to deterministic-policy multi-agent gradient methods on this particular reward scale, not a uniform algorithm-family or environment-level property.

### Sporadic NaN cells on extension seeds

Beyond the focal pattern on `ApacheProject-v0`, M3DDPG produces sporadic NaN on extension seeds 106-108 across five additional environments — `ReputationMarket-v0` (TR-2) and the four reciprocity environments `AppleAppStore-v0`, `GiftExchange-v0`, `GraduatedSanction-v0`, and `ReciprocalDilemma-v0` (all TR-4). Together with three MADDPG cells on `LoyaltyTeam-v0` (seeds 101-103), the aggregate beyond-Apache NaN count is documented in the released `aggregates/all_nan_cells_v2.txt` file.

In ranking aggregations on this site, NaN cells are excluded under a pre-registered censoring rule and rankings are computed over cells with defined returns.

### MASAC training-time instability on TR-3

Independently of NaN-divergence, MASAC exhibits training-time instability (large peak-to-median ratios in the training-return time series) on a non-trivial fraction of cells across the benchmark. At the criterion `peak/median > 5×` applied to `metrics.training_returns` time series:

- **41 of 600 MASAC cells (6.8%)** exhibit instability across the full benchmark.
- Most affected environments concentrate on `PartnerHoldUp-v0` (all reward modes) and `PlatformEcosystem-v0`.
- All 41 unstable cells produce defined final returns; none diverge to NaN.

> **Source:** `aggregates/masac_instability_v2_TIMESERIES.txt`. The criterion `peak / median(|training_returns|) > 5` flags cells whose training trajectory exhibits return excursions much larger than the typical scale.

The instability is consequential because a learning curve with large excursions can produce a final converged return that is misleadingly close to the median, masking that the trajectory passed through high-return states the policy did not consolidate. MASAC's per-step return variance on these cells is a methodological caution rather than a failure mode.

---

## What reward-type ablation contributes methodologically

A single-mode benchmark publication leaves three structural questions unanswered:

1. **Does the leader change with reward mutuality?** If so, the published ranking is conditional on a calibration choice the reader may not share.
2. **Is the algorithm's return derived from partner-payoff coupling, or from solitary learning that ignores the coupling?** The `D_ij` contribution analysis distinguishes these.
3. **Does the algorithm fail under conditions different from the published evaluation mode?** The reward-induced failure modes show that some algorithms produce valid output under one mode and zero valid output under another.

Reward-type ablation answers each question by holding the environment fixed and varying only the reward signal. The methodology is not redundant with single-mode evaluation; it surfaces structure that single-mode evaluation cannot.