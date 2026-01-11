# Benchmark Results

**Comprehensive MARL Algorithm Evaluation on Coopetition-Gym**

This section presents empirical results from evaluating 20 multi-agent reinforcement learning algorithms across all 10 Coopetition-Gym environments. These benchmarks provide evidence-based guidance for algorithm selection and validate the theoretical foundations established in TR-1 and TR-2.

---

## Benchmark Overview

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Algorithms Evaluated** | 20 |
| **Environments** | 10 |
| **Seeds per Algorithm** | 5 (learning) / 1 (heuristic) |
| **Total Experiments** | 760 |
| **Evaluation Episodes** | 100 per experiment |
| **Total Episodes** | 76,000 |
| **Hardware** | 8x NVIDIA A100-SXM4-40GB |
| **Total Compute** | 3,844 GPU-hours |

### Algorithm Categories

| Category | Algorithms | Description |
|----------|------------|-------------|
| **Heuristic** | Random, Constant_035, Constant_050, Constant_075, TitForTat | Non-learning baselines |
| **Independent** | IPPO, ISAC, IA2C | Agents learn independently |
| **CTDE-Continuous** | MAPPO, MADDPG, MATD3, MASAC | Centralized critic, continuous actions |
| **CTDE-Discrete** | QMIX, VDN, COMA | Value decomposition methods |
| **Opponent-Modeling** | LOLA, M3DDPG | Model opponent learning |
| **Population** | SelfPlay_PPO, FCP | Population-based training |
| **Mean-Field** | MeanFieldAC | Mean-field approximation |

### Environment Categories

| Category | Environments | Agents |
|----------|--------------|--------|
| **Dyadic** | TrustDilemma-v0, PartnerHoldUp-v0 | 2 |
| **Ecosystem** | PlatformEcosystem-v0, DynamicPartnerSelection-v0 | 4-5 |
| **Benchmark** | RecoveryRace-v0, SynergySearch-v0 | 2 |
| **Validated** | SLCD-v0, RenaultNissan-v0 | 2 |
| **Extended** | CooperativeNegotiation-v0, ReputationMarket-v0 | 2-4 |

---

## Key Findings

### Finding 1: Simple Heuristics Outperform Complex Learning

The most striking result is that constant-cooperation heuristics achieve the highest returns across all environments.

| Rank | Algorithm | Category | Mean Return | 95% CI |
|------|-----------|----------|-------------|--------|
| 1 | Constant_050 | Heuristic | 72,494 | +/- 11,563 |
| 2 | Constant_075 | Heuristic | 69,594 | +/- 12,213 |
| 3 | ISAC | Independent | 65,563 | +/- 5,703 |
| 4 | Constant_035 | Heuristic | 53,390 | +/- 10,632 |
| 5 | VDN | CTDE-Discrete | 51,916 | +/- 4,694 |

**Interpretation:** In coopetitive environments, predictable cooperation at moderate levels (50%) creates stable trust dynamics that learning algorithms struggle to discover. This validates TR-2's theoretical prediction that trust-building requires consistent signaling.

### Finding 2: Population-Based Methods Fail Catastrophically

Self-play and fictitious co-play converge to near-defection equilibria:

| Algorithm | Mean Return | Cooperation Rate | Final Trust |
|-----------|-------------|------------------|-------------|
| SelfPlay_PPO | 21,136 | 11.8% | 6.8% |
| FCP | 21,019 | 11.7% | 6.8% |
| IPPO | 20,981 | 11.6% | 6.5% |

**Interpretation:** Population-based methods optimize for Nash equilibria, which in coopetitive games are often Pareto-suboptimal. Without mechanisms to escape defection spirals, these methods destroy trust and forfeit cooperative surplus.

### Finding 3: Trust Predicts Long-Term Performance

Statistical analysis reveals strong correlations:

| Correlation | Pearson r | Interpretation |
|-------------|-----------|----------------|
| Trust-Return | 0.552 | Moderate positive |
| Cooperation-Return | 0.522 | Moderate positive |
| Trust-Cooperation | 0.894 | Strong positive |

**Interpretation:** Trust is not merely a side-effect but a causal driver of performance. Algorithms that maintain trust achieve significantly higher returns (Cohen's d > 1.0 for trust-builders vs. trust-eroders).

### Finding 4: CTDE Methods Cluster Together

All CTDE-Continuous methods achieve nearly identical results:

| Algorithm | Mean Return | Cooperation | Trust |
|-----------|-------------|-------------|-------|
| M3DDPG | 50,916 | 75.8% | 73.5% |
| MATD3 | 50,913 | 75.8% | 73.5% |
| MASAC | 50,908 | 75.8% | 73.5% |
| MADDPG | 50,903 | 75.8% | 73.5% |

**Interpretation:** The centralized critic dominates actor architecture differences. In coopetitive settings, access to joint observations enables coordination regardless of policy gradient variant.

---

## Algorithm Rankings

### Overall Performance (All Environments)

| Rank | Algorithm | Mean Return | Std Dev | Cooperation | Trust |
|------|-----------|-------------|---------|-------------|-------|
| 1 | Constant_050 | 72,494 | 41,726 | 55.0% | 98.7% |
| 2 | Constant_075 | 69,594 | 44,054 | 78.7% | 98.7% |
| 3 | ISAC | 65,563 | 35,634 | 80.3% | 98.6% |
| 4 | Constant_035 | 53,390 | 38,359 | 38.5% | 55.5% |
| 5 | VDN | 51,916 | 29,344 | 49.6% | 61.9% |
| 6 | M3DDPG | 50,916 | 31,504 | 75.8% | 73.5% |
| 7 | MATD3 | 50,913 | 31,505 | 75.8% | 73.5% |
| 8 | MASAC | 50,908 | 31,507 | 75.8% | 73.5% |
| 9 | MADDPG | 50,903 | 31,510 | 75.8% | 73.5% |
| 10 | MeanFieldAC | 48,368 | 45,155 | 46.8% | 33.9% |
| 11 | COMA | 44,342 | 30,248 | 54.6% | 58.8% |
| 12 | QMIX | 44,340 | 30,249 | 54.6% | 58.8% |
| 13 | TitForTat | 38,556 | 22,317 | 35.2% | 48.9% |
| 14 | LOLA | 37,896 | 27,479 | 51.3% | 50.3% |
| 15 | MAPPO | 35,423 | 30,480 | 36.0% | 27.1% |
| 16 | IA2C | 28,089 | 25,335 | 19.6% | 15.3% |
| 17 | Random | 24,939 | 17,311 | 47.7% | 7.1% |
| 18 | SelfPlay_PPO | 21,136 | 13,559 | 11.8% | 6.8% |
| 19 | FCP | 21,019 | 13,475 | 11.7% | 6.8% |
| 20 | IPPO | 20,981 | 13,449 | 11.6% | 6.5% |

### Category Performance

| Category | N | Mean Return | 95% CI | Cooperation | Trust | Training (hrs) |
|----------|---|-------------|--------|-------------|-------|----------------|
| Heuristic | 50 | 51,795 | +/- 10,477 | 51.0% | 61.8% | 0.0 |
| Mean-Field | 10 | 48,368 | +/- 27,987 | 46.8% | 33.9% | 4.4 |
| CTDE-Continuous | 200 | 47,037 | +/- 4,399 | 65.8% | 61.9% | 11.2 |
| CTDE-Discrete | 150 | 46,866 | +/- 4,795 | 53.0% | 59.8% | 3.0 |
| Opponent-Modeling | 100 | 44,406 | +/- 5,905 | 63.5% | 61.9% | 6.5 |
| Independent | 150 | 38,211 | +/- 5,243 | 37.2% | 40.1% | 1.8 |
| Population | 100 | 21,077 | +/- 2,636 | 11.8% | 6.8% | 2.0 |

---

## Environment-Specific Results

### Best Algorithm per Environment

| Environment | Best Overall | Return | Best Learning | Return |
|-------------|--------------|--------|---------------|--------|
| PlatformEcosystem-v0 | Constant_035 | 146,967 | VDN | 123,704 |
| ReputationMarket-v0 | Constant_075 | 163,761 | ISAC | 144,730 |
| DynamicPartnerSelection-v0 | Constant_050 | 107,314 | ISAC | 81,620 |
| RecoveryRace-v0 | Constant_050 | 59,508 | ISAC | 57,348 |
| SLCD-v0 | Constant_035 | 50,946 | ISAC | 45,555 |
| RenaultNissan-v0 | Constant_035 | 46,406 | ISAC | 43,033 |
| TrustDilemma-v0 | Constant_050 | 46,267 | ISAC | 44,660 |
| PartnerHoldUp-v0 | Constant_050 | 45,320 | ISAC | 43,237 |
| CooperativeNegotiation-v0 | Constant_050 | 43,948 | ISAC | 38,321 |
| SynergySearch-v0 | Constant_050 | 41,994 | ISAC | 40,439 |

**Key Observation:** ISAC (Independent Soft Actor-Critic) is the best learning algorithm across 9 of 10 environments, with VDN winning only on PlatformEcosystem-v0.

---

## Head-to-Head Comparison

Win rates showing percentage of environments where row algorithm beats column algorithm:

| | Constant_050 | ISAC | MADDPG | MAPPO | IPPO | Random |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Constant_050** | - | 100% | 100% | 100% | 100% | 100% |
| **ISAC** | 0% | - | 100% | 100% | 100% | 100% |
| **MADDPG** | 0% | 0% | - | 80% | 90% | 90% |
| **MAPPO** | 0% | 0% | 20% | - | 90% | 50% |
| **IPPO** | 0% | 0% | 10% | 10% | - | 10% |
| **Random** | 0% | 0% | 10% | 50% | 90% | - |

**Interpretation:** Clear dominance hierarchy exists: Heuristics > ISAC > CTDE > Independent > Population. Random beats IPPO in 90% of environments, demonstrating that naive independent learning is counterproductive in coopetitive settings.

---

## Statistical Significance

### Effect Sizes (Cohen's d)

| Comparison | Cohen's d | Effect Size |
|------------|-----------|-------------|
| Constant_050 vs IPPO | +2.50 | Large |
| ISAC vs IPPO | +1.66 | Large |
| Constant_050 vs Random | +1.49 | Large |
| MADDPG vs IPPO | +1.24 | Large |
| Constant_050 vs MAPPO | +1.14 | Large |

All key comparisons show large effect sizes (d > 0.8), indicating robust and practically significant differences.

---

## Reproducibility

### Seed Variance Analysis

| Algorithm | Mean CV% | Stability |
|-----------|----------|-----------|
| FCP | 1.4% | High |
| IPPO | 1.5% | High |
| SelfPlay_PPO | 1.5% | High |
| ISAC | 2.9% | High |
| VDN | 19.6% | Low |
| MADDPG | 22.7% | Low |
| MAPPO | 30.5% | Low |
| LOLA | 41.6% | Low |
| QMIX | 41.9% | Low |

**Note:** High stability for population methods reflects convergence to the same (suboptimal) defection equilibrium. Low stability for CTDE methods indicates sensitivity to initialization, suggesting the need for multiple seeds in research.

---

## Recommendations

### For Practitioners

1. **Start with Constant_050**: Before investing in complex learning, establish a baseline with 50% constant cooperation.

2. **Use ISAC for learning**: If learning is required, Independent SAC achieves the best balance of performance and training efficiency.

3. **Avoid population methods**: Self-play and FCP consistently underperform in coopetitive settings.

4. **Monitor trust dynamics**: Track trust as a leading indicator of long-term performance.

### For Researchers

1. **Report multiple seeds**: CTDE methods show high variance; 5+ seeds are recommended.

2. **Include heuristic baselines**: Many learning papers omit simple baselines that may outperform.

3. **Analyze trust trajectories**: Raw returns obscure the mechanism of success/failure.

4. **Consider training cost**: MAPPO requires 20+ hours per run vs. 3 hours for ISAC.

---

## Navigation

- [Algorithm Comparison](algorithm_comparison.md) - Detailed algorithm analysis
- [Trust Dynamics](trust_dynamics.md) - Empirical validation of TR-2
- [Environment Analysis](environment_analysis.md) - Per-environment deep dives
- [Research Insights](research_insights.md) - Theoretical implications

---

## Citation

If you use these benchmark results, please cite:

```bibtex
@software{coopetition_gym_benchmarks,
  title = {Coopetition-Gym Benchmark Results: 20 MARL Algorithms on 10 Environments},
  author = {Pant, Vik and Yu, Eric},
  year = {2026},
  institution = {Faculty of Information, University of Toronto},
  note = {760 experiments, 76,000 evaluation episodes, 3,844 GPU-hours}
}
```
