# Benchmark Results

**Comprehensive MARL Algorithm Evaluation on Coopetition-Gym v1**

This benchmark suite evaluates how multi-agent reinforcement learning (MARL) algorithms behave on environments grounded in formal models of strategic coopetition. The headline finding is simple to state and consequential to act on: the leading algorithm is not the same algorithm in every setting. Whether centralized critics or independent learners come out ahead depends jointly on the mechanism class of the environment (interdependence, trust, collective action, reciprocity) and on the reward configuration (private, integrated, or cooperative). A single ranking table per environment, the prevailing convention in MARL benchmarking, can miss this structure or even invert it.

> Numerical claims on this page have been cross-checked against the analysis aggregates released with the package. Each table identifies its source file in the released dataset so that a reader can reproduce the figure directly.

---

## What this benchmark contains

| Component | Count |
|-----------|-------|
| Environments | 20 (5 per mechanism class) |
| Mechanism classes | 4 (TR-1, TR-2, TR-3, TR-4) |
| Reward configurations | 3 (private, integrated, cooperative) |
| Training algorithms | 16 |
| Game-theoretic oracle baselines | 7 |
| Heuristic baselines | 2 (Random, TitForTat) |
| Constant-action policies | 101 (0% through 100% in 1% increments) |
| Random seeds (baseline) | 7 (seeds 99-105) |
| Random seeds (extension) | 6 (seeds 106-111, partial coverage) |
| Reference compute | NVIDIA RTX 5090 fleet, ~$10,500 USD |

**Action space:** A single uniform scalar action per agent in `[0, endowment]`. Actions are interpreted by the environment dynamics; the `[0, endowment]` action space is standard in trust games, public-goods games, and continuous prisoner's dilemmas.

**Three programming interfaces:** Gymnasium (single-agent style with agents in the action array), PettingZoo Parallel (simultaneous moves), and PettingZoo AEC (sequential moves). All three expose the same underlying environment dynamics.

---

## Five primary findings

### 1. Paradigm crossover under reward-type ablation

The ranking of algorithms changes when the reward configuration changes. On at least one validated case study environment (`AppleAppStore-v0`), the leading paradigm class flips between independent learning (under private reward) and centralized-training-with-decentralized-execution (under integrated and cooperative reward). The crossover means a benchmark that publishes only one ranking is potentially actively misleading for a reader whose deployment setting matches a different reward configuration.

### 2. Mechanism-class split between paradigms

Under the calibrated integrated reward at the n=10+ extension fold, the four mechanism classes do not all favor the same paradigm. The split is **2-2**: centralized critics lead on TR-1 (interdependence) and TR-4 (reciprocity); independent learners lead on TR-2 (trust) and TR-3 (collective action). This is inconsistent with a "CTDE always dominates" reading of cooperative MARL benchmarks.

### 3. Implicit cooperation through structural incentives

On all five TR-3 collective action environments, the Independent Soft Actor-Critic (ISAC) algorithm exceeds the highest mean episodic return achievable by any constant-action (fixed-cooperation-level) policy. Exceedance ranges from +0.62% to +1.29% across the five environments and is positive on every seed. The mechanism is not temporal exploitation (the behavioral audit rules this out) but adaptive sequencing within the episode.

### 4. Reward-induced failure modes

Three deterministic-policy multi-agent algorithms (MADDPG, MATD3, M3DDPG) exhibit complete training divergence on the 6-agent `ApacheProject-v0` environment under integrated and cooperative reward configurations, while predominantly converging under private reward. This is not a quantitative ranking shift; it is a qualitative absence of valid output. A benchmark that evaluated these algorithms under only one reward mode would produce a partial and misleading picture of their behavior.

### 5. Interdependence coefficient contribution scales by mechanism class

The fraction of an algorithm's return that derives from the partner-payoff terms (the `D_ij` coupling) varies systematically by tier. Median contributions span from approximately 24% (TR-2 trust) to approximately 59% (TR-3 collective action). On a small minority of TR-2 algorithm-environment pairs, the contribution is negative: incorporating partner payoffs measurably worsens the learned policy on those pairs.

---

## Continue reading

| Document | Topic |
|----------|-------|
| [Algorithm Comparison](algorithm_comparison.md) | Per-tier rankings of all 16 training algorithms under all three reward configurations |
| [Environment Analysis](environment_analysis.md) | Per-environment results for the 20 environments, organized by mechanism class |
| [Case Study Validation](case_study_validation.md) | The four historically calibrated environments and their validation scores |
| [Reward-Type Ablation](reward_type_ablation.md) | The methodology, the crossover finding, and the reward-induced failure modes |

---

## How to reproduce these results

The reference experimental study is released as a public reproducibility package alongside the v1 release. Reproduction requires:

- The `coopetition_gym` package (this repository).
- The `experiments/` reproducibility folder (this repository, `experiments/`).
- The full result archive on Hugging Face: `coopetition-gym-logs` (CC-BY-4.0, Croissant 1.0 manifest).

Cells reported in this documentation are taken from the analysis aggregates released in the dataset (`aggregates/returns_summary_v2.csv`, `aggregates/tier_summary_v2.txt`, `aggregates/oracle_exceedance_v2.txt`, `aggregates/dij_contribution_summary_v2.txt`, `aggregates/all_nan_cells_v2.txt`). Each table on the supporting pages identifies the source file from which its values were computed.

---

## Citation

```bibtex
@software{coopetition_gym,
  title = {Coopetition-Gym: Environments for Mixed-Motive Multi-Agent Reinforcement Learning},
  author = {Pant, Vik and Yu, Eric},
  year = {2026},
  institution = {Faculty of Information and Department of Computer Science, University of Toronto}
}
```