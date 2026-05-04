# Reproducing Paper Results

This document describes how to reproduce the empirical results reported in:

> Pant, V. and Yu, E. (2026). *Reward-Type Ablation Reveals Mechanism-Dependent Algorithm Rankings in Mixed-Motive Multi-Agent Evaluation.* Manuscript in preparation.

The paper presents two complementary empirical artifacts: 1. A **training dataset** of 17,930 experiment result files across 128 algorithms × 20 environments × 3 reward types × 7 seeds.
2. A **behavioral audit** of 1,116 experiment result files characterizing the exploitation gradient under integrated reward.

Both artifacts are released as versioned datasets on HuggingFace Hub. This document describes how to regenerate either artifact from source and how to map each paper table and figure to its producing script.

---

## 1. Environment Setup

```bash
git clone https://github.com/vikpant/strategic-coopetition.git
cd strategic-coopetition
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl,viz]"
pytest coopetition_gym/tests/ -v
```

All 143 tests should pass. If any fail, do not proceed, file an issue with the test failure output.

## 2. Dataset Access

### 2.1 Training Dataset (17,930 files)

```bash
# HuggingFace CLI: full repo (training_runs/ + behavioral_audit/ + lr_ablation/ +
# case_study_calibration/ + tier_1_5_2d_slcd/) is at vikpant/coopetition-gym-logs.
pip install huggingface_hub
huggingface-cli download vikpant/coopetition-gym-logs --repo-type dataset --local-dir data/
```

Expected folder structure after extraction:

```
data/training/
├── baseline_integrated/       # 16,835 files — main results (integrated reward)
├── ablation_private/          # 2,450 files — private reward ablation
├── ablation_cooperative/      # 2,450 files — cooperative reward ablation
├── case_study/                # 3,402 files — validated case study calibrations
├── france_bonus_isac_integrated/  # 21 files
├── local_bonus/               # 70 files
└── network_sensitivity/       # 480 files — capacity-sensitivity analysis
```

### 2.2 Behavioral Audit Dataset (1,116 files)

The behavioral audit lives in the same repo under `behavioral_audit/`. After Section 2.1 it is already present at `data/behavioral_audit/`. To pull only the behavioral-audit subset:

```bash
huggingface-cli download vikpant/coopetition-gym-logs --repo-type dataset --local-dir data/ \
    --include 'behavioral_audit/*'
```

Expected folder structure:

```
data/audit/
├── action_audit/              # 1,056 files — static response-surface audit
└── temporal_audit/            # 60 files — temporal deviation audit
```

## 3. Paper Table and Figure Mapping

| Paper Artifact | Producing Script | Input Data | Expected Runtime |
|---|---|---|---|
| Table 1 (Paradigm boundary) | `scripts/analyze_paradigm_boundary.py` | `data/training/baseline_integrated/` | ~5 minutes |
| Table 2 (ISAC oracle exceedance) | `scripts/analyze_oracle_exceedance.py` | `data/training/baseline_integrated/` | ~2 minutes |
| Table 3 (D_ij contribution) | `scripts/analyze_dij_contribution.py` | `data/training/baseline_integrated/` + `ablation_private/` + `ablation_cooperative/` | ~10 minutes |
| Figure 1 (Crossover on AppleAppStore-v0) | `scripts/plot_crossover.py` | All three reward ablation folders | ~1 minute |
| Appendix A (Full rankings by TR tier) | `scripts/analyze_rankings.py` | `data/training/baseline_integrated/` | ~3 minutes |
| Appendix B (Network sensitivity) | `scripts/analyze_sensitivity.py` | `data/training/network_sensitivity/` | ~2 minutes |
| Appendix C (Case study calibration) | `scripts/analyze_case_studies.py` | `data/training/case_study/` | ~3 minutes |
| Appendix F (Exploitation audit) | `scripts/analyze_audits.py` | `data/audit/` | ~1 minute |

All scripts write results to `results/` with the same filename structure used in the paper source.

## 4. Regenerating the Training Dataset from Scratch

> **Warning**: Full regeneration requires approximately 3,400 GPU-hours across 16 training algorithms × 20 environments × 3 reward conditions × 7+ seeds (plus 135-cell controlled critic-lr ablation on ApacheProject-v0). The reference evaluation cost approximately $10,500 USD on commodity cloud NVIDIA RTX 5090 GPUs. Rates vary by provider; budget accordingly.

### 4.1 Single-experiment launch

```bash
python scripts/run_experiment.py \
    --algorithm ISAC \
    --environment TrustDilemma-v0 \
    --reward-type integrated \
    --seed 99 \
    --timesteps 500000 \
    --output data/training/baseline_integrated/
```

### 4.2 Full evaluation orchestration

```bash
python scripts/orchestrator.py \
    --algorithms ISAC,COMA,MASAC,QMIX,VDN,MADDPG,MATD3,M3DDPG,MeanFieldAC,FCP,LOLA,IndependentREINFORCE,IPPO,IA2C,MAPPO,SelfPlay_PPO,Random,TitForTat \
    --environments ApacheProject-v0,CoalitionFormation-v0,LoyaltyTeam-v0,PublicGoods-v0,TeamProduction-v0,SLCD-v0,TrustDilemma-v0,PartnerHoldUp-v0,PlatformEcosystem-v0,DynamicPartnerSelection-v0,CooperativeNegotiation-v0,RecoveryRace-v0,ReputationMarket-v0,RenaultNissan-v0,SynergySearch-v0,ReciprocalDilemma-v0,GiftExchange-v0,IndirectReciprocity-v0,GraduatedSanction-v0,AppleAppStore-v0 \
    --reward-types private,integrated,cooperative \
    --seeds 99,100,101,102,103,104,105 \
    --max-gpu-workers 24 \
    --enable-checkpoints \
    --output data/training/
```

The `--enable-checkpoints` flag is mandatory; prior runs without checkpointing lost 30–40 GPU-hours when instances crashed without checkpointing.

## 5. Regenerating the Behavioral Audit

The audit requires only CPU and completes in approximately 10 minutes with 8 workers.

### 5.1 Static response-surface audit (1,056 experiments)

```bash
python scripts/eval_action_audit.py \
    --max-workers 8 \
    --eval-episodes 100 \
    --seeds 99,100,101 \
    --output data/audit/action_audit/
```

Produces 1,056 JSON files. Expected runtime: approximately 1 hour with 8 workers.

### 5.2 Temporal deviation audit (60 experiments)

```bash
python scripts/eval_temporal_audit.py \
    --max-workers 8 \
    --seeds 99,100,101 \
    --output data/audit/temporal_audit/
```

Produces 60 JSON files. Expected runtime: approximately 10 minutes with 8 workers.

### 5.3 Audit analysis

```bash
python scripts/analyze_audits.py \
    --static-dir data/audit/action_audit/ \
    --temporal-dir data/audit/temporal_audit/ \
    --output results/audit_analysis.txt
```

## 6. Specifications

### 6.1 Seeds

The seven seeds used in the reference evaluation are: **99, 100, 101, 102, 103, 104, 105**.

The behavioral audit uses a subset: **99, 100, 101**.

Seeds are passed to `numpy.random.default_rng()`, `torch.manual_seed()`, and the environment's `reset(seed=...)` call. All randomness in the evaluation is seeded; results are deterministic given seed, algorithm hyperparameters, and hardware.

### 6.2 Algorithms

Sixteen training algorithms:

- **CTDE (centralized training, decentralized execution)**, 9: MADDPG, MATD3, M3DDPG, MASAC, QMIX, VDN, COMA, MAPPO, MeanFieldAC
- **Independent learning**, 7: ISAC, IPPO, IA2C, FCP, SelfPlay_PPO, LOLA, IndependentREINFORCE

Two heuristic baselines (no training): Random, TitForTat

Seven game-theoretic oracles:

- `Oracle_Equilibrium`, TR-1 interdependence equilibrium (Nash reference)
- `Oracle_TrustAware`, TR-2 trust-aware equilibrium
- `Oracle_Nash`, TR-3 Nash equilibrium (lower bound)
- `Oracle_Loyalty`, TR-3 social optimum (upper bound)
- `Oracle_SocialOptimum`, TR-3 social optimum (equivalent to Oracle_Loyalty)
- `Oracle_ReciprocityEquilibrium`, TR-4 Nash-style equilibrium (lower bound)
- `Oracle_BoundedReciprocity`, TR-4 cooperation upper bound

One hundred and one constant-action policies (cooperation fractions from 0 to 1 in 0.01 increments) complete the 128-algorithm benchmark set.

### 6.3 Hardware

The reference evaluation used cloud NVIDIA RTX 5090 GPU instances for all policy training and tuning. All training results are hardware-invariant within floating-point tolerance because seeds are propagated through PyTorch's deterministic mode. No GPU is required for the behavioral audit.

## 7. Known Deviations

- **MeanFieldAC** is evaluated only on environments with N ≥ 3 agents (12 environments). The mean-field approximation degenerates for N = 2. This is documented in the paper's experimental setup section.
- **62 files contain NaN returns** from documented training instability: 21 MASAC on TR-3 under baseline, 21 MADDPG/MATD3/M3DDPG on ApacheProject-v0 under cooperative reward, 20 MADDPG on AppleAppStore-v0 in network sensitivity. Analysis scripts exclude these; the exclusion does not affect any paper finding.

## 8. Questions and Issues

Open an issue at https://github.com/vikpant/strategic-coopetition/issues. Include:

- The script you ran
- The full command line
- The Python version and platform
- The full error traceback
- The relevant lines from the script's log output

## Technical Reports

- TR-1: [Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity](https://arxiv.org/pdf/2510.18802) (arXiv:2510.18802)
- TR-2: [Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics](https://arxiv.org/pdf/2510.24909) (arXiv:2510.24909)
- TR-3: [Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty](https://arxiv.org/pdf/2601.16237) (arXiv:2601.16237)
- TR-4: [Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity](https://arxiv.org/pdf/2604.01240) (arXiv:2604.01240)
