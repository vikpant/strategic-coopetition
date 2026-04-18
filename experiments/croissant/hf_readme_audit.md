---
license: cc-by-4.0
task_categories:
  - reinforcement-learning
tags:
  - multi-agent-reinforcement-learning
  - behavioral-audit
  - exploitation-gradient
  - alignment
  - mixed-motive
  - coopetition
pretty_name: Coopetition-Gym Behavioral Audit
size_categories:
  - 1K<n<10K
---

# Coopetition-Gym Behavioral Audit

Behavioral audit results from the NeurIPS 2026 Coopetition-Gym campaign.
1,116 JSON files across two subsets:

- **Static response-surface audit** (1,056 files): 18 policies × 20 environments
  × 3 seeds, minus 24 MeanFieldAC dyadic exclusions. Sweeps uniform
  cooperation from 0% to 100% and tests unilateral deviation at four
  cooperation levels.
- **Temporal deviation audit** (60 files): 20 environments × 3 seeds. Tests
  whether agents can accumulate cooperative capital then defect, via binary
  late-defection switchpoints and gradual ramp-down strategies.

The audit empirically bounds the exploitation gradient under integrated
reward (paper Appendix F). Key finding: binary switchpoint strategies are
universally blocked (0 exploitative outcomes across 504 tests). Gradual
ramp-down produces marginal exploitation on 6 of 20 environments
(+0.004% to +0.41% of baseline return), predominantly TR-4 reciprocity.

**Companion paper**: *Reward-Type Ablation Reveals Mechanism-Dependent
Algorithm Rankings in Mixed-Motive Multi-Agent Evaluation*, NeurIPS 2026
Evaluations and Datasets Track.

**Companion code**: https://github.com/vikpant/strategic-coopetition

**Companion training dataset**: https://huggingface.co/datasets/vikpant/coopetition-gym-v1

---

## Quick Start

```bash
pip install huggingface_hub
huggingface-cli download vikpant/coopetition-gym-audit --repo-type dataset --local-dir data/audit
tar -xzf data/audit/behavioral_audits.tar.gz -C data/audit/
```

Expected structure after extraction:

```
data/audit/
├── action_audit/              # 1,056 static response-surface files
│   ├── ISAC_TrustDilemma-v0_99_audit.json
│   ├── COMA_LoyaltyTeam-v0_100_audit.json
│   └── ...
└── temporal_audit/            # 60 temporal deviation files
    ├── TrustDilemma-v0_99_temporal.json
    ├── LoyaltyTeam-v0_100_temporal.json
    └── ...
```

## Schema

Two record types.

### Static audit (`action_audit/*_audit.json`)

| Field | Type | Description |
|---|---|---|
| `algorithm` | str | Algorithm label (one of 18 policies) |
| `environment` | str | Environment ID |
| `seed` | int | Random seed {99, 100, 101} |
| `n_agents` | int | Number of agents |
| `endowment` | float | Per-agent endowment per step |
| `response_surface` | object | Per-agent-returns at 21 cooperation levels (0–100% in 5% increments) |
| `optimal_coop_level` | float | Cooperation level maximizing mean return |
| `exploitation_analysis` | list | 4 unilateral-deviation tests (at 20%, 40%, 60%, 80% cooperation) |
| `n_exploitative` | int | Count of test levels classified exploitative |

### Temporal audit (`temporal_audit/*_temporal.json`)

| Field | Type | Description |
|---|---|---|
| `environment` | str | Environment ID |
| `seed` | int | Random seed |
| `episode_length` | int | Steps per episode |
| `baseline` | object | Full-cooperation reference |
| `full_defection` | object | Agent 0 defects throughout |
| `late_defection` | list | 9 switchpoint tests (50%–99% of episode) |
| `early_defection` | list | 3 early-defection duration tests |
| `gradual_defection` | object | Linear ramp-down over final 20% |
| `temporal_profile.vulnerability_class` | str | `immune`, `terminal_only`, `late_vulnerable`, `partially_vulnerable`, or `broadly_vulnerable` |

See the [Croissant metadata](croissant.json) for the complete machine-readable
schema.

## Key Findings (paper Appendix F)

1. **Algorithm-independent exploitation**: the static audit's exploitation
   count is identical across all 18 policies for every environment. The
   exploitation gradient is a structural property of the environment, not
   the learning paradigm.

2. **Universal binary-switchpoint immunity**: 504 temporal switchpoint tests,
   zero exploitative outcomes. Within-step state updates (trust erosion,
   loyalty degradation, reciprocity sanctions) close the first-mover
   advantage window.

3. **Marginal gradual-ramp-down exploitation**: the gradual strategy (linear
   cooperation reduction over the final 20% of an episode) evades per-step
   sanctions on 6 environments — but gains are two orders of magnitude
   smaller than the corresponding losses to partners.

4. **TR-3 universal immunity**: all five TR-3 collective action environments
   are immune to every temporal strategy. Loyalty accumulation creates
   multiplicative coupling that neutralizes deviation gains regardless of
   timing.

## Reproducibility

Regenerate the audits (runs on CPU, no GPU required):

```bash
git clone https://github.com/vikpant/strategic-coopetition.git
cd strategic-coopetition
pip install -e ./coopetition_gym

# Static audit (~1 hour on 8 CPU cores)
python -m experiments.audit static --output data/audit/action_audit/ --max-workers 8

# Temporal audit (~10 minutes on 8 CPU cores)
python -m experiments.audit temporal --output data/audit/temporal_audit/ --max-workers 8

# Cross-audit analysis
python -m experiments.audit analyze \
    --static-dir data/audit/action_audit/ \
    --temporal-dir data/audit/temporal_audit/ \
    --output data/analysis/audit_analysis.txt
```

## Validation

```bash
python -m experiments.validate audit data/audit/
```

Expected: **1,056 static files, 60 temporal files, 0 schema errors**.

## Limitations

The static audit tests fixed-action deviations from uniform cooperation
levels (cross-sectional); it does not test trained-policy dynamics directly.
The temporal audit covers binary switchpoint, early-defection, and
gradual-ramp-down strategies; a trained RL policy could in principle
discover strategies outside this set. Users interpreting the "universal
immunity to binary switchpoint strategies" result should note that coverage
is limited to the audit's strategy set.

## Citation

```bibtex
@inproceedings{pant2026rewardtype,
    title={Reward-Type Ablation Reveals Mechanism-Dependent Algorithm Rankings in Mixed-Motive Multi-Agent Evaluation},
    author={Pant, Vik and Yu, Eric},
    booktitle={NeurIPS 2026 Evaluations and Datasets Track},
    year={2026}
}
```

## License

CC-BY-4.0 (Creative Commons Attribution 4.0 International).

## Maintenance

- **Issues**: https://github.com/vikpant/strategic-coopetition/issues
- **Contact**: vik.pant@utoronto.ca
- **Changelog**: https://github.com/vikpant/strategic-coopetition/blob/master/CHANGELOG.md
