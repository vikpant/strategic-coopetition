# Experiments — Reproducibility Package

This directory contains the consolidated orchestration, evaluation, analysis,
and validation code used to produce the datasets released with the companion
research paper *Reward-Type Ablation Reveals Mechanism-Dependent Algorithm
Rankings in Mixed-Motive Multi-Agent Evaluation*.

See [../REPRODUCE.md](../REPRODUCE.md) for step-by-step instructions to reproduce
paper tables and figures.

## Module Layout

| Module | Purpose |
|---|---|
| [config.py](config.py) | Single source of truth for all defaults (seeds, algorithms, environments, safety settings). Every other module imports from here. |
| [algorithms.py](algorithms.py) | 16 training algorithms + 7 game-theoretic oracles + 2 heuristic baselines + 101 constant-action policies. |
| [campaign.py](campaign.py) | Unified orchestrator with subcommands: `baseline`, `private`, `cooperative`, `sensitivity`. Safety defaults on (checkpoints, disk monitoring, thermal monitoring, backpressure). |
| [sensitivity.py](sensitivity.py) | Network capacity sensitivity analysis. Invoked via `python -m experiments.campaign sensitivity` or directly. |
| [evaluate.py](evaluate.py) | Policy evaluation and return aggregation. Subcommands `agent`, `aggregate`. |
| [analyze.py](analyze.py) | Analysis pipeline. Subcommands: `all`, `returns-summary`, `oracle-comparison`, `tier-summary`, `masac-instability`, `training-metrics`, `learning-curves`, `plots`, `reward-ablation`. |
| [audit.py](audit.py) | Behavioral audit (Appendix F). Subcommands `static`, `temporal`, `analyze`. |
| [validate.py](validate.py) | Dataset integrity checks. Verifies file counts, schema, and expected NaN counts. |
| [monitor.py](monitor.py) | Local-friendly monitor. Subcommands `watch`, `clean-checkpoints`, `disk-status`, `health-check`. |

## Usage

From the repository root, after `pip install -e ./coopetition_gym`:

```bash
# Reproduce the baseline training campaign (3,400 GPU-hours, approximately $8,100)
python -m experiments.campaign baseline \
    --enable-checkpoints \
    --output data/training/baseline_integrated/

# Reproduce the behavioral audit (under 1 hour on 8 CPU cores)
python -m experiments.audit static --output data/audit/static/
python -m experiments.audit temporal --output data/audit/temporal/

# Regenerate paper tables and figures from released datasets
python -m experiments.analyze paradigm-boundary --input data/training/baseline_integrated/
python -m experiments.analyze oracle-exceedance  --input data/training/baseline_integrated/
python -m experiments.analyze dij-contribution  --input data/training/
python -m experiments.audit    analyze          --static-dir data/audit/static/ --temporal-dir data/audit/temporal/

# Validate a downloaded dataset before running analysis
python -m experiments.validate training data/training/
python -m experiments.validate audit    data/audit/
```

## Design Principles

1. **Single source of truth** — [config.py](config.py) defines every default. Other modules import from it rather than redefining values.
2. **Safety defaults on** — Checkpoints, disk monitoring, and progress reporting are enabled by default. Opt-out flags exist for special cases but the defaults reflect lessons learned from the original campaign.
3. **Idempotent** — Every module that writes output checks for existing files and skips completed work. Safe to re-run without losing progress.
4. **Config-driven** — Campaign types, algorithm selections, and environment lists are data (in config.py), not code. Adding a new campaign type or algorithm is a dataclass addition, not a new module.
5. **Determinism** — All randomness is seeded. Given the same seed, algorithm hyperparameters, and hardware, results are reproducible within floating-point tolerance.

## Relationship to the Original Campaign Scripts

The original training campaign was orchestrated by a collection of scripts
scattered across three directories (orchestrator.py, orchestrator_reward_ablation.py,
run_network_sensitivity.py, and per-instance shell scripts). Those scripts produced
the released datasets. This consolidated package reorganizes the same logic into
a cleaner structure suitable for public release. The numerical outputs of the
consolidated code match the original campaign outputs within floating-point
tolerance (verified by a regression smoke test).

Campaign-specific one-off scripts (per-instance launchers, patch scripts,
cloud-orchestration utilities) are not included here — they are historical
artifacts that reference specific cloud instance IDs and have no reuse value.
