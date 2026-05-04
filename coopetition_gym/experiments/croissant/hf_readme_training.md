---
license: cc-by-4.0
task_categories:
  - reinforcement-learning
tags:
  - multi-agent-reinforcement-learning
  - mixed-motive
  - coopetition
  - benchmark
  - reward-type-ablation
  - pettingzoo
  - gymnasium
pretty_name: Coopetition-Gym v1 Training Results
size_categories:
  - 10K<n<100K
---

# Coopetition-Gym v1, Training Results

Training results from the Coopetition-Gym v1 benchmark campaign.
17,930 JSON files, each recording the outcome of training one of 16
reinforcement learning algorithms, 7 game-theoretic oracles, 2 heuristic
baselines, or 101 constant-action policies on one of 20 mixed-motive
multi-agent environments under one of three reward configurations (private,
integrated, cooperative) with one of seven random seeds.

**Companion technical report**: *Coopetition-Gym v1: A Formally Grounded
Platform for Mixed-Motive Multi-Agent Reinforcement Learning under Strategic
Coopetition*. Pant and Yu, arXiv preprint (May 2026; canonical arXiv ID forthcoming).

**Companion code**: https://github.com/vikpant/strategic-coopetition

**Companion audit dataset**: https://huggingface.co/datasets/vikpant/coopetition-gym-logs

---

## Quick Start

```bash
pip install huggingface_hub
huggingface-cli download vikpant/coopetition-gym-logs \
    --repo-type dataset --local-dir data/ \
    --include "training_runs/*"
```

The training corpus is delivered as 950 JSONL shards (`training_runs_NNNN.jsonl`, ~5 MB each) under `data/training_runs/`. Each line of each shard is one training-run record (one JSON object per training experiment). Shards can be read line-by-line without an extraction step:

```python
import json
from pathlib import Path

for shard in sorted(Path("data/training_runs").glob("*.jsonl")):
    with open(shard) as fh:
        for line in fh:
            record = json.loads(line)
            # record contains: algorithm, environment, training_seed,
            # status, training_time_seconds, evaluation_time_seconds,
            # metrics, timestamp, gpu_id, tr_mode
```

## Schema

Each JSON file has the following top-level structure:

| Field | Type | Description |
|---|---|---|
| `algorithm` | str | Algorithm name (e.g., `ISAC`, `COMA`) |
| `environment` | str | Environment ID (e.g., `TrustDilemma-v0`) |
| `training_seed` | int | Seed in {99, 100, 101, 102, 103, 104, 105} |
| `status` | str | `success` for released files |
| `training_time_seconds` | float | Wall-clock training time |
| `evaluation_time_seconds` | float | Wall-clock evaluation time |
| `metrics` | object | Aggregated per-episode and per-seed metrics |
| `timestamp` | str | ISO 8601 completion timestamp |
| `gpu_id` | int | GPU index used (−1 for CPU) |
| `tr_mode` | str | TR tier label (`tr1`, `tr2`, `tr3`, `tr4`) |

The nested `metrics` object contains `mean_return`, `std_return`,
`mean_cooperation_rate`, `mean_final_trust`, `training_returns`,
`training_timesteps`, `training_metrics` (gradient-level diagnostics by step),
and TR-tier-specific `tr_metrics`.

See the [Croissant metadata](croissant.json) for the complete machine-readable
schema with JSONPath extractions.

## Known Data Characteristics

- **62 NaN-return files** from documented training instabilities are retained
  for transparency: 21 MASAC on TR-3 environments under baseline, 21 MADDPG
  /MATD3/M3DDPG on ApacheProject-v0 under cooperative reward, 20 MADDPG on
  AppleAppStore-v0 in the network sensitivity analysis.
- **MeanFieldAC** is evaluated only on environments with ≥ 3 agents
  (12 of 20 environments). The mean-field approximation is degenerate for
  two-agent settings.
- **Seeds** are arbitrary but fixed. The 7 seeds (99–105) are a design
  decision, not a sample from any population.

## Reproducibility

Reproduce the paper's tables and figures from this dataset:

```bash
git clone https://github.com/vikpant/strategic-coopetition.git
cd strategic-coopetition
pip install -e ./coopetition_gym

# Point at the extracted dataset
python -m experiments.analyze all \
    --input-dir data/training/baseline_integrated/ \
    --output-dir data/analysis/

# Reward-type ablation comparison
python -m experiments.analyze reward-ablation \
    --input-baseline    data/training/baseline_integrated/ \
    --input-private     data/training/ablation_private/ \
    --input-cooperative data/training/ablation_cooperative/ \
    --output-dir        data/analysis/reward_ablation/
```

Regenerate the dataset from scratch (3,400 GPU-hours, approximately $8,100 USD
on commodity cloud GPUs):

```bash
python -m experiments.campaign baseline --enable-checkpoints \
    --output data/training/baseline_integrated/
python -m experiments.campaign private --output data/training/ablation_private/
python -m experiments.campaign cooperative --output data/training/ablation_cooperative/
```

See [REPRODUCE.md](https://github.com/vikpant/strategic-coopetition/blob/master/REPRODUCE.md)
for full instructions.

## Validation

Check dataset integrity after download:

```bash
python -m experiments.validate training data/training/
```

Expected output: **17,930 files, 62 expected NaN entries, 0 failed experiments**.

## Limitations

This dataset should **not** be used to:

- Train policies for deployment in actual business settings without
  extensive domain-specific validation. The validated case studies are
  abstract models, not operational authorizations.
- Claim results about empirical human cooperative behavior. The dataset
  reflects agent behavior under synthetic reward structures.

See the repository's [DATASHEET.md](https://github.com/vikpant/strategic-coopetition/blob/master/DATASHEET.md)
for the complete Gebru et al. datasheet.

## Citation

```bibtex
@misc{pant2026coopetitiongym,
    title={Coopetition-Gym v1: A Formally Grounded Platform for Mixed-Motive
           Multi-Agent Reinforcement Learning under Strategic Coopetition},
    author={Pant, Vik and Yu, Eric},
    year={2026},
    publisher={arXiv},
    note={Companion technical report; arXiv ID forthcoming}
}

@software{pant2026coopetitiongym_software,
    author={Pant, Vik and Yu, Eric},
    title={Coopetition-Gym: reproducibility package for the Coopetition-Gym benchmark},
    version={1.0.0},
    year={2026},
    publisher={Zenodo},
    doi={10.5281/zenodo.20015197}
}
```

Software archival deposit: <https://doi.org/10.5281/zenodo.20015197> (concept DOI; resolves to latest version).

The benchmark environments are formalized in four foundational technical reports:
arXiv:2510.18802 (TR-1), arXiv:2510.24909 (TR-2), arXiv:2601.16237 (TR-3),
arXiv:2604.01240 (TR-4).

## License

CC-BY-4.0 (Creative Commons Attribution 4.0 International). Users may share
and adapt the dataset for any purpose, including commercial, provided
attribution is given to the original authors.

## Maintenance

- **Issues and corrections**: https://github.com/vikpant/strategic-coopetition/issues
- **Contact**: vik.pant@mail.utoronto.ca
- **Changelog**: https://github.com/vikpant/strategic-coopetition/blob/master/CHANGELOG.md

The v1 dataset is frozen for reproducibility. Future extensions will be
released as versioned successors with their own datasheets.

## Technical Reports

- TR-1: [Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity](https://arxiv.org/pdf/2510.18802) (arXiv:2510.18802)
- TR-2: [Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics](https://arxiv.org/pdf/2510.24909) (arXiv:2510.24909)
- TR-3: [Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty](https://arxiv.org/pdf/2601.16237) (arXiv:2601.16237)
- TR-4: [Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity](https://arxiv.org/pdf/2604.01240) (arXiv:2604.01240)
