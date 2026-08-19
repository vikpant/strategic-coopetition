# Reproducing Coopetition-Gym v1 Results

This is the root reproduction pointer for the repository. Each component
below carries its own detailed instructions; this page maps claims to the
artifact that reproduces them.

## Installation

```bash
pip install -e ./coopetition_gym
python -c "import coopetition_gym; print(coopetition_gym.__version__)"
```

Note: run Python from outside the repository root (or from any other
directory) after installation. The project folder is itself named
`coopetition_gym`, so an interpreter started at the repository root
resolves the folder, not the installed package.

## Package test suite

```bash
python -m pytest coopetition_gym/coopetition_gym/tests/ -q   # 143 tests
```

## Technical-report validation suites (TR-1 through TR-4)

```bash
python TR_validation/TR1_foundations/TR1_validation_suite.py --experiment tr
```

Expected for the S-LCD case study: 58/60 under the logarithmic value
specification and 46/60 under the power alternative. Suites for TR-2
through TR-4 live in the sibling `TR_validation/` folders; each scores
its case study's behavioral-correspondence rubric programmatically.

## Reference experimental study (training corpus)

The consolidated orchestration module is `coopetition_gym/experiments/`
(`campaign.py`, `algorithms.py`, `evaluate.py`, `analyze.py`,
`validate.py`, `config.py`). The released training corpus is hosted at
https://huggingface.co/datasets/vikpant/coopetition-gym-logs
(`training_runs/`: 949 JSONL shards, 27,649 rows = 27,613 result records
across 17,930 algorithm–environment–seed cells, plus 36 progress-log
rows; per-shard record counts and MD5 checksums in
`training_runs/training_runs_manifest.csv`). Baseline seeds are 99–105;
the canonical evaluation fold extends to seeds 106–108;
stability-characterization runs cover seeds 109–112.

## Behavioral audit corpus

`behavioral_audit/` in the same dataset (1,116 records: 1,056 static
sweep + 60 temporal schedules), with its own manifest.

## Two-dimensional SLCD extension

`coopetition_gym/extensions/slcd_2d/` carries its own `REPRODUCE.md`.

## Provenance

Campaign-era package code state: git tag `v1.0.0-campaign`. Initial
public release: `v1.0.0`. Later patch releases carry documentation and
packaging-metadata changes only; see `CHANGELOG.md`.
