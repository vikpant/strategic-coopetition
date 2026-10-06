# Reproducing and checking experiment artifacts

The working tree is a **1.0.8 candidate**; the previous published tag is
`v1.0.7`. This candidate changes environment configuration, reward propagation,
experiment completion checks, and analysis. New runs belong to a new campaign.
They must not be merged into historical results as if these were documentation
changes only.

## Install the current source

Use Python 3.10–3.12 when also installing the two-dimensional SLCD extension.
From a fresh checkout:

```bash
git clone https://github.com/vikpant/strategic-coopetition.git
cd strategic-coopetition
python -m venv .venv
source .venv/bin/activate
python -m pip install './coopetition_gym[dev,experiments]'
python -m pip install ./coopetition_gym/extensions/slcd_2d
python -c "import coopetition_gym as cg; print(cg.__version__, len(cg.list_environments()))"
```

The installed experiment namespace is `experiments`, alongside
`coopetition_gym`. The commands below run from the repository root after this
installation. Installing the package normally also avoids relying on the
checkout's outer `coopetition_gym/` directory as an importable package. If you
change source afterward, reinstall before checking installed commands.

## Run a small real-environment check

```bash
python -m experiments.smoke --output data/smoke
```

Use a new, empty output directory. This runs the `Random` policy for two seeds
under each of three reward treatments on a short TrustDilemma episode. It does
**no training**. For each treatment it writes `raw/results.jsonl`, validates two
native result records, and writes `returns.csv`. These six runs check the
installation and result pipeline; they do not reproduce benchmark scores.

The individual pipeline commands are:

```bash
python -m experiments.validate training data/smoke/integrated/raw --expected-records 2
python -m experiments.analyze returns-summary \
    --input-dir data/smoke/integrated/raw \
    --output data/smoke/integrated/returns.csv
python -m experiments.analyze all \
    --input-dir data/smoke/integrated/raw \
    --output-dir data/smoke/integrated/analysis
python -m experiments.analyze reward-ablation \
    --input-baseline data/smoke/integrated/raw \
    --input-private data/smoke/private/raw \
    --input-cooperative data/smoke/cooperative/raw \
    --output-dir data/smoke/comparison
```

Summary CSVs record the actual eligible seed list, treatment, configuration,
and `comparison_basis=historical-roster`. Reward-ablation differences are emitted
only when all arms share the same observed seeds and recorded scientific
configuration; otherwise the CSV marks them incomparable and leaves differences
blank. Matching partial legacy metadata is not proof of a controlled design.
Stable algorithm names identify
implementations; they do not establish a controlled independent-learning
versus CTDE comparison. See the current
[implementation protocol](coopetition_gym/docs/benchmarks/implementation_protocol.md)
and [score provenance](coopetition_gym/docs/benchmarks/score_provenance.md).

## Tests

```bash
python -m pytest coopetition_gym/coopetition_gym/tests/ \
    coopetition_gym/experiments/tests/ \
    coopetition_gym/extensions/slcd_2d/tests/ -k "not test_ippo_trains_on_2d"
```

Tests include synthetic JSON/JSONL fixtures and the small real-environment
smoke check. The filter excludes the existing extension IPPO training test.
These commands do not launch a training campaign or regenerate technical
report artifacts. The extension source and its tests are under
`coopetition_gym/extensions/slcd_2d/`; its installed import name is `slcd_2d`.

## Analyze historical datasets with explicit provenance

The released corpus is linked from
[Hugging Face](https://huggingface.co/datasets/vikpant/coopetition-gym-logs).
Use a recorded dataset revision and the manifest for that revision. Counts of
shards, physical rows, result records, and unique experimental cells are
separate quantities; the validator does not assume a historical total or a
fixed expected NaN count.

Readers accept recursive `.json` and `.jsonl` input. Pass a campaign's `raw/`
result directory, not its container with manifests, logs, or checkpoints.
Sensitivity results are under `OUTPUT/REWARD_TYPE/raw/`; select a homogeneous
configuration before a comparison. Native results contain
`algorithm`, `environment`, `training_seed`, `status`, and nested
`metrics.mean_return`. Identifiable progress rows are counted separately.
Validation reports failed/nonfinite outcomes and conflicting or repeated cells.
Numerical analysis excludes failed/nonfinite outcomes and rejects ambiguous
comparisons across treatments, configurations, campaigns, or repeated seeds.

For a revision whose manifest uses supported path, count, and MD5/SHA256
columns, an optional check is:

```bash
python -m experiments.validate training data/training_runs \
    --manifest data/training_runs/training_runs_manifest.csv
```

Manifest paths are relative to the supplied dataset directory; its row counts
refer to physical JSON objects, including progress rows. A successful manifest
check establishes file integrity, not treatment attribution or paper provenance.
Use `python -m experiments.validate training --help` for optional declared
result counts and explicit legacy treatment context.

Historical notes identify seeds 99–108 as a canonical evaluation fold and
109–112 as stability runs. That is a revision-specific convention, not an
automatic loader rule. Select a fold only after checking its manifest and
experimental design, for example:

```bash
python -m experiments.analyze returns-summary \
    --input-dir data/selected_results \
    --reward-type integrated --seeds 99,100,101,102,103,104,105,106,107,108 \
    --output data/analysis/returns.csv
```

`--reward-type` selects labelled records and explicitly supplies the treatment
for legacy rows that lack it. Establish that assignment from source evidence
first; a directory name does not establish the treatment. The assignment is
reported in output provenance. The named inputs to `reward-ablation` likewise
supply explicit treatment context for missing legacy metadata. There is no
implicit rerun selection, historical deduplication, or configuration averaging.

The corrected pipeline does **not** certify that all historical paper tables
can be regenerated from the public corpus. Exact campaign implementations,
treatments, folds, exclusions, and artifact revisions must be reconciled first.
The original campaign tag `v1.0.0-campaign` and later release tags remain
historical provenance references, not interchangeable executions.

## Other research artifacts

- Technical-report suites and saved outputs: [`TR_validation/`](TR_validation/).
  They remain historical artifacts; the current experiment fixes do not
  reconcile conflicting case-study scores or authorize overwriting saved outputs.
- Paper-specific support and limitations:
  [`papers/neurips_ed_2026/REPRODUCE.md`](papers/neurips_ed_2026/REPRODUCE.md).
- New campaign options: `python -m experiments.campaign --help`.
  Training resources must be sized for the actual machine and declared design;
  the smoke check is not a training-capacity estimate.
