# Reproduction status of the public paper bundle

This bundle documents artifacts associated with *Reward-Type Ablation Reveals
Mechanism-Dependent Algorithm Rankings in Mixed-Motive Multi-Agent Evaluation*.
The current source is a **1.0.8 candidate**, following published tag v1.0.7.
Environment, reward-propagation, experiment-completion, and analysis behavior
has changed. These fixes support new campaigns and artifact inspection; they
do not establish that historical paper results have been reproduced.

The supported installation and smoke route is maintained in the root
[REPRODUCE.md](../../REPRODUCE.md). From the repository root:

```bash
python -m pip install './coopetition_gym[dev,experiments]'
python -m pip install ./coopetition_gym/extensions/slcd_2d
python -m experiments.smoke --output data/paper-support-smoke
python -m experiments.validate training data/paper-support-smoke/integrated/raw --expected-records 2
python -m experiments.analyze returns-summary \
    --input-dir data/paper-support-smoke/integrated/raw \
    --output data/paper-support-smoke/integrated/returns.csv
```

Use a new, empty smoke directory. The smoke executes two seeds of a short
`Random` policy episode under each reward treatment, writes native JSONL,
validates the records, and checks summaries. It performs no learning and is
not evidence for the paper's rankings or case-study scores.

## What current tooling can establish

The installed `experiments` namespace supports recursive JSON and JSONL
results with `training_seed` and nested `metrics.mean_return`. It separates
identifiable progress rows, checks native schemas and finite successful
outcomes, detects repeated full-context cells, and can check an explicit
manifest's counts and checksums. Analysis records actual seeds, treatment,
configuration, and legacy treatment assignments.

Available analysis commands are `all`, `returns-summary`, `oracle-comparison`,
`tier-summary`, `masac-instability`, `training-metrics`, `learning-curves`,
`plots`, and `reward-ablation`. The old `scripts/*.py` table/figure mapping and
old `data/training/` layout are not a supported reproduction recipe. There is
no current promise that a named command recreates a particular paper table.

Stable algorithm IDs and the historical analysis grouping provide traceability.
Outputs mark `comparison_basis=historical-roster`. They do not establish
independent-learning versus CTDE equivalence, isolate a causal mechanism, or
certify a publication claim. The current
[implementation protocol](../../coopetition_gym/docs/benchmarks/implementation_protocol.md)
sets out these distinctions.

## What historical reproduction still requires

The released data are hosted at
[Hugging Face](https://huggingface.co/datasets/vikpant/coopetition-gym-logs).
Record the exact dataset revision and inspect its manifest. Reconcile the code
and algorithm implementations, environment configuration, effective training
and evaluation reward modes, budgets, seed fold, failure exclusions, and
rerun selection against the claim being checked. Physical shards, physical
rows, result records, and unique experimental cells must be counted separately.
A checksum check proves integrity of those files, not their scientific attribution.

Historical records identify seeds 99–108 as a canonical evaluation fold and
109–112 as stability characterization. Verify that convention for the selected
revision before using `--seeds`; the reader does not apply it automatically.
For a homogeneous set whose treatment is established from source evidence,
`--reward-type integrated` selects labelled integrated records and explicitly
assigns integrated treatment to legacy rows missing metadata. Directory names
alone are insufficient evidence. Named reward-ablation inputs likewise provide
explicit treatment context and require the same provenance review.

Mixed comparison groups and repeated cells are rejected rather than silently
averaged or deduplicated. Where the public records do not establish the needed
attribution, exact paper reproduction remains unresolved. New results from the
candidate must carry a new campaign identity and must not replace historical
records or frozen technical-report outputs. Case-study score discrepancies
require a separate artifact/version ruling; this document selects no new scores.
