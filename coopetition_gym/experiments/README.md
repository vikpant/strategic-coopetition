# Experiment tooling

This package contains campaign orchestration, evaluation, record validation,
and analysis. The working tree is a 1.0.8 candidate with behavior changes from
published v1.0.7. It supports new, explicitly identified campaigns and careful
inspection of historical records. It does not certify that the historical
paper tables are reproduced by a new run.

Install from the repository root:

```bash
python -m pip install './coopetition_gym[dev,experiments]'
```

The installed namespace is `experiments`. The complete setup, extension install,
tests, and provenance rules are in the root [REPRODUCE.md](../../REPRODUCE.md).

## Small end-to-end check

```bash
python -m experiments.smoke --output data/smoke
python -m experiments.validate training data/smoke/integrated/raw --expected-records 2
python -m experiments.analyze returns-summary \
    --input-dir data/smoke/integrated/raw \
    --output data/smoke/integrated/returns.csv
python -m experiments.analyze reward-ablation \
    --input-baseline data/smoke/integrated/raw \
    --input-private data/smoke/private/raw \
    --input-cooperative data/smoke/cooperative/raw \
    --output-dir data/smoke/comparison
```

Choose a new output directory for the smoke command. It executes six short
`Random` policy runs, with no training, and verifies the written JSONL records
and summaries. The generated data are installation fixtures, not paper results.

## Modules and commands

| Module | Supported entry points |
|---|---|
| `campaign` | `baseline`, `private`, `cooperative`, `sensitivity`; use subcommand `--help` before a new campaign |
| `evaluate` | `agent`, `aggregate` |
| `analyze` | `all`, `returns-summary`, `oracle-comparison`, `tier-summary`, `masac-instability`, `training-metrics`, `learning-curves`, `plots`, `reward-ablation` |
| `audit` | `static`, `temporal`, `analyze` |
| `validate` | `training`, `audit`, `schema` |
| `smoke` | Small real-environment result/validation/analysis check |
| `records` | Shared recursive JSON/JSONL reader, schema/quality checks, comparison identity, manifest checks |
| `config` | Explicit defaults, campaign allocation, and separately named historical analysis grouping |

For example, `python -m experiments.analyze all --help` lists the accepted
options. Historical commands named `paradigm-boundary`, `oracle-exceedance`, or
`dij-contribution` are not aliases for current commands.

## Records and comparisons

Results use `training_seed` and nested `metrics.mean_return`. The loader
preserves source locations and original provenance, separates identifiable
progress rows, and reads individual JSON files or JSONL shards recursively.
The validator reports schema problems, failed/nonfinite results, duplicate
full-context cells, and optional manifest count/checksum mismatches. Historic
corpus totals and expected NaN counts are not default validation criteria.

Analysis uses valid successful outcomes, reports observed seed counts, and
rejects mixed treatments, configurations, campaigns, or repeated seed cells
within a comparison. Supply `--seeds` to select a verified evaluation fold.
Use `--reward-type` only with established treatment evidence: it selects labelled
rows and explicitly assigns missing legacy metadata. Named `reward-ablation`
inputs also assign their stated treatment when legacy metadata is absent.
Outputs disclose these assignments and the configuration of each arm.
Cross-treatment differences require identical observed seeds and matching
recorded scientific configuration; unmatched arms are marked incomparable and
have blank differences. Pass result `raw/` directories; campaign containers
also contain manifests/logs and are not result datasets.

`all` loads each unified input once. Its numerical summaries exclude invalid
outcomes while its diagnostics retain observed failures and nonfinite values.
`masac-instability` reports supplied telemetry and missing telemetry without
assuming historical counts or a cause of failure.

Algorithm IDs and the historical tier grouping remain stable for traceability.
Generated summaries identify `comparison_basis=historical-roster`; the names do
not establish that implementations are equivalent to independent-learning or
CTDE paradigms. See the [implementation protocol](../docs/benchmarks/implementation_protocol.md).
A new controlled comparison requires a separately specified
protocol and new campaign provenance.

## Historical release limits

The public corpus and older paper bundles have differing formats, counts,
folds, and provenance statements. Select a dataset revision and inspect its
manifest before analysis. Repeated cells require an explicit upstream selection;
the loader does not infer which run is canonical. Corrected behavior in this
candidate requires new runs with new identities, not silent replacement of
historical artifacts. Refer to the root reproduction guide for the current
supported route and the limits of historical reproduction.
