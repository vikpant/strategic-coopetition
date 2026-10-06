# Benchmark implementation and historical results

Coopetition-Gym supplies 20 environments, three reward treatments, and 126
reference identifiers: 16 training implementations, 7 oracles, 2 heuristics and
101 constant-action configurations. The current source is the unreleased 1.0.8
candidate. Its runtime and analysis repairs require new runs with new provenance.

Start with the [implementation protocol](implementation_protocol.md). IPPO,
IA2C and ISAC use joint controllers; historical I/IND and C/CTDE roster labels do
not establish a controlled comparison of independent learning and decentralized
execution. Case-specific scoring conflicts are recorded in the
[score provenance register](score_provenance.md).

- [Algorithm comparison](algorithm_comparison.md): previously reported numerical
  tables retained with their historical labels and verification limits.
- [Environment defaults](environment_analysis.md): current factory defaults,
  distinguished from historical campaign and analysis groupings.
- [Reward-type ablation](reward_type_ablation.md): current reward semantics and
  the requirements for comparing treatments.
- [Case-study validation](case_study_validation.md): source artifacts and scoring
  interpretation.

Use the repository [reproduction guide](https://github.com/vikpant/strategic-coopetition/blob/master/REPRODUCE.md)
for a small environment-to-report check and explicit JSON/JSONL analysis. It does
not claim that every historical paper table can be regenerated from public data
without reconciling source revision, treatments, seeds, exclusions and manifests.
