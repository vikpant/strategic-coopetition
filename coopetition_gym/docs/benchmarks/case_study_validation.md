# Case-study validation

The public validation code, saved outputs and prose summaries do not all agree.
Use the [score provenance register](score_provenance.md), which records the
source for each value and leaves the Apache and Apple canonical scores unresolved.
The four rubrics are case-specific correspondence checks, not comparable accuracy
rates. Validation code and saved outputs remain intact.

The source suites are in `TR_validation/TR1_foundations`, `TR2_trust`,
`TR3_loyalty` and `TR4_reciprocity`. Their READMEs describe parameters and CLI
options. Run suites in a separate copy or choose an output directory so existing
results are preserved. A new run should record the commit, command, parameter
set, rubric, dependency versions and output checksum.

RL return comparisons belong to the [implementation protocol](implementation_protocol.md)
and [algorithm comparison](algorithm_comparison.md). An oracle exceedance or a
learning return does not resolve a historical case-study rubric conflict.
