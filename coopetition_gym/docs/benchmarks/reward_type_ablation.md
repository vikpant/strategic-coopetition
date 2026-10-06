# Reward-type ablation

The current reward implementation supports three modes. Each environment returns
one reward per agent; a joint SB3 controller then sums that vector.

| Mode | Per-agent reward |
| --- | --- |
| `private` | The agent's own private payoff |
| `integrated` | The environment's integrated utility, including its configured partner-payoff coupling and mechanism-specific terms |
| `cooperative` | The mean of the integrated utilities, shared by every agent |

Select a mode with `cg.make(env_id, reward_type="private")`. The configured
interdependence matrix and environment dynamics stay in place: private reward
selection does not literally zero the stored matrix. Cooperative reward is not
the sum of private payoffs. See `CoopetitionEnv.step` in the base implementation.

Version 1.0.8 repairs ignored keyword overrides and propagates mode explicitly
through campaigns and evaluation. Do not infer that historical directories named
`private` or `cooperative` prove which objective their runs actually used.

## Comparisons

Hold source revision, algorithm parameters, environment configuration, requested
training budget, evaluation protocol and seed fold fixed across treatments.
Analyze homogeneous inputs and record failed/nonfinite exclusions. Reward labels
missing from legacy data must be supplied from source evidence; the analyzer
records explicit assignments. Repeated seeds/configuration variants require an
explicit upstream selection rather than automatic pooling.

For compatible inputs the analysis command reports absolute/relative changes in
observed returns. A difference in return across objectives is not, on its own,
a causal estimate of a mechanism's effect: the objective scale and learned policy
can both change. Historical I/IND versus C/CTDE labels likewise do not establish
an independent-learning versus centralized-training comparison; see the
[implementation protocol](implementation_protocol.md).

## Previously reported Apple comparison

These values are retained from the pre-repair documentation. They have not been
regenerated or certified by the 1.0.8 fixes. The cited historical artifact was
`aggregates/returns_summary_v2.csv`; its exact snapshot and fold need reconciliation.

| Treatment | COMA return | ISAC return | Reported gap (ISAC − COMA, relative to COMA) |
| --- | ---: | ---: | ---: |
| Private | 23,191 | 24,854 | +7.2% |
| Integrated | 40,670 | 39,953 | −1.8% |
| Cooperative | 40,673 | 38,979 | −4.2% |

This is a historical implementation comparison. It does not establish the claimed
learning-paradigm or mechanism explanation without the controls above. Other
historical tier tables remain on [algorithm comparison](algorithm_comparison.md).
Use the [reproduction guide](https://github.com/vikpant/strategic-coopetition/blob/master/REPRODUCE.md)
for working commands.
