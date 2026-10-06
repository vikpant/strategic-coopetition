# Implementation and interpretation protocol

This describes the unreleased 1.0.8 source candidate. Reproduce an archived result
using its exact source revision, dataset snapshot, configuration and seed fold.
The runtime and analysis fixes in this candidate do not retroactively validate
historical reward assignments or regenerate published results.

## Algorithm identifiers

The roster has 16 training implementations, 7 reference oracles, 2 heuristics and
101 constant-action configurations (126 identifiers). Identifiers remain stable
for compatibility with recorded data. They are not independent confirmations of
the scientific properties implied by the algorithm names.

| Historical identifier | Implemented controller |
| --- | --- |
| IPPO / `IndependentPPO` | One SB3 PPO model consuming the joint observation, producing the joint action, and optimizing summed agent rewards |
| IA2C / `IndependentA2C` | One SB3 A2C model with the same joint input/output and reward aggregation |
| ISAC / `IndependentSAC` | One SB3 SAC model with the same joint input/output and reward aggregation |
| SelfPlay_PPO | Wraps the IPPO implementation; inherits its joint-controller semantics |
| IndependentREINFORCE | Separate per-agent policy networks/optimizers; this does not establish equivalence to the three SB3 implementations above |

See `experiments/algorithms.py`, especially `SingleAgentWrapper`. Tables using
`I`, `IND`, `C`, or `CTDE` preserve historical roster labels. A comparison between
those rosters is not by itself a controlled comparison of independent learning
and centralized training with decentralized execution. Such a claim needs
verified actor/critic information access, reward aggregation, parameter sharing,
and matched budgets for every compared implementation. Numeric historical tables
are retained as reported observations, not re-certified by this repair.

## Rewards and resets

Every environment factory now honors explicit `reward_type` overrides, including
when the factory supplies a configuration object. The private mode uses private
payoffs; integrated mode applies the environment's integrated utility; cooperative
mode shares the mean integrated utility across agents. Cooperative mode does not
mean the sum of private payoffs. Changing reward mode leaves the mechanism
configuration in place. Both PettingZoo wrappers call the environment's public
reset so subclass episode state is initialized on every reset.

The Gymnasium interface returns a reward vector and a joint action space.
Registration disables the scalar-reward passive checker; registration alone does
not make this a scalar-reward single-agent task. Use PettingZoo for agent-indexed
interaction or an explicitly documented scalar aggregation for SB3.

## Groups, units and completion

`config.ENVIRONMENTS_BY_TR` retains the historical campaign allocation.
`config.ANALYSIS_ENVIRONMENTS_BY_TR` retains the different historical analyzer
allocation. These mappings are named separately; neither determines a case
study's source technical report. Per-environment outputs are preferable when
comparing artifacts whose grouping provenance is incomplete.

An analysis unit is one recorded algorithm/environment/seed/configuration under
one reward treatment. Duplicate records and mixed variants are rejected; failed
and nonfinite results are disclosed and excluded from successful-return estimates.
The combined command reads a corpus once. Standard deviation across seeds uses
the sample definition; SEM uses the number of unique contributing seeds.

Campaign success requires finite evaluation and measured training steps meeting
the requested budget for learners. A resume flag, completion-state key, filename,
or checkpoint alone is insufficient. Versioned manifests distinguish new output
from historical folders. Legacy records without this evidence can be analyzed
when their context is established, but do not prove completed training.

See the repository [reproduction guide](https://github.com/vikpant/strategic-coopetition/blob/master/REPRODUCE.md) for commands and
[score provenance](score_provenance.md) for case-study evidence limits.
