# Datasheet for the Coopetition-Gym Datasets

Following the framework proposed by Gebru et al., *Datasheets for Datasets*,
Communications of the ACM (2021). This datasheet covers the two datasets
released alongside the companion research paper *Reward-Type Ablation Reveals
Mechanism-Dependent Algorithm Rankings in Mixed-Motive Multi-Agent Evaluation*.

| Dataset | Purpose | Size | HuggingFace repo |
|---|---|---|---|
| **coopetition-gym-v1** | Training results | 25,708 JSON files | `vikpant/coopetition-gym-v1` |
| **coopetition-gym-audit** | Behavioral audit | 1,116 JSON files | `vikpant/coopetition-gym-audit` |

---

## Motivation

### For what purpose were the datasets created?

To support the companion research paper demonstrating that algorithm
rankings in mixed-motive multi-agent reinforcement learning are mechanism-
dependent, specifically, that the dominance of Centralized Training with
Decentralized Execution (CTDE) over independent learning breaks down
systematically when the reward function structure is varied.

The datasets enable independent researchers to:

* Reproduce the paper's tables and figures using the published analysis
  pipeline (`experiments.analyze`).
* Extend the empirical analysis to new algorithms or new reward configurations
  without re-running the 3,400-GPU-hour reference evaluation.
* Audit the training dynamics and behavioral characteristics of each algorithm.
* Verify the exploitation-gradient claims made in the societal impact
  discussion (Appendix G).

### Who created the datasets?

Vik Pant and Eric Yu, Faculty of Information, University of Toronto.

### Who funded their creation?

Cloud compute costs (approximately $8,100 USD) were self-funded by the first
author. No external funding was involved.

---

## Composition

### What do the instances represent?

**Training dataset (`coopetition-gym-v1`)**: each instance is one training
experiment, the result of training one algorithm on one environment under
one reward configuration with one random seed. Each instance is a single JSON
file with the following top-level structure:

| Field | Type | Description |
|---|---|---|
| `algorithm` | str | Algorithm name (e.g., `ISAC`, `COMA`) |
| `environment` | str | Environment ID (e.g., `TrustDilemma-v0`) |
| `training_seed` | int | Seed in {99, 100, 101, 102, 103, 104, 105} |
| `status` | str | `success` or `failed` |
| `training_time_seconds` | float | Wall-clock training time |
| `evaluation_time_seconds` | float | Wall-clock evaluation time |
| `metrics` | object | Aggregated per-episode and per-seed metrics |
| `timestamp` | str | ISO 8601 completion timestamp |
| `gpu_id` | int | GPU index used during training (−1 for CPU) |
| `tr_mode` | str | TR tier label (`tr1`, `tr2`, `tr3`, `tr4`) |

The nested `metrics` object contains:

* `mean_return`, `std_return`, evaluation return statistics
* `mean_cooperation_rate`, mean fraction of endowment contributed
* `mean_final_trust`, mean final-step trust level
* `training_returns`, `training_timesteps`, training-time return curve
* `training_metrics`, gradient-level diagnostics (loss values by step)
* `tr_metrics`, TR-tier-specific domain metrics

**Behavioral audit dataset (`coopetition-gym-audit`)**: two subsets covering
the static response-surface audit (1,056 JSON files) and the temporal
deviation audit (60 JSON files). See paper Appendix F for the full schema and
the file `experiments/validate.py` (run
`python -m experiments.validate schema static_audit`) for the full field list.

### How many instances are there in total?

* Training dataset: **25,708** JSON files across 7 subfolders:
  `baseline_integrated/` (16,835), `ablation_private/` (2,450),
  `ablation_cooperative/` (2,450), `case_study/` (3,402),
  `france_bonus_isac_integrated/` (21), `local_bonus/` (70),
  `network_sensitivity/` (480).
* Audit dataset: **1,116** JSON files (1,056 static + 60 temporal).

### Does the dataset contain all possible instances or is it a sample?

The training dataset is a *complete factorial* over the published experimental
design: 16 training algorithms × 20 environments × 3 reward types × 7 seeds,
minus MeanFieldAC's 24 exclusions on two-agent environments (the mean-field
approximation is degenerate for N=2), plus supplementary case study and
sensitivity experiments. The 62 NaN-return files arise from documented
training instabilities in specific algorithm-environment-reward combinations
and are retained for transparency; they are excluded from the paper's numerical
analyses.

The audit dataset is a complete factorial over 18 policies × 20 environments
× 3 seeds, minus 24 MeanFieldAC dyadic exclusions = 1,056 static-audit files;
plus 20 environments × 3 seeds = 60 temporal-audit files.

### What data does each instance consist of?

Raw scalar metrics, training curves, and diagnostic telemetry from simulation
experiments. No images, audio, text, or user-generated content. No personally
identifiable information of any kind.

### Is any information missing from individual instances?

For 62 training-dataset files, certain per-episode metrics are `NaN` due to
documented training instability (21 MASAC baseline, 21 MADDPG/MATD3/M3DDPG
cooperative, 20 MADDPG network-sensitivity). These are retained for full
transparency of the evaluation output; `experiments.validate training` reports
the 62-count as an expected invariant.

### Are relationships between individual instances made explicit?

Yes. Instances are identified by the tuple
`(algorithm, environment, seed, reward_type)`. The file-naming convention
`{algorithm}_{environment}_{seed}.json` within the corresponding reward-type
subfolder encodes all four dimensions.

### Are there recommended data splits?

The dataset is a factorial design, not a train/test split. Researchers
typically aggregate across seeds for statistical summaries (see
`experiments.evaluate aggregate`) or compare pairs of conditions
(e.g., private vs. integrated reward).

### Are there any errors, sources of noise, or redundancies?

* **Seed variance**: training algorithms are stochastic; the same
  (algorithm, environment, reward_type) triple with different seeds may
  produce returns differing by up to 20% at convergence. This is not error;
  it is the stochasticity the 7-seed design is meant to quantify.
* **Documented NaN returns**: 62 files, as noted above.
* **No redundancy**: each instance is a unique experiment.

### Is the dataset self-contained, or does it link to external resources?

Self-contained. Each JSON file encodes the full result of one experiment.
The environment simulation code required to regenerate or extend the dataset
lives in the companion GitHub repository
(`https://github.com/vikpant/strategic-coopetition`).

### Does the dataset contain data that might be considered confidential or sensitive?

No. All data is synthetic simulation output from open-source environments.
No human subjects, no proprietary information, no trade secrets. The case
study calibrations (Samsung-Sony, Renault-Nissan, Apache, Apple) are based
on publicly documented historical business decisions.

### Does the dataset contain data that might be offensive, insulting, threatening, or otherwise cause anxiety?

No.

---

## Collection Process

### How was the data acquired?

By running the experiments in `experiments/campaign.py` on cloud GPU
infrastructure over approximately 3,400 GPU-hours distributed across 7 cloud
instances during March–April 2026. Each experiment consisted of training one
algorithm on one environment for 500,000 or 1,000,000 environment steps
(depending on environment category), followed by evaluation of the trained
policy on 100 episodes.

### What mechanisms or procedures were used to collect the data?

* Orchestration: `experiments/campaign.py` (unified orchestrator with
  bin-packed GPU allocation and resume-aware dispatch).
* Simulation: Coopetition-Gym (Gymnasium and PettingZoo environment suite).
* Training: implementation details for each of 16 algorithms, 7 oracles,
  2 heuristics, and 101 constant-action policies are in
  `experiments/algorithms.py`.
* Result persistence: one JSON file per experiment, written atomically on
  training completion.

### Over what timeframe was the data collected?

Primary training period: February 13, 2026 – April 14, 2026.
Behavioral audit: April 15, 2026.

### Were any ethical review processes conducted?

No human subjects were involved. No institutional review board approval
was required or sought. The work is pure computational simulation of
abstract coopetitive dynamics.

---

## Preprocessing

### Was any preprocessing done?

The released dataset is the *raw* orchestrator output. No post-hoc filtering
or transformation was applied. The analysis pipeline (`experiments/analyze.py`)
performs aggregation (e.g., mean ± std across seeds) at analysis time, not at
collection time.

### Is the software used to preprocess the data available?

Yes. `experiments/analyze.py` contains the full analysis pipeline used to
produce the paper's tables and figures from the raw JSON files.

---

## Uses

### Has the dataset been used for any tasks already?

The companion research paper. No other uses at time of release.

### Is there a repository that links to any or all papers or systems that use the dataset?

The GitHub repository README lists the companion research paper as the
primary user. Future uses will be tracked as they become available.

### What other tasks could the dataset be used for?

* Developing algorithms specifically tuned for mixed-motive multi-agent
  settings.
* Meta-analysis across reward configurations and mechanism classes.
* Uncertainty quantification and Bayesian modeling of MARL outcomes.
* Failure-mode characterization (see the paper's MASAC instability analysis).
* Educational material illustrating the reward-type ablation methodology.

### Are there tasks for which the dataset should not be used?

The dataset should **not** be used for:

* Training policies for deployment in actual business settings without
  extensive additional validation. The validated case studies (Samsung-Sony,
  Renault-Nissan, Apache, Apple) model real-world partnerships at an abstract
  level; transferring policies to operational decision-making requires
  domain-specific validation beyond the scope of this work.
* Claims about human cooperative behavior. The dataset reflects agent
  behavior under synthetic reward structures; it does not reflect empirical
  human behavior.

### Are there any risks of harm?

Limited, but noted in Appendix G of the paper: 1. The integrated reward configuration permits exploitation when the private
   gain exceeds the weighted partner loss. The behavioral audit empirically
   bounds this vulnerability (see Appendix F), but deploying policies trained
   under this configuration without further auditing is not recommended.
2. Policies that appear cooperative at a static cooperation level may be
   instrumentally motivated. The paper's methodology (reward-type ablation)
   is designed to surface such cases; users of derived policies should
   apply similar scrutiny.

---

## Distribution

### Will the dataset be distributed to third parties?

Yes, via HuggingFace Hub under CC-BY-4.0.

### When will the dataset be released?

Training dataset and audit dataset: on or before May 6, 2026, to coincide
with the companion research paper release.

### Will the dataset be distributed under a copyright or IP license?

CC-BY-4.0 (Creative Commons Attribution 4.0 International). Users may
share and adapt the dataset for any purpose, including commercial, provided
attribution is given to the original authors.

### Have any third parties imposed IP-based or other restrictions on the data?

No.

### Do any export controls or regulatory restrictions apply?

None known.

---

## Maintenance

### Who will be supporting, hosting, and maintaining the dataset?

Vik Pant. HuggingFace Hub provides hosting; GitHub Issues on the companion
repository (`vikpant/strategic-coopetition`) provides the primary issue
tracker.

### How can the owner be contacted?

Via GitHub Issues at `https://github.com/vikpant/strategic-coopetition/issues`
or email vik.pant@utoronto.ca.

### Is there an erratum?

None at release. Any corrections will be recorded in the GitHub repository's
CHANGELOG.md and on the HuggingFace dataset page.

### Will the dataset be updated?

The **v1** dataset released alongside the companion research paper is frozen
as a reproducibility reference. Future extensions (e.g., biaxial action spaces, dynamic D_ij) will
be released as separate versioned datasets (v2, v3, ...) with their own
datasheets.

### Will old versions continue to be supported?

Yes. HuggingFace preserves version history; archived tags will remain
accessible indefinitely.

### If others want to extend, augment, or build on the dataset, is there a mechanism for them to do so?

Yes. The reproducibility package in `experiments/` supports:

* Running additional algorithms on the existing environments.
* Running the existing algorithms on new environments.
* Running reward ablations on any (algorithm, environment) pair.
* Running behavioral audits on new (algorithm, environment) pairs.

Contributors should submit pull requests to the GitHub repository. Extended
datasets derived from the released v1 dataset should cite the original work
and release under CC-BY-4.0 (per the terms of the original license).
