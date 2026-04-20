# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] — 2026-04-19

### Documentation clarifications

- **Clarified role of `core/collective_action.py`** — module docstring
  rewritten to state that this module provides support utilities
  (dataclasses, state-tracking containers) used by the TR-3
  environments, and that the authoritative TR-3 paper formalism is
  implemented in `envs/collective_action_envs.py`. No code changes.
- **Clarified role of `core/reciprocity.py`** — module docstring
  rewritten to state that this module provides support utilities used
  by the TR-4 environments, and that the authoritative TR-4 paper
  formalism is implemented in `envs/reciprocity_envs.py`. No code
  changes.
- **Added architectural pointer to `envs/collective_action_envs.py`** —
  module docstring now explicitly identifies this file as the
  authoritative TR-3 implementation and points to the `core/` helper
  location.
- **Added architectural pointer to `envs/reciprocity_envs.py`** —
  module docstring now explicitly identifies this file as the
  authoritative TR-4 implementation and points to the `core/` helper
  location.

### Provenance notes

- The code state that produced the 25,708-file training dataset and
  the 1,116-file behavioral audit dataset is preserved at the git tag
  `v1.0.0-campaign` on `master`. Users who require byte-exact
  reproduction of campaign-era package behavior should pin to that
  tag.
- All 143 pytest tests pass on v1.0.1 with identical behavior to
  v1.0.0. No computational code, no class or function signatures, no
  import paths, and no numerical outputs have changed.

### Not changed

- `experiments/` reproducibility package (unchanged)
- Algorithm implementations (unchanged)
- Dataset formats or contents (unchanged)
- `coopetition_gym` public API surface (unchanged)
- Class and function names in `core/collective_action.py` and
  `core/reciprocity.py` (unchanged; these remain stable across v1.x).
  Consolidation of these helper modules into `envs/` or renaming to
  `core/*_support.py` is reserved for v2.0.0, where a SemVer-major
  break is already planned.

---

## [1.0.0] — 2026-04-18

### Added

- **`experiments/` reproducibility package** (13,663 lines across 9 modules).
  Single entry point for reproducing every table and figure in the companion
  research paper *Reward-Type Ablation Reveals Mechanism-Dependent Algorithm
  Rankings in Mixed-Motive Multi-Agent Evaluation*.
  - `config.py` — single source of truth for all defaults (seeds, reward
    types, environments, algorithms, oracle references, safety settings).
  - `algorithms.py` — 16 training algorithms (IPPO, IA2C, ISAC, LOLA,
    SelfPlay_PPO, IndependentREINFORCE, FCP; MAPPO, MADDPG, MATD3, MASAC,
    M3DDPG, QMIX, VDN, COMA, MeanFieldAC), 7 game-theoretic oracles,
    2 heuristic baselines (Random, TitForTat), 101 constant-action policies.
  - `campaign.py` — unified orchestrator with subcommands `baseline`,
    `private`, `cooperative`. Safety defaults on (checkpoints, GPU memory
    monitoring, thermal monitoring, dynamic backpressure).
  - `sensitivity.py` — network capacity sensitivity analysis.
  - `audit.py` — behavioral audit (static response surface + temporal
    deviation). Subcommands `static`, `temporal`, `analyze`.
  - `evaluate.py` — policy evaluation and cross-seed aggregation.
    Subcommands `agent`, `aggregate`.
  - `analyze.py` — paper table and figure generators. Nine subcommands
    covering returns summary, oracle comparison, MASAC instability,
    learning curves, plots, reward-type ablation, and more.
  - `validate.py` — dataset integrity checks. Subcommands `training`,
    `audit`, `schema`.
  - `monitor.py` — local-friendly progress, health, and disk monitor.
    Subcommands `watch`, `clean-checkpoints`, `disk-status`, `health-check`.
- **`REPRODUCE.md`** — step-by-step instructions for reproducing paper
  results from the released datasets.
- **`.github/workflows/tests.yml`** — pytest matrix on Python 3.9/3.10/3.11/3.12
  × ubuntu-latest.
- **`.github/workflows/install.yml`** — install-verification matrix on
  macos-latest / ubuntu-latest / windows-latest × Python 3.10/3.12,
  exercising Gymnasium and PettingZoo Parallel APIs.
- **Root `LICENSE`** (MIT) covering the whole repository.
- **`CITATION.cff`** for the GitHub "Cite this repository" button.
- **`DATASHEET.md`** following Gebru et al. for the released datasets.
- **`CHANGELOG.md`** (this file).

### Changed

- Root `README.md` expanded from a minimal profile to a full project landing
  page with quick-start, installation, reproducibility pointer, case study
  summary, and technical report references.
- `coopetition_gym/README.md` extended with a new oracles section documenting
  all 7 game-theoretic oracles and their TR-tier applicability, plus a
  companion-paper callout block.
- Case study validation scores corrected to match the authoritative
  `TR_validation/` suite values (Apache 52/60 = 86.7%; Apple 48/55 = 87.3%;
  Samsung-Sony and Renault-Nissan unchanged at 58/60 = 96.7% and
  49/60 = 81.7% respectively).
- Paper and checklist corrected from 128 → 126 algorithms, with the breakdown
  disaggregated as "16 training algorithms, 2 heuristic baselines,
  7 game-theoretic oracles, and 101 constant-action policies".
- Total campaign cost updated from approximate placeholder to the authoritative
  invoice total of $8,100 USD.
- Package version bumped from `0.3.0` to `1.0.0`.

### Fixed

- **Namespace-package shadowing** — when running from the repository root,
  Python resolved `coopetition_gym` to the outer folder (which has no
  top-level `__init__.py`) as a namespace package, causing attribute errors
  when attempting to call `coopetition_gym.make(...)`. Fixed consistently
  across every consolidated module via a dedicated import helper that
  prepends the inner-package parent and drops stale imports.
- **Install verification workflow** — Windows GitHub Actions runners default
  to PowerShell, which does not understand the bash heredoc syntax used in
  the install verification steps. Added `defaults.run.shell: bash` and
  `working-directory: coopetition_gym` to avoid both the heredoc issue and
  the namespace shadow in CI.

### Validated

- **143 pytest tests pass** on Python 3.9, 3.10, 3.11, 3.12 (ubuntu-latest).
- **6 install-verification jobs pass** on macOS, Ubuntu, and Windows
  × Python 3.10, 3.12.
- **Deterministic regression** (Constant_50 on TrustDilemma-v0 seed 99) on a
  fresh clone of the public GitHub repository matches the released dataset
  within 1.7 × 10⁻⁷ relative error.
- **Training regression** — IA2C (via SB3) and IndependentREINFORCE (custom
  PyTorch) both train to completion through the consolidated orchestrator.

### Not Included in the Public Release (Archived)

Campaign-specific infrastructure referenced specific cloud-instance
identifiers and SSH credentials, and has been archived to
`.claude/experiments/archive/` rather than published:

- Per-instance launch shell scripts (`run_instance*.sh`)
- Algorithm-specific patch scripts (`isac_metrics_*.sh`, `oracle_nash_rerun.sh`)
- Instance deployment and tarball scripts (`setup_remote.sh`,
  `create_results_tarball.sh`, `monitor_all_instances.sh`)
- Cloud-orchestration monitoring (`background_monitor.py`'s SSH-polling
  functions)
- Dataset extraction helpers tied to specific tarball filenames
  (`post_process_results.py`, `merge_dataset.py`)

---

## [0.3.0] — 2026 (pre-consolidation)

Editable-install package release of `coopetition-gym` supporting Gymnasium and
PettingZoo APIs. Internal release; used during the v1 training campaign.

## [0.2.0] — 2025

Initial public release of the Coopetition-Gym environment suite with 10
environments and supporting utilities.
