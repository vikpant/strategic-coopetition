# extensions/

Post-v1 research code that is **not part of the `coopetition_gym` package**.

## Policy

Each subdirectory in this folder is a self-contained extension that: 1. **Does not modify** `coopetition_gym/`, `experiments/`, or `TR_validation/`.
2. **Does not get imported** by `coopetition_gym/` (enforced by CI).
3. Is **opt-in**: reviewers install separately, e.g. `pip install -e ./extensions/<name>/`.
4. Uses env-id suffix `v1extN` so downstream tooling can distinguish extension envs from v1 envs.

The `coopetition_gym` package on PyPI/GitHub remains frozen at v1 semantics. Extensions live here so reviewers can reproduce post-v1 findings without contaminating the base package.

## Contents

| Extension | Purpose | Status |
| --- | --- | --- |
| [`slcd_2d/`](slcd_2d/) | 2-action-dimension SLCD sanity check (cooperation + appropriation) | In development |

See `VERSION_COMPAT.md` for the base-package version each extension requires.
