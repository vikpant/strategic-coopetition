# Reproducing the 2D SLCD Prototype

This guide targets **unreleased extension candidate 0.1.1** and **unreleased base candidate 1.0.8**. It describes source checks and a small oracle simulation. It does not assert that the candidate reproduces historical training campaigns.

## Install from the repository root

Use Python 3.10+ and a suitable existing or fresh virtual environment:

```bash
python -m pip install -e './coopetition_gym[dev,experiments]'
python -m pip install -e './coopetition_gym/extensions/slcd_2d[dev]'
```

The installed import is `slcd_2d`; source imports from the package directory use `extensions.slcd_2d`.

## Bounded verification without training

From the repository root:

```bash
cd coopetition_gym
python -B -m pytest -p no:cacheprovider extensions/slcd_2d/tests \
    -k 'not test_ippo_trains_on_2d'
```

This checks integrated-reward compatibility at zero appropriation, reward routing, action bounds, reset behavior, dilution invariants, equilibrium properties, calibration objectives, and algorithm construction. The excluded test performs a short IPPO training run. Passing the remaining tests does not establish the learning performance of the seven supported learner adapters.

An installed-package smoke check requires no model training:

```python
from slcd_2d import SLCDAppropriationEnv
from slcd_2d.algorithms import list_algorithms

env = SLCDAppropriationEnv(max_steps=40)
obs, info = env.reset(seed=42)
obs, rewards, terminated, truncated, info = env.step([50.0, 0.3, 50.0, 0.3])
assert obs.shape == (15,) and rewards.shape == (2,)
assert len(list_algorithms()) == 8  # Seven learner adapters and one oracle.
env.close()
```

## Optional oracle simulation

After installation, run from a working directory where results may be written:

```bash
python -m slcd_2d.campaign --seeds 106,107,108 --steps 40 \
    --output ./results/slcd_2d/smoke
```

This solves the configured appropriation equilibrium and evaluates an oracle policy; it does not train a policy. Inspect convergence, action bounds, return vectors, trust trajectories, and the saved calibration. Compare outputs numerically with an explicit tolerance; solver and floating-point differences preclude a general bit-for-bit guarantee.

## Provenance

Record the Git revision, source and installed distribution versions, Python/dependency versions, calibration, seeds, horizon and exact command alongside each new output. From the repository root, checksums of the relevant source files can be collected with:

```bash
sha256sum coopetition_gym/extensions/slcd_2d/env.py \
    coopetition_gym/extensions/slcd_2d/utility.py \
    coopetition_gym/extensions/slcd_2d/oracle.py \
    coopetition_gym/extensions/slcd_2d/calibration.json
```

These are fresh provenance records, not replacements for historical manifests. The backward-compatibility tests use absolute tolerance `1e-3` for integrated-reward trajectories at `p=0`; they do not establish case-study rubric scores.

## Training and limitations

`IPPO`, `ISAC`, `IA2C`, `MAPPO`, `MADDPG`, `MATD3` and `MASAC` are available through `slcd_2d.algorithms`; `Oracle_Appropriation` is the eighth entry. The basic `campaign` module remains oracle-only, while the tiered campaign modules support learner runs. Calibration supports endpoint and waypoint objectives, but these are model targets rather than independent historical validation.

Preflight modules include short training and must be invoked only when that work is intended. Their `--skip-pytest` option skips source tests, not the training gate. An installed preflight needs `--repo-root` for source tests; it no longer assumes an author's home directory.

The prototype remains specific to SLCD. Its calibration and simulated dissolution behavior do not prove why the historical joint venture ended. For the distinction between recorded scores and current claims, see [score provenance](../../docs/benchmarks/score_provenance.md).
