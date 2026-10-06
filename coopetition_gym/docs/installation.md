# Installation Guide

This guide describes **Coopetition-Gym 1.0.8, an unreleased source candidate**, and the optional **SLCD extension 0.1.1, also unreleased**. Installing the published PyPI package does not establish that these candidate fixes are present.

## Install from this repository

Use Python 3.9+ for the base package, or Python 3.10+ when using the SLCD extension. From the repository root:

```bash
git clone https://github.com/vikpant/strategic-coopetition.git
cd strategic-coopetition
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './coopetition_gym[dev,experiments]'
```

On Windows, activate with `.venv\Scripts\activate`. If an appropriate environment already exists, use it instead of creating another one. The install command belongs at the repository root; the Python examples and test commands below run from the package directory:

```bash
cd coopetition_gym
```

## Dependencies

The base package declares Gymnasium >=0.29, PettingZoo >=1.24, NumPy >=1.21, SciPy >=1.7, PyTorch >=2.0, and Stable-Baselines3 >=2.0. PyTorch and Stable-Baselines3 are core dependencies.

| Extra | Contents |
|---|---|
| `dev` | pytest, pytest-cov, Black, isort, mypy, flake8 |
| `viz` | matplotlib, seaborn |
| `experiments` | matplotlib, seaborn, pandas, diptest |
| `rl` | Compatibility alias for the already-required PyTorch and Stable-Baselines3 dependencies |
| `all` | All extras above |

The candidate wheel includes the `experiments` Python package. Source tests and research datasets are separate from installed runtime packages.

## Verify the base package

```python
import coopetition_gym as cg
import gymnasium as gym
import numpy as np

print(cg.__version__, len(cg.list_environments()))  # 1.0.8, 20 for this candidate
with gym.make("coopetition_gym:TrustDilemma-v0", reward_type="private") as env:
    obs, info = env.reset(seed=42)
    obs, rewards, terminated, truncated, info = env.step(
        np.array([60.0, 55.0], dtype=np.float32)
    )
    assert obs.shape == (15,)
    assert rewards.shape == (2,)
```

The Gymnasium factory preserves **one reward per agent**. A scalar-reward learner therefore needs an explicit adapter. `experiments.algorithms.MultiAgentToSingleAgentWrapper` sums the rewards for a joint controller; this changes the learning objective and must be reported. See the [wrapper reference](api/wrappers.md#scalar-reward-adapter).

## Optional SLCD extension

From the repository root, after installing the base candidate:

```bash
python -m pip install -e './coopetition_gym/extensions/slcd_2d[dev]'
```

The installed import is `slcd_2d`:

```python
from slcd_2d import SLCDAppropriationEnv

env = SLCDAppropriationEnv(reward_type="private")
obs, info = env.reset(seed=42)
obs, rewards, terminated, truncated, info = env.step([50.0, 0.2, 50.0, 0.2])
env.close()
```

Source-checkout imports such as `extensions.slcd_2d` also work from the package directory. The extension provides seven learner adapters and one oracle; see its [source README](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/extensions/slcd_2d/README.md).

## Tests without training

From `strategic-coopetition/coopetition_gym`:

```bash
python -B -m pytest -p no:cacheprovider -k 'not test_ippo_trains_on_2d'
```

Pytest discovers the core, experiment, and extension tests. The excluded extension test performs a short IPPO training run. Running the command without the exclusion includes that training test. No fresh benchmark campaign is required for these checks.

## Troubleshooting

Check `python -m pip show coopetition-gym` and `python -c "import coopetition_gym; print(coopetition_gym.__file__)"` in the same interpreter. The repository contains an outer `coopetition_gym/` directory and an inner Python package; running examples from the package directory avoids confusing the outer namespace with installed code. Source `__version__` and installed distribution metadata can differ in an old editable environment.

CPU execution is sufficient for environment smoke checks. GPU requirements depend on the selected algorithm and workload; no GPU capacity is implied by a successful import.

Report reproducible problems at the [repository issue tracker](https://github.com/vikpant/strategic-coopetition/issues). Continue with the [quickstart](tutorials/quickstart.md), [API reference](api/index.md), or [environment catalog](api/quick_reference.md#available-environments).
