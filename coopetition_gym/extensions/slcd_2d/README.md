# 2D SLCD Extension

**Unreleased candidate:** extension `0.1.1`, requiring base `coopetition-gym>=1.0.8`. These source changes do not publish either distribution. **Environment ID:** `SLCDAppropriation-v1ext0`. This remains a prototype with a separate registry from the 20 base environments.

Each agent chooses cooperation `c_i` and appropriation effort `p_i`. The prototype explores a value-creation/value-capture tension using the Samsung-Sony setting; its numerical behavior does not establish a historical causal explanation for dissolution.

## Formalism

The flattened action is `[c_0, p_0, c_1, p_1]`, with `c_i` bounded by the agent's endowment and `p_i` in `[0, 1]`.

```text
π_i(c, p) = (e_i - c_i - κ p_i)
          + θ ln(1 + c_i)
          + α_i S(c) (1 - β mean(p))
          + η p_i S(c) - ξ p_i²
S(c) = γ (∏ c_i)^(1/N)
U_i(c, p) = π_i(c, p) + Σ_{j≠i} T_ij D_ij π_j(c, p)
```

`T_ij` is effective trust after reputation limits. Defaults are recorded in [calibration.json](calibration.json): `κ=0.5`, `β=0.6`, `η=0.4`, `ξ=15`. They are a coarse prototype calibration.

For the tested integrated-reward trajectories, setting every `p_i=0` matches the base SLCD rewards within absolute tolerance `1e-3`; see [backward-compatibility tests](tests/test_backward_compat.py). This is a numerical compatibility check, not a rerun of a historical case-study validation rubric. Floating-point, solver and platform differences can affect results.

## Install

From the **repository root**, install both source candidates into the chosen Python 3.10+ environment:

```bash
python -m pip install -e './coopetition_gym[dev,experiments]'
python -m pip install -e './coopetition_gym/extensions/slcd_2d[dev]'
```

The installed package name is `slcd_2d`. Source imports using `extensions.slcd_2d` remain available from the `coopetition_gym/` package directory.

```python
from slcd_2d import SLCDAppropriationEnv

env = SLCDAppropriationEnv(reward_type="private", max_steps=40)
obs, info = env.reset(seed=42)
obs, rewards, terminated, truncated, info = env.step([50.0, 0.3, 50.0, 0.3])
assert obs.shape == (15,)
assert rewards.shape == (2,)
env.close()
```

Reward modes are `integrated`, `private` and `cooperative`; an explicit argument overrides `COOPETITION_REWARD_TYPE`. Reset clears appropriation metrics.

## Algorithms and commands

The extension registry in [algorithms.py](algorithms.py) contains **seven learner adapters**: `IPPO`, `ISAC`, `IA2C`, `MAPPO`, `MADDPG`, `MATD3`, `MASAC`; plus `Oracle_Appropriation`. These reuse the packaged `experiments.algorithms` implementations. The first three historical IDs use joint controllers with summed rewards; they should not be described as decentralized independent learners solely from their names.

The basic `campaign` module runs the oracle only. `campaign_tier1` and `campaign_tier15` provide the broader orchestration; `calibrate` supplies endpoint and waypoint objectives. Availability of these modules is separate from validation of a completed training campaign.

An oracle-only smoke run, after installation, can be written to a local results directory:

```bash
python -m slcd_2d.campaign --seeds 106,107,108 --steps 40 \
    --output ./results/slcd_2d/smoke
```

From the repository root, run extension tests without the short IPPO training test:

```bash
cd coopetition_gym
python -B -m pytest -p no:cacheprovider extensions/slcd_2d/tests \
    -k 'not test_ippo_trains_on_2d'
```

## Preflight checks

Installed module entry points are `python -m slcd_2d.pre_launch_check_tier1` and `python -m slcd_2d.pre_launch_check_tier15`. Both accept `--repo-root /path/to/strategic-coopetition` for source tests, or `--skip-pytest` when no checkout is available. Source module entry points use `extensions.slcd_2d` instead.

Preflights include a short IPPO training gate; `--skip-pytest` does **not** skip that gate. They are preparation for an intentionally requested campaign, not an import-only check.

See [REPRODUCE.md](REPRODUCE.md) for bounded checks, output provenance, and limitations. The extension is not registered as a base Gymnasium environment and is not part of the frozen historical experiment artifacts.
