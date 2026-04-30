# Reproducing the 2D SLCD Sanity Check

## Prerequisites

- Python 3.10+ (tested on 3.12)
- `coopetition_gym` v0.2.0+ installed in editable mode
- `pip install scipy pytest numpy gymnasium`

## Steps

```bash
# 1. From repo root
cd strategic-coopetition

# 2. Run the full test suite (13 tests, ~1 second)
pytest extensions/slcd_2d/tests/ -v

# Expected output:
#   test_backward_compat.py::test_constant_trajectory_matches_v1       PASSED
#   test_backward_compat.py::test_varying_trajectory_matches_v1        PASSED
#   test_backward_compat.py::test_extreme_corner_trajectory_matches_v1 PASSED
#   test_nash_interior.py::test_equilibrium_converges                  PASSED
#   test_nash_interior.py::test_equilibrium_is_interior                PASSED
#   test_nash_interior.py::test_oracle_action_is_valid                 PASSED
#   test_nash_interior.py::test_oracle_utility_beats_zero_appropriation PASSED
#   test_shapes.py::test_action_space_shape                            PASSED
#   test_shapes.py::test_obs_space_matches_v1                          PASSED
#   test_shapes.py::test_step_returns_float32_rewards                  PASSED
#   test_shapes.py::test_action_is_clipped_to_bounds                   PASSED
#   test_shapes.py::test_invalid_action_shape_raises                   PASSED
#   test_shapes.py::test_appropriation_changes_reward                  PASSED
#   13 passed in ~1s

# 3. Run the oracle sanity campaign (3 seeds, ~1 second total)
python -m extensions.slcd_2d.campaign \
    --seeds 106,107,108 --steps 40 \
    --output .claude/experiments/slcd_2d/smoke/

# Expected output (identical across seeds — oracle is deterministic):
#   [seed=106] return=[6066.37 6012.89] c*=[26.77 27.55] p*=[0.071 0.056] final_trust=0.000
#   [seed=107] return=[6066.37 6012.89] c*=[26.77 27.55] p*=[0.071 0.056] final_trust=0.000
#   [seed=108] return=[6066.37 6012.89] c*=[26.77 27.55] p*=[0.071 0.056] final_trust=0.000
```

## What to check

1. **Backward compatibility**: `test_constant_trajectory_matches_v1`, `test_varying_trajectory_matches_v1`, and `test_extreme_corner_trajectory_matches_v1` all pass. When `p_i ≡ 0`, the 2D env produces the same reward stream as v1 `SLCDEnv` (tolerance 1e-3).
2. **Interior Nash**: `test_equilibrium_is_interior` passes. `c*` is in `(1, 99)` and `p*` is in `(1e-3, 0.999)`.
3. **Oracle dominates zero-appropriation**: `test_oracle_utility_beats_zero_appropriation` passes, confirming that appropriation is a strictly-better action than `p=0` at the calibrated parameters.

## Reproducibility checksum

```bash
cd extensions/slcd_2d
sha256sum env.py utility.py oracle.py calibration.json | sort
```

Any re-run with the same code should give bit-identical smoke-campaign outputs; numeric equilibrium values are reproducible to ~6 decimal places across BLAS implementations.

## Known limitations

- Oracle only; no training algorithms wired in yet.
- Calibration `(κ, β, η, ξ)` is a first pass — not fit to the SLCD 2004–11 dissolution timeline. Fine-tuning is a follow-up task.
- Extension is scoped to SLCD; transferring the 2D formalism to other v1 environments is out of scope for this sanity check.
