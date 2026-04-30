# 2D SLCD Sanity Check

Post-v1 extension adding a second per-agent action dimension to the Samsung-Sony S-LCD environment.

**Env id:** `SLCDAppropriation-v1ext0`
**Status:** prototype, sanity check only (not in `coopetition_gym` v1)

## Why a second dimension?

TR-1 through TR-4 formalize *how agents coordinate* (interdependence, trust, loyalty, reciprocity). None formalize the **value-creation vs. value-capture tension** at the heart of coopetition (Brandenburger–Nalebuff, Ritala–Hurmelinna-Laukkanen). In the Samsung-Sony S-LCD joint venture, the two firms jointly invested in fab capacity (cooperation) while each pushed branded TVs that competed on the panels' downstream margin (appropriation). The dissolution in 2011 is what happens when appropriation pressure outruns cooperative returns, a dynamic v1 cannot currently express.

## Formalism

Per-agent action is `(c_i, p_i)` where

- `c_i ∈ [0, e_i]`, cooperation (TR-1 primitive, unchanged)
- `p_i ∈ [0, 1]`, appropriation effort (new)

Extended private payoff:

```
π_i^2D(c, p) = (e_i - c_i - κ·p_i)         # endowment net of both costs
             + θ·ln(1 + c_i)                # TR-1 individual value
             + α_i · S(c) · (1 - β·p̄)       # diluted synergy
             + η · p_i · S(c)               # private capture of joint output
             - ξ · p_i²                     # convex cost
```

where `S(c) = γ · (∏ c_i)^(1/N)` is the v1 synergistic surplus, `p̄ = (1/N)·Σ p_j`, and `(κ, β, η, ξ)` are the calibration parameters in [`calibration.json`](calibration.json).

Integrated utility follows v1:

```
U_i^2D(c, p) = π_i^2D(c, p) + Σ_{j≠i} T_ij · D_ij · π_j^2D(c, p)
```

where `T_ij` is effective trust (capped by reputation damage) exactly as in v1.

## Backward compatibility

When every `p_i ≡ 0`, the 2D formulation reduces to v1 integrated utility bit-exact (tolerance 1e-3, dominated by float32/float64 casting). This is enforced by [`tests/test_backward_compat.py`](tests/test_backward_compat.py).

## Default equilibrium

With calibration `(κ=0.5, β=0.6, η=0.4, ξ=15)` on the v1 SLCD parameters:

| Quantity | Samsung (agent 0) | Sony (agent 1) |
| --- | --- | --- |
| `c*` | 26.77 | 27.55 |
| `p*` | 0.071 | 0.056 |
| `U*` | 243.80 | 275.96 |

Converged in 7 iterations via scipy-based iterated best response.

Since `c* ≈ 27 < baseline 30`, trust erodes over an episode and the oracle trajectory ends at `mean_trust = 0` by step 40, the "dissolution" mode appearing endogenously.

## Installation

```bash
# Base package (v1) must be installed first
pip install -e .
# Then the extension (opt-in)
pip install -e ./extensions/slcd_2d/   # OR just import from the repo
```

## Running

Oracle smoke campaign:

```bash
python -m extensions.slcd_2d.campaign \
    --seeds 106,107,108 --steps 40 \
    --output .claude/experiments/slcd_2d/smoke/
```

Tests:

```bash
pytest extensions/slcd_2d/tests/ -v
```

## File map

| File | Purpose |
| --- | --- |
| `env.py` | `SLCDAppropriationEnv`, subclass of `SLCDEnv` with 2D action space |
| `utility.py` | Pure 2D utility math (`compute_2d_integrated_utilities`) |
| `oracle.py` | `AppropriationOracle`, solves interior `(c*, p*)` Nash |
| `calibration.json` | Default `(κ, β, η, ξ)` values |
| `campaign.py` | Stand-alone smoke/sanity runner |
| `tests/` | Backward-compat, shape, Nash-interior tests |
| `REPRODUCE.md` | Exact commands for reviewers |

## What this extension deliberately does *not* do

- Does not modify `coopetition_gym/` (read-only)
- Does not register into `coopetition_gym.envs._ENVIRONMENT_REGISTRY`
- Does not extend `experiments/campaign.py`, has its own `campaign.py`
- Does not ship training-algorithm support; only the oracle is implemented
