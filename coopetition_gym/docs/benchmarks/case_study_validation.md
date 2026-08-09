# Case Study Validation

Four of the twenty environments in Coopetition-Gym v1 are calibrated to historically documented coopetitive relationships. Calibration extracts interdependence coefficients and other model parameters from qualitative coding of strategic dependencies in archival sources, then verifies that the calibrated environment produces simulation trajectories qualitatively consistent with documented historical outcomes.

> **Source:** `TR_validation/` directory in this repository. Each environment's score is reproduced when running the corresponding validation suite (`TR1_validation_suite.py`, `TR2_validation_suite.py`, `TR3_validation_suite.py`, `TR4_validation_suite.py`).

---

## Validation scores

| Environment | Case study | Mechanism class | Validation score | Source script |
|-------------|-----------|-----------------|------------------|---------------|
| `SLCD-v0` | Samsung-Sony LCD joint venture (2004-2011) | TR-1 (interdependence) | 58/60 log, 46/60 power | `TR_validation/TR1_foundations/TR1_validation_suite.py` |
| `RenaultNissan-v0` | Renault-Nissan Alliance (1999-present) | TR-2 (trust dynamics) | 49/60 | `TR_validation/TR2_trust/TR2_validation_suite.py` |
| `ApacheProject-v0` | Apache HTTP Server community (1995-2023) | TR-3 (collective action) | 45/60 | `TR_validation/TR3_loyalty/TR3_validation_suite.py` |
| `AppleAppStore-v0` | Apple iOS App Store ecosystem (2008-2024) | TR-4 (reciprocity) | 43/51 | `TR_validation/TR4_reciprocity/TR4_validation_suite.py` |

> **Note on SLCD-v0:**

> **What the 60-point figure is.** It is a **plausibility score**, not an accuracy rate. It
> counts weighted rubric conditions — convergence, and cooperation and value increases falling
> inside bands calibrated to the documented S-LCD record — and it is informative chiefly as a
> comparison *between* functional specifications. Against a random-parameter null of 20,000
> draws (D drawn uniformly from [0,1]², α from [0.30, 0.70]), the logarithmic mean is 54.2 and
> the power mean is 45.8, and logarithmic outscores power in 96.8% of draws. Individual case
> totals should therefore be read as confirming that a specification behaves plausibly under
> that case's parameters, not as a measure of predictive accuracy for that case.
 The validation suite shipped in this repository scores SLCD-v0 at 58/60 on the logarithmic specification and 46/60 on the power specification; these are the scores that running `TR1_validation_suite.py` from this repository produces today.

---

## What the scores mean

Each validation suite implements a Behavioral Correspondence Protocol scoring rubric that asks whether the calibrated environment, when run with the case-study parameters, produces trajectories matching coded historical outcomes on a battery of qualitative criteria. A score of 49/60 means the calibrated environment matches 49 of 60 codeable behavioral predictions of the case study.

The four scores are not directly comparable across cases because the rubrics are case-specific (they enumerate phenomena that the case study documents and that the model should reproduce). They are independently meaningful as evidence that each formal mechanism captures something empirically observable, and collectively suggestive that the package's mechanism classes have empirical referents.

---

## Sample validation summaries

The case studies span four very different coopetitive settings: a manufacturing joint venture (SLCD), a long-running automotive alliance (Renault-Nissan), an open-source software community (Apache HTTP Server), and a platform ecosystem (Apple iOS App Store). Each case has its own coding scheme and its own specific phenomena to reproduce.

### SLCD-v0 (TR-1)

Validation runs the calibrated logarithmic value function against documented investment patterns from the Samsung-Sony LCD joint venture (2004-2011). The validation suite reports 58/60 on the logarithmic specification with `θ=20`, `γ=0.65`, plus 46/60 on a power-function alternative — together establishing that the logarithmic form fits this case better than the alternative.

Source: `TR_validation/TR1_foundations/README.md` and `TR_validation/TR1_foundations/tr1_results.json`.

### RenaultNissan-v0 (TR-2)

Validation runs the two-layer trust model (immediate trust `T_ij` and exponentially smoothed reputation `R_ij` with 3:1 negativity bias λ⁺=0.10, λ⁻=0.30) against documented trust-trajectory phases of the Renault-Nissan Alliance. The model reproduces the documented hysteresis (slow rebuild after the 2018 governance crisis) and the asymmetric trust-decay pattern characteristic of the alliance's published case-study analyses.

Source: `TR_validation/TR2_trust/README.md`.

### ApacheProject-v0 (TR-3)

Validation runs the loyalty-modified team-production model against documented contribution patterns in the Apache HTTP Server community (1995-2023). The model reproduces the empirically observed 96.5% free-riding baseline accuracy and 100% loyalty monotonicity required of the formalism.

Source: `TR_validation/TR3_loyalty/README.md`.

### AppleAppStore-v0 (TR-4)

Validation runs the bounded-reciprocity model with `φ(x) = tanh(κx)` against documented developer-platform interactions in the Apple iOS App Store (2008-2024). The model reproduces six behavioral targets: cooperation emergence (97.5%), defection punishment (100.0%), forgiveness dynamics (87.9%), asymmetric differentiation (100.0%), trust-reciprocity interaction (100.0%), and bounded responses (100.0%).

Source: `TR_validation/TR4_reciprocity/README.md`.

---

## ISAC oracle exceedance on TR-3 case study

A noteworthy result on the TR-3 collective-action environments, including the calibrated `ApacheProject-v0` case, is that ISAC (Independent Soft Actor-Critic) exceeds the highest mean episodic return achievable by any constant-action policy. The exceedance is small but consistent across all five TR-3 environments and all seeds.

| Environment | ISAC return | Oracle_Loyalty return | Gap |
|-------------|-----------:|----------------------:|----:|
| `ApacheProject-v0` | 5,539,736 | 5,484,826 | +1.00% |
| `CoalitionFormation-v0` | 424,560 | 421,152 | +0.81% |
| `LoyaltyTeam-v0` | 124,120 | 123,359 | +0.62% |
| `PublicGoods-v0` | 183,372 | 182,166 | +0.66% |
| `TeamProduction-v0` | 90,548 | 89,390 | +1.29% |

> **Source:** `aggregates/oracle_exceedance_v2.txt`. Oracle_Loyalty is the highest-return constant-action policy under loyalty-modified payoffs at the calibrated `θ`. The exceedance is small because Oracle_Loyalty is already a strong reference; the exceedance is consequential because it indicates that ISAC's stochastic policy discovers adaptive within-episode action sequences that no fixed-cooperation-level policy can match.

The exceedance is positive on every seed-environment pair. The behavioral audit (separately documented) rules out temporal exploitation as the mechanism: zero exploitative outcomes on 504 binary switchpoint tests.

---

## Reproducing the validation scores

```bash
# From the repository root, with the package installed:
cd TR_validation/TR1_foundations
python TR1_validation_suite.py     # SLCD: 58/60 logarithmic, 46/60 power

cd ../TR2_trust
python TR2_validation_suite.py     # Renault-Nissan: 49/60

cd ../TR3_loyalty
python TR3_validation_suite.py     # Apache: 45/60

cd ../TR4_reciprocity
python TR4_validation_suite.py     # Apple App Store: 43/51
```

Each suite emits a JSON results file in its own directory (`tr1_results.json`, `tr2_results.json`, `tr3_results.json`, `validation_summary.json`) so that scores can be inspected programmatically.