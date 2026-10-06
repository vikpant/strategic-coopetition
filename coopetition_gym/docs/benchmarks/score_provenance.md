# Case-study score provenance

Reviewed against public source artifacts on 2026-10-06. This register records
conflicting evidence; it does not select new canonical scores. Rubric scores
measure case-specific correspondence/plausibility, not held-out prediction
accuracy. Numerators and denominators from different rubrics are not interchangeable.

The source references below are repository-relative. Historical documentation
values refer to commit `8ac7db4c0ac6e2859a1fb382e01c538edcdd03a0`
(tag `v1.0.7`); saved datasets and validation code are unchanged by this repair.

| Case | Recorded value | Source artifact | Evidence and status |
| --- | --- | --- | --- |
| Samsung–Sony S-LCD | 58/60 logarithmic; 46/60 power | `TR_validation/TR1_foundations/README.md`, `tr1_results.json` | Source-specific rubric alternatives. Keep their specification attached. |
| Samsung–Sony S-LCD | 59/60 (98.3%) | Historical `papers/neurips_ed_2026/croissant.json` calibration description | Metadata claim conflicts with the public validation artifact; not adopted. |
| Renault–Nissan | 49/60 | `TR_validation/TR2_trust/README.md`; `enhanced_validation_results/renault_nissan_enhanced_results.json` | Source-specific result; the old top-level validation README incorrectly assigned this case to TR-1. |
| Apache HTTP Server | 45/60 | Historical root `README.md` and `coopetition_gym/docs/benchmarks/case_study_validation.md` | Documentation claim; no matching saved result is established here. |
| Apache HTTP Server | 52/60 (86.7%) | `TR_validation/TR3_loyalty/README.md` and validation script narrative | Competing documented claim. Canonical score remains unresolved pending a matched rubric/run artifact. |
| Apple iOS App Store | 43/51 | `TR_validation/TR4_reciprocity/TR4_validation_output/apple_ios_results.json` (validation_score); historical root README | Saved result with its own denominator. |
| Apple iOS App Store | 48/55 (87.3%) | `TR_validation/TR4_reciprocity/README.md` and validation script narrative | Different documented rubric/denominator. Does not supersede saved 43/51 without a provenance decision. |

To reconcile a case, identify the intended report revision, rubric matrix,
parameter set, exact source commit, command and output checksum together. Preserve
the older artifact and publish a separate reconciliation record. Running a suite
with new code does not establish which historical claim was intended.

The current package provides case-informed environments; none of these four
scores is an RL benchmark return. Do not substitute case rubric scores for
algorithm evaluation or convert a documentation conflict into an averaged score.
