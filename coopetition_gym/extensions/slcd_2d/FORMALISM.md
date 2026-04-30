# Formalism Notes, `SLCDAppropriation-v1ext0`

Supplementary to [README.md](README.md). Codifies three modeling choices that
reviewers should be able to cite directly.

## 1. Mean-field (commons-externality) dilution, NOT private penalty

The synergy-dilution factor is `(1 - β · p̄)` with `p̄ = (1/N) · Σ p_j`, applied
to the **shared** synergy term `α_i · γ · g(c)` only. The individual value
terms `θ · ln(1 + c_i)` are **not** scaled by dilution.

This is a deliberate modeling choice. An alternative formulation using
`(1 - β · p_i)` (private penalty) would collapse coopetition into bilateral
bargaining with a private cost on appropriation, losing the Tragedy-of-the-
Commons externality that makes coopetition a distinct phenomenon
(Brandenburger–Nalebuff 1996; Ritala–Hurmelinna-Laukkanen 2013; Gnyawali–Park
2011). Our choice instantiates: *when Samsung appropriates value from the
S-LCD joint venture, it degrades the relational substrate that makes the
joint synergy function productive for both firms*.

Regression test `tests/test_dilution_invariant.py` locks the invariant:
`f_i(c_i)` is unaffected by `p`; only `α_i · γ · g(c)` is.

## 2. Instantaneous dilution, NOT dynamic trust erosion

The dilution term `(1 - β · p̄)` acts on the **same-period** synergy. It does
not propagate to the next period's trust state. Dynamic erosion of trust
from appropriation is TR-2 territory; modeling it here would duplicate (or
worse, interfere with) the v1 trust dynamics.

This is a scalar-extension modeling choice for the appendix. A reviewer
looking for the full multi-period dynamic model will find it in TR-2 and
the main coopetition_gym trust mechanics. A hybrid term
`(1 - β · p̄ - β' · p_i)` with a small individual component for direct
appropriation costs (management attention, legal overhead) would be
defensible but adds a parameter without changing the qualitative storynot adopted.

## 3. Sensitivity scope, (η, β) at baseline-calibrated (κ, ξ)

The Tier 1.5 sensitivity sweep in `campaign_tier15.py` varies `(η, β)` over
a 5×5 grid while holding `(κ, ξ)` **fixed at the baseline-calibrated values**.
This is NOT a joint 4-parameter sensitivity analysis. The reported response
surface should be read as:

> "conditional sensitivity of the 2D dynamics to `(η, β)`, at calibrated
> `(κ, ξ)`"

A joint 4-parameter sweep would require 5⁴ = 625 cells × ≥20 seeds = >12,500
runs, Tier 2 scope. The Tier 1.5 design is defensible if (and only if) the
text makes this framing explicit. The `manifest.json` emitted by
`campaign_tier15.py` records `sensitivity_scope: "eta_beta_at_fixed_kappa_xi"`
for downstream tooling.

## 4. Calibration objectives, endpoint AND waypoint, reported side-by-side

Stage C of the Tier 1.5 campaign runs coordinate-descent calibration of
`(κ, ξ)` twice and reports the results side-by-side in
`dual_summary.json` with `|Δκ|`, `|Δξ|`, and an `agree_within_tol` flag
(tolerances 0.05 and 1.0 respectively).

### 4.1 Endpoint objective

Squared deviation of final trust and final appropriation from targets
`{0.0, 0.30}`. Calibrates the *attractor*.

### 4.2 Waypoint objective, historical anchors

Squared deviation at `{T/4, T/2, 3T/4, T}` for trust plus a final-
appropriation penalty. Calibrates the *trajectory* against the SLCD
2004–2011 historical arc. The waypoint positions (times) are fixed by
dated primary sources; the target *magnitudes* are registered under three
reviewer-proposed schedules selectable via
`--waypoint-target-set {A_flat_peak, A_rising, B_monotonic}`.

**Dated primary sources (time anchors, invariant across schedules):**

| Waypoint | Date | Anchor | Citation |
|---|---|---|---|
| `T=0`   | April 2004 | S-LCD Corporation JV founded | Samsung Display corporate history |
| `T/4`   | April 2005 | S-LCD begins shipment of 7th-generation TFT panels | Samsung Display corporate history |
| `T/2`   | August 2007 | S-LCD begins shipment of 8th-generation TFT panels; peak capacity 150K panels/month by April 2008 | Samsung Display corporate history |
| `3T/4`  | July 30, 2009 | Sony Corp 6-K filing announcing Sharp SDP joint venture (34% stake), first dated, SEC-filed supply-chain divergence | SEC EDGAR 0000313838-09-005328 |
| `T`     | December 26, 2011 | Joint Samsung-Sony announcement that Samsung acquires Sony's entire stake; formal dissolution | Sony/Samsung joint press release 2011-12-26 |

Qualitative arc: *coopetition as virtuous cycle* (Gnyawali & Park 2011).
Asymmetric appropriation, "Samsung has been able to appropriate a greater
share from the benefits" (Gnyawali & Park 2011, p. 657), motivates the
appropriation side of the objective.

### 4.3 Target magnitudes, two reviewer readings, both registered

Two friendly reviewers read the magnitudes at `T/4` and `T/2` differently,
and the disagreement is substantive enough to keep both versions available.
Both agree on `{3T/4, T}` (erosion onset, dissolution).

| Schedule | `T/4` | `T/2` | `3T/4` | `T` | View |
|---|---|---|---|---|---|
| `A_flat_peak` (**default**) | 0.85 | 0.85 | 0.30 | 0.0 | **Rev A, JV-internal**: Gen-7 ramp then Gen-8 capital commitment is peak cooperation within the JV. Flat-peak variant is used because it does not require TR-2 trust dynamics to produce rising trust above initialization (0.65). |
| `A_rising` | 0.85 | 0.90 | 0.30 | 0.0 | **Rev A, JV-internal, aspirational**: 8th-gen commitment pushes trust higher than Gen-7 ramp. Requires TR-2 to produce rising trust under steady cooperation; verified reachable in practice for `c > baseline=30`. |
| `B_monotonic` | 0.60 | 0.40 | 0.15 | 0.0 | **Rev B, firm-level**: Samsung's downstream TV market-share overtake of Sony during 2007–2008 is read as trust erosion already underway by `T/2`. |

**Why `A_flat_peak` is the default.** The env measures JV-internal trust via
TR-2's cooperation-vs-baseline signal. Samsung overtaking Sony in downstream
TV market share is a firm-level competitive fact, not an observable signal
in the JV-internal TR-2 state. Rev A's JV-internal view aligns with what
the env can actually represent. Flat-peak is the conservative variantit does not force TR-2 to exceed its initial trust value.

**Appendix language (recommended).** From Rev A's suggestion:

> "Waypoint targets are specified at {T/4, T/2, 3T/4, T} corresponding to
> the four dated phases of the Samsung–Sony S-LCD joint venture
> (2004–2011): 7th-generation ramp-up (Samsung Display, 2005), 8th-
> generation capacity peak (Samsung Display, 2007–2008), trust-erosion
> onset (Sony Corp SEC 6-K filing, 2009-07-30, announcing Sharp SDP joint
> venture at 34% stake), and formal dissolution (joint Sony-Samsung
> announcement, 2011-12-26). Trust trajectory shape follows the qualitative
> coopetition-as-virtuous-cycle characterization of Gnyawali & Park (2011).
> Target magnitudes reflect authors' inference from the cited primary and
> secondary sources; sensitivity to target magnitudes is examined in
> Tier 2 by running calibration under three registered schedules."

### 4.4 Agreement and divergence

- If endpoint and waypoint calibrations converge to the same `(κ, ξ)`
  within tolerance: trajectory and attractor are mutually consistent.
  Strong appendix story.
- If they diverge: the disagreement is itself a reportable finding and
  identifies an open question for Tier 2 (either the waypoint magnitudes
  are wrong, or the env dynamics cannot simultaneously fit attractor and
  trajectory, both informative).

## 5. Headline Tier 0 finding

The uncalibrated Tier 0 equilibrium `p* ≈ (0.07, 0.06)` is interior and low
but non-zero. Pure bilateral bargaining predicts either `p ≈ 0` (full trust
maintenance) or `p → 1` (dissolution). An interior low-but-nonzero `p*` is
the signature of *calculated appropriation under trust constraint*, exactly
what historians describe Samsung as doing 2008–2010. Report this as a
headline, not a footnote.

## Technical Reports

- TR-2: [Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics](https://arxiv.org/pdf/2510.24909) (arXiv:2510.24909)
