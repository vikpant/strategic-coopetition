"""Multi-waypoint calibration objective."""

from __future__ import annotations

import numpy as np

from extensions.slcd_2d.calibrate import (
    DEFAULT_WAYPOINT_TARGETS,
    default_objective,
    waypoint_objective,
)


def _make_stats(trust_curve, final_appr):
    return {
        "eval_mean_trust_curve": trust_curve,
        "eval_final_trust_mean": float(trust_curve[-1]),
        "eval_final_appropriation_mean": [final_appr],
    }


def test_waypoint_zero_at_perfect_trajectory():
    """Trajectory hitting all 4 waypoints exactly + appropriate p -> objective = 0."""
    T = 40
    curve = np.linspace(1.0, 0.0, T + 1)  # replaced per-waypoint below
    idx_e, idx_m, idx_l = T // 4, T // 2, (3 * T) // 4
    curve = list(curve)
    curve[idx_e] = DEFAULT_WAYPOINT_TARGETS["trust_early"]
    curve[idx_m] = DEFAULT_WAYPOINT_TARGETS["trust_mid"]
    curve[idx_l] = DEFAULT_WAYPOINT_TARGETS["trust_late"]
    curve[-1] = DEFAULT_WAYPOINT_TARGETS["trust_final"]

    stats = _make_stats(curve, DEFAULT_WAYPOINT_TARGETS["appr_final"])
    loss = waypoint_objective(stats)
    assert loss < 1e-9, f"Expected 0 at perfect trajectory, got {loss}"


def test_waypoint_penalizes_wrong_midpoint():
    """Two trajectories with same endpoint but different mid-points -> different loss."""
    T = 40
    idx_m = T // 2
    tgt = DEFAULT_WAYPOINT_TARGETS

    good = [1.0] * (T + 1)
    good[T // 4] = tgt["trust_early"]
    good[idx_m] = tgt["trust_mid"]
    good[(3 * T) // 4] = tgt["trust_late"]
    good[-1] = tgt["trust_final"]

    bad = good.copy()
    bad[idx_m] = tgt["trust_mid"] + 0.4  # wrong at mid-point only

    L_good = waypoint_objective(_make_stats(good, tgt["appr_final"]))
    L_bad = waypoint_objective(_make_stats(bad, tgt["appr_final"]))
    assert L_bad > L_good, "Wrong mid-point should cost more than zero"
    assert L_bad - L_good > 0.1, f"Penalty too weak: delta={L_bad-L_good}"


def test_endpoint_rewards_wrong_trajectory_with_right_endpoint():
    """Endpoint objective is indifferent to mid-trajectory — the motivation for waypoint."""
    T = 40
    good_at_endpoint_only = [0.0] * (T + 1)  # immediate collapse
    good_at_endpoint_only[-1] = 0.0

    realistic = [0.65] * (T + 1)
    realistic[-1] = 0.0

    L_instant = default_objective(_make_stats(good_at_endpoint_only, 0.30))
    L_realistic = default_objective(_make_stats(realistic, 0.30))
    assert abs(L_instant - L_realistic) < 1e-9, (
        "Endpoint objective should NOT distinguish these — that's the reviewer's complaint."
    )

    # Waypoint DOES distinguish
    L_instant_wp = waypoint_objective(_make_stats(good_at_endpoint_only, 0.30))
    L_realistic_wp = waypoint_objective(_make_stats(realistic, 0.30))
    assert L_instant_wp > L_realistic_wp, "Waypoint must penalize immediate collapse"


def test_waypoint_falls_back_without_curve():
    """If curve is missing, waypoint gracefully degrades to endpoint objective."""
    stats = {
        "eval_final_trust_mean": 0.0,
        "eval_final_appropriation_mean": [0.30],
    }
    L = waypoint_objective(stats)
    L_endpoint = default_objective(stats)
    assert abs(L - L_endpoint) < 1e-12


def test_registered_target_sets_present():
    """All three reviewer-proposed schedules are registered and well-formed."""
    from extensions.slcd_2d.calibrate import (
        WAYPOINT_TARGET_REGISTRY,
        DEFAULT_WAYPOINT_TARGETS,
        WAYPOINT_TARGETS_A_FLAT_PEAK,
    )
    assert set(WAYPOINT_TARGET_REGISTRY) == {"A_flat_peak", "A_rising", "B_monotonic"}
    required_keys = {"trust_early", "trust_mid", "trust_late", "trust_final", "appr_final"}
    for name, schedule in WAYPOINT_TARGET_REGISTRY.items():
        assert set(schedule.keys()) == required_keys, f"{name} missing keys"
        assert schedule["trust_final"] == 0.0, f"{name} must dissolve at T"
    assert DEFAULT_WAYPOINT_TARGETS is WAYPOINT_TARGETS_A_FLAT_PEAK, (
        "Default must be A_flat_peak per reviewer-adjudication decision"
    )


def test_reviewer_schedules_disagree_at_mid():
    """Verify the disagreement the reviewers actually have: at T/2."""
    from extensions.slcd_2d.calibrate import WAYPOINT_TARGET_REGISTRY
    rev_a = WAYPOINT_TARGET_REGISTRY["A_flat_peak"]["trust_mid"]
    rev_b = WAYPOINT_TARGET_REGISTRY["B_monotonic"]["trust_mid"]
    # Rev A says trust is high at T/2 (peak cooperation); Rev B says declining
    assert rev_a > rev_b, "Rev A's T/2 should be higher than Rev B's (documented disagreement)"
    assert rev_a - rev_b >= 0.4, "Disagreement should be material (>0.4)"


def test_flat_peak_reachable_by_tr2_under_steady_cooperation():
    """TR-2 with SLCD parameters must be able to reach trust ~0.85 under steady
    high cooperation. Otherwise the A_flat_peak schedule is physically unreachable
    by the env dynamics and we would need to switch to B_monotonic.
    """
    from extensions.slcd_2d import SLCDAppropriationEnv
    import numpy as np
    env = SLCDAppropriationEnv(max_steps=40)
    env.reset(seed=0)
    c = 50.0  # well above baseline=30 -> positive trust signal every step
    action = np.array([c, 0.0, c, 0.0], dtype=np.float32)
    trust_at_10 = None
    for step in range(20):
        _, _, _, _, info = env.step(action)
        if step == 9:
            trust_at_10 = float(info["mean_trust"])
    assert trust_at_10 is not None and trust_at_10 >= 0.80, (
        f"TR-2 should be able to reach mean_trust>=0.80 under steady c=50 by step 10; "
        f"got {trust_at_10}. If this fails, switch default to B_monotonic."
    )
