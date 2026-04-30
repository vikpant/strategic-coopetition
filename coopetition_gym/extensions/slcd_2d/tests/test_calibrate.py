"""Calibration module unit tests (no actual training — synthetic objective)."""

from __future__ import annotations

import numpy as np

from extensions.slcd_2d.calibrate import (
    DEFAULT_TARGETS,
    _parabolic_vertex,
    default_objective,
)


def test_parabolic_vertex_finds_interior_minimum():
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = (xs - 2.3) ** 2 + 0.1  # minimum at x=2.3
    v = _parabolic_vertex(xs, ys, bounds=(0.0, 4.0))
    assert abs(v - 2.3) < 0.05, f"vertex {v} far from 2.3"


def test_parabolic_vertex_clips_to_bounds():
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = xs  # monotone increasing, vertex would extrapolate negative
    v = _parabolic_vertex(xs, ys, bounds=(0.0, 4.0))
    assert 0.0 <= v <= 4.0


def test_parabolic_vertex_handles_degenerate_linear():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([1.0, 2.0, 3.0])  # a = 0 after quadratic fit (near-linear)
    v = _parabolic_vertex(xs, ys, bounds=(0.0, 2.0))
    assert 0.0 <= v <= 2.0


def test_default_objective_zero_at_targets():
    stats = {
        "eval_final_trust_mean": DEFAULT_TARGETS["final_trust_mean"],
        "eval_final_appropriation_mean": [DEFAULT_TARGETS["final_appropriation_mean"]],
    }
    obj = default_objective(stats)
    assert abs(obj) < 1e-12


def test_default_objective_monotone_in_deviation():
    target = DEFAULT_TARGETS
    s1 = {
        "eval_final_trust_mean": target["final_trust_mean"] + 0.1,
        "eval_final_appropriation_mean": [target["final_appropriation_mean"]],
    }
    s2 = {
        "eval_final_trust_mean": target["final_trust_mean"] + 0.3,
        "eval_final_appropriation_mean": [target["final_appropriation_mean"]],
    }
    assert default_objective(s1) < default_objective(s2)
