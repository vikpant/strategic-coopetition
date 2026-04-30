"""Nash interior-equilibrium test for the 2D formulation.

At the calibrated (kappa, beta, eta, xi), the 2D best-response iteration must
converge to an interior point — not a corner solution at p=0 or p=1. A corner
at p=0 would make the appropriation dimension trivial; a corner at p=1 would
mean appropriation always dominates.
"""

from __future__ import annotations

import numpy as np

from extensions.slcd_2d import (
    AppropriationOracle,
    SLCDAppropriationEnv,
    solve_appropriation_equilibrium,
    load_default_appropriation_params,
)


def test_equilibrium_converges():
    env = SLCDAppropriationEnv()
    eq = solve_appropriation_equilibrium(
        endowments=np.asarray(env.endowments, dtype=np.float64),
        alpha=np.asarray(env.alpha, dtype=np.float64),
        D=np.asarray(env.D, dtype=np.float64),
        theta=float(env.value_params.theta),
        gamma=float(env.value_params.gamma),
        appr_params=env.appr_params,
    )
    assert eq.converged, f"Did not converge in {eq.iterations} iterations"


def test_equilibrium_is_interior():
    env = SLCDAppropriationEnv()
    eq = solve_appropriation_equilibrium(
        endowments=np.asarray(env.endowments, dtype=np.float64),
        alpha=np.asarray(env.alpha, dtype=np.float64),
        D=np.asarray(env.D, dtype=np.float64),
        theta=float(env.value_params.theta),
        gamma=float(env.value_params.gamma),
        appr_params=env.appr_params,
    )
    assert np.all(eq.cooperation > 1.0), f"c* too close to 0: {eq.cooperation}"
    assert np.all(eq.cooperation < 99.0), f"c* too close to e_i: {eq.cooperation}"
    assert np.all(eq.appropriation > 1e-3), f"p* pinned at 0: {eq.appropriation}"
    assert np.all(eq.appropriation < 0.999), f"p* pinned at 1: {eq.appropriation}"


def test_oracle_action_is_valid():
    env = SLCDAppropriationEnv()
    oracle = AppropriationOracle(env)
    action, _ = oracle.predict(obs=None, deterministic=True)
    assert action.shape == env.action_space.shape
    assert env.action_space.contains(action.astype(np.float32)), (
        f"Oracle action {action} out of bounds"
    )


def test_oracle_utility_beats_zero_appropriation():
    """Oracle should not pick p=0 if an interior optimum exists."""
    env = SLCDAppropriationEnv()
    oracle = AppropriationOracle(env)
    action_oracle, _ = oracle.predict(obs=None)

    env.reset(seed=1)
    _, r_oracle, _, _, _ = env.step(action_oracle)

    p_zero_action = action_oracle.copy()
    p_zero_action[1::2] = 0.0
    env.reset(seed=1)
    _, r_zero, _, _, _ = env.step(p_zero_action)

    assert r_oracle.sum() >= r_zero.sum() - 1e-3, (
        f"Oracle utility {r_oracle.sum()} < p=0 utility {r_zero.sum()}"
    )
