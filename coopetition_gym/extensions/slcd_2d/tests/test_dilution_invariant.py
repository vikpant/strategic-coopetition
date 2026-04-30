"""Lock the invariant that (1 - beta * p_bar) dilutes synergy only, not individual value.

Reviewer A's concern: 'If your implementation applies (1 - beta * p_bar) to V as a
whole, that's a bug - the individual f_i should be unaffected. If it applies only
to g, you're fine.'

The test below fixes c and varies only p. If the implementation is correct, the
p-dependent reward contribution must be independent of the individual f_i(c_i)
terms. We verify by decomposing the payoff analytically against the formula.
"""

from __future__ import annotations

import numpy as np

from extensions.slcd_2d.utility import (
    AppropriationParameters,
    compute_2d_private_payoffs,
)


def _expected_payoff_term_by_term(c, p, endowments, alpha, theta, gamma, params):
    """Ground-truth formula (spelled out), for comparison."""
    n = len(c)
    individual_value = theta * np.log1p(c)
    synergy = gamma * np.prod(np.maximum(c, 0.0)) ** (1.0 / n)
    p_bar = float(np.mean(p))
    return (
        (endowments - c - params.kappa * p)
        + individual_value
        + alpha * synergy * (1.0 - params.beta * p_bar)
        + params.eta * p * synergy
        - params.xi * p ** 2
    )


def test_individual_value_independent_of_p():
    """At fixed c, vary p. The change in payoff must NOT depend on individual_value.

    Formally: pi(c, p1) - pi(c, p2) should match the synergy+cost terms only.
    """
    params = AppropriationParameters()
    c = np.array([50.0, 50.0], dtype=np.float64)
    endowments = np.array([100.0, 100.0])
    alpha = np.array([0.55, 0.45])
    theta, gamma = 20.0, 0.65

    p1 = np.array([0.0, 0.0])
    p2 = np.array([0.5, 0.5])

    pi1 = compute_2d_private_payoffs(c, p1, endowments, alpha, theta, gamma, params)
    pi2 = compute_2d_private_payoffs(c, p2, endowments, alpha, theta, gamma, params)

    # Expected difference: only cost, dilution, capture, convex terms change
    synergy = gamma * np.prod(c) ** 0.5
    p_bar1 = p1.mean()
    p_bar2 = p2.mean()
    expected_diff = (
        (-params.kappa * p2 + params.kappa * p1)
        + alpha * synergy * ((1.0 - params.beta * p_bar2) - (1.0 - params.beta * p_bar1))
        + params.eta * (p2 - p1) * synergy
        - params.xi * (p2 ** 2 - p1 ** 2)
    )
    actual_diff = pi2 - pi1
    assert np.allclose(actual_diff, expected_diff, atol=1e-9), (
        f"Expected diff {expected_diff}, actual {actual_diff}. "
        "This suggests individual_value is being scaled by (1 - beta * p_bar)."
    )


def test_individual_value_component_extractable_at_p_zero():
    """At p=0, the payoff equals v1 exactly: (e - c) + theta*log(1+c) + alpha*gamma*g(c).

    If individual_value were being diluted, this would disagree with v1 at p=0.
    """
    params = AppropriationParameters()
    c = np.array([30.0, 70.0], dtype=np.float64)
    p = np.zeros(2)
    endowments = np.array([100.0, 100.0])
    alpha = np.array([0.55, 0.45])
    theta, gamma = 20.0, 0.65

    pi = compute_2d_private_payoffs(c, p, endowments, alpha, theta, gamma, params)

    synergy = gamma * np.prod(c) ** 0.5
    expected = (endowments - c) + theta * np.log1p(c) + alpha * synergy
    assert np.allclose(pi, expected, atol=1e-9), (
        f"At p=0, payoff must equal v1 formula exactly. Got {pi}, expected {expected}."
    )


def test_dilution_zero_when_beta_zero():
    """beta=0 disables the commons externality; only cost/capture/convex terms matter."""
    params_no_dil = AppropriationParameters(kappa=0.5, beta=0.0, eta=0.4, xi=15.0)
    c = np.array([40.0, 60.0], dtype=np.float64)
    endowments = np.array([100.0, 100.0])
    alpha = np.array([0.55, 0.45])
    theta, gamma = 20.0, 0.65

    p_low = np.array([0.1, 0.2])
    p_high = np.array([0.8, 0.9])

    pi_low = compute_2d_private_payoffs(c, p_low, endowments, alpha, theta, gamma, params_no_dil)
    pi_high = compute_2d_private_payoffs(c, p_high, endowments, alpha, theta, gamma, params_no_dil)
    diff = pi_high - pi_low

    synergy = gamma * np.prod(c) ** 0.5
    expected_diff = (
        -params_no_dil.kappa * (p_high - p_low)
        + params_no_dil.eta * (p_high - p_low) * synergy
        - params_no_dil.xi * (p_high ** 2 - p_low ** 2)
    )
    assert np.allclose(diff, expected_diff, atol=1e-9), (
        "With beta=0, the mean-field dilution path must contribute zero to the diff."
    )
