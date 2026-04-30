"""Oracle that solves the 2D (c*, p*) Nash equilibrium numerically.

Uses iterated best response with scipy's bounded scalar minimizer applied
coordinate-wise (c_i then p_i). Returns the equilibrium action vector in the
Gymnasium-flat layout [c_0, p_0, c_1, p_1, ...].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from .utility import (
    AppropriationParameters,
    compute_2d_integrated_utilities,
    compute_2d_private_payoffs,
)


@dataclass
class AppropriationEquilibrium:
    cooperation: NDArray[np.floating]
    appropriation: NDArray[np.floating]
    utilities: NDArray[np.floating]
    iterations: int
    converged: bool

    def to_flat_action(self) -> np.ndarray:
        n = len(self.cooperation)
        flat = np.empty(2 * n, dtype=np.float32)
        flat[0::2] = self.cooperation.astype(np.float32)
        flat[1::2] = self.appropriation.astype(np.float32)
        return flat


def _utility_of_agent(
    i: int,
    c: NDArray[np.floating],
    p: NDArray[np.floating],
    endowments: NDArray[np.floating],
    alpha: NDArray[np.floating],
    D: NDArray[np.floating],
    theta: float,
    gamma: float,
    appr_params: AppropriationParameters,
    trust_matrix: Optional[NDArray[np.floating]],
) -> float:
    utils = compute_2d_integrated_utilities(
        c=c,
        p=p,
        endowments=endowments,
        alpha=alpha,
        D=D,
        theta=theta,
        gamma=gamma,
        appr_params=appr_params,
        trust_matrix=trust_matrix,
    )
    return float(utils[i])


def solve_appropriation_equilibrium(
    endowments: NDArray[np.floating],
    alpha: NDArray[np.floating],
    D: NDArray[np.floating],
    theta: float,
    gamma: float,
    appr_params: AppropriationParameters,
    trust_matrix: Optional[NDArray[np.floating]] = None,
    max_iterations: int = 200,
    tolerance: float = 1e-5,
    initial_c: Optional[NDArray[np.floating]] = None,
    initial_p: Optional[NDArray[np.floating]] = None,
) -> AppropriationEquilibrium:
    """Iterated best response over (c, p) for each agent."""
    n = len(endowments)
    c = (endowments * 0.5) if initial_c is None else np.asarray(initial_c, dtype=np.float64).copy()
    p = np.full(n, 0.1, dtype=np.float64) if initial_p is None else np.asarray(initial_p, dtype=np.float64).copy()

    converged = False
    for iteration in range(max_iterations):
        c_old = c.copy()
        p_old = p.copy()

        for i in range(n):
            def neg_util_c(c_i: float, i=i) -> float:
                c_trial = c.copy()
                c_trial[i] = c_i
                return -_utility_of_agent(
                    i, c_trial, p, endowments, alpha, D,
                    theta, gamma, appr_params, trust_matrix,
                )

            res_c = minimize_scalar(
                neg_util_c, bounds=(0.0, float(endowments[i])), method="bounded"
            )
            c[i] = float(res_c.x)

            def neg_util_p(p_i: float, i=i) -> float:
                p_trial = p.copy()
                p_trial[i] = p_i
                return -_utility_of_agent(
                    i, c, p_trial, endowments, alpha, D,
                    theta, gamma, appr_params, trust_matrix,
                )

            res_p = minimize_scalar(
                neg_util_p, bounds=(0.0, 1.0), method="bounded"
            )
            p[i] = float(res_p.x)

        delta = max(np.max(np.abs(c - c_old)), np.max(np.abs(p - p_old)))
        if delta < tolerance:
            converged = True
            break

    utilities = compute_2d_integrated_utilities(
        c=c, p=p,
        endowments=endowments, alpha=alpha, D=D,
        theta=theta, gamma=gamma, appr_params=appr_params,
        trust_matrix=trust_matrix,
    )

    return AppropriationEquilibrium(
        cooperation=c,
        appropriation=p,
        utilities=utilities,
        iterations=iteration + 1,
        converged=converged,
    )


class AppropriationOracle:
    """Deterministic policy returning the pre-computed 2D equilibrium action.

    Matches the minimal oracle interface used in experiments/algorithms.py
    (a ``predict(obs, deterministic=True)`` method returning an action and
    a ``state`` placeholder).
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs) -> None:
        self.n_agents = int(getattr(env, "n_agents"))
        self.endowments = np.asarray(getattr(env, "endowments"), dtype=np.float64)
        self.alpha = np.asarray(getattr(env, "alpha"), dtype=np.float64)
        self.D = np.asarray(getattr(env, "D"), dtype=np.float64)
        self.theta = float(env.config.value_params.theta)
        self.gamma = float(env.config.value_params.gamma)
        appr_params = getattr(env, "appr_params", None)
        if appr_params is None:
            from .utility import AppropriationParameters
            appr_params = AppropriationParameters()
        self.appr_params = appr_params

        self.equilibrium = solve_appropriation_equilibrium(
            endowments=self.endowments,
            alpha=self.alpha,
            D=self.D,
            theta=self.theta,
            gamma=self.gamma,
            appr_params=self.appr_params,
        )
        self._action = self.equilibrium.to_flat_action()

    def predict(self, obs, deterministic: bool = True):
        return self._action.copy(), None
