"""2D utility for SLCD appropriation extension.

Extends the TR-1 integrated-utility formulation with a second per-agent action
dimension p_i (appropriation effort). When p_i is identically zero for all
agents, the 2D utility reduces to the v1 integrated utility bit-exact.

Formulation
-----------
Let c_i in [0, e_i] be the TR-1 cooperation level and p_i in [0, 1] be
appropriation effort. Write p_bar = (1/N) * sum_j p_j. With v1 synergistic
surplus S(c) = gamma * (prod_i c_i)^(1/N), the 2D private payoff is:

    pi_i^{2D}(c, p) = (e_i - c_i - kappa * p_i)          # cost
                      + theta * ln(1 + c_i)               # TR-1 individual value
                      + alpha_i * S(c) * (1 - beta * p_bar)  # diluted synergy
                      + eta * p_i * S(c)                  # private capture
                      - xi * p_i ** 2                     # convex cost

The 2D integrated utility follows the v1 form:

    U_i^{2D}(c, p) = pi_i^{2D}(c, p) + sum_{j != i} w_ij * pi_j^{2D}(c, p)

where w_ij = T_ij * D_ij if trust is enabled else D_ij. Backward compatibility
(p == 0 => v1 U_i) is enforced by tests/test_backward_compat.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AppropriationParameters:
    """Parameters of the second action dimension. See calibration.json."""

    kappa: float = 0.5
    beta: float = 0.6
    eta: float = 0.4
    xi: float = 15.0

    def __post_init__(self) -> None:
        for name in ("kappa", "beta", "eta", "xi"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.beta > 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")


def compute_2d_private_payoffs(
    c: NDArray[np.floating],
    p: NDArray[np.floating],
    endowments: NDArray[np.floating],
    alpha: NDArray[np.floating],
    theta: float,
    gamma: float,
    appr_params: AppropriationParameters,
) -> NDArray[np.floating]:
    """Compute pi_i^{2D} for all agents."""
    c = np.asarray(c, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = len(c)

    individual_value = theta * np.log1p(c)
    synergy = gamma * np.prod(np.maximum(c, 0.0)) ** (1.0 / n) if n > 0 else 0.0
    p_bar = float(np.mean(p))

    payoffs = (
        (endowments - c - appr_params.kappa * p)
        + individual_value
        + alpha * synergy * (1.0 - appr_params.beta * p_bar)
        + appr_params.eta * p * synergy
        - appr_params.xi * p ** 2
    )
    return payoffs


def compute_2d_integrated_utilities(
    c: NDArray[np.floating],
    p: NDArray[np.floating],
    endowments: NDArray[np.floating],
    alpha: NDArray[np.floating],
    D: NDArray[np.floating],
    theta: float,
    gamma: float,
    appr_params: AppropriationParameters,
    trust_matrix: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Compute U_i^{2D} for all agents.

    Parameters
    ----------
    trust_matrix
        Optional NxN trust matrix T. If provided, partner weighting becomes
        w_ij = T_ij * D_ij (matching v1 `compute_integrated_utility` semantics
        for trust-enabled environments).
    """
    payoffs = compute_2d_private_payoffs(
        c, p, endowments, alpha, theta, gamma, appr_params
    )
    n = len(payoffs)

    weights = D.astype(np.float64).copy()
    if trust_matrix is not None:
        weights = weights * trust_matrix.astype(np.float64)
    np.fill_diagonal(weights, 0.0)

    utilities = payoffs + weights @ payoffs
    return utilities
