"""Tier 1 sensitivity sweep definition.

Defines the (eta, beta) × seeds × algorithms grid that Tier 1 explores. The
choice of eta, beta as the swept axes (and kappa, xi held at defaults) reflects
a closed-form marginal analysis: d U_i / d p_i has leading terms eta*S(c) and
-alpha_i*S(c)*beta/N. These two parameters dominate the interior-Nash location.

Grid size: 5 x 5 = 25 (eta, beta) cells + 1 baseline cell at default calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, List, Tuple

from .utility import AppropriationParameters


DEFAULT_ETAS: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)
DEFAULT_BETAS: Tuple[float, ...] = (0.30, 0.45, 0.60, 0.75, 0.90)


@dataclass(frozen=True)
class SweepCell:
    cell_id: str
    eta: float
    beta: float
    is_baseline: bool

    def to_params(self, kappa: float, xi: float) -> AppropriationParameters:
        return AppropriationParameters(kappa=kappa, beta=self.beta, eta=self.eta, xi=xi)


def tier1_cells(
    etas: Tuple[float, ...] = DEFAULT_ETAS,
    betas: Tuple[float, ...] = DEFAULT_BETAS,
    baseline_eta: float = 0.40,
    baseline_beta: float = 0.60,
) -> List[SweepCell]:
    """Return the Tier 1 (eta, beta) grid plus a labelled baseline cell."""
    cells: List[SweepCell] = []
    for eta, beta in product(etas, betas):
        is_baseline = (
            abs(eta - baseline_eta) < 1e-9 and abs(beta - baseline_beta) < 1e-9
        )
        cells.append(
            SweepCell(
                cell_id=f"eta{eta:.2f}_beta{beta:.2f}",
                eta=float(eta),
                beta=float(beta),
                is_baseline=is_baseline,
            )
        )
    return cells


def enumerate_runs(
    cells: List[SweepCell],
    algorithms: List[str],
    seeds: List[int],
    reward_types: List[str],
) -> Iterator[dict]:
    """Yield one dict per (cell, algo, seed, reward_type) combination."""
    for cell in cells:
        for algo in algorithms:
            for seed in seeds:
                for reward_type in reward_types:
                    yield {
                        "cell_id": cell.cell_id,
                        "eta": cell.eta,
                        "beta": cell.beta,
                        "algorithm": algo,
                        "seed": seed,
                        "reward_type": reward_type,
                        "is_baseline": cell.is_baseline,
                    }
