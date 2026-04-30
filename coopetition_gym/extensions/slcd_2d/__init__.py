"""2D SLCD extension (post-v1 sanity check).

This package is NOT part of `coopetition_gym`. It is an opt-in extension
that subclasses `SLCDEnv` to add a second action dimension (appropriation
effort) per agent.

Usage
-----
    from extensions.slcd_2d import make_slcd_2d
    env = make_slcd_2d()

The extension registry is local to this module — the base-package registry
in ``coopetition_gym.envs`` is deliberately left untouched.
"""

from __future__ import annotations

from typing import Any, Dict

from .env import SLCDAppropriationEnv, load_default_appropriation_params
from .oracle import (
    AppropriationEquilibrium,
    AppropriationOracle,
    solve_appropriation_equilibrium,
)
from .utility import (
    AppropriationParameters,
    compute_2d_integrated_utilities,
    compute_2d_private_payoffs,
)

ENV_ID = "SLCDAppropriation-v1ext0"

_EXTENSION_REGISTRY: Dict[str, Any] = {
    ENV_ID: SLCDAppropriationEnv,
}


def list_extension_envs() -> list[str]:
    return list(_EXTENSION_REGISTRY.keys())


def make_slcd_2d(**kwargs: Any) -> SLCDAppropriationEnv:
    """Construct the 2D SLCD environment with default calibration."""
    return SLCDAppropriationEnv(**kwargs)


__all__ = [
    "ENV_ID",
    "SLCDAppropriationEnv",
    "AppropriationOracle",
    "AppropriationEquilibrium",
    "AppropriationParameters",
    "compute_2d_integrated_utilities",
    "compute_2d_private_payoffs",
    "load_default_appropriation_params",
    "solve_appropriation_equilibrium",
    "list_extension_envs",
    "make_slcd_2d",
]
