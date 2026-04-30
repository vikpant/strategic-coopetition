"""Training-algorithm adapters for the 2D SLCD extension.

Delegates to the v1 implementations in ``experiments.algorithms`` — we
deliberately do not re-implement SB3 wiring, checkpointing, or metric capture.
This keeps the extension thin and ensures parity with the main campaign.

Exposes a small algorithm registry (name -> constructor) used by
``campaign_tier1.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .env import SLCDAppropriationEnv
from .oracle import AppropriationOracle


def _build_ippo(env, device: str, seed: int, **kwargs):
    from experiments.algorithms import IndependentPPO
    return IndependentPPO(env, device=device, seed=seed, **kwargs)


def _build_isac(env, device: str, seed: int, **kwargs):
    from experiments.algorithms import IndependentSAC
    return IndependentSAC(env, device=device, seed=seed, **kwargs)


def _build_ia2c(env, device: str, seed: int, **kwargs):
    from experiments.algorithms import IndependentA2C
    return IndependentA2C(env, device=device, seed=seed, **kwargs)


def _build_mappo(env, device: str, seed: int, **kwargs):
    from experiments.algorithms import MAPPO
    return MAPPO(env, device=device, seed=seed, **kwargs)


def _build_maddpg(env, device: str, seed: int, **kwargs):
    from experiments.algorithms import MADDPG
    return MADDPG(env, device=device, seed=seed, **kwargs)


def _build_matd3(env, device: str, seed: int, **kwargs):
    from experiments.algorithms import MATD3
    return MATD3(env, device=device, seed=seed, **kwargs)


def _build_masac(env, device: str, seed: int, **kwargs):
    from experiments.algorithms import MASAC
    return MASAC(env, device=device, seed=seed, **kwargs)


def _build_oracle(env, device: str, seed: int, **kwargs):
    return AppropriationOracle(env, device=device, seed=seed, **kwargs)


ALGORITHM_REGISTRY: Dict[str, Callable[..., Any]] = {
    "IPPO": _build_ippo,
    "ISAC": _build_isac,
    "IA2C": _build_ia2c,
    "MAPPO": _build_mappo,
    "MADDPG": _build_maddpg,
    "MATD3": _build_matd3,
    "MASAC": _build_masac,
    "Oracle_Appropriation": _build_oracle,
}

# Algorithms that perform better on CPU than GPU for this env's small MLP
# (128x128). SB3 explicitly warns about on-policy PPO/A2C with MlpPolicy being
# GPU-suboptimal; empirical Tier 1.5 trial on 5090 showed IPPO at ~30 min/run
# on GPU vs ~5 min/run on CPU. The orchestrator routes these to CPU regardless
# of --num-gpus; off-policy and centralized-critic algos stay on GPU.
CPU_PREFERRED_ALGORITHMS = frozenset({"IPPO", "IA2C"})


def prefers_cpu(algorithm: str) -> bool:
    return algorithm in CPU_PREFERRED_ALGORITHMS


def list_algorithms() -> list[str]:
    return sorted(ALGORITHM_REGISTRY.keys())


def build_algorithm(name: str, env: SLCDAppropriationEnv, device: str = "cpu",
                    seed: int = 0, **kwargs):
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm {name!r}. Available: {list_algorithms()}")
    return ALGORITHM_REGISTRY[name](env, device=device, seed=seed, **kwargs)
