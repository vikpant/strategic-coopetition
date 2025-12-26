"""
================================================================================
COOPETITION-GYM: Multi-Agent Reinforcement Learning for Strategic Coopetition
================================================================================

A Gymnasium and PettingZoo compatible library for studying coopetitive dynamics
in multi-agent systems, based on validated game-theoretic research.

Quick Start (Gymnasium API):
>>> import coopetition_gym
>>> env = coopetition_gym.make("TrustDilemma-v0")
>>> obs, info = env.reset(seed=42)
>>> obs, rewards, terminated, truncated, info = env.step([50.0, 50.0])

Quick Start (PettingZoo Parallel API):
>>> env = coopetition_gym.make_parallel("TrustDilemma-v0")
>>> observations, infos = env.reset(seed=42)
>>> actions = {agent: env.action_space(agent).sample() for agent in env.agents}
>>> observations, rewards, terminations, truncations, infos = env.step(actions)

Authors: Vik Pant, Eric Yu - University of Toronto
License: MIT
================================================================================
"""

__version__ = "0.2.0"
__author__ = "Vik Pant, Eric Yu"

from .core import (
    # TR-1: Value Functions
    ValueSpecification, ValueFunctionParameters, individual_value, 
    synergy_function, total_value, create_slcd_parameters,
    # TR-1: Interdependence
    InterdependenceMatrix, create_slcd_interdependence,
    create_renault_nissan_interdependence, create_symmetric_interdependence,
    # TR-2: Trust Dynamics
    TrustParameters, TrustState, TrustDynamicsModel,
    # Equilibrium
    PayoffParameters, compute_rewards, solve_equilibrium,
    # TR-3: Collective Action (skeleton)
    CollectiveActionParameters, CollectiveActionState, CollectiveActionModel,
    # TR-4: Reciprocity (skeleton)
    ReciprocityParameters, ReciprocityState, ReciprocityModel,
)

from .envs import (
    # Base classes
    CoopetitionEnv, EnvironmentConfig, AbstractCoopetitionEnv,
    # Environments
    TrustDilemmaEnv, PartnerHoldUpEnv,
    PlatformEcosystemEnv, DynamicPartnerSelectionEnv,
    RecoveryRaceEnv, SynergySearchEnv,
    SLCDEnv, RenaultNissanEnv,
    CooperativeNegotiationEnv, ReputationMarketEnv,
    # Wrappers (v0.2.0)
    ObservationConfig, CoopetitionParallelEnv, CoopetitionAECEnv,
    # Factory functions
    make, make_parallel, make_aec, list_environments,
)


def version():
    return __version__


def info():
    print(f"COOPETITION-GYM v{__version__} | {len(list_environments())} environments")
    print("APIs: Gymnasium (make), PettingZoo Parallel (make_parallel), PettingZoo AEC (make_aec)")


__all__ = [
    "__version__", "version", "info",
    # TR-1
    "ValueSpecification", "ValueFunctionParameters", "individual_value",
    "synergy_function", "total_value", "create_slcd_parameters",
    "InterdependenceMatrix", "create_slcd_interdependence",
    "create_renault_nissan_interdependence", "create_symmetric_interdependence",
    # TR-2
    "TrustParameters", "TrustState", "TrustDynamicsModel",
    "PayoffParameters", "compute_rewards", "solve_equilibrium",
    # TR-3 (skeleton)
    "CollectiveActionParameters", "CollectiveActionState", "CollectiveActionModel",
    # TR-4 (skeleton)
    "ReciprocityParameters", "ReciprocityState", "ReciprocityModel",
    # Environments
    "CoopetitionEnv", "EnvironmentConfig", "AbstractCoopetitionEnv",
    "TrustDilemmaEnv", "PartnerHoldUpEnv",
    "PlatformEcosystemEnv", "DynamicPartnerSelectionEnv",
    "RecoveryRaceEnv", "SynergySearchEnv",
    "SLCDEnv", "RenaultNissanEnv",
    "CooperativeNegotiationEnv", "ReputationMarketEnv",
    # Wrappers (v0.2.0)
    "ObservationConfig", "CoopetitionParallelEnv", "CoopetitionAECEnv",
    # Factory functions
    "make", "make_parallel", "make_aec", "list_environments",
]
