"""
================================================================================
COOPETITION-GYM: Environments Module (v0.2.0)
================================================================================

This module provides Gymnasium and PettingZoo-compatible environments for
multi-agent strategic coopetition research.

Environment Categories:
-----------------------
1. Dyadic (Micro): TrustDilemma-v0, PartnerHoldUp-v0
2. Ecosystem (Macro): PlatformEcosystem-v0, DynamicPartnerSelection-v0
3. Research Benchmarks: RecoveryRace-v0, SynergySearch-v0
4. Validated Case Studies: SLCD-v0, RenaultNissan-v0
5. Extended: CooperativeNegotiation-v0, ReputationMarket-v0

API Modes (v0.2.0):
-------------------
- make(): Gymnasium API (backward compatible)
- make_parallel(): PettingZoo ParallelEnv (simultaneous moves)
- make_aec(): PettingZoo AECEnv (sequential moves)

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from typing import Optional

from .base import (
    CoopetitionEnv, 
    MultiAgentCoopetitionEnv, 
    EnvironmentConfig,
    AbstractCoopetitionEnv,
)
from .dyadic_envs import TrustDilemmaEnv, PartnerHoldUpEnv
from .ecosystem_envs import PlatformEcosystemEnv, DynamicPartnerSelectionEnv
from .benchmark_envs import RecoveryRaceEnv, SynergySearchEnv
from .case_study_envs import SLCDEnv, RenaultNissanEnv, AlliancePhase
from .extended_envs import CooperativeNegotiationEnv, ReputationMarketEnv

from .wrappers import (
    ObservationConfig,
    CoopetitionParallelEnv,
    CoopetitionAECEnv,
)

_ENVIRONMENT_REGISTRY = {
    "TrustDilemma-v0": TrustDilemmaEnv,
    "PartnerHoldUp-v0": PartnerHoldUpEnv,
    "PlatformEcosystem-v0": PlatformEcosystemEnv,
    "DynamicPartnerSelection-v0": DynamicPartnerSelectionEnv,
    "RecoveryRace-v0": RecoveryRaceEnv,
    "SynergySearch-v0": SynergySearchEnv,
    "SLCD-v0": SLCDEnv,
    "RenaultNissan-v0": RenaultNissanEnv,
    "CooperativeNegotiation-v0": CooperativeNegotiationEnv,
    "ReputationMarket-v0": ReputationMarketEnv,
}


def list_environments():
    """List all available environment IDs."""
    return list(_ENVIRONMENT_REGISTRY.keys())


def make(env_id: str, **kwargs):
    """
    Create a Gymnasium-compatible environment by ID.
    
    This is the v0.1.0-compatible interface.
    
    Args:
        env_id: Environment identifier
        **kwargs: Additional arguments passed to environment
        
    Returns:
        CoopetitionEnv instance
    """
    if env_id not in _ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment: {env_id}. Available: {list_environments()}")
    return _ENVIRONMENT_REGISTRY[env_id](**kwargs)


def make_parallel(
    env_id: str,
    obs_config: Optional[ObservationConfig] = None,
    render_mode: Optional[str] = None,
    **kwargs
) -> CoopetitionParallelEnv:
    """
    Create a PettingZoo ParallelEnv for simultaneous-move games.
    
    All agents submit actions simultaneously without observing each other's
    current-round decisions.
    
    Args:
        env_id: Environment identifier
        obs_config: Observation configuration for information asymmetry
        render_mode: Rendering mode ("human", "ansi", or None)
        **kwargs: Additional arguments passed to base environment
        
    Returns:
        CoopetitionParallelEnv instance
        
    Example:
        >>> env = make_parallel("TrustDilemma-v0")
        >>> observations, infos = env.reset(seed=42)
        >>> actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        >>> observations, rewards, terminations, truncations, infos = env.step(actions)
    """
    if env_id not in _ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment: {env_id}. Available: {list_environments()}")
    
    # Get base environment class and create config
    env_class = _ENVIRONMENT_REGISTRY[env_id]
    
    # Create base environment instance
    base_env = env_class(**kwargs)
    
    # Override obs_config if provided
    if obs_config is not None:
        base_env.obs_config = obs_config
    
    return CoopetitionParallelEnv(base_env, render_mode=render_mode)


def make_aec(
    env_id: str,
    obs_config: Optional[ObservationConfig] = None,
    render_mode: Optional[str] = None,
    **kwargs
) -> CoopetitionAECEnv:
    """
    Create a PettingZoo AECEnv for sequential-move games.
    
    Agents take turns, with later movers observing earlier movers' choices.
    Suitable for Stackelberg games and TR-4 reciprocity analysis.
    
    Args:
        env_id: Environment identifier
        obs_config: Observation configuration for information asymmetry
        render_mode: Rendering mode ("human", "ansi", or None)
        **kwargs: Additional arguments passed to base environment
        
    Returns:
        CoopetitionAECEnv instance
        
    Example:
        >>> env = make_aec("TrustDilemma-v0")
        >>> env.reset(seed=42)
        >>> for agent in env.agent_iter():
        ...     observation, reward, termination, truncation, info = env.last()
        ...     action = policy(observation) if not termination else None
        ...     env.step(action)
    """
    if env_id not in _ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment: {env_id}. Available: {list_environments()}")
    
    # Get base environment class
    env_class = _ENVIRONMENT_REGISTRY[env_id]
    
    # Create base environment instance
    base_env = env_class(**kwargs)
    
    # Override obs_config if provided
    if obs_config is not None:
        base_env.obs_config = obs_config
    
    return CoopetitionAECEnv(base_env, render_mode=render_mode)


__all__ = [
    # Base classes
    "CoopetitionEnv", 
    "MultiAgentCoopetitionEnv", 
    "EnvironmentConfig",
    "AbstractCoopetitionEnv",
    # Dyadic environments
    "TrustDilemmaEnv", 
    "PartnerHoldUpEnv",
    # Ecosystem environments
    "PlatformEcosystemEnv", 
    "DynamicPartnerSelectionEnv",
    # Benchmark environments
    "RecoveryRaceEnv", 
    "SynergySearchEnv",
    # Case study environments
    "SLCDEnv", 
    "RenaultNissanEnv", 
    "AlliancePhase",
    # Extended environments
    "CooperativeNegotiationEnv", 
    "ReputationMarketEnv",
    # Wrappers
    "ObservationConfig",
    "CoopetitionParallelEnv",
    "CoopetitionAECEnv",
    # Factory functions
    "make", 
    "make_parallel",
    "make_aec",
    "list_environments",
]
