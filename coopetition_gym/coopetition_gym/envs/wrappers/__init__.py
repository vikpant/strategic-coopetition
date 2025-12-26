"""
================================================================================
COOPETITION-GYM: PettingZoo Wrappers
================================================================================

This module provides PettingZoo-compatible wrappers for coopetition environments.

Wrappers:
---------
- CoopetitionParallelEnv: Simultaneous-move games (ParallelEnv)
- CoopetitionAECEnv: Sequential-move games (AECEnv)
- ObservationConfig: Agent-specific observation configuration

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from .observation_config import ObservationConfig
from .parallel_wrapper import CoopetitionParallelEnv
from .aec_wrapper import CoopetitionAECEnv

__all__ = [
    "ObservationConfig",
    "CoopetitionParallelEnv",
    "CoopetitionAECEnv",
]
