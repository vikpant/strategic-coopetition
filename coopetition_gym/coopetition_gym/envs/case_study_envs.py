"""
================================================================================
COOPETITION-GYM: Validated Case Study Environments
================================================================================

This module implements environments based on real-world case studies that were
empirically validated in the research papers. These serve as the gold standard
benchmarks where we know the "ground truth" parameters from actual business data.

Environments:
-------------
1. SLCD-v0: Samsung-Sony S-LCD Joint Venture (TR-1, arXiv:2510.18802)
2. RenaultNissan-v0: Renault-Nissan Alliance multi-phase (TR-2, arXiv:2510.24909)

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, Optional, Tuple
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.base import CoopetitionEnv, EnvironmentConfig
from core.value_functions import ValueFunctionParameters, ValueSpecification
from core.trust_dynamics import TrustParameters
from core.interdependence import (
    create_slcd_interdependence,
    create_renault_nissan_interdependence
)


class SLCDEnv(CoopetitionEnv):
    """
    Samsung-Sony S-LCD Joint Venture Environment (SLCD-v0)
    
    Validated parameters achieving 58/60 score against historical data.
    
    Agents:
    - Agent 0: Samsung Electronics
    - Agent 1: Sony Corporation
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "SLCD-v0",
        "validation_score": "58/60",
        "source": "TR-1 arXiv:2510.18802 Section 8"
    }
    
    def __init__(
        self,
        max_steps: int = 100,
        trust_enabled: bool = True,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        # Validated value parameters from TR-1 §8.3
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=20.0,
            gamma=0.65
        )
        
        # Validated interdependence
        interdependence = create_slcd_interdependence()
        
        trust_params = TrustParameters(
            lambda_plus=0.08,
            lambda_minus=0.28,
            mu_R=0.50,
            delta_R=0.02,
            xi=0.45,
            kappa=1.0,
            initial_trust=0.65
        )
        
        config = EnvironmentConfig(
            n_agents=2,
            max_steps=max_steps,
            endowments=np.array([100.0, 100.0]),
            alpha=np.array([0.55, 0.45]),
            interdependence_matrix=interdependence.matrix,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=trust_enabled,
            baselines=np.array([30.0, 30.0]),
            reward_type="integrated",
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        self.agent_names = ["Samsung", "Sony"]
    
    def _get_legacy_info(self) -> Dict[str, Any]:
        info = super()._get_legacy_info()
        info["agent_names"] = self.agent_names
        if "actions" in self._state:
            info["samsung_investment"] = float(self._state["actions"][0])
            info["sony_investment"] = float(self._state["actions"][1])
        return info


class AlliancePhase(Enum):
    """Phases of the Renault-Nissan Alliance lifecycle."""
    FORMATION = "formation"
    MATURE = "mature"
    CRISIS = "crisis"
    STRAINED = "strained"


class RenaultNissanEnv(CoopetitionEnv):
    """
    Renault-Nissan Alliance Environment (RenaultNissan-v0)
    
    Multi-phase environment validated in TR-2.
    
    Agents:
    - Agent 0: Nissan Motor Corporation
    - Agent 1: Renault SA
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "RenaultNissan-v0",
        "source": "TR-2 arXiv:2510.24909 Section 9"
    }
    
    PHASE_CONFIGS = {
        AlliancePhase.FORMATION: {
            "initial_trust": 0.45,
            "initial_reputation": 0.05,
            "period": "1999-2002"
        },
        AlliancePhase.MATURE: {
            "initial_trust": 0.70,
            "initial_reputation": 0.02,
            "period": "2002-2018"
        },
        AlliancePhase.CRISIS: {
            "initial_trust": 0.30,
            "initial_reputation": 0.45,
            "period": "2018-2020"
        },
        AlliancePhase.STRAINED: {
            "initial_trust": 0.40,
            "initial_reputation": 0.35,
            "period": "2020-2025"
        }
    }
    
    def __init__(
        self,
        phase: str = "mature",
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        if isinstance(phase, str):
            phase = AlliancePhase(phase.lower())
        self.phase = phase
        self.phase_config = self.PHASE_CONFIGS[phase]
        
        interdependence = create_renault_nissan_interdependence(phase.value)
        
        trust_params = TrustParameters(
            lambda_plus=0.08,
            lambda_minus=0.25,
            mu_R=0.55,
            delta_R=0.02,
            xi=0.50,
            kappa=1.0,
            initial_trust=self.phase_config["initial_trust"],
            initial_reputation=self.phase_config["initial_reputation"]
        )
        
        value_params = ValueFunctionParameters(
            specification=ValueSpecification.LOGARITHMIC,
            theta=22.0,
            gamma=0.58
        )
        
        config = EnvironmentConfig(
            n_agents=2,
            max_steps=max_steps,
            endowments=np.array([90.0, 100.0]),
            alpha=np.array([0.48, 0.52]),
            interdependence_matrix=interdependence.matrix,
            value_params=value_params,
            trust_params=trust_params,
            trust_enabled=True,
            baselines=np.array([27.0, 30.0]),
            reward_type="integrated",
            render_mode=render_mode
        )
        
        super().__init__(config=config, **kwargs)
        self.agent_names = ["Nissan", "Renault"]
    
    def _get_legacy_info(self) -> Dict[str, Any]:
        info = super()._get_legacy_info()
        info["phase"] = self.phase.value
        info["period"] = self.phase_config["period"]
        info["agent_names"] = self.agent_names
        return info


def make_slcd(**kwargs) -> SLCDEnv:
    return SLCDEnv(**kwargs)


def make_renault_nissan(**kwargs) -> RenaultNissanEnv:
    return RenaultNissanEnv(**kwargs)
