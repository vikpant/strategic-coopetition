"""
================================================================================
COOPETITION-GYM: Observation Configuration
================================================================================

This module provides the ObservationConfig dataclass for configuring
agent-specific observation structure, enabling information asymmetry.

Philosophical Importance:
-------------------------
Trust only has meaning when agents cannot directly observe each other's
internal states. By default, agents see their own trust toward others
but NOT others' trust toward them, preserving epistemic uncertainty.

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ObservationConfig:
    """
    Configure agent-specific observation structure.
    
    This is philosophically critical: trust only has meaning when agents
    cannot directly observe each other's internal states. By default,
    agents see their own trust toward others but NOT others' trust toward them.
    
    Attributes:
        own_actions_visible: Agent sees their own last action
        others_actions_visible: Agent sees others' last actions (public behavior)
        action_history_depth: How many past rounds to include
        
        own_trust_row_visible: Agent sees T[i,:] - their trust toward others
        others_trust_toward_self_visible: Agent sees T[:,i] - others' trust in them
            DEFAULT=False: This preserves epistemic uncertainty
        full_trust_matrix_visible: Legacy mode - see everything (breaks asymmetry)
        
        own_reputation_visible: Agent sees their own reputation
        public_reputation_visible: Agent sees public reputation signals
        
        interdependence_visible: Agent sees dependency structure
        step_count_visible: Agent sees current timestep
        
        private_info_keys: Environment-specific private information
    """
    # Actions
    own_actions_visible: bool = True
    others_actions_visible: bool = True
    action_history_depth: int = 1
    
    # Trust (KEY SETTINGS FOR INFORMATION ASYMMETRY)
    own_trust_row_visible: bool = True
    others_trust_toward_self_visible: bool = False  # DEFAULT FALSE
    full_trust_matrix_visible: bool = False         # LEGACY MODE
    
    # Reputation
    own_reputation_visible: bool = True
    public_reputation_visible: bool = True
    
    # Structure
    interdependence_visible: bool = True
    step_count_visible: bool = True
    
    # Extension point
    private_info_keys: List[str] = field(default_factory=list)
    
    @classmethod
    def full_observability(cls) -> "ObservationConfig":
        """Legacy mode: all agents see everything (v0.1.0 behavior)"""
        return cls(full_trust_matrix_visible=True)
    
    @classmethod
    def realistic_asymmetry(cls) -> "ObservationConfig":
        """Recommended: agents cannot read others' minds"""
        return cls(
            own_trust_row_visible=True,
            others_trust_toward_self_visible=False,
            full_trust_matrix_visible=False
        )
    
    @classmethod
    def minimal(cls) -> "ObservationConfig":
        """Minimal observations: only own actions and trust"""
        return cls(
            own_actions_visible=True,
            others_actions_visible=False,
            own_trust_row_visible=True,
            others_trust_toward_self_visible=False,
            full_trust_matrix_visible=False,
            own_reputation_visible=False,
            public_reputation_visible=False,
            interdependence_visible=False,
            step_count_visible=False
        )
