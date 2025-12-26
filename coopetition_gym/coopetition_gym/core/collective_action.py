"""
================================================================================
COOPETITION-GYM: Collective Action & Loyalty Module (TR-3)
================================================================================

This module will implement TR-3 when available. Currently provides skeleton
interfaces for forward compatibility.

TR-3 Focus Areas:
-----------------
1. Collective Action Problems
   - N-agent coordination challenges
   - Critical mass thresholds for partnership success
   - Free-rider detection and response

2. Loyalty Dynamics
   - Commitment persistence over time
   - Loyalty rewards for sustained cooperation
   - Defection penalties and exclusion mechanisms

3. Coalition Stability
   - Conditions for partnership sustainability
   - Entry/exit dynamics
   - Minimum viable coalition size

Key Distinction:
----------------
TR-3 concerns INTERNAL partnership dynamics (coordination within coalition),
NOT external market competition between firms.

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class CollectiveActionParameters:
    """
    Parameters for collective action dynamics (TR-3).
    
    These will be refined when TR-3 equations are available.
    
    Attributes:
        free_rider_threshold: Cooperation rate below which agent is free-riding
        free_rider_penalty: Reward multiplier for detected free-riders
        loyalty_horizon: Number of past steps to consider for loyalty calculation
        loyalty_bonus_rate: Bonus multiplier for loyal agents
        loyalty_threshold: Minimum loyalty score for "loyal" status
        min_coalition_size: Minimum viable partnership size
        critical_mass_fraction: Fraction of agents needed for coordination success
        exclusion_threshold: Loyalty score below which agent risks exclusion
        coordination_bonus: Reward multiplier when all agents coordinate
        coordination_threshold: Cooperation rate for coordination bonus
    """
    # Free-rider detection
    free_rider_threshold: float = 0.3       # Below this = free-riding
    free_rider_penalty: float = 0.5         # Reward multiplier for free-riders
    
    # Loyalty mechanics
    loyalty_horizon: int = 10               # Lookback for loyalty calculation
    loyalty_bonus_rate: float = 0.1         # Bonus per unit loyalty score
    loyalty_threshold: float = 0.6          # Minimum for "loyal" status
    
    # Coalition stability
    min_coalition_size: int = 2             # Minimum viable partnership
    critical_mass_fraction: float = 0.5     # Fraction needed for success
    exclusion_threshold: float = 0.2        # Below this = risk exclusion
    
    # Group-level effects
    coordination_bonus: float = 0.2         # Bonus when all cooperate above threshold
    coordination_threshold: float = 0.5     # Cooperation rate for coordination bonus


@dataclass
class CollectiveActionState:
    """
    Track collective action state across episode.
    
    Attributes:
        loyalty_scores: Per-agent loyalty scores in [0, 1]
        coalition_members: List of agent indices in active coalition
        excluded_agents: List of agent indices excluded from coalition
        coordination_achieved: Whether group coordinated this round
    """
    loyalty_scores: NDArray[np.floating]      # Per-agent loyalty [0,1]
    coalition_members: List[int]              # Active coalition members
    excluded_agents: List[int]                # Agents excluded from coalition
    coordination_achieved: bool               # Whether group coordinated this round
    
    @classmethod
    def create_initial(cls, n_agents: int) -> "CollectiveActionState":
        """Create initial state with neutral loyalty and full coalition."""
        return cls(
            loyalty_scores=np.full(n_agents, 0.5, dtype=np.float32),
            coalition_members=list(range(n_agents)),
            excluded_agents=[],
            coordination_achieved=False
        )


class CollectiveActionModel:
    """
    Model for computing collective action effects.
    
    Skeleton implementation—will be expanded with TR-3 equations.
    
    This model handles:
    - Free-rider detection and penalties
    - Loyalty score calculation
    - Coordination bonuses
    - Coalition membership management
    """
    
    def __init__(self, params: Optional[CollectiveActionParameters] = None):
        """
        Initialize the collective action model.
        
        Args:
            params: Model parameters. Uses defaults if not provided.
        """
        self.params = params or CollectiveActionParameters()
    
    def detect_free_riders(
        self,
        actions: NDArray[np.floating],
        endowments: NDArray[np.floating],
        baselines: NDArray[np.floating]
    ) -> List[int]:
        """
        Identify agents contributing below threshold.
        
        Free-rider: cooperation rate < free_rider_threshold
        
        Args:
            actions: Current action array
            endowments: Agent endowments
            baselines: Baseline expectations
            
        Returns:
            List of agent indices detected as free-riders
        """
        cooperation_rates = actions / endowments
        return [
            i for i, rate in enumerate(cooperation_rates)
            if rate < self.params.free_rider_threshold
        ]
    
    def compute_loyalty_scores(
        self,
        action_history: List[NDArray[np.floating]],
        endowments: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Compute loyalty scores based on sustained cooperation.
        
        Loyalty = mean cooperation rate over loyalty_horizon
        
        Args:
            action_history: List of past action arrays
            endowments: Agent endowments
            
        Returns:
            Array of loyalty scores in [0, 1]
        """
        n_agents = len(endowments)
        
        if len(action_history) == 0:
            return np.full(n_agents, 0.5, dtype=np.float32)
        
        horizon = min(len(action_history), self.params.loyalty_horizon)
        recent = action_history[-horizon:]
        
        scores = np.zeros(n_agents, dtype=np.float32)
        for i in range(n_agents):
            agent_actions = np.array([h[i] for h in recent])
            scores[i] = np.mean(agent_actions / endowments[i])
        
        return np.clip(scores, 0.0, 1.0)
    
    def check_coordination(
        self,
        actions: NDArray[np.floating],
        endowments: NDArray[np.floating]
    ) -> bool:
        """
        Check if group achieved coordination (all above threshold).
        
        Args:
            actions: Current action array
            endowments: Agent endowments
            
        Returns:
            True if all agents cooperated above coordination_threshold
        """
        cooperation_rates = actions / endowments
        return bool(np.all(cooperation_rates >= self.params.coordination_threshold))
    
    def compute_reward_modifiers(
        self,
        actions: NDArray[np.floating],
        endowments: NDArray[np.floating],
        action_history: List[NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Compute collective action modifiers for rewards.
        
        Returns array of multipliers, one per agent.
        
        Modifiers include:
        - Free-rider penalty (< 1.0)
        - Loyalty bonus (> 1.0)
        - Coordination bonus (> 1.0)
        
        Args:
            actions: Current action array
            endowments: Agent endowments
            action_history: List of past action arrays
            
        Returns:
            Array of reward multipliers
        """
        n_agents = len(actions)
        modifiers = np.ones(n_agents, dtype=np.float32)
        
        # Free-rider penalty
        baselines = endowments * 0.3  # Default baseline
        free_riders = self.detect_free_riders(actions, endowments, baselines)
        for i in free_riders:
            modifiers[i] *= self.params.free_rider_penalty
        
        # Loyalty bonus
        loyalty_scores = self.compute_loyalty_scores(action_history, endowments)
        for i in range(n_agents):
            if loyalty_scores[i] >= self.params.loyalty_threshold:
                modifiers[i] *= (1.0 + self.params.loyalty_bonus_rate * loyalty_scores[i])
        
        # Coordination bonus (group-level)
        if self.check_coordination(actions, endowments):
            modifiers *= (1.0 + self.params.coordination_bonus)
        
        return modifiers
    
    def update_state(
        self,
        state: CollectiveActionState,
        actions: NDArray[np.floating],
        endowments: NDArray[np.floating],
        action_history: List[NDArray[np.floating]]
    ) -> CollectiveActionState:
        """
        Update collective action state after round.
        
        Args:
            state: Current collective action state
            actions: Current action array
            endowments: Agent endowments
            action_history: List of past action arrays
            
        Returns:
            Updated CollectiveActionState
        """
        # Update loyalty scores
        loyalty_scores = self.compute_loyalty_scores(action_history, endowments)
        
        # Check for exclusion (placeholder for TR-3 exclusion logic)
        excluded = state.excluded_agents.copy()
        coalition = state.coalition_members.copy()
        
        for i in coalition:
            if loyalty_scores[i] < self.params.exclusion_threshold:
                # Risk exclusion (could add voting mechanic here)
                pass  # Placeholder for TR-3 exclusion logic
        
        return CollectiveActionState(
            loyalty_scores=loyalty_scores,
            coalition_members=coalition,
            excluded_agents=excluded,
            coordination_achieved=self.check_coordination(actions, endowments)
        )
    
    def compute_critical_mass_achieved(
        self,
        actions: NDArray[np.floating],
        endowments: NDArray[np.floating]
    ) -> bool:
        """
        Check if critical mass of agents are cooperating.
        
        Args:
            actions: Current action array
            endowments: Agent endowments
            
        Returns:
            True if fraction of cooperating agents >= critical_mass_fraction
        """
        cooperation_rates = actions / endowments
        cooperating = np.sum(cooperation_rates >= self.params.coordination_threshold)
        return cooperating / len(actions) >= self.params.critical_mass_fraction
