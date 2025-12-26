"""
================================================================================
COOPETITION-GYM: Reciprocity & Conditionality Module (TR-4)
================================================================================

This module will implement TR-4 when available. Currently provides skeleton
interfaces for forward compatibility.

TR-4 Focus Areas:
-----------------
1. Conditional Cooperation
   - "I cooperate if you cooperated"
   - Threshold-based reciprocity
   - Gradual vs. binary responses

2. Sequential Reciprocity
   - Turn-based reaction dynamics
   - Observation before action
   - Strategic timing of cooperation/defection

3. Forgiveness & Punishment
   - Recovery from defection episodes
   - Punishment severity and duration
   - Forgiveness conditions

4. History Dependence
   - How past trajectories shape present choices
   - Memory horizons
   - Weighted recency effects

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class ReciprocityParameters:
    """
    Parameters for reciprocity dynamics (TR-4).
    
    These will be refined when TR-4 equations are available.
    
    Attributes:
        reciprocity_threshold: Partner cooperation rate to trigger reciprocity
        reciprocity_sensitivity: How strongly to match partner behavior
        forgiveness_rate: How quickly past defections are forgiven
        forgiveness_threshold: Consecutive cooperations to trigger forgiveness
        punishment_severity: Reward reduction during punishment
        punishment_duration: Rounds of punishment per defection
        grim_trigger: If True, defection triggers permanent punishment
        memory_horizon: How far back to consider
        recency_weight: Weight for recent vs. distant history
    """
    # Conditional cooperation
    reciprocity_threshold: float = 0.5      # Partner cooperation rate to trigger reciprocity
    reciprocity_sensitivity: float = 1.0    # How strongly to match partner behavior
    
    # Forgiveness
    forgiveness_rate: float = 0.1           # How quickly past defections forgiven
    forgiveness_threshold: int = 3          # Consecutive cooperations to trigger forgiveness
    
    # Punishment
    punishment_severity: float = 0.5        # Reward reduction during punishment
    punishment_duration: int = 3            # Rounds of punishment per defection
    grim_trigger: bool = False              # If True, defection triggers permanent punishment
    
    # History
    memory_horizon: int = 10                # How far back to consider
    recency_weight: float = 0.8             # Weight for recent vs. distant history


@dataclass
class ReciprocityState:
    """
    Track reciprocity state per agent pair.
    
    Attributes:
        punishment_counters: Remaining punishment rounds [n x n]
        forgiveness_counters: Consecutive cooperations [n x n]
        cooperation_history: Rolling cooperation rates [n x n]
        grim_triggered: Permanent punishment flags [n x n]
    """
    punishment_counters: NDArray[np.integer]    # Remaining punishment rounds [n x n]
    forgiveness_counters: NDArray[np.integer]   # Consecutive cooperations [n x n]
    cooperation_history: NDArray[np.floating]   # Rolling cooperation rates [n x n]
    grim_triggered: NDArray[np.bool_]           # Permanent punishment flags [n x n]
    
    @classmethod
    def create_initial(cls, n_agents: int) -> "ReciprocityState":
        """Create initial state with no punishments and neutral history."""
        return cls(
            punishment_counters=np.zeros((n_agents, n_agents), dtype=np.int32),
            forgiveness_counters=np.zeros((n_agents, n_agents), dtype=np.int32),
            cooperation_history=np.full((n_agents, n_agents), 0.5, dtype=np.float32),
            grim_triggered=np.zeros((n_agents, n_agents), dtype=bool)
        )


class ReciprocityModel:
    """
    Model for computing reciprocity effects.
    
    Skeleton implementation—will be expanded with TR-4 equations.
    
    This model handles:
    - Cooperation signal computation
    - Reciprocity recommendations
    - Punishment and forgiveness tracking
    - History-dependent behavior
    """
    
    def __init__(self, params: Optional[ReciprocityParameters] = None):
        """
        Initialize the reciprocity model.
        
        Args:
            params: Model parameters. Uses defaults if not provided.
        """
        self.params = params or ReciprocityParameters()
    
    def compute_cooperation_signal(
        self,
        actions: NDArray[np.floating],
        baselines: NDArray[np.floating],
        endowments: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Compute cooperation signals from actions.
        
        Signal > 0: cooperative
        Signal < 0: defection
        
        Args:
            actions: Current action array
            baselines: Baseline expectations
            endowments: Agent endowments
            
        Returns:
            Array of cooperation signals in [-1, 1]
        """
        cooperation_rates = (actions - baselines) / (endowments - baselines + 1e-8)
        return np.clip(cooperation_rates, -1.0, 1.0)
    
    def should_cooperate(
        self,
        agent_idx: int,
        partner_idx: int,
        action_history: List[NDArray[np.floating]],
        baselines: NDArray[np.floating]
    ) -> float:
        """
        Compute recommended cooperation level based on reciprocity.
        
        Returns cooperation multiplier [0, 1] for agent toward partner.
        
        Args:
            agent_idx: Index of the agent making decision
            partner_idx: Index of the partner
            action_history: List of past action arrays
            baselines: Baseline expectations
            
        Returns:
            Cooperation multiplier in [0, 1]
        """
        if len(action_history) == 0:
            return 1.0  # Cooperate initially
        
        # Look at partner's recent behavior toward threshold
        horizon = min(len(action_history), self.params.memory_horizon)
        recent = action_history[-horizon:]
        
        partner_actions = np.array([h[partner_idx] for h in recent])
        partner_rate = np.mean(partner_actions) / baselines[partner_idx]
        
        # Apply recency weighting
        if len(recent) > 1:
            weights = np.array([
                self.params.recency_weight ** (len(recent) - 1 - i) 
                for i in range(len(recent))
            ])
            weights /= weights.sum()
            partner_rate = np.average(partner_actions / baselines[partner_idx], weights=weights)
        
        # Reciprocity logic
        if partner_rate >= self.params.reciprocity_threshold:
            return 1.0  # Full cooperation
        else:
            # Gradual reduction based on partner's cooperation
            return float(max(0.0, partner_rate / self.params.reciprocity_threshold))
    
    def compute_punishment_modifier(
        self,
        state: ReciprocityState,
        agent_idx: int
    ) -> float:
        """
        Compute reward modifier based on punishment state.
        
        If agent is being punished by others, their rewards are reduced.
        
        Args:
            state: Current reciprocity state
            agent_idx: Index of the agent
            
        Returns:
            Reward multiplier in (0, 1]
        """
        # Check if any partner is punishing this agent
        punishments = state.punishment_counters[:, agent_idx]
        
        if np.any(punishments > 0) or np.any(state.grim_triggered[:, agent_idx]):
            return 1.0 - self.params.punishment_severity
        
        return 1.0
    
    def compute_reward_modifiers(
        self,
        state: ReciprocityState,
        n_agents: int
    ) -> NDArray[np.floating]:
        """
        Compute reciprocity modifiers for all agents.
        
        Args:
            state: Current reciprocity state
            n_agents: Number of agents
            
        Returns:
            Array of reward multipliers
        """
        modifiers = np.ones(n_agents, dtype=np.float32)
        for i in range(n_agents):
            modifiers[i] = self.compute_punishment_modifier(state, i)
        return modifiers
    
    def update_state(
        self,
        state: ReciprocityState,
        actions: NDArray[np.floating],
        baselines: NDArray[np.floating],
        endowments: NDArray[np.floating]
    ) -> ReciprocityState:
        """
        Update reciprocity state after round.
        
        Args:
            state: Current reciprocity state
            actions: Current action array
            baselines: Baseline expectations
            endowments: Agent endowments
            
        Returns:
            Updated ReciprocityState
        """
        n_agents = len(actions)
        signals = self.compute_cooperation_signal(actions, baselines, endowments)
        
        new_punishment = state.punishment_counters.copy()
        new_forgiveness = state.forgiveness_counters.copy()
        new_history = state.cooperation_history.copy()
        new_grim = state.grim_triggered.copy()
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                
                # Update cooperation history (exponential moving average)
                alpha = 1.0 / self.params.memory_horizon
                new_history[i, j] = (1 - alpha) * new_history[i, j] + alpha * ((signals[j] + 1) / 2)
                
                # Check for defection
                if signals[j] < 0:
                    new_punishment[i, j] = self.params.punishment_duration
                    new_forgiveness[i, j] = 0
                    
                    if self.params.grim_trigger:
                        new_grim[i, j] = True
                else:
                    # Decrement punishment
                    new_punishment[i, j] = max(0, new_punishment[i, j] - 1)
                    
                    # Increment forgiveness counter
                    new_forgiveness[i, j] += 1
        
        return ReciprocityState(
            punishment_counters=new_punishment,
            forgiveness_counters=new_forgiveness,
            cooperation_history=new_history,
            grim_triggered=new_grim
        )
    
    def check_forgiveness(
        self,
        state: ReciprocityState,
        agent_idx: int,
        partner_idx: int
    ) -> bool:
        """
        Check if agent should forgive partner.
        
        Args:
            state: Current reciprocity state
            agent_idx: Index of the agent considering forgiveness
            partner_idx: Index of the partner to potentially forgive
            
        Returns:
            True if forgiveness conditions are met
        """
        consecutive_coop = state.forgiveness_counters[agent_idx, partner_idx]
        return consecutive_coop >= self.params.forgiveness_threshold
