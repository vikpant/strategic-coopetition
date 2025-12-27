"""
================================================================================
COOPETITION-GYM: PettingZoo Parallel Wrapper
================================================================================

This module implements the PettingZoo ParallelEnv wrapper for simultaneous-move
coopetition games.

All agents submit actions simultaneously without observing each other's
current-round decisions. Suitable for:
- Joint venture investment decisions
- Simultaneous capacity commitments
- TR-1 strategic investment scenarios

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import AbstractCoopetitionEnv


class CoopetitionParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for simultaneous-move coopetition games.
    
    All agents submit actions simultaneously without observing each other's
    current-round decisions. Suitable for:
    - Joint venture investment decisions
    - Simultaneous capacity commitments
    - TR-1 strategic investment scenarios
    
    Example:
        >>> from coopetition_gym.envs import make_parallel
        >>> env = make_parallel("TrustDilemma-v0")
        >>> observations, infos = env.reset(seed=42)
        >>> actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        >>> observations, rewards, terminations, truncations, infos = env.step(actions)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "name": "coopetition_parallel"}
    
    def __init__(
        self, 
        base_env: "AbstractCoopetitionEnv",
        render_mode: Optional[str] = None
    ):
        """
        Initialize the parallel wrapper.
        
        Args:
            base_env: AbstractCoopetitionEnv instance containing game logic
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        self.base_env = base_env
        self.render_mode = render_mode
        
        self.possible_agents = base_env.possible_agents
        self.agents = self.possible_agents.copy()
        
        # Build per-agent spaces (public attributes required by PettingZoo API)
        self.observation_spaces = {}
        self.action_spaces = {}

        # Initialize base env to compute observation sizes
        base_env._init_state(seed=0)

        for i, agent in enumerate(self.possible_agents):
            # Observation space matches agent-specific observation
            obs_sample = base_env.get_observation_for(agent)
            self.observation_spaces[agent] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(obs_sample),),
                dtype=np.float32
            )

            # Action space: cooperation level [0, endowment]
            self.action_spaces[agent] = spaces.Box(
                low=0.0,
                high=float(base_env.endowments[i]),
                shape=(1,),
                dtype=np.float32
            )
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> tuple[Dict[str, NDArray], Dict[str, Dict]]:
        """
        Reset and return initial observations.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional configuration overrides
            
        Returns:
            observations: Dict mapping agent_id to observation array
            infos: Dict mapping agent_id to info dict
        """
        self.base_env._init_state(seed=seed)
        self.agents = self.possible_agents.copy()
        
        observations = {
            agent: self.base_env.get_observation_for(agent) 
            for agent in self.agents
        }
        infos = {
            agent: self.base_env.get_info_for(agent) 
            for agent in self.agents
        }
        
        return observations, infos
    
    def step(
        self, 
        actions: Dict[str, NDArray]
    ) -> tuple[
        Dict[str, NDArray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Dict]
    ]:
        """
        Process simultaneous actions from all agents.
        
        Args:
            actions: Dict mapping agent_id to action array
            
        Returns:
            observations: Dict of next observations
            rewards: Dict of rewards
            terminations: Dict of termination flags
            truncations: Dict of truncation flags
            infos: Dict of info dicts
        """
        results = self.base_env.process_actions(actions)
        
        observations = {
            agent: self.base_env.get_observation_for(agent) 
            for agent in self.agents
        }
        rewards = results["rewards"]
        terminations = results["terminations"]
        truncations = results["truncations"]
        infos = {
            agent: self.base_env.get_info_for(agent) 
            for agent in self.agents
        }
        
        # Remove finished agents
        self.agents = [
            agent for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]
        
        return observations, rewards, terminations, truncations, infos
    
    @property
    def num_agents(self) -> int:
        """Return the number of currently active agents."""
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Return the maximum number of agents."""
        return len(self.possible_agents)

    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for specific agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for specific agent."""
        return self.action_spaces[agent]
    
    def render(self) -> Optional[str]:
        """Render current state."""
        if hasattr(self.base_env, 'render'):
            return self.base_env.render()
        return None
    
    def close(self):
        """Clean up resources."""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()
    
    def state(self) -> NDArray:
        """
        Return the global state (for centralized training).
        
        Returns the full observation concatenated for all agents.
        """
        all_obs = [
            self.base_env.get_observation_for(agent)
            for agent in self.possible_agents
        ]
        return np.concatenate(all_obs)
