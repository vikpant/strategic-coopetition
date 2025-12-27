"""
================================================================================
COOPETITION-GYM: PettingZoo AEC Wrapper
================================================================================

This module implements the PettingZoo AECEnv wrapper for sequential-move
coopetition games.

Agents take turns, with later movers observing earlier movers' choices.
Suitable for:
- Stackelberg (leader-follower) games
- Sequential reciprocity (TR-4)
- Turn-based trust building/breaking

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import AbstractCoopetitionEnv


class CoopetitionAECEnv(AECEnv):
    """
    PettingZoo AECEnv wrapper for sequential-move coopetition games.
    
    Agents take turns, with later movers observing earlier movers' choices.
    Suitable for:
    - Stackelberg (leader-follower) games
    - Sequential reciprocity (TR-4)
    - Turn-based trust building/breaking
    
    Example:
        >>> from coopetition_gym.envs import make_aec
        >>> env = make_aec("TrustDilemma-v0")
        >>> env.reset(seed=42)
        >>> for agent in env.agent_iter():
        ...     observation, reward, termination, truncation, info = env.last()
        ...     action = policy(observation) if not termination else None
        ...     env.step(action)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "name": "coopetition_aec"}
    
    def __init__(
        self,
        base_env: "AbstractCoopetitionEnv",
        render_mode: Optional[str] = None
    ):
        """
        Initialize the AEC wrapper.
        
        Args:
            base_env: AbstractCoopetitionEnv instance containing game logic
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        super().__init__()
        
        self.base_env = base_env
        self.render_mode = render_mode
        
        self.possible_agents = base_env.possible_agents
        self.agents = self.possible_agents.copy()
        
        # Turn management
        self._agent_selector: Optional[AgentSelector] = None
        self.agent_selection: Optional[str] = None
        
        # Accumulate actions within round
        self._current_round_actions: Dict[str, NDArray] = {}
        
        # Per-agent state
        self.rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        self.terminations: Dict[str, bool] = {agent: False for agent in self.possible_agents}
        self.truncations: Dict[str, bool] = {agent: False for agent in self.possible_agents}
        self.infos: Dict[str, Dict] = {agent: {} for agent in self.possible_agents}
        
        self._cumulative_rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        self._pending_rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        
        # Initialize base env to compute observation sizes
        base_env._init_state(seed=0)
        
        # Build spaces (public attributes required by PettingZoo API)
        self.observation_spaces = {}
        self.action_spaces = {}

        # Extra dimension for revealed actions in sequential mode
        base_obs_sample = base_env.get_observation_for(self.possible_agents[0])
        base_obs_len = len(base_obs_sample)
        extra_dim = len(self.possible_agents)  # Revealed actions

        for i, agent in enumerate(self.possible_agents):
            self.observation_spaces[agent] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(base_obs_len + extra_dim,),
                dtype=np.float32
            )
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
    ):
        """
        Reset for new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional configuration overrides
        """
        self.base_env._init_state(seed=seed)
        
        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self._current_round_actions = {}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: self.base_env.get_info_for(agent) for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
    
    def step(self, action):
        """
        Process current agent's action, advance to next.
        
        Args:
            action: Action for the current agent
        """
        current_agent = self.agent_selection
        
        # Handle dead step - agent was terminated/truncated last turn
        if self.terminations.get(current_agent, False) or self.truncations.get(current_agent, False):
            self._was_dead_step(action)
            return
        
        # Clear cumulative reward for this agent (they're about to act on it)
        self._cumulative_rewards[current_agent] = 0.0
        
        # Store action
        if action is not None:
            self._current_round_actions[current_agent] = np.asarray(action, dtype=np.float32)
        
        # Count only non-terminated agents for round completion
        active_agents = [a for a in self.agents 
                         if not self.terminations.get(a, False) and not self.truncations.get(a, False)]
        
        # Check if round complete
        if len(self._current_round_actions) == len(active_agents):
            # Process full round
            results = self.base_env.process_actions(self._current_round_actions)
            
            # Update state - set rewards for active agents
            for agent in active_agents:
                self.rewards[agent] = results["rewards"][agent]
                self.terminations[agent] = results["terminations"][agent]
                self.truncations[agent] = results["truncations"][agent]
                self.infos[agent] = self.base_env.get_info_for(agent)
            
            # Accumulate rewards into _cumulative_rewards
            self._accumulate_rewards()
            
            # Clear round
            self._current_round_actions = {}
        else:
            # Round NOT complete - clear rewards dict
            self._clear_rewards()
        
        # Advance to next agent (selector handles the cycling)
        if self.agents:
            self.agent_selection = self._agent_selector.next()
    
    def _was_dead_step(self, action):
        """
        Handle step for terminated/truncated agent.
        Agent is removed from all tracking dicts after this.
        """
        agent = self.agent_selection
        
        # Verify action is None for dead agent
        assert action is None, f"Dead agents should receive None action, got {action}"
        
        # Remove from all tracking structures
        if agent in self.rewards:
            del self.rewards[agent]
        if agent in self._cumulative_rewards:
            del self._cumulative_rewards[agent]
        if agent in self.terminations:
            del self.terminations[agent]
        if agent in self.truncations:
            del self.truncations[agent]
        if agent in self.infos:
            del self.infos[agent]
        
        # Remove from agents list
        if agent in self.agents:
            self.agents.remove(agent)
        
        # Clear rewards for remaining agents (prevent double-counting)
        self._clear_rewards()
        
        # Update selector and advance
        if self.agents:
            self._agent_selector = AgentSelector(self.agents)
            self.agent_selection = self._agent_selector.next()
    
    def observe(self, agent: str) -> NDArray:
        """
        Get observation for agent, including revealed actions from earlier movers.
        
        Args:
            agent: Agent to get observation for
            
        Returns:
            Observation array with base observation + revealed actions
        """
        base_obs = self.base_env.get_observation_for(agent)
        
        # Build revealed actions vector
        revealed = []
        for other in self.possible_agents:
            if other in self._current_round_actions:
                revealed.append(float(self._current_round_actions[other].flatten()[0]))
            else:
                revealed.append(-1.0)  # Not yet revealed
        
        return np.concatenate([
            base_obs, 
            np.array(revealed, dtype=np.float32)
        ]).astype(np.float32)
    
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
            self.observe(agent)
            for agent in self.possible_agents
        ]
        return np.concatenate(all_obs)
