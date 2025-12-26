"""
================================================================================
COOPETITION-GYM: Base Environment Class (v0.2.0)
================================================================================

This module implements the foundational environment classes for coopetition:

- AbstractCoopetitionEnv: API-agnostic game logic (TR-1, TR-2, TR-3, TR-4 hooks)
- CoopetitionEnv: Gymnasium-compatible wrapper (backward compatible)
- MultiAgentCoopetitionEnv: PettingZoo-style hints

Key v0.2.0 Changes:
-------------------
- AbstractCoopetitionEnv separates game logic from API presentation
- ObservationConfig enables agent-specific observations (information asymmetry)
- Action history tracking for TR-4 reciprocity
- Extension hooks for TR-3 collective action
- process_actions() provides API-agnostic action processing

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, Optional, Tuple, Union, List, SupportsFloat
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
import logging
from dataclasses import dataclass, field
from copy import deepcopy
from abc import ABC

# Import core mathematical modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.value_functions import (
    ValueFunctionParameters,
    ValueSpecification,
    total_value,
    individual_value,
)
from core.interdependence import (
    InterdependenceMatrix,
    create_symmetric_interdependence,
)
from core.trust_dynamics import (
    TrustParameters,
    TrustState,
    TrustDynamicsModel,
)
from core.equilibrium import (
    PayoffParameters,
    compute_rewards,
    compute_all_private_payoffs,
    compute_all_integrated_utilities,
    compute_complete_payoffs,
)

# Import observation config
from .wrappers.observation_config import ObservationConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """
    Configuration for a coopetition environment.
    
    Attributes:
        n_agents: Number of agents in the environment
        max_steps: Maximum timesteps per episode
        endowments: Initial resource endowment for each agent
        alpha: Bargaining shares for synergy distribution
        interdependence_matrix: NxN dependency structure
        value_params: Configuration for value calculations
        trust_params: Configuration for trust dynamics
        trust_enabled: Whether to model trust evolution
        baselines: Expected baseline actions for trust signal calculation
        reward_type: Type of reward ("integrated", "private", or "cooperative")
        normalize_rewards: Whether to normalize rewards to [-1, 1] range
        reward_scale: Scaling factor for rewards
        render_mode: Rendering mode ("human", "ansi", or None)
    """
    n_agents: int = 2
    max_steps: int = 100
    endowments: Optional[NDArray[np.floating]] = None
    alpha: Optional[NDArray[np.floating]] = None
    interdependence_matrix: Optional[NDArray[np.floating]] = None
    value_params: Optional[ValueFunctionParameters] = None
    trust_params: Optional[TrustParameters] = None
    trust_enabled: bool = True
    baselines: Optional[NDArray[np.floating]] = None
    reward_type: str = "integrated"
    normalize_rewards: bool = False
    reward_scale: float = 1.0
    render_mode: Optional[str] = None
    
    def __post_init__(self):
        """Initialize defaults based on n_agents."""
        if self.endowments is None:
            self.endowments = np.full(self.n_agents, 100.0)
        
        if self.alpha is None:
            self.alpha = np.full(self.n_agents, 1.0 / self.n_agents)
        
        if self.interdependence_matrix is None:
            self.interdependence_matrix = create_symmetric_interdependence(
                self.n_agents, 0.5
            ).matrix
        
        if self.value_params is None:
            self.value_params = ValueFunctionParameters()
        
        if self.trust_params is None:
            self.trust_params = TrustParameters()
        
        if self.baselines is None:
            self.baselines = self.endowments * 0.3


class AbstractCoopetitionEnv(ABC):
    """
    Abstract base class separating game logic from API presentation.
    
    This class implements core mathematical dynamics:
    - Value creation (TR-1)
    - Trust evolution (TR-2)
    - Reward computation
    - Extension hooks for TR-3 (collective action) and TR-4 (reciprocity)
    
    Concrete API wrappers (Parallel, AEC, Gymnasium) use this class.
    
    Key Design Principles:
    1. Single source of truth for game mechanics
    2. Agent-specific observations via ObservationConfig
    3. Action history tracking for TR-4 reciprocity
    4. Extension hooks for TR-3 collective action
    """
    
    def __init__(
        self, 
        config: EnvironmentConfig,
        obs_config: Optional[ObservationConfig] = None
    ):
        """
        Initialize the abstract environment.
        
        Args:
            config: Environment configuration
            obs_config: Observation configuration for information asymmetry
        """
        self.config = config
        self.obs_config = obs_config or ObservationConfig.realistic_asymmetry()
        
        # Core attributes
        self.n_agents = config.n_agents
        self.max_steps = config.max_steps
        self.endowments = config.endowments.astype(np.float32)
        self.alpha = config.alpha.astype(np.float32)
        self.baselines = config.baselines.astype(np.float32)
        self.D = config.interdependence_matrix.astype(np.float32)
        self.value_params = config.value_params
        self.trust_params = config.trust_params
        self.trust_enabled = config.trust_enabled
        self.reward_type = config.reward_type
        self.normalize_rewards = config.normalize_rewards
        self.reward_scale = config.reward_scale
        
        # Agent identifiers
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents.copy()
        
        # Initialize mathematical models
        self.interdependence = InterdependenceMatrix(self.D)
        self.trust_model = TrustDynamicsModel(self.trust_params) if self.trust_enabled else None
        self.payoff_params = PayoffParameters(
            value_params=self.value_params,
            endowments=self.endowments,
            alpha=self.alpha,
            interdependence=self.interdependence
        )
        
        # State (initialized in reset)
        self._state: Dict[str, np.ndarray] = {}
        self._trust_state: Optional[TrustState] = None
        self._step_count: int = 0
        
        # Action history for TR-4 reciprocity
        self._action_history: List[np.ndarray] = []
        
        # Episode tracking
        self._episode_rewards: List[np.ndarray] = []
        
        # Random generator
        self._np_random: Optional[np.random.Generator] = None
    
    def _init_state(self, seed: Optional[int] = None) -> None:
        """Initialize/reset game state."""
        # Only create new generator if not already set by Gymnasium's reset
        if self._np_random is None:
            if seed is not None:
                self._np_random = np.random.default_rng(seed)
            else:
                self._np_random = np.random.default_rng()
        
        self._step_count = 0
        self._action_history = []
        self._episode_rewards = []
        self.agents = self.possible_agents.copy()
        
        # Initialize trust state
        if self.trust_enabled and self.trust_model is not None:
            self._trust_state = self.trust_model.create_initial_state(self.n_agents)
        else:
            self._trust_state = TrustState(
                trust_matrix=np.ones((self.n_agents, self.n_agents), dtype=np.float32),
                reputation_matrix=np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
            )
        
        # Initialize state dict
        self._state = {
            "actions": self.baselines.copy(),
            "trust": self._trust_state.trust_matrix.copy(),
            "reputation": self._trust_state.reputation_matrix.copy(),
        }
    
    def process_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Process complete round of actions and update state.
        
        This is the core game logic, independent of API format.
        
        Args:
            actions: Dict mapping agent_id to action array
        
        Returns:
            Dict with 'rewards', 'terminations', 'truncations' per agent
        """
        # Convert dict to array
        action_array = np.array([
            float(np.asarray(actions[agent]).flatten()[0]) 
            for agent in self.possible_agents
        ], dtype=np.float32)
        
        # Clip to valid range
        action_array = np.clip(action_array, 0.0, self.endowments)
        
        # Store in history (for TR-4)
        self._action_history.append(action_array.copy())
        
        # Update state
        self._state["actions"] = action_array
        
        # Evolve trust (TR-2)
        if self.trust_enabled and self._trust_state is not None and self.trust_model is not None:
            self._trust_state = self.trust_model.update(
                self._trust_state, action_array, self.baselines, self.D
            )
            self._state["trust"] = self._trust_state.trust_matrix.copy()
            self._state["reputation"] = self._trust_state.reputation_matrix.copy()
        
        # Compute base rewards
        rewards_array = compute_rewards(
            actions=action_array,
            params=self.payoff_params,
            trust_state=self._trust_state if self.trust_enabled else None,
            reward_type=self.reward_type,
            normalize=self.normalize_rewards,
            reward_scale=self.reward_scale
        ).astype(np.float32)
        
        # Apply TR-3 collective action modifier (extension hook)
        collective_modifier = self._compute_collective_action_modifier(action_array)
        rewards_array = rewards_array * collective_modifier
        
        # Apply TR-4 reciprocity modifier (extension hook)
        for i in range(self.n_agents):
            reciprocity_modifier = self._compute_reciprocity_modifier(i)
            rewards_array[i] *= reciprocity_modifier
        
        self._episode_rewards.append(rewards_array.copy())
        self._step_count += 1
        
        # Check termination
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        return {
            "rewards": {agent: float(rewards_array[i]) for i, agent in enumerate(self.possible_agents)},
            "terminations": {agent: terminated for agent in self.possible_agents},
            "truncations": {agent: truncated for agent in self.possible_agents},
        }
    
    def get_observation_for(self, agent: str) -> np.ndarray:
        """
        Build agent-specific observation based on ObservationConfig.
        
        This method implements information asymmetry—each agent only
        sees what they're configured to observe.
        """
        agent_idx = self.possible_agents.index(agent)
        obs_parts = []
        
        # Own actions
        if self.obs_config.own_actions_visible:
            obs_parts.append(np.array([self._state["actions"][agent_idx]], dtype=np.float32))
        
        # Others' actions (publicly observable behavior)
        if self.obs_config.others_actions_visible:
            others_actions = np.delete(self._state["actions"], agent_idx)
            obs_parts.append(others_actions.astype(np.float32))
        
        # Trust information
        if self.obs_config.full_trust_matrix_visible:
            # Legacy mode: full visibility
            obs_parts.append(self._state["trust"].flatten().astype(np.float32))
        else:
            if self.obs_config.own_trust_row_visible:
                # I see my trust toward others: T[i, :]
                obs_parts.append(self._state["trust"][agent_idx, :].astype(np.float32))
            
            if self.obs_config.others_trust_toward_self_visible:
                # I see others' trust toward me: T[:, i]
                obs_parts.append(self._state["trust"][:, agent_idx].astype(np.float32))
        
        # Reputation
        if self.obs_config.own_reputation_visible:
            obs_parts.append(self._state["reputation"][agent_idx, :].astype(np.float32))
        
        if self.obs_config.public_reputation_visible:
            # Mean reputation damage per agent (public signal)
            public_rep = np.array([
                self._state["reputation"][j, :].mean() 
                for j in range(self.n_agents) if j != agent_idx
            ], dtype=np.float32)
            if len(public_rep) > 0:
                obs_parts.append(public_rep)
        
        # Interdependence structure
        if self.obs_config.interdependence_visible:
            obs_parts.append(self.D[agent_idx, :].astype(np.float32))
        
        # Step count (normalized)
        if self.obs_config.step_count_visible:
            obs_parts.append(np.array([self._step_count / self.max_steps], dtype=np.float32))
        
        if not obs_parts:
            # Fallback: return at least current actions
            return self._state["actions"].astype(np.float32)
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def get_info_for(self, agent: str) -> Dict[str, Any]:
        """Build agent-specific info dictionary."""
        agent_idx = self.possible_agents.index(agent)
        
        return {
            "step": self._step_count,
            "own_action": float(self._state["actions"][agent_idx]),
            "own_trust_mean": float(self._state["trust"][agent_idx, :].mean()),
            "cooperation_rate": float(self._state["actions"][agent_idx] / self.endowments[agent_idx]),
        }
    
    def _check_terminated(self) -> bool:
        """Check for terminal state. Override in subclasses."""
        return False
    
    def _check_truncated(self) -> bool:
        """Check for time limit."""
        return self._step_count >= self.max_steps
    
    # =========================================================================
    # TR-3 Extension Hooks: Collective Action & Loyalty
    # =========================================================================
    
    def _compute_collective_action_modifier(self, actions: np.ndarray) -> np.ndarray:
        """
        Compute collective action effects on rewards.
        
        Override in subclasses to implement TR-3:
        - Free-rider penalties
        - Loyalty bonuses
        - Coalition stability effects
        
        Returns: Array of multipliers, one per agent (default: all 1.0)
        """
        return np.ones(self.n_agents, dtype=np.float32)
    
    def compute_loyalty_score(self, agent_idx: int, lookback: int = 10) -> float:
        """
        Compute loyalty score based on sustained cooperation.
        
        Extension point for TR-3 loyalty mechanics.
        """
        if len(self._action_history) < lookback:
            return 0.5  # Neutral
        
        recent = self._action_history[-lookback:]
        agent_actions = np.array([h[agent_idx] for h in recent])
        cooperation_rates = agent_actions / self.endowments[agent_idx]
        
        return float(np.mean(cooperation_rates))
    
    def detect_free_riders(self, threshold: float = 0.3) -> List[int]:
        """
        Identify agents contributing below threshold.
        
        Extension point for TR-3 free-rider detection.
        """
        current_rates = self._state["actions"] / self.endowments
        return [i for i, rate in enumerate(current_rates) if rate < threshold]
    
    # =========================================================================
    # TR-4 Extension Hooks: Reciprocity & Conditionality
    # =========================================================================
    
    def _compute_reciprocity_modifier(self, agent_idx: int) -> float:
        """
        Compute reciprocity effect on agent's reward.
        
        Override in subclasses to implement TR-4:
        - Tit-for-tat responses
        - Forgiveness dynamics
        - Conditional cooperation bonuses
        
        Returns: Multiplier for agent's reward (default: 1.0)
        """
        return 1.0
    
    def get_action_history(self, lookback: int = 5) -> List[np.ndarray]:
        """Return recent action history for reciprocity analysis."""
        if not self._action_history:
            return []
        return self._action_history[-lookback:]
    
    def compute_partner_cooperation_trend(self, agent_idx: int, lookback: int = 5) -> float:
        """
        Compute trend in partners' cooperation toward this agent.
        
        Extension point for TR-4 reciprocity signals.
        Returns: Positive if partners increasing cooperation, negative if decreasing
        """
        history = self.get_action_history(lookback)
        if len(history) < 2:
            return 0.0
        
        partner_indices = [j for j in range(self.n_agents) if j != agent_idx]
        
        early = np.mean([history[0][j] for j in partner_indices])
        late = np.mean([history[-1][j] for j in partner_indices])
        
        return float(late - early) / np.mean(self.endowments)
    
    # =========================================================================
    # Sequential Game Support (for AEC)
    # =========================================================================
    
    @property
    def supports_sequential(self) -> bool:
        """Whether this environment supports AEC (sequential) mode."""
        return True
    
    def get_move_order(self) -> List[str]:
        """
        Define move order for AEC mode. Override for custom orders.
        
        Options:
        - Fixed order (default)
        - Role-based (leader first)
        - Random each round
        - State-dependent
        """
        return self.possible_agents.copy()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Compute statistics for the current episode."""
        if not self._episode_rewards:
            return {}
        
        rewards_array = np.array(self._episode_rewards)
        
        return {
            "total_steps": self._step_count,
            "total_rewards": np.sum(rewards_array, axis=0).tolist(),
            "mean_reward_per_step": np.mean(rewards_array, axis=0).tolist(),
            "final_trust": self._trust_state.mean_trust() if self._trust_state else 1.0,
            "final_reputation_damage": self._trust_state.mean_reputation_damage() if self._trust_state else 0.0,
            "mean_cooperation_rate": float(np.mean(self._state["actions"] / self.endowments)),
        }
    
    def get_trust_state(self) -> Optional[TrustState]:
        """Return the current trust state for analysis."""
        return self._trust_state
    
    def get_interdependence(self) -> InterdependenceMatrix:
        """Return the interdependence matrix."""
        return self.interdependence
    
    def set_baselines(self, baselines: NDArray[np.floating]):
        """Update baseline expectations for trust calculation."""
        self.baselines = baselines.astype(np.float32)
    
    def render(self) -> Optional[str]:
        """Render current state."""
        output_lines = [
            f"\n{'='*60}",
            f"  COOPETITION ENVIRONMENT - Step {self._step_count}",
            f"{'='*60}",
            f"\nAgents: {self.n_agents}",
            f"Actions: {self._state.get('actions', [])}",
            f"Endowments: {self.endowments}",
        ]
        
        if self._state.get("actions") is not None:
            output_lines.append(f"Cooperation Rate: {np.mean(self._state['actions'] / self.endowments):.1%}")
        
        output_lines.extend([
            f"\nTrust Matrix:",
            str(np.round(self._state.get('trust', np.array([])), 3)),
            f"\nMean Trust: {self._trust_state.mean_trust():.3f}" if self._trust_state else "",
            f"Total Value: {total_value(self._state.get('actions', self.baselines), self.value_params):.2f}",
            f"{'='*60}\n"
        ])
        
        return "\n".join(output_lines)
    
    def close(self):
        """Clean up resources."""
        pass


class CoopetitionEnv(gym.Env, AbstractCoopetitionEnv):
    """
    Gymnasium-compatible environment (backward compatible with v0.1.0).
    
    This class wraps AbstractCoopetitionEnv with the standard Gymnasium API.
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        **kwargs
    ):
        """
        Initialize the Gymnasium-compatible environment.
        
        Args:
            config: Environment configuration
            **kwargs: Additional arguments passed to config if config is None
        """
        if config is None:
            config = EnvironmentConfig(**kwargs)
        
        # Use full observability for backward compatibility
        AbstractCoopetitionEnv.__init__(self, config, ObservationConfig.full_observability())
        
        # Define Gymnasium spaces
        obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=float(np.max(self.endowments)),
            shape=(self.n_agents,), dtype=np.float32
        )
        
        self.render_mode = config.render_mode
    
    def _compute_obs_dim(self) -> int:
        """Compute observation dimension for legacy mode."""
        # actions(n) + trust(n²) + reputation(n²) + interdep(n²) + step(1)
        return self.n_agents + 3 * (self.n_agents ** 2) + 1
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Gymnasium reset interface."""
        # Call parent reset for proper Gymnasium seeding
        super().reset(seed=seed)
        # Create our own generator with the same seed for consistency
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()
        self._init_state()
        obs = self._get_legacy_observation()
        info = self._get_legacy_info()
        return obs, info
    
    def step(
        self,
        action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Gymnasium step interface."""
        action = np.asarray(action, dtype=np.float32)
        actions_dict = {
            agent: np.array([action[i]]) 
            for i, agent in enumerate(self.possible_agents)
        }
        
        results = self.process_actions(actions_dict)
        
        obs = self._get_legacy_observation()
        rewards = np.array([results["rewards"][a] for a in self.possible_agents], dtype=np.float32)
        terminated = results["terminations"][self.possible_agents[0]]
        truncated = results["truncations"][self.possible_agents[0]]
        info = self._get_legacy_info()
        info["actions"] = self._state["actions"].copy()
        
        return obs, rewards, terminated, truncated, info
    
    def _get_legacy_observation(self) -> np.ndarray:
        """Build v0.1.0 compatible observation (full visibility)."""
        return np.concatenate([
            self._state["actions"],
            self._state["trust"].flatten(),
            self._state["reputation"].flatten(),
            self.D.flatten(),
            np.array([self._step_count], dtype=np.float32)
        ]).astype(np.float32)
    
    def _get_legacy_info(self) -> Dict[str, Any]:
        """Build v0.1.0 compatible info dict."""
        return {
            "step": self._step_count,
            "mean_trust": float(self._trust_state.mean_trust()) if self._trust_state else 1.0,
            "mean_reputation_damage": float(self._trust_state.mean_reputation_damage()) if self._trust_state else 0.0,
            "total_value": float(total_value(self._state["actions"], self.value_params)),
            "mean_cooperation": float(np.mean(self._state["actions"])),
            "cooperation_rate": float(np.mean(self._state["actions"] / self.endowments)),
        }
    
    def render(self) -> Optional[str]:
        """Render current state."""
        if self.render_mode is None:
            return None
        
        output = AbstractCoopetitionEnv.render(self)
        
        if self.render_mode == "human":
            print(output)
            return None
        return output


class MultiAgentCoopetitionEnv(CoopetitionEnv):
    """
    Extended CoopetitionEnv with PettingZoo-style multi-agent interface hints.
    
    Provides additional methods and properties that align with PettingZoo ParallelEnv.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Per-agent spaces
        self._observation_spaces = {
            agent: self.observation_space for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=float(self.endowments[i]),
                shape=(1,),
                dtype=np.float32
            )
            for i, agent in enumerate(self.possible_agents)
        }
    
    def observation_space_for(self, agent: str) -> spaces.Space:
        """Get observation space for a specific agent."""
        return self._observation_spaces[agent]
    
    def action_space_for(self, agent: str) -> spaces.Space:
        """Get action space for a specific agent."""
        return self._action_spaces[agent]
    
    def step_dict(
        self,
        actions: Dict[str, NDArray]
    ) -> Tuple[Dict[str, NDArray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """
        PettingZoo-style step with dictionary inputs/outputs.
        """
        # Convert dict actions to array
        action_array = np.array([
            actions[agent].flatten()[0] for agent in self.possible_agents
        ], dtype=np.float32)
        
        # Call parent step
        obs, rewards, terminated, truncated, info = self.step(action_array)
        
        # Convert to dicts
        observations = {agent: obs for agent in self.possible_agents}
        rewards_dict = {agent: rewards[i] for i, agent in enumerate(self.possible_agents)}
        terminateds = {agent: terminated for agent in self.possible_agents}
        truncateds = {agent: truncated for agent in self.possible_agents}
        infos = {agent: info.copy() for agent in self.possible_agents}
        
        return observations, rewards_dict, terminateds, truncateds, infos
