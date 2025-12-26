"""
================================================================================
COOPETITION-GYM: Utilities Module
================================================================================

Utility functions for analysis, episode running, and baseline policies.

Authors: Vik Pant, Eric Yu
License: MIT
================================================================================
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
import json


@dataclass
class EpisodeResult:
    """Container for episode evaluation results."""
    total_steps: int
    total_rewards: NDArray
    mean_reward_per_step: NDArray
    final_trust: float
    mean_cooperation_rate: float
    actions_history: List[NDArray]
    rewards_history: List[NDArray]
    trust_history: List[float]
    terminated: bool
    truncated: bool
    
    def to_dict(self) -> Dict:
        return {
            "total_steps": self.total_steps,
            "total_rewards": self.total_rewards.tolist(),
            "final_trust": self.final_trust,
            "mean_cooperation_rate": self.mean_cooperation_rate,
        }


def run_episode(env, policy: Optional[Callable] = None, seed: Optional[int] = None) -> EpisodeResult:
    """Run a complete episode with optional policy."""
    obs, info = env.reset(seed=seed)
    
    actions_history, rewards_history = [], []
    trust_history = [info.get("mean_trust", 1.0)]
    done, terminated, truncated = False, False, False
    
    while not done:
        action = policy(obs) if policy else env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        actions_history.append(np.asarray(action))
        rewards_history.append(np.asarray(rewards))
        trust_history.append(info.get("mean_trust", 1.0))
    
    rewards_array = np.array(rewards_history)
    actions_array = np.array(actions_history)
    coop_rates = actions_array / env.endowments
    
    return EpisodeResult(
        total_steps=len(actions_history),
        total_rewards=np.sum(rewards_array, axis=0),
        mean_reward_per_step=np.mean(rewards_array, axis=0),
        final_trust=trust_history[-1],
        mean_cooperation_rate=float(np.mean(coop_rates)),
        actions_history=actions_history,
        rewards_history=rewards_history,
        trust_history=trust_history,
        terminated=terminated,
        truncated=truncated
    )


def make_constant_policy(level: float):
    """Policy that always cooperates at a fixed level."""
    def policy(obs):
        return np.array([level, level])
    return policy


def make_proportional_policy(fraction: float = 0.5):
    """Policy that invests a fraction of endowment."""
    def policy(obs):
        return np.array([100.0 * fraction, 100.0 * fraction])
    return policy


def aggregate_results(results: List[EpisodeResult]) -> Dict:
    """Compute aggregate statistics over multiple episodes."""
    return {
        "n_episodes": len(results),
        "mean_total_reward": np.mean([r.total_rewards for r in results], axis=0).tolist(),
        "mean_final_trust": float(np.mean([r.final_trust for r in results])),
        "mean_cooperation_rate": float(np.mean([r.mean_cooperation_rate for r in results])),
    }


__all__ = ["EpisodeResult", "run_episode", "make_constant_policy", 
           "make_proportional_policy", "aggregate_results"]
