"""Algorithm implementations used in the NeurIPS 2026 benchmark.

This module contains the exact algorithm implementations used to produce
the released training dataset (25,708 files). It consolidates the original
``algorithms.py`` and ``algorithms_extended.py`` from the campaign source
tree into a single module. Implementations are preserved byte-identical
to the originals; no behavioral changes were made during consolidation.

Classes defined here fall into four groups:

* **Heuristic baselines**: :class:`RandomPolicy`, :class:`TitForTatPolicy`,
  and the :class:`ConstantPolicy` family (101 instances parameterized by
  cooperation level).

* **Game-theoretic oracles** (7): :class:`CoopetitiveEquilibriumOracle`,
  :class:`NashEquilibriumOracle`, :class:`SocialOptimumOracle`,
  :class:`TrustAwareEquilibriumOracle`, :class:`LoyaltyAugmentedOracle`,
  :class:`ReciprocityEquilibriumOracle`, :class:`BoundedReciprocityOracle`.
  These compute actions analytically from environment parameters; they do
  not train.

* **Training algorithms** (16): :class:`IndependentPPO`,
  :class:`IndependentSAC`, :class:`IndependentA2C`,
  :class:`IndependentREINFORCE`, :class:`MAPPO`, :class:`MADDPG`,
  :class:`MATD3`, :class:`MASAC`, :class:`QMIX`, :class:`VDN`,
  :class:`COMA`, :class:`LOLA`, :class:`M3DDPG`, :class:`SelfPlayPPO`,
  :class:`FictitiousCoPlay`, :class:`MeanFieldActorCritic`.

* **Experimental, not used in paper results**:
  :class:`DynamicLoyaltyOracle`. Kept for reference but not registered in
  :mod:`experiments.config`. Do not use to reproduce paper numbers.

All trainable algorithms inherit from :class:`BaseAlgorithm` and implement a
common API: ``__init__(env, device, seed, **params)``, ``train(total_timesteps)``,
``predict(obs, deterministic=True)``, ``save(path)``, and ``load(path)``.

See :mod:`experiments.config` for the algorithm-to-environment applicability
matrix (e.g. MeanFieldAC restriction to N>=3 environments, TR-specific oracle
restrictions).
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from collections import deque
from contextlib import nullcontext
import torch
import torch.nn as nn

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)


# =============================================================================
# Multiprocessing-safe coopetition_gym import
# =============================================================================

def _import_coopetition_gym():
    """Import ``coopetition_gym`` from the installed package.

    The repository layout has a top-level folder ``coopetition_gym/`` with
    the actual package at ``coopetition_gym/coopetition_gym/``. When running
    from the repository root (or a multiprocessing worker launched from
    there), Python resolves ``coopetition_gym`` to the outer folder as a
    namespace package, shadowing the installed editable package. This helper
    inserts the inner package parent at the front of ``sys.path`` and drops
    any stale namespace-package import.

    Safe to call from the main process or worker subprocesses.
    """
    import importlib

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inner_package_parent = os.path.join(repo_root, "coopetition_gym")

    sys.modules.pop("coopetition_gym", None)
    if inner_package_parent not in sys.path:
        sys.path.insert(0, inner_package_parent)
    return importlib.import_module("coopetition_gym")


# ============================================================================
# Efficient Replay Buffer (replaces collections.deque)
# ============================================================================

class ReplayBuffer:
    """Replay buffer using pre-allocated numpy arrays.

    Provides O(1) insertion and O(1) random-access sampling,
    replacing collections.deque which has O(n) random access.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = np.sum(reward) if isinstance(reward, np.ndarray) else reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a batch. Returns contiguous numpy arrays."""
        indices = np.random.choice(self.size, batch_size, replace=True)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


# ============================================================================
# Base Algorithm Class
# ============================================================================

class BaseAlgorithm(ABC):
    """Base class for all algorithms."""

    def __init__(
        self,
        env,
        device: str = "cpu",
        seed: int = 0,
        **kwargs
    ):
        self.env = env
        self.device = device
        self.seed = seed
        self.kwargs = kwargs

        # Set seeds
        np.random.seed(seed)

        # Extract environment info
        self.obs_space = env.observation_space
        self.action_space = env.action_space

        # Training curve recording
        self.training_returns = []
        self.training_timesteps = []  # Timestep at which each episode ended
        self.training_metrics = TrainingMetrics()

        # Determine number of agents
        if hasattr(env, 'num_agents'):
            self.n_agents = env.num_agents
        elif hasattr(self.action_space, 'shape') and len(self.action_space.shape) > 0:
            # Infer from action space shape
            self.n_agents = self.action_space.shape[0] if len(self.action_space.shape) == 1 else 1
        else:
            self.n_agents = 2  # Default for dyadic

    @abstractmethod
    def train(self, total_timesteps: int):
        """Train the algorithm."""
        pass

    def train_with_callback(self, total_timesteps: int, callback=None):
        """
        Train the algorithm with optional progress callback.

        Parameters
        ----------
        total_timesteps : int
            Total number of timesteps to train
        callback : callable, optional
            Function called periodically with current timestep: callback(step)
            Used for progress logging and checkpointing.

        This default implementation simply calls train() without callbacks.
        Subclasses should override to support progress callbacks.
        """
        # Default: just call train without callback support
        self.train(total_timesteps)

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action given observation."""
        pass

    def save(self, path: str):
        """Save model to path."""
        pass

    def load(self, path: str):
        """Load model from path."""
        pass

    def close(self):
        """Cleanup resources."""
        pass


class TrainingMetrics:
    """Lightweight metric accumulator for algorithm-level logging.

    Records per-update metrics in a buffer, flushed to history at regular intervals.
    Designed for 1M+ timestep runs: stores ~200 data points (flushed every 5000 steps).
    """

    def __init__(self, log_interval=5000):
        self.log_interval = log_interval
        self._accum = {}
        self._counts = {}
        self.history = {}

    def record(self, name, value):
        if name not in self._accum:
            self._accum[name] = 0.0
            self._counts[name] = 0
            self.history[name] = []
        self._accum[name] += float(value)
        self._counts[name] += 1

    def flush(self, timestep):
        for name in list(self._accum.keys()):
            if self._counts[name] > 0:
                avg = self._accum[name] / self._counts[name]
                self.history[name].append([timestep, round(avg, 6)])
                self._accum[name] = 0.0
                self._counts[name] = 0

    def to_dict(self):
        # Flush any remaining un-flushed data before export
        if any(self._counts.get(n, 0) > 0 for n in self._accum):
            self.flush(-1)  # -1 indicates final flush
        return dict(self.history)


# ============================================================================
# Heuristic Policies (No Training)
# ============================================================================

class RandomPolicy(BaseAlgorithm):
    """Uniformly random actions."""

    def train(self, total_timesteps: int):
        """No training needed."""
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return random action."""
        return self.action_space.sample()


class ConstantPolicy(BaseAlgorithm):
    """Fixed cooperation level policy."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, level: float = 0.5, **kwargs):
        super().__init__(env, device, seed, **kwargs)
        self.level = level

        # Compute constant action
        if hasattr(self.action_space, 'high'):
            self.constant_action = self.level * self.action_space.high
        else:
            self.constant_action = np.full(self.action_space.shape, self.level * 100)

    def train(self, total_timesteps: int):
        """No training needed."""
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return constant action."""
        return self.constant_action.astype(np.float32)


class TitForTatPolicy(BaseAlgorithm):
    """Tit-for-Tat strategy adapted for continuous actions."""

    def __init__(self, env, device: str = "cpu", seed: int = 0,
                 initial_level: float = 0.5, **kwargs):
        super().__init__(env, device, seed, **kwargs)
        self.initial_level = initial_level

        # Initial action
        if hasattr(self.action_space, 'high'):
            self.last_action = self.initial_level * self.action_space.high
        else:
            self.last_action = np.full(self.action_space.shape, self.initial_level * 100)

        self.first_step = True

    def train(self, total_timesteps: int):
        """No training needed."""
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Mirror partner's last action."""
        if self.first_step:
            self.first_step = False
            return self.last_action.astype(np.float32)

        # Extract partner actions from observation
        # Assuming obs contains [agent_0_action, agent_1_action, ...]
        try:
            # Get partner's last action and match it
            if len(obs) >= self.n_agents:
                partner_actions = obs[:self.n_agents]
                # Mirror: each agent plays what their partner played
                action = np.zeros(self.n_agents)
                for i in range(self.n_agents):
                    partner_idx = (i + 1) % self.n_agents
                    action[i] = partner_actions[partner_idx]

                # Clip to valid range
                if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                    action = np.clip(action, self.action_space.low, self.action_space.high)

                self.last_action = action
                return action.astype(np.float32)
        except Exception:
            pass

        return self.last_action.astype(np.float32)


# ============================================================================
# Oracle Policies (Equilibrium-Based Baselines)
# ============================================================================

class CoopetitiveEquilibriumOracle(BaseAlgorithm):
    """
    Oracle policy that computes Coopetitive Equilibrium actions.

    Based on TR-1: "Computational Foundations for Strategic Coopetition"
    Uses best-response iteration to find equilibrium actions given
    environment parameters (gamma, theta, interdependence matrix).

    This provides an upper bound on what rational agents should achieve
    if they had perfect knowledge of the game structure.
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        # Extract environment parameters
        self.endowments = getattr(env, 'endowments', np.full(self.n_agents, 100.0))
        self.alpha = getattr(env, 'alpha', np.full(self.n_agents, 1.0 / self.n_agents))
        self.D = getattr(env, 'D', np.eye(self.n_agents) * 0.5)

        # Get value function parameters
        if hasattr(env, 'config') and hasattr(env.config, 'value_params'):
            self.gamma = getattr(env.config.value_params, 'gamma', 0.65)
            self.theta = getattr(env.config.value_params, 'theta', 20.0)
        else:
            self.gamma = 0.65
            self.theta = 20.0

        # Compute equilibrium actions once
        self.equilibrium_actions = self._compute_equilibrium()

    def _log_f(self, a: float) -> float:
        """Logarithmic value function: f(a) = theta * ln(1 + a)"""
        return self.theta * np.log(1 + max(a, 0))

    def _synergy_g(self, actions: np.ndarray) -> float:
        """Synergy function: geometric mean of actions."""
        positive_actions = np.maximum(actions, 1e-10)
        return np.prod(positive_actions) ** (1.0 / len(actions))

    def _utility(self, i: int, actions: np.ndarray) -> float:
        """
        Compute integrated utility for agent i (TR-1 Equation 13).

        U_i(a) = π_i(a) + Σ_{j≠i} D_ij · π_j(a)

        Where private payoff: π_k(a) = (e_k - a_k) + f(a_k) + α_k·S(a)
        And synergistic surplus: S(a) = γ·g(a)
        """
        # Synergistic surplus: S(a) = γ·g(a)
        synergy = self.gamma * self._synergy_g(actions)

        # Compute all private payoffs: π_k = (e_k - a_k) + f(a_k) + α_k·S(a)
        private_payoffs = np.zeros(self.n_agents)
        for k in range(self.n_agents):
            private_payoffs[k] = (
                (self.endowments[k] - actions[k]) +
                self._log_f(actions[k]) +
                self.alpha[k] * synergy
            )

        # Integrated utility: U_i = π_i + Σ_{j≠i} D_ij · π_j
        utility = private_payoffs[i]
        for j in range(self.n_agents):
            if i != j:
                utility += self.D[i, j] * private_payoffs[j]

        return utility

    def _best_response(self, i: int, actions: np.ndarray) -> float:
        """Find best response for agent i given others' actions."""
        from scipy.optimize import minimize_scalar

        def neg_utility(a_i):
            test_actions = actions.copy()
            test_actions[i] = a_i
            return -self._utility(i, test_actions)

        result = minimize_scalar(
            neg_utility,
            bounds=(0, self.endowments[i]),
            method='bounded'
        )
        return result.x

    def _compute_equilibrium(self, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Compute Coopetitive Equilibrium via best-response iteration."""
        # Initialize at midpoint
        actions = self.endowments * 0.5

        for _ in range(max_iter):
            old_actions = actions.copy()

            for i in range(self.n_agents):
                actions[i] = self._best_response(i, actions)

            if np.max(np.abs(actions - old_actions)) < tol:
                break

        return actions

    def train(self, total_timesteps: int):
        """No training needed - equilibrium computed analytically."""
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return equilibrium actions."""
        return self.equilibrium_actions.astype(np.float32)


class NashEquilibriumOracle(BaseAlgorithm):
    """
    Oracle policy that computes Nash Equilibrium for team production.

    Based on TR-3: "Formalizing Collective Action and Loyalty"
    For team production games, the symmetric Nash equilibrium is:
    a* = (omega * beta / (c * n^(2-beta)))^(1/(1-beta))

    This represents the free-riding equilibrium where each agent
    contributes just enough given others' contributions.
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        # TR-3 specific parameters
        if hasattr(env, 'tr3_params'):
            params = env.tr3_params
            self.omega = getattr(params, 'omega', 1.0)
            self.beta = getattr(params, 'beta', 0.5)
            self.c = getattr(params, 'c', 1.0)
        else:
            # Default team production parameters
            self.omega = 1.0
            self.beta = 0.5
            self.c = 1.0

        # Compute Nash equilibrium effort
        self.nash_effort = self._compute_nash_equilibrium()

        # Get action bounds
        if hasattr(self.action_space, 'high'):
            self.action_high = self.action_space.high
        else:
            self.action_high = np.full(self.n_agents, 100.0)

    def _compute_nash_equilibrium(self) -> float:
        """
        Analytical free-riding equilibrium: a* = (omega*beta / (c * n^(2-beta)))^(1/(1-beta))
        """
        numerator = self.omega * self.beta
        denominator = self.c * (self.n_agents ** (2 - self.beta))

        if self.beta < 1.0 and denominator > 0:
            nash = (numerator / denominator) ** (1.0 / (1.0 - self.beta))
        else:
            nash = 0.5 * 100  # Fallback to midpoint

        return nash

    def train(self, total_timesteps: int):
        """No training needed - equilibrium computed analytically."""
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return Nash equilibrium actions for all agents."""
        actions = np.full(self.n_agents, self.nash_effort, dtype=np.float32)

        # Clip to valid range
        actions = np.clip(actions, 0, self.action_high)

        return actions


class SocialOptimumOracle(BaseAlgorithm):
    """
    Oracle policy that plays the socially optimal (cooperative) actions.

    For team production games, the social optimum maximizes total welfare.
    This represents the Pareto-efficient outcome that a benevolent
    social planner would implement.

    Social optimum effort: a_SO = (omega * beta / c)^(1/(1-beta))
    (Higher than Nash because it internalizes positive externalities)
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        # TR-3 specific parameters
        if hasattr(env, 'tr3_params'):
            params = env.tr3_params
            self.omega = getattr(params, 'omega', 1.0)
            self.beta = getattr(params, 'beta', 0.5)
            self.c = getattr(params, 'c', 1.0)
        else:
            self.omega = 1.0
            self.beta = 0.5
            self.c = 1.0

        # Compute social optimum effort
        self.social_optimum_effort = self._compute_social_optimum()

        # Get action bounds
        if hasattr(self.action_space, 'high'):
            self.action_high = self.action_space.high
        else:
            self.action_high = np.full(self.n_agents, 100.0)

    def _compute_social_optimum(self) -> float:
        """
        Social optimum: a_SO = (omega * beta / c)^(1/(1-beta)) / n
        Division by n distributes the socially optimal total effort equally across agents.
        """
        numerator = self.omega * self.beta
        denominator = self.c

        if self.beta < 1.0 and denominator > 0:
            social_opt = (numerator / denominator) ** (1.0 / (1.0 - self.beta)) / self.n_agents
        else:
            social_opt = 0.75 * 100  # Fallback to higher cooperation

        return social_opt

    def train(self, total_timesteps: int):
        """No training needed - optimum computed analytically."""
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return socially optimal actions for all agents."""
        actions = np.full(self.n_agents, self.social_optimum_effort, dtype=np.float32)

        # Clip to valid range
        actions = np.clip(actions, 0, self.action_high)

        return actions


class TrustAwareEquilibriumOracle(BaseAlgorithm):
    """
    Oracle policy that computes trust-aware equilibrium actions.

    Based on TR-2: "Formalizing Trust and Reputation Dynamics"
    Extends TR-1 Coopetitive Equilibrium with trust-dependent utility:

    U_i(a, T) = π_i(a) + Σ D_ij·π_j(a) + ρ·T_ij·(a_j - baseline_j)·a_i

    The trust term creates incentive for higher cooperation when trust is high,
    and accounts for how sustained cooperation builds trust over time.

    This Oracle approximates the steady-state optimal policy by:
    1. Computing base TR-1 equilibrium
    2. Adjusting upward for expected trust benefits from sustained cooperation
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        # Extract environment parameters (TR-1 base)
        self.endowments = getattr(env, 'endowments', np.full(self.n_agents, 100.0))
        self.alpha = getattr(env, 'alpha', np.full(self.n_agents, 1.0 / self.n_agents))
        self.D = getattr(env, 'D', np.eye(self.n_agents) * 0.5)
        self.baselines = getattr(env, 'baselines', self.endowments * 0.3)

        # Get value function parameters
        if hasattr(env, 'config') and hasattr(env.config, 'value_params'):
            self.gamma = getattr(env.config.value_params, 'gamma', 0.65)
            self.theta = getattr(env.config.value_params, 'theta', 20.0)
        else:
            self.gamma = 0.65
            self.theta = 20.0

        # TR-2 trust parameters
        if hasattr(env, 'config') and hasattr(env.config, 'trust_params'):
            trust_params = env.config.trust_params
            self.rho = getattr(trust_params, 'rho', 0.2)
            self.lambda_plus = getattr(trust_params, 'lambda_plus', 0.10)
            self.lambda_minus = getattr(trust_params, 'lambda_minus', 0.30)
        else:
            self.rho = 0.2
            self.lambda_plus = 0.10
            self.lambda_minus = 0.30

        # Compute trust-aware equilibrium
        self.equilibrium_actions = self._compute_trust_aware_equilibrium()

    def _log_f(self, a: float) -> float:
        """Logarithmic value function: f(a) = theta * ln(1 + a)"""
        return self.theta * np.log(1 + max(a, 0))

    def _synergy_g(self, actions: np.ndarray) -> float:
        """Synergy function: geometric mean of actions."""
        positive_actions = np.maximum(actions, 1e-10)
        return np.prod(positive_actions) ** (1.0 / len(actions))

    def _utility_with_trust(self, i: int, actions: np.ndarray, trust: float) -> float:
        """
        Compute TR-2 trust-augmented utility for agent i.

        U_i^T(a) = π_i(a) + Σ_{j≠i} D_ij·π_j(a) + ρ·T_ij·(a_j - baseline_j)·a_i

        Where private payoff: π_k(a) = (e_k - a_k) + f(a_k) + α_k·S(a)
        And synergistic surplus: S(a) = γ·g(a)
        """
        a_i = actions[i]

        # Synergistic surplus: S(a) = γ·g(a)
        synergy = self.gamma * self._synergy_g(actions)

        # Compute all private payoffs: π_k = (e_k - a_k) + f(a_k) + α_k·S(a)
        private_payoffs = np.zeros(self.n_agents)
        for k in range(self.n_agents):
            private_payoffs[k] = (
                (self.endowments[k] - actions[k]) +
                self._log_f(actions[k]) +
                self.alpha[k] * synergy
            )

        # Integrated utility: U_i = π_i + Σ_{j≠i} D_ij · π_j
        utility = private_payoffs[i]
        trust_benefit = 0.0
        for j in range(self.n_agents):
            if i != j:
                utility += self.D[i, j] * private_payoffs[j]

                # TR-2 trust reciprocity term
                trust_benefit += self.rho * trust * (actions[j] - self.baselines[j]) * a_i

        return utility + trust_benefit

    def _compute_trust_aware_equilibrium(self) -> np.ndarray:
        """
        Compute trust-aware equilibrium via iterative best-response.

        Key insight: At steady-state with sustained cooperation, trust approaches 1.0.
        We compute equilibrium assuming steady-state high trust.
        """
        from scipy.optimize import minimize_scalar

        # Assume steady-state trust from sustained cooperation
        steady_state_trust = 0.95  # High trust from predictable cooperation

        # Initialize at midpoint
        actions = self.endowments * 0.5

        def best_response(i: int, actions: np.ndarray) -> float:
            def neg_utility(a_i):
                test_actions = actions.copy()
                test_actions[i] = a_i
                return -self._utility_with_trust(i, test_actions, steady_state_trust)

            result = minimize_scalar(
                neg_utility,
                bounds=(0, self.endowments[i]),
                method='bounded'
            )
            return result.x

        # Best-response iteration
        for _ in range(100):
            old_actions = actions.copy()
            for i in range(self.n_agents):
                actions[i] = best_response(i, actions)
            if np.max(np.abs(actions - old_actions)) < 1e-6:
                break

        return actions

    def train(self, total_timesteps: int):
        """No training needed - equilibrium computed analytically."""
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return trust-aware equilibrium actions."""
        return self.equilibrium_actions.astype(np.float32)


class LoyaltyAugmentedOracle(BaseAlgorithm):
    """
    Oracle policy for TR-3 environments with loyalty mechanisms.

    Based on TR-3: "Formalizing Collective Action and Loyalty"
    Computes optimal actions under loyalty-augmented utility:

    U_i(a; θ_i) = (1/n)·Q(a) - c·(1 - φ_C·θ_i)·a_i + φ_B·θ_i·π̄_{-i}

    Where:
    - Q(a) = ω·(Σa_i)^β is team production
    - θ_i is loyalty level (0 to 1)
    - φ_B is loyalty benefit strength (welfare internalization)
    - φ_C is cost tolerance strength

    At high loyalty (θ → 1), agents approach social optimum.
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        # TR-3 parameters
        if hasattr(env, 'tr3_params'):
            params = env.tr3_params
            self.omega = getattr(params, 'omega', 1.0)
            self.beta = getattr(params, 'beta', 0.5)
            self.c = getattr(params, 'c', 1.0)
            self.phi_B = getattr(params, 'phi_B', 0.8)
            self.phi_C = getattr(params, 'phi_C', 0.3)
        else:
            self.omega = 1.0
            self.beta = 0.5
            self.c = 1.0
            self.phi_B = 0.8
            self.phi_C = 0.3

        # Assume high loyalty at steady-state
        self.loyalty = 0.9

        # Compute loyalty-augmented equilibrium
        self.equilibrium_effort = self._compute_loyalty_equilibrium()

        # Get action bounds
        if hasattr(self.action_space, 'high'):
            self.action_high = self.action_space.high
        else:
            self.action_high = np.full(self.n_agents, 100.0)

    def _compute_loyalty_equilibrium(self) -> float:
        """
        Compute equilibrium effort under loyalty-augmented utility.

        With loyalty θ, the effective cost is c·(1 - φ_C·θ) and agents
        internalize φ_B·θ of teammates' welfare.

        The equilibrium interpolates between Nash (θ=0) and social optimum (θ=1).
        """
        # Effective cost reduction from loyalty
        effective_c = self.c * (1 - self.phi_C * self.loyalty)

        # Welfare internalization factor
        internalization = 1 + self.phi_B * self.loyalty * (self.n_agents - 1)

        # Modified equilibrium: accounts for both cost reduction and internalization
        # This is an approximation; full solution requires solving FOC
        numerator = self.omega * self.beta * internalization
        denominator = self.n_agents * effective_c

        if self.beta < 1.0 and denominator > 0:
            effort = (numerator / denominator) ** (1.0 / (1.0 - self.beta))
        else:
            effort = 0.75 * 100  # Fallback

        return effort

    def train(self, total_timesteps: int):
        """No training needed - equilibrium computed analytically."""
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return loyalty-augmented equilibrium actions."""
        actions = np.full(self.n_agents, self.equilibrium_effort, dtype=np.float32)
        actions = np.clip(actions, 0, self.action_high)
        return actions


class ReciprocityEquilibriumOracle(BaseAlgorithm):
    """
    Oracle policy computing the reciprocity-augmented equilibrium.

    Based on TR-4 Algorithm 1: iterative best-response solver under the
    complete utility function with reciprocity modifier (Eq 44):

    U_i = π_base + U_interdep + U_trust + U_recip

    At the cooperative steady state, cooperation signals s_ij = 0 (no deviation
    from norm) so the reciprocity modifier itself is zero. However, reciprocity
    shifts the equilibrium UPWARD because each agent internalizes the marginal
    deterrence value: deviating downward from the cooperative level triggers
    bounded punishment φ(s) = tanh(κs) from all partners (Proposition 4.7).

    The equilibrium incorporates this deterrence as a shadow incentive:
    ∂U_recip/∂a_i ≈ λ_R · Σ_j T_ij · (1+ω·D_ij) · ρ_ij · κ · (1/endow_i)

    This term (from φ'(0) = κ) increases the marginal benefit of cooperation,
    pushing equilibrium actions above the TR-1/TR-2 baseline.
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        # TR-1 base parameters
        self.endowments = getattr(env, 'endowments', np.full(self.n_agents, 100.0))
        self.alpha = getattr(env, 'alpha', np.full(self.n_agents, 1.0 / self.n_agents))
        self.D = getattr(env, 'D', np.eye(self.n_agents) * 0.5)
        self.baselines = getattr(env, 'baselines', self.endowments * 0.3)

        # Value function parameters
        if hasattr(env, 'config') and hasattr(env.config, 'value_params'):
            self.gamma = getattr(env.config.value_params, 'gamma', 0.65)
            self.theta = getattr(env.config.value_params, 'theta', 20.0)
        else:
            self.gamma = 0.65
            self.theta = 20.0

        # TR-2 trust parameters
        if hasattr(env, 'config') and hasattr(env.config, 'trust_params'):
            tp = env.config.trust_params
            self.initial_trust = getattr(tp, 'initial_trust', 0.50)
        else:
            self.initial_trust = 0.50

        # TR-4 reciprocity parameters
        if hasattr(env, 'tr4_params'):
            p = env.tr4_params
            self.rho_0 = p.rho_0
            self.eta = p.eta
            self.kappa_recip = p.kappa
            self.k = p.k
            self.lambda_R = p.lambda_R
            self.omega = p.omega
        else:
            self.rho_0 = 1.0
            self.eta = 1.0
            self.kappa_recip = 1.0
            self.k = 5
            self.lambda_R = 1.0
            self.omega = 0.6

        # Precompute reciprocity sensitivities: ρ_ij = ρ_0 · D_ij^η (Eq 23)
        self.rho = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    self.rho[i, j] = self.rho_0 * (max(float(self.D[i, j]), 1e-10) ** self.eta)

        # Compute steady-state equilibrium
        self.equilibrium_actions = self._compute_equilibrium()

    def _log_f(self, a: float) -> float:
        """f(a) = θ·ln(1 + a)"""
        return self.theta * np.log(1 + max(a, 0))

    def _synergy_g(self, actions: np.ndarray) -> float:
        """Synergy: geometric mean of actions."""
        positive = np.maximum(actions, 1e-10)
        return np.prod(positive) ** (1.0 / len(actions))

    def _utility(self, i: int, actions: np.ndarray, trust_level: float) -> float:
        """
        Complete utility: π_base + U_interdep + U_trust + reciprocity_incentive.

        The reciprocity incentive models the marginal deterrence value at
        the cooperative equilibrium. Agent i's cooperation above baseline is
        sustained because reducing it would generate negative signals
        s_ij < 0, triggering punishment from all partners.

        Approximation: the incentive is proportional to
        λ_R · Σ_j T_ij · (1+ω·D_ij) · ρ_ij · κ · (a_i - baseline_i) / endow_i
        """
        synergy = self.gamma * self._synergy_g(actions)

        # Private payoffs: π_k = (e_k - a_k) + f(a_k) + α_k·S(a)
        private_payoffs = np.zeros(self.n_agents)
        for k_idx in range(self.n_agents):
            private_payoffs[k_idx] = (
                (self.endowments[k_idx] - actions[k_idx]) +
                self._log_f(actions[k_idx]) +
                self.alpha[k_idx] * synergy
            )

        # Base + interdependence
        utility = private_payoffs[i]
        for j in range(self.n_agents):
            if i != j:
                utility += self.D[i, j] * private_payoffs[j]

        # TR-2 trust benefit (from sustained cooperation)
        rho_trust = 0.2  # Trust response coefficient
        for j in range(self.n_agents):
            if i != j:
                utility += rho_trust * trust_level * (actions[j] - self.baselines[j]) * actions[i] / self.endowments[i]

        # TR-4 reciprocity deterrence incentive
        cooperation_above_baseline = max(actions[i] - self.baselines[i], 0.0)
        deterrence = 0.0
        for j in range(self.n_agents):
            if i != j:
                deterrence += (
                    self.lambda_R *
                    trust_level *
                    (1 + self.omega * float(self.D[i, j])) *
                    self.rho[i, j] *
                    self.kappa_recip
                )
        utility += deterrence * cooperation_above_baseline / max(self.endowments[i], 1.0)

        return utility

    def _compute_equilibrium(self) -> np.ndarray:
        """
        Algorithm 1: Iterative best-response with reciprocity incentive.

        Assumes steady-state trust from sustained cooperation.
        """
        from scipy.optimize import minimize_scalar

        trust_level = min(0.95, self.initial_trust + 0.35)
        actions = self.endowments * 0.5

        for _ in range(100):
            old_actions = actions.copy()
            for i in range(self.n_agents):
                def neg_u(a_i, _i=i):
                    test = actions.copy()
                    test[_i] = a_i
                    return -self._utility(_i, test, trust_level)

                result = minimize_scalar(
                    neg_u,
                    bounds=(0, self.endowments[i]),
                    method='bounded'
                )
                actions[i] = result.x

            if np.max(np.abs(actions - old_actions)) < 1e-6:
                break

        return actions

    def train(self, total_timesteps: int):
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.equilibrium_actions.astype(np.float32)


class BoundedReciprocityOracle(BaseAlgorithm):
    """
    Oracle policy implementing TR-4 cooperation evolution dynamics.

    Based on TR-4 cooperation evolution equation (Section 8.4):

    a_i^{t+1} = a_i^t + α·[Σ_j λ_R·T_ij·(1+ω·D_ij)·ρ_ij·φ(s_ij)]
                - δ·(a_i^t - baseline_i)

    Where:
    - s_ij = a_j^t - ā_j^{t-k:t-1}  (cooperation signal, Eq 19)
    - φ(x) = tanh(κx)               (bounded response, Eq 21)
    - ρ_ij = ρ_0 · D_ij^η           (reciprocity sensitivity, Eq 23)
    - α = 0.12                       (adjustment rate)
    - δ = 0.05                       (mean-reversion toward baseline)

    This dynamic oracle maintains action history across steps within an
    episode. Since it controls all agents, mutual cooperation creates
    positive signals that push all actions upward, demonstrating how
    bounded reciprocity sustains cooperation above baseline.
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        # Environment parameters
        self.endowments = getattr(env, 'endowments', np.full(self.n_agents, 100.0))
        self.D = getattr(env, 'D', np.eye(self.n_agents) * 0.5)
        self.baselines = getattr(env, 'baselines', self.endowments * 0.3)

        # TR-4 parameters
        if hasattr(env, 'tr4_params'):
            p = env.tr4_params
            self.rho_0 = p.rho_0
            self.eta = p.eta
            self.kappa_recip = p.kappa
            self.k = p.k
            self.lambda_R = p.lambda_R
            self.omega = p.omega
        else:
            self.rho_0 = 1.0
            self.eta = 1.0
            self.kappa_recip = 1.0
            self.k = 5
            self.lambda_R = 1.0
            self.omega = 0.6

        # Trust initial value
        if hasattr(env, 'config') and hasattr(env.config, 'trust_params'):
            self.initial_trust = getattr(env.config.trust_params, 'initial_trust', 0.50)
        else:
            self.initial_trust = 0.50

        # Evolution dynamics parameters (from TR-4 Section 8.4)
        self.adjustment_rate = 0.12    # α
        self.mean_reversion = 0.05     # δ

        # Precompute reciprocity sensitivities
        self.rho = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    self.rho[i, j] = self.rho_0 * (max(float(self.D[i, j]), 1e-10) ** self.eta)

        # Dynamic state
        self.action_history = []
        self.current_actions = self.baselines.copy() + (self.endowments - self.baselines) * 0.4
        self.trust = np.full((self.n_agents, self.n_agents), self.initial_trust)
        np.fill_diagonal(self.trust, 1.0)

        # Episode tracking
        self._max_steps = getattr(env, 'max_steps',
                                  getattr(getattr(env, 'config', None), 'max_steps', 200))
        self._step = 0

        # Get action bounds
        if hasattr(self.action_space, 'high'):
            self.action_high = self.action_space.high
        else:
            self.action_high = self.endowments.copy()

    def _compute_signals(self) -> np.ndarray:
        """Compute cooperation signals from action history (Eq 19-20)."""
        signals = np.zeros((self.n_agents, self.n_agents))
        t = len(self.action_history)
        if t == 0:
            return signals

        k = min(self.k, t)
        last_actions = self.action_history[-1]

        for j in range(self.n_agents):
            recent = [self.action_history[-(idx + 1)][j] for idx in range(k)]
            avg_j = np.mean(recent)
            for i in range(self.n_agents):
                if i != j:
                    signals[i, j] = last_actions[j] - avg_j

        return signals

    def _update_trust(self, signals):
        """Simple trust update based on cooperation signals."""
        lambda_plus = 0.10
        lambda_minus = 0.30

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                s = signals[i, j]
                if s > 0:
                    delta_t = lambda_plus * s * max(0, 0.95 - self.trust[i, j])
                else:
                    delta_t = lambda_minus * s * self.trust[i, j]
                self.trust[i, j] = np.clip(self.trust[i, j] + delta_t, 0.0, 1.0)

    def train(self, total_timesteps: int):
        pass

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Compute next actions using cooperation evolution dynamics.

        a_i += α·[Σ_j λ_R·T_ij·(1+ω·D_ij)·ρ_ij·tanh(κ·s_ij)]
             - δ·(a_i - baseline_i)
        """
        self._step += 1

        # Detect episode boundary and reset
        if self._step > self._max_steps:
            self._step = 1
            self.action_history = []
            self.current_actions = self.baselines.copy() + (self.endowments - self.baselines) * 0.4
            self.trust = np.full((self.n_agents, self.n_agents), self.initial_trust)
            np.fill_diagonal(self.trust, 1.0)

        # Compute cooperation signals
        signals = self._compute_signals()

        if len(self.action_history) > 0:
            self._update_trust(signals)

        # Cooperation evolution equation (TR-4 Section 8.4)
        new_actions = self.current_actions.copy()
        for i in range(self.n_agents):
            reciprocity_push = 0.0
            for j in range(self.n_agents):
                if i != j:
                    phi_s = np.tanh(self.kappa_recip * signals[i, j])
                    reciprocity_push += (
                        self.lambda_R *
                        self.trust[i, j] *
                        (1 + self.omega * float(self.D[i, j])) *
                        self.rho[i, j] *
                        phi_s
                    )

            # Update: a_i += α·push - δ·(a_i - baseline)
            mean_reversion_term = self.mean_reversion * (self.current_actions[i] - self.baselines[i])
            new_actions[i] += (
                self.adjustment_rate * reciprocity_push * self.endowments[i]
                - mean_reversion_term
            )

        # Clip to valid range
        new_actions = np.clip(new_actions, 0, self.action_high)

        # Store and advance
        self.current_actions = new_actions.copy()
        self.action_history.append(new_actions.copy())

        # Bound history length
        if len(self.action_history) > self.k + 10:
            self.action_history = self.action_history[-(self.k + 10):]

        return new_actions.astype(np.float32)


# ============================================================================
# Environment Wrapper for SB3 Compatibility
# ============================================================================

import gymnasium as gym


class MultiAgentToSingleAgentWrapper(gym.Env):
    """
    Wrapper to convert multi-agent coopetition_gym environments to single-agent
    format for stable-baselines3 compatibility.

    This wrapper:
    1. Sums multi-agent rewards into a scalar (cooperative objective)
    2. Keeps all other env properties intact
    3. Inherits from gymnasium.Env for SB3 compatibility
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Episode return tracking for training curves
        self._episode_return = 0.0
        self.episode_returns = []
        self.episode_timesteps = []  # Timestep at which each episode ended
        self._total_steps = 0  # Running timestep counter

        # Copy other attributes
        if hasattr(env, 'num_agents'):
            self.num_agents = env.num_agents
        if hasattr(env, 'render_mode'):
            self.render_mode = env.render_mode

    def reset(self, seed=None, options=None):
        self._episode_return = 0.0
        # Pass seed to underlying env
        if seed is not None:
            result = self.env.reset(seed=seed)
        else:
            result = self.env.reset()

        if isinstance(result, tuple):
            obs, info = result
            return obs, info
        return result, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += 1

        # Convert multi-agent reward array to scalar sum
        if isinstance(reward, np.ndarray):
            scalar_reward = float(np.sum(reward))
        elif isinstance(reward, (list, tuple)):
            scalar_reward = float(sum(reward))
        else:
            scalar_reward = float(reward)

        self._episode_return += scalar_reward
        if terminated or truncated:
            self.episode_returns.append(self._episode_return)
            self.episode_timesteps.append(self._total_steps)
            self._episode_return = 0.0

        return obs, scalar_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# ============================================================================
# SB3 Metric Capture via Logger Output Format
# ============================================================================

def _create_metric_capture_format(training_metrics, flush_interval=5000):
    """
    Factory for SB3 logger KVWriter that captures training metrics at dump() time.

    SB3's logger.dump() calls write() on all output formats BEFORE clearing
    name_to_value. This guarantees metric capture for both on-policy (PPO/A2C)
    and off-policy (SAC/TD3) algorithms, unlike callback-based capture which
    fails for off-policy because dump() clears metrics immediately after train().

    IMPORTANT: Must inherit from stable_baselines3.common.logger.KVWriter
    because Logger.dump() checks isinstance(_format, KVWriter) before calling
    write(). Without this inheritance, write() is silently never called.

    Usage: Register with model.logger.output_formats AFTER _setup_learn()
    has configured the logger (e.g., in a BaseCallback._on_training_start).
    """
    from stable_baselines3.common.logger import KVWriter

    class MetricCaptureFormat(KVWriter):
        def __init__(self):
            self.training_metrics = training_metrics
            self.flush_interval = flush_interval
            self._next_flush = flush_interval
            self._last_step = 0

        def write(self, key_values, key_excluded, step=0):
            self._last_step = step
            for key, value in key_values.items():
                if key.startswith('train/'):
                    metric_name = key.split('/')[-1]
                    try:
                        self.training_metrics.record(metric_name, float(value))
                    except (TypeError, ValueError):
                        pass
            if step >= self._next_flush:
                self.training_metrics.flush(step)
                self._next_flush = step + self.flush_interval

        def close(self):
            if self._last_step > 0:
                self.training_metrics.flush(self._last_step)

    return MetricCaptureFormat()


class ISACMetricCallback:
    """
    SB3 BaseCallback for ISAC (off-policy SAC) metric capture.

    Registers MetricCaptureFormat on the model's logger in _on_training_start(),
    which fires AFTER _setup_learn() has created the final logger. This avoids
    the bug where registering before learn() attaches to a logger that gets
    replaced by _setup_learn().

    Also handles the orchestrator's progress callback via _on_step().
    """

    def __init__(self, training_metrics, user_callback=None, flush_interval=5000):
        from stable_baselines3.common.callbacks import BaseCallback

        # We dynamically subclass BaseCallback to avoid import at module level
        outer_training_metrics = training_metrics
        outer_user_callback = user_callback
        outer_flush_interval = flush_interval

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.training_metrics = outer_training_metrics
                self.user_callback = outer_user_callback
                self.flush_interval = outer_flush_interval
                self._capture = None

            def _on_training_start(self):
                """Called AFTER _setup_learn() configures the new logger."""
                self._capture = _create_metric_capture_format(
                    self.training_metrics, self.flush_interval
                )
                if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'output_formats'):
                    self.model.logger.output_formats.append(self._capture)

            def _on_step(self):
                """Called after each env step — route to orchestrator progress callback
                and capture SAC training metrics directly from logger.name_to_value.

                SB3 off-policy algorithms (SAC) call _on_step every env step (500K+
                calls). We throttle to every 5000 steps for disk I/O and metric capture.

                IMPORTANT: We capture metrics here via name_to_value (like SB3ProgressCallback)
                rather than relying solely on MetricCaptureFormat + logger.dump(). The dump()
                approach fails when training completes with fewer episodes than log_interval,
                because dump() never fires and write() is never called.
                """
                if self.num_timesteps % 5000 == 0:
                    # Progress callback
                    if self.user_callback is not None:
                        try:
                            self.user_callback(self.num_timesteps)
                        except Exception:
                            pass

                    # Capture SAC metrics directly from logger.name_to_value
                    if self.training_metrics is not None and hasattr(self.model, 'logger'):
                        try:
                            name_to_value = getattr(self.model.logger, 'name_to_value', {})
                            for key in ('train/actor_loss', 'train/critic_loss',
                                        'train/ent_coef', 'train/ent_coef_loss',
                                        'train/learning_rate', 'train/n_updates',
                                        'train/loss'):
                                if key in name_to_value:
                                    metric_name = key.split('/')[-1]
                                    try:
                                        self.training_metrics.record(metric_name, float(name_to_value[key]))
                                    except (TypeError, ValueError):
                                        pass
                            self.training_metrics.flush(self.num_timesteps)
                        except Exception:
                            pass
                return True

            def _on_training_end(self):
                """Flush remaining metrics when learn() finishes."""
                # Final capture from name_to_value for any metrics not yet captured
                if self.training_metrics is not None and hasattr(self.model, 'logger'):
                    try:
                        name_to_value = getattr(self.model.logger, 'name_to_value', {})
                        for key in ('train/actor_loss', 'train/critic_loss',
                                    'train/ent_coef', 'train/ent_coef_loss',
                                    'train/learning_rate', 'train/n_updates',
                                    'train/loss'):
                            if key in name_to_value:
                                metric_name = key.split('/')[-1]
                                try:
                                    self.training_metrics.record(metric_name, float(name_to_value[key]))
                                except (TypeError, ValueError):
                                    pass
                        if self.training_metrics._accum:
                            self.training_metrics.flush(self.num_timesteps)
                    except Exception:
                        pass
                if self._capture:
                    self._capture.close()

        self._callback = _Callback()

    @property
    def callback(self):
        return self._callback


# ============================================================================
# SB3 Callback Wrapper for Progress Logging
# ============================================================================

class SB3ProgressCallback:
    """
    Callback wrapper for stable-baselines3 algorithms.

    Converts our simple callback(step) interface to SB3's callback format.
    SB3 callbacks are called after each rollout collection (every n_steps).

    Also captures SB3's internal training metrics (policy_loss, value_loss,
    entropy_loss, clip_fraction, approx_kl) into a TrainingMetrics object
    so they appear in the result JSON alongside custom algorithm metrics.
    """

    def __init__(self, user_callback=None, training_metrics=None):
        self.user_callback = user_callback
        self.training_metrics = training_metrics
        self.n_calls = 0

    def __call__(self, locals_dict, globals_dict=None):
        """
        Called by SB3 after each rollout.

        In SB3, `self.num_timesteps` in the model tracks total timesteps.
        We extract it from locals_dict['self'].
        """
        self.n_calls += 1
        model = locals_dict.get('self')
        if model is not None and hasattr(model, 'num_timesteps'):
            step = model.num_timesteps

            # Progress callback
            if self.user_callback is not None:
                try:
                    self.user_callback(step)
                except Exception:
                    pass  # Don't crash training on callback errors

            # Capture SB3 training metrics into our TrainingMetrics accumulator
            if self.training_metrics is not None and hasattr(model, 'logger'):
                try:
                    name_to_value = getattr(model.logger, 'name_to_value', {})
                    for key in ('train/policy_gradient_loss', 'train/value_loss',
                                'train/entropy_loss', 'train/clip_fraction',
                                'train/approx_kl', 'train/loss',
                                'train/actor_loss', 'train/critic_loss',
                                'train/ent_coef'):
                        if key in name_to_value:
                            metric_name = key.split('/')[-1]
                            self.training_metrics.record(metric_name, name_to_value[key])
                    # Flush every 5000 steps
                    if step % 5000 == 0:
                        self.training_metrics.flush(step)
                except Exception:
                    pass  # Don't crash training on metric capture errors

        return True  # Continue training


# ============================================================================
# Independent Learning Algorithms
# ============================================================================

class IndependentPPO(BaseAlgorithm):
    """Independent PPO - each agent runs PPO independently."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            raise ImportError("stable_baselines3 required for IPPO")

        # Extract hyperparameters
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.n_steps = kwargs.get('n_steps', 2048)
        self.batch_size = kwargs.get('batch_size', 64)
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.vf_coef = kwargs.get('vf_coef', 0.5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.net_arch = kwargs.get('net_arch', [128, 128])

        # Wrap environment with multi-agent to single-agent adapter
        self.wrapped_env = MultiAgentToSingleAgentWrapper(env)
        self.vec_env = DummyVecEnv([lambda e=self.wrapped_env: e])

        # Create model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            policy_kwargs=dict(net_arch=self.net_arch),
            device=device,
            seed=seed,
            verbose=0,
        )

    def train(self, total_timesteps: int):
        """Train the model."""
        sb3_callback = SB3ProgressCallback(training_metrics=self.training_metrics)
        self.model.learn(total_timesteps=total_timesteps, callback=sb3_callback)
        self.training_returns = list(self.wrapped_env.episode_returns)
        self.training_timesteps = list(self.wrapped_env.episode_timesteps)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train the model with progress callback support."""
        sb3_callback = SB3ProgressCallback(callback, training_metrics=self.training_metrics)
        self.model.learn(total_timesteps=total_timesteps, callback=sb3_callback)
        self.training_returns = list(self.wrapped_env.episode_returns)
        self.training_timesteps = list(self.wrapped_env.episode_timesteps)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, env=self.vec_env)


class IndependentSAC(BaseAlgorithm):
    """Independent SAC - each agent runs SAC independently."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            raise ImportError("stable_baselines3 required for ISAC")

        # Extract hyperparameters
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.buffer_size = kwargs.get('buffer_size', 100000)
        self.batch_size = kwargs.get('batch_size', 256)
        self.tau = kwargs.get('tau', 0.005)
        self.gamma = kwargs.get('gamma', 0.99)
        self.ent_coef = kwargs.get('ent_coef', 'auto')
        self.net_arch = kwargs.get('net_arch', [128, 128])

        # Wrap environment with multi-agent to single-agent adapter
        self.wrapped_env = MultiAgentToSingleAgentWrapper(env)
        self.vec_env = DummyVecEnv([lambda e=self.wrapped_env: e])

        # Create model
        self.model = SAC(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            policy_kwargs=dict(net_arch=self.net_arch),
            device=device,
            seed=seed,
            verbose=0,
        )

    def train(self, total_timesteps: int):
        """Train using ISACMetricCallback for proper off-policy metric capture."""
        mc = ISACMetricCallback(self.training_metrics)
        self.model.learn(total_timesteps=total_timesteps, callback=mc.callback)
        self.training_returns = list(self.wrapped_env.episode_returns)
        self.training_timesteps = list(self.wrapped_env.episode_timesteps)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train with orchestrator progress callback + off-policy metric capture."""
        mc = ISACMetricCallback(self.training_metrics, user_callback=callback)
        self.model.learn(total_timesteps=total_timesteps, callback=mc.callback)
        self.training_returns = list(self.wrapped_env.episode_returns)
        self.training_timesteps = list(self.wrapped_env.episode_timesteps)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        self.model.save(path)


class IndependentA2C(BaseAlgorithm):
    """Independent A2C - each agent runs A2C independently."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        try:
            from stable_baselines3 import A2C
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            raise ImportError("stable_baselines3 required for IA2C")

        # Extract hyperparameters
        self.learning_rate = kwargs.get('learning_rate', 7e-4)
        self.n_steps = kwargs.get('n_steps', 5)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 1.0)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.vf_coef = kwargs.get('vf_coef', 0.5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.net_arch = kwargs.get('net_arch', [128, 128])

        # Wrap environment with multi-agent to single-agent adapter
        self.wrapped_env = MultiAgentToSingleAgentWrapper(env)
        self.vec_env = DummyVecEnv([lambda e=self.wrapped_env: e])

        # Create model
        self.model = A2C(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            policy_kwargs=dict(net_arch=self.net_arch),
            device=device,
            seed=seed,
            verbose=0,
        )

    def train(self, total_timesteps: int):
        sb3_callback = SB3ProgressCallback(training_metrics=self.training_metrics)
        self.model.learn(total_timesteps=total_timesteps, callback=sb3_callback)
        self.training_returns = list(self.wrapped_env.episode_returns)
        self.training_timesteps = list(self.wrapped_env.episode_timesteps)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train the model with progress callback support."""
        sb3_callback = SB3ProgressCallback(callback, training_metrics=self.training_metrics)
        self.model.learn(total_timesteps=total_timesteps, callback=sb3_callback)
        self.training_returns = list(self.wrapped_env.episode_returns)
        self.training_timesteps = list(self.wrapped_env.episode_timesteps)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action


# ============================================================================
# CTDE Algorithms (Centralized Training, Decentralized Execution)
# ============================================================================

class MAPPO(BaseAlgorithm):
    """Multi-Agent PPO with shared critic."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.n_steps = kwargs.get('n_steps', 2048)
        self.batch_size = kwargs.get('batch_size', 64)
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.net_arch = kwargs.get('net_arch', [128, 128])

        # Observation and action dimensions
        self.obs_dim = self.obs_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # Build networks
        self.device_torch = torch.device(device)
        torch.manual_seed(seed)

        # Shared critic (takes global state)
        self.critic = self._build_network(self.obs_dim, 1).to(self.device_torch)

        # Per-agent actors
        self.actors = nn.ModuleList([
            self._build_network(self.obs_dim, 1)  # Each agent outputs 1 action
            for _ in range(self.n_agents)
        ]).to(self.device_torch)

        # Log std for actions - create parameters on device directly
        self.log_stds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, device=self.device_torch))
            for _ in range(self.n_agents)
        ])

        # Optimizer
        all_params = list(self.critic.parameters())
        for actor in self.actors:
            all_params.extend(actor.parameters())
        all_params.extend(self.log_stds)
        self.optimizer = optim.Adam(all_params, lr=self.learning_rate)

    def _build_network(self, input_dim: int, output_dim: int) -> 'nn.Module':
        import torch.nn as nn
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def train(self, total_timesteps: int):
        """Train without callback."""
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train with optional progress callback."""
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        """Internal training loop for MAPPO with optional callback support."""
        import torch

        obs, _ = self.env.reset(seed=self.seed)
        episode_return = 0.0
        timesteps = 0

        while timesteps < total_timesteps:
            # Collect rollout
            observations = []
            actions = []
            rewards = []
            dones = []
            log_probs = []
            values = []

            for _ in range(self.n_steps):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_torch)

                    # Get actions from each actor
                    agent_actions = []
                    agent_log_probs = []
                    for i, actor in enumerate(self.actors):
                        mean = actor(obs_tensor)
                        std = torch.exp(self.log_stds[i])
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        agent_actions.append(action.item())
                        agent_log_probs.append(log_prob)

                    action_array = np.array(agent_actions, dtype=np.float32)

                    # Clip actions
                    if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                        action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

                    # Get value
                    value = self.critic(obs_tensor)

                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action_array)
                done = terminated or truncated

                step_reward = np.sum(reward) if isinstance(reward, np.ndarray) else reward
                episode_return += step_reward

                observations.append(obs)
                actions.append(action_array)
                rewards.append(step_reward)
                dones.append(done)
                log_probs.append(torch.stack(agent_log_probs).mean())
                values.append(value)

                obs = next_obs
                timesteps += 1

                # Call progress callback periodically
                if callback is not None and timesteps % 5000 == 0:
                    callback(timesteps)
                if timesteps % 5000 == 0:
                    self.training_metrics.flush(timesteps)

                if done:
                    self.training_returns.append(float(episode_return))
                    self.training_timesteps.append(timesteps)
                    episode_return = 0.0
                    obs, _ = self.env.reset()

            # Compute advantages and update
            self._update(observations, actions, rewards, dones, log_probs, values)

    def _update(self, observations, actions, rewards, dones, old_log_probs, old_values):
        """PPO update step."""
        import torch

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device_torch)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device_torch)
        dones_tensor = torch.FloatTensor(dones).to(self.device_torch)
        old_log_probs_tensor = torch.stack(old_log_probs).detach().to(self.device_torch)
        old_values_tensor = torch.cat(old_values).squeeze().detach().to(self.device_torch)

        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = old_values_tensor[t + 1].item()

            delta = rewards_tensor[t] + self.gamma * next_value * (1 - dones_tensor[t]) - old_values_tensor[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_tensor[t]) * gae
            advantages.append(gae)
            returns.append(gae + old_values_tensor[t].item())

        advantages = advantages[::-1]
        returns = returns[::-1]

        returns_tensor = torch.FloatTensor(returns).to(self.device_torch)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device_torch)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Pre-convert actions to GPU tensor once for all epochs
        actions_array = np.array(actions)
        actions_gpu = torch.tensor(actions_array, device=self.device_torch, dtype=torch.float32)

        # PPO epochs
        for _ in range(self.n_epochs):
            # Batched forward pass: one call per actor instead of per-sample
            agent_log_probs_list = []
            for j, actor in enumerate(self.actors):
                means = actor(obs_tensor)  # (n_steps, 1) in one batched call
                std = torch.exp(self.log_stds[j])
                dist = torch.distributions.Normal(means, std)
                lps = dist.log_prob(actions_gpu[:, j:j+1])  # (n_steps, 1)
                agent_log_probs_list.append(lps)

            new_log_probs_tensor = torch.stack(agent_log_probs_list, dim=1).mean(dim=1).squeeze()
            new_values = self.critic(obs_tensor).squeeze()

            # Policy loss
            ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * ((new_values - returns_tensor) ** 2).mean()

            # Entropy bonus
            entropy = 0
            for log_std in self.log_stds:
                entropy += 0.5 * (1 + torch.log(2 * np.pi * torch.exp(log_std) ** 2)).mean()
            entropy /= self.n_agents

            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.training_metrics.record('policy_loss', policy_loss.item())
            self.training_metrics.record('value_loss', value_loss.item())
            self.training_metrics.record('entropy', float(entropy) if isinstance(entropy, (int, float)) else entropy.item())
            self.training_metrics.record('clip_fraction', ((ratio - 1).abs() > self.clip_range).float().mean().item())

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_torch)

            actions = []
            for i, actor in enumerate(self.actors):
                mean = actor(obs_tensor)
                if deterministic:
                    action = mean.item()
                else:
                    std = torch.exp(self.log_stds[i])
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample().item()
                actions.append(action)

        action_array = np.array(actions, dtype=np.float32)
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

        return action_array

    def save(self, path: str):
        torch.save({
            'actors': self.actors.state_dict(),
            'critic': self.critic.state_dict(),
            'log_stds': self.log_stds.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.actors.load_state_dict(data['actors'])
        self.critic.load_state_dict(data['critic'])
        self.log_stds.load_state_dict(data['log_stds'])


class MADDPG(BaseAlgorithm):
    """Multi-Agent Deep Deterministic Policy Gradient."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate_actor = kwargs.get('learning_rate_actor', 1e-4)
        self.learning_rate_critic = kwargs.get('learning_rate_critic', 1e-3)
        self.buffer_size = kwargs.get('buffer_size', 100000)
        self.batch_size = kwargs.get('batch_size', 64)
        self.tau = kwargs.get('tau', 0.005)
        self.gamma = kwargs.get('gamma', 0.99)
        self.net_arch = kwargs.get('net_arch', [128, 128])

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)
        self.update_every = kwargs.get('update_every', 4)

        # Enable cudnn autotuner for fixed-size inputs
        if 'cuda' in str(device):
            torch.backends.cudnn.benchmark = True
        self.use_amp = ('cuda' in str(device)) and torch.cuda.is_available()

        self.obs_dim = self.obs_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # Per-agent actors and critics
        self.actors = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.target_actors = nn.ModuleList()
        self.target_critics = nn.ModuleList()
        self.actor_optimizers = []
        self.critic_optimizers = []

        for i in range(self.n_agents):
            # Actor: obs -> action
            actor = self._build_network(self.obs_dim, 1).to(self.device_torch)
            target_actor = self._build_network(self.obs_dim, 1).to(self.device_torch)
            target_actor.load_state_dict(actor.state_dict())

            # Critic: obs + all_actions -> Q
            critic_input_dim = self.obs_dim + self.action_dim
            critic = self._build_network(critic_input_dim, 1).to(self.device_torch)
            target_critic = self._build_network(critic_input_dim, 1).to(self.device_torch)
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=self.learning_rate_actor))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=self.learning_rate_critic))

        # Replay buffer - numpy circular buffer with O(1) random access
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dim, self.action_dim)

        # Exploration noise
        self.noise_scale = 0.1

    def _build_network(self, input_dim: int, output_dim: int):
        import torch.nn as nn
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def train(self, total_timesteps: int):
        """Train without callback."""
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train with optional progress callback."""
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        """Internal training implementation with optional callback support."""
        import torch

        obs, _ = self.env.reset(seed=self.seed)
        timesteps = 0
        episode_return = 0.0

        while timesteps < total_timesteps:
            # Select action with noise
            action = self._select_action(obs, add_noise=True)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_return += float(np.sum(reward) if isinstance(reward, np.ndarray) else reward)

            # Store transition
            self.buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            timesteps += 1

            # Call progress callback periodically
            if callback is not None and timesteps % 5000 == 0:
                callback(timesteps)
            if timesteps % 5000 == 0:
                self.training_metrics.flush(timesteps)

            if done:
                self.training_returns.append(float(episode_return))
                self.training_timesteps.append(timesteps)
                episode_return = 0.0
                obs, _ = self.env.reset()

            # Update if buffer has enough samples (every N steps)
            if len(self.buffer) >= self.batch_size and timesteps % self.update_every == 0:
                self._update()

            # Decay noise
            self.noise_scale = max(0.01, self.noise_scale * 0.9999)

    def _select_action(self, obs: np.ndarray, add_noise: bool = False) -> np.ndarray:
        import torch

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device_torch)

            actions = []
            for i, actor in enumerate(self.actors):
                action = actor(obs_tensor).cpu().numpy()[0, 0]
                if add_noise:
                    action += np.random.normal(0, self.noise_scale)
                actions.append(action)

        action_array = np.array(actions, dtype=np.float32)

        # Scale from [-1, 1] to action space
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action_array = (action_array + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
            action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

        return action_array

    def _update(self):
        import torch
        import torch.nn.functional as F

        # Sample batch - O(1) access from numpy circular buffer
        obs_np, actions_np, rewards_np, next_obs_np, dones_np = self.buffer.sample(self.batch_size)
        obs_batch = torch.from_numpy(obs_np).to(self.device_torch)
        action_batch = torch.from_numpy(actions_np).to(self.device_torch)
        reward_batch = torch.from_numpy(rewards_np).to(self.device_torch)
        next_obs_batch = torch.from_numpy(next_obs_np).to(self.device_torch)
        done_batch = torch.from_numpy(dones_np).to(self.device_torch)

        amp_ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()

        # Pre-compute target actions ONCE outside agent loop (reduces N^2 to N)
        with torch.no_grad():
            target_actions = [ta(next_obs_batch) for ta in self.target_actors]
            target_actions_tensor = torch.cat(target_actions, dim=1)
            target_critic_input = torch.cat([next_obs_batch, target_actions_tensor], dim=1)

        # Pre-compute all actor outputs (detached) ONCE outside agent loop
        with torch.no_grad():
            all_actor_outputs = [actor(obs_batch) for actor in self.actors]

        # Critic input from replay buffer actions (same for all agents)
        critic_input = torch.cat([obs_batch, action_batch], dim=1)

        # Update each agent
        for i in range(self.n_agents):
            # Target Q value
            with torch.no_grad(), amp_ctx:
                target_q = self.target_critics[i](target_critic_input).squeeze()
                target_q = reward_batch + self.gamma * (1 - done_batch) * target_q

            # Current Q value
            with amp_ctx:
                current_q = self.critics[i](critic_input).squeeze()
                critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Actor loss - only recompute actor[i] with gradient
            with amp_ctx:
                current_actions = list(all_actor_outputs)
                current_actions[i] = self.actors[i](obs_batch)
                current_actions_tensor = torch.cat(current_actions, dim=1)
                actor_critic_input = torch.cat([obs_batch, current_actions_tensor], dim=1)
                actor_loss = -self.critics[i](actor_critic_input).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            self.training_metrics.record('critic_loss', critic_loss.item())
            self.training_metrics.record('actor_loss', actor_loss.item())
            self.training_metrics.record('q_mean', current_q.mean().item())

        # Soft update targets
        for i in range(self.n_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._select_action(obs, add_noise=not deterministic)

    def save(self, path: str):
        torch.save({
            'actors': self.actors.state_dict(),
            'critics': self.critics.state_dict(),
            'target_actors': self.target_actors.state_dict(),
            'target_critics': self.target_critics.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.actors.load_state_dict(data['actors'])
        self.critics.load_state_dict(data['critics'])
        self.target_actors.load_state_dict(data['target_actors'])
        self.target_critics.load_state_dict(data['target_critics'])


class MATD3(MADDPG):
    """Multi-Agent TD3 (Twin Delayed DDPG)."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        # Add TD3-specific params
        kwargs.setdefault('policy_noise', 0.2)
        kwargs.setdefault('noise_clip', 0.5)
        kwargs.setdefault('policy_delay', 2)
        super().__init__(env, device, seed, **kwargs)

        self.policy_noise = kwargs.get('policy_noise', 0.2)
        self.noise_clip = kwargs.get('noise_clip', 0.5)
        self.policy_delay = kwargs.get('policy_delay', 2)
        self.update_counter = 0

        import torch.nn as nn
        import torch.optim as optim

        # Add twin critics
        self.critics2 = nn.ModuleList()
        self.target_critics2 = nn.ModuleList()
        self.critic2_optimizers = []

        critic_input_dim = self.obs_dim + self.action_dim
        for i in range(self.n_agents):
            critic2 = self._build_network(critic_input_dim, 1).to(self.device_torch)
            target_critic2 = self._build_network(critic_input_dim, 1).to(self.device_torch)
            target_critic2.load_state_dict(critic2.state_dict())

            self.critics2.append(critic2)
            self.target_critics2.append(target_critic2)
            self.critic2_optimizers.append(optim.Adam(critic2.parameters(), lr=self.learning_rate_critic))

    def _update(self):
        """TD3 update with clipped double-Q, target smoothing, and delayed policy updates."""
        import torch
        import torch.nn.functional as F
        from contextlib import nullcontext

        obs_np, actions_np, rewards_np, next_obs_np, dones_np = self.buffer.sample(self.batch_size)
        obs_batch = torch.from_numpy(obs_np).to(self.device_torch)
        action_batch = torch.from_numpy(actions_np).to(self.device_torch)
        reward_batch = torch.from_numpy(rewards_np).to(self.device_torch)
        next_obs_batch = torch.from_numpy(next_obs_np).to(self.device_torch)
        done_batch = torch.from_numpy(dones_np).to(self.device_torch)

        amp_ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()
        self.update_counter += 1

        # Target actions with smoothing noise (TD3 target policy smoothing)
        with torch.no_grad():
            target_actions = []
            for ta in self.target_actors:
                ta_out = ta(next_obs_batch)
                noise = torch.clamp(
                    torch.randn_like(ta_out) * self.policy_noise,
                    -self.noise_clip, self.noise_clip
                )
                ta_out = torch.clamp(ta_out + noise, -1.0, 1.0)
                target_actions.append(ta_out)
            target_actions_tensor = torch.cat(target_actions, dim=1)
            target_critic_input = torch.cat([next_obs_batch, target_actions_tensor], dim=1)

        # Pre-compute actor outputs (detached) for actor loss
        with torch.no_grad():
            all_actor_outputs = [actor(obs_batch) for actor in self.actors]

        critic_input = torch.cat([obs_batch, action_batch], dim=1)

        for i in range(self.n_agents):
            # Clipped double-Q: use min of two target critics
            with torch.no_grad(), amp_ctx:
                target_q1 = self.target_critics[i](target_critic_input).squeeze()
                target_q2 = self.target_critics2[i](target_critic_input).squeeze()
                target_q = torch.min(target_q1, target_q2)
                target_q = reward_batch + self.gamma * (1 - done_batch) * target_q

            # Update critic 1
            with amp_ctx:
                q1 = self.critics[i](critic_input).squeeze()
                critic1_loss = F.mse_loss(q1, target_q)
            self.critic_optimizers[i].zero_grad()
            critic1_loss.backward()
            self.critic_optimizers[i].step()

            # Update critic 2
            with amp_ctx:
                q2 = self.critics2[i](critic_input).squeeze()
                critic2_loss = F.mse_loss(q2, target_q)
            self.critic2_optimizers[i].zero_grad()
            critic2_loss.backward()
            self.critic2_optimizers[i].step()

            # Delayed actor update (every policy_delay steps)
            if self.update_counter % self.policy_delay == 0:
                with amp_ctx:
                    current_actions = list(all_actor_outputs)
                    current_actions[i] = self.actors[i](obs_batch)
                    current_actions_tensor = torch.cat(current_actions, dim=1)
                    actor_critic_input = torch.cat([obs_batch, current_actions_tensor], dim=1)
                    actor_loss = -self.critics[i](actor_critic_input).mean()

                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[i].step()
                self.training_metrics.record('actor_loss', actor_loss.item())

            self.training_metrics.record('critic1_loss', critic1_loss.item())
            self.training_metrics.record('critic2_loss', critic2_loss.item())
            self.training_metrics.record('q_mean', torch.min(q1, q2).mean().item())

        # Soft update both target critic pairs (delayed)
        if self.update_counter % self.policy_delay == 0:
            for i in range(self.n_agents):
                for tp, p in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
                for tp, p in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
                for tp, p in zip(self.target_critics2[i].parameters(), self.critics2[i].parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self, path: str):
        torch.save({
            'actors': self.actors.state_dict(),
            'critics': self.critics.state_dict(),
            'critics2': self.critics2.state_dict(),
            'target_actors': self.target_actors.state_dict(),
            'target_critics': self.target_critics.state_dict(),
            'target_critics2': self.target_critics2.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.actors.load_state_dict(data['actors'])
        self.critics.load_state_dict(data['critics'])
        self.critics2.load_state_dict(data['critics2'])
        self.target_actors.load_state_dict(data['target_actors'])
        self.target_critics.load_state_dict(data['target_critics'])
        self.target_critics2.load_state_dict(data['target_critics2'])


class MASAC(BaseAlgorithm):
    """Multi-Agent Soft Actor-Critic (Haarnoja et al. 2018, CTDE extension).

    Extends SAC to multi-agent setting with:
    - Stochastic Gaussian actors (per-agent, decentralized)
    - Twin centralized critics (per-agent, input = obs + all_actions)
    - Entropy-regularized TD targets: y = r + γ(min(Q1',Q2') - α·log π(a'|s'))
    - Auto-tuning temperature α via dual gradient descent on target entropy
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate_actor = kwargs.get('learning_rate_actor', 3e-4)
        self.learning_rate_critic = kwargs.get('learning_rate_critic', 1e-3)
        self.buffer_size = kwargs.get('buffer_size', 10000)
        self.batch_size = kwargs.get('batch_size', 64)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)
        self.net_arch = kwargs.get('net_arch', [128, 128])
        self.update_every = kwargs.get('update_every', 4)

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)

        if 'cuda' in str(device):
            torch.backends.cudnn.benchmark = True
        # Disable AMP: squashed Gaussian log_prob needs FP32 precision
        self.use_amp = False

        self.obs_dim = self.obs_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # Per-agent stochastic actors, twin critics, target critics
        self.actors = nn.ModuleList()
        self.log_stds = nn.ParameterList()
        self.critics1 = nn.ModuleList()
        self.critics2 = nn.ModuleList()
        self.target_critics1 = nn.ModuleList()
        self.target_critics2 = nn.ModuleList()
        self.actor_optimizers = []
        self.critic_optimizers = []

        # Auto-tuning alpha (one per agent)
        self.target_entropy = -self.action_dim / self.n_agents  # -dim per agent
        self.log_alphas = []
        self.alpha_optimizers = []

        critic_input_dim = self.obs_dim + self.action_dim

        for i in range(self.n_agents):
            # Stochastic actor: obs -> mean (tanh bounded)
            actor = self._build_actor(self.obs_dim, 1).to(self.device_torch)
            log_std = nn.Parameter(torch.zeros(1, device=self.device_torch))

            # Twin centralized critics: obs + all_actions -> Q
            c1 = self._build_critic(critic_input_dim, 1).to(self.device_torch)
            c2 = self._build_critic(critic_input_dim, 1).to(self.device_torch)
            tc1 = self._build_critic(critic_input_dim, 1).to(self.device_torch)
            tc2 = self._build_critic(critic_input_dim, 1).to(self.device_torch)
            tc1.load_state_dict(c1.state_dict())
            tc2.load_state_dict(c2.state_dict())

            self.actors.append(actor)
            self.log_stds.append(log_std)
            self.critics1.append(c1)
            self.critics2.append(c2)
            self.target_critics1.append(tc1)
            self.target_critics2.append(tc2)

            self.actor_optimizers.append(
                optim.Adam(list(actor.parameters()) + [log_std], lr=self.learning_rate_actor))
            self.critic_optimizers.append(
                optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=self.learning_rate_critic))

            # Learnable log_alpha for entropy auto-tuning
            log_alpha = torch.zeros(1, device=self.device_torch, requires_grad=True)
            self.log_alphas.append(log_alpha)
            self.alpha_optimizers.append(optim.Adam([log_alpha], lr=self.learning_rate_actor))

        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dim, self.action_dim)

    def _build_actor(self, input_dim, output_dim):
        import torch.nn as nn
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def _build_critic(self, input_dim, output_dim):
        import torch.nn as nn
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _sample_action(self, obs_tensor, agent_idx):
        """Sample action from stochastic Gaussian policy with log_prob."""
        import torch

        mean = self.actors[agent_idx](obs_tensor)
        log_std = self.log_stds[agent_idx].clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        # Reparameterization trick
        x = dist.rsample()
        action = torch.tanh(x)
        # Squashed Gaussian log_prob (Haarnoja eq. 21)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def train(self, total_timesteps: int):
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        import torch

        obs, _ = self.env.reset(seed=self.seed)
        timesteps = 0
        episode_return = 0.0

        while timesteps < total_timesteps:
            action = self._select_action(obs, stochastic=True)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_return += float(np.sum(reward) if isinstance(reward, np.ndarray) else reward)

            self.buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            timesteps += 1

            if callback is not None and timesteps % 5000 == 0:
                callback(timesteps)
            if timesteps % 5000 == 0:
                self.training_metrics.flush(timesteps)

            if done:
                self.training_returns.append(float(episode_return))
                self.training_timesteps.append(timesteps)
                episode_return = 0.0
                obs, _ = self.env.reset()

            if len(self.buffer) >= self.batch_size and timesteps % self.update_every == 0:
                self._update()

    def _select_action(self, obs: np.ndarray, stochastic: bool = False) -> np.ndarray:
        import torch

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device_torch)
            actions = []
            for i in range(self.n_agents):
                if stochastic:
                    action, _ = self._sample_action(obs_tensor, i)
                    actions.append(action.cpu().numpy()[0, 0])
                else:
                    mean = self.actors[i](obs_tensor)
                    actions.append(mean.cpu().numpy()[0, 0])

        action_array = np.array(actions, dtype=np.float32)
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action_array = (action_array + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
            action_array = np.clip(action_array, self.action_space.low, self.action_space.high)
        return action_array

    def _update(self):
        import torch
        import torch.nn.functional as F
        from contextlib import nullcontext

        obs_np, actions_np, rewards_np, next_obs_np, dones_np = self.buffer.sample(self.batch_size)
        obs_batch = torch.from_numpy(obs_np).to(self.device_torch)
        action_batch = torch.from_numpy(actions_np).to(self.device_torch)
        reward_batch = torch.from_numpy(rewards_np).to(self.device_torch)
        next_obs_batch = torch.from_numpy(next_obs_np).to(self.device_torch)
        done_batch = torch.from_numpy(dones_np).to(self.device_torch)

        amp_ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()

        # Compute next actions from all agents (stochastic, with log_probs)
        with torch.no_grad():
            next_actions_list = []
            next_log_probs_list = []
            for i in range(self.n_agents):
                na, nlp = self._sample_action(next_obs_batch, i)
                next_actions_list.append(na)
                next_log_probs_list.append(nlp)
            next_actions_cat = torch.cat(next_actions_list, dim=1)

        # Detached current actions for actor critic input
        with torch.no_grad():
            all_actor_outputs = [actor(obs_batch) for actor in self.actors]

        critic_input = torch.cat([obs_batch, action_batch], dim=1)
        next_critic_input = torch.cat([next_obs_batch, next_actions_cat], dim=1)

        for i in range(self.n_agents):
            alpha = self.log_alphas[i].exp().detach()

            # --- Twin critic update ---
            with torch.no_grad(), amp_ctx:
                target_q1 = self.target_critics1[i](next_critic_input).squeeze()
                target_q2 = self.target_critics2[i](next_critic_input).squeeze()
                min_target_q = torch.min(target_q1, target_q2)
                # Entropy-regularized TD target
                target = reward_batch + self.gamma * (1 - done_batch) * (
                    min_target_q - alpha * next_log_probs_list[i].squeeze())

            with amp_ctx:
                q1 = self.critics1[i](critic_input).squeeze()
                q2 = self.critics2[i](critic_input).squeeze()
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critics1[i].parameters()) + list(self.critics2[i].parameters()), 1.0)
            self.critic_optimizers[i].step()

            # --- Stochastic actor update ---
            with amp_ctx:
                current_actions = list(all_actor_outputs)
                sampled_action_i, log_prob_i = self._sample_action(obs_batch, i)
                current_actions[i] = sampled_action_i
                current_actions_cat = torch.cat(current_actions, dim=1)
                actor_critic_input = torch.cat([obs_batch, current_actions_cat], dim=1)

                q1_pi = self.critics1[i](actor_critic_input).squeeze()
                q2_pi = self.critics2[i](actor_critic_input).squeeze()
                min_q_pi = torch.min(q1_pi, q2_pi)

                # SAC actor loss: maximize Q - alpha * log_prob (entropy bonus)
                actor_loss = (alpha * log_prob_i.squeeze() - min_q_pi).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actors[i].parameters()) + [self.log_stds[i]], 1.0)
            self.actor_optimizers[i].step()

            # --- Alpha auto-tuning ---
            alpha_loss = -(self.log_alphas[i] * (
                log_prob_i.squeeze().detach() + self.target_entropy)).mean()

            self.alpha_optimizers[i].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizers[i].step()

            # Record metrics
            self.training_metrics.record('critic_loss', critic_loss.item())
            self.training_metrics.record('actor_loss', actor_loss.item())
            self.training_metrics.record('entropy', -log_prob_i.mean().item())
            self.training_metrics.record('alpha', alpha.item())
            self.training_metrics.record('q_mean', min_q_pi.mean().item())

        # Soft update target critics
        for i in range(self.n_agents):
            for tp, p in zip(self.target_critics1[i].parameters(), self.critics1[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for tp, p in zip(self.target_critics2[i].parameters(), self.critics2[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._select_action(obs, stochastic=not deterministic)

    def save(self, path: str):
        state = {
            'actors': self.actors.state_dict(),
            'critics1': self.critics1.state_dict(),
            'critics2': self.critics2.state_dict(),
            'target_critics1': self.target_critics1.state_dict(),
            'target_critics2': self.target_critics2.state_dict(),
            'log_stds': self.log_stds.state_dict(),
            'log_alphas': [la.data.clone() for la in self.log_alphas],
        }
        torch.save(state, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.actors.load_state_dict(data['actors'])
        self.critics1.load_state_dict(data['critics1'])
        self.critics2.load_state_dict(data['critics2'])
        self.target_critics1.load_state_dict(data['target_critics1'])
        self.target_critics2.load_state_dict(data['target_critics2'])
        self.log_stds.load_state_dict(data['log_stds'])
        if 'log_alphas' in data:
            for i, la_data in enumerate(data['log_alphas']):
                self.log_alphas[i].data.copy_(la_data)


# ============================================================================
# Value Decomposition (Discrete Action) Algorithms
# ============================================================================

class DiscreteActionWrapper:
    """Wrapper to discretize continuous actions."""

    def __init__(self, n_bins: int = 11, low: float = 0.0, high: float = 100.0):
        self.n_bins = n_bins
        self.low = low
        self.high = high
        self.bin_values = np.linspace(low, high, n_bins)

    def discrete_to_continuous(self, discrete_action: int) -> float:
        return self.bin_values[discrete_action]

    def continuous_to_discrete(self, continuous_action: float) -> int:
        return int(np.argmin(np.abs(self.bin_values - continuous_action)))


class QMIXMixingNetwork(torch.nn.Module):
    """QMIX mixing network with hypernetworks (Rashid et al. 2018).

    Generates mixing weights from global state with non-negative weight
    constraint (torch.abs) to enforce monotonicity / IGM condition:
    argmax_u Q_tot = (argmax_u1 Q1, ..., argmax_uN QN).
    """

    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        import torch.nn as nn
        import torch.nn.functional as F

        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # Hyper-w1: state -> abs -> (n_agents * embed_dim) weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim))
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hyper-w2: state -> abs -> (embed_dim * 1) weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim))
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1))

    def forward(self, agent_qs, state):
        """Mix agent Q-values with state-conditioned non-negative weights.

        Args:
            agent_qs: (B, n_agents) individual agent Q-values
            state: (B, state_dim) global state (observation)
        Returns:
            q_total: (B,) mixed Q-value
        """
        import torch
        import torch.nn.functional as F

        B = agent_qs.size(0)

        # First layer: non-negative weights from hypernetwork
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer: non-negative weights from hypernetwork
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)

        return (torch.bmm(hidden, w2) + b2).view(B)


class QMIX(BaseAlgorithm):
    """QMIX: Monotonic Value Function Factorization (Rashid et al. 2018)."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate = kwargs.get('learning_rate', 5e-4)
        self.buffer_size = kwargs.get('buffer_size', 5000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.gamma = kwargs.get('gamma', 0.99)
        self.action_bins = kwargs.get('action_bins', 11)

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)
        self.update_every = kwargs.get('update_every', 1)

        # Enable cudnn autotuner for fixed-size inputs
        if 'cuda' in str(device):
            torch.backends.cudnn.benchmark = True
        # PATCH: Disable AMP for QMIX - causes dtype mismatch (Float vs Half) in backward pass
        self.use_amp = False

        self.obs_dim = self.obs_space.shape[0]

        # Action discretizer
        action_low = self.action_space.low[0] if hasattr(self.action_space, 'low') else 0.0
        action_high = self.action_space.high[0] if hasattr(self.action_space, 'high') else 100.0
        self.discretizer = DiscreteActionWrapper(self.action_bins, action_low, action_high)

        # Per-agent Q networks
        self.q_networks = nn.ModuleList([
            self._build_q_network() for _ in range(self.n_agents)
        ]).to(self.device_torch)

        self.target_q_networks = nn.ModuleList([
            self._build_q_network() for _ in range(self.n_agents)
        ]).to(self.device_torch)

        for i in range(self.n_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())

        # Mixing network (hypernetwork-based for QMIX, None for VDN)
        self.mixer = self._build_mixer()
        if self.mixer is not None:
            self.mixer = self.mixer.to(self.device_torch)
            self.target_mixer = self._build_mixer().to(self.device_torch)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        else:
            self.target_mixer = None

        # Optimizer
        params = []
        if self.mixer is not None:
            params.extend(list(self.mixer.parameters()))
        for qnet in self.q_networks:
            params.extend(qnet.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        # Replay buffer - numpy circular buffer with O(1) random access
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dim, self.action_space.shape[0])

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def _build_q_network(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_bins)
        )

    def _build_mixer(self):
        """Build QMIX hypernetwork mixing network."""
        return QMIXMixingNetwork(self.n_agents, self.obs_dim)

    def train(self, total_timesteps: int):
        """Train without callback."""
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train with optional progress callback."""
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        """Internal training implementation with optional callback support."""
        import torch

        obs, _ = self.env.reset(seed=self.seed)
        timesteps = 0
        episode_return = 0.0

        while timesteps < total_timesteps:
            # Select action with epsilon-greedy
            action = self._select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_return += float(np.sum(reward) if isinstance(reward, np.ndarray) else reward)

            # Store transition
            self.buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            timesteps += 1

            # Call progress callback periodically (every 1000 steps)
            if callback is not None and timesteps % 5000 == 0:
                callback(timesteps)
            if timesteps % 5000 == 0:
                self.training_metrics.flush(timesteps)

            if done:
                self.training_returns.append(float(episode_return))
                self.training_timesteps.append(timesteps)
                episode_return = 0.0
                obs, _ = self.env.reset()

            # Update (every N steps to reduce gradient computation overhead)
            if len(self.buffer) >= self.batch_size and timesteps % self.update_every == 0:
                self._update()

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Update targets periodically
            if timesteps % 1000 == 0:
                for i in range(self.n_agents):
                    self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
                if self.mixer is not None and self.target_mixer is not None:
                    self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        import torch

        if np.random.random() < self.epsilon:
            # Random discrete actions
            discrete_actions = [np.random.randint(0, self.action_bins) for _ in range(self.n_agents)]
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device_torch)
                discrete_actions = []
                for i, qnet in enumerate(self.q_networks):
                    q_values = qnet(obs_tensor)
                    discrete_actions.append(q_values.argmax(dim=1).item())

        # Convert to continuous
        continuous_actions = np.array([
            self.discretizer.discrete_to_continuous(a) for a in discrete_actions
        ], dtype=np.float32)

        return continuous_actions

    def _update(self):
        import torch
        import torch.nn.functional as F

        # Sample batch - O(1) access from numpy circular buffer
        obs_np, _, rewards_np, next_obs_np, dones_np = self.buffer.sample(self.batch_size)
        obs_batch = torch.from_numpy(obs_np).to(self.device_torch)
        reward_batch = torch.from_numpy(rewards_np).to(self.device_torch)
        next_obs_batch = torch.from_numpy(next_obs_np).to(self.device_torch)
        done_batch = torch.from_numpy(dones_np).to(self.device_torch)

        amp_ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()

        # Get Q values
        with amp_ctx:
            q_values = []
            for i, qnet in enumerate(self.q_networks):
                q = qnet(obs_batch).max(dim=1)[0]
                q_values.append(q)
            q_values_tensor = torch.stack(q_values, dim=1)

            # Mix Q values (QMIX uses hypernetwork conditioned on state)
            if self.mixer is not None:
                q_total = self.mixer(q_values_tensor, obs_batch)
            else:
                q_total = q_values_tensor.sum(dim=1)

        # Target Q values
        with torch.no_grad(), amp_ctx:
            target_q_values = []
            for i, target_qnet in enumerate(self.target_q_networks):
                target_q = target_qnet(next_obs_batch).max(dim=1)[0]
                target_q_values.append(target_q)
            target_q_values_tensor = torch.stack(target_q_values, dim=1)
            if self.target_mixer is not None:
                target_q_total = self.target_mixer(target_q_values_tensor, next_obs_batch)
            else:
                target_q_total = target_q_values_tensor.sum(dim=1)
            target = reward_batch + self.gamma * (1 - done_batch) * target_q_total

        # Loss
        loss = F.mse_loss(q_total, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_metrics.record('td_loss', loss.item())
        self.training_metrics.record('q_total_mean', q_total.mean().item())
        self.training_metrics.record('epsilon', self.epsilon)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        old_epsilon = self.epsilon
        self.epsilon = 0.0 if deterministic else self.epsilon
        action = self._select_action(obs)
        self.epsilon = old_epsilon
        return action

    def save(self, path: str):
        state = {
            'q_networks': self.q_networks.state_dict(),
            'target_q_networks': self.target_q_networks.state_dict(),
            'epsilon': self.epsilon,
        }
        if self.mixer is not None:
            state['mixer'] = self.mixer.state_dict()
            state['target_mixer'] = self.target_mixer.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.q_networks.load_state_dict(data['q_networks'])
        self.target_q_networks.load_state_dict(data['target_q_networks'])
        if 'epsilon' in data:
            self.epsilon = data['epsilon']
        if self.mixer is not None and 'mixer' in data:
            self.mixer.load_state_dict(data['mixer'])
            self.target_mixer.load_state_dict(data['target_mixer'])


class VDN(QMIX):
    """Value Decomposition Networks (additive factorization)."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        # Call BaseAlgorithm init directly to avoid QMIX mixer setup
        BaseAlgorithm.__init__(self, env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate = kwargs.get('learning_rate', 5e-4)
        self.buffer_size = kwargs.get('buffer_size', 5000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.gamma = kwargs.get('gamma', 0.99)
        self.action_bins = kwargs.get('action_bins', 11)

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)
        self.update_every = kwargs.get('update_every', 1)

        # Enable cudnn autotuner for fixed-size inputs
        if 'cuda' in str(device):
            torch.backends.cudnn.benchmark = True
        # PATCH: Disable AMP for VDN - causes dtype mismatch (Float vs Half) in backward pass
        self.use_amp = False

        self.obs_dim = self.obs_space.shape[0]

        # Action discretizer
        action_low = self.action_space.low[0] if hasattr(self.action_space, 'low') else 0.0
        action_high = self.action_space.high[0] if hasattr(self.action_space, 'high') else 100.0
        self.discretizer = DiscreteActionWrapper(self.action_bins, action_low, action_high)

        # Per-agent Q networks
        self.q_networks = nn.ModuleList([
            self._build_q_network() for _ in range(self.n_agents)
        ]).to(self.device_torch)

        self.target_q_networks = nn.ModuleList([
            self._build_q_network() for _ in range(self.n_agents)
        ]).to(self.device_torch)

        for i in range(self.n_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())

        # VDN has no mixer - just sum Q values
        self.mixer = None
        self.target_mixer = None

        # Optimizer - only Q networks, no mixer
        params = []
        for qnet in self.q_networks:
            params.extend(qnet.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        # Replay buffer - numpy circular buffer with O(1) random access
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dim, self.action_space.shape[0])

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def _build_mixer(self):
        # VDN doesn't use a mixer
        return None

    def _update(self):
        import torch
        import torch.nn.functional as F

        # Sample batch - O(1) access from numpy circular buffer
        obs_np, _, rewards_np, next_obs_np, dones_np = self.buffer.sample(self.batch_size)
        obs_batch = torch.from_numpy(obs_np).to(self.device_torch)
        reward_batch = torch.from_numpy(rewards_np).to(self.device_torch)
        next_obs_batch = torch.from_numpy(next_obs_np).to(self.device_torch)
        done_batch = torch.from_numpy(dones_np).to(self.device_torch)

        amp_ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()

        # Get Q values and sum them (VDN)
        with amp_ctx:
            q_values = []
            for qnet in self.q_networks:
                q = qnet(obs_batch).max(dim=1)[0]
                q_values.append(q)
            q_total = sum(q_values)

        # Target Q values
        with torch.no_grad(), amp_ctx:
            target_q_values = []
            for target_qnet in self.target_q_networks:
                target_q = target_qnet(next_obs_batch).max(dim=1)[0]
                target_q_values.append(target_q)
            target_q_total = sum(target_q_values)
            target = reward_batch + self.gamma * (1 - done_batch) * target_q_total

        loss = F.mse_loss(q_total, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_metrics.record('td_loss', loss.item())
        self.training_metrics.record('q_total_mean', q_total.mean().item())
        self.training_metrics.record('epsilon', self.epsilon)


class COMA(BaseAlgorithm):
    """Counterfactual Multi-Agent Policy Gradients (Foerster et al. 2018).

    Actor-critic with:
    - Decentralized stochastic actors (softmax over discrete actions)
    - Centralized critic: Q(state, other_agents_actions) -> Q-values per action
    - Counterfactual baseline: b_a = Σ_{u'} π(u'|s) · Q(s, u_{-a}, u')
    - Advantage: A = Q(s,u) - b_a (per-agent credit assignment)
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate_actor = kwargs.get('learning_rate_actor', 3e-4)
        self.learning_rate_critic = kwargs.get('learning_rate_critic', 1e-3)
        self.gamma = kwargs.get('gamma', 0.99)
        self.action_bins = kwargs.get('action_bins', 11)
        self.n_steps = kwargs.get('n_steps', 128)
        self.net_arch = kwargs.get('net_arch', [128, 128])
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)

        if 'cuda' in str(device):
            torch.backends.cudnn.benchmark = True
        self.use_amp = False  # Discrete actions, small networks

        self.obs_dim = self.obs_space.shape[0]

        # Action discretizer (same as QMIX/VDN)
        action_low = self.action_space.low[0] if hasattr(self.action_space, 'low') else 0.0
        action_high = self.action_space.high[0] if hasattr(self.action_space, 'high') else 100.0
        self.discretizer = DiscreteActionWrapper(self.action_bins, action_low, action_high)

        # Per-agent decentralized actors: obs -> softmax(logits) over action_bins
        self.actors = nn.ModuleList()
        self.actor_optimizers = []
        for i in range(self.n_agents):
            actor = self._build_actor(self.obs_dim, self.action_bins).to(self.device_torch)
            self.actors.append(actor)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=self.learning_rate_actor))

        # Per-agent centralized critics:
        # Input: obs + other_agents' one-hot actions = obs_dim + (n_agents-1)*action_bins
        # Output: Q-values for this agent's action_bins actions
        other_actions_dim = max(0, (self.n_agents - 1) * self.action_bins)
        critic_input_dim = self.obs_dim + other_actions_dim

        self.critics = nn.ModuleList()
        self.critic_optimizers = []
        for i in range(self.n_agents):
            critic = self._build_critic(critic_input_dim, self.action_bins).to(self.device_torch)
            self.critics.append(critic)
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=self.learning_rate_critic))

        # Exploration epsilon
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9998

    def _build_actor(self, input_dim, output_dim):
        import torch.nn as nn
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_critic(self, input_dim, output_dim):
        import torch.nn as nn
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _get_other_actions_onehot(self, discrete_actions, agent_idx):
        """Build one-hot encoded other agents' actions tensor."""
        import torch

        parts = []
        for j in range(self.n_agents):
            if j == agent_idx:
                continue
            onehot = torch.zeros(self.action_bins, device=self.device_torch)
            onehot[discrete_actions[j]] = 1.0
            parts.append(onehot)
        if parts:
            return torch.cat(parts)
        return torch.zeros(0, device=self.device_torch)

    def _get_other_actions_onehot_batch(self, discrete_actions_batch, agent_idx):
        """Batch version: discrete_actions_batch is (B, n_agents) LongTensor."""
        import torch

        B = discrete_actions_batch.size(0)
        parts = []
        for j in range(self.n_agents):
            if j == agent_idx:
                continue
            onehot = torch.zeros(B, self.action_bins, device=self.device_torch)
            onehot.scatter_(1, discrete_actions_batch[:, j:j+1], 1.0)
            parts.append(onehot)
        if parts:
            return torch.cat(parts, dim=1)
        return torch.zeros(B, 0, device=self.device_torch)

    def train(self, total_timesteps: int):
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        import torch
        import torch.nn.functional as F

        obs, _ = self.env.reset(seed=self.seed)
        timesteps = 0
        episode_return = 0.0

        while timesteps < total_timesteps:
            # Collect n_steps of experience
            batch_obs = []
            batch_discrete_actions = []
            batch_rewards = []
            batch_dones = []

            for _ in range(self.n_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device_torch)

                # Select discrete actions from each actor (epsilon-greedy over softmax)
                discrete_actions = []
                with torch.no_grad():
                    for i in range(self.n_agents):
                        logits = self.actors[i](obs_tensor)
                        probs = F.softmax(logits, dim=-1).squeeze(0)
                        if np.random.random() < self.epsilon:
                            action_idx = np.random.randint(0, self.action_bins)
                        else:
                            action_idx = torch.multinomial(probs, 1).item()
                        discrete_actions.append(action_idx)

                # Convert to continuous for env
                continuous_actions = np.array([
                    self.discretizer.discrete_to_continuous(a) for a in discrete_actions
                ], dtype=np.float32)

                next_obs, reward, terminated, truncated, info = self.env.step(continuous_actions)
                done = terminated or truncated
                step_reward = float(np.sum(reward) if isinstance(reward, np.ndarray) else reward)
                episode_return += step_reward

                batch_obs.append(obs.copy())
                batch_discrete_actions.append(discrete_actions)
                batch_rewards.append(step_reward)
                batch_dones.append(done)

                obs = next_obs
                timesteps += 1

                if callback is not None and timesteps % 5000 == 0:
                    callback(timesteps)
                if timesteps % 5000 == 0:
                    self.training_metrics.flush(timesteps)

                if done:
                    self.training_returns.append(float(episode_return))
                    self.training_timesteps.append(timesteps)
                    episode_return = 0.0
                    obs, _ = self.env.reset()

                if timesteps >= total_timesteps:
                    break

            if len(batch_obs) == 0:
                break

            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(batch_rewards):
                G = r + self.gamma * G
                returns.append(G)
            returns = returns[::-1]
            returns_t = torch.tensor(returns, device=self.device_torch, dtype=torch.float32)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            B = len(batch_obs)
            obs_t = torch.tensor(np.array(batch_obs), device=self.device_torch, dtype=torch.float32)
            actions_t = torch.tensor(batch_discrete_actions, device=self.device_torch, dtype=torch.long)

            # Update each agent
            total_critic_loss = 0.0
            total_actor_loss = 0.0
            total_cf_adv = 0.0

            for i in range(self.n_agents):
                # Build critic input: obs + other agents' one-hot actions
                other_onehot = self._get_other_actions_onehot_batch(actions_t, i)
                critic_input = torch.cat([obs_t, other_onehot], dim=1)

                # Critic outputs Q-values for all of agent i's actions: (B, action_bins)
                q_all = self.critics[i](critic_input)

                # Q for taken action
                q_taken = q_all.gather(1, actions_t[:, i:i+1]).squeeze(1)

                # Critic loss: MSE against discounted returns
                critic_loss = F.mse_loss(q_taken, returns_t)

                self.critic_optimizers[i].zero_grad()
                critic_loss.backward()
                self.critic_optimizers[i].step()

                # Actor update with counterfactual baseline
                logits = self.actors[i](obs_t)
                probs = F.softmax(logits, dim=-1)  # (B, action_bins)
                log_probs = F.log_softmax(logits, dim=-1)

                # Recompute Q with no grad for advantage
                with torch.no_grad():
                    q_all_detached = self.critics[i](critic_input)
                    q_taken_detached = q_all_detached.gather(1, actions_t[:, i:i+1]).squeeze(1)

                    # Counterfactual baseline: b = Σ_{u'} π(u'|s) · Q(s, u_{-i}, u')
                    baseline = (probs.detach() * q_all_detached).sum(dim=1)

                    # Counterfactual advantage
                    advantage = q_taken_detached - baseline

                # Policy gradient with counterfactual advantage
                log_prob_taken = log_probs.gather(1, actions_t[:, i:i+1]).squeeze(1)
                entropy = -(probs * log_probs).sum(dim=1).mean()
                actor_loss = -(log_prob_taken * advantage).mean() - self.entropy_coef * entropy

                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[i].step()

                total_critic_loss += critic_loss.item()
                total_actor_loss += actor_loss.item()
                total_cf_adv += advantage.mean().item()

            self.training_metrics.record('critic_loss', total_critic_loss / self.n_agents)
            self.training_metrics.record('actor_loss', total_actor_loss / self.n_agents)
            self.training_metrics.record('counterfactual_advantage_mean', total_cf_adv / self.n_agents)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device_torch)
            discrete_actions = []
            for i in range(self.n_agents):
                logits = self.actors[i](obs_tensor)
                if deterministic:
                    action_idx = logits.argmax(dim=-1).item()
                else:
                    probs = F.softmax(logits, dim=-1).squeeze(0)
                    action_idx = torch.multinomial(probs, 1).item()
                discrete_actions.append(action_idx)

        continuous_actions = np.array([
            self.discretizer.discrete_to_continuous(a) for a in discrete_actions
        ], dtype=np.float32)
        return continuous_actions

    def save(self, path: str):
        torch.save({
            'actors': self.actors.state_dict(),
            'critics': self.critics.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.actors.load_state_dict(data['actors'])
        self.critics.load_state_dict(data['critics'])
        if 'epsilon' in data:
            self.epsilon = data['epsilon']


# ============================================================================
# Opponent Modeling Algorithms
# ============================================================================

class LOLA(BaseAlgorithm):
    """Learning with Opponent-Learning Awareness."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.opponent_lr = kwargs.get('opponent_lr', 1e-3)
        self.n_lookahead = kwargs.get('n_lookahead', 1)
        self.gamma = kwargs.get('gamma', 0.99)
        self.net_arch = kwargs.get('net_arch', [128, 128])

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)

        self.obs_dim = self.obs_space.shape[0]

        # Policy networks for each agent
        self.policies = nn.ModuleList([
            self._build_policy() for _ in range(self.n_agents)
        ]).to(self.device_torch)

        # Log std for actions - create parameters on device directly
        self.log_stds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, device=self.device_torch)) for _ in range(self.n_agents)
        ])

        # Separate optimizers
        self.optimizers = [
            optim.Adam(list(self.policies[i].parameters()) + [self.log_stds[i]], lr=self.learning_rate)
            for i in range(self.n_agents)
        ]

    def _build_policy(self):
        import torch.nn as nn
        layers = []
        prev_dim = self.obs_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def train(self, total_timesteps: int):
        """Train without callback."""
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train with optional progress callback."""
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        """LOLA training: policy gradient with opponent-learning awareness.

        Key difference from REINFORCE: each agent accounts for how opponents
        will update their policies, differentiating through the opponent's
        anticipated parameter change (first-order approximation).
        """
        import torch

        obs, _ = self.env.reset(seed=self.seed)
        timesteps = 0
        episode_return = 0.0

        while timesteps < total_timesteps:
            # Collect trajectories — use torch.no_grad() for action selection,
            # store (obs, action_value, reward) tuples. Log probs are recomputed
            # during the update phase when they need computation graph connection.
            trajectories = [[] for _ in range(self.n_agents)]

            for _ in range(128):  # Trajectory length
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device_torch)

                    actions = []
                    for i in range(self.n_agents):
                        mean = self.policies[i](obs_tensor)
                        std = torch.exp(self.log_stds[i])
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample()
                        actions.append(action.item())

                action_array = np.array(actions, dtype=np.float32)
                if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                    action_array = (action_array + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
                    action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

                next_obs, reward, terminated, truncated, info = self.env.step(action_array)
                done = terminated or truncated
                episode_return += float(np.sum(reward) if isinstance(reward, np.ndarray) else reward)

                for i in range(self.n_agents):
                    r_i = reward[i] if isinstance(reward, np.ndarray) else reward / self.n_agents
                    trajectories[i].append((obs.copy(), actions[i], r_i))

                obs = next_obs
                timesteps += 1

                if callback is not None and timesteps % 5000 == 0:
                    callback(timesteps)
                if timesteps % 5000 == 0:
                    self.training_metrics.flush(timesteps)

                if done:
                    self.training_returns.append(float(episode_return))
                    self.training_timesteps.append(timesteps)
                    episode_return = 0.0
                    obs, _ = self.env.reset()

            # LOLA update: each agent anticipates opponent learning
            # Pre-batch all trajectory data to GPU once (avoid per-sample transfers)
            all_obs_gpu = {}
            all_actions_gpu = {}
            all_returns_gpu = {}
            for k in range(self.n_agents):
                obs_list = [o for o, _, _ in trajectories[k]]
                act_list = [a for _, a, _ in trajectories[k]]
                rew_list = [r for _, _, r in trajectories[k]]
                all_obs_gpu[k] = torch.tensor(
                    np.array(obs_list), device=self.device_torch, dtype=torch.float32
                )
                all_actions_gpu[k] = torch.tensor(
                    [[a] for a in act_list], device=self.device_torch, dtype=torch.float32
                )
                # Compute discounted returns (append + reverse, not insert(0))
                returns_k = []
                G = 0
                for r in reversed(rew_list):
                    G = r + self.gamma * G
                    returns_k.append(G)
                returns_k = returns_k[::-1]
                returns_k = torch.tensor(returns_k, device=self.device_torch, dtype=torch.float32)
                all_returns_gpu[k] = (returns_k - returns_k.mean()) / (returns_k.std() + 1e-8)

            for i in range(self.n_agents):
                returns_i = all_returns_gpu[i]

                # Batched log prob computation for agent i
                means_i = self.policies[i](all_obs_gpu[i])  # (T, 1)
                std_i = torch.exp(self.log_stds[i])
                dist_i = torch.distributions.Normal(means_i, std_i)
                log_probs_i = dist_i.log_prob(all_actions_gpu[i]).squeeze()  # (T,)

                # Vectorized naive policy gradient
                naive_loss_i = -(log_probs_i * returns_i).sum()

                # LOLA correction: multi-step opponent anticipation
                # For each opponent j, simulate n_lookahead gradient steps on
                # their policy, then compute how agent i's loss changes at the
                # opponent's anticipated parameters (Foerster et al. 2018).
                lola_correction = torch.zeros(1, device=self.device_torch)
                for j in range(self.n_agents):
                    if j == i:
                        continue

                    returns_j = all_returns_gpu[j]

                    # Clone opponent parameters for lookahead simulation
                    # Use dict form for torch.func.functional_call
                    anticipated_params = {
                        name: p.clone()
                        for name, p in self.policies[j].named_parameters()
                    }
                    anticipated_log_std = self.log_stds[j].clone()

                    # Multi-step lookahead: simulate opponent's gradient updates
                    for lookahead_step in range(self.n_lookahead):
                        # Forward pass at anticipated opponent params
                        means_j_ant = torch.func.functional_call(
                            self.policies[j], anticipated_params, all_obs_gpu[j])
                        std_j_ant = torch.exp(anticipated_log_std)
                        dist_j_ant = torch.distributions.Normal(means_j_ant, std_j_ant)
                        log_probs_j_ant = dist_j_ant.log_prob(all_actions_gpu[j]).squeeze()

                        # Opponent's naive loss at anticipated params
                        loss_j_ant = -(log_probs_j_ant * returns_j).sum()

                        # Gradient of opponent's loss w.r.t. anticipated params
                        grad_targets = list(anticipated_params.values()) + [anticipated_log_std]
                        grads = torch.autograd.grad(
                            loss_j_ant, grad_targets,
                            create_graph=True, retain_graph=True, allow_unused=True)

                        # Simulate opponent gradient descent step
                        param_names = list(anticipated_params.keys())
                        for k, name in enumerate(param_names):
                            if grads[k] is not None:
                                anticipated_params[name] = anticipated_params[name] - self.opponent_lr * grads[k]
                        if grads[-1] is not None:
                            anticipated_log_std = anticipated_log_std - self.opponent_lr * grads[-1]

                    # Compute agent i's loss at opponent's ANTICIPATED parameters
                    # This is the LOLA insight: differentiate through the opponent's
                    # anticipated update to get the correction term
                    means_j_final = torch.func.functional_call(
                        self.policies[j], anticipated_params, all_obs_gpu[j])
                    std_j_final = torch.exp(anticipated_log_std)
                    dist_j_final = torch.distributions.Normal(means_j_final, std_j_final)
                    log_probs_j_final = dist_j_final.log_prob(all_actions_gpu[j]).squeeze()

                    # Agent i cares about how j's anticipated behavior affects returns
                    # Cross-agent LOLA correction term
                    lola_correction = lola_correction + (-(log_probs_j_final * returns_i).sum() * 0.01)

                total_loss = naive_loss_i + lola_correction

                self.optimizers[i].zero_grad()
                total_loss.backward()
                self.optimizers[i].step()

                self.training_metrics.record('naive_loss', naive_loss_i.item())
                self.training_metrics.record('lola_correction', lola_correction.item())
                self.training_metrics.record('total_loss', total_loss.item())

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_torch)

            actions = []
            for i in range(self.n_agents):
                mean = self.policies[i](obs_tensor)
                if deterministic:
                    action = mean.item()
                else:
                    std = torch.exp(self.log_stds[i])
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample().item()
                actions.append(action)

        action_array = np.array(actions, dtype=np.float32)
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action_array = (action_array + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
            action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

        return action_array

    def save(self, path: str):
        torch.save({
            'policies': self.policies.state_dict(),
            'log_stds': self.log_stds.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.policies.load_state_dict(data['policies'])
        self.log_stds.load_state_dict(data['log_stds'])


class IndependentREINFORCE(BaseAlgorithm):
    """Independent REINFORCE (Williams 1992) for multi-agent settings.

    Each agent independently runs vanilla policy gradient (REINFORCE) with
    no inter-agent information sharing. This is the simplest MARL baseline —
    each agent treats other agents as part of the environment.

    Structurally identical to LOLA but without the opponent-learning correction
    term, making it the natural ablation baseline for LOLA and other
    opponent-aware methods.
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.gamma = kwargs.get('gamma', 0.99)
        self.net_arch = kwargs.get('net_arch', [128, 128])

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)

        self.obs_dim = self.obs_space.shape[0]

        # Per-agent policy networks (decentralized execution)
        self.policies = nn.ModuleList([
            self._build_policy() for _ in range(self.n_agents)
        ]).to(self.device_torch)

        # Log std for continuous actions
        self.log_stds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, device=self.device_torch)) for _ in range(self.n_agents)
        ])

        # Independent optimizers — no parameter sharing
        self.optimizers = [
            optim.Adam(list(self.policies[i].parameters()) + [self.log_stds[i]], lr=self.learning_rate)
            for i in range(self.n_agents)
        ]

    def _build_policy(self):
        import torch.nn as nn
        layers = []
        prev_dim = self.obs_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def train(self, total_timesteps: int):
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        """Independent REINFORCE: vanilla per-agent policy gradient.

        Each agent independently collects trajectories and updates its own
        policy using the REINFORCE estimator. No opponent modeling, no
        centralized critic, no shared information.
        """
        import torch

        obs, _ = self.env.reset(seed=self.seed)
        timesteps = 0
        episode_return = 0.0

        while timesteps < total_timesteps:
            # Collect trajectories
            trajectories = [[] for _ in range(self.n_agents)]

            for _ in range(128):  # Trajectory length
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device_torch)

                    actions = []
                    for i in range(self.n_agents):
                        mean = self.policies[i](obs_tensor)
                        std = torch.exp(self.log_stds[i])
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample()
                        actions.append(action.item())

                action_array = np.array(actions, dtype=np.float32)
                if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                    action_array = (action_array + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
                    action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

                next_obs, reward, terminated, truncated, info = self.env.step(action_array)
                done = terminated or truncated
                episode_return += float(np.sum(reward) if isinstance(reward, np.ndarray) else reward)

                for i in range(self.n_agents):
                    r_i = reward[i] if isinstance(reward, np.ndarray) else reward / self.n_agents
                    trajectories[i].append((obs.copy(), actions[i], r_i))

                obs = next_obs
                timesteps += 1

                if callback is not None and timesteps % 5000 == 0:
                    callback(timesteps)
                if timesteps % 5000 == 0:
                    self.training_metrics.flush(timesteps)

                if done:
                    self.training_returns.append(float(episode_return))
                    self.training_timesteps.append(timesteps)
                    episode_return = 0.0
                    obs, _ = self.env.reset()

            # Independent REINFORCE update for each agent
            for i in range(self.n_agents):
                obs_list = [o for o, _, _ in trajectories[i]]
                act_list = [a for _, a, _ in trajectories[i]]
                rew_list = [r for _, _, r in trajectories[i]]

                obs_gpu = torch.tensor(
                    np.array(obs_list), device=self.device_torch, dtype=torch.float32)
                actions_gpu = torch.tensor(
                    [[a] for a in act_list], device=self.device_torch, dtype=torch.float32)

                # Discounted returns (append + reverse, not insert(0))
                returns = []
                G = 0
                for r in reversed(rew_list):
                    G = r + self.gamma * G
                    returns.append(G)
                returns = returns[::-1]
                returns = torch.tensor(returns, device=self.device_torch, dtype=torch.float32)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                # REINFORCE: log π(a|s) · R(τ)
                means = self.policies[i](obs_gpu)
                std = torch.exp(self.log_stds[i])
                dist = torch.distributions.Normal(means, std)
                log_probs = dist.log_prob(actions_gpu).squeeze()

                policy_loss = -(log_probs * returns).sum()

                self.optimizers[i].zero_grad()
                policy_loss.backward()
                self.optimizers[i].step()

                self.training_metrics.record('policy_loss', policy_loss.item())

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_torch)

            actions = []
            for i in range(self.n_agents):
                mean = self.policies[i](obs_tensor)
                if deterministic:
                    action = mean.item()
                else:
                    std = torch.exp(self.log_stds[i])
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample().item()
                actions.append(action)

        action_array = np.array(actions, dtype=np.float32)
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action_array = (action_array + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
            action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

        return action_array

    def save(self, path: str):
        torch.save({
            'policies': self.policies.state_dict(),
            'log_stds': self.log_stds.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.policies.load_state_dict(data['policies'])
        self.log_stds.load_state_dict(data['log_stds'])


class M3DDPG(MADDPG):
    """Minimax Multi-Agent DDPG."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)
        self.minimax_weight = kwargs.get('minimax_weight', 0.5)

    def _update(self):
        """Minimax MADDPG: blends standard Q maximization with worst-case opponent Q."""
        import torch
        import torch.nn.functional as F
        from contextlib import nullcontext

        obs_np, actions_np, rewards_np, next_obs_np, dones_np = self.buffer.sample(self.batch_size)
        obs_batch = torch.from_numpy(obs_np).to(self.device_torch)
        action_batch = torch.from_numpy(actions_np).to(self.device_torch)
        reward_batch = torch.from_numpy(rewards_np).to(self.device_torch)
        next_obs_batch = torch.from_numpy(next_obs_np).to(self.device_torch)
        done_batch = torch.from_numpy(dones_np).to(self.device_torch)

        amp_ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()

        with torch.no_grad():
            target_actions = [ta(next_obs_batch) for ta in self.target_actors]
            target_actions_tensor = torch.cat(target_actions, dim=1)
            target_critic_input = torch.cat([next_obs_batch, target_actions_tensor], dim=1)

        with torch.no_grad():
            all_actor_outputs = [actor(obs_batch) for actor in self.actors]

        critic_input = torch.cat([obs_batch, action_batch], dim=1)

        for i in range(self.n_agents):
            # Standard critic update
            with torch.no_grad(), amp_ctx:
                target_q = self.target_critics[i](target_critic_input).squeeze()
                target_q = reward_batch + self.gamma * (1 - done_batch) * target_q

            with amp_ctx:
                current_q = self.critics[i](critic_input).squeeze()
                critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Minimax actor loss: blend own Q with worst-case opponent Q
            with amp_ctx:
                current_actions = list(all_actor_outputs)
                current_actions[i] = self.actors[i](obs_batch)
                current_actions_tensor = torch.cat(current_actions, dim=1)
                actor_critic_input = torch.cat([obs_batch, current_actions_tensor], dim=1)

                # Standard: maximize own Q
                standard_q = self.critics[i](actor_critic_input).mean()

                # Minimax: consider worst-case opponent Q
                opponent_qs = []
                for j in range(self.n_agents):
                    if j != i:
                        opponent_qs.append(self.critics[j](actor_critic_input).mean())

                if opponent_qs:
                    max_opponent_q = torch.stack(opponent_qs).max()
                    actor_loss = -(1 - self.minimax_weight) * standard_q + self.minimax_weight * max_opponent_q
                else:
                    actor_loss = -standard_q

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            self.training_metrics.record('critic_loss', critic_loss.item())
            self.training_metrics.record('actor_loss', actor_loss.item())
            self.training_metrics.record('standard_q', standard_q.item())
            if opponent_qs:
                self.training_metrics.record('max_opponent_q', max_opponent_q.item())

        # Soft update targets
        for i in range(self.n_agents):
            for tp, p in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for tp, p in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


# ============================================================================
# Population-Based Algorithms
# ============================================================================

class SelfPlayPPO(BaseAlgorithm):
    """Self-Play with PPO."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        self.opponent_update_freq = kwargs.get('opponent_update_freq', 10000)

        # Use IPPO as base
        self._ippo = IndependentPPO(env, device, seed, **kwargs)

        # Store opponent policy (copy of self)
        self.opponent_weights = None
        self.update_counter = 0

    def train(self, total_timesteps: int):
        """Train without callback."""
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train with optional progress callback."""
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        """Internal training implementation with optional callback support."""
        # Share our training_metrics with the inner IPPO so SB3 callback records there
        self._ippo.training_metrics = self.training_metrics

        # Periodic opponent update
        steps_per_update = self.opponent_update_freq

        trained = 0
        while trained < total_timesteps:
            steps_to_train = min(steps_per_update, total_timesteps - trained)
            self._ippo.train_with_callback(steps_to_train, callback)
            trained += steps_to_train

            # Update opponent (save current policy)
            self.opponent_weights = self._ippo.model.get_parameters()

        self.training_returns = self._ippo.training_returns

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._ippo.predict(obs, deterministic)


class FictitiousCoPlay(BaseAlgorithm):
    """Fictitious Co-Play (Strouse et al. 2021).

    Trains a learner agent (agent 0) against randomly sampled historical
    partner policies. Uses MAPPO-style per-agent actors with PPO updates.

    Key mechanism:
    - Agent 0 is the learner (receives gradient updates)
    - Agents 1..N-1 are partners loaded from frozen historical checkpoints
    - At each checkpoint_freq interval, save learner weights to population
    - Sample partner from population (biased toward recent with sample_recent_prob)
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import copy

        self.checkpoint_freq = kwargs.get('checkpoint_freq', 10000)
        self.sample_recent_prob = kwargs.get('sample_recent_prob', 0.5)
        self.min_population = kwargs.get('min_population', 3)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.n_steps = kwargs.get('n_steps', 2048)
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.net_arch = kwargs.get('net_arch', [128, 128])

        self.obs_dim = self.obs_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)

        # Shared critic (takes global state)
        self.critic = self._build_network(self.obs_dim, 1).to(self.device_torch)

        # Per-agent actors
        self.actors = nn.ModuleList([
            self._build_network(self.obs_dim, 1) for _ in range(self.n_agents)
        ]).to(self.device_torch)

        # Log std for actions
        self.log_stds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, device=self.device_torch))
            for _ in range(self.n_agents)
        ])

        # Optimizer: only learner (agent 0) actor + shared critic
        learner_params = (list(self.actors[0].parameters()) +
                          list(self.critic.parameters()) +
                          [self.log_stds[0]])
        self.optimizer = optim.Adam(learner_params, lr=self.learning_rate)

        # Policy checkpoint population (list of serialized state_dicts)
        self.policy_population = []

    def _build_network(self, input_dim, output_dim):
        import torch.nn as nn
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _save_learner_to_population(self):
        """Save current learner (agent 0) policy to the checkpoint population."""
        import copy
        checkpoint = {
            'actor': copy.deepcopy(self.actors[0].state_dict()),
            'log_std': self.log_stds[0].data.clone()
        }
        self.policy_population.append(checkpoint)

    def _load_partner_from_population(self):
        """Sample a historical checkpoint and load into partner actors (agents 1..N-1)."""
        import torch

        if len(self.policy_population) < self.min_population:
            return  # Keep self-play until enough diversity

        # Biased sampling: sample_recent_prob chance of most recent, else uniform
        if np.random.random() < self.sample_recent_prob:
            idx = len(self.policy_population) - 1
        else:
            idx = np.random.randint(0, len(self.policy_population))

        checkpoint = self.policy_population[idx]

        # Load into all partner actors (agents 1..N-1) — frozen, no grad
        for i in range(1, self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actor'])
            self.log_stds[i].data.copy_(checkpoint['log_std'])
            # Freeze partner actors
            for param in self.actors[i].parameters():
                param.requires_grad = False

    def train(self, total_timesteps: int):
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        import torch

        obs, _ = self.env.reset(seed=self.seed)
        episode_return = 0.0
        timesteps = 0
        steps_since_checkpoint = 0

        while timesteps < total_timesteps:
            # Collect rollout
            observations = []
            actions_list = []
            rewards_list = []
            dones_list = []
            log_probs_list = []
            values_list = []

            for _ in range(self.n_steps):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_torch)

                    agent_actions = []
                    agent_log_probs = []
                    for i, actor in enumerate(self.actors):
                        mean = actor(obs_tensor)
                        std = torch.exp(self.log_stds[i])
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        agent_actions.append(action.item())
                        agent_log_probs.append(log_prob)

                    action_array = np.array(agent_actions, dtype=np.float32)
                    if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                        action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

                    value = self.critic(obs_tensor)

                next_obs, reward, terminated, truncated, info = self.env.step(action_array)
                done = terminated or truncated

                step_reward = np.sum(reward) if isinstance(reward, np.ndarray) else reward
                episode_return += step_reward

                observations.append(obs)
                actions_list.append(action_array)
                rewards_list.append(step_reward)
                dones_list.append(done)
                log_probs_list.append(torch.stack(agent_log_probs).mean())
                values_list.append(value)

                obs = next_obs
                timesteps += 1
                steps_since_checkpoint += 1

                if callback is not None and timesteps % 5000 == 0:
                    callback(timesteps)
                if timesteps % 5000 == 0:
                    self.training_metrics.flush(timesteps)

                if done:
                    self.training_returns.append(float(episode_return))
                    self.training_timesteps.append(timesteps)
                    episode_return = 0.0
                    obs, _ = self.env.reset()

                # Checkpoint and partner swap at checkpoint_freq intervals
                if steps_since_checkpoint >= self.checkpoint_freq:
                    self._save_learner_to_population()
                    self._load_partner_from_population()
                    steps_since_checkpoint = 0

            # PPO update (only learner actor + shared critic get gradients)
            self._update(observations, actions_list, rewards_list, dones_list,
                         log_probs_list, values_list)

    def _update(self, observations, actions, rewards, dones, old_log_probs, old_values):
        """PPO update for learner agent (agent 0) + shared critic."""
        import torch

        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device_torch)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device_torch)
        dones_tensor = torch.FloatTensor(dones).to(self.device_torch)
        old_log_probs_tensor = torch.stack(old_log_probs).to(self.device_torch)
        old_values_tensor = torch.cat(old_values).squeeze().to(self.device_torch)

        # Compute GAE advantages
        with torch.no_grad():
            values = self.critic(obs_tensor).squeeze()
            advantages = torch.zeros_like(rewards_tensor)
            last_gae = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                delta = rewards_tensor[t] + self.gamma * next_value * (1 - dones_tensor[t]) - values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones_tensor[t]) * last_gae

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for epoch in range(self.n_epochs):
            # Forward pass through learner actor (agent 0)
            actor_output = self.actors[0](obs_tensor)
            std = torch.exp(self.log_stds[0])
            dist = torch.distributions.Normal(actor_output.squeeze(), std)

            # Actions for agent 0
            actions_array = np.array(actions)
            agent0_actions = torch.FloatTensor(actions_array[:, 0]).to(self.device_torch)
            new_log_probs = dist.log_prob(agent0_actions)
            entropy = dist.entropy().mean()

            new_values = self.critic(obs_tensor).squeeze()

            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()

            loss = policy_loss + value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.training_metrics.record('policy_loss', policy_loss.item())
        self.training_metrics.record('value_loss', value_loss.item())
        self.training_metrics.record('population_size', len(self.policy_population))

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_torch)

            actions = []
            for i, actor in enumerate(self.actors):
                mean = actor(obs_tensor)
                if deterministic:
                    actions.append(mean.item())
                else:
                    std = torch.exp(self.log_stds[i])
                    dist = torch.distributions.Normal(mean, std)
                    actions.append(dist.sample().item())

        action_array = np.array(actions, dtype=np.float32)
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action_array = np.clip(action_array, self.action_space.low, self.action_space.high)
        return action_array

    def save(self, path: str):
        torch.save({
            'actors': self.actors.state_dict(),
            'critic': self.critic.state_dict(),
            'log_stds': self.log_stds.state_dict(),
            'policy_population': self.policy_population,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.actors.load_state_dict(data['actors'])
        self.critic.load_state_dict(data['critic'])
        self.log_stds.load_state_dict(data['log_stds'])
        if 'policy_population' in data:
            self.policy_population = data['policy_population']


# ============================================================================
# Mean-Field Algorithms
# ============================================================================

class MeanFieldActorCritic(BaseAlgorithm):
    """Mean-Field Actor-Critic for large N-agent systems."""

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.n_steps = kwargs.get('n_steps', 2048)
        self.net_arch = kwargs.get('net_arch', [128, 128])

        self.device_torch = torch.device(device)
        torch.manual_seed(seed)

        self.obs_dim = self.obs_space.shape[0]

        # Shared policy (all agents use same policy)
        self.policy = self._build_network(self.obs_dim + 1, 1).to(self.device_torch)  # +1 for mean action
        self.critic = self._build_network(self.obs_dim + 1, 1).to(self.device_torch)
        # nn.Parameter needs to be created on device or moved with a module
        self.log_std = nn.Parameter(torch.zeros(1, device=self.device_torch))

        params = list(self.policy.parameters()) + list(self.critic.parameters()) + [self.log_std]
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

    def _build_network(self, input_dim: int, output_dim: int):
        import torch.nn as nn
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def train(self, total_timesteps: int):
        """Train without callback."""
        self._train_impl(total_timesteps, callback=None)

    def train_with_callback(self, total_timesteps: int, callback=None):
        """Train with optional progress callback."""
        self._train_impl(total_timesteps, callback=callback)

    def _train_impl(self, total_timesteps: int, callback=None):
        """Internal training implementation with optional callback support."""
        import torch

        obs, _ = self.env.reset(seed=self.seed)
        timesteps = 0
        mean_action = 0.5  # Initial mean action estimate
        episode_return = 0.0

        while timesteps < total_timesteps:
            observations = []
            actions_list = []
            rewards = []
            dones = []

            for _ in range(self.n_steps):
                obs_with_mean = np.concatenate([obs, [mean_action]])

                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_with_mean).unsqueeze(0).to(self.device_torch)

                    # Get action from policy
                    mean = self.policy(obs_tensor)
                    std = torch.exp(self.log_std)
                    dist = torch.distributions.Normal(mean, std)
                    action_sample = dist.sample()

                    # All agents use same policy
                    actions = []
                    for _ in range(self.n_agents):
                        a = dist.sample().item()
                        actions.append(a)

                action_array = np.array(actions, dtype=np.float32)
                if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                    action_array = (action_array + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
                    action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

                next_obs, reward, terminated, truncated, info = self.env.step(action_array)
                done = terminated or truncated

                step_reward = np.sum(reward) if isinstance(reward, np.ndarray) else reward
                episode_return += float(step_reward)

                observations.append(obs_with_mean)
                actions_list.append(action_array)
                rewards.append(step_reward)
                dones.append(done)

                # Update mean action estimate
                if hasattr(self.action_space, 'high'):
                    mean_action = 0.9 * mean_action + 0.1 * (action_array.mean() / self.action_space.high[0])
                else:
                    mean_action = 0.9 * mean_action + 0.1 * (action_array.mean() / 100)

                obs = next_obs
                timesteps += 1

                # Call progress callback periodically (every 1000 steps)
                if callback is not None and timesteps % 5000 == 0:
                    callback(timesteps)
                if timesteps % 5000 == 0:
                    self.training_metrics.flush(timesteps)

                if done:
                    self.training_returns.append(float(episode_return))
                    self.training_timesteps.append(timesteps)
                    episode_return = 0.0
                    obs, _ = self.env.reset()
                    mean_action = 0.5

            # Simple policy gradient update
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.append(G)
            returns = returns[::-1]

            returns = torch.FloatTensor(returns).to(self.device_torch)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device_torch)
            values = self.critic(obs_tensor).squeeze()
            advantages = returns - values.detach()

            # Policy loss
            means = self.policy(obs_tensor)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(means.squeeze(), std)

            actions_tensor = torch.FloatTensor([a.mean() for a in actions_list]).to(self.device_torch)
            if hasattr(self.action_space, 'high'):
                actions_normalized = (actions_tensor / self.action_space.high[0]) * 2 - 1
            else:
                actions_normalized = (actions_tensor / 100) * 2 - 1

            log_probs = dist.log_prob(actions_normalized)
            policy_loss = -(log_probs * advantages).mean()

            # Value loss
            value_loss = 0.5 * ((values - returns) ** 2).mean()

            # Total loss
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.training_metrics.record('policy_loss', policy_loss.item())
            self.training_metrics.record('value_loss', value_loss.item())
            self.training_metrics.record('mean_action', float(mean_action))

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        import torch

        mean_action = 0.5  # Use default mean action for evaluation
        obs_with_mean = np.concatenate([obs, [mean_action]])

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_with_mean).unsqueeze(0).to(self.device_torch)

            mean = self.policy(obs_tensor)

            if deterministic:
                action = mean.item()
            else:
                std = torch.exp(self.log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().item()

        # All agents get same action (mean-field assumption)
        actions = np.full(self.n_agents, action, dtype=np.float32)

        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            actions = (actions + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions

    def save(self, path: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'log_std': self.log_std.data,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device_torch, weights_only=False)
        self.policy.load_state_dict(data['policy'])
        self.critic.load_state_dict(data['critic'])
        self.log_std.data.copy_(data['log_std'])


# =============================================================================
# Experimental oracle (NOT used in paper results)
# =============================================================================

class DynamicLoyaltyOracle(BaseAlgorithm):
    """Dynamic-programming oracle for TR-3 environments (experimental).

    .. warning::
        This oracle is preserved for reference but was **not** used in the
        released dataset. It is not registered in :mod:`experiments.config`
        and should not be used to reproduce paper numbers. Use
        :class:`LoyaltyAugmentedOracle` or :class:`SocialOptimumOracle` for
        TR-3 oracle baselines instead.

    Unlike :class:`SocialOptimumOracle` and :class:`LoyaltyAugmentedOracle`
    (which use static formulas at fixed ``theta=0.9``), this oracle computes
    the optimal action sequence over the full episode by accounting for
    dynamic loyalty evolution.

    The key insight: loyalty starts at ``theta(0)=0.5`` and grows with
    cooperation. Higher loyalty increases rewards at future steps. A
    forward-looking oracle should cooperate more in early steps to build
    loyalty, then exploit the accumulated loyalty for higher rewards in
    later steps.

    Algorithm:

    1. Forward simulation: for a range of candidate constant action levels,
       simulate the loyalty trajectory and compute total episodic return.
    2. Grid search with refinement: coarse grid then fine grid around best.
    3. The oracle plays the action level that maximizes total return.

    This is computationally tractable because all agents are symmetric in
    TR-3 (same endowments, same loyalty dynamics) and the loyalty update is
    deterministic given actions. Searching over constant action levels is
    near-optimal because loyalty dynamics are monotonic under sustained
    cooperation.
    """

    def __init__(self, env, device: str = "cpu", seed: int = 0, **kwargs):
        super().__init__(env, device, seed, **kwargs)

        if hasattr(env, "tr3_params"):
            params = env.tr3_params
            self.omega = getattr(params, "omega", 1.0)
            self.beta = getattr(params, "beta", 0.5)
            self.c = getattr(params, "c", 1.0)
            self.phi_B = getattr(params, "phi_B", 0.8)
            self.phi_C = getattr(params, "phi_C", 0.3)
            self.loyalty_horizon = getattr(params, "loyalty_horizon", 10)
        else:
            self.omega = 1.0
            self.beta = 0.5
            self.c = 1.0
            self.phi_B = 0.8
            self.phi_C = 0.3
            self.loyalty_horizon = 10

        if hasattr(env, "max_steps"):
            self.horizon = env.max_steps
        elif hasattr(env, "spec") and hasattr(env.spec, "max_episode_steps"):
            self.horizon = env.spec.max_episode_steps or 100
        else:
            self.horizon = 100

        if hasattr(env, "endowments"):
            self.endowments = env.endowments
        else:
            self.endowments = np.full(self.n_agents, 100.0)

        if hasattr(self.action_space, "high"):
            self.action_high = self.action_space.high
        else:
            self.action_high = np.full(self.n_agents, 100.0)

        self.optimal_action = self._compute_optimal_action()

    def _simulate_episode(self, action_level: float) -> float:
        """Simulate one episode with all agents playing ``action_level``.

        Returns total episodic return (sum of rewards across all steps).
        Models loyalty dynamics: initial loyalty 0.5, loyalty = mean
        cooperation rate over ``loyalty_horizon``, cooperation rate =
        action / endowment.
        """
        n = self.n_agents
        endow = self.endowments[0] if len(self.endowments) > 0 else 100.0
        coop_rate = np.clip(action_level / endow, 0.0, 1.0)

        total_return = 0.0
        action_history_rates = []

        for t in range(self.horizon):
            action_history_rates.append(coop_rate)

            horizon_window = min(len(action_history_rates), self.loyalty_horizon)
            recent_rates = action_history_rates[-horizon_window:]
            loyalty = np.clip(np.mean(recent_rates), 0.0, 1.0)

            # Team production: Q(a) = omega * (sum a_i)^beta
            total_effort = action_level * n
            Q = self.omega * (total_effort ** self.beta)

            # Base payoff per agent: (1/n)*Q - c*a_i
            base_payoff = Q / n - self.c * action_level
            avg_teammate_payoff = base_payoff  # symmetric case

            # Loyalty modifier
            loyalty_mod = loyalty * (
                self.phi_B * avg_teammate_payoff + self.phi_C * self.c * action_level
            )

            step_reward = base_payoff + loyalty_mod
            total_return += step_reward * n

        return total_return

    def _compute_optimal_action(self) -> float:
        """Find the action level that maximizes total episodic return.

        Uses a coarse grid search over [0, action_max] in 50 steps, then
        refines with a fine grid of 100 steps around the best coarse value.
        """
        endow = self.endowments[0] if len(self.endowments) > 0 else 100.0
        action_max = min(endow, self.action_high[0] if len(self.action_high) > 0 else 100.0)

        best_action = 0.0
        best_return = -1e18
        coarse_step = action_max / 50.0

        for i in range(51):
            action = i * coarse_step
            ret = self._simulate_episode(action)
            if ret > best_return:
                best_return = ret
                best_action = action

        lo = max(0.0, best_action - coarse_step)
        hi = min(action_max, best_action + coarse_step)
        fine_step = (hi - lo) / 100.0

        for i in range(101):
            action = lo + i * fine_step
            ret = self._simulate_episode(action)
            if ret > best_return:
                best_return = ret
                best_action = action

        return best_action

    def train(self, total_timesteps: int):
        """No training needed; the optimal action is computed at init time."""
        return

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return the optimal constant action for every agent."""
        actions = np.full(self.n_agents, self.optimal_action, dtype=np.float32)
        actions = np.clip(actions, 0, self.action_high)
        return actions


# =============================================================================
# Algorithm class registry
# =============================================================================

#: Maps algorithm class names to class objects. Used by orchestration code
#: to resolve the class name stored in :class:`experiments.config.AlgorithmSpec`.
#: Every class listed in :data:`experiments.config.ALL_ALGORITHMS` appears here.
ALGORITHM_CLASSES: Dict[str, type] = {
    # Heuristic baselines
    "RandomPolicy": RandomPolicy,
    "ConstantPolicy": ConstantPolicy,
    "TitForTatPolicy": TitForTatPolicy,
    # Game-theoretic oracles
    "CoopetitiveEquilibriumOracle": CoopetitiveEquilibriumOracle,
    "NashEquilibriumOracle": NashEquilibriumOracle,
    "SocialOptimumOracle": SocialOptimumOracle,
    "TrustAwareEquilibriumOracle": TrustAwareEquilibriumOracle,
    "LoyaltyAugmentedOracle": LoyaltyAugmentedOracle,
    "ReciprocityEquilibriumOracle": ReciprocityEquilibriumOracle,
    "BoundedReciprocityOracle": BoundedReciprocityOracle,
    # Training algorithms — independent learners
    "IndependentPPO": IndependentPPO,
    "IndependentA2C": IndependentA2C,
    "IndependentSAC": IndependentSAC,
    "IndependentREINFORCE": IndependentREINFORCE,
    "LOLA": LOLA,
    "SelfPlayPPO": SelfPlayPPO,
    "FictitiousCoPlay": FictitiousCoPlay,
    # Training algorithms — CTDE
    "MAPPO": MAPPO,
    "MADDPG": MADDPG,
    "MATD3": MATD3,
    "MASAC": MASAC,
    "M3DDPG": M3DDPG,
    "QMIX": QMIX,
    "VDN": VDN,
    "COMA": COMA,
    "MeanFieldActorCritic": MeanFieldActorCritic,
    # Experimental — not registered in config
    "DynamicLoyaltyOracle": DynamicLoyaltyOracle,
}


def get_algorithm_class(class_name: str) -> type:
    """Return the algorithm class matching ``class_name``.

    The lookup covers every class in :data:`ALGORITHM_CLASSES`. For the set
    of classes that are registered in :mod:`experiments.config`, the mapping
    is also consistent with :attr:`experiments.config.AlgorithmSpec.class_name`.

    Args:
        class_name: Class name as stored in :class:`experiments.config.AlgorithmSpec`.

    Returns:
        The class object.

    Raises:
        KeyError: If ``class_name`` is not in :data:`ALGORITHM_CLASSES`.
    """
    if class_name not in ALGORITHM_CLASSES:
        raise KeyError(
            f"Unknown algorithm class: {class_name!r}. "
            f"Known classes: {sorted(ALGORITHM_CLASSES)}"
        )
    return ALGORITHM_CLASSES[class_name]


def make_algorithm(spec, env, device: str = "cpu", seed: int = 0):
    """Instantiate an algorithm from an :class:`experiments.config.AlgorithmSpec`.

    Args:
        spec: The algorithm specification (from :mod:`experiments.config`).
        env: The environment instance to pass to the algorithm.
        device: ``'cpu'`` or ``'cuda'``.
        seed: Random seed for this instance.

    Returns:
        An instantiated algorithm ready for ``train`` and ``predict``.
    """
    cls = get_algorithm_class(spec.class_name)
    return cls(env=env, device=device, seed=seed, **spec.params)
