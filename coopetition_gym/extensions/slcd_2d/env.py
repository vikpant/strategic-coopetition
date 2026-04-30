"""SLCD environment with 2 action dimensions per agent.

Subclasses `coopetition_gym.envs.SLCDEnv` to add an appropriation dimension
p_i in [0, 1] alongside the v1 cooperation dimension c_i in [0, e_i].

When every p_i is clamped to zero, this environment's reward stream is
bit-exact to the v1 SLCDEnv — enforced by tests/test_backward_compat.py.

Env id
------
`SLCDAppropriation-v1ext0` (suffix `v1ext0` signals "extension of v1, variant 0").
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray

from coopetition_gym.envs import SLCDEnv

from .utility import (
    AppropriationParameters,
    compute_2d_integrated_utilities,
    compute_2d_private_payoffs,
)

VALID_REWARD_TYPES = ("integrated", "private", "cooperative")


_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"


def load_default_appropriation_params() -> AppropriationParameters:
    with _CALIBRATION_PATH.open() as fh:
        data = json.load(fh)
    return AppropriationParameters(**data["parameters"])


class SLCDAppropriationEnv(SLCDEnv):
    """Samsung-Sony SLCD with a second (appropriation) action dimension.

    Per-agent action is ordered ``(c_i, p_i)``. The Gymnasium-flat action
    space has shape ``(2 * n_agents,)``: ``[c_0, p_0, c_1, p_1, ...]``.

    Notes
    -----
    Appropriation reward is computed *in place of* v1's base reward for this
    step — not added on top. This keeps backward compat exact: the v1 formula
    is the p=0 special case of the 2D formula, not an independent term.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "SLCDAppropriation-v1ext0",
        "source": "extensions/slcd_2d (post-v1; not part of coopetition_gym)",
    }

    def __init__(
        self,
        appr_params: Optional[AppropriationParameters] = None,
        reward_type: Optional[str] = None,
        max_steps: int = 100,
        trust_enabled: bool = True,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_steps=max_steps,
            trust_enabled=trust_enabled,
            render_mode=render_mode,
            **kwargs,
        )
        self.appr_params: AppropriationParameters = (
            appr_params if appr_params is not None else load_default_appropriation_params()
        )
        if reward_type is None:
            reward_type = os.environ.get("COOPETITION_REWARD_TYPE", "integrated").lower()
        if reward_type not in VALID_REWARD_TYPES:
            raise ValueError(
                f"reward_type must be one of {VALID_REWARD_TYPES}, got {reward_type!r}"
            )
        self.reward_type = reward_type

        per_agent_low = np.tile([0.0, 0.0], self.n_agents).astype(np.float32)
        per_agent_high = np.empty(2 * self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            per_agent_high[2 * i] = float(self.endowments[i])
            per_agent_high[2 * i + 1] = 1.0
        self.action_space = spaces.Box(
            low=per_agent_low, high=per_agent_high, dtype=np.float32
        )

        self._appropriation: np.ndarray = np.zeros(self.n_agents, dtype=np.float32)

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------
    def _split_action(
        self, action: NDArray[np.floating]
    ) -> Tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2 * self.n_agents:
            raise ValueError(
                f"Expected action of shape ({2 * self.n_agents},), got {action.shape}"
            )
        c = action[0::2].copy()
        p = action[1::2].copy()
        c = np.clip(c, 0.0, self.endowments)
        p = np.clip(p, 0.0, 1.0)
        return c, p

    # ------------------------------------------------------------------
    # Gymnasium step override
    # ------------------------------------------------------------------
    def step(
        self, action: ActType
    ) -> Tuple[ObsType, np.ndarray, bool, bool, Dict[str, Any]]:
        c, p = self._split_action(np.asarray(action))

        self._action_history.append(c.copy())
        self._state["actions"] = c.astype(np.float32)
        self._appropriation = p.astype(np.float32)

        if (
            self.trust_enabled
            and self._trust_state is not None
            and self.trust_model is not None
        ):
            self._trust_state = self.trust_model.update(
                self._trust_state, c, self.baselines, self.D
            )
            self._state["trust"] = self._trust_state.trust_matrix.copy()
            self._state["reputation"] = self._trust_state.reputation_matrix.copy()

        trust_matrix = None
        if self.trust_enabled and self._trust_state is not None:
            n = self.n_agents
            eff = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        eff[i, j] = self._trust_state.get_effective_trust(i, j)
            trust_matrix = eff
        if self.reward_type == "private":
            rewards64 = compute_2d_private_payoffs(
                c=c, p=p,
                endowments=self.endowments.astype(np.float64),
                alpha=self.alpha.astype(np.float64),
                theta=float(self.value_params.theta),
                gamma=float(self.value_params.gamma),
                appr_params=self.appr_params,
            )
        elif self.reward_type == "integrated":
            rewards64 = compute_2d_integrated_utilities(
                c=c, p=p,
                endowments=self.endowments.astype(np.float64),
                alpha=self.alpha.astype(np.float64),
                D=self.D.astype(np.float64),
                theta=float(self.value_params.theta),
                gamma=float(self.value_params.gamma),
                appr_params=self.appr_params,
                trust_matrix=trust_matrix,
            )
        else:  # cooperative
            integrated = compute_2d_integrated_utilities(
                c=c, p=p,
                endowments=self.endowments.astype(np.float64),
                alpha=self.alpha.astype(np.float64),
                D=self.D.astype(np.float64),
                theta=float(self.value_params.theta),
                gamma=float(self.value_params.gamma),
                appr_params=self.appr_params,
                trust_matrix=trust_matrix,
            )
            rewards64 = np.full_like(integrated, float(np.mean(integrated)))
        utilities = rewards64.astype(np.float32)

        self._episode_rewards.append(utilities.copy())
        self._step_count += 1

        terminated = self._check_terminated()
        truncated = self._check_truncated()

        obs = self._get_legacy_observation()
        info = self._get_legacy_info()
        info["cooperation"] = c.copy()
        info["appropriation"] = p.copy()
        info["private_payoffs_2d"] = compute_2d_private_payoffs(
            c=c,
            p=p,
            endowments=self.endowments.astype(np.float64),
            alpha=self.alpha.astype(np.float64),
            theta=float(self.value_params.theta),
            gamma=float(self.value_params.gamma),
            appr_params=self.appr_params,
        )
        return obs, utilities, terminated, truncated, info

    def _get_legacy_info(self) -> Dict[str, Any]:
        info = super()._get_legacy_info()
        info["appropriation_mean"] = float(np.mean(self._appropriation))
        return info
