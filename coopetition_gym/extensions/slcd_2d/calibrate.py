"""Coordinate-descent calibration of (kappa, xi) for the 2D SLCD appropriation model.

Tier 1.5 scope. Depends only on numpy + scipy (no skopt/sklearn) so the
reproducibility bar is low.

Algorithm
---------
1. Hold (eta, beta) fixed at the Tier 1 baseline (or user-supplied).
2. For each outer iteration:
   - Sweep kappa at fixed xi over ``grid_resolution`` points, evaluate objective
     at each (averaging K inner IPPO training runs), fit a parabola, pick the
     analytic vertex (clipped to bounds).
   - Repeat along xi at the new kappa.
3. Stop when the outer step size drops below ``tol`` or after ``max_outer`` iters.

Default objective
-----------------
The objective is a sum of two squared deviations designed to produce
SLCD-dissolution-like endpoints:

    L(kappa, xi) = (final_trust_mean - 0.0)^2
                 + (final_appropriation_mean - 0.30)^2

Lower = better. Reviewers can substitute their own objective by passing
``objective_fn`` — the function must accept an ``eval_stats`` dict (as produced
by campaign_tier1._evaluate) and return a scalar.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .utility import AppropriationParameters

logger = logging.getLogger("slcd2d.calibrate")


DEFAULT_TARGETS = {"final_trust_mean": 0.0, "final_appropriation_mean": 0.30}

# Multi-waypoint targets mapped to SLCD 2004-2011 historical arc.
# Two readings were proposed by reviewers; both are registered here.
# See FORMALISM.md sec 4 for historical anchors and the JV-internal vs
# firm-level distinction that drives the disagreement.

# Reviewer A (JV-internal view): Gen-7 ramp (T/4, April 2005 start), Gen-8
# capital commitment + peak capacity (T/2, Aug 2007 - Apr 2008), trust
# erosion onset (3T/4, Sony SEC 6-K 2009-07-30 Sharp SDP joint venture at
# 34% stake), dissolution (T, joint announcement 2011-12-26).
WAYPOINT_TARGETS_A_FLAT_PEAK = {
    "trust_early": 0.85, "trust_mid": 0.85, "trust_late": 0.30,
    "trust_final": 0.0, "appr_final": 0.30,
}
WAYPOINT_TARGETS_A_RISING = {
    "trust_early": 0.85, "trust_mid": 0.90, "trust_late": 0.30,
    "trust_final": 0.0, "appr_final": 0.30,
}

# Reviewer B (firm-level-competition view): T/2 captures Samsung's downstream
# market-share takeover of Sony during 2007-2008, hence trust already declining.
WAYPOINT_TARGETS_B_MONOTONIC = {
    "trust_early": 0.60, "trust_mid": 0.40, "trust_late": 0.15,
    "trust_final": 0.0, "appr_final": 0.30,
}

WAYPOINT_TARGET_REGISTRY = {
    "A_flat_peak":  WAYPOINT_TARGETS_A_FLAT_PEAK,
    "A_rising":     WAYPOINT_TARGETS_A_RISING,
    "B_monotonic":  WAYPOINT_TARGETS_B_MONOTONIC,
}

# Default: A_flat_peak.
#   - aligns with what TR-2 trust dynamics actually measure (JV-internal
#     cooperation, not firm-level market share)
#   - requires no assumption that TR-2 produces rising trust above the
#     initialization value of 0.65 (which v1 parameters may or may not)
#   - anchored in primary SEC filings and Samsung Display corporate history
#   - Rev B's B_monotonic remains available via --waypoint-target-set
DEFAULT_WAYPOINT_TARGETS = WAYPOINT_TARGETS_A_FLAT_PEAK


def default_objective(eval_stats: Dict, targets: Dict[str, float] = None) -> float:
    """Endpoint-only objective. Sum of squared deviations. Lower is better."""
    targets = targets if targets is not None else DEFAULT_TARGETS
    ft = float(eval_stats.get("eval_final_trust_mean", 1.0))
    fa_per_agent = eval_stats.get("eval_final_appropriation_mean", [0.0])
    fa = float(np.mean(fa_per_agent))
    return (ft - targets["final_trust_mean"]) ** 2 + (fa - targets["final_appropriation_mean"]) ** 2


def waypoint_objective(
    eval_stats: Dict,
    targets: Dict[str, float] = None,
    weights: Dict[str, float] = None,
) -> float:
    """Multi-waypoint trajectory objective.

    Anchors the trajectory at four points (T/4, T/2, 3T/4, T) rather than fitting
    only the endpoint. This prevents parameter regimes that produce wrong dynamics
    (instant collapse, oscillation) from winning because they happen to land at
    the right endpoint. Matches the Samsung-Sony SLCD historical arc.

    Requires ``eval_stats`` to contain ``eval_mean_trust_curve`` and
    ``eval_final_appropriation_mean`` (produced by campaign_tier1._evaluate).
    Falls back gracefully to endpoint-only if the curve is missing.
    """
    targets = targets if targets is not None else DEFAULT_WAYPOINT_TARGETS
    weights = weights if weights is not None else {
        "trust_early": 1.0, "trust_mid": 1.0, "trust_late": 1.0,
        "trust_final": 1.0, "appr_final": 1.0,
    }

    trust_curve = eval_stats.get("eval_mean_trust_curve", None)
    if trust_curve is None or len(trust_curve) < 4:
        return default_objective(eval_stats)

    curve = np.asarray(trust_curve, dtype=np.float64)
    T = len(curve) - 1  # curve has T+1 samples (reset + T steps)
    idx_early = max(1, T // 4)
    idx_mid = max(1, T // 2)
    idx_late = max(1, (3 * T) // 4)

    t_early = float(curve[idx_early])
    t_mid = float(curve[idx_mid])
    t_late = float(curve[idx_late])
    t_final = float(curve[-1])

    fa = float(np.mean(eval_stats.get("eval_final_appropriation_mean", [0.0])))

    loss = (
        weights["trust_early"] * (t_early - targets["trust_early"]) ** 2
        + weights["trust_mid"] * (t_mid - targets["trust_mid"]) ** 2
        + weights["trust_late"] * (t_late - targets["trust_late"]) ** 2
        + weights["trust_final"] * (t_final - targets["trust_final"]) ** 2
        + weights["appr_final"] * (fa - targets["appr_final"]) ** 2
    )
    return float(loss)


OBJECTIVE_REGISTRY = {
    "endpoint": default_objective,
    "waypoint": waypoint_objective,
}


def _parabolic_vertex(xs: np.ndarray, ys: np.ndarray, bounds: Tuple[float, float]) -> float:
    """Fit y = a x^2 + b x + c; return clipped vertex -b/(2a) if a>0 else argmin x."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 3:
        return float(xs[np.argmin(ys)])
    coeffs = np.polyfit(xs, ys, 2)
    a, b, _ = coeffs
    if a <= 1e-9:
        vertex = float(xs[np.argmin(ys)])
    else:
        vertex = -b / (2.0 * a)
    lo, hi = bounds
    return float(np.clip(vertex, lo, hi))


def _evaluate_point(
    kappa: float,
    xi: float,
    eta: float,
    beta: float,
    seeds: List[int],
    algorithm: str,
    timesteps: int,
    max_steps: int,
    eval_episodes: int,
    device: str,
) -> Dict:
    """Run K=len(seeds) inner training runs; average eval_stats across runs."""
    from .env import SLCDAppropriationEnv
    from .algorithms import build_algorithm
    from .campaign_tier1 import _evaluate

    params = AppropriationParameters(kappa=kappa, beta=beta, eta=eta, xi=xi)
    trusts: List[float] = []
    appr_means: List[float] = []
    trust_curves: List[List[float]] = []

    for seed in seeds:
        os.environ["COOPETITION_REWARD_TYPE"] = "integrated"
        env = SLCDAppropriationEnv(appr_params=params, max_steps=max_steps,
                                    reward_type="integrated")
        from .algorithms import prefers_cpu
        effective_device = "cpu" if prefers_cpu(algorithm) else device
        algo = build_algorithm(algorithm, env, device=effective_device, seed=seed)
        if hasattr(algo, "train"):
            algo.train(total_timesteps=timesteps)
        eval_env = SLCDAppropriationEnv(appr_params=params, max_steps=max_steps,
                                         reward_type="integrated")
        eval_stats = _evaluate(algo, eval_env, num_episodes=eval_episodes, seed=seed)
        trusts.append(float(eval_stats["eval_final_trust_mean"]))
        appr_means.append(float(np.mean(eval_stats["eval_final_appropriation_mean"])))
        curve = eval_stats.get("eval_mean_trust_curve", [])
        if curve:
            trust_curves.append(curve)

    if trust_curves:
        max_len = max(len(c) for c in trust_curves)
        mean_curve = [
            float(np.mean([c[i] for c in trust_curves if i < len(c)]))
            for i in range(max_len)
        ]
    else:
        mean_curve = []

    return {
        "eval_final_trust_mean": float(np.mean(trusts)),
        "eval_final_trust_std": float(np.std(trusts)),
        "eval_final_appropriation_mean": [float(np.mean(appr_means))],
        "eval_final_appropriation_std": float(np.std(appr_means)),
        "eval_mean_trust_curve": mean_curve,
        "inner_n": len(seeds),
    }


@dataclass
class CalibrationStep:
    iter_idx: int
    axis: str  # "kappa" or "xi"
    kappa: float
    xi: float
    sweep_values: List[float]
    sweep_objectives: List[float]
    vertex: float
    objective_at_vertex: float


@dataclass
class CalibrationResult:
    kappa_star: float
    xi_star: float
    final_objective: float
    eta_fixed: float
    beta_fixed: float
    history: List[CalibrationStep] = field(default_factory=list)
    converged: bool = False
    outer_iters: int = 0


def calibrate(
    eta: float,
    beta: float,
    kappa_bounds: Tuple[float, float] = (0.1, 2.0),
    xi_bounds: Tuple[float, float] = (5.0, 30.0),
    initial_kappa: float = 0.5,
    initial_xi: float = 15.0,
    inner_seeds: Optional[List[int]] = None,
    algorithm: str = "IPPO",
    grid_resolution: int = 5,
    max_outer: int = 2,
    tol: float = 0.02,
    timesteps: int = 500_000,
    max_steps: int = 40,
    eval_episodes: int = 10,
    device: str = "cpu",
    objective_fn: Optional[Callable[[Dict], float]] = None,
    targets: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> CalibrationResult:
    """Coordinate-descent calibration of (kappa, xi) at fixed (eta, beta)."""
    if inner_seeds is None:
        inner_seeds = list(range(300, 310))

    if objective_fn is None:
        obj_fn = lambda s: default_objective(s, targets)
    elif isinstance(objective_fn, str):
        if objective_fn not in OBJECTIVE_REGISTRY:
            raise ValueError(f"Unknown objective_fn {objective_fn!r}; "
                             f"choose one of {list(OBJECTIVE_REGISTRY)}")
        chosen = OBJECTIVE_REGISTRY[objective_fn]
        obj_fn = (lambda s: chosen(s, targets)) if targets else chosen
    else:
        obj_fn = objective_fn

    kappa = float(np.clip(initial_kappa, *kappa_bounds))
    xi = float(np.clip(initial_xi, *xi_bounds))
    history: List[CalibrationStep] = []
    prev_kappa = prev_xi = None

    for iter_idx in range(max_outer):
        # Sweep kappa at fixed xi
        kappa_grid = np.linspace(kappa_bounds[0], kappa_bounds[1], grid_resolution)
        kappa_objs = []
        for k in kappa_grid:
            stats = _evaluate_point(
                kappa=float(k), xi=xi, eta=eta, beta=beta,
                seeds=inner_seeds, algorithm=algorithm,
                timesteps=timesteps, max_steps=max_steps,
                eval_episodes=eval_episodes, device=device,
            )
            kappa_objs.append(obj_fn(stats))
            logger.info("  sweep[kappa=%.3f xi=%.2f] -> obj=%.6f", k, xi, kappa_objs[-1])
        kappa_star = _parabolic_vertex(kappa_grid, np.asarray(kappa_objs), kappa_bounds)
        stats_at_kappa_star = _evaluate_point(
            kappa=kappa_star, xi=xi, eta=eta, beta=beta,
            seeds=inner_seeds, algorithm=algorithm,
            timesteps=timesteps, max_steps=max_steps,
            eval_episodes=eval_episodes, device=device,
        )
        obj_at_kappa_star = obj_fn(stats_at_kappa_star)
        history.append(CalibrationStep(
            iter_idx=iter_idx, axis="kappa",
            kappa=kappa_star, xi=xi,
            sweep_values=kappa_grid.tolist(),
            sweep_objectives=kappa_objs,
            vertex=kappa_star, objective_at_vertex=obj_at_kappa_star,
        ))
        kappa = kappa_star

        # Sweep xi at fixed kappa
        xi_grid = np.linspace(xi_bounds[0], xi_bounds[1], grid_resolution)
        xi_objs = []
        for x in xi_grid:
            stats = _evaluate_point(
                kappa=kappa, xi=float(x), eta=eta, beta=beta,
                seeds=inner_seeds, algorithm=algorithm,
                timesteps=timesteps, max_steps=max_steps,
                eval_episodes=eval_episodes, device=device,
            )
            xi_objs.append(obj_fn(stats))
            logger.info("  sweep[kappa=%.3f xi=%.2f] -> obj=%.6f", kappa, x, xi_objs[-1])
        xi_star = _parabolic_vertex(xi_grid, np.asarray(xi_objs), xi_bounds)
        stats_at_xi_star = _evaluate_point(
            kappa=kappa, xi=xi_star, eta=eta, beta=beta,
            seeds=inner_seeds, algorithm=algorithm,
            timesteps=timesteps, max_steps=max_steps,
            eval_episodes=eval_episodes, device=device,
        )
        obj_at_xi_star = obj_fn(stats_at_xi_star)
        history.append(CalibrationStep(
            iter_idx=iter_idx, axis="xi",
            kappa=kappa, xi=xi_star,
            sweep_values=xi_grid.tolist(),
            sweep_objectives=xi_objs,
            vertex=xi_star, objective_at_vertex=obj_at_xi_star,
        ))
        xi = xi_star

        if (prev_kappa is not None and prev_xi is not None and
                abs(kappa - prev_kappa) < tol and abs(xi - prev_xi) < tol):
            converged = True
            break
        prev_kappa, prev_xi = kappa, xi
    else:
        converged = False

    result = CalibrationResult(
        kappa_star=kappa,
        xi_star=xi,
        final_objective=history[-1].objective_at_vertex if history else float("inf"),
        eta_fixed=eta,
        beta_fixed=beta,
        history=history,
        converged=converged,
        outer_iters=len(history) // 2,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "kappa_star": result.kappa_star,
            "xi_star": result.xi_star,
            "final_objective": result.final_objective,
            "eta_fixed": result.eta_fixed,
            "beta_fixed": result.beta_fixed,
            "converged": result.converged,
            "outer_iters": result.outer_iters,
            "history": [asdict(s) for s in result.history],
        }
        with output_path.open("w") as fh:
            json.dump(serializable, fh, indent=2)
    return result
