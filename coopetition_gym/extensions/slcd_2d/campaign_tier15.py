"""Tier 1.5 campaign orchestrator.

Three stages, each resumable independently:

  A) Sensitivity sweep    — IPPO + ISAC × 25 (eta, beta) cells × 30 seeds = 1500 runs
  B) Algorithm verify     — MADDPG + MASAC × 1 baseline cell × 30 seeds   =   60 runs
  C) Calibration fit      — coordinate-descent on (kappa, xi)              =  200 runs
                            (20 grid points × 10 inner IPPO runs)

Total: ~1,760 runs. ~900 GPU-hr on 8×4090 (one day, ~$20–25).

Delegates to:
  - ``campaign_tier1.py`` for stages A and B (reuses its _run_one worker)
  - ``calibrate.py`` for stage C

CLI
---
  python -m extensions.slcd_2d.campaign_tier15 \
      --output /workspace/results_slcd2d_tier15/ \
      --seeds 200-229 \
      --num-gpus 8 --max-workers 32

Stages can be selected with ``--stages A,B,C`` (default: all three).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from .algorithms import list_algorithms
from .calibrate import WAYPOINT_TARGET_REGISTRY, calibrate
from .campaign_tier1 import (
    _assign_device,
    _parse_floats,
    _parse_seeds,
    _run_one,
)
from .env import VALID_REWARD_TYPES
from .sweep import DEFAULT_BETAS, DEFAULT_ETAS, enumerate_runs, tier1_cells

logger = logging.getLogger("slcd2d.tier15")


BASELINE_ETA = 0.40
BASELINE_BETA = 0.60


def _run_stage_sweep(
    algorithms: List[str],
    seeds: List[int],
    reward_types: List[str],
    etas,
    betas,
    kappa: float,
    xi: float,
    timesteps: int,
    max_steps: int,
    eval_episodes: int,
    max_workers: int,
    num_gpus: int,
    output_root: Path,
) -> dict:
    cells = tier1_cells(etas=etas, betas=betas,
                        baseline_eta=BASELINE_ETA, baseline_beta=BASELINE_BETA)
    runs = list(enumerate_runs(cells, algorithms, seeds, reward_types))
    logger.info("Stage A sweep: %d runs", len(runs))
    return _dispatch_runs(runs, timesteps, max_steps, kappa, xi,
                          eval_episodes, max_workers, num_gpus, output_root)


def _run_stage_algo_verify(
    algorithms: List[str],
    seeds: List[int],
    reward_types: List[str],
    kappa: float,
    xi: float,
    timesteps: int,
    max_steps: int,
    eval_episodes: int,
    max_workers: int,
    num_gpus: int,
    output_root: Path,
) -> dict:
    baseline_only = tier1_cells(
        etas=(BASELINE_ETA,), betas=(BASELINE_BETA,),
        baseline_eta=BASELINE_ETA, baseline_beta=BASELINE_BETA,
    )
    runs = list(enumerate_runs(baseline_only, algorithms, seeds, reward_types))
    logger.info("Stage B algo-verify: %d runs (baseline cell only)", len(runs))
    return _dispatch_runs(runs, timesteps, max_steps, kappa, xi,
                          eval_episodes, max_workers, num_gpus, output_root)


def _run_stage_calibrate(
    kappa_bounds, xi_bounds,
    initial_kappa: float,
    initial_xi: float,
    inner_seeds: List[int],
    timesteps: int,
    max_steps: int,
    eval_episodes: int,
    grid_resolution: int,
    max_outer: int,
    num_gpus: int,
    output_root: Path,
    waypoint_target_set: str = "A_flat_peak",
) -> dict:
    """Run calibration TWICE (endpoint + waypoint) and report side-by-side.

    Per reviewer feedback: if both calibrations converge to (kappa, xi) within
    tolerance, the trajectory and attractor are mutually consistent and the
    appendix story is much stronger. If they diverge, that divergence is
    itself a reportable finding.
    """
    if waypoint_target_set not in WAYPOINT_TARGET_REGISTRY:
        raise ValueError(f"Unknown waypoint_target_set {waypoint_target_set!r}; "
                         f"choose one of {list(WAYPOINT_TARGET_REGISTRY)}")
    waypoint_targets = WAYPOINT_TARGET_REGISTRY[waypoint_target_set]

    logger.info("Stage C calibrate (dual): fitting (kappa, xi) at fixed eta=%.2f beta=%.2f",
                BASELINE_ETA, BASELINE_BETA)
    logger.info("  waypoint target set: %s = %s", waypoint_target_set, waypoint_targets)
    device = "cuda:0" if num_gpus > 0 else "cpu"

    summary: dict = {
        "eta_fixed": BASELINE_ETA,
        "beta_fixed": BASELINE_BETA,
        "waypoint_target_set": waypoint_target_set,
        "waypoint_targets": waypoint_targets,
    }

    for objective_name in ("endpoint", "waypoint"):
        logger.info("  -> running calibration with objective=%s", objective_name)
        targets_for_run = waypoint_targets if objective_name == "waypoint" else None
        result = calibrate(
            eta=BASELINE_ETA,
            beta=BASELINE_BETA,
            kappa_bounds=kappa_bounds,
            xi_bounds=xi_bounds,
            initial_kappa=initial_kappa,
            initial_xi=initial_xi,
            inner_seeds=inner_seeds,
            algorithm="IPPO",
            grid_resolution=grid_resolution,
            max_outer=max_outer,
            timesteps=timesteps,
            max_steps=max_steps,
            eval_episodes=eval_episodes,
            device=device,
            objective_fn=objective_name,
            targets=targets_for_run,
            output_path=output_root / "calibration" / f"{objective_name}_result.json",
        )
        summary[objective_name] = {
            "kappa_star": result.kappa_star,
            "xi_star": result.xi_star,
            "final_objective": result.final_objective,
            "converged": result.converged,
            "outer_iters": result.outer_iters,
        }

    # Side-by-side comparison
    ep = summary["endpoint"]
    wp = summary["waypoint"]
    d_kappa = abs(ep["kappa_star"] - wp["kappa_star"])
    d_xi = abs(ep["xi_star"] - wp["xi_star"])
    summary["agreement"] = {
        "delta_kappa": d_kappa,
        "delta_xi": d_xi,
        "agree_within_tol": bool(d_kappa < 0.05 and d_xi < 1.0),
    }
    logger.info("Stage C agreement: delta_kappa=%.4f delta_xi=%.4f agree=%s",
                d_kappa, d_xi, summary["agreement"]["agree_within_tol"])

    with (output_root / "calibration" / "dual_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _dispatch_runs(runs, timesteps, max_steps, kappa, xi,
                   eval_episodes, max_workers, num_gpus, output_root):
    n_ok = n_skip = n_fail = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for idx, run in enumerate(runs):
            device = _assign_device(idx, num_gpus)
            futures.append(pool.submit(
                _run_one, run, timesteps, max_steps,
                kappa, xi, eval_episodes, device, str(output_root),
            ))
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if res["status"] == "ok":
                n_ok += 1
                logger.info("[%d/%d] OK %s/%s s=%d in %.1fs",
                            i, len(runs), res["run"]["algorithm"],
                            res["run"]["cell_id"], res["run"]["seed"], res["wall_s"])
            elif res["status"] == "skipped":
                n_skip += 1
            else:
                n_fail += 1
                logger.error("[%d/%d] FAIL %s: %s", i, len(runs),
                             res["run"].get("cell_id", "?"), res.get("error", ""))
    return {
        "runs_ok": n_ok, "runs_skipped": n_skip, "runs_failed": n_fail,
        "wall_s": time.time() - started,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Tier 1.5 2D SLCD campaign")
    p.add_argument("--stages", type=str, default="A,B,C",
                   help="Which stages to run: A=sweep, B=algo-verify, C=calibrate")
    p.add_argument("--sweep-algorithms", type=str, default="IPPO,ISAC")
    p.add_argument("--verify-algorithms", type=str, default="MADDPG,MASAC")
    p.add_argument("--seeds", type=str, default="200-229",
                   help="30 seeds by default (vs. 20 in Tier 1)")
    p.add_argument("--reward-types", type=str, default="integrated")
    p.add_argument("--etas", type=str, default=None)
    p.add_argument("--betas", type=str, default=None)
    p.add_argument("--kappa", type=float, default=0.5,
                   help="Initial kappa for stages A, B; starting point for stage C.")
    p.add_argument("--xi", type=float, default=15.0,
                   help="Initial xi for stages A, B; starting point for stage C.")
    p.add_argument("--kappa-bounds", type=str, default="0.1,2.0")
    p.add_argument("--xi-bounds", type=str, default="5.0,30.0")
    p.add_argument("--calibrate-inner-seeds", type=str, default="300-309",
                   help="Inner seeds for each calibration evaluation (default 10).")
    p.add_argument("--calibrate-grid-resolution", type=int, default=5)
    p.add_argument("--calibrate-max-outer", type=int, default=2)
    p.add_argument("--waypoint-target-set", type=str, default="A_flat_peak",
                   choices=list(WAYPOINT_TARGET_REGISTRY.keys()),
                   help="Which reviewer's waypoint target schedule to fit against. "
                        "A_flat_peak = Rev A JV-internal peak-cooperation (default); "
                        "A_rising = Rev A rising-then-falling; "
                        "B_monotonic = Rev B firm-level-competition monotonic decline.")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-workers", type=int, default=32)
    p.add_argument("--num-gpus", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s | %(levelname)s | %(message)s")

    stages = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ("A", "B", "C"):
            raise SystemExit(f"Unknown stage {s!r}; use subset of A,B,C")

    sweep_algos = [a.strip() for a in args.sweep_algorithms.split(",") if a.strip()]
    verify_algos = [a.strip() for a in args.verify_algorithms.split(",") if a.strip()]
    reward_types = [r.strip() for r in args.reward_types.split(",") if r.strip()]
    seeds = _parse_seeds(args.seeds)
    inner_seeds = _parse_seeds(args.calibrate_inner_seeds)
    etas = _parse_floats(args.etas, DEFAULT_ETAS)
    betas = _parse_floats(args.betas, DEFAULT_BETAS)
    kappa_bounds = tuple(float(x) for x in args.kappa_bounds.split(","))
    xi_bounds = tuple(float(x) for x in args.xi_bounds.split(","))

    for algo in sweep_algos + verify_algos:
        if algo not in list_algorithms():
            raise SystemExit(f"Unknown algorithm {algo!r}")
    for rt in reward_types:
        if rt not in VALID_REWARD_TYPES:
            raise SystemExit(f"Invalid reward_type {rt!r}")

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tier": "1.5",
        "sensitivity_scope": "eta_beta_at_fixed_kappa_xi",
        "waypoint_target_set": args.waypoint_target_set,
        "waypoint_targets": WAYPOINT_TARGET_REGISTRY[args.waypoint_target_set],
        "stages": stages,
        "sweep_algorithms": sweep_algos,
        "verify_algorithms": verify_algos,
        "seeds": seeds,
        "inner_calibration_seeds": inner_seeds,
        "reward_types": reward_types,
        "etas": list(etas),
        "betas": list(betas),
        "kappa": args.kappa,
        "xi": args.xi,
        "kappa_bounds": list(kappa_bounds),
        "xi_bounds": list(xi_bounds),
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "max_workers": args.max_workers,
        "num_gpus": args.num_gpus,
    }
    with (output_root / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Tier 1.5 manifest written: %s", output_root / "manifest.json")

    if args.dry_run:
        sweep_runs = len(etas) * len(betas) * len(sweep_algos) * len(seeds) * len(reward_types)
        verify_runs = len(verify_algos) * len(seeds) * len(reward_types)
        cal_runs = args.calibrate_max_outer * 2 * args.calibrate_grid_resolution * len(inner_seeds)
        logger.info("DRY RUN — Stage A: %d, Stage B: %d, Stage C: ~%d. Total ~%d runs.",
                    sweep_runs, verify_runs, cal_runs, sweep_runs + verify_runs + cal_runs)
        return 0

    summary = {}
    if "A" in stages:
        logger.info("=== Stage A: sensitivity sweep ===")
        summary["stage_A"] = _run_stage_sweep(
            algorithms=sweep_algos, seeds=seeds, reward_types=reward_types,
            etas=etas, betas=betas, kappa=args.kappa, xi=args.xi,
            timesteps=args.timesteps, max_steps=args.max_steps,
            eval_episodes=args.eval_episodes,
            max_workers=args.max_workers, num_gpus=args.num_gpus,
            output_root=output_root,
        )

    if "B" in stages:
        logger.info("=== Stage B: algorithm verification ===")
        summary["stage_B"] = _run_stage_algo_verify(
            algorithms=verify_algos, seeds=seeds, reward_types=reward_types,
            kappa=args.kappa, xi=args.xi,
            timesteps=args.timesteps, max_steps=args.max_steps,
            eval_episodes=args.eval_episodes,
            max_workers=args.max_workers, num_gpus=args.num_gpus,
            output_root=output_root,
        )

    if "C" in stages:
        logger.info("=== Stage C: calibration fit ===")
        summary["stage_C"] = _run_stage_calibrate(
            kappa_bounds=kappa_bounds, xi_bounds=xi_bounds,
            initial_kappa=args.kappa, initial_xi=args.xi,
            inner_seeds=inner_seeds,
            timesteps=args.timesteps, max_steps=args.max_steps,
            eval_episodes=args.eval_episodes,
            grid_resolution=args.calibrate_grid_resolution,
            max_outer=args.calibrate_max_outer,
            num_gpus=args.num_gpus,
            output_root=output_root,
            waypoint_target_set=args.waypoint_target_set,
        )

    with (output_root / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Tier 1.5 done. Summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
