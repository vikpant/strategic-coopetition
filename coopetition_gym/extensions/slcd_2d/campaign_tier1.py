"""Tier 1 cloud-ready campaign for the 2D SLCD extension.

Scope (per the tiered plan):
- 1 environment (SLCDAppropriation-v1ext0)
- 1 reward type by default (integrated; add --reward-types for ablation)
- 2-3 continuous-action algorithms (IPPO, ISAC by default)
- 5x5 (eta, beta) sensitivity grid + labelled baseline
- 20 seeds (configurable)

Design choices
--------------
- Per-run checkpointing via SB3 model.save at completion (not per-step; each
  run is short enough that step-level checkpointing is overkill).
- Resume: a run is skipped if its final result JSON already exists.
- Parallelism: concurrent.futures.ProcessPoolExecutor. Each worker picks its
  GPU device via round-robin on CUDA_VISIBLE_DEVICES.
- Output layout matches main campaign conventions:
      <output>/<reward_type>/<algo>/<cell_id>/<seed>/{result.json, model.zip}

CLI
---
  python -m extensions.slcd_2d.campaign_tier1 \
      --algorithms IPPO,ISAC --seeds 200-219 \
      --timesteps 500000 --max-workers 32 \
      --output /workspace/results_slcd2d_tier1/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .algorithms import build_algorithm, list_algorithms, prefers_cpu
from .env import VALID_REWARD_TYPES, SLCDAppropriationEnv
from .sweep import DEFAULT_BETAS, DEFAULT_ETAS, enumerate_runs, tier1_cells
from .utility import AppropriationParameters

logger = logging.getLogger("slcd2d.tier1")


def _parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif chunk:
            seeds.append(int(chunk))
    return seeds


def _parse_floats(spec: Optional[str], default):
    if not spec:
        return default
    return tuple(float(x) for x in spec.split(","))


def _run_output_path(root: Path, run: dict) -> Path:
    return (
        root
        / run["reward_type"]
        / run["algorithm"]
        / run["cell_id"]
        / f"seed_{run['seed']}"
    )


def _build_env(run: dict, kappa: float, xi: float, max_steps: int) -> SLCDAppropriationEnv:
    params = AppropriationParameters(kappa=kappa, beta=run["beta"], eta=run["eta"], xi=xi)
    return SLCDAppropriationEnv(
        appr_params=params,
        reward_type=run["reward_type"],
        max_steps=max_steps,
    )


def _evaluate(algo, env, num_episodes: int, seed: int) -> Dict[str, object]:
    """Run deterministic evaluation episodes and collect per-step trajectories.

    Returns endpoint statistics plus mean per-step trust and appropriation curves
    (averaged across episodes). The trajectories are what enables the multi-
    waypoint calibration objective in ``calibrate.py``.
    """
    returns = np.zeros((num_episodes, env.n_agents), dtype=np.float64)
    final_trust = np.zeros(num_episodes, dtype=np.float64)
    final_appr = np.zeros((num_episodes, env.n_agents), dtype=np.float64)

    trust_traj_by_ep: List[List[float]] = []
    appr_traj_by_ep: List[List[List[float]]] = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + 10_000 + ep)
        trust_steps = [float(info.get("mean_trust", 0.0))]
        appr_steps: List[List[float]] = []
        done = False
        while not done:
            if hasattr(algo, "predict"):
                result = algo.predict(obs, deterministic=True)
                action = result[0] if isinstance(result, tuple) else result
            else:
                action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            returns[ep] += r
            trust_steps.append(float(info.get("mean_trust", 0.0)))
            appr_steps.append([float(x) for x in info.get("appropriation", np.zeros(env.n_agents))])
            done = term or trunc
        trust_traj_by_ep.append(trust_steps)
        appr_traj_by_ep.append(appr_steps)
        final_trust[ep] = float(info.get("mean_trust", 0.0))
        final_appr[ep] = info.get("appropriation", np.zeros(env.n_agents))

    # Pad trajectories to equal length for a mean curve (episodes rarely truncate early on SLCD)
    max_len_trust = max(len(t) for t in trust_traj_by_ep)
    max_len_appr = max(len(a) for a in appr_traj_by_ep)
    mean_trust_curve = np.array([
        float(np.mean([t[i] for t in trust_traj_by_ep if i < len(t)]))
        for i in range(max_len_trust)
    ])
    mean_appr_curve = np.array([
        float(np.mean([np.mean(a[i]) for a in appr_traj_by_ep if i < len(a)]))
        for i in range(max_len_appr)
    ])

    return {
        "eval_mean_return": returns.mean(axis=0).tolist(),
        "eval_std_return": returns.std(axis=0).tolist(),
        "eval_final_trust_mean": float(final_trust.mean()),
        "eval_final_appropriation_mean": final_appr.mean(axis=0).tolist(),
        "eval_mean_trust_curve": mean_trust_curve.tolist(),
        "eval_mean_appropriation_curve": mean_appr_curve.tolist(),
        "eval_episodes": num_episodes,
    }


def _run_one(
    run: dict,
    timesteps: int,
    max_steps: int,
    kappa: float,
    xi: float,
    eval_episodes: int,
    device: str,
    output_root: str,
) -> dict:
    """Execute a single (cell, algo, seed, reward_type) run. Runs in a worker process."""
    out_dir = _run_output_path(Path(output_root), run)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    if result_path.exists():
        return {"status": "skipped", "run": run, "reason": "result.json exists"}

    started = time.time()
    try:
        os.environ["COOPETITION_REWARD_TYPE"] = run["reward_type"]
        env = _build_env(run, kappa=kappa, xi=xi, max_steps=max_steps)
        effective_device = "cpu" if prefers_cpu(run["algorithm"]) else device
        algo = build_algorithm(run["algorithm"], env, device=effective_device, seed=run["seed"])

        if hasattr(algo, "train"):
            algo.train(total_timesteps=timesteps)
            training_returns = list(getattr(algo, "training_returns", []))
        else:
            training_returns = []

        eval_env = _build_env(run, kappa=kappa, xi=xi, max_steps=max_steps)
        eval_stats = _evaluate(algo, eval_env, num_episodes=eval_episodes, seed=run["seed"])

        model_path = out_dir / "model.zip"
        if hasattr(algo, "save"):
            try:
                algo.save(str(model_path))
            except Exception as save_err:
                logger.warning("save failed for %s: %s", out_dir, save_err)

        record = {
            "run": run,
            "kappa": kappa,
            "xi": xi,
            "max_steps": max_steps,
            "timesteps": timesteps,
            "device": effective_device,
            "device_assigned": device,
            "training_returns_last10": training_returns[-10:] if training_returns else [],
            "training_returns_n": len(training_returns),
            "wall_clock_s": time.time() - started,
            **eval_stats,
        }
        with result_path.open("w") as fh:
            json.dump(record, fh, indent=2)
        return {"status": "ok", "run": run, "wall_s": record["wall_clock_s"]}
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        err_path = out_dir / "error.log"
        with err_path.open("w") as fh:
            fh.write(f"{exc}\n\n{tb}")
        return {"status": "failed", "run": run, "error": str(exc)}


def _assign_device(worker_idx: int, num_gpus: int) -> str:
    if num_gpus <= 0:
        return "cpu"
    return f"cuda:{worker_idx % num_gpus}"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Tier 1 campaign for SLCDAppropriation-v1ext0")
    p.add_argument("--algorithms", type=str, default="IPPO,ISAC",
                   help=f"Comma-separated subset of {list_algorithms()}")
    p.add_argument("--seeds", type=str, default="200-219",
                   help="Comma-separated seeds with optional ranges (e.g. '200-219,225')")
    p.add_argument("--reward-types", type=str, default="integrated",
                   help=f"Comma-separated subset of {list(VALID_REWARD_TYPES)}")
    p.add_argument("--etas", type=str, default=None,
                   help=f"Override eta grid (comma-separated). Default {DEFAULT_ETAS}")
    p.add_argument("--betas", type=str, default=None,
                   help=f"Override beta grid (comma-separated). Default {DEFAULT_BETAS}")
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--xi", type=float, default=15.0)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--num-gpus", type=int, default=0,
                   help="Round-robin GPU assignment across workers. 0 = CPU-only.")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the run matrix and exit without executing.")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s | %(levelname)s | %(message)s")

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    reward_types = [r.strip() for r in args.reward_types.split(",") if r.strip()]
    seeds = _parse_seeds(args.seeds)
    etas = _parse_floats(args.etas, DEFAULT_ETAS)
    betas = _parse_floats(args.betas, DEFAULT_BETAS)

    for r in reward_types:
        if r not in VALID_REWARD_TYPES:
            raise SystemExit(f"Invalid reward type {r!r}; valid: {VALID_REWARD_TYPES}")
    for a in algorithms:
        if a not in list_algorithms():
            raise SystemExit(f"Unknown algorithm {a!r}; valid: {list_algorithms()}")

    cells = tier1_cells(etas=etas, betas=betas)
    runs = list(enumerate_runs(cells, algorithms, seeds, reward_types))
    logger.info("Matrix: %d cells x %d algos x %d seeds x %d reward_types = %d runs",
                len(cells), len(algorithms), len(seeds), len(reward_types), len(runs))

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w") as fh:
        json.dump({
            "algorithms": algorithms,
            "seeds": seeds,
            "reward_types": reward_types,
            "etas": list(etas),
            "betas": list(betas),
            "kappa": args.kappa,
            "xi": args.xi,
            "timesteps": args.timesteps,
            "eval_episodes": args.eval_episodes,
            "total_runs": len(runs),
        }, fh, indent=2)

    if args.dry_run:
        for i, r in enumerate(runs[:20]):
            logger.info("run[%d]: %s", i, r)
        if len(runs) > 20:
            logger.info("... (%d more)", len(runs) - 20)
        return 0

    futures = []
    n_ok = n_skip = n_fail = 0
    started_all = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        for idx, run in enumerate(runs):
            device = _assign_device(idx, args.num_gpus)
            futures.append(pool.submit(
                _run_one, run, args.timesteps, args.max_steps,
                args.kappa, args.xi, args.eval_episodes, device, str(output_root),
            ))
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if res["status"] == "ok":
                n_ok += 1
                logger.info("[%d/%d] OK %s in %.1fs", i, len(runs),
                            res["run"]["cell_id"], res["wall_s"])
            elif res["status"] == "skipped":
                n_skip += 1
            else:
                n_fail += 1
                logger.error("[%d/%d] FAIL %s: %s", i, len(runs),
                             res["run"]["cell_id"], res["error"])

    logger.info("Done. ok=%d skipped=%d failed=%d total_wall=%.1fs",
                n_ok, n_skip, n_fail, time.time() - started_all)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
