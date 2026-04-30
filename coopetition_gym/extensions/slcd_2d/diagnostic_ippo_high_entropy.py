"""Diagnostic: re-run IPPO at baseline (eta=0.20, beta=0.60) with ent_coef=0.05.

Rev A's upstream question: the Tier 1.5 Stage A finding that ISAC finds the
interior Nash but IPPO does not is only a clean "off-policy vs on-policy"
claim if both algorithms ran comparable exploration budgets. SB3 default PPO
uses ent_coef=0.01 (fixed); SAC uses ent_coef='auto' (adaptive max-entropy).
The asymmetry is a referee objection waiting to happen.

This diagnostic runs IPPO with ent_coef=0.05 (5x the default) on the SAME cell
where Stage A's IPPO collapsed to p=0. If IPPO with higher exploration still
collapses, the off-policy-vs-on-policy finding is robust. If it finds the
interior Nash, the finding is about exploration budget, not algorithm family.

Design
------
- Single cell: eta=0.20, beta=0.60 (matches observed Stage A data)
- 30 seeds (200-229) for apples-to-apples comparison with Stage A IPPO
- 500k timesteps (matches Stage A)
- CPU device (per the CPU_PREFERRED_ALGORITHMS routing)
- Output: /workspace/results_ippo_high_entropy_diag/seed_{n}/result.json

Runtime: ~6 min/run on 96 vCPU with 12 parallel workers + OMP_NUM_THREADS=2
(= 24 threads, leaving ~72 CPU cores for the main Tier 1.5 campaign).
Expected total wall: ~30 * 6 / 12 = ~15 min.

Usage
-----
  python -m extensions.slcd_2d.diagnostic_ippo_high_entropy \\
      --output /workspace/results_ippo_high_entropy_diag/ \\
      --max-workers 12
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
from typing import List, Optional

import numpy as np

logger = logging.getLogger("slcd2d.diag_high_entropy")


def _run_one(seed: int, timesteps: int, max_steps: int, eval_episodes: int,
             ent_coef: float, output_root: str) -> dict:
    """Run a single IPPO training with the given ent_coef."""
    import sys
    sys.path.insert(0, "/workspace/strategic-coopetition")
    from extensions.slcd_2d.env import SLCDAppropriationEnv
    from extensions.slcd_2d.utility import AppropriationParameters
    from extensions.slcd_2d.campaign_tier1 import _evaluate
    from experiments.algorithms import IndependentPPO

    out_dir = Path(output_root) / f"ent{ent_coef:.2f}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        return {"status": "skipped", "seed": seed, "ent_coef": ent_coef}

    started = time.time()
    try:
        os.environ["COOPETITION_REWARD_TYPE"] = "integrated"
        params = AppropriationParameters(kappa=0.5, beta=0.60, eta=0.20, xi=15.0)
        env = SLCDAppropriationEnv(appr_params=params, reward_type="integrated",
                                    max_steps=max_steps)
        algo = IndependentPPO(
            env, device="cpu", seed=seed,
            ent_coef=ent_coef, n_steps=2048, batch_size=64,
        )
        algo.train(total_timesteps=timesteps)

        eval_env = SLCDAppropriationEnv(appr_params=params, reward_type="integrated",
                                         max_steps=max_steps)
        eval_stats = _evaluate(algo, eval_env, num_episodes=eval_episodes, seed=seed)

        # Capture achieved policy entropy at convergence (Rev A's Q).
        # Two paths: (a) pull SB3 logger's last-logged entropy_loss, (b) compute
        # analytically from the trained policy's action distribution on a batch
        # of real observations. Both stored if available.
        policy_entropy_analytical = None
        policy_entropy_sb3_loss = None
        try:
            import numpy as np
            import torch
            sb3_vals = getattr(algo.model.logger, "name_to_value", {}) or {}
            ent_loss = sb3_vals.get("train/entropy_loss")
            if ent_loss is not None and ent_coef > 0:
                # SB3's PPO loss aggregates -ent_coef * entropy into entropy_loss
                policy_entropy_sb3_loss = -float(ent_loss) / float(ent_coef)
            obs_sample, _ = eval_env.reset(seed=seed + 999_999)
            obs_batch = np.tile(obs_sample.astype(np.float32), (64, 1))
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_batch)
                dist = algo.model.policy.get_distribution(obs_t)
                policy_entropy_analytical = float(dist.entropy().mean().item())
        except Exception:
            pass

        record = {
            "diagnostic": "ippo_high_entropy",
            "ent_coef": ent_coef,
            "cell": {"eta": 0.20, "beta": 0.60, "kappa": 0.5, "xi": 15.0},
            "seed": seed,
            "timesteps": timesteps,
            "max_steps": max_steps,
            "device": "cpu",
            "wall_clock_s": time.time() - started,
            "training_returns_last10": list(getattr(algo, "training_returns", []))[-10:],
            "training_returns_n": len(getattr(algo, "training_returns", [])),
            "policy_entropy_analytical": policy_entropy_analytical,
            "policy_entropy_sb3_loss": policy_entropy_sb3_loss,
            **eval_stats,
        }
        with result_path.open("w") as fh:
            json.dump(record, fh, indent=2)
        return {"status": "ok", "seed": seed, "ent_coef": ent_coef,
                "wall_s": record["wall_clock_s"]}
    except Exception as exc:
        tb = traceback.format_exc()
        (out_dir / "error.log").write_text(f"{exc}\n\n{tb}")
        return {"status": "failed", "seed": seed, "ent_coef": ent_coef,
                "error": str(exc)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="200-229")
    p.add_argument("--ent-coef", type=float, default=0.05,
                   help="(Deprecated) Single ent_coef. Use --ent-coefs for a list.")
    p.add_argument("--ent-coefs", type=str, default=None,
                   help="Comma-separated list of ent_coef values to sweep.")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-workers", type=int, default=12,
                   help="Keep low to avoid disrupting main Tier 1.5 campaign on same box.")
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    seeds: List[int] = []
    for chunk in args.seeds.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif chunk:
            seeds.append(int(chunk))

    if args.ent_coefs:
        ent_coefs = [float(x) for x in args.ent_coefs.split(",") if x.strip()]
    else:
        ent_coefs = [args.ent_coef]

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w") as fh:
        json.dump({
            "diagnostic": "ippo_high_entropy",
            "ent_coefs": ent_coefs,
            "cell": {"eta": 0.20, "beta": 0.60, "kappa": 0.5, "xi": 15.0},
            "seeds": seeds,
            "timesteps": args.timesteps,
            "max_workers": args.max_workers,
        }, fh, indent=2)

    total_runs = len(seeds) * len(ent_coefs)
    logger.info("Launching %d runs (%d seeds x %d ent_coefs=%s) on %d workers",
                total_runs, len(seeds), len(ent_coefs), ent_coefs, args.max_workers)

    n_ok = n_skip = n_fail = 0
    started_all = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = []
        for ent_coef in ent_coefs:
            for seed in seeds:
                futures.append(pool.submit(
                    _run_one, seed, args.timesteps, args.max_steps,
                    args.eval_episodes, ent_coef, str(output_root),
                ))
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if res["status"] == "ok":
                n_ok += 1
                logger.info("[%d/%d] OK ent=%.2f seed=%d in %.1fs", i, total_runs,
                            res.get("ent_coef", 0), res["seed"], res["wall_s"])
            elif res["status"] == "skipped":
                n_skip += 1
            else:
                n_fail += 1
                logger.error("[%d/%d] FAIL seed=%d: %s", i, total_runs,
                             res["seed"], res.get("error", ""))

    logger.info("Done. ok=%d skipped=%d failed=%d total_wall=%.1fs",
                n_ok, n_skip, n_fail, time.time() - started_all)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
