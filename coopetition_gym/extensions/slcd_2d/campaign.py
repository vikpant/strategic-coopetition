"""Stand-alone smoke/sanity campaign for the 2D SLCD extension.

Runs a small matrix of (algorithm, seed) on `SLCDAppropriation-v1ext0`. This is
deliberately *not* wired into ``experiments/campaign.py`` so that the v1
reproducibility release stays unchanged.

Usage
-----
    python -m extensions.slcd_2d.campaign --seeds 106,107,108 --steps 40 \
        --output .claude/experiments/slcd_2d/smoke/

Currently supports only the oracle baseline (no training). Training-algorithm
support is a follow-up once the environment is validated.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

from . import AppropriationOracle, SLCDAppropriationEnv


def run_oracle_episode(seed: int, steps: int) -> dict:
    env = SLCDAppropriationEnv(max_steps=steps)
    oracle = AppropriationOracle(env)
    action = oracle._action

    obs, info = env.reset(seed=seed)
    returns = np.zeros(env.n_agents, dtype=np.float64)
    trust_trajectory: List[float] = [float(info["mean_trust"])]
    appr_trajectory: List[list] = []

    for t in range(steps):
        obs, r, term, trunc, info = env.step(action)
        returns += r
        trust_trajectory.append(float(info["mean_trust"]))
        appr_trajectory.append([float(x) for x in info["appropriation"]])
        if term or trunc:
            break

    return {
        "seed": seed,
        "algorithm": "Oracle_Appropriation",
        "env": "SLCDAppropriation-v1ext0",
        "episode_return": returns.tolist(),
        "equilibrium_cooperation": oracle.equilibrium.cooperation.tolist(),
        "equilibrium_appropriation": oracle.equilibrium.appropriation.tolist(),
        "equilibrium_utilities": oracle.equilibrium.utilities.tolist(),
        "equilibrium_iterations": oracle.equilibrium.iterations,
        "equilibrium_converged": oracle.equilibrium.converged,
        "trust_trajectory": trust_trajectory,
        "appropriation_trajectory": appr_trajectory,
        "calibration": asdict(oracle.appr_params),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2D SLCD sanity campaign")
    parser.add_argument(
        "--seeds", type=str, default="106,107,108",
        help="Comma-separated seeds (default: 106,107,108)",
    )
    parser.add_argument(
        "--steps", type=int, default=40,
        help="Episode length in steps (default: 40 to match v1 SLCD horizon)",
    )
    parser.add_argument(
        "--output", type=str, default=".claude/experiments/slcd_2d/smoke/",
        help="Output directory for per-seed JSON files",
    )
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        result = run_oracle_episode(seed=seed, steps=args.steps)
        out_file = out_dir / f"oracle_seed{seed}.json"
        with out_file.open("w") as fh:
            json.dump(result, fh, indent=2)
        print(
            f"[seed={seed}] return={np.array(result['episode_return']).round(2)} "
            f"c*={np.array(result['equilibrium_cooperation']).round(2)} "
            f"p*={np.array(result['equilibrium_appropriation']).round(3)} "
            f"final_trust={result['trust_trajectory'][-1]:.3f} -> {out_file}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
