# =============================================================================
# THREAD LIMITING - MUST BE SET BEFORE ANY IMPORTS
# =============================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
# =============================================================================

"""Network capacity sensitivity analysis for the Coopetition-Gym v1 benchmark.

Runs a subset of the training algorithms at multiple network capacities to
verify that the paper's findings are not artifacts of the baseline
``[128, 128]`` architecture used in the main campaign.

Algorithm matrix (from :data:`SENSITIVITY_ALGORITHMS`):
    ISAC, MADDPG, MAPPO, COMA, QMIX. Hyperparameters are copied exactly from
    the main-campaign specs in :mod:`experiments.campaign`; only ``net_arch``
    is varied per experiment.

Available network capacities (from :data:`experiments.config.SENSITIVITY_NET_SIZES`):
    ``[64, 64]``, ``[128, 128]``, ``[256, 256]``, ``[512, 512]``,
    ``[1024, 1024]``. The ``[128, 128]`` baseline is typically skipped
    because the main campaign already covers that point. The default runs
    only ``[64, 64]``; broader sweeps require explicit ``--net-sizes``.

Design:

* Wraps :func:`experiments.campaign.run_single_experiment` so algorithm
  execution is byte-identical to the main campaign.
* Manages its own experiment matrix with ``net_arch`` as an additional axis.
* Injects ``net_arch`` into the algorithm's ``params`` dict before dispatch.
* Output filenames include a ``net{W}x{W}`` tag:
  ``{algo}_{env}_{seed}_net{W}x{W}.json``.
* Resume-aware: scans existing result files on startup and skips completed
  experiments.

Usage::

    # Default architecture with one worker
    python -m experiments.sensitivity --max-gpu-workers 1 \\
        --output data/training/network_sensitivity/

    # Subset for distributed execution
    python -m experiments.sensitivity --algorithms MADDPG --max-gpu-workers 1 ...
    python -m experiments.sensitivity --algorithms ISAC,MAPPO --max-gpu-workers 1 ...

Also accessible via the unified campaign CLI::

    python -m experiments.campaign sensitivity --max-gpu-workers 1 \\
        --output data/training/network_sensitivity/
"""

import sys
from copy import deepcopy
import json
import time
import math
import random
import logging
import argparse
import traceback
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

# =============================================================================
# CONFIGURATION
# =============================================================================

from .config import ENVIRONMENT_BY_ID, TRAINING_SEEDS, SENSITIVITY_NET_SIZES
from .campaign import (
    TRAINING_ALGORITHMS, TIMESTEPS_BY_CATEGORY, CAMPAIGN_SCHEMA_VERSION,
    BUDGET_VERSION, REWARD_VERSION, REWARD_TYPES, NumpyEncoder,
    _identity_hash, _source_version, _source_fingerprint, _campaign_result_complete, _json_safe,
)

# A small architecture and one worker are the defaults; larger sweeps require
# an explicit --net-sizes selection after hardware sizing.
DEFAULT_NET_SIZES = [[64, 64]]
AVAILABLE_NET_SIZES = [list(size) for size in SENSITIVITY_NET_SIZES]
BASELINE_NET_SIZE = [128, 128]

# Preserve the campaign's algorithm parameters programmatically. Network width
# is the only scientific algorithm setting changed by this module.
SENSITIVITY_ALGORITHMS = {
    item["name"]: deepcopy(item) for item in TRAINING_ALGORITHMS
    if item["name"] in {"ISAC", "MADDPG", "MAPPO", "COMA", "QMIX"}
}
SENSITIVITY_ENVIRONMENTS = [
    asdict(ENVIRONMENT_BY_ID[name]) for name in (
        "TrustDilemma-v0", "LoyaltyTeam-v0", "ApacheProject-v0", "RecoveryRace-v0",
        "GraduatedSanction-v0", "SLCD-v0", "PartnerHoldUp-v0", "ReciprocalDilemma-v0",
    )
]


def ensure_campaign_identity(output_dir, n_eval_episodes, timesteps_override=None):
    """Prevent incompatible sensitivity configurations sharing an output tree."""
    identity = {
        "kind": "network-sensitivity", "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "source_version": _source_version(), "source_fingerprint": _source_fingerprint(),
        "reward_version": REWARD_VERSION,
        "budget_version": BUDGET_VERSION, "timesteps_by_category": TIMESTEPS_BY_CATEGORY,
        "timesteps_override": timesteps_override, "n_eval_episodes": n_eval_episodes,
        "algorithms": SENSITIVITY_ALGORITHMS, "environments": SENSITIVITY_ENVIRONMENTS,
    }
    campaign_id = _identity_hash(identity)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sensitivity.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if (existing.get("campaign_id") != campaign_id
                or _identity_hash(existing.get("configuration")) != campaign_id):
            raise ValueError("Output directory belongs to a different sensitivity configuration; use a new directory")
        return campaign_id
    if any(output_dir.glob("raw/*.json")) or any(output_dir.glob("*/raw/*.json")):
        raise ValueError("Existing sensitivity results have no manifest; use a new output directory")
    with path.open("x") as handle:
        json.dump({"campaign_id": campaign_id, "configuration": identity}, handle,
                  sort_keys=True, allow_nan=False, cls=NumpyEncoder)
    return campaign_id


def _sensitivity_result_complete(record, campaign_id, n_eval_episodes, timesteps_override=None):
    if not campaign_id or n_eval_episodes is None:
        return False
    if not _campaign_result_complete(record, campaign_id) or record.get("requires_training") is not True:
        return False
    net_arch = record.get("net_arch")
    if (not isinstance(net_arch, list) or not net_arch
            or any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in net_arch)):
        return False
    if record.get("net_arch_tag") != net_arch_tag(net_arch):
        return False
    if record.get("evaluation_episodes_requested") != n_eval_episodes:
        return False
    spec = next((env for env in SENSITIVITY_ENVIRONMENTS if env["id"] == record["environment"]), None)
    algorithm = SENSITIVITY_ALGORITHMS.get(record["algorithm"])
    if spec is None or algorithm is None:
        return False
    expected_algo = deepcopy(algorithm)
    expected_algo["params"]["net_arch"] = net_arch
    expected_algo["gpu_memory_gb"] = estimate_vram_gb(net_arch, record["algorithm"], spec["n_agents"])
    if record.get("algorithm_config") != expected_algo:
        return False
    expected_env = dict(spec, reward_type=record["reward_type"])
    if record.get("environment_config") != expected_env:
        return False
    budget = (timesteps_override if timesteps_override is not None else
              TIMESTEPS_BY_CATEGORY.get(spec.get("category", "dyadic"), 500000))
    return record.get("training_steps_requested") == budget


# =============================================================================
# VRAM ESTIMATION
# =============================================================================

def estimate_vram_gb(net_arch: List[int], algo_name: str, n_agents: int) -> float:
    """Estimate VRAM usage for a given network size configuration."""
    # Base VRAM from algorithm overhead (buffers, optimizer states)
    base = {"ISAC": 1.0, "MADDPG": 1.5, "MAPPO": 1.0}.get(algo_name, 1.5)
    # Network parameter scaling: roughly proportional to sum of W*W products
    param_factor = sum(w * w for w in net_arch) / (128 * 128)  # Relative to baseline
    # Agent scaling for MADDPG (centralized critic sees all agents)
    agent_factor = n_agents if algo_name == "MADDPG" else 1.0
    return base + 0.5 * param_factor * (1.0 + 0.3 * agent_factor)


# =============================================================================
# EXPERIMENT MATRIX
# =============================================================================

def net_arch_tag(net_arch: List[int]) -> str:
    """Create filename-safe tag for a net_arch, e.g., 'net64x64'."""
    return "net" + "x".join(str(w) for w in net_arch)


def build_experiment_matrix(
    algorithms: Optional[List[str]],
    environments: Optional[List[str]],
    seeds: List[int],
    net_sizes: List[List[int]],
    reward_types: List[str],
    completed_keys: Set[str],
) -> List[Dict[str, Any]]:
    """Build the sensitivity experiment matrix."""
    if algorithms and set(algorithms) - set(SENSITIVITY_ALGORITHMS):
        raise ValueError("Unknown sensitivity algorithm")
    known_envs = {env["id"] for env in SENSITIVITY_ENVIRONMENTS}
    if environments and set(environments) - known_envs:
        raise ValueError("Unknown sensitivity environment")
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if not reward_types or set(reward_types) - set(REWARD_TYPES):
        raise ValueError("Unknown sensitivity reward type")
    if not net_sizes or any(not size or any(not isinstance(width, int) or isinstance(width, bool) or width <= 0
                                            for width in size) for size in net_sizes):
        raise ValueError("Network widths must be positive integers")
    if (len(set(seeds)) != len(seeds) or len(set(reward_types)) != len(reward_types)
            or len({tuple(size) for size in net_sizes}) != len(net_sizes)):
        raise ValueError("Sensitivity axes must be distinct to prevent duplicate concurrent cells")
    experiments = []

    algos = SENSITIVITY_ALGORITHMS
    if algorithms:
        algos = {k: v for k, v in algos.items() if k in algorithms}

    envs = SENSITIVITY_ENVIRONMENTS
    if environments:
        envs = [e for e in envs if e["id"] in environments]

    for algo_name, algo_config in algos.items():
        for env_config in envs:
            for net_arch in net_sizes:
                for reward_type in reward_types:
                    for seed in seeds:
                        tag = net_arch_tag(net_arch)
                        key = f"{algo_name}_{env_config['id']}_{seed}_{tag}_{reward_type}"

                        if key in completed_keys:
                            continue

                        # Deep copy algo config and inject net_arch
                        ac = {k: (v.copy() if isinstance(v, dict) else v)
                              for k, v in algo_config.items()}
                        ac["params"] = algo_config["params"].copy()
                        ac["params"]["net_arch"] = list(net_arch)

                        # Adjust VRAM estimate for larger networks
                        ac["gpu_memory_gb"] = estimate_vram_gb(
                            net_arch, algo_name, env_config["n_agents"]
                        )

                        experiments.append({
                            "algo_config": ac,
                            "env_config": env_config,
                            "seed": seed,
                            "net_arch": list(net_arch),
                            "reward_type": reward_type,
                            "key": key,
                        })

    # Sort: slow algorithms first (MADDPG), then by env size (ApacheProject),
    # then by net size (1024 first) — so heavy experiments start immediately
    speed_order = {"slow": 0, "medium": 1, "fast": 2}
    experiments.sort(key=lambda e: (
        speed_order.get(e["algo_config"].get("speed", "medium"), 1),
        -e["env_config"]["n_agents"],
        -max(e["net_arch"]),
    ))

    return experiments


# =============================================================================
# RESULT SCANNING
# =============================================================================

def scan_completed(raw_dir: Path, campaign_id=None, n_eval_episodes=None,
                   timesteps_override=None) -> Set[str]:
    """Resume only finite, measured results matching the requested configuration."""
    completed = set()
    if not raw_dir.exists() or not campaign_id or n_eval_episodes is None:
        return completed
    for filepath in raw_dir.glob("*.json"):
        try:
            data = json.loads(filepath.read_text())
            if _sensitivity_result_complete(data, campaign_id, n_eval_episodes, timesteps_override):
                key = f"{data['algorithm']}_{data['environment']}_{data['training_seed']}_{data['net_arch_tag']}_{data['reward_type']}"
                if filepath.stem == key:
                    completed.add(key)
        except (ValueError, KeyError, OSError, TypeError):
            continue
    return completed


# =============================================================================
# EXPERIMENT RUNNER WRAPPER
# =============================================================================

def run_sensitivity_experiment(
    algo_config: Dict[str, Any],
    env_config: Dict[str, Any],
    seed: int,
    net_arch: List[int],
    reward_type: str,
    n_eval_episodes: int,
    gpu_id: int,
    enable_gpu_isolation: bool,
    checkpoint_dir: Optional[Path],
    checkpoint_interval: int,
    log_file: Optional[str],
    progress_dir: Optional[Path],
    raw_dir: str,
    campaign_id: str = "",
    timesteps_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single sensitivity experiment, wrapping run_single_experiment."""
    from .campaign import run_single_experiment

    tag = net_arch_tag(net_arch)
    algo_name = algo_config["name"]
    env_id = env_config["id"]

    # Call the existing experiment runner
    try:
        result = run_single_experiment(
            algo_config=algo_config,
            env_config=env_config,
            training_seed=seed,
            reward_type=reward_type,
            campaign_id=campaign_id,
            timesteps_override=timesteps_override,
            n_eval_episodes=n_eval_episodes,
            gpu_id=gpu_id,
            enable_gpu_isolation=enable_gpu_isolation,
            reduced_buffer_level=0,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            log_file=log_file,
            progress_dir=progress_dir,
        )
    except Exception as e:
        return {
            "key": f"{algo_name}_{env_id}_{seed}_{tag}_{reward_type}",
            "status": "failed",
            "filename": f"{algo_name}_{env_id}_{seed}_{tag}_{reward_type}.json",
            "training_time": 0,
            "mean_return": None,
            "error": str(e)[:200],
        }

    if result is None:
        return {
            "key": f"{algo_name}_{env_id}_{seed}_{tag}_{reward_type}",
            "status": "failed",
            "filename": f"{algo_name}_{env_id}_{seed}_{tag}_{reward_type}.json",
            "training_time": 0,
            "mean_return": None,
            "error": "run_single_experiment returned None",
        }

    # Convert result to dict and add sensitivity metadata
    result_dict = result.to_dict()
    result_dict["net_arch"] = net_arch
    result_dict["net_arch_tag"] = tag
    valid = _sensitivity_result_complete(result_dict, campaign_id or result_dict.get("campaign_id"),
                                         n_eval_episodes, timesteps_override)
    if result_dict.get("reward_type") != reward_type:
        valid = False
    if result_dict.get("status") == "success" and not valid:
        result_dict["status"] = "failed"
        result_dict["error_message"] = "Sensitivity result failed configuration or completion validation"

    # Save with sensitivity-aware filename
    filename = f"{algo_name}_{env_id}_{seed}_{tag}_{reward_type}.json"
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    filepath = raw_path / filename

    tmp = filepath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_json_safe(result_dict), separators=(",", ":"),
                              cls=NumpyEncoder, allow_nan=False))
    tmp.replace(filepath)

    return {
        "key": f"{algo_name}_{env_id}_{seed}_{tag}_{reward_type}",
        "status": result_dict["status"],
        "filename": filename,
        "training_time": result.training_time_seconds,
        "mean_return": (result_dict.get("metrics") or {}).get("mean_return"),
    }


# =============================================================================
# GPU ALLOCATION
# =============================================================================

def detect_gpus() -> int:
    """Detect available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
            return n
    except ImportError:
        pass
    return 0


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Network Size Sensitivity Analysis — Phase 4-NET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full sweep on 16× RTX 4090
    python -m experiments.sensitivity --max-gpu-workers 80 --output results/sensitivity

    # MADDPG only (Instance 1)
    python -m experiments.sensitivity --algorithms MADDPG --max-gpu-workers 1

    # ISAC + MAPPO (Instance 2)
    python -m experiments.sensitivity --algorithms ISAC,MAPPO --max-gpu-workers 1

    # Dry run
    python -m experiments.sensitivity --dry-run
        """
    )

    parser.add_argument("--output", type=str, default="results_sensitivity",
                        help="Output directory")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated algorithms (default: ISAC,MADDPG,MAPPO,COMA,QMIX)")
    parser.add_argument("--environments", type=str, default=None,
                        help="Comma-separated environments")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, TRAINING_SEEDS)),
                        help="Comma-separated seeds")
    parser.add_argument("--net-sizes", type=str, default=None,
                        help="Explicit space-separated architectures (default: 64,64); size larger sweeps for the actual hardware")
    parser.add_argument("--reward-types", type=str, default="integrated,private",
                        help="Comma-separated reward types")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes")
    parser.add_argument("--max-gpu-workers", type=int, default=1,
                        help="Max concurrent experiments (default: 1); size overrides for the actual hardware")
    parser.add_argument("--resume", action="store_true",
                        help="Compatibility flag; valid matching results are always skipped")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show experiment matrix without running")
    parser.add_argument("--enable-checkpoints", action="store_true",
                        help="Enable training checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=100000,
                        help="Steps between checkpoints")

    parser.add_argument("--timesteps-override", type=int, default=None,
                        help="Explicit reduced training budget per learner")
    args = parser.parse_args(argv)
    if args.eval_episodes < 1 or args.max_gpu_workers < 1 or args.checkpoint_interval < 1:
        parser.error("evaluation episodes, workers, and checkpoint interval must be positive")
    if args.timesteps_override is not None and args.timesteps_override < 1:
        parser.error("--timesteps-override must be positive")

    # Parse arguments
    algorithms = args.algorithms.split(",") if args.algorithms else None
    environments = args.environments.split(",") if args.environments else None
    seeds = [int(s) for s in args.seeds.split(",")]
    reward_types = [r.strip() for r in args.reward_types.split(",")]

    if args.net_sizes:
        net_sizes = [[int(x) for x in spec.split(",")]
                     for spec in args.net_sizes.split()]
    else:
        net_sizes = DEFAULT_NET_SIZES

    # Validate the full selection before creating output or dispatching workers.
    build_experiment_matrix(algorithms, environments, seeds, net_sizes, reward_types, set())
    output_dir = Path(args.output)
    campaign_id = ensure_campaign_identity(output_dir, args.eval_episodes, args.timesteps_override)
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    progress_dir = output_dir / "progress"

    for d in [raw_dir, logs_dir, progress_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "sensitivity.log"),
        ]
    )
    logger = logging.getLogger("sensitivity")

    logger.info("=" * 70)
    logger.info("NETWORK SIZE SENSITIVITY ANALYSIS — Phase 4-NET")
    logger.info("=" * 70)
    logger.info(f"Algorithms: {algorithms or list(SENSITIVITY_ALGORITHMS.keys())}")
    logger.info(f"Environments: {[e['id'] for e in SENSITIVITY_ENVIRONMENTS]}")
    logger.info(f"Net sizes: {net_sizes}")
    logger.info(f"Reward types: {reward_types}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {output_dir}")

    # Detect hardware
    logger.info("\nHardware:")
    num_gpus = detect_gpus()
    logger.info(f"  GPUs: {num_gpus}")
    logger.info(f"  CPUs: {mp.cpu_count()}")
    logger.info(f"  Max GPU workers: {args.max_gpu_workers}")

    # Revalidate on-disk evidence regardless of the compatibility --resume flag.
    completed_keys = set()
    for rt in reward_types:
        completed_keys.update(scan_completed(output_dir / rt / "raw", campaign_id,
                                             args.eval_episodes, args.timesteps_override))
    completed_keys.update(scan_completed(raw_dir, campaign_id, args.eval_episodes,
                                         args.timesteps_override))
    logger.info(f"  Resumed: {len(completed_keys)} validated experiments found")

    # Build experiment matrix
    experiments = build_experiment_matrix(
        algorithms=algorithms,
        environments=environments,
        seeds=seeds,
        net_sizes=net_sizes,
        reward_types=reward_types,
        completed_keys=completed_keys,
    )

    logger.info(f"\nExperiment matrix: {len(experiments)} experiments to run")

    # Summary by algorithm × environment × net_size
    summary = defaultdict(int)
    for exp in experiments:
        algo = exp["algo_config"]["name"]
        env = exp["env_config"]["id"]
        tag = net_arch_tag(exp["net_arch"])
        summary[(algo, env, tag)] += 1

    for (algo, env, tag), count in sorted(summary.items()):
        logger.info(f"  {algo:10s} × {env:25s} × {tag:12s}: {count} experiments")

    if args.dry_run:
        logger.info("\nDRY RUN — would run the above experiments. Exiting.")
        return

    if not experiments:
        logger.info("No experiments to run (all completed or empty matrix).")
        return

    # GPU round-robin allocation
    gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [-1]
    gpu_cycle = 0

    completed = 0
    failed = 0
    start_time = time.time()
    spawn_ctx = mp.get_context('spawn')

    logger.info(f"\nStarting {len(experiments)} experiments with {args.max_gpu_workers} workers...")

    with ProcessPoolExecutor(
        max_workers=args.max_gpu_workers,
        mp_context=spawn_ctx,
    ) as executor:
        futures = {}

        for exp in experiments:
            # Round-robin GPU assignment
            gpu_id = -1 if exp["algo_config"].get("cpu_only") else gpu_ids[gpu_cycle % len(gpu_ids)]
            gpu_cycle += 1

            # Determine output subdirectory by reward type
            rt = exp["reward_type"]
            exp_raw_dir = str(output_dir / rt / "raw")

            checkpoint_dir = None
            if args.enable_checkpoints:
                cd = args.checkpoint_dir or str(output_dir / "checkpoints")
                checkpoint_dir = Path(cd) / rt / net_arch_tag(exp["net_arch"])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            future = executor.submit(
                run_sensitivity_experiment,
                algo_config=exp["algo_config"],
                env_config=exp["env_config"],
                seed=exp["seed"],
                net_arch=exp["net_arch"],
                reward_type=exp["reward_type"],
                n_eval_episodes=args.eval_episodes,
                gpu_id=gpu_id,
                enable_gpu_isolation=True,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=args.checkpoint_interval,
                log_file=str(logs_dir / "workers.log"),
                progress_dir=progress_dir,
                raw_dir=exp_raw_dir,
                campaign_id=campaign_id,
                timesteps_override=args.timesteps_override,
            )
            futures[future] = exp["key"]

        # Collect results
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                if result["status"] == "success":
                    completed += 1
                    elapsed = time.time() - start_time
                    rate = completed / (elapsed / 3600) if elapsed > 0 else 0
                    logger.info(
                        f"  [{completed + failed}/{len(experiments)}] "
                        f"{result['filename']}: "
                        f"return={result['mean_return']:.1f} "
                        f"time={result['training_time']:.0f}s "
                        f"({rate:.1f}/hr)"
                    )
                else:
                    failed += 1
                    logger.warning(f"  FAILED: {key}")
            except Exception as e:
                failed += 1
                logger.error(f"  ERROR: {key}: {str(e)[:200]}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"NETWORK SENSITIVITY ANALYSIS COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total time: {elapsed/3600:.1f} hours")
    logger.info(f"  Output: {output_dir}")

    # Print per-reward-type file counts
    for rt in reward_types:
        rt_raw = output_dir / rt / "raw"
        if rt_raw.exists():
            count = len(list(rt_raw.glob("*.json")))
            logger.info(f"  {rt}: {count} result files")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()