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
    the main-campaign specs in :mod:`experiments.config`; only ``net_arch``
    is varied per experiment.

Network capacities (from :data:`experiments.config.SENSITIVITY_NET_SIZES`):
    ``[64, 64]``, ``[128, 128]``, ``[256, 256]``, ``[512, 512]``,
    ``[1024, 1024]``. The ``[128, 128]`` baseline is typically skipped
    because the main campaign already covers that point.

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

    # Full sweep
    python -m experiments.sensitivity --max-gpu-workers 40 \\
        --output data/training/network_sensitivity/

    # Subset for distributed execution
    python -m experiments.sensitivity --algorithms MADDPG --max-gpu-workers 40 ...
    python -m experiments.sensitivity --algorithms ISAC,MAPPO --max-gpu-workers 40 ...

Also accessible via the unified campaign CLI::

    python -m experiments.campaign sensitivity --max-gpu-workers 40 \\
        --output data/training/network_sensitivity/
"""

import sys
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

# Prepend the parent of the inner ``coopetition_gym`` package so the import
# machinery resolves the installed editable package instead of the outer
# namespace-package directory. This mirrors the fix applied in
# ``experiments.audit`` and ``experiments.algorithms``.
_THIS_DIR = Path(__file__).resolve().parent       # .../experiments
_PROJECT_ROOT = _THIS_DIR.parent                   # repository root
_GYM_PATH = str(_PROJECT_ROOT / "coopetition_gym")
if _GYM_PATH not in sys.path:
    sys.path.insert(0, _GYM_PATH)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Network sizes to test (powers of two from 64 to 1024)
# [128,128] is the baseline — exists in main campaign data, not re-run here
DEFAULT_NET_SIZES = [[64, 64], [256, 256], [512, 512], [1024, 1024]]
BASELINE_NET_SIZE = [128, 128]  # Reference only — not run

# =============================================================================
# ALGORITHM CONFIGURATIONS — MUST MATCH orchestrator.py EXACTLY
# =============================================================================
# CRITICAL: These params are copied PROGRAMMATICALLY from orchestrator.py
# TRAINING_ALGORITHMS definitions. The ONLY parameter that changes across
# sensitivity experiments is net_arch. All other params MUST be identical
# to Campaign 1 baseline to enable valid cross-campaign comparison.
#
# Source: orchestrator.py lines 413-515 (TRAINING_ALGORITHMS)
# Verified: 2026-04-11 via programmatic extraction
#
# RULE: NEVER hand-type algorithm params. Always extract from orchestrator.py
# and verify before launching. See feedback_sensitivity_params.md in memory.
# =============================================================================
SENSITIVITY_ALGORITHMS = {
    "ISAC": {
        "name": "ISAC", "class": "IndependentSAC",
        "requires_training": True, "gpu_memory_gb": 3.0, "cpu_only": False,
        "speed": "medium",
        "params": {
            # EXACT match: orchestrator.py ISAC params
            "learning_rate": 3e-4, "buffer_size": 100000, "batch_size": 256,
            "tau": 0.005, "gamma": 0.99,
            "net_arch": [128, 128],  # Will be overridden per experiment
        },
    },
    "MADDPG": {
        "name": "MADDPG", "class": "MADDPG",
        "requires_training": True, "gpu_memory_gb": 4.0, "cpu_only": False,
        "speed": "slow",
        "params": {
            # EXACT match: orchestrator.py MADDPG params
            "learning_rate_actor": 1e-4, "learning_rate_critic": 1e-3,
            "buffer_size": 100000, "batch_size": 256,
            "tau": 0.005, "gamma": 0.99,
            "net_arch": [128, 128],  # Will be overridden per experiment
        },
    },
    "MAPPO": {
        "name": "MAPPO", "class": "MAPPO",
        "requires_training": True, "gpu_memory_gb": 0.0, "cpu_only": True,
        "speed": "medium",
        "params": {
            # EXACT match: orchestrator.py MAPPO params
            # NOTE: MAPPO is cpu_only=True in orchestrator — runs on CPU, not GPU
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
            "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": 0.01, "share_critic": True,
            "net_arch": [128, 128],  # Will be overridden per experiment
        },
    },
    "COMA": {
        "name": "COMA", "class": "COMA",
        "requires_training": True, "gpu_memory_gb": 1.5, "cpu_only": False,
        "speed": "fast",
        "params": {
            # EXACT match: orchestrator.py COMA params
            # NOTE: orchestrator only sets learning_rate and gamma for COMA
            # All other params use algorithm class defaults in algorithms.py
            "learning_rate": 5e-4, "gamma": 0.99,
            "net_arch": [128, 128],  # Will be overridden per experiment
        },
    },
    "QMIX": {
        "name": "QMIX", "class": "QMIX",
        "requires_training": True, "gpu_memory_gb": 2.5, "cpu_only": False,
        "speed": "medium",
        "params": {
            # EXACT match: orchestrator.py QMIX params
            "learning_rate": 5e-4, "buffer_size": 5000, "batch_size": 32,
            "gamma": 0.99, "action_bins": 11,
            "net_arch": [128, 128],  # Will be overridden per experiment
        },
    },
}

# Environments: coverage across all 4 TR tiers and agent counts
# Set A (original): TR-1 + TR-3, agent counts 2/3/7
# Set B (extended): TR-2 + TR-4 + case study, agent counts 2/4/6
SENSITIVITY_ENVIRONMENTS = [
    # --- Set A: Original 3 (Italy + France instances) ---
    {"id": "TrustDilemma-v0", "horizon": 100, "category": "dyadic",
     "n_agents": 2, "tr": "tr1"},
    {"id": "LoyaltyTeam-v0", "horizon": 100, "category": "collective_action",
     "n_agents": 3, "tr": "tr3"},
    {"id": "ApacheProject-v0", "horizon": 100, "category": "collective_action",
     "n_agents": 7, "tr": "tr3"},
    # --- Set B: Extended (California instance) ---
    {"id": "RecoveryRace-v0", "horizon": 100, "category": "benchmark",
     "n_agents": 2, "tr": "tr2"},
    {"id": "GraduatedSanction-v0", "horizon": 100, "category": "reciprocity",
     "n_agents": 6, "tr": "tr4"},
    {"id": "SLCD-v0", "horizon": 100, "category": "dyadic",
     "n_agents": 2, "tr": "tr2"},
    # --- Set C: Symmetric coverage (2 per TR) ---
    # EXACT match: orchestrator.py environment configs
    {"id": "PartnerHoldUp-v0", "horizon": 100, "category": "dyadic",
     "n_agents": 2, "tr": "tr1"},
    {"id": "ReciprocalDilemma-v0", "horizon": 100, "category": "dyadic",
     "n_agents": 2, "tr": "tr4"},
]

REWARD_TYPES = ["integrated", "private"]


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

def scan_completed(raw_dir: Path) -> Set[str]:
    """Scan for already-completed sensitivity experiments."""
    completed = set()
    if not raw_dir.exists():
        return completed

    for filepath in raw_dir.glob("*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
            if data.get("status") == "success":
                algo = data["algorithm"]
                env = data["environment"]
                seed = data["training_seed"]
                tag = data.get("net_arch_tag", "")
                rt = data.get("reward_type", "integrated")
                if tag:
                    key = f"{algo}_{env}_{seed}_{tag}_{rt}"
                    completed.add(key)
        except (json.JSONDecodeError, KeyError, OSError):
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
) -> Dict[str, Any]:
    """Run a single sensitivity experiment, wrapping run_single_experiment."""
    # Make sure the repository root (which contains the ``experiments``
    # package) is on ``sys.path`` so this worker subprocess can import it.
    _experiments_dir = str(Path(__file__).resolve().parent)
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    # And prepend the inner coopetition_gym parent to beat the namespace-package
    # shadowing (see experiments.audit._import_coopetition_gym).
    _gym_parent = str(Path(_repo_root) / "coopetition_gym")
    if _gym_parent not in sys.path:
        sys.path.insert(0, _gym_parent)

    # Set reward type environment variable before any env creation.
    os.environ['COOPETITION_REWARD_TYPE'] = reward_type

    from experiments.campaign import run_single_experiment

    tag = net_arch_tag(net_arch)
    algo_name = algo_config["name"]
    env_id = env_config["id"]

    # Call the existing experiment runner
    try:
        result = run_single_experiment(
            algo_config=algo_config,
            env_config=env_config,
            training_seed=seed,
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
            "filename": f"{algo_name}_{env_id}_{seed}_{tag}.json",
            "training_time": 0,
            "mean_return": None,
            "error": str(e)[:200],
        }

    if result is None:
        return {
            "key": f"{algo_name}_{env_id}_{seed}_{tag}_{reward_type}",
            "status": "failed",
            "filename": f"{algo_name}_{env_id}_{seed}_{tag}.json",
            "training_time": 0,
            "mean_return": None,
            "error": "run_single_experiment returned None",
        }

    # Convert result to dict and add sensitivity metadata
    result_dict = result.to_dict()
    result_dict["net_arch"] = net_arch
    result_dict["net_arch_tag"] = tag
    result_dict["reward_type"] = reward_type

    # Save with sensitivity-aware filename
    filename = f"{algo_name}_{env_id}_{seed}_{tag}.json"
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    filepath = raw_path / filename

    # Use custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                if math.isnan(obj) or math.isinf(obj):
                    return str(obj)
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(filepath, 'w') as f:
        json.dump(result_dict, f, separators=(',', ':'), cls=NumpyEncoder)

    return {
        "key": f"{algo_name}_{env_id}_{seed}_{tag}_{reward_type}",
        "status": result.status,
        "filename": filename,
        "training_time": result.training_time_seconds,
        "mean_return": result_dict.get("metrics", {}).get("mean_return"),
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

def main():
    parser = argparse.ArgumentParser(
        description="Network Size Sensitivity Analysis — Phase 4-NET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full sweep on 16× RTX 4090
    python run_network_sensitivity.py --max-gpu-workers 80 --output results/sensitivity

    # MADDPG only (Instance 1)
    python run_network_sensitivity.py --algorithms MADDPG --max-gpu-workers 40

    # ISAC + MAPPO (Instance 2)
    python run_network_sensitivity.py --algorithms ISAC,MAPPO --max-gpu-workers 40

    # Dry run
    python run_network_sensitivity.py --dry-run
        """
    )

    parser.add_argument("--output", type=str, default="results_sensitivity",
                        help="Output directory")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated algorithms (default: ISAC,MADDPG,MAPPO)")
    parser.add_argument("--environments", type=str, default=None,
                        help="Comma-separated environments")
    parser.add_argument("--seeds", type=str, default="99,100,101,102,103",
                        help="Comma-separated seeds")
    parser.add_argument("--net-sizes", type=str, default=None,
                        help="Space-separated net arch specs (e.g., '64,64 256,256 512,512 1024,1024')")
    parser.add_argument("--reward-types", type=str, default="integrated,private",
                        help="Comma-separated reward types")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes")
    parser.add_argument("--max-gpu-workers", type=int, default=40,
                        help="Max concurrent GPU experiments")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show experiment matrix without running")
    parser.add_argument("--enable-checkpoints", action="store_true",
                        help="Enable training checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=100000,
                        help="Steps between checkpoints")

    args = parser.parse_args()

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

    output_dir = Path(args.output)
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

    # Scan completed experiments
    completed_keys = set()
    if args.resume:
        # Scan per-reward-type subdirectories
        for rt in reward_types:
            rt_raw = output_dir / rt / "raw"
            completed_keys.update(scan_completed(rt_raw))
        # Also scan flat raw dir
        completed_keys.update(scan_completed(raw_dir))
        logger.info(f"  Resumed: {len(completed_keys)} completed experiments found")

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
            gpu_id = gpu_ids[gpu_cycle % len(gpu_ids)]
            gpu_cycle += 1

            # Determine output subdirectory by reward type
            rt = exp["reward_type"]
            exp_raw_dir = str(output_dir / rt / "raw")

            checkpoint_dir = None
            if args.enable_checkpoints:
                cd = args.checkpoint_dir or str(output_dir / "checkpoints")
                checkpoint_dir = Path(cd)
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