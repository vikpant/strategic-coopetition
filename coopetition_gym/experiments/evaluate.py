"""Policy evaluation and result aggregation.

This module consolidates ``evaluate.py`` (episode-level policy evaluation)
and ``aggregate_results.py`` (cross-seed aggregation of training results)
from the campaign source tree into one command-line tool with two
subcommands:

* ``agent`` — Evaluate a trained policy or heuristic by running ``n_episodes``
  episodes on a specified environment, recording per-episode returns, final
  trust, cooperation rate, and episode length. Writes one
  :class:`EvaluationResult` JSON per invocation.

* ``aggregate`` — Scan a directory of training result files and produce
  per-(algorithm, environment) summary statistics averaged across seeds.
  Writes a CSV summary and a ``summary.json`` with overall statistics.

The aggregation assumes the standard training-result schema
(see :func:`experiments.validate.print_schema`):

* Each file contains ``algorithm``, ``environment``, ``seed``, and a
  ``metrics`` subdict with ``mean_return``, ``mean_final_trust``,
  ``mean_cooperation_rate``.

Usage::

    # Evaluate a trained policy
    python -m experiments.evaluate agent \\
        --algorithm ISAC --environment TrustDilemma-v0 \\
        --seeds 99,100,101 --episodes 100 \\
        --output data/evaluation/isac_td.json

    # Aggregate training results into a CSV summary
    python -m experiments.evaluate aggregate \\
        --input-dir data/training/baseline_integrated/ \\
        --output-dir data/analysis/baseline_summary/
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from coopetition_gym.experiments import config


logger = logging.getLogger(__name__)


# =============================================================================
# Multiprocessing-safe coopetition_gym import
# =============================================================================

def _import_coopetition_gym():
    """Import ``coopetition_gym`` bypassing the outer-folder namespace shadow.

    See :func:`experiments.algorithms._import_coopetition_gym` for rationale.
    """
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inner_package_parent = os.path.join(repo_root, "coopetition_gym")

    sys.modules.pop("coopetition_gym", None)
    if inner_package_parent not in sys.path:
        sys.path.insert(0, inner_package_parent)
    return importlib.import_module("coopetition_gym")


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class EpisodeResult:
    """Result from a single evaluation episode."""

    seed: int
    episode_return: float
    final_trust: float
    cooperation_rate: float
    episode_length: int
    terminated_early: bool
    per_step_rewards: Optional[List[float]] = None
    per_step_trust: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Aggregated evaluation results across seeds.

    Contains per-episode entries and aggregate statistics (mean and std) for
    returns, trust, cooperation rate, and episode length. Also records the
    early-termination rate, useful for detecting policy collapse.
    """

    algorithm: str
    environment: str
    n_episodes: int
    seed_range: Tuple[int, int]

    mean_return: float
    std_return: float
    mean_final_trust: float
    std_final_trust: float
    mean_cooperation_rate: float
    std_cooperation_rate: float
    mean_episode_length: float
    early_termination_rate: float

    episodes: List[EpisodeResult]

    evaluation_time_seconds: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["episodes"] = [
            ep.to_dict() if hasattr(ep, "to_dict") else ep for ep in self.episodes
        ]
        return result


@dataclass
class AggregatedResult:
    """Per-(algorithm, environment) summary statistics across seeds."""

    algorithm: str
    environment: str
    n_seeds: int
    seeds: List[int]

    # Means across seeds
    mean_return: float
    std_return_across_seeds: float
    mean_final_trust: float
    mean_cooperation_rate: float
    mean_training_time_seconds: float

    # Means of per-seed stds (indicative of within-seed variability)
    mean_within_seed_std_return: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Episode-level evaluation
# =============================================================================

def _run_single_episode(
    agent,
    env_id: str,
    seed: int,
    deterministic: bool = True,
    record_trajectory: bool = False,
) -> EpisodeResult:
    """Run one evaluation episode against a freshly constructed environment."""
    coopetition_gym = _import_coopetition_gym()
    env = coopetition_gym.make(env_id)
    obs, info = env.reset(seed=seed)

    episode_return = 0.0
    steps = 0
    terminated = False
    truncated = False

    per_step_rewards = [] if record_trajectory else None
    per_step_trust = [] if record_trajectory else None
    actions_sum = 0.0
    action_count = 0

    while not (terminated or truncated):
        try:
            action = agent.predict(obs, deterministic=deterministic)
        except Exception as exc:
            logger.warning(f"Agent prediction failed at step {steps}: {exc}")
            action = env.action_space.sample()

        if not isinstance(action, np.ndarray):
            action = np.array(action)

        obs, reward, terminated, truncated, info = env.step(action)

        step_reward = float(np.sum(reward)) if isinstance(reward, np.ndarray) else float(reward)
        episode_return += step_reward
        steps += 1

        if hasattr(env.action_space, "high"):
            denom = float(np.mean(env.action_space.high)) or 1.0
        else:
            denom = 100.0
        actions_sum += float(np.mean(action)) / denom
        action_count += 1

        if record_trajectory:
            per_step_rewards.append(step_reward)
            per_step_trust.append(info.get("mean_trust", 0.0))

    env.close()

    final_trust = info.get("mean_trust", 0.0)
    cooperation_rate = actions_sum / max(action_count, 1)
    # "terminated_early" is true if the episode terminated before reaching the
    # nominal horizon. We approximate with steps < environment horizon; since
    # we don't have the horizon here, fall back to a common default.
    ep_spec = config.ENVIRONMENT_BY_ID.get(env_id)
    horizon = ep_spec.horizon if ep_spec else 100
    terminated_early = bool(terminated and steps < horizon)

    return EpisodeResult(
        seed=seed,
        episode_return=episode_return,
        final_trust=final_trust,
        cooperation_rate=cooperation_rate,
        episode_length=steps,
        terminated_early=terminated_early,
        per_step_rewards=per_step_rewards,
        per_step_trust=per_step_trust,
    )


def evaluate_agent(
    agent,
    env_id: str,
    n_episodes: int = 100,
    seed_start: int = 0,
    deterministic: bool = True,
    record_trajectories: bool = False,
    verbose: bool = False,
    algorithm_name: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate a trained agent on an environment.

    Runs ``n_episodes`` episodes with consecutive seeds starting at
    ``seed_start``. Failures in a single episode are logged and recorded as
    zero-return, early-terminated episodes so a transient failure does not
    abort the whole evaluation.

    Args:
        agent: Object with a ``predict(obs, deterministic)`` method.
        env_id: Environment identifier (e.g. ``"TrustDilemma-v0"``).
        n_episodes: Number of evaluation episodes.
        seed_start: Starting seed. Subsequent seeds are ``seed_start + i``.
        deterministic: Whether to use deterministic actions.
        record_trajectories: Whether to record per-step rewards and trust.
            Increases memory usage proportionally to episode length.
        verbose: Whether to log progress every 10 episodes.
        algorithm_name: Optional name for the ``algorithm`` field. Falls back
            to ``type(agent).__name__``.

    Returns:
        An :class:`EvaluationResult` containing per-episode entries and
        aggregate statistics.
    """
    start_time = time.time()
    episodes: List[EpisodeResult] = []

    for i in range(n_episodes):
        seed = seed_start + i
        try:
            episodes.append(_run_single_episode(
                agent=agent, env_id=env_id, seed=seed,
                deterministic=deterministic, record_trajectory=record_trajectories,
            ))
        except Exception as exc:
            logger.error(f"Episode {i} (seed={seed}) failed: {exc}")
            episodes.append(EpisodeResult(
                seed=seed, episode_return=0.0, final_trust=0.0,
                cooperation_rate=0.0, episode_length=0, terminated_early=True,
            ))
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{n_episodes} episodes")

    eval_time = time.time() - start_time
    returns = [ep.episode_return for ep in episodes]
    trusts = [ep.final_trust for ep in episodes]
    coop_rates = [ep.cooperation_rate for ep in episodes]
    lengths = [ep.episode_length for ep in episodes]
    early_terms = [ep.terminated_early for ep in episodes]

    return EvaluationResult(
        algorithm=algorithm_name or type(agent).__name__,
        environment=env_id,
        n_episodes=n_episodes,
        seed_range=(seed_start, seed_start + n_episodes - 1),
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_final_trust=float(np.mean(trusts)),
        std_final_trust=float(np.std(trusts)),
        mean_cooperation_rate=float(np.mean(coop_rates)),
        std_cooperation_rate=float(np.std(coop_rates)),
        mean_episode_length=float(np.mean(lengths)),
        early_termination_rate=float(np.mean(early_terms)),
        episodes=episodes,
        evaluation_time_seconds=eval_time,
        timestamp=datetime.now().isoformat(),
    )


def evaluate_heuristic(
    policy_fn: Callable[[np.ndarray, Any], np.ndarray],
    env_id: str,
    n_episodes: int = 100,
    seed_start: int = 0,
    policy_name: str = "Heuristic",
) -> EvaluationResult:
    """Evaluate a heuristic policy function.

    Wraps ``policy_fn`` in an object with a ``predict`` method so it can be
    passed to :func:`evaluate_agent`. The wrapper constructs a fresh env
    for environment-aware policies at each episode.
    """

    class _HeuristicWrapper:
        def __init__(self, fn, env_id_):
            self.fn = fn
            self.env_id = env_id_

        def predict(self, obs, deterministic: bool = True):
            coopetition_gym = _import_coopetition_gym()
            env = coopetition_gym.make(self.env_id)
            try:
                return self.fn(obs, env)
            finally:
                env.close()

    return evaluate_agent(
        agent=_HeuristicWrapper(policy_fn, env_id),
        env_id=env_id,
        n_episodes=n_episodes,
        seed_start=seed_start,
        deterministic=True,
        algorithm_name=policy_name,
    )


def write_evaluation_result(result: EvaluationResult, output_path: Path) -> None:
    """Write an :class:`EvaluationResult` to ``output_path`` as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


# =============================================================================
# Training-result aggregation
# =============================================================================

def load_training_results(input_dir: Path, status_filter: str = "success") -> List[Dict[str, Any]]:
    """Load training result JSON files from ``input_dir`` (non-recursive).

    Args:
        input_dir: Directory containing ``*.json`` files produced by the
            campaign orchestrator.
        status_filter: Only include results whose ``status`` matches. Pass
            ``None`` to include all.

    Returns:
        List of parsed dicts, with an extra ``_source_file`` entry recording
        the filename.
    """
    results: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse {path.name}: {exc}")
            continue
        except OSError as exc:
            logger.error(f"Error reading {path.name}: {exc}")
            continue

        if status_filter is not None and data.get("status") != status_filter:
            continue

        data["_source_file"] = path.name
        results.append(data)

    logger.info(f"Loaded {len(results)} results from {input_dir}")
    return results


def aggregate_by_algorithm_environment(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, AggregatedResult]]:
    """Aggregate training results by (environment, algorithm) across seeds.

    Returns a nested dict: ``{environment_id: {algorithm_name: AggregatedResult}}``.
    Only ``status == 'success'`` entries are considered; the caller should
    filter upstream with :func:`load_training_results`.
    """
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        env = r.get("environment")
        algo = r.get("algorithm")
        metrics = r.get("metrics") or {}
        if env is None or algo is None or not metrics:
            continue
        grouped[env][algo].append({
            "seed": r.get("training_seed", r.get("seed", 0)),
            "metrics": metrics,
            "training_time": r.get("training_time_seconds", 0.0),
        })

    aggregated: Dict[str, Dict[str, AggregatedResult]] = {}
    for env_id, algo_results in grouped.items():
        aggregated[env_id] = {}
        for algo_name, seed_rows in algo_results.items():
            if not seed_rows:
                continue
            mean_returns = [row["metrics"].get("mean_return", 0.0) for row in seed_rows]
            std_returns = [row["metrics"].get("std_return", 0.0) for row in seed_rows]
            trusts = [row["metrics"].get("mean_final_trust", 0.0) for row in seed_rows]
            coop = [row["metrics"].get("mean_cooperation_rate", 0.0) for row in seed_rows]
            times = [row["training_time"] for row in seed_rows]
            aggregated[env_id][algo_name] = AggregatedResult(
                algorithm=algo_name,
                environment=env_id,
                n_seeds=len(seed_rows),
                seeds=sorted(int(row["seed"]) for row in seed_rows),
                mean_return=float(np.mean(mean_returns)),
                std_return_across_seeds=float(np.std(mean_returns)),
                mean_final_trust=float(np.mean(trusts)),
                mean_cooperation_rate=float(np.mean(coop)),
                mean_training_time_seconds=float(np.mean(times)),
                mean_within_seed_std_return=float(np.mean(std_returns)),
            )
    return aggregated


def write_summary_csv(
    aggregated: Dict[str, Dict[str, AggregatedResult]],
    output_path: Path,
) -> None:
    """Write a flat CSV summarizing every (environment, algorithm) pair."""
    rows = []
    for env_id in sorted(aggregated.keys()):
        for algo_name in sorted(aggregated[env_id].keys()):
            agg = aggregated[env_id][algo_name]
            rows.append({
                "environment": env_id,
                "algorithm": algo_name,
                "n_seeds": agg.n_seeds,
                "mean_return": f"{agg.mean_return:.4f}",
                "std_return_across_seeds": f"{agg.std_return_across_seeds:.4f}",
                "mean_final_trust": f"{agg.mean_final_trust:.4f}",
                "mean_cooperation_rate": f"{agg.mean_cooperation_rate:.4f}",
                "mean_training_time_seconds": f"{agg.mean_training_time_seconds:.2f}",
                "mean_within_seed_std_return": f"{agg.mean_within_seed_std_return:.4f}",
            })

    if not rows:
        logger.warning("No rows to write — aggregation produced no entries.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary CSV written to {output_path}")


def write_overall_statistics(
    results: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a ``summary.json`` with corpus-level statistics.

    Reports total experiments, success/failure counts, success rate, total
    training time, and the unique algorithm and environment sets present in
    the corpus. Useful for verifying that a downloaded dataset is complete.
    """
    all_status = [r.get("status") for r in results]
    n_total = len(results)
    n_success = sum(1 for s in all_status if s == "success")
    n_failed = sum(1 for s in all_status if s == "failed")

    total_train_seconds = sum(
        r.get("training_time_seconds", 0.0) for r in results if r.get("status") == "success"
    )

    unique_algos = sorted({r.get("algorithm") for r in results if r.get("algorithm")})
    unique_envs = sorted({r.get("environment") for r in results if r.get("environment")})

    summary = {
        "generated_at": datetime.now().isoformat(),
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": n_success / n_total if n_total else 0.0,
        "total_training_hours": total_train_seconds / 3600.0,
        "unique_algorithms": unique_algos,
        "unique_environments": unique_envs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Overall summary written to {output_path}")
    logger.info(f"  success rate: {summary['success_rate']:.1%}")
    logger.info(f"  total training hours: {summary['total_training_hours']:.2f}")


# =============================================================================
# CLI
# =============================================================================

def _cmd_agent(args: argparse.Namespace) -> int:
    """Run the ``agent`` subcommand: evaluate an agent on one environment.

    Constructs a fresh algorithm instance from :mod:`experiments.algorithms`
    using the spec in :mod:`experiments.config`. For algorithms that require
    training, the caller is responsible for providing a checkpoint via
    the ``--load`` argument.
    """
    from experiments import algorithms

    coopetition_gym = _import_coopetition_gym()
    env = coopetition_gym.make(args.environment)

    spec = config.ALGORITHM_BY_NAME.get(args.algorithm)
    if spec is None:
        logger.error(f"Unknown algorithm: {args.algorithm}")
        return 2

    agent = algorithms.make_algorithm(spec, env, device=args.device, seed=args.seed_start)
    if args.load and spec.requires_training:
        agent.load(args.load)

    result = evaluate_agent(
        agent=agent,
        env_id=args.environment,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=args.deterministic,
        record_trajectories=args.record_trajectories,
        verbose=args.verbose,
        algorithm_name=args.algorithm,
    )

    write_evaluation_result(result, args.output)
    env.close()
    logger.info(f"Mean return: {result.mean_return:.2f} (std {result.std_return:.2f})")
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    """Run the ``aggregate`` subcommand."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_training_results(input_dir, status_filter="success")
    if not results:
        logger.error(f"No successful results found in {input_dir}")
        return 1

    aggregated = aggregate_by_algorithm_environment(results)
    write_summary_csv(aggregated, output_dir / "summary.csv")

    # Also write the corpus-level overall summary (considers all statuses).
    all_results = load_training_results(input_dir, status_filter=None)
    write_overall_statistics(all_results, output_dir / "summary.json")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ap = sub.add_parser("agent", help="Evaluate a single policy on one environment.")
    ap.add_argument("--algorithm", required=True,
                    help="Algorithm name from experiments.config.ALGORITHM_BY_NAME.")
    ap.add_argument("--environment", required=True,
                    help="Environment ID from experiments.config.ENVIRONMENT_BY_ID.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output JSON path for the EvaluationResult.")
    ap.add_argument("--episodes", type=int, default=100,
                    help="Number of evaluation episodes.")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="Starting seed; subsequent seeds are seed_start+i.")
    ap.add_argument("--deterministic", action="store_true", default=True,
                    help="Use deterministic actions (default: true).")
    ap.add_argument("--record-trajectories", action="store_true",
                    help="Record per-step rewards and trust (large memory).")
    ap.add_argument("--device", default="cpu",
                    help="Torch device: 'cpu' or 'cuda' (default: cpu).")
    ap.add_argument("--load", type=str, default=None,
                    help="Path to load a trained policy checkpoint from.")
    ap.add_argument("--verbose", action="store_true",
                    help="Log progress every 10 episodes.")
    ap.set_defaults(func=_cmd_agent)

    gp = sub.add_parser("aggregate", help="Aggregate training results into summary tables.")
    gp.add_argument("--input-dir", required=True,
                    help="Directory of training result JSON files.")
    gp.add_argument("--output-dir", required=True,
                    help="Output directory for summary.csv and summary.json.")
    gp.set_defaults(func=_cmd_aggregate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
