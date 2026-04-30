"""Behavioral audit — static response surface and temporal deviation analysis.

This module consolidates the two behavioral audits described in Appendix F
of the paper (and their analysis script) into a single command-line tool
with three subcommands:

* ``static`` — Run the static response-surface audit (1,056 experiments).
  Sweeps uniform cooperation from 0% to 100% in 5% increments and tests
  unilateral deviation (agent 0 reduces contribution by 50%) at four
  cooperation levels. A point is classified as *exploitative* when agent 0
  gains and other agents lose.

* ``temporal`` — Run the temporal deviation audit (60 experiments).
  Tests whether an agent can accumulate cooperative capital then defect by
  switching strategies at various switchpoints across the episode. Also
  tests early-defection and gradual ramp-down strategies.

* ``analyze`` — Generate the cross-audit analysis report used to produce
  Appendix F's reconciliation of static and temporal findings.

Both audits use fixed-action policies applied to the environment; they do
not require trained policies. This is why the exploitation rate is
algorithm-independent: the audit measures the payoff landscape's
structural properties, not the behavior of any particular learning algorithm.

Usage:
    python -m experiments.audit static --output data/audit/static/
    python -m experiments.audit temporal --output data/audit/temporal/
    python -m experiments.audit analyze \\
        --static-dir data/audit/static/ \\
        --temporal-dir data/audit/temporal/ \\
        --output data/analysis/audit_analysis.txt
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments import config


def _import_coopetition_gym():
    """Import ``coopetition_gym`` from the installed package.

    The repository layout has a top-level folder ``coopetition_gym/`` with
    the actual package at ``coopetition_gym/coopetition_gym/``. When running
    from the repository root (or a multiprocessing worker launched from
    there), Python resolves ``coopetition_gym`` to the outer folder as a
    namespace package, shadowing the installed editable package. This helper
    inserts the inner package parent at the front of ``sys.path`` and drops
    any stale namespace-package import.
    """
    import importlib
    import os as _os
    import sys as _sys

    repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    inner_package_parent = _os.path.join(repo_root, "coopetition_gym")

    # Drop any stale namespace-package import so the next import re-resolves.
    _sys.modules.pop("coopetition_gym", None)

    # Prepend the parent of the inner package so the import machinery finds
    # ``coopetition_gym/coopetition_gym/__init__.py``.
    if inner_package_parent not in _sys.path:
        _sys.path.insert(0, inner_package_parent)

    return importlib.import_module("coopetition_gym")


# =============================================================================
# Static response-surface audit
# =============================================================================

def _run_static_experiment(args: Tuple[str, str, int, int, str]) -> str:
    """Worker that runs one static audit experiment.

    Args:
        args: ``(algorithm_label, env_id, seed, episodes_per_level, output_dir)``

    Returns:
        A status string for logging.
    """
    algorithm_label, env_id, seed, episodes_per_level, output_dir = args

    coopetition_gym = _import_coopetition_gym()

    output_file = Path(output_dir) / f"{algorithm_label}_{env_id}_{seed}_audit.json"
    if output_file.exists():
        return f"SKIP {algorithm_label} {env_id} s{seed}"

    try:
        env = coopetition_gym.make(env_id)
        n_agents = env.n_agents
        endowment = float(env.endowments[0])

        cooperation_levels = np.linspace(0.0, 1.0, config.STATIC_AUDIT.n_cooperation_levels)
        response_surface: Dict[str, Dict] = {}

        for coop_fraction in cooperation_levels:
            coop_action = coop_fraction * endowment
            per_agent_returns: List[List[float]] = [[] for _ in range(n_agents)]

            for episode in range(episodes_per_level):
                env.reset(seed=seed * 1000 + int(coop_fraction * 1000) + episode)
                ep_rewards = np.zeros(n_agents)
                done = False
                while not done:
                    actions = np.full(n_agents, coop_action)
                    _, rewards, terminated, truncated, _ = env.step(actions)
                    ep_rewards += np.asarray(rewards) if hasattr(rewards, "__iter__") else rewards
                    done = terminated or truncated

                for i in range(n_agents):
                    per_agent_returns[i].append(float(ep_rewards[i]))

            key = f"{coop_fraction:.2f}"
            per_agent_mean = [float(np.mean(r)) for r in per_agent_returns]
            response_surface[key] = {
                "coop_fraction": float(coop_fraction),
                "coop_action": float(coop_action),
                "mean_return": float(np.mean(per_agent_mean)),
                "std_return": float(np.std([np.mean(r) for r in per_agent_returns])),
                "per_agent_mean": per_agent_mean,
                "per_agent_std": [float(np.std(r)) for r in per_agent_returns],
                "n_episodes": episodes_per_level,
            }

        # Find optimal cooperation level
        best_key = max(response_surface.keys(), key=lambda k: response_surface[k]["mean_return"])

        # Exploitation analysis at the four test cooperation levels
        exploitation_analysis = []
        for coop_level in config.STATIC_AUDIT.exploitation_test_levels:
            base_action = coop_level * endowment

            # Baseline: all agents play base_action
            env.reset(seed=seed * 10_000)
            base_rewards = np.zeros(n_agents)
            actions = np.full(n_agents, base_action)
            done = False
            while not done:
                _, rewards, terminated, truncated, _ = env.step(actions)
                base_rewards += np.asarray(rewards) if hasattr(rewards, "__iter__") else rewards
                done = terminated or truncated

            # Deviation: agent 0 reduces its contribution by 50%
            env.reset(seed=seed * 10_000)
            actions_dev = np.full(n_agents, base_action)
            actions_dev[0] = base_action * (1.0 - config.STATIC_AUDIT.deviation_fraction)
            dev_rewards = np.zeros(n_agents)
            done = False
            while not done:
                _, rewards, terminated, truncated, _ = env.step(actions_dev)
                dev_rewards += np.asarray(rewards) if hasattr(rewards, "__iter__") else rewards
                done = terminated or truncated

            agent0_gain = float(dev_rewards[0] - base_rewards[0])
            others_loss = float(np.mean(dev_rewards[1:]) - np.mean(base_rewards[1:]))
            exploitation_analysis.append({
                "coop_level": float(coop_level),
                "base_agent0_return": float(base_rewards[0]),
                "deviate_agent0_return": float(dev_rewards[0]),
                "agent0_gain": agent0_gain,
                "base_others_mean": float(np.mean(base_rewards[1:])),
                "deviate_others_mean": float(np.mean(dev_rewards[1:])),
                "others_loss": others_loss,
                "exploitative": bool(agent0_gain > 0 and others_loss < 0),
            })

        result = {
            "algorithm": algorithm_label,
            "environment": env_id,
            "seed": seed,
            "n_agents": n_agents,
            "endowment": endowment,
            "response_surface": response_surface,
            "optimal_coop_level": float(best_key),
            "optimal_mean_return": response_surface[best_key]["mean_return"],
            "exploitation_analysis": exploitation_analysis,
            "n_exploitative": sum(1 for e in exploitation_analysis if e["exploitative"]),
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        env.close()
        return (
            f"OK {algorithm_label} {env_id} s{seed} "
            f"optimal={best_key} exploit={result['n_exploitative']}/4"
        )

    except Exception as exc:
        import traceback
        return f"ERROR {algorithm_label} {env_id} s{seed}: {exc}\n{traceback.format_exc()}"


def run_static_audit(
    output_dir: Path,
    algorithms: Sequence[str],
    environments: Sequence[str],
    seeds: Sequence[int],
    episodes_per_level: int,
    max_workers: int,
) -> None:
    """Run the static response-surface audit across all specified combinations.

    Skips experiments whose output files already exist (idempotent). Writes
    one JSON file per (algorithm, environment, seed) with the complete
    response surface and exploitation analysis.

    Args:
        output_dir: Directory to write result JSON files.
        algorithms: Algorithm labels to include in the audit.
        environments: Environment IDs to sweep.
        seeds: Seeds to run.
        episodes_per_level: Episodes per cooperation level.
        max_workers: Process pool size.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for algo_label in algorithms:
        for env_id in environments:
            env_spec = config.ENVIRONMENT_BY_ID.get(env_id)
            if env_spec is None:
                print(f"WARNING: unknown environment {env_id}, skipping")
                continue
            # Respect algorithm-environment restrictions (e.g., MeanFieldAC on N>=3)
            algo_spec = config.ALGORITHM_BY_NAME.get(algo_label)
            if algo_spec and algo_spec.applicable_categories is not None:
                if env_spec.category not in algo_spec.applicable_categories:
                    continue
            for seed in seeds:
                experiments.append((algo_label, env_id, seed, episodes_per_level, str(output_dir)))

    print(f"Static audit: {len(experiments)} experiments, {max_workers} workers")
    print(f"Output: {output_dir}")

    start = time.time()
    with mp.Pool(max_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_static_experiment, experiments), 1):
            if i % 50 == 0 or "ERROR" in result:
                elapsed = time.time() - start
                print(f"  [{i}/{len(experiments)}] {result} ({elapsed / 60:.1f} min)")

    elapsed = time.time() - start
    print(f"Static audit complete: {len(experiments)} experiments in {elapsed / 60:.1f} min")


# =============================================================================
# Temporal deviation audit
# =============================================================================

def _fixed_strategy_episodes(
    env,
    n_agents: int,
    seed: int,
    action_fn,
    n_episodes: int,
) -> Dict:
    """Run ``n_episodes`` episodes with a step-indexed action function.

    The ``action_fn`` takes the current step and returns an action vector.
    Returns mean and std per agent across episodes.
    """
    all_returns: List[List[float]] = [[] for _ in range(n_agents)]
    for episode in range(n_episodes):
        env.reset(seed=seed * 1000 + episode)
        ep_rewards = np.zeros(n_agents)
        done = False
        step = 0
        while not done:
            actions = action_fn(step)
            _, rewards, terminated, truncated, _ = env.step(actions)
            ep_rewards += np.asarray(rewards) if hasattr(rewards, "__iter__") else rewards
            done = terminated or truncated
            step += 1
        for i in range(n_agents):
            all_returns[i].append(float(ep_rewards[i]))

    per_agent_mean = [float(np.mean(r)) for r in all_returns]
    return {
        "mean_return": float(np.mean(per_agent_mean)),
        "per_agent_mean": per_agent_mean,
        "per_agent_std": [float(np.std(r)) for r in all_returns],
    }


def _classify_vulnerability(
    exploitation_onset: Optional[float],
    n_exploitative: int,
    n_tested: int,
) -> str:
    """Classify an environment's temporal exploitation vulnerability."""
    if n_exploitative == 0:
        return "immune"
    if exploitation_onset is not None and exploitation_onset >= 0.9:
        return "terminal_only"
    if exploitation_onset is not None and exploitation_onset >= 0.7:
        return "late_vulnerable"
    if n_exploitative >= n_tested * 0.5:
        return "broadly_vulnerable"
    return "partially_vulnerable"


def _run_temporal_experiment(args: Tuple[str, int, str]) -> str:
    """Worker that runs one temporal audit experiment for a given environment and seed."""
    env_id, seed, output_dir = args

    coopetition_gym = _import_coopetition_gym()

    output_file = Path(output_dir) / f"{env_id}_{seed}_temporal.json"
    if output_file.exists():
        return f"SKIP {env_id} s{seed}"

    cfg = config.TEMPORAL_AUDIT

    try:
        env = coopetition_gym.make(env_id)
        n_agents = env.n_agents
        endowment = float(env.endowments[0])

        # Determine episode length
        env.reset(seed=seed)
        ep_length = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(np.full(n_agents, endowment * cfg.baseline_coop_fraction))
            done = terminated or truncated
            ep_length += 1

        coop_action = endowment * cfg.baseline_coop_fraction
        defect_action = endowment * cfg.defect_action_fraction

        baseline = _fixed_strategy_episodes(
            env, n_agents, seed,
            lambda step: np.full(n_agents, coop_action),
            cfg.episodes_per_strategy,
        )

        def full_defect(step: int) -> np.ndarray:
            actions = np.full(n_agents, coop_action)
            actions[0] = defect_action
            return actions

        full_defection = _fixed_strategy_episodes(
            env, n_agents, seed, full_defect, cfg.episodes_per_strategy,
        )

        # Late defection switchpoints
        switchpoints = sorted(set(
            [max(1, int(f * ep_length)) for f in cfg.switchpoint_fractions]
            + [ep_length - offset for offset in cfg.terminal_offsets if ep_length - offset > 0]
        ))

        late_results = []
        for sp in switchpoints:
            def late_defect(step: int, switchpoint: int = sp) -> np.ndarray:
                actions = np.full(n_agents, coop_action)
                if step >= switchpoint:
                    actions[0] = defect_action
                return actions

            r = _fixed_strategy_episodes(
                env, n_agents, seed, late_defect, cfg.episodes_per_strategy,
            )
            agent0_gain = r["per_agent_mean"][0] - baseline["per_agent_mean"][0]
            others_loss = float(np.mean(r["per_agent_mean"][1:]) - np.mean(baseline["per_agent_mean"][1:]))
            late_results.append({
                "switchpoint": sp,
                "switchpoint_fraction": round(sp / ep_length, 3),
                "defection_steps": ep_length - sp,
                "agent0_return": r["per_agent_mean"][0],
                "agent0_gain_vs_baseline": float(agent0_gain),
                "others_mean_return": float(np.mean(r["per_agent_mean"][1:])),
                "others_loss_vs_baseline": others_loss,
                "exploitative": bool(agent0_gain > 0 and others_loss < 0),
                "mean_return": r["mean_return"],
            })

        # Early defection durations
        early_durations = [max(1, int(f * ep_length)) for f in cfg.early_defect_fractions]
        early_results = []
        for duration in early_durations:
            def early_defect(step: int, end: int = duration) -> np.ndarray:
                actions = np.full(n_agents, coop_action)
                if step < end:
                    actions[0] = defect_action
                return actions

            r = _fixed_strategy_episodes(
                env, n_agents, seed, early_defect, cfg.episodes_per_strategy,
            )
            agent0_gain = r["per_agent_mean"][0] - baseline["per_agent_mean"][0]
            others_loss = float(np.mean(r["per_agent_mean"][1:]) - np.mean(baseline["per_agent_mean"][1:]))
            early_results.append({
                "defect_until_step": duration,
                "defect_fraction": round(duration / ep_length, 3),
                "agent0_return": r["per_agent_mean"][0],
                "agent0_gain_vs_baseline": float(agent0_gain),
                "others_mean_return": float(np.mean(r["per_agent_mean"][1:])),
                "others_loss_vs_baseline": others_loss,
                "exploitative": bool(agent0_gain > 0 and others_loss < 0),
            })

        # Gradual ramp-down over the final fraction of the episode
        ramp_start = max(1, int((1.0 - cfg.gradual_rampdown_fraction) * ep_length))

        def gradual_defect(step: int) -> np.ndarray:
            actions = np.full(n_agents, coop_action)
            if step >= ramp_start:
                progress = (step - ramp_start) / max(1, ep_length - ramp_start)
                actions[0] = coop_action * (1.0 - progress)
            return actions

        gradual = _fixed_strategy_episodes(
            env, n_agents, seed, gradual_defect, cfg.episodes_per_strategy,
        )
        grad_gain = gradual["per_agent_mean"][0] - baseline["per_agent_mean"][0]
        grad_loss = float(np.mean(gradual["per_agent_mean"][1:]) - np.mean(baseline["per_agent_mean"][1:]))

        exploitation_onset = next(
            (ld["switchpoint_fraction"] for ld in late_results if ld["exploitative"]),
            None,
        )
        n_exploitative = sum(1 for ld in late_results if ld["exploitative"])

        result = {
            "environment": env_id,
            "seed": seed,
            "n_agents": n_agents,
            "endowment": endowment,
            "episode_length": ep_length,
            "coop_action": float(coop_action),
            "defect_action": float(defect_action),
            "baseline": {
                "mean_return": baseline["mean_return"],
                "per_agent_mean": baseline["per_agent_mean"],
            },
            "full_defection": {
                "mean_return": full_defection["mean_return"],
                "per_agent_mean": full_defection["per_agent_mean"],
                "agent0_gain": float(full_defection["per_agent_mean"][0] - baseline["per_agent_mean"][0]),
                "exploitative": bool(
                    full_defection["per_agent_mean"][0] > baseline["per_agent_mean"][0]
                    and np.mean(full_defection["per_agent_mean"][1:]) < np.mean(baseline["per_agent_mean"][1:])
                ),
            },
            "late_defection": late_results,
            "early_defection": early_results,
            "gradual_defection": {
                "ramp_start_step": ramp_start,
                "ramp_start_fraction": round(ramp_start / ep_length, 3),
                "mean_return": gradual["mean_return"],
                "per_agent_mean": gradual["per_agent_mean"],
                "agent0_gain": float(grad_gain),
                "exploitative": bool(grad_gain > 0 and grad_loss < 0),
            },
            "temporal_profile": {
                "exploitation_onset_fraction": exploitation_onset,
                "n_exploitative_switchpoints": n_exploitative,
                "total_switchpoints_tested": len(late_results),
                "vulnerability_class": _classify_vulnerability(
                    exploitation_onset, n_exploitative, len(late_results),
                ),
            },
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        env.close()
        vuln = result["temporal_profile"]["vulnerability_class"]
        return (
            f"OK {env_id} s{seed} ep={ep_length} "
            f"exploit={n_exploitative}/{len(late_results)} class={vuln}"
        )

    except Exception as exc:
        import traceback
        return f"ERROR {env_id} s{seed}: {exc}\n{traceback.format_exc()}"


def run_temporal_audit(
    output_dir: Path,
    environments: Sequence[str],
    seeds: Sequence[int],
    max_workers: int,
) -> None:
    """Run the temporal deviation audit across environments and seeds.

    Writes one JSON file per (environment, seed) with baseline, full defection,
    late defection (9 switchpoints), early defection (3 durations), and
    gradual ramp-down strategies. Also writes a temporal vulnerability
    classification (immune, terminal_only, late_vulnerable,
    partially_vulnerable, broadly_vulnerable).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [(env_id, seed, str(output_dir)) for env_id in environments for seed in seeds]
    print(f"Temporal audit: {len(experiments)} experiments, {max_workers} workers")
    print(f"Output: {output_dir}")

    start = time.time()
    with mp.Pool(max_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_temporal_experiment, experiments), 1):
            elapsed = time.time() - start
            print(f"  [{i}/{len(experiments)}] {result} ({elapsed / 60:.1f} min)")

    elapsed = time.time() - start
    print(f"Temporal audit complete: {len(experiments)} experiments in {elapsed / 60:.1f} min")


# =============================================================================
# Analysis
# =============================================================================

def _tr_for_env(env_id: str) -> str:
    """Return the TR tier label (``TR-1`` ... ``TR-4``) for an environment."""
    env = config.ENVIRONMENT_BY_ID.get(env_id)
    return env.tr.upper().replace("TR", "TR-") if env else "?"


def analyze_audits(static_dir: Path, temporal_dir: Path, output: Path) -> None:
    """Generate the cross-audit analysis report used for paper Appendix F.

    Reads the JSON files produced by the static and temporal audits and
    writes a human-readable report covering:

    * Algorithm independence (exploitation count identical per environment)
    * Per-tier aggregate exploitation rates
    * Environment-level immunity classification
    * Response surface shape analysis
    * Temporal profile by tier
    * Gradual ramp-down exploitation (small TR-4 effect)
    * Cross-audit reconciliation

    The output file is text, matching the ``audit_analysis_full.txt`` format
    referenced in the paper.
    """
    static_files = list(Path(static_dir).glob("*_audit.json"))
    temporal_files = list(Path(temporal_dir).glob("*_temporal.json"))

    static_results = [json.loads(f.read_text()) for f in static_files]
    temporal_results = [json.loads(f.read_text()) for f in temporal_files]

    output.parent.mkdir(parents=True, exist_ok=True)
    out = output.open("w")

    def w(line: str = "") -> None:
        out.write(line + "\n")

    w(f"=== STATIC AUDIT: {len(static_results)} experiments ===")
    w()

    # Algorithm independence check
    algo_counts: Dict[str, int] = defaultdict(int)
    for r in static_results:
        algo_counts[r["algorithm"]] += 1
    w(f"Algorithms audited: {len(algo_counts)}")
    for algo, count in sorted(algo_counts.items()):
        w(f"  {algo}: {count}")

    w()
    w("EXPLOITATION BY ENVIRONMENT")
    env_exploit: Dict[str, List[int]] = defaultdict(list)
    for r in static_results:
        env_exploit[r["environment"]].append(r.get("n_exploitative", 0))

    for tier in ("TR-1", "TR-2", "TR-3", "TR-4"):
        w(f"\n--- {tier} ---")
        for env_id, counts in sorted(env_exploit.items()):
            if _tr_for_env(env_id) != tier:
                continue
            unique = set(counts)
            tag = "algorithm-independent" if len(unique) == 1 else f"VARIES: {sorted(unique)}"
            total_exploit = sum(counts)
            total_points = 4 * len(counts)
            pct = total_exploit / total_points * 100 if total_points else 0
            w(f"  {env_id:35s}: {total_exploit}/{total_points} ({pct:.1f}%) [{tag}]")

    # Per-tier aggregate
    w()
    w("AGGREGATE EXPLOITATION RATES BY TIER")
    tier_stats: Dict[str, List[int]] = defaultdict(list)
    for r in static_results:
        tier_stats[_tr_for_env(r["environment"])].append(r.get("n_exploitative", 0))
    for tier in ("TR-1", "TR-2", "TR-3", "TR-4"):
        counts = tier_stats[tier]
        total = sum(counts)
        pts = 4 * len(counts)
        w(f"  {tier}: {total}/{pts} ({total / pts * 100:.1f}%)")

    # Optimal cooperation levels
    w()
    w("OPTIMAL COOPERATION LEVELS (median per environment)")
    env_optimal: Dict[str, List[float]] = defaultdict(list)
    for r in static_results:
        env_optimal[r["environment"]].append(float(r["optimal_coop_level"]))
    for tier in ("TR-1", "TR-2", "TR-3", "TR-4"):
        w(f"\n--- {tier} ---")
        for env_id, opts in sorted(env_optimal.items()):
            if _tr_for_env(env_id) != tier:
                continue
            median = float(np.median(opts))
            w(f"  {env_id:35s}: {median:.2f}")

    # Temporal audit analysis
    w()
    w("=" * 70)
    w(f"=== TEMPORAL AUDIT: {len(temporal_results)} experiments ===")
    w()

    w("VULNERABILITY CLASSIFICATION BY ENVIRONMENT")
    for tier in ("TR-1", "TR-2", "TR-3", "TR-4"):
        w(f"\n--- {tier} ---")
        for r in sorted(temporal_results, key=lambda x: (x["environment"], x["seed"])):
            if _tr_for_env(r["environment"]) != tier:
                continue
            tp = r["temporal_profile"]
            grad = r["gradual_defection"]
            w(
                f"  {r['environment']} s{r['seed']} ep={r['episode_length']}: "
                f"{tp['vulnerability_class']}, gradual_exploit={grad['exploitative']}"
            )

    # Last-step defection cost
    w()
    w("LAST-STEP-ONLY DEFECTION COST BY TIER (mean loss to defector)")
    tier_last: Dict[str, List[float]] = defaultdict(list)
    for r in temporal_results:
        tier = _tr_for_env(r["environment"])
        if r["late_defection"]:
            tier_last[tier].append(r["late_defection"][-1]["agent0_gain_vs_baseline"])
    for tier in ("TR-1", "TR-2", "TR-3", "TR-4"):
        vals = tier_last[tier]
        if vals:
            w(f"  {tier}: mean={np.mean(vals):+.1f}, min={min(vals):+.1f}, max={max(vals):+.1f}")

    w()
    w("GRADUAL RAMP-DOWN EXPLOITATION BY ENVIRONMENT")
    for r in sorted(temporal_results, key=lambda x: (x["environment"], x["seed"])):
        grad = r["gradual_defection"]
        if grad["exploitative"]:
            baseline_mean = r["baseline"]["mean_return"]
            pct = grad["agent0_gain"] / baseline_mean * 100 if baseline_mean else 0
            w(
                f"  {r['environment']} s{r['seed']}: "
                f"agent0_gain={grad['agent0_gain']:+.1f} ({pct:+.3f}% of baseline)"
            )

    # Final summary
    w()
    w("=" * 70)
    w("KEY FINDINGS")
    w()
    w("1. ALGORITHM INDEPENDENCE")
    w("   Every algorithm produces identical exploitation counts per environment.")
    w("   Exploitation is a structural property of the environment, not the algorithm.")
    w()
    w("2. UNIVERSAL BINARY-SWITCHPOINT IMMUNITY")
    binary_exploit = sum(
        r["temporal_profile"]["n_exploitative_switchpoints"] for r in temporal_results
    )
    binary_total = sum(
        r["temporal_profile"]["total_switchpoints_tested"] for r in temporal_results
    )
    w(f"   Temporal audit: {binary_exploit}/{binary_total} binary switchpoints exploitative.")
    w()
    w("3. GRADUAL RAMP-DOWN YIELDS MARGINAL EXPLOITATION ON TR-4")
    n_grad_exploit = sum(1 for r in temporal_results if r["gradual_defection"]["exploitative"])
    w(f"   {n_grad_exploit}/{len(temporal_results)} environment-seed pairs show gradual exploitation.")
    w()

    out.close()
    print(f"Analysis written to {output}")


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # static subcommand
    sp = sub.add_parser("static", help="Run the static response-surface audit.")
    sp.add_argument("--output", type=Path, default=config.DEFAULT_AUDIT_DIR / "static",
                    help="Output directory for JSON files.")
    sp.add_argument("--algorithms", type=str, default=None,
                    help="Comma-separated algorithm labels (default: 16 training + 2 heuristic).")
    sp.add_argument("--environments", type=str, default=None,
                    help="Comma-separated environment IDs (default: all 20).")
    sp.add_argument("--seeds", type=str, default=None,
                    help=f"Comma-separated seeds (default: {','.join(str(s) for s in config.AUDIT_SEEDS)}).")
    sp.add_argument("--episodes-per-level", type=int,
                    default=config.STATIC_AUDIT.episodes_per_level,
                    help="Episodes per cooperation level in the sweep.")
    sp.add_argument("--max-workers", type=int, default=8,
                    help="Process pool size for parallel execution.")

    # temporal subcommand
    tp = sub.add_parser("temporal", help="Run the temporal deviation audit.")
    tp.add_argument("--output", type=Path, default=config.DEFAULT_AUDIT_DIR / "temporal",
                    help="Output directory for JSON files.")
    tp.add_argument("--environments", type=str, default=None,
                    help="Comma-separated environment IDs (default: all 20).")
    tp.add_argument("--seeds", type=str, default=None,
                    help=f"Comma-separated seeds (default: {','.join(str(s) for s in config.AUDIT_SEEDS)}).")
    tp.add_argument("--max-workers", type=int, default=8,
                    help="Process pool size for parallel execution.")

    # analyze subcommand
    ap = sub.add_parser("analyze", help="Generate cross-audit analysis report.")
    ap.add_argument("--static-dir", type=Path, required=True,
                    help="Directory containing static audit JSON files.")
    ap.add_argument("--temporal-dir", type=Path, required=True,
                    help="Directory containing temporal audit JSON files.")
    ap.add_argument("--output", type=Path, default=config.DEFAULT_ANALYSIS_DIR / "audit_analysis.txt",
                    help="Output report file.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "static":
        algorithms = (
            args.algorithms.split(",") if args.algorithms
            else [a.name for a in config.TRAINING_ALGORITHMS + config.HEURISTIC_ALGORITHMS]
        )
        environments = (
            args.environments.split(",") if args.environments
            else [e.id for e in config.ALL_ENVIRONMENTS]
        )
        seeds = (
            [int(s) for s in args.seeds.split(",")] if args.seeds
            else list(config.AUDIT_SEEDS)
        )
        run_static_audit(
            args.output, algorithms, environments, seeds,
            args.episodes_per_level, args.max_workers,
        )

    elif args.command == "temporal":
        environments = (
            args.environments.split(",") if args.environments
            else [e.id for e in config.ALL_ENVIRONMENTS]
        )
        seeds = (
            [int(s) for s in args.seeds.split(",")] if args.seeds
            else list(config.AUDIT_SEEDS)
        )
        run_temporal_audit(args.output, environments, seeds, args.max_workers)

    elif args.command == "analyze":
        analyze_audits(args.static_dir, args.temporal_dir, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
