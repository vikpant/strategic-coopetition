"""Dataset integrity checks for the released training and audit datasets.

This module validates that a downloaded dataset matches the expected structure
and content. It provides three subcommands:

* ``training`` — Validate a directory of training result JSON files.
  Checks file count against :data:`experiments.config.EXPECTED_TRAINING_FILES`,
  verifies algorithm-environment combinations match the expected matrix,
  identifies NaN returns (62 are documented and expected), and confirms
  no failed experiments.

* ``audit`` — Validate a directory of behavioral audit JSON files.
  Checks file count against :data:`experiments.config.EXPECTED_AUDIT_STATIC_FILES`
  and :data:`experiments.config.EXPECTED_AUDIT_TEMPORAL_FILES`, verifies
  schema, and reports the exploitation classifications.

* ``schema`` — Print the expected schema for training and audit result files.
  Useful for users inspecting a single JSON file to understand its structure.

Usage:
    python -m experiments.validate training data/training/
    python -m experiments.validate audit data/audit/
    python -m experiments.validate schema training
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from coopetition_gym.experiments import config


# =============================================================================
# Training dataset validation
# =============================================================================

def validate_training_dataset(data_dir: Path) -> int:
    """Validate a training result directory.

    Returns the number of issues found. Zero means the dataset is clean.
    """
    issues = 0
    all_files: List[Path] = []
    for subfolder in data_dir.iterdir():
        if subfolder.is_dir():
            all_files.extend(subfolder.rglob("*.json"))

    print(f"Scanning {data_dir}...")
    print(f"Found {len(all_files):,} JSON files (expected {config.EXPECTED_TRAINING_FILES:,})")

    if len(all_files) != config.EXPECTED_TRAINING_FILES:
        delta = len(all_files) - config.EXPECTED_TRAINING_FILES
        print(f"  FILE COUNT MISMATCH: {delta:+,}")
        issues += 1

    algo_env_seed: Counter = Counter()
    nan_files: List[Path] = []
    failed_files: List[Path] = []
    parse_errors: List[Path] = []

    for f in all_files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            parse_errors.append(f)
            continue

        status = data.get("status", "unknown")
        if status == "failed":
            failed_files.append(f)

        returns = data.get("final_mean_returns") or data.get("mean_returns") or []
        if any(isinstance(r, float) and math.isnan(r) for r in returns):
            nan_files.append(f)

        algo = data.get("algorithm", "?")
        env = data.get("environment", "?")
        seed = data.get("seed", "?")
        algo_env_seed[(algo, env)] += 1

    print(f"\n{'Parse errors:':<30s} {len(parse_errors)}")
    print(f"{'Failed experiments:':<30s} {len(failed_files)}")
    print(f"{'NaN returns (62 expected):':<30s} {len(nan_files)}")

    if parse_errors:
        issues += len(parse_errors)
        for f in parse_errors[:10]:
            print(f"  PARSE ERROR: {f.name}")

    if failed_files:
        issues += len(failed_files)
        for f in failed_files[:10]:
            print(f"  FAILED: {f.name}")

    if len(nan_files) != 62:
        print(f"  UNEXPECTED NaN COUNT: {len(nan_files)} (expected 62)")
        issues += 1

    print(f"\nUnique (algorithm, environment) pairs: {len(algo_env_seed)}")
    return issues


# =============================================================================
# Audit dataset validation
# =============================================================================

def validate_audit_dataset(data_dir: Path) -> int:
    """Validate a behavioral audit result directory.

    Expects two subdirectories: ``static/`` with 1,056 files and
    ``temporal/`` with 60 files. Checks file counts, schema, and reports
    exploitation statistics.
    """
    issues = 0

    static_files = list((data_dir / "static").glob("*_audit.json"))
    temporal_files = list((data_dir / "temporal").glob("*_temporal.json"))

    print(f"Static audit files: {len(static_files):,} (expected {config.EXPECTED_AUDIT_STATIC_FILES:,})")
    print(f"Temporal audit files: {len(temporal_files):,} (expected {config.EXPECTED_AUDIT_TEMPORAL_FILES:,})")

    if len(static_files) != config.EXPECTED_AUDIT_STATIC_FILES:
        print(f"  STATIC COUNT MISMATCH: {len(static_files) - config.EXPECTED_AUDIT_STATIC_FILES:+,}")
        issues += 1

    if len(temporal_files) != config.EXPECTED_AUDIT_TEMPORAL_FILES:
        print(f"  TEMPORAL COUNT MISMATCH: {len(temporal_files) - config.EXPECTED_AUDIT_TEMPORAL_FILES:+,}")
        issues += 1

    # Validate static file schema
    static_schema_errors = 0
    total_exploitative = 0
    total_test_points = 0
    for f in static_files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            static_schema_errors += 1
            continue
        required = ("algorithm", "environment", "seed", "response_surface",
                    "exploitation_analysis", "n_exploitative")
        if not all(k in data for k in required):
            static_schema_errors += 1
            continue
        total_exploitative += data["n_exploitative"]
        total_test_points += 4  # Four test cooperation levels per experiment

    if static_schema_errors:
        print(f"  STATIC SCHEMA ERRORS: {static_schema_errors}")
        issues += static_schema_errors

    if total_test_points:
        pct = total_exploitative / total_test_points * 100
        print(f"Static exploitation rate: {total_exploitative:,}/{total_test_points:,} ({pct:.1f}%)")

    # Validate temporal file schema
    temporal_schema_errors = 0
    binary_exploit = 0
    binary_total = 0
    gradual_exploit = 0
    for f in temporal_files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            temporal_schema_errors += 1
            continue
        required = ("environment", "seed", "late_defection", "gradual_defection",
                    "temporal_profile")
        if not all(k in data for k in required):
            temporal_schema_errors += 1
            continue
        binary_exploit += data["temporal_profile"]["n_exploitative_switchpoints"]
        binary_total += data["temporal_profile"]["total_switchpoints_tested"]
        if data["gradual_defection"]["exploitative"]:
            gradual_exploit += 1

    if temporal_schema_errors:
        print(f"  TEMPORAL SCHEMA ERRORS: {temporal_schema_errors}")
        issues += temporal_schema_errors

    print(f"Temporal binary switchpoint exploitation: {binary_exploit}/{binary_total}")
    print(f"Temporal gradual ramp-down exploitation: {gradual_exploit}/{len(temporal_files)} files")

    return issues


# =============================================================================
# Schema reference
# =============================================================================

TRAINING_SCHEMA = {
    "algorithm": "str — algorithm name (e.g., ISAC, COMA)",
    "environment": "str — environment ID (e.g., TrustDilemma-v0)",
    "seed": "int — seed in {99, 100, 101, 102, 103, 104, 105}",
    "reward_type": "str — one of {private, integrated, cooperative}",
    "status": "str — 'success' for all released results",
    "final_mean_returns": "list[float] — mean return per agent at end of training",
    "training_returns": "list[list[float]] — per-step returns during training (training algos only)",
    "training_metrics": "dict — training-time metrics (actor_loss, critic_loss, ...)",
    "n_agents": "int — number of agents in the environment",
    "elapsed_seconds": "float — wall-clock training time",
}

STATIC_AUDIT_SCHEMA = {
    "algorithm": "str — algorithm label used for the audit",
    "environment": "str — environment ID",
    "seed": "int — seed in {99, 100, 101}",
    "n_agents": "int",
    "endowment": "float — agent endowment per step",
    "response_surface": "dict[str, dict] — cooperation fraction to {mean_return, per_agent_mean, ...}",
    "optimal_coop_level": "float — cooperation level that maximizes mean_return",
    "exploitation_analysis": "list[dict] — one entry per test cooperation level",
    "n_exploitative": "int — count of test levels where agent 0 gains and others lose",
}

TEMPORAL_AUDIT_SCHEMA = {
    "environment": "str — environment ID",
    "seed": "int — seed in {99, 100, 101}",
    "n_agents": "int",
    "episode_length": "int — steps per episode",
    "baseline": "dict — full-cooperation baseline result",
    "full_defection": "dict — agent 0 defects throughout",
    "late_defection": "list[dict] — one entry per switchpoint",
    "early_defection": "list[dict] — one entry per early-defect duration",
    "gradual_defection": "dict — linear ramp-down over final 20%",
    "temporal_profile": "dict — vulnerability classification summary",
}


def print_schema(kind: str) -> None:
    """Print the JSON schema for a given result file type."""
    schemas = {
        "training": TRAINING_SCHEMA,
        "static_audit": STATIC_AUDIT_SCHEMA,
        "temporal_audit": TEMPORAL_AUDIT_SCHEMA,
    }
    schema = schemas.get(kind)
    if schema is None:
        print(f"Unknown schema: {kind}. Available: {', '.join(schemas)}")
        sys.exit(2)

    print(f"Schema for {kind} result files:")
    for field, description in schema.items():
        print(f"  {field}: {description}")


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("training", help="Validate the training dataset.")
    tr.add_argument("data_dir", type=Path, help="Directory containing training subfolders.")

    au = sub.add_parser("audit", help="Validate the behavioral audit dataset.")
    au.add_argument("data_dir", type=Path, help="Directory containing static/ and temporal/ subdirectories.")

    sc = sub.add_parser("schema", help="Print the JSON schema for a result file type.")
    sc.add_argument("kind", choices=["training", "static_audit", "temporal_audit"])

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "training":
        issues = validate_training_dataset(args.data_dir)
    elif args.command == "audit":
        issues = validate_audit_dataset(args.data_dir)
    elif args.command == "schema":
        print_schema(args.kind)
        return 0
    else:
        return 2

    if issues:
        print(f"\nVALIDATION FAILED: {issues} issue(s)")
        return 1
    print("\nValidation clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
