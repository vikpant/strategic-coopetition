"""Validate native experiment records in recursive JSON/JSONL datasets.

Training validation reports schema, success, finiteness, treatment provenance,
duplicate full-context cells, and optional declared counts/checksums. Historic
corpus sizes and expected NaN counts are never assumed. Failed outcomes remain
in the dataset and produce validation issues rather than being silently dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .records import (REWARD_TYPES, RecordError, duplicate_issues, inspect_result,
                      read_records, record_kind, verify_manifest, _nonfinite_paths)


# =============================================================================
# Training dataset validation
# =============================================================================

def _report(issues):
    for issue in issues[:30]:
        print(f"  ISSUE: {issue}")
    if len(issues) > 30:
        print(f"  ... {len(issues) - 30} additional issues")
    return len(issues)


def validate_training_dataset(data_dir: Path, *, manifest=None, expected_records=None,
                              reward_type=None) -> int:
    """Validate all result rows, retaining failures and distinct treatment arms."""
    issues, results, progress = [], [], 0
    try:
        raw_records = read_records(data_dir)
    except RecordError as exc:
        return _report([str(exc)])
    for raw in raw_records:
        try:
            kind = record_kind(raw)
        except RecordError as exc:
            issues.append(f"{raw['_source']}: {exc}")
            continue
        if kind == "progress":
            progress += 1
            continue
        record = dict(raw)
        if "reward_type" not in record and reward_type is not None:
            record["reward_type"] = reward_type
            record["_reward_type_assumption"] = "explicit validator option"
        if "reward_type" not in record:
            issues.append(f"{record['_source']}: missing reward_type; supply explicit legacy context with --reward-type")
        issues.extend(f"{record['_source']}: {issue}" for issue in inspect_result(record))
        results.append(record)
    if not results:
        issues.append("no result records found")
    if expected_records is not None and len(results) != expected_records:
        issues.append(f"result count {len(results)} != declared {expected_records}")
    issues.extend(duplicate_issues(results))
    if manifest is not None:
        try:
            issues.extend(verify_manifest(data_dir, manifest))
        except (OSError, ValueError) as exc:
            issues.append(f"cannot read manifest: {exc}")
    print(f"Result records: {len(results)}; progress records: {progress}")
    print(f"Reward treatments: {dict(Counter(str(r.get('reward_type', 'unknown')) for r in results))}")
    print(f"Observed training seeds: {sorted({r['training_seed'] for r in results if isinstance(r.get('training_seed'), int) and not isinstance(r['training_seed'], bool)})}")
    return _report(issues)


def validate_audit_dataset(data_dir: Path, *, manifest=None, expected_static=None,
                           expected_temporal=None) -> int:
    """Validate static/temporal audit objects without a fixed historic corpus size."""
    issues, counts, seen = [], Counter(), set()
    try:
        records = read_records(data_dir)
    except RecordError as exc:
        return _report([str(exc)])
    for record in records:
        source = record['_source']
        if 'response_surface' in record:
            kind = 'static'
            required = ('algorithm', 'environment', 'seed', 'response_surface',
                        'exploitation_analysis', 'n_exploitative')
        elif 'temporal_profile' in record:
            kind = 'temporal'
            required = ('environment', 'seed', 'late_defection', 'gradual_defection', 'temporal_profile')
        else:
            issues.append(f"{source}: unrecognized audit record")
            continue
        counts[kind] += 1
        missing = [key for key in required if key not in record]
        if missing:
            issues.append(f"{source}: missing fields {missing}")
            continue
        seed = record['seed']
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            issues.append(f"{source}: seed must be a nonnegative integer")
        if not isinstance(record['environment'], str) or not record['environment']:
            issues.append(f"{source}: environment must be a nonempty string")
        issues.extend(f"{source}: nonfinite {name}" for name in _nonfinite_paths(record, 'record'))
        try:
            key = (kind, record.get('algorithm'), record['environment'], seed, record.get('reward_type'))
            if key in seen:
                issues.append(f"{source}: duplicate audit cell {key}")
            seen.add(key)
        except TypeError:
            issues.append(f"{source}: malformed audit identity")
        if kind == 'static':
            entries = record['exploitation_analysis']
            if not isinstance(record['response_surface'], dict) or not record['response_surface']:
                issues.append(f"{source}: response_surface must be a nonempty object")
            if not isinstance(entries, list) or any(not isinstance(e, dict) or not isinstance(e.get('exploitative'), bool) for e in entries):
                issues.append(f"{source}: malformed exploitation_analysis")
            elif record['n_exploitative'] != sum(e['exploitative'] for e in entries):
                issues.append(f"{source}: n_exploitative disagrees with per-level results")
        else:
            profile = record['temporal_profile']
            if not isinstance(profile, dict) or not all(isinstance(profile.get(key), int) and not isinstance(profile[key], bool) and profile[key] >= 0 for key in ('n_exploitative_switchpoints', 'total_switchpoints_tested')):
                issues.append(f"{source}: malformed temporal_profile counts")
            elif profile['n_exploitative_switchpoints'] > profile['total_switchpoints_tested']:
                issues.append(f"{source}: exploitative switchpoints exceed total")
            if not isinstance(record['gradual_defection'], dict) or not isinstance(record['gradual_defection'].get('exploitative'), bool):
                issues.append(f"{source}: malformed gradual_defection")
    for kind, expected in (('static', expected_static), ('temporal', expected_temporal)):
        if expected is not None and counts[kind] != expected:
            issues.append(f"{kind} count {counts[kind]} != declared {expected}")
    if not sum(counts.values()):
        issues.append("no audit records found")
    if manifest is not None:
        try:
            issues.extend(verify_manifest(data_dir, manifest))
        except (OSError, ValueError) as exc:
            issues.append(f"cannot read manifest: {exc}")
    print(f"Static audit records: {counts['static']}; temporal audit records: {counts['temporal']}")
    return _report(issues)


# =============================================================================
# Schema reference
# =============================================================================

TRAINING_SCHEMA = {
    "algorithm": "nonempty string — algorithm implementation label",
    "environment": "nonempty string — environment identifier",
    "training_seed": "nonnegative integer — no implicit canonical fold",
    "reward_type": "private, integrated, or cooperative (explicit legacy context permitted)",
    "status": "success or failed; failed outcomes are reported as issues",
    "metrics": "object containing finite numeric mean_return; numeric metrics must be finite",
    "campaign_id / config_hash": "preserved configuration provenance when provided",
    "requires_training / training_steps_requested / training_steps_completed": "checked together when provided; learners must complete a positive budget",
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
    tr.add_argument("data_dir", type=Path, help="JSON/JSONL file or recursive result directory.")
    tr.add_argument("--manifest", type=Path, help="Optional CSV of relative paths, physical row counts, and MD5/SHA256 hashes.")
    tr.add_argument("--expected-records", type=int, help="Explicit expected result count, excluding progress rows.")
    tr.add_argument("--reward-type", choices=sorted(REWARD_TYPES), help="Explicit treatment context for legacy rows missing reward_type.")

    au = sub.add_parser("audit", help="Validate the behavioral audit dataset.")
    au.add_argument("data_dir", type=Path, help="JSON/JSONL file or recursive audit directory.")
    au.add_argument("--manifest", type=Path)
    au.add_argument("--expected-static", type=int)
    au.add_argument("--expected-temporal", type=int)

    sc = sub.add_parser("schema", help="Print the JSON schema for a result file type.")
    sc.add_argument("kind", choices=["training", "static_audit", "temporal_audit"])

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "training":
        issues = validate_training_dataset(args.data_dir, manifest=args.manifest, expected_records=args.expected_records, reward_type=args.reward_type)
    elif args.command == "audit":
        issues = validate_audit_dataset(args.data_dir, manifest=args.manifest, expected_static=args.expected_static, expected_temporal=args.expected_temporal)
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
