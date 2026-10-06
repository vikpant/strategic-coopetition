"""Strict readers for per-run JSON and JSONL experiment records.

Readers preserve every original field and attach ``_source`` for diagnostics.
They do not infer reward treatments, canonical folds, or preferred reruns from
filenames or folder names. Repeated cells require an explicit upstream decision.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Optional

REWARD_TYPES = {"private", "integrated", "cooperative"}


class RecordError(ValueError):
    """Input cannot be interpreted as an unambiguous experiment dataset."""


def is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonfinite_paths(value, prefix="metrics"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _nonfinite_paths(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _nonfinite_paths(child, f"{prefix}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        yield prefix


def inspect_result(record):
    """Return schema/quality issues for one *successful* native result.

    Failed and nonfinite outcomes remain readable for validation and diagnostics,
    but cannot enter a successful-result aggregate or campaign completion set.
    New training evidence is checked when present; old records are not claimed
    to prove that their requested training budget was completed.
    """
    if not isinstance(record, dict):
        return ["result must be an object"]
    issues = []
    for name in ("algorithm", "environment"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            issues.append(f"{name} must be a nonempty string")
    for name in ("record_type", "type", "event"):
        if name in record and not isinstance(record[name], str):
            issues.append(f"{name} must be a string when present")
    seed = record.get("training_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        issues.append("training_seed must be a nonnegative integer")
    if "seed" in record and record["seed"] != seed:
        issues.append("seed and training_seed disagree")
    if record.get("status") != "success":
        issues.append(f"status is {record.get('status')!r}, not success")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        issues.append("metrics must be an object")
    elif not is_finite_number(metrics.get("mean_return")):
        issues.append("metrics.mean_return must be a finite number")
    if isinstance(metrics, dict):
        issues.extend(f"nonfinite {path}" for path in _nonfinite_paths(metrics))
    if "reward_type" in record and (not isinstance(record["reward_type"], str) or record["reward_type"] not in REWARD_TYPES):
        issues.append("reward_type must be private, integrated, or cooperative")
    evidence = {"requires_training", "training_steps_requested", "training_steps_completed"}
    if evidence.intersection(record):
        if not isinstance(record.get("requires_training"), bool):
            issues.append("requires_training must be a boolean when training evidence is present")
        for key in evidence - {"requires_training"}:
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                issues.append(f"{key} must be a nonnegative integer")
        requested = record.get("training_steps_requested")
        completed = record.get("training_steps_completed")
        if record.get("requires_training") is True:
            if not is_finite_number(requested) or requested <= 0:
                issues.append("learner training_steps_requested must be positive")
            if is_finite_number(requested) and is_finite_number(completed) and completed < requested:
                issues.append("training budget was not completed")
    return issues


def is_successful_result(record):
    """Whether the record supplies valid, finite successful-result evidence."""
    return not inspect_result(record)


def record_kind(record):
    """Distinguish explicit progress rows from results without inventing data."""
    if not isinstance(record, dict):
        raise RecordError("record must be a JSON object")
    for name in ("record_type", "type", "event"):
        if name in record and not isinstance(record[name], str):
            raise RecordError(f"{name} must be a string when present")
    marker = record.get("record_type", record.get("type", record.get("event")))
    result_shaped = "metrics" in record or "status" in record or "training_seed" in record
    if isinstance(marker, str) and marker in {"progress", "training_progress", "progress_log"}:
        if record.get("status") == "success" and isinstance(record.get("metrics"), dict) and "mean_return" in record["metrics"]:
            raise RecordError("progress row also claims to be a completed result")
        return "progress"
    if "status" not in record and not (isinstance(record.get("metrics"), dict) and "mean_return" in record["metrics"]) and any(key in record for key in ("step", "timestep", "timesteps")) and any(key in record for key in ("algorithm", "environment", "progress_id")):
        return "progress"
    if result_shaped:
        return "result"
    raise RecordError("unrecognized record; expected a result or explicitly identified progress row")


def read_records(path):
    """Read a file or recursively read JSON/JSONL files, retaining line sources."""
    path = Path(path)
    if not path.exists():
        raise RecordError(f"input does not exist: {path}")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl"})
    if not files:
        raise RecordError(f"no JSON or JSONL files found: {path}")
    records = []
    for source in files:
        try:
            if source.suffix.lower() == ".jsonl":
                with source.open(encoding="utf-8") as handle:
                    items = [(index, json.loads(line)) for index, line in enumerate(handle, 1) if line.strip()]
            else:
                payload = json.loads(source.read_text(encoding="utf-8"))
                items = list(enumerate(payload, 1)) if isinstance(payload, list) else [(None, payload)]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RecordError(f"cannot read {source}: {exc}") from exc
        for line, record in items:
            locator = f"{source}:{line}" if line is not None else str(source)
            if not isinstance(record, dict):
                raise RecordError(f"{locator}: record must be an object")
            if source.name in {"campaign.json", "sensitivity.json"} and "campaign_id" in record and isinstance(record.get("configuration"), dict):
                raise RecordError(f"{source}: campaign container metadata is not a result dataset; "
                                  "select OUTPUT/raw (or OUTPUT/REWARD_TYPE/raw for sensitivity) as the input")
            if any(str(key).startswith("_") for key in record):
                raise RecordError(f"{locator}: underscore-prefixed keys are reserved for reader provenance")
            records.append(dict(record, _source=locator))
    return records


# These describe execution or outcomes, not the scientific configuration.
_RUN_FIELDS = {
    "algorithm", "environment", "training_seed", "seed", "status", "metrics",
    "timestamp", "gpu_id", "worker_id", "hostname", "error_message",
    "training_time_seconds", "evaluation_time_seconds", "elapsed_seconds",
    "training_steps_completed", "record_type", "type", "event", "reward_type",
}


def comparison_context(record):
    """Preserve all non-output configuration, including unfamiliar future fields."""
    return {key: value for key, value in record.items()
            if key not in _RUN_FIELDS and not key.startswith("_")}


def _frozen(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_identity(record):
    """Full cell identity; configuration variants are never silently collapsed."""
    return (record.get("algorithm"), record.get("environment"), record.get("training_seed"),
            record.get("reward_type"), _frozen(comparison_context(record)))


def duplicate_issues(records):
    seen, issues = {}, []
    for record in records:
        try:
            identity = record_identity(record)
            hash(identity)
        except (TypeError, ValueError) as exc:
            issues.append(f"{record.get('_source', '<record>')}: invalid configuration: {exc}")
            continue
        if identity in seen:
            issues.append(f"duplicate experimental cell: {seen[identity]} and {record.get('_source', '<record>')}; select the intended run explicitly")
        else:
            seen[identity] = record.get("_source", "<record>")
    return issues


def ensure_comparable(records):
    """Require one treatment and one configuration per algorithm/environment."""
    issues = duplicate_issues(records)
    if issues:
        raise RecordError("; ".join(dict.fromkeys(issues)))
    modes = {r.get("reward_type") for r in records}
    if None in modes:
        issues.append("missing reward_type; supply --reward-type explicitly for legacy input")
    if len(modes) > 1:
        issues.append("mixed reward treatments; select one treatment or use reward-ablation")
    contexts, cells = {}, set()
    for record in records:
        pair = (record["algorithm"], record["environment"])
        context = _frozen(comparison_context(record))
        if pair in contexts and contexts[pair] != context:
            issues.append(f"multiple configurations/campaigns for {pair}; select a homogeneous input")
        contexts[pair] = context
        cell = (*pair, record["training_seed"])
        if cell in cells:
            issues.append(f"repeated algorithm/environment/seed {cell}; no rerun selection is implicit")
        cells.add(cell)
    if issues:
        raise RecordError("; ".join(dict.fromkeys(issues)))


def load_records(path, *, reward_type=None, seeds: Optional[Iterable[int]] = None,
                 successful_only=True, comparable=True):
    """Load native records; optional treatment supplies *missing* legacy metadata.

    An explicit treatment also selects matching labelled records. It never
    relabels an incompatible result. Any record with missing treatment receives
    ``_reward_type_assumption`` so the caller can disclose the supplied context.
    """
    if reward_type is not None and (not isinstance(reward_type, str) or reward_type not in REWARD_TYPES):
        raise RecordError(f"unknown reward_type: {reward_type!r}")
    if seeds is None:
        seed_set = None
    else:
        try:
            seed_values = list(seeds)
        except TypeError as exc:
            raise RecordError("selected seeds must be an iterable of nonnegative integers") from exc
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seed_values):
            raise RecordError("selected seeds must be nonnegative integers")
        seed_set = set(seed_values)
    result, excluded, progress = [], [], 0
    for raw in read_records(path):
        try:
            kind = record_kind(raw)
        except RecordError as exc:
            raise RecordError(f"{raw['_source']}: {exc}") from exc
        if kind == "progress":
            progress += 1
            continue
        record = dict(raw)
        # Validate identity before selection so malformed metadata is never
        # mistaken for a different treatment/fold and silently discarded.
        issues = inspect_result(record)
        identity_issues = [issue for issue in issues if issue.startswith(("algorithm ", "environment ", "training_seed ", "seed and ", "reward_type "))]
        if identity_issues:
            raise RecordError(f"{record['_source']}: {'; '.join(identity_issues)}")
        if reward_type is not None:
            if "reward_type" not in record:
                record["reward_type"] = reward_type
                record["_reward_type_assumption"] = "explicit caller option"
            elif record["reward_type"] != reward_type:
                continue
        if seed_set is not None and record["training_seed"] not in seed_set:
            continue
        if record.get("status") == "failed" and successful_only:
            excluded.append((record["_source"], issues))
            continue
        if issues and successful_only:
            # A failed outcome or nonfinite metric is an observed exclusion;
            # malformed scientific identity is not silently discarded.
            structural = [issue for issue in issues if not issue.startswith(("status is", "nonfinite "))
                          and issue != "metrics.mean_return must be a finite number"]
            if structural or not isinstance(record.get("metrics"), dict) or "mean_return" not in record["metrics"]:
                raise RecordError(f"{record['_source']}: {'; '.join(issues)}")
            excluded.append((record["_source"], issues))
            continue
        result.append(record)
    if not result:
        raise RecordError(f"no eligible result records in {path} (progress={progress}, excluded={len(excluded)})")
    if excluded:
        import warnings
        warnings.warn(f"excluded {len(excluded)} failed/nonfinite result records: " + "; ".join(source for source, _ in excluded[:5]), RuntimeWarning)
    if comparable:
        ensure_comparable(result)
    return result


def verify_manifest(data_dir, manifest):
    """Verify an explicit CSV manifest of relative paths, row counts and hashes.

    Accepted path columns: path, file, filename, shard. Count columns:
    record_count, records, row_count, rows, n_records. Hash columns: md5/sha256.
    Counts refer to physical JSON objects, including progress rows. No historic
    corpus size or expected NaN count is built into this check.
    """
    root = Path(data_dir).resolve()
    if root.is_file():
        root = root.parent
    issues = []
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ["manifest contains no rows"]
    seen = set()
    for index, row in enumerate(rows, 2):
        name = next((row.get(key) for key in ("path", "file", "filename", "shard") if row.get(key)), None)
        if not name:
            issues.append(f"manifest row {index}: missing path/file/filename/shard")
            continue
        source = (root / name).resolve()
        if root not in source.parents:
            issues.append(f"manifest row {index}: path escapes dataset root")
            continue
        if source in seen:
            issues.append(f"manifest row {index}: duplicate path {name}")
        seen.add(source)
        if not source.is_file():
            issues.append(f"manifest file missing: {name}")
            continue
        for algorithm in ("md5", "sha256"):
            expected = (row.get(algorithm) or "").strip().lower()
            if expected:
                actual = hashlib.new(algorithm, source.read_bytes()).hexdigest()
                if actual != expected:
                    issues.append(f"{name}: {algorithm} mismatch")
        count = next((row.get(key) for key in ("record_count", "records", "row_count", "rows", "n_records") if row.get(key)), None)
        if count is not None:
            try:
                actual = len(read_records(source))
                if actual != int(count):
                    issues.append(f"{name}: record count {actual} != {count}")
            except (ValueError, OSError) as exc:
                issues.append(f"{name}: cannot verify count: {exc}")
    return issues
