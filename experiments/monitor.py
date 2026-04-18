"""Local-friendly progress, health, and disk monitor.

This module provides the local-machine subset of the monitoring
infrastructure used during the original campaign. The cloud-orchestration
pieces of ``background_monitor.py`` (SSH polling of Vast.ai instances,
remote snapshot creation, per-instance relaunch functions) are campaign-
specific and have been archived; they have no reuse value outside that
deployment.

Subcommands::

    watch              Live progress display for a running campaign.
                       Polls the output directory every few seconds and
                       reports completion counts, success rate, and ETA.

    clean-checkpoints  Checkpoint rotation. For each unique
                       ``(algorithm, environment, seed)`` combination in
                       the checkpoint directory, keeps only the checkpoint
                       with the highest step number and deletes older
                       ones. Safe to run while training is in progress.

    disk-status        Report disk usage for a given path. Exits non-zero
                       if usage exceeds the configured threshold
                       (``experiments.config.SAFETY.disk_pressure_threshold``).
                       Intended for cron-like invocation.

    health-check       Scan a campaign output directory and classify
                       experiments by status (success, failed, in-progress,
                       stalled). Useful after an interrupted campaign to
                       decide what needs to be re-run.

All subcommands exit quickly and are safe to invoke from a cron job or a
wrapper shell script. No subcommand polls remote instances, opens SSH
connections, or modifies data outside the specified input directory.

Usage::

    # Watch a running campaign
    python -m experiments.monitor watch data/training/baseline_integrated/

    # Rotate old checkpoints (keeps only the latest per experiment)
    python -m experiments.monitor clean-checkpoints data/checkpoints/

    # Exit 1 if the output disk is above the 80% threshold
    python -m experiments.monitor disk-status data/training/

    # Health check after an interrupted campaign
    python -m experiments.monitor health-check data/training/baseline_integrated/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from experiments import config


logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint rotation
# =============================================================================

#: Regex matching checkpoint files: ``{ALGO}_{ENV}_{SEED}_step_{NNNNNN}.pt``.
#: Environments are matched against the known list from
#: :data:`experiments.config.ENVIRONMENT_BY_ID` to avoid false matches on
#: algorithm names that contain underscores.
_CKPT_ENV_PATTERN = "|".join(re.escape(e.id) for e in config.ALL_ENVIRONMENTS)
_CKPT_RE = re.compile(
    rf"^(?P<algo>.+)_(?P<env>{_CKPT_ENV_PATTERN})_(?P<seed>\d+)_step_(?P<step>\d+)\.pt$"
)


def parse_checkpoint_name(filename: str) -> Optional[Dict[str, object]]:
    """Parse a checkpoint filename into (algorithm, environment, seed, step).

    Returns ``None`` if the filename does not match the expected pattern.
    The environment component must be one of the 20 registered environments.
    """
    m = _CKPT_RE.match(filename)
    if m is None:
        return None
    return {
        "algorithm": m.group("algo"),
        "environment": m.group("env"),
        "seed": int(m.group("seed")),
        "step": int(m.group("step")),
        "filename": filename,
    }


def rotate_checkpoints(checkpoint_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    """Delete all but the latest checkpoint per ``(algo, env, seed)`` triple.

    Args:
        checkpoint_dir: Directory containing ``*.pt`` checkpoint files.
        dry_run: If True, report what would be deleted without deleting.

    Returns:
        A dict with counts: ``{"kept": K, "deleted": D, "bytes_freed": B}``.

    Checkpoint filename format: ``{algo}_{env}_{seed}_step_{step}.pt``.
    Files that do not match are left untouched.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    by_experiment: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    skipped = 0
    for path in checkpoint_dir.glob("*.pt"):
        parsed = parse_checkpoint_name(path.name)
        if parsed is None:
            skipped += 1
            continue
        parsed["path"] = path
        by_experiment[(parsed["algorithm"], parsed["environment"], parsed["seed"])].append(parsed)

    kept = 0
    deleted = 0
    bytes_freed = 0
    for key, checkpoints in by_experiment.items():
        checkpoints.sort(key=lambda c: c["step"])
        latest = checkpoints[-1]
        kept += 1
        for old in checkpoints[:-1]:
            size = old["path"].stat().st_size
            bytes_freed += size
            if dry_run:
                logger.info(f"  [dry-run] would delete {old['filename']} ({size / 1e6:.1f} MB)")
            else:
                old["path"].unlink()
                logger.info(f"  deleted {old['filename']} ({size / 1e6:.1f} MB)")
            deleted += 1

    if skipped:
        logger.info(f"  {skipped} files did not match checkpoint pattern; left untouched")

    return {"kept": kept, "deleted": deleted, "bytes_freed": bytes_freed}


# =============================================================================
# Disk pressure
# =============================================================================

def disk_usage(path: Path) -> Dict[str, float]:
    """Return disk usage statistics for the filesystem containing ``path``.

    Keys: ``total_gb``, ``used_gb``, ``free_gb``, ``used_fraction``.
    """
    usage = shutil.disk_usage(str(path))
    return {
        "total_gb": usage.total / 1e9,
        "used_gb": usage.used / 1e9,
        "free_gb": usage.free / 1e9,
        "used_fraction": usage.used / usage.total if usage.total else 0.0,
    }


def check_disk_pressure(
    path: Path,
    threshold: float = config.SAFETY.disk_pressure_threshold,
) -> bool:
    """Return True if the disk containing ``path`` is above the threshold.

    The threshold defaults to :attr:`experiments.config.SafetyConfig.disk_pressure_threshold`.
    """
    stats = disk_usage(path)
    return stats["used_fraction"] >= threshold


# =============================================================================
# Health check
# =============================================================================

def classify_experiments(
    output_dir: Path,
    stall_threshold_seconds: float = 1800.0,
) -> Dict[str, List[Path]]:
    """Classify result files in ``output_dir`` by status.

    Buckets:

    * ``success`` — ``status == "success"``
    * ``failed`` — ``status == "failed"``
    * ``in_progress`` — ``status`` missing/other, modified within
      ``stall_threshold_seconds``
    * ``stalled`` — ``status`` missing/other, not modified within
      ``stall_threshold_seconds``

    Returns a dict mapping each bucket name to the list of paths.
    """
    buckets: Dict[str, List[Path]] = {
        "success": [],
        "failed": [],
        "in_progress": [],
        "stalled": [],
    }
    now = time.time()

    for path in sorted(output_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            buckets["failed"].append(path)
            continue

        status = data.get("status")
        if status == "success":
            buckets["success"].append(path)
        elif status == "failed":
            buckets["failed"].append(path)
        else:
            age = now - path.stat().st_mtime
            if age < stall_threshold_seconds:
                buckets["in_progress"].append(path)
            else:
                buckets["stalled"].append(path)
    return buckets


def summarize_health(buckets: Dict[str, List[Path]]) -> str:
    """Render a short health-check summary as a multi-line string."""
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return "No result files found."
    lines = [f"Total files: {total}"]
    for bucket in ("success", "failed", "in_progress", "stalled"):
        pct = len(buckets[bucket]) / total * 100
        lines.append(f"  {bucket:12s}: {len(buckets[bucket]):6d}  ({pct:5.1f}%)")
    return "\n".join(lines)


# =============================================================================
# Live progress watch
# =============================================================================

def count_results(output_dir: Path) -> Tuple[int, int]:
    """Return ``(total_files, success_files)`` for a campaign directory.

    Scans recursively for ``*.json`` and classifies each by ``status``.
    """
    total = 0
    success = 0
    for path in output_dir.rglob("*.json"):
        total += 1
        try:
            data = json.loads(path.read_text())
            if data.get("status") == "success":
                success += 1
        except (json.JSONDecodeError, OSError):
            pass
    return total, success


def watch_campaign(
    output_dir: Path,
    interval_seconds: float = 10.0,
    target_count: Optional[int] = None,
) -> None:
    """Display live progress of a running campaign.

    Polls ``output_dir`` every ``interval_seconds`` and prints the current
    file count, success rate, rate of change (files per minute), and ETA to
    the target count if provided.

    Run until Ctrl-C. Safe to leave running during a long campaign; does
    not interfere with the orchestrator.

    Args:
        output_dir: The campaign's top-level output directory.
        interval_seconds: Seconds between polls.
        target_count: Expected total file count. Used to compute ETA.
            If ``None``, ETA is not shown.
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    start_time = time.time()
    start_total, _ = count_results(output_dir)
    last_total = start_total
    last_time = start_time

    print(f"Watching {output_dir} (initial count: {start_total:,})")
    if target_count:
        print(f"Target: {target_count:,} files")
    print("Press Ctrl-C to stop.\n")

    try:
        while True:
            time.sleep(interval_seconds)
            total, success = count_results(output_dir)
            now = time.time()

            delta = total - last_total
            dt = now - last_time
            rate_per_min = delta / dt * 60 if dt > 0 else 0

            success_rate = success / total * 100 if total else 0

            eta_str = ""
            if target_count and rate_per_min > 0 and total < target_count:
                remaining = target_count - total
                eta_seconds = remaining / (rate_per_min / 60)
                hours = int(eta_seconds / 3600)
                minutes = int((eta_seconds % 3600) / 60)
                eta_str = f"  ETA: {hours}h{minutes:02d}m"

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"total: {total:6,}  "
                f"success: {success:6,} ({success_rate:5.1f}%)  "
                f"+{delta:4d} / {dt:.0f}s  "
                f"({rate_per_min:5.1f}/min){eta_str}"
            )
            last_total = total
            last_time = now
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        total_delta = last_total - start_total
        print(f"\nStopped after {elapsed / 60:.1f} minutes.")
        print(f"Completed: {total_delta:,} new files ({total_delta / (elapsed / 60):.1f}/min average)")


# =============================================================================
# CLI
# =============================================================================

def _cmd_watch(args: argparse.Namespace) -> int:
    watch_campaign(
        output_dir=args.output_dir,
        interval_seconds=args.interval,
        target_count=args.target,
    )
    return 0


def _cmd_clean_checkpoints(args: argparse.Namespace) -> int:
    stats = rotate_checkpoints(args.checkpoint_dir, dry_run=args.dry_run)
    print(
        f"{'DRY-RUN: ' if args.dry_run else ''}"
        f"kept {stats['kept']} latest, "
        f"{'would delete' if args.dry_run else 'deleted'} {stats['deleted']} old, "
        f"{'would free' if args.dry_run else 'freed'} {stats['bytes_freed'] / 1e9:.2f} GB"
    )
    return 0


def _cmd_disk_status(args: argparse.Namespace) -> int:
    stats = disk_usage(args.path)
    pct = stats["used_fraction"] * 100
    threshold_pct = args.threshold * 100
    print(
        f"Disk at {args.path}:\n"
        f"  total:    {stats['total_gb']:8.1f} GB\n"
        f"  used:     {stats['used_gb']:8.1f} GB  ({pct:5.1f}%)\n"
        f"  free:     {stats['free_gb']:8.1f} GB\n"
        f"  threshold:{threshold_pct:5.1f}%"
    )
    if stats["used_fraction"] >= args.threshold:
        print(f"ALERT: disk usage exceeds {threshold_pct:.0f}% threshold.")
        return 1
    return 0


def _cmd_health_check(args: argparse.Namespace) -> int:
    buckets = classify_experiments(args.output_dir, stall_threshold_seconds=args.stall_threshold)
    print(summarize_health(buckets))
    if args.list_stalled and buckets["stalled"]:
        print("\nStalled experiments:")
        for path in buckets["stalled"][:20]:
            print(f"  {path}")
        if len(buckets["stalled"]) > 20:
            print(f"  ... and {len(buckets['stalled']) - 20} more")
    # Non-zero exit if there are stalled or failed experiments, so the
    # caller (typically a CI job) can treat this as an actionable error.
    if buckets["stalled"] or buckets["failed"]:
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local progress, health, and disk monitor for Coopetition-Gym "
                    "campaigns. Run one of the subcommands below.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- watch
    wp = sub.add_parser("watch", help="Live progress display.")
    wp.add_argument("output_dir", type=Path, help="Campaign output directory.")
    wp.add_argument("--interval", type=float, default=10.0,
                    help="Seconds between polls (default: 10).")
    wp.add_argument("--target", type=int, default=None,
                    help="Expected total file count for ETA calculation.")
    wp.set_defaults(func=_cmd_watch)

    # -- clean-checkpoints
    cp = sub.add_parser("clean-checkpoints",
                        help="Keep only the latest checkpoint per experiment.")
    cp.add_argument("checkpoint_dir", type=Path,
                    help="Directory containing .pt checkpoint files.")
    cp.add_argument("--dry-run", action="store_true",
                    help="Report what would be deleted without deleting.")
    cp.set_defaults(func=_cmd_clean_checkpoints)

    # -- disk-status
    dp = sub.add_parser("disk-status",
                        help="Report disk usage, exit 1 if above threshold.")
    dp.add_argument("path", type=Path,
                    help="Path whose disk should be inspected.")
    dp.add_argument("--threshold", type=float,
                    default=config.SAFETY.disk_pressure_threshold,
                    help="Fraction threshold (default: 0.80).")
    dp.set_defaults(func=_cmd_disk_status)

    # -- health-check
    hp = sub.add_parser("health-check",
                        help="Classify experiments by status.")
    hp.add_argument("output_dir", type=Path,
                    help="Campaign output directory.")
    hp.add_argument("--stall-threshold", type=float, default=1800.0,
                    help="Seconds of inactivity before marking stalled "
                         "(default: 1800 = 30 min).")
    hp.add_argument("--list-stalled", action="store_true",
                    help="Print paths of stalled experiments.")
    hp.set_defaults(func=_cmd_health_check)

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
