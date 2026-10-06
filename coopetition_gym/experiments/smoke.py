"""Small real-environment check of result writing, validation and analysis.

This runs random policies without training. Its outputs are installation test
artifacts, not reproductions of scientific benchmark scores.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import coopetition_gym as cg

from . import analyze, config, validate
from .records import is_successful_result


def run(output: Path):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Smoke output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for mode in config.REWARD_TYPES:
        raw = output / mode / "raw"
        raw.mkdir(parents=True)
        records = []
        for seed in (99, 100):
            env = cg.make("TrustDilemma-v0", reward_type=mode, max_steps=4)
            try:
                env.reset(seed=seed)
                env.action_space.seed(seed)
                total, steps = 0.0, 0
                done = False
                while not done:
                    _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                    total += float(np.sum(reward))
                    steps += 1
                    done = terminated or truncated
                record = {
                    "algorithm": "Random", "environment": "TrustDilemma-v0",
                    "training_seed": seed, "reward_type": mode, "status": "success",
                    "requires_training": False, "training_steps_requested": 0,
                    "training_steps_completed": 0, "source_version": cg.__version__,
                    "evaluation_config": {"episodes": 1, "max_steps": 4},
                    "purpose": "installation-smoke",
                    "metrics": {"mean_return": total, "mean_episode_length": steps},
                }
                assert is_successful_result(record)
                records.append(record)
            finally:
                env.close()
        (raw / "results.jsonl").write_text(
            "\n".join(json.dumps(row, allow_nan=False) for row in records) + "\n",
            encoding="utf-8",
        )
        assert validate.main(["training", str(raw), "--expected-records", "2"]) == 0
        report = output / mode / "returns.csv"
        assert analyze.main(["returns-summary", "--input-dir", str(raw),
                             "--output", str(report)]) == 0
        with report.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1 and int(rows[0]["n_seeds"]) == 2
        assert rows[0]["reward_type"] == mode
        assert np.isclose(float(rows[0]["mean_return"]), np.mean([r["metrics"]["mean_return"] for r in records]), atol=5e-5, rtol=0)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke"))
    args = parser.parse_args(argv)
    print(f"Smoke check passed: {run(args.output)} (6 random-policy runs, no training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
