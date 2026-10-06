"""Run from outside the source checkout after installing both wheels."""
from pathlib import Path
from importlib import metadata
import subprocess
import json
import sys
import tempfile

import numpy as np
import gymnasium as gym
import coopetition_gym as cg
import experiments
import slcd_2d


def main():
    checkout = Path(sys.argv[1]).resolve()
    for module in (cg, experiments, slcd_2d):
        location = Path(module.__file__).resolve()
        assert checkout not in location.parents, (module.__name__, location)
        print(f"Installed {module.__name__}: {location}")
    assert metadata.version("coopetition-gym") == cg.__version__ == experiments.__version__
    assert metadata.version("coopetition-gym-slcd2d") == "0.1.1"
    assert len(cg.list_environments()) == 20
    for mode in ("private", "integrated", "cooperative"):
        env = gym.make("coopetition_gym:TrustDilemma-v0", reward_type=mode)
        try:
            assert env.unwrapped.reward_type == mode
            env.reset(seed=42)
            _, reward, _, _, _ = env.step(np.array([60.0, 55.0]))
            assert np.shape(reward) == (2,) and np.isfinite(reward).all()
        finally:
            env.close()
        for make in (cg.make_parallel, cg.make_aec):
            wrapped = make("CoalitionFormation-v0", reward_type=mode)
            try:
                wrapped.reset(seed=42)
                assert len(wrapped.agents) == 6
                assert len(wrapped.base_env._coalition_members) == 6
            finally:
                wrapped.close()
    for module in ("campaign", "audit", "evaluate", "analyze", "validate", "monitor", "sensitivity", "smoke"):
        subprocess.run([sys.executable, "-m", f"experiments.{module}", "--help"], check=True,
                       stdout=subprocess.DEVNULL)
    for module in ("campaign", "pre_launch_check_tier1", "pre_launch_check_tier15"):
        subprocess.run([sys.executable, "-m", f"slcd_2d.{module}", "--help"], check=True,
                       stdout=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory(prefix="coopetition-install-") as tmp:
        subprocess.run([sys.executable, "-m", "experiments.smoke", "--output", str(Path(tmp) / "smoke")], check=True)
        campaign = Path(tmp) / "campaign"
        command = [sys.executable, "-m", "experiments.campaign", "private",
                   "--algorithms", "Random", "--environments", "TrustDilemma-v0",
                   "--seeds", "99,100", "--eval-episodes", "1", "--output", str(campaign),
                   "--max-workers", "1", "--no-monitoring", "--no-ram-monitoring",
                   "--no-thermal-monitoring", "--no-backpressure", "--no-checkpoints"]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        outputs = sorted((campaign / "raw").glob("*.json"))
        assert len(outputs) == 2
        contents = {p.name: p.read_bytes() for p in outputs}
        for path in outputs:
            record = json.loads(path.read_text())
            assert record["reward_type"] == record["actual_reward_type"] == "private"
            assert record["requires_training"] is False
            assert record["training_steps_completed"] == 0
            assert record["status"] == "success" and record["source_fingerprint"]
        subprocess.run(command + ["--resume"], check=True, stdout=subprocess.DEVNULL)
        assert contents == {p.name: p.read_bytes() for p in outputs}
        subprocess.run([sys.executable, "-m", "experiments.validate", "training",
                        str(campaign / "raw"), "--expected-records", "2"], check=True)
        subprocess.run([sys.executable, "-m", "experiments.analyze", "returns-summary",
                        "--input-dir", str(campaign / "raw"), "--output", str(campaign / "returns.csv")], check=True)
        subprocess.run([sys.executable, "-m", "slcd_2d.campaign", "--seeds", "99", "--steps", "4",
                        "--output", str(Path(tmp) / "extension")], check=True)
    print("Installed-wheel verification passed; no learners trained.")


if __name__ == "__main__":
    main()
