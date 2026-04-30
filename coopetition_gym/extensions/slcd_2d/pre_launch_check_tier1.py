"""Pre-flight verification for Tier 1 cloud activation.

Run on the remote Vast.ai instance before starting the campaign. Each check
is independent and the script exits non-zero on the first failure with a
clear diagnostic, matching the gate protocol used for the main campaign.

Gates
-----
1. Python version >= 3.10
2. Required packages importable (coopetition_gym, SB3, torch, scipy, gymnasium)
3. `extensions.slcd_2d` importable
4. Pytest passes on the extension test suite
5. v1 SLCD env usable (smoke make + step)
6. SLCDAppropriationEnv reset + step returns (15,) obs and (2,) reward
7. Reward-type routing honors COOPETITION_REWARD_TYPE
8. SB3 IPPO constructs + trains 128 steps
9. Oracle_Appropriation converges and returns valid action
10. CUDA visible and usable if --require-gpu
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path("/home/vik_p/projects/strategic-coopetition")


def _ok(name: str, detail: str = "") -> None:
    print(f"[OK] {name}{': ' + detail if detail else ''}")


def _fail(name: str, detail: str) -> None:
    print(f"[FAIL] {name}: {detail}")
    sys.exit(1)


def gate_python() -> None:
    if sys.version_info < (3, 10):
        _fail("python>=3.10", f"got {sys.version_info[:3]}")
    _ok("python>=3.10", f"{'.'.join(map(str, sys.version_info[:3]))}")


def gate_packages() -> None:
    required = ["numpy", "scipy", "gymnasium", "coopetition_gym",
                "stable_baselines3", "torch"]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError as e:
            missing.append(f"{pkg}: {e}")
    if missing:
        _fail("required packages", "; ".join(missing))
    _ok("required packages")


def gate_extension_importable() -> None:
    try:
        import extensions.slcd_2d  # noqa: F401
        from extensions.slcd_2d import SLCDAppropriationEnv  # noqa: F401
        from extensions.slcd_2d.algorithms import list_algorithms  # noqa: F401
    except Exception as e:
        _fail("import extensions.slcd_2d", traceback.format_exc())
    _ok("import extensions.slcd_2d")


def gate_pytest(repo_root: Path) -> None:
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "extensions/slcd_2d/tests/", "-q"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        _fail("pytest", str(e))
    if result.returncode != 0:
        _fail("pytest", result.stdout[-800:] + result.stderr[-800:])
    _ok("pytest", result.stdout.splitlines()[-1].strip())


def gate_v1_env() -> None:
    from coopetition_gym.envs import make
    env = make("SLCD-v0")
    obs, _ = env.reset(seed=0)
    import numpy as np
    obs2, r, t, tr, _ = env.step(np.array([50.0, 50.0], dtype=np.float32))
    _ok("v1 SLCD-v0 step", f"obs.shape={obs.shape}, r.shape={r.shape}")


def gate_2d_env() -> None:
    import numpy as np
    from extensions.slcd_2d import SLCDAppropriationEnv
    env = SLCDAppropriationEnv(max_steps=40)
    obs, _ = env.reset(seed=0)
    _, r, _, _, info = env.step(np.array([50.0, 0.3, 50.0, 0.3], dtype=np.float32))
    assert obs.shape == (15,), f"obs shape {obs.shape}"
    assert r.shape == (2,), f"reward shape {r.shape}"
    assert "appropriation" in info, "info missing appropriation"
    _ok("2D env step", f"obs.shape={obs.shape}, r.shape={r.shape}")


def gate_reward_type_routing() -> None:
    from extensions.slcd_2d import SLCDAppropriationEnv
    for rt in ("integrated", "private", "cooperative"):
        os.environ["COOPETITION_REWARD_TYPE"] = rt
        env = SLCDAppropriationEnv()
        assert env.reward_type == rt, f"reward_type env-var routing broke for {rt}"
    del os.environ["COOPETITION_REWARD_TYPE"]
    _ok("COOPETITION_REWARD_TYPE routing")


def gate_ippo_trains() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.algorithms import IndependentPPO
    from extensions.slcd_2d import SLCDAppropriationEnv
    env = SLCDAppropriationEnv(max_steps=40)
    algo = IndependentPPO(env, device="cpu", seed=0, n_steps=64, batch_size=32)
    algo.train(total_timesteps=128)
    _ok("IPPO trains 128 steps on 2D env")


def gate_oracle() -> None:
    from extensions.slcd_2d import AppropriationOracle, SLCDAppropriationEnv
    env = SLCDAppropriationEnv(max_steps=40)
    oracle = AppropriationOracle(env)
    assert oracle.equilibrium.converged, "Oracle did not converge"
    action, _ = oracle.predict(obs=None, deterministic=True)
    assert env.action_space.contains(action), f"oracle action out of bounds: {action}"
    _ok("Oracle_Appropriation converges", f"iters={oracle.equilibrium.iterations}")


def gate_cuda(required: bool) -> None:
    try:
        import torch
    except ImportError:
        if required:
            _fail("torch cuda", "torch not installed")
        _ok("torch cuda (skipped, not required)")
        return
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        _ok("cuda", f"{n} device(s); 0={torch.cuda.get_device_name(0)}")
    elif required:
        _fail("cuda", "required but not available")
    else:
        _ok("cuda (skipped, not required)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tier 1 pre-flight check")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--skip-pytest", action="store_true",
                   help="Skip gate 4 (runs the test suite). Useful for quick checks.")
    args = p.parse_args(argv)

    checks: List[Tuple[str, Callable[[], None]]] = [
        ("1/python", gate_python),
        ("2/packages", gate_packages),
        ("3/extension_import", gate_extension_importable),
    ]
    if not args.skip_pytest:
        checks.append(("4/pytest", lambda: gate_pytest(args.repo_root)))
    checks.extend([
        ("5/v1_env", gate_v1_env),
        ("6/2d_env", gate_2d_env),
        ("7/reward_routing", gate_reward_type_routing),
        ("8/ippo_trains", gate_ippo_trains),
        ("9/oracle", gate_oracle),
        ("10/cuda", lambda: gate_cuda(args.require_gpu)),
    ])
    for tag, fn in checks:
        print(f"--- gate {tag} ---")
        fn()
    print("\nAll gates passed. Cleared for Tier 1 activation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
