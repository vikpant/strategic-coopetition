"""Tier 1.5 pre-flight check.

Extends Tier 1 with gates specific to the additional stages:
  11. MADDPG, MATD3, MASAC constructible on 2D env
  12. calibrate module importable and synthetic objective passes unit invariants

Runs Tier 1 gates first (delegates), then adds these.
"""

from __future__ import annotations

import argparse
import sys

from extensions.slcd_2d.pre_launch_check_tier1 import (
    gate_2d_env,
    gate_cuda,
    gate_extension_importable,
    gate_ippo_trains,
    gate_oracle,
    gate_packages,
    gate_pytest,
    gate_python,
    gate_reward_type_routing,
    gate_v1_env,
)


def _ok(name: str, detail: str = "") -> None:
    print(f"[OK] {name}{': ' + detail if detail else ''}")


def _fail(name: str, detail: str) -> None:
    print(f"[FAIL] {name}: {detail}")
    sys.exit(1)


def gate_tier15_algorithms() -> None:
    sys.path.insert(0, "/home/vik_p/projects/strategic-coopetition")
    from extensions.slcd_2d import SLCDAppropriationEnv
    from extensions.slcd_2d.algorithms import build_algorithm, list_algorithms

    required = {"MADDPG", "MATD3", "MASAC", "MAPPO"}
    missing = required - set(list_algorithms())
    if missing:
        _fail("tier15 algorithms in registry", f"missing {missing}")

    env = SLCDAppropriationEnv(max_steps=40)
    for name in sorted(required):
        try:
            algo = build_algorithm(name, env, device="cpu", seed=0)
            if not (hasattr(algo, "predict") or hasattr(algo, "train")):
                _fail(f"{name} interface", "missing predict/train")
        except Exception as e:
            _fail(f"{name} construction", str(e))
    _ok("tier15 algorithms constructible")


def gate_calibrate_module() -> None:
    import numpy as np
    from extensions.slcd_2d.calibrate import (
        DEFAULT_TARGETS, _parabolic_vertex, default_objective,
    )
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = (xs - 2.3) ** 2 + 0.1
    v = _parabolic_vertex(xs, ys, bounds=(0.0, 4.0))
    if abs(v - 2.3) > 0.1:
        _fail("parabolic_vertex", f"got {v}, expected ~2.3")
    zero = default_objective({
        "eval_final_trust_mean": DEFAULT_TARGETS["final_trust_mean"],
        "eval_final_appropriation_mean": [DEFAULT_TARGETS["final_appropriation_mean"]],
    })
    if zero > 1e-9:
        _fail("default_objective", f"not zero at targets, got {zero}")
    _ok("calibrate module invariants")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tier 1.5 pre-flight")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--skip-pytest", action="store_true")
    args = p.parse_args(argv)

    # Tier 1 gates (delegated)
    print("--- gate 1/python ---"); gate_python()
    print("--- gate 2/packages ---"); gate_packages()
    print("--- gate 3/extension_import ---"); gate_extension_importable()
    if not args.skip_pytest:
        from pathlib import Path
        print("--- gate 4/pytest ---")
        gate_pytest(Path("/home/vik_p/projects/strategic-coopetition"))
    print("--- gate 5/v1_env ---"); gate_v1_env()
    print("--- gate 6/2d_env ---"); gate_2d_env()
    print("--- gate 7/reward_routing ---"); gate_reward_type_routing()
    print("--- gate 8/ippo_trains ---"); gate_ippo_trains()
    print("--- gate 9/oracle ---"); gate_oracle()
    print("--- gate 10/cuda ---"); gate_cuda(args.require_gpu)

    # Tier 1.5 additions
    print("--- gate 11/tier15_algorithms ---"); gate_tier15_algorithms()
    print("--- gate 12/calibrate ---"); gate_calibrate_module()

    print("\nAll Tier 1.5 gates passed. Cleared for activation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
