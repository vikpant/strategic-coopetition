"""Tier 1.5 extended-algorithm buildability tests.

These verify that MADDPG, MATD3, MASAC, MAPPO construct successfully on the
2D env. We do NOT train them here (that would take minutes); full training
is covered by the smoke campaign.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


@pytest.mark.parametrize("algo_name", ["MADDPG", "MATD3", "MASAC", "MAPPO"])
def test_tier15_algorithm_constructs(algo_name):
    import sys
    sys.path.insert(0, "/home/vik_p/projects/strategic-coopetition")
    from extensions.slcd_2d import SLCDAppropriationEnv
    from extensions.slcd_2d.algorithms import build_algorithm

    env = SLCDAppropriationEnv(max_steps=40)
    algo = build_algorithm(algo_name, env, device="cpu", seed=0)
    assert algo is not None
    assert hasattr(algo, "predict") or hasattr(algo, "train")


def test_all_eight_in_registry():
    from extensions.slcd_2d.algorithms import list_algorithms
    expected = {"IPPO", "ISAC", "IA2C", "MAPPO", "MADDPG", "MATD3", "MASAC",
                "Oracle_Appropriation"}
    assert set(list_algorithms()) == expected
