"""Gymnasium registration and portable preflight regressions."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import coopetition_gym as cg


@pytest.mark.parametrize("env_id", cg.list_environments())
def test_gymnasium_factory_preserves_multi_agent_contract(env_id):
    env = gym.make(f"coopetition_gym:{env_id}", reward_type="private", max_steps=2)
    try:
        assert env.spec.disable_env_checker
        assert env.unwrapped.reward_type == "private"
        obs, info = env.reset(seed=41)
        assert isinstance(info, dict)
        assert env.observation_space.contains(obs)
        action = env.unwrapped.endowments * 0.6
        _, rewards, terminated, truncated, _ = env.step(action)
        assert rewards.shape == (env.unwrapped.n_agents,)
        assert np.all(np.isfinite(rewards))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_plugin_entry_point_loads_and_registration_is_idempotent():
    entry_point = importlib.metadata.EntryPoint(
        name="__root__", group="gymnasium.envs",
        value="coopetition_gym:_register_gymnasium_envs",
    )
    register = entry_point.load()
    original = {env_id: gym.spec(env_id) for env_id in cg.list_environments()}
    register()
    register()
    for env_id, spec in original.items():
        assert gym.spec(env_id) is spec


def test_explicit_module_factory_works_in_a_fresh_process(tmp_path):
    # Keep the dependency paths of this interpreter while excluding startup
    # hooks. The child deliberately does not import coopetition_gym itself.
    code = f"sys.path[:] = {sys.path!r}\n" + """
import gymnasium as gym
import numpy as np
with gym.make("coopetition_gym:TrustDilemma-v0") as env:
    env.reset(seed=41)
    rewards = env.step(np.array([50.0, 50.0], dtype=np.float32))[1]
    assert rewards.shape == (2,)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-c", "import sys\n" + code],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("package_root", [False, True])
def test_preflight_resolves_source_tests_and_uses_current_python(tmp_path, monkeypatch, package_root):
    from extensions.slcd_2d import pre_launch_check_tier1 as preflight

    source_package = tmp_path / "coopetition_gym"
    tests = source_package / "extensions" / "slcd_2d" / "tests"
    tests.mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="38 passed\n", stderr="")

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    preflight.gate_pytest(source_package if package_root else tmp_path)
    command, kwargs = calls[0]
    assert command[0] == sys.executable
    assert command[-2:] == [str(tests), "-q"]
    assert kwargs["cwd"] == str(source_package)


def test_installed_preflight_requires_source_checkout_for_pytest(capsys):
    from extensions.slcd_2d import pre_launch_check_tier1 as preflight

    with pytest.raises(SystemExit) as exc:
        preflight.gate_pytest(None)
    assert exc.value.code == 1
    assert "--repo-root" in capsys.readouterr().out
