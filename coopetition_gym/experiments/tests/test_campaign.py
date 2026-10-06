"""Campaign correctness tests using small fake environments and algorithms."""
import json
import sys
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import campaign


class FakeEnv:
    def __init__(self, reward_type="integrated", reward=1.0):
        self.reward_type = reward_type
        self.max_steps = 2
        self.n_agents = 2
        self.unwrapped = self
        self.reward = reward
        self.action_space = SimpleNamespace(shape=(2,))
        self.closed = False
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        return np.zeros(2), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(2), np.full(2, self.reward), False, self.steps >= self.max_steps, {"mean_trust": 0.5}

    def close(self):
        self.closed = True


class FakeLearner:
    def __init__(self, env, **kwargs):
        self.env = env

    def train(self, total_timesteps):
        for _ in range(total_timesteps):
            self.env.step(np.zeros(2))

    def predict(self, observation, deterministic=True):
        return np.zeros(2)


ALGORITHM = {"name": "Fake", "class": "FakeLearner", "requires_training": True, "params": {}}
ENVIRONMENT = {"id": "TrustDilemma-v0", "category": "dyadic", "tr": "tr2", "horizon": 999, "n_agents": 999}


@pytest.fixture
def fake_runtime(monkeypatch):
    environments = []
    def create(env_id, seed=None, reward_type="integrated"):
        env = FakeEnv(reward_type)
        environments.append(env)
        return env, None
    monkeypatch.setattr(campaign, "_setup_path", lambda: None)
    monkeypatch.setattr(campaign, "create_environment", create)
    monkeypatch.setattr(campaign, "get_algorithm_class", lambda config: FakeLearner)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    return environments


def run_fake(**kwargs):
    options = dict(timesteps_override=3, reward_type="private")
    options.update(kwargs)
    return campaign.run_single_experiment(ALGORITHM, ENVIRONMENT, 7, 2, **options)


@pytest.fixture
def make_orchestrator(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "detect_hardware", lambda: {"num_gpus": 0, "num_vcpus": 12})
    monkeypatch.setattr(campaign.signal, "signal", lambda *args: None)
    def make(**kwargs):
        options = dict(modes=["tr2"], output_dir=tmp_path, algorithms=["Random"],
                       environments=["TrustDilemma-v0"], seeds=[7], n_eval_episodes=2,
                       reward_type="private", timesteps_override=3, max_cpu_workers=1,
                       enable_checkpoints=False, enable_ram_monitoring=False,
                       enable_memory_monitoring=False, enable_backpressure=False,
                       enable_nvlink_scheduling=False)
        options.update(kwargs)
        return campaign.UnifiedOrchestrator(campaign.OrchestratorConfig(**options))
    return make



@pytest.mark.parametrize("cpus, expected", [(1, 1), (2, 1), (4, 1), (8, 1), (12, 4), (240, 232)])
def test_cpu_only_worker_limit_keeps_a_worker_available(cpus, expected):
    limits = campaign.compute_safe_worker_limits({"num_gpus": 0, "num_vcpus": cpus})
    assert limits == {"gpu_workers": 0, "cpu_workers": expected,
                      "max_per_gpu": 0, "total_workers": expected}


@pytest.mark.parametrize("cpus", [1, 2, 4, 8, 12, 240])
def test_cpu_only_worker_limit_honors_explicit_single_worker(cpus):
    limits = campaign.compute_safe_worker_limits(
        {"num_gpus": 0, "num_vcpus": cpus}, max_workers=1)
    assert limits["cpu_workers"] == limits["total_workers"] == 1


@pytest.mark.parametrize("max_workers, expected", [(None, 3), (1, 1)])
def test_cpu_only_worker_override_still_respects_total_cap(max_workers, expected):
    limits = campaign.compute_safe_worker_limits(
        {"num_gpus": 0, "num_vcpus": 2}, max_workers=max_workers, max_cpu_workers=3)
    assert limits["cpu_workers"] == limits["total_workers"] == expected


def test_worker_measures_training_and_actual_environment(fake_runtime):
    result = run_fake()
    assert result.status == "success", result.error_message
    assert result.training_steps_completed == result.training_steps_requested == 3
    assert result.horizon == result.n_agents == 2
    assert result.reward_type == result.actual_reward_type == "private"
    assert all(env.reward_type == "private" for env in fake_runtime)
    assert len(fake_runtime) == 3  # One training plus two evaluation environments.
    assert campaign._campaign_result_complete(result.to_dict())


def test_noop_training_does_not_complete_even_after_constructor_step(fake_runtime, monkeypatch):
    class NoOp(FakeLearner):
        def __init__(self, env, **kwargs):
            super().__init__(env, **kwargs)
            env.step(np.zeros(2))
        def train(self, total_timesteps):
            pass
    monkeypatch.setattr(campaign, "get_algorithm_class", lambda config: NoOp)
    result = run_fake(timesteps_override=1)
    assert result.status == "failed"
    assert result.training_steps_completed == 0
    assert "observed steps" in result.error_message


def test_failed_training_retains_partial_evidence(fake_runtime, monkeypatch):
    class Fails(FakeLearner):
        def train(self, total_timesteps):
            self.env.step(np.zeros(2))
            raise RuntimeError("training failed")
    monkeypatch.setattr(campaign, "get_algorithm_class", lambda config: Fails)
    result = run_fake()
    assert result.status == "failed"
    assert result.training_steps_completed == 1


@pytest.mark.parametrize("metric", ["mean_return", "mean_final_trust", "std_cooperation_rate"])
def test_nonfinite_final_metrics_fail(fake_runtime, monkeypatch, metric):
    original = campaign.evaluate_agent
    def evaluate(*args, **kwargs):
        values = original(*args, **kwargs)
        values[metric] = float("nan")
        return values
    monkeypatch.setattr(campaign, "evaluate_agent", evaluate)
    result = run_fake()
    assert result.status == "failed"
    assert "finite final" in result.error_message


def test_missing_training_evidence_and_tampered_metadata_fail(fake_runtime):
    record = run_fake().to_dict()
    del record["training_steps_completed"]
    assert not campaign._campaign_result_complete(record)
    record = run_fake().to_dict()
    record["horizon"] = 999
    assert not campaign._campaign_result_complete(record)


def test_environment_reward_mismatch_and_creation_error_fail_closed(monkeypatch):
    import coopetition_gym
    monkeypatch.setattr(campaign, "_setup_path", lambda: None)
    env = FakeEnv("integrated")
    monkeypatch.setattr(coopetition_gym, "make", lambda *args, **kwargs: env)
    created, error = campaign.create_environment("Fake-v0", reward_type="private")
    assert created is None and "did not apply" in error and env.closed
    def fails(*args, **kwargs):
        raise RuntimeError("constructor failed")
    monkeypatch.setattr(coopetition_gym, "make", fails)
    created, error = campaign.create_environment("Fake-v0", reward_type="cooperative")
    assert created is None and "constructor failed" in error


def test_reward_and_budget_environment_variables_do_not_control_workers(fake_runtime, monkeypatch):
    monkeypatch.setenv("COOPETITION_REWARD_TYPE", "cooperative")
    monkeypatch.setenv("COOPETITION_TIMESTEPS_OVERRIDE", "999")
    result = run_fake()
    assert result.status == "success"
    assert result.reward_type == "private" and result.training_steps_requested == 3


def test_output_directory_rejects_mixed_treatment_and_budget(make_orchestrator):
    first = make_orchestrator()
    with pytest.raises(ValueError, match="different campaign"):
        make_orchestrator(reward_type="cooperative")
    with pytest.raises(ValueError, match="different campaign"):
        make_orchestrator(timesteps_override=4)
    expanded = make_orchestrator(seeds=[7, 8])
    assert expanded.campaign_id == first.campaign_id


def test_unidentified_legacy_results_are_preserved(make_orchestrator, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    legacy = raw / "legacy.json"
    legacy.write_text('{"status":"success"}')
    with pytest.raises(ValueError, match="no campaign manifest"):
        make_orchestrator()
    assert legacy.read_text() == '{"status":"success"}'


def test_resume_revalidates_files_and_ignores_stale_state(fake_runtime, make_orchestrator):
    orchestrator = make_orchestrator()
    result = run_fake(campaign_id=orchestrator.campaign_id)
    assert orchestrator._save_result(result)
    key = "Fake_TrustDilemma-v0_7"
    assert key in orchestrator.completed_keys
    # Stale entries in a state snapshot must never suppress pending work.
    orchestrator.state_file.write_text(json.dumps({"completed": ["Random_TrustDilemma-v0_7", key]}))
    reloaded = make_orchestrator(resume=True)
    assert reloaded.completed_keys == {key}
    assert len(reloaded.cpu_experiments) == 1
    result_path = orchestrator.raw_dir / (key + ".json")
    result_path.unlink()
    reloaded._save_state()
    assert reloaded.completed_keys == set()
    assert json.loads(reloaded.state_file.read_text())["completed"] == []


def test_invalid_success_is_saved_failed_and_never_completed(fake_runtime, make_orchestrator):
    orchestrator = make_orchestrator()
    result = run_fake(campaign_id=orchestrator.campaign_id)
    result.metrics["mean_return"] = float("inf")
    assert not orchestrator._save_result(result)
    assert result.status == "failed"
    record = json.loads(next(orchestrator.raw_dir.glob("*.json")).read_text())
    assert record["metrics"]["mean_return"] is None
    orchestrator._scan_completed_results()
    assert not orchestrator.completed_keys


def test_result_scan_rejects_corrupt_and_modified_files(fake_runtime, make_orchestrator):
    orchestrator = make_orchestrator()
    result = run_fake(campaign_id=orchestrator.campaign_id)
    orchestrator._save_result(result)
    path = next(orchestrator.raw_dir.glob("*.json"))
    record = json.loads(path.read_text())
    record["training_steps_completed"] = 0
    path.write_text(json.dumps(record))
    (orchestrator.raw_dir / "broken.json").write_text("{")
    orchestrator._scan_completed_results()
    assert not orchestrator.completed_keys


def test_cpu_pool_forwards_config_and_only_counts_valid_success(fake_runtime, make_orchestrator, monkeypatch):
    orchestrator = make_orchestrator()
    submissions = []
    result = run_fake(campaign_id=orchestrator.campaign_id)
    result.status = "failed"
    class FakePool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args, **kwargs):
            submissions.append(kwargs)
            future = Future()
            future.set_result(result)
            return future
    monkeypatch.setattr(campaign, "ProcessPoolExecutor", FakePool)
    completed, failed = orchestrator._run_cpu_experiments(None)
    assert (completed, failed) == (0, 1)
    assert not orchestrator.completed_keys
    assert submissions == [orchestrator._worker_options()]


def test_legacy_checkpoint_filename_does_not_claim_training(fake_runtime, tmp_path, monkeypatch):
    class NoOp(FakeLearner):
        def train(self, total_timesteps):
            pass
        def load(self, path):
            pytest.fail("An unverified checkpoint must never be loaded")
    monkeypatch.setattr(campaign, "get_algorithm_class", lambda config: NoOp)
    (tmp_path / "Fake_TrustDilemma-v0_7_step_999.pt").write_text("legacy")
    result = run_fake(checkpoint_dir=tmp_path)
    assert result.status == "failed" and result.training_steps_completed == 0


def test_cli_uses_config_seed_defaults_and_explicit_treatment(monkeypatch, tmp_path):
    captured = []
    class FakeOrchestrator:
        def __init__(self, config):
            captured.append(config)
        def run(self):
            pass
    monkeypatch.setattr(campaign, "UnifiedOrchestrator", FakeOrchestrator)
    campaign.main(["cooperative", "--output", str(tmp_path), "--timesteps-override", "3", "--dry-run"])
    assert captured[0].reward_type == "cooperative"
    assert captured[0].seeds == list(campaign.TRAINING_SEEDS)
    assert captured[0].timesteps_override == 3


def test_numpy_nonfinite_diagnostic_cannot_hide_in_success(fake_runtime):
    record = run_fake().to_dict()
    record["metrics"]["training_returns"] = np.array([1.0, np.nan])
    assert not campaign._campaign_result_complete(record)


def test_verified_checkpoints_resume_but_tampered_payloads_do_not(fake_runtime, tmp_path, monkeypatch):
    calls = {"train": 0, "load": 0}
    class CheckpointLearner(FakeLearner):
        def train(self, total_timesteps):
            calls["train"] += 1
            super().train(total_timesteps)
        def save(self, path):
            from pathlib import Path
            Path(path).write_text("verified fake model")
        def load(self, path):
            calls["load"] += 1
    monkeypatch.setattr(campaign, "get_algorithm_class", lambda config: CheckpointLearner)
    first = run_fake(checkpoint_dir=tmp_path)
    assert first.status == "success", first.error_message
    second = run_fake(checkpoint_dir=tmp_path)
    assert second.status == "success", second.error_message
    assert second.training_steps_completed == 3
    assert calls == {"train": 1, "load": 1}
    checkpoint = next(tmp_path.rglob("*.pt"))
    checkpoint.write_text("tampered model")
    third = run_fake(checkpoint_dir=tmp_path)
    assert third.status == "success", third.error_message
    assert calls == {"train": 2, "load": 1}


def test_gpu_pool_does_not_mark_failed_results_completed(fake_runtime, make_orchestrator, monkeypatch):
    orchestrator = make_orchestrator()
    orchestrator.worker_limits["gpu_workers"] = 1
    orchestrator.hardware["num_gpus"] = 1
    orchestrator.gpu_experiments = [(ALGORITHM, ENVIRONMENT, 7)]
    orchestrator.gpu_manager = SimpleNamespace(allocate=lambda *args: 0, release=lambda *args: None)
    result = run_fake(campaign_id=orchestrator.campaign_id)
    result.status = "failed"
    submissions = []
    class FakePool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args, **kwargs):
            submissions.append(kwargs)
            future = Future()
            future.set_result(result)
            return future
    monkeypatch.setattr(campaign, "ProcessPoolExecutor", FakePool)
    completed, failed = orchestrator._run_gpu_experiments(None)
    assert (completed, failed) == (0, 1)
    assert not orchestrator.completed_keys
    assert submissions == [orchestrator._worker_options()]


def test_source_changes_under_same_version_reject_output_reuse(make_orchestrator, monkeypatch):
    make_orchestrator()
    monkeypatch.setattr(campaign, "_source_fingerprint", lambda: "changed-source")
    with pytest.raises(ValueError, match="different campaign"):
        make_orchestrator()


@pytest.mark.parametrize("action", [np.array([np.inf, 0.0]), np.array([0.0]), np.zeros((1, 2))])
def test_campaign_rejects_invalid_policy_outputs(fake_runtime, monkeypatch, action):
    class InvalidPolicy(FakeLearner):
        def predict(self, observation, deterministic=True):
            return action
    monkeypatch.setattr(campaign, "get_algorithm_class", lambda config: InvalidPolicy)
    result = run_fake()
    assert result.status == "failed"
    assert "invalid action shape" in result.error_message
    assert all(env.closed for env in fake_runtime)


def test_campaign_never_fabricates_missing_final_trust(fake_runtime, monkeypatch):
    def create(env_id, seed=None, reward_type="private"):
        env = FakeEnv(reward_type)
        original = env.step
        def step(action):
            obs, reward, terminated, truncated, info = original(action)
            return obs, reward, terminated, truncated, {}
        env.step = step
        return env, None
    monkeypatch.setattr(campaign, "create_environment", create)
    result = run_fake()
    assert result.status == "failed"
    assert "mean_trust is missing" in result.error_message


def test_duplicate_seeds_cannot_race_the_same_output_cell(make_orchestrator):
    with pytest.raises(ValueError, match="distinct"):
        make_orchestrator(seeds=[7, 7])
