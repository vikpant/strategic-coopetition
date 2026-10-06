"""Orchestrator failures must reach callers without discarding valid results."""
import json
from unittest.mock import Mock

import pytest

from experiments import campaign


@pytest.fixture
def make_orchestrator(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "detect_hardware", lambda: {"num_gpus": 0, "num_vcpus": 2})
    monkeypatch.setattr(campaign.signal, "signal", lambda *args: None)
    monkeypatch.setattr(campaign.UnifiedOrchestrator, "_print_resource_summary", lambda self: None)

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


def save_success(orchestrator, experiment):
    """Supply a schema-valid fake result through the real atomic writer."""
    algorithm, environment, seed = experiment
    metrics = dict(mean_return=1.0, std_return=0.0, mean_final_trust=0.5,
                   std_final_trust=0.0, mean_cooperation_rate=0.5,
                   std_cooperation_rate=0.0, mean_episode_length=2.0,
                   episodes_evaluated=2)
    result = campaign.ExperimentResult(
        algorithm=algorithm["name"], environment=environment["id"],
        training_seed=seed, status="success", metrics=metrics,
        reward_type="private", actual_reward_type="private",
        source_version=campaign._source_version(),
        source_fingerprint=campaign._source_fingerprint(),
        campaign_id=orchestrator.campaign_id, horizon=2, n_agents=2,
        evaluation_episodes_requested=2, algorithm_config=algorithm,
        environment_config={**environment, "horizon": 2, "n_agents": 2,
                            "reward_type": "private"})
    result.config_hash = campaign._identity_hash(campaign._result_configuration(result.to_dict()))
    assert orchestrator._save_result(result)
    return f"{result.algorithm}_{result.environment}_{seed}"


@pytest.mark.parametrize("pool", ["CPU", "GPU"])
def test_pool_error_propagates_after_monitor_shutdown_and_state_save(make_orchestrator, monkeypatch, pool):
    orchestrator = make_orchestrator()
    if pool == "GPU":
        orchestrator.gpu_experiments = orchestrator.cpu_experiments
        orchestrator.cpu_experiments = []
    monitor = Mock()
    monitor.get_current_status.return_value = {"oom_count": 0}
    orchestrator.gpu_monitor = monitor
    orchestrator.system_monitor = Mock()

    def fail(_context):
        raise ValueError("pool startup failed")

    monkeypatch.setattr(orchestrator, f"_run_{pool.lower()}_experiments", fail)
    with pytest.raises(RuntimeError, match=f"{pool} pool: pool startup failed") as error:
        orchestrator.run()
    assert isinstance(error.value.__cause__, ValueError)
    monitor.start.assert_called_once()
    monitor.stop.assert_called_once()
    orchestrator.system_monitor.stop.assert_called_once()
    assert json.loads(orchestrator.state_file.read_text())["completed"] == []


@pytest.mark.parametrize("counts, interrupted", [((0, 1), False), ((0, 0), True), ((1, 0), False)])
def test_failed_interrupted_or_unpersisted_cells_fail(make_orchestrator, monkeypatch, counts, interrupted):
    orchestrator = make_orchestrator()
    orchestrator.shutdown_requested = interrupted
    monkeypatch.setattr(orchestrator, "_run_cpu_experiments", lambda context: counts)
    with pytest.raises(RuntimeError, match="1 missing result"):
        orchestrator.run()
    assert json.loads(orchestrator.state_file.read_text())["completed"] == []


def test_other_pool_failure_preserves_valid_partial_result(make_orchestrator, monkeypatch):
    orchestrator = make_orchestrator(seeds=[7, 8])
    orchestrator.gpu_experiments = [orchestrator.cpu_experiments.pop()]
    saved = []

    def finish_cpu(_context):
        saved.append(save_success(orchestrator, orchestrator.cpu_experiments[0]))
        return 1, 0

    def fail_gpu(_context):
        raise ValueError("GPU unavailable")

    monkeypatch.setattr(orchestrator, "_run_cpu_experiments", finish_cpu)
    monkeypatch.setattr(orchestrator, "_run_gpu_experiments", fail_gpu)
    with pytest.raises(RuntimeError, match="1/2 completed"):
        orchestrator.run()
    assert json.loads(orchestrator.state_file.read_text())["completed"] == saved
    assert (orchestrator.raw_dir / f"{saved[0]}.json").exists()
    resumed = make_orchestrator(seeds=[7, 8], resume=True)
    assert len(resumed.cpu_experiments) == 1
    assert resumed.completed_keys == set(saved)


def test_success_and_zero_pending_resume_return_normally(make_orchestrator, monkeypatch):
    orchestrator = make_orchestrator()

    def finish(_context):
        save_success(orchestrator, orchestrator.cpu_experiments[0])
        return 1, 0

    monkeypatch.setattr(orchestrator, "_run_cpu_experiments", finish)
    assert orchestrator.run() is None
    resumed = make_orchestrator(resume=True)
    assert not resumed.cpu_experiments and not resumed.gpu_experiments
    monkeypatch.setattr(resumed, "_run_cpu_experiments", Mock(side_effect=AssertionError("pool called")))
    assert resumed.run() is None
    resumed._run_cpu_experiments.assert_not_called()


def test_dry_run_does_not_require_results_or_start_pools(make_orchestrator, monkeypatch):
    orchestrator = make_orchestrator(dry_run=True)
    pool = Mock(side_effect=AssertionError("pool called"))
    monkeypatch.setattr(orchestrator, "_run_cpu_experiments", pool)
    assert orchestrator.run() is None
    pool.assert_not_called()
    assert not orchestrator.state_file.exists()
