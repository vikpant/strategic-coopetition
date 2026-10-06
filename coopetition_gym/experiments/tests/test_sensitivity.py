"""Sensitivity provenance and resume checks without model training."""
import json
from copy import deepcopy

import pytest

from experiments import campaign, sensitivity


def experiment(reward_type="private"):
    return sensitivity.build_experiment_matrix(
        ["MADDPG"], ["TrustDilemma-v0"], [7], [[64, 64]], [reward_type], set()
    )[0]


def result_for(exp, campaign_id="sensitivity-test"):
    env_config = dict(exp["env_config"], reward_type=exp["reward_type"])
    record = campaign.ExperimentResult(
        algorithm=exp["algo_config"]["name"], environment=env_config["id"], training_seed=7,
        status="success", reward_type=exp["reward_type"], actual_reward_type=exp["reward_type"],
        source_version=campaign._source_version(), source_fingerprint=campaign._source_fingerprint(),
        campaign_id=campaign_id,
        horizon=env_config["horizon"], n_agents=env_config["n_agents"], requires_training=True,
        training_steps_requested=3, training_steps_completed=3, evaluation_episodes_requested=2,
        algorithm_config=deepcopy(exp["algo_config"]), environment_config=env_config,
        metrics={"mean_return": 1.0, "std_return": 0.0, "mean_final_trust": 0.5,
                 "std_final_trust": 0.0, "mean_cooperation_rate": 0.0,
                 "std_cooperation_rate": 0.0, "mean_episode_length": 2.0,
                 "episodes_evaluated": 2},
    )
    record.config_hash = campaign._identity_hash(campaign._result_configuration(record.to_dict()))
    return record


def run_fake(tmp_path, monkeypatch, exp=None, result=None):
    exp = exp or experiment()
    result = result or result_for(exp)
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return result
    monkeypatch.setattr(campaign, "run_single_experiment", run)
    summary = sensitivity.run_sensitivity_experiment(
        algo_config=exp["algo_config"], env_config=exp["env_config"], seed=7,
        net_arch=exp["net_arch"], reward_type=exp["reward_type"], n_eval_episodes=2,
        gpu_id=-1, enable_gpu_isolation=True, checkpoint_dir=None, checkpoint_interval=100,
        log_file=None, progress_dir=None, raw_dir=str(tmp_path),
        campaign_id="sensitivity-test", timesteps_override=3,
    )
    return summary, calls


def test_algorithm_parameters_come_from_campaign_and_matrix_copies_them():
    originals = {algo["name"]: algo for algo in campaign.TRAINING_ALGORITHMS}
    for name, config in sensitivity.SENSITIVITY_ALGORITHMS.items():
        assert config == originals[name]
    exp = experiment()
    assert exp["algo_config"]["params"]["net_arch"] == [64, 64]
    assert sensitivity.SENSITIVITY_ALGORITHMS["MADDPG"]["params"]["net_arch"] == [128, 128]
    expected = deepcopy(originals["MADDPG"]["params"])
    expected["net_arch"] = [64, 64]
    assert exp["algo_config"]["params"] == expected


def test_valid_result_is_explicit_and_resumable(tmp_path, monkeypatch):
    summary, calls = run_fake(tmp_path, monkeypatch)
    assert summary["status"] == "success"
    assert summary["filename"].endswith("_net64x64_private.json")
    assert calls[0]["reward_type"] == "private"
    assert calls[0]["timesteps_override"] == 3
    assert calls[0]["campaign_id"] == "sensitivity-test"
    assert sensitivity.scan_completed(tmp_path, "sensitivity-test", 2, 3) == {experiment()["key"]}
    assert sensitivity.scan_completed(tmp_path) == set()  # No unspecified completion claim.


def test_changed_budget_evaluation_identity_and_source_do_not_resume(tmp_path, monkeypatch):
    run_fake(tmp_path, monkeypatch)
    assert not sensitivity.scan_completed(tmp_path, "another-campaign", 2, 3)
    assert not sensitivity.scan_completed(tmp_path, "sensitivity-test", 3, 3)
    assert not sensitivity.scan_completed(tmp_path, "sensitivity-test", 2, 4)
    monkeypatch.setattr(campaign, "_source_version", lambda: "changed-version")
    assert not sensitivity.scan_completed(tmp_path, "sensitivity-test", 2, 3)


def test_tampered_hyperparameters_do_not_resume_even_with_recomputed_hash(tmp_path, monkeypatch):
    summary, _ = run_fake(tmp_path, monkeypatch)
    path = tmp_path / summary["filename"]
    record = json.loads(path.read_text())
    record["algorithm_config"]["params"]["learning_rate_actor"] *= 10
    record["config_hash"] = campaign._identity_hash(campaign._result_configuration(record))
    path.write_text(json.dumps(record))
    assert not sensitivity.scan_completed(tmp_path, "sensitivity-test", 2, 3)


def test_nonfinite_success_is_written_as_failed_json(tmp_path, monkeypatch):
    result = result_for(experiment())
    result.metrics["mean_return"] = float("nan")
    summary, _ = run_fake(tmp_path, monkeypatch, result=result)
    assert summary["status"] == "failed"
    saved = json.loads((tmp_path / summary["filename"]).read_text())
    assert saved["status"] == "failed" and saved["metrics"]["mean_return"] is None
    assert not sensitivity.scan_completed(tmp_path, "sensitivity-test", 2, 3)


def test_failed_result_without_metrics_is_persisted(tmp_path, monkeypatch):
    result = result_for(experiment())
    result.status = "failed"
    result.metrics = None
    summary, _ = run_fake(tmp_path, monkeypatch, result=result)
    assert summary["status"] == "failed" and summary["mean_return"] is None
    assert (tmp_path / summary["filename"]).exists()


def test_flat_outputs_do_not_overwrite_another_reward_treatment(tmp_path, monkeypatch):
    first, _ = run_fake(tmp_path, monkeypatch, exp=experiment("private"))
    second, _ = run_fake(tmp_path, monkeypatch, exp=experiment("integrated"))
    assert first["status"] == second["status"] == "success"
    assert first["filename"] != second["filename"]
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_wrapper_never_relabels_an_actual_reward_mismatch(tmp_path, monkeypatch):
    result = result_for(experiment("integrated"))
    summary, _ = run_fake(tmp_path, monkeypatch, exp=experiment("private"), result=result)
    assert summary["status"] == "failed"
    record = json.loads((tmp_path / summary["filename"]).read_text())
    assert record["reward_type"] == record["actual_reward_type"] == "integrated"
    assert not sensitivity.scan_completed(tmp_path, "sensitivity-test", 2, 3)


def test_manifest_rejects_changed_scientific_configuration(tmp_path, monkeypatch):
    first = sensitivity.ensure_campaign_identity(tmp_path, 2, 3)
    assert sensitivity.ensure_campaign_identity(tmp_path, 2, 3) == first
    with pytest.raises(ValueError, match="different sensitivity"):
        sensitivity.ensure_campaign_identity(tmp_path, 3, 3)
    with pytest.raises(ValueError, match="different sensitivity"):
        sensitivity.ensure_campaign_identity(tmp_path, 2, 4)
    monkeypatch.setattr(sensitivity, "_source_version", lambda: "changed-version")
    with pytest.raises(ValueError, match="different sensitivity"):
        sensitivity.ensure_campaign_identity(tmp_path, 2, 3)


def test_legacy_sensitivity_results_are_preserved(tmp_path):
    raw = tmp_path / "private" / "raw"
    raw.mkdir(parents=True)
    (raw / "legacy.json").write_text('{"status":"success"}')
    with pytest.raises(ValueError, match="no manifest"):
        sensitivity.ensure_campaign_identity(tmp_path, 2, 3)
    assert (raw / "legacy.json").read_text() == '{"status":"success"}'


def test_matrix_rejects_unknown_or_invalid_configurations():
    with pytest.raises(ValueError, match="algorithm"):
        sensitivity.build_experiment_matrix(["unknown"], None, [7], [[64, 64]], ["private"], set())
    with pytest.raises(ValueError, match="widths"):
        sensitivity.build_experiment_matrix(None, None, [7], [[0]], ["private"], set())
    with pytest.raises(ValueError, match="reward"):
        sensitivity.build_experiment_matrix(None, None, [7], [[64]], ["unknown"], set())


def test_defaults_use_one_worker_and_small_architecture(tmp_path, monkeypatch):
    captured = []
    class FakePool:
        def __init__(self, **kwargs):
            captured.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, *args, **kwargs):
            from concurrent.futures import Future
            future = Future()
            future.set_result({"status": "failed"})
            return future
    monkeypatch.setattr(sensitivity, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(sensitivity, "detect_gpus", lambda: 0)
    sensitivity.main(["--output", str(tmp_path), "--algorithms", "MAPPO", "--environments",
                      "TrustDilemma-v0", "--seeds", "7", "--reward-types", "private"])
    assert captured[0]["max_workers"] == 1
    assert sensitivity.DEFAULT_NET_SIZES == [[64, 64]]


def test_sensitivity_manifest_rejects_same_version_source_changes(tmp_path, monkeypatch):
    sensitivity.ensure_campaign_identity(tmp_path, 2, 3)
    monkeypatch.setattr(sensitivity, "_source_fingerprint", lambda: "changed-source")
    with pytest.raises(ValueError, match="different sensitivity"):
        sensitivity.ensure_campaign_identity(tmp_path, 2, 3)


def test_duplicate_sensitivity_axes_cannot_race_the_same_output_cell():
    with pytest.raises(ValueError, match="distinct"):
        sensitivity.build_experiment_matrix(None, None, [7, 7], [[64]], ["private"], set())
    with pytest.raises(ValueError, match="distinct"):
        sensitivity.build_experiment_matrix(None, None, [7], [[64], [64]], ["private"], set())
