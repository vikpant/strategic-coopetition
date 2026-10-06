"""Validator checks the schema written by campaign evaluation."""
import json

from experiments import validate


def record(seed=99, **extra):
    return {"algorithm": "Random", "environment": "TrustDilemma-v0",
            "training_seed": seed, "reward_type": "integrated", "status": "success",
            "metrics": {"mean_return": 3.0}, **extra}


def write_rows(tmp_path, rows):
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_native_small_dataset_needs_no_historical_counts(tmp_path, capsys):
    write_rows(tmp_path, [record(), record(seed=100), {"type": "progress", "step": 2}])
    assert validate.validate_training_dataset(tmp_path, expected_records=2) == 0
    assert "progress records: 1" in capsys.readouterr().out
    assert validate.validate_training_dataset(tmp_path, expected_records=3) > 0


def test_nested_nonfinite_returns_fail_validation(tmp_path, capsys):
    write_rows(tmp_path, [record(metrics={"mean_return": float("nan")})])
    assert validate.validate_training_dataset(tmp_path) > 0
    assert "metrics.mean_return" in capsys.readouterr().out


def test_failed_duplicate_and_missing_seed_are_not_clean(tmp_path):
    write_rows(tmp_path, [record(), record(), record(seed=True), record(seed=100, status="failed")])
    assert validate.validate_training_dataset(tmp_path) >= 3


def test_distinct_treatments_can_be_validated_together(tmp_path):
    write_rows(tmp_path, [record(), record(reward_type="private")])
    assert validate.validate_training_dataset(tmp_path) == 0


def test_missing_treatment_is_reported_or_explicitly_supplied(tmp_path):
    legacy = record()
    del legacy["reward_type"]
    write_rows(tmp_path, [legacy])
    assert validate.validate_training_dataset(tmp_path) > 0
    assert validate.validate_training_dataset(tmp_path, reward_type="integrated") == 0


def test_audit_jsonl_without_fixed_historical_counts(tmp_path):
    audit = {"algorithm": "Random", "environment": "TrustDilemma-v0", "seed": 99,
             "response_surface": {"0.5": {"mean_return": 1.0}},
             "exploitation_analysis": [{"exploitative": False}], "n_exploitative": 0}
    write_rows(tmp_path, [audit])
    assert validate.validate_audit_dataset(tmp_path, expected_static=1) == 0
    audit["n_exploitative"] = 1
    write_rows(tmp_path, [audit])
    assert validate.validate_audit_dataset(tmp_path) > 0
