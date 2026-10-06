"""Input/provenance behavior using synthetic artifacts only."""
import csv
import hashlib
import json

import pytest

from experiments.records import (RecordError, inspect_result, is_successful_result,
                                 load_records, read_records, verify_manifest)


def result(seed=99, reward_type="integrated", value=10.0, **extra):
    return {"algorithm": "ISAC", "environment": "TrustDilemma-v0",
            "training_seed": seed, "reward_type": reward_type, "status": "success",
            "metrics": {"mean_return": value}, **extra}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_recursive_json_and_jsonl_preserve_record_provenance(tmp_path):
    record = result(config={"critic_lr": 0.001}, campaign_id="fixture")
    write_jsonl(tmp_path / "nested" / "shard.jsonl", [record, {"type": "progress", "step": 2}])
    (tmp_path / "other.json").write_text(json.dumps(result(seed=100, config={"critic_lr": 0.001}, campaign_id="fixture")))
    loaded = load_records(tmp_path)
    assert len(loaded) == 2
    assert {row["training_seed"] for row in loaded} == {99, 100}
    assert loaded[0]["config"] == {"critic_lr": 0.001}
    assert loaded[0]["_source"].endswith("shard.jsonl:1")


def test_progress_with_training_seed_is_not_a_completed_result(tmp_path):
    write_jsonl(tmp_path / "runs.jsonl", [result(), {"algorithm": "ISAC", "training_seed": 99, "step": 8}])
    assert len(load_records(tmp_path)) == 1


@pytest.mark.parametrize("change,match", [
    ({"reward_type": "private"}, "mixed reward"),
    ({"config": {"critic_lr": 0.02}}, "multiple configurations"),
    ({"campaign_id": "other"}, "multiple configurations"),
])
def test_comparisons_reject_mixed_arms(tmp_path, change, match):
    write_jsonl(tmp_path / "runs.jsonl", [result(), result(seed=100, **change)])
    with pytest.raises(RecordError, match=match):
        load_records(tmp_path)


def test_config_variants_preserved_for_validation_but_not_pooled(tmp_path):
    write_jsonl(tmp_path / "runs.jsonl", [result(config={"width": 32}), result(config={"width": 64})])
    rows = load_records(tmp_path, comparable=False)
    assert len(rows) == 2
    with pytest.raises(RecordError, match="multiple configurations"):
        load_records(tmp_path)


@pytest.mark.parametrize("second", [result(), result(value=20.0)])
def test_repeated_cells_require_explicit_selection(tmp_path, second):
    write_jsonl(tmp_path / "runs.jsonl", [result(), second])
    with pytest.raises(RecordError, match="duplicate experimental cell"):
        load_records(tmp_path)


def test_missing_treatment_requires_explicit_context(tmp_path):
    legacy = result()
    del legacy["reward_type"]
    write_jsonl(tmp_path / "runs.jsonl", [legacy])
    with pytest.raises(RecordError, match="missing reward_type"):
        load_records(tmp_path)
    rows = load_records(tmp_path, reward_type="private")
    assert rows[0]["reward_type"] == "private"
    assert rows[0]["_reward_type_assumption"] == "explicit caller option"


def test_explicit_fold_and_treatment_select_without_relabelling(tmp_path):
    write_jsonl(tmp_path / "runs.jsonl", [result(seed=99), result(seed=109), result(reward_type="private")])
    rows = load_records(tmp_path, reward_type="integrated", seeds=[99])
    assert [(r["training_seed"], r["reward_type"]) for r in rows] == [(99, "integrated")]


def test_nonfinite_failed_and_partial_learning_are_not_completed():
    assert not is_successful_result(result(value=float("nan")))
    assert not is_successful_result(result(status="failed"))
    assert not is_successful_result(result(requires_training=True, training_steps_requested=100, training_steps_completed=99))
    assert is_successful_result(result(requires_training=False, training_steps_requested=0, training_steps_completed=0))
    assert is_successful_result(result(requires_training=True, training_steps_requested=100, training_steps_completed=100))


def test_failed_and_nonfinite_rows_reported_and_excluded(tmp_path):
    write_jsonl(tmp_path / "runs.jsonl", [result(), result(seed=100, value=float("inf")), result(seed=101, status="failed", metrics=None)])
    with pytest.warns(RuntimeWarning, match="excluded 2"):
        rows = load_records(tmp_path)
    assert len(rows) == 1


def test_malformed_identity_and_unrecognized_records_fail(tmp_path):
    write_jsonl(tmp_path / "runs.jsonl", [result(training_seed=True)])
    with pytest.raises(RecordError, match="training_seed"):
        load_records(tmp_path)
    write_jsonl(tmp_path / "runs.jsonl", [{"surprise": 1}])
    with pytest.raises(RecordError, match="unrecognized"):
        load_records(tmp_path)


def test_manifest_physical_row_counts_and_checksum(tmp_path):
    shard = tmp_path / "runs.jsonl"
    write_jsonl(shard, [result(), {"type": "progress"}])
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "record_count", "md5"])
        writer.writeheader()
        writer.writerow({"filename": shard.name, "record_count": 2,
                         "md5": hashlib.md5(shard.read_bytes()).hexdigest()})
    assert verify_manifest(tmp_path, manifest) == []
    shard.write_text(shard.read_text() + json.dumps(result(seed=100)) + "\n")
    issues = verify_manifest(tmp_path, manifest)
    assert any("md5 mismatch" in issue for issue in issues)
    assert any("record count" in issue for issue in issues)


def test_campaign_container_requires_explicit_raw_input(tmp_path):
    write_jsonl(tmp_path / "raw" / "results.jsonl", [result()])
    (tmp_path / "campaign.json").write_text(json.dumps({"campaign_id": "fixture", "configuration": {}}))
    with pytest.raises(RecordError, match="select OUTPUT/raw"):
        load_records(tmp_path)
    assert len(load_records(tmp_path / "raw")) == 1


@pytest.mark.parametrize("bad", [[], {}])
def test_malformed_reward_is_reported_before_treatment_filter(tmp_path, bad):
    malformed = result(reward_type=bad)
    assert any("reward_type" in issue for issue in inspect_result(malformed))
    assert not is_successful_result(malformed)
    write_jsonl(tmp_path / "runs.jsonl", [result(), malformed])
    with pytest.raises(RecordError, match="reward_type"):
        load_records(tmp_path, reward_type="integrated")


@pytest.mark.parametrize("marker", ["record_type", "type", "event"])
@pytest.mark.parametrize("bad", [[], {}])
def test_nonstring_record_marker_is_a_schema_error(tmp_path, marker, bad):
    malformed = result(**{marker: bad})
    assert not is_successful_result(malformed)
    write_jsonl(tmp_path / "runs.jsonl", [result(), malformed])
    with pytest.raises(RecordError, match=f"{marker} must be a string"):
        load_records(tmp_path)


def test_malformed_seed_is_reported_before_fold_filter(tmp_path):
    write_jsonl(tmp_path / "runs.jsonl", [result(), result(training_seed=[])])
    with pytest.raises(RecordError, match="training_seed must be"):
        load_records(tmp_path, seeds=[99])


@pytest.mark.parametrize("option", [{"reward_type": []}, {"seeds": [[]]}, {"seeds": 99}])
def test_malformed_reader_options_raise_record_error(tmp_path, option):
    write_jsonl(tmp_path / "runs.jsonl", [result()])
    with pytest.raises(RecordError):
        load_records(tmp_path, **option)
