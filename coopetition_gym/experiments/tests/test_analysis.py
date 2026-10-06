"""Regression tests for analysis entry points and experimental units."""
import json
from types import SimpleNamespace

import pytest

from experiments import analyze


def test_all_loads_unified_input_once(tmp_path, monkeypatch):
    for seed, value in [(99, 10.0), (100, 20.0)]:
        record = {"algorithm": "ISAC", "environment": "TrustDilemma-v0",
                  "training_seed": seed, "status": "success", "reward_type": "integrated",
                  "metrics": {"mean_return": value}}
        (tmp_path / f"ISAC_TrustDilemma-v0_{seed}.json").write_text(json.dumps(record))
    captured = []
    monkeypatch.setattr(analyze, "_analyze_all_main", lambda records: captured.extend(records))
    analyze._cmd_all(SimpleNamespace(input_dir=str(tmp_path), output_dir=str(tmp_path)))
    result = analyze.compute_returns_summary(captured)[("ISAC", "TrustDilemma-v0")]
    assert result["n"] == 2
    assert result["mean"] == 15
    assert result["std"] == pytest.approx(50 ** 0.5)
    assert result["sem"] == 5


def test_summary_excludes_failed_and_nonfinite_results():
    records = [{"algo": "ISAC", "env": "TrustDilemma-v0", "seed": i,
                "status": status, "mean_return": value}
               for i, (status, value) in enumerate([
                   ("success", 3.0), ("failed", 100.0),
                   ("success", float("nan")), ("success", float("inf"))])]
    summary = analyze.compute_returns_summary(records)[("ISAC", "TrustDilemma-v0")]
    assert summary["n"] == 1
    assert summary["mean"] == 3


def test_jsonl_all_and_individual_cli_agree(tmp_path, monkeypatch):
    import csv
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows = [{"algorithm": "Random", "environment": "TrustDilemma-v0",
             "training_seed": seed, "status": "success", "reward_type": "integrated",
             "metrics": {"mean_return": value}}
            for seed, value in [(99, 10.0), (100, 20.0)]]
    (input_dir / "runs.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    # Plotting does not affect this statistics regression; all actual tabular
    # analysis functions still run through the CLI on the fixture.
    monkeypatch.setattr(analyze, "make_plots", lambda *args: None)
    individual = tmp_path / "individual.csv"
    output = tmp_path / "all"
    assert analyze.main(["returns-summary", "--input-dir", str(input_dir), "--output", str(individual)]) == 0
    assert analyze.main(["all", "--input-dir", str(input_dir), "--output-dir", str(output)]) == 0
    assert individual.read_text() == (output / "returns_summary.csv").read_text()
    with individual.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["n_seeds"] == "2"
    assert json.loads(row["seeds"]) == [99, 100]
    assert row["reward_type"] == "integrated"
    assert float(row["sem_return"]) == 5


def test_summary_rejects_repeated_seed_records():
    from experiments.records import RecordError
    row = {"algo": "Random", "env": "TrustDilemma-v0", "seed": 99,
           "status": "success", "mean_return": 3.0}
    with pytest.raises(RecordError, match="repeated seed"):
        analyze.compute_returns_summary([row, row])


def test_all_keeps_nonfinite_diagnostics_out_of_summary(tmp_path, monkeypatch):
    import csv
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows = [
        {"algorithm": "MASAC", "environment": "TrustDilemma-v0", "training_seed": 99,
         "status": "success", "reward_type": "integrated",
         "metrics": {"mean_return": 10.0, "training_metrics": {"critic_loss": [[1, 2.0]]}}},
        {"algorithm": "MASAC", "environment": "TrustDilemma-v0", "training_seed": 100,
         "status": "success", "reward_type": "integrated",
         "metrics": {"mean_return": 100.0, "training_metrics": {"critic_loss": [[1, float("inf")]]}}},
    ]
    (input_dir / "runs.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(analyze, "make_plots", lambda *args: None)
    output = tmp_path / "analysis"
    assert analyze.main(["all", "--input-dir", str(input_dir), "--output-dir", str(output)]) == 0
    with (output / "returns_summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["n_seeds"] == "1"
    assert float(row["mean_return"]) == 10
    assert row["comparison_basis"] == "historical-roster"
    diagnostic = (output / "masac_instability.txt").read_text()
    assert "Input records: 2" in diagnostic
    assert "observed nonfinite loss: 1/2" in diagnostic


def test_reward_ablation_retains_arm_seed_and_configuration(tmp_path):
    import csv
    directories = {}
    for mode, seed, value in [("integrated", 99, 10), ("private", 100, 5), ("cooperative", 101, 15)]:
        directory = tmp_path / mode
        directory.mkdir()
        row = {"algorithm": "Random", "environment": "TrustDilemma-v0",
               "training_seed": seed, "status": "success", "reward_type": mode,
               "campaign_id": f"fixture-{mode}", "metrics": {"mean_return": value}}
        (directory / "result.json").write_text(json.dumps(row))
        directories[mode] = str(directory)
    output = tmp_path / "comparison"
    analyze.compare_reward_configurations(directories["integrated"], directories["private"], directories["cooperative"], str(output))
    with (output / "reward_ablation_summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["seeds_integrated"]) == [99]
    assert json.loads(row["seeds_private"]) == [100]
    assert json.loads(row["configuration_cooperative"])["campaign_id"] == "fixture-cooperative"
    assert row["comparison_basis"] == "historical-roster"
    assert row["comparison_status"] == "incomparable: different seed folds"
    assert row["delta_integrated_minus_private"] == ""
    assert row["delta_cooperative_minus_integrated"] == ""


@pytest.mark.parametrize("difference", [
    {"horizon": 200}, {"training_steps_requested": 200},
    {"source_version": "different"}, {"source_fingerprint": "different"},
    {"algorithm_config": {"params": {"learning_rate": 0.02}}},
    {"evaluation_config": {"episodes": 20}},
])
def test_cross_arm_context_rejects_scientific_changes(difference):
    first = {"seeds": [99, 100], "configuration": {"horizon": 100,
             "training_steps_requested": 100, "source_version": "1.0.8",
             "source_fingerprint": "fixture", "algorithm_config": {"params": {"learning_rate": 0.01}},
             "evaluation_config": {"episodes": 10}}}
    changed = {"seeds": [99, 100], "configuration": dict(first["configuration"], **difference)}
    assert analyze._cross_arm_status([first, first, changed]).startswith("incomparable")


def test_cross_arm_comparison_ignores_only_treatment_identity():
    arms = [{"seeds": [99, 100], "configuration": {
        "horizon": 100, "source_fingerprint": "same", "actual_reward_type": mode,
        "campaign_id": mode, "config_hash": mode,
        "environment_config": {"n_agents": 2, "reward_type": mode}}}
        for mode in ("private", "integrated", "cooperative")]
    assert analyze._cross_arm_status(arms) == "matching recorded seeds and context"
