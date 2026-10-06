"""Evaluation must preserve objective/policy and expose invalid episodes."""
from types import SimpleNamespace
import numpy as np
import pytest
from experiments import evaluate


class TinyEnv:
    def __init__(self, reward_type, **kwargs):
        self.reward_type = reward_type
        self.max_steps = 1
        self.action_space = SimpleNamespace(seed=lambda seed: None, high=np.array([100.0]), shape=(1,))
        self.closed = False
    def reset(self, seed=None):
        return np.array([0.0]), {}
    def step(self, action):
        value = {"private": 1.0, "integrated": 2.0, "cooperative": 3.0}[self.reward_type]
        return np.array([0.0]), np.array([value]), False, True, {"mean_trust": 0.4}
    def close(self):
        self.closed = True


@pytest.mark.parametrize("mode,value", [("private", 1), ("integrated", 2), ("cooperative", 3)])
def test_evaluation_preserves_reward(mode, value, monkeypatch):
    made = []
    def make(env_id, **kwargs):
        made.append(TinyEnv(**kwargs))
        return made[-1]
    monkeypatch.setattr(evaluate, "_import_coopetition_gym", lambda: SimpleNamespace(make=make))
    result = evaluate.evaluate_agent(SimpleNamespace(predict=lambda *args, **kw: [50]),
                                     "Tiny", n_episodes=2, reward_type=mode)
    assert result.mean_return == value and result.reward_type == mode
    assert len(made) == 2 and all(env.closed for env in made)


@pytest.mark.parametrize("bad", ["prediction", "nonfinite"])
def test_evaluation_failure_is_not_replaced_by_random_or_zero(bad, monkeypatch):
    env = TinyEnv("integrated")
    monkeypatch.setattr(evaluate, "_import_coopetition_gym", lambda: SimpleNamespace(make=lambda *a, **kw: env))
    def predict(*args, **kwargs):
        if bad == "prediction":
            raise RuntimeError("bad checkpoint")
        return [float("nan")]
    with pytest.raises((RuntimeError, ValueError)):
        evaluate.evaluate_agent(SimpleNamespace(predict=predict), "Tiny", n_episodes=1)
    assert env.closed


def test_smoke_pipeline(tmp_path):
    from experiments.smoke import run
    run(tmp_path / "smoke")
    assert len(list((tmp_path / "smoke").rglob("returns.csv"))) == 3


def test_missing_diagnostics_stay_unknown_and_provenance_is_saved(tmp_path):
    import csv
    import json
    rows = [{"algorithm": "Random", "environment": "TrustDilemma-v0",
             "training_seed": seed, "status": "success", "reward_type": "private",
             "source_version": "fixture", "metrics": {"mean_return": value}}
            for seed, value in [(99, 10.0), (100, 20.0)]]
    aggregated = evaluate.aggregate_by_algorithm_environment(rows)
    result = aggregated["TrustDilemma-v0"]["Random"]
    assert result.mean_final_trust is None
    assert result.mean_training_time_seconds is None
    assert result.std_return_across_seeds == pytest.approx(50 ** 0.5)
    path = tmp_path / "summary.csv"
    evaluate.write_summary_csv(aggregated, path)
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["mean_cooperation_rate"] == ""
    assert json.loads(row["seeds"]) == [99, 100]
    assert json.loads(row["configuration"])["source_version"] == "fixture"
    assert row["reward_type"] == "private"


def test_heuristic_observes_the_live_episode_state(monkeypatch):
    env = TinyEnv("private")
    env.max_steps = 2
    env.current_step = 0
    def step(action):
        env.current_step += 1
        return np.array([env.current_step]), np.array([1.0]), False, env.current_step == 2, {"mean_trust": 0.5}
    env.step = step
    made = []
    def make(*args, **kwargs):
        made.append(kwargs)
        return env
    monkeypatch.setattr(evaluate, "_import_coopetition_gym", lambda: SimpleNamespace(make=make))
    seen = []
    def policy(obs, live):
        assert live is env
        seen.append(live.current_step)
        return [50]
    result = evaluate.evaluate_heuristic(policy, "Tiny", n_episodes=1, reward_type="private")
    assert seen == [0, 1] and len(made) == 1
    assert result.mean_return == 2 and env.closed
