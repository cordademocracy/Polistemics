import json

from src.collect.cost_tracker import CostTracker


def test_log_creates_file(tmp_path):
    tracker = CostTracker("test_exp", experiment_dir=tmp_path)
    tracker.log("model_a", 100, 50, 0.001, 500.0)
    assert (tmp_path / "costs.jsonl").exists()


def test_log_appends_entries(tmp_path):
    tracker = CostTracker("test_exp", experiment_dir=tmp_path)
    tracker.log("model_a", 100, 50, 0.001, 500.0)
    tracker.log("model_b", 200, 80, 0.003, 800.0)
    lines = (tmp_path / "costs.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["model_id"] == "model_a"
    assert entry["tokens_input"] == 100


def test_log_handles_none_cost(tmp_path):
    tracker = CostTracker("test_exp", experiment_dir=tmp_path)
    tracker.log("model_a", 100, 50, None, 500.0)
    entry = json.loads((tmp_path / "costs.jsonl").read_text().strip())
    assert entry["cost_usd"] is None
