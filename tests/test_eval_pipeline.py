from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.common.io import append_output_to_path, save_ground_truth
from src.common.schemas import (
    DatasetItem,
    LLMOutput,
    OutputStatus,
    PromptVariation,
    RefusalType,
    StanceLabel,
)
from src.evaluate.filter import OutputFilter
from src.evaluate.panel import PartialPanelResult
from src.evaluate.pipeline import EvaluationPipeline, _prune_incomplete
from src.metrics.base import BaseMetric
from src.metrics.registry import METRIC_REGISTRY

# ---------------------------------------------------------------------------
# Dummy metric for testing
# ---------------------------------------------------------------------------


class DummyMetric(BaseMetric):
    """Simple correctness metric for pipeline tests."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def aggregation_method(self) -> str:
        return "mean"

    async def score(self, output: LLMOutput, ground_truth: DatasetItem) -> dict[str, float]:
        correct = 1.0 if output.predicted_stance == ground_truth.stance_label else 0.0
        return {"correct": correct}

    def aggregate(self, run_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["correct"] for s in run_scores]
        return {"correct": sum(values) / len(values)}

    @property
    def primary_score_key(self) -> str:
        return "mean_correct"

    def summarize(self, batch_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["correct"] for s in batch_scores]
        return {"mean_correct": sum(values) / len(values)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gt_item(obs_id: str, party_id: str = "de_spd") -> DatasetItem:
    return DatasetItem(
        observation_id=obs_id,
        election_id="bundestagswahl2025",
        party_id=party_id,
        party_name="SPD" if party_id == "de_spd" else "CDU/CSU",
        party_anonymized="Party 01",
        statement_id="s001",
        statement_number=None,
        statement_text="Test statement",
        statement_category=None,
        stance_label=StanceLabel.AGREE,
        rationale_text=None,
        has_rationale=False,
        ie_name="baseline",
        ie_chunks=[],
    )


def _make_output(
    obs_id: str,
    party_id: str = "de_spd",
    model_id: str = "model_a",
    prompt_variation: PromptVariation = PromptVariation.MINIMAL,
    run_index: int = 0,
    temperature: float = 0.0,
    status: OutputStatus = OutputStatus.SUCCESS,
) -> LLMOutput:
    return LLMOutput(
        observation_id=obs_id,
        statement_id="s001",
        party_id=party_id,
        experiment_id="test",
        model_id=model_id,
        prompt_variation=prompt_variation,
        run_index=run_index,
        temperature=temperature,
        predicted_stance=StanceLabel.AGREE,
        predicted_explanation="test explanation",
        timestamp=datetime.now(UTC),
        latency_ms=100.0,
        tokens_input=50,
        tokens_output=100,
        cost_usd=0.001,
        status=status,
        error_message=None,
        refusal_type=RefusalType.NONE,
        ie_name="baseline",
        condition_id="baseline__real__evidence__none",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def register_dummy_metric():
    """Register DummyMetric for all tests, clean up after."""
    METRIC_REGISTRY["dummy"] = DummyMetric
    yield
    METRIC_REGISTRY.pop("dummy", None)


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    """Create experiment dir with 2 outputs + ground truth."""
    exp_dir = tmp_path / "test_experiment"
    exp_dir.mkdir()

    # Ground truth
    gt_items = [_make_gt_item("obs1", "de_spd"), _make_gt_item("obs2", "de_cdu")]
    save_ground_truth(gt_items, exp_dir / "ground_truth.jsonl")

    # Outputs
    outputs_path = exp_dir / "outputs.jsonl"
    append_output_to_path(_make_output("obs1", "de_spd"), outputs_path)
    append_output_to_path(_make_output("obs2", "de_cdu"), outputs_path)

    return exp_dir


@pytest.fixture
def experiment_dir_with_k_runs(tmp_path: Path) -> Path:
    """Create experiment dir with K=3 runs per observation."""
    exp_dir = tmp_path / "test_experiment_k"
    exp_dir.mkdir()

    gt_items = [_make_gt_item("obs1", "de_spd")]
    save_ground_truth(gt_items, exp_dir / "ground_truth.jsonl")

    outputs_path = exp_dir / "outputs.jsonl"
    for k in range(3):
        append_output_to_path(_make_output("obs1", "de_spd", run_index=k), outputs_path)

    return exp_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_pipeline_scores_outputs(experiment_dir: Path) -> None:
    """Verify item_scores are produced for each successful output."""
    pipeline = EvaluationPipeline(experiment_dir)
    result = await pipeline.run(metric_names=["dummy"])

    assert "dummy" in result.metrics
    item_scores = result.metrics["dummy"].item_scores
    assert len(item_scores) == 2
    assert all(s.metric_name == "dummy" for s in item_scores)
    assert all(s.scores["correct"] == 1.0 for s in item_scores)


async def test_pipeline_aggregates_runs(experiment_dir_with_k_runs: Path) -> None:
    """Verify aggregated_scores combine K runs (k_runs=3)."""
    pipeline = EvaluationPipeline(experiment_dir_with_k_runs)
    result = await pipeline.run(metric_names=["dummy"])

    aggregated = result.metrics["dummy"].aggregated_scores
    assert len(aggregated) == 1
    assert aggregated[0].k_runs == 3
    assert aggregated[0].scores["correct"] == 1.0
    assert aggregated[0].aggregation_method == "mean"


async def test_pipeline_persists_scores(experiment_dir: Path) -> None:
    """Verify score files exist on disk after run."""
    pipeline = EvaluationPipeline(experiment_dir)
    await pipeline.run(metric_names=["dummy"])

    scores_dir = experiment_dir / "scores" / "dummy"
    assert (scores_dir / "item_scores.jsonl").exists()
    assert (scores_dir / "aggregated_scores.jsonl").exists()


async def test_pipeline_skips_existing(experiment_dir: Path) -> None:
    """Run twice, second run should load from disk (check result still has scores)."""
    pipeline = EvaluationPipeline(experiment_dir)

    # First run — computes and persists
    result1 = await pipeline.run(metric_names=["dummy"])
    assert len(result1.metrics["dummy"].item_scores) == 2

    # Second run — should load from disk
    result2 = await pipeline.run(metric_names=["dummy"])
    assert len(result2.metrics["dummy"].item_scores) == 2
    assert len(result2.metrics["dummy"].aggregated_scores) == len(
        result1.metrics["dummy"].aggregated_scores
    )


async def test_pipeline_force_recomputes(experiment_dir: Path) -> None:
    """Run with force=True on existing scores — should recompute."""
    pipeline = EvaluationPipeline(experiment_dir)

    # First run
    await pipeline.run(metric_names=["dummy"])

    # Force recompute
    result = await pipeline.run(metric_names=["dummy"], force=True)
    assert len(result.metrics["dummy"].item_scores) == 2


async def test_pipeline_applies_filter(experiment_dir: Path) -> None:
    """Filter to one party, verify only that party's scores."""
    pipeline = EvaluationPipeline(experiment_dir)
    result = await pipeline.run(
        metric_names=["dummy"],
        output_filter=OutputFilter(party_ids=["de_spd"]),
    )

    item_scores = result.metrics["dummy"].item_scores
    assert len(item_scores) == 1
    assert item_scores[0].party_id == "de_spd"


async def test_pipeline_skips_failed_outputs(tmp_path: Path) -> None:
    """Include a PARSE_ERROR output, verify it's skipped."""
    exp_dir = tmp_path / "test_experiment_fail"
    exp_dir.mkdir()

    gt_items = [_make_gt_item("obs1", "de_spd"), _make_gt_item("obs2", "de_cdu")]
    save_ground_truth(gt_items, exp_dir / "ground_truth.jsonl")

    outputs_path = exp_dir / "outputs.jsonl"
    append_output_to_path(_make_output("obs1", "de_spd"), outputs_path)
    append_output_to_path(
        _make_output("obs2", "de_cdu", status=OutputStatus.PARSE_ERROR), outputs_path
    )

    pipeline = EvaluationPipeline(exp_dir)
    result = await pipeline.run(metric_names=["dummy"])

    # Only the SUCCESS output should be scored
    assert len(result.metrics["dummy"].item_scores) == 1
    assert result.metrics["dummy"].item_scores[0].observation_id == "obs1"


async def test_pipeline_unknown_metric(experiment_dir: Path) -> None:
    """Request metric not in registry, verify it's skipped gracefully."""
    pipeline = EvaluationPipeline(experiment_dir)
    result = await pipeline.run(metric_names=["nonexistent_metric"])

    assert "nonexistent_metric" not in result.metrics
    assert len(result.metrics) == 0


# ---------------------------------------------------------------------------
# Factory-based metric resolution (experiment_config path)
# ---------------------------------------------------------------------------


class FactoryDummyMetric(BaseMetric):
    """Metric only available via factory, not registry."""

    @property
    def name(self) -> str:
        return "factory_dummy"

    @property
    def aggregation_method(self) -> str:
        return "mean"

    @property
    def primary_score_key(self) -> str:
        return "value"

    async def score(self, output: LLMOutput, ground_truth: DatasetItem) -> dict[str, float]:
        return {"value": 1.0}

    def aggregate(self, run_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["value"] for s in run_scores]
        return {"value": sum(values) / len(values)}

    def summarize(self, batch_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["value"] for s in batch_scores]
        return {"mean_value": sum(values) / len(values)}


async def test_pipeline_resolves_factory_metric(experiment_dir: Path) -> None:
    """Verify factory metrics are resolved when experiment_config is provided."""
    from unittest.mock import patch

    factory_metrics = {"factory_dummy": FactoryDummyMetric}

    pipeline = EvaluationPipeline(experiment_dir)
    with patch.object(pipeline, "_build_factory_metrics", return_value=factory_metrics):
        result = await pipeline.run(metric_names=["factory_dummy"])

    assert "factory_dummy" in result.metrics
    assert len(result.metrics["factory_dummy"].item_scores) == 2


async def test_pipeline_prefers_registry_over_factory(experiment_dir: Path) -> None:
    """Registry metrics take precedence over factory metrics."""
    pipeline = EvaluationPipeline(experiment_dir)
    # "dummy" is in the registry (from autouse fixture)
    result = await pipeline.run(metric_names=["dummy"])

    assert "dummy" in result.metrics
    assert len(result.metrics["dummy"].item_scores) == 2


# ---------------------------------------------------------------------------
# Incomplete tracking (Fix 3c)
# ---------------------------------------------------------------------------


class PartialMetric(BaseMetric):
    """Metric that always raises PartialPanelResult."""

    @property
    def name(self) -> str:
        return "partial_metric"

    @property
    def aggregation_method(self) -> str:
        return "mean"

    @property
    def primary_score_key(self) -> str:
        return "value"

    async def score(self, output: LLMOutput, ground_truth: DatasetItem) -> dict[str, float]:
        raise PartialPanelResult(n_succeeded=2, n_expected=3)

    def aggregate(self, run_scores: list[dict[str, float]]) -> dict[str, float]:
        return {"value": 0.0}

    def summarize(self, batch_scores: list[dict[str, float]]) -> dict[str, float]:
        return {"value": 0.0}


@pytest.fixture
def register_partial_metric():
    """Register PartialMetric for incomplete-tracking tests, clean up after."""
    METRIC_REGISTRY["partial_metric"] = PartialMetric
    yield
    METRIC_REGISTRY.pop("partial_metric", None)


async def test_incomplete_written_on_partial_panel(
    tmp_path: Path, register_partial_metric: None
) -> None:
    """PartialPanelResult writes to incomplete.jsonl; no item_scores written."""
    exp_dir = tmp_path / "exp_partial"
    exp_dir.mkdir()

    gt_items = [_make_gt_item("obs1", "de_spd")]
    save_ground_truth(gt_items, exp_dir / "ground_truth.jsonl")

    outputs_path = exp_dir / "outputs.jsonl"
    append_output_to_path(_make_output("obs1", "de_spd"), outputs_path)

    pipeline = EvaluationPipeline(exp_dir)
    await pipeline.run(metric_names=["partial_metric"])

    scores_dir = exp_dir / "scores" / "partial_metric"
    incomplete_path = scores_dir / "incomplete.jsonl"

    # incomplete.jsonl must exist with exactly 1 record
    assert incomplete_path.exists()
    lines = [ln for ln in incomplete_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["observation_id"] == "obs1"
    assert record["model_id"] == "model_a"
    assert record["prompt_variation"] == PromptVariation.MINIMAL.value
    assert record["n_succeeded"] == 2
    assert record["n_expected"] == 3

    # No item scores should have been written
    assert not (scores_dir / "item_scores.jsonl").exists()


def test_scores_complete_false_when_incomplete_nonempty(tmp_path: Path) -> None:
    """_scores_complete returns False when incomplete.jsonl has content."""
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "item_scores.jsonl").write_text('{"x": 1}\n')
    (scores_dir / "aggregated_scores.jsonl").write_text('{"x": 1}\n')
    (scores_dir / "incomplete.jsonl").write_text('{"observation_id": "obs1"}\n')

    pipeline = EvaluationPipeline(tmp_path)
    assert pipeline._scores_complete(scores_dir) is False


def test_scores_complete_true_when_incomplete_absent(tmp_path: Path) -> None:
    """_scores_complete returns True when both score files exist and no incomplete."""
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "item_scores.jsonl").write_text('{"x": 1}\n')
    (scores_dir / "aggregated_scores.jsonl").write_text('{"x": 1}\n')

    pipeline = EvaluationPipeline(tmp_path)
    assert pipeline._scores_complete(scores_dir) is True


def test_scores_complete_true_when_incomplete_empty(tmp_path: Path) -> None:
    """_scores_complete returns True when incomplete.jsonl exists but is empty."""
    scores_dir = tmp_path / "scores" / "dummy"
    scores_dir.mkdir(parents=True)
    (scores_dir / "item_scores.jsonl").write_text("")
    (scores_dir / "aggregated_scores.jsonl").write_text("")
    (scores_dir / "incomplete.jsonl").write_bytes(b"")  # zero-length file

    pipeline = EvaluationPipeline(tmp_path)
    assert pipeline._scores_complete(scores_dir) is True


def test_prune_incomplete_removes_resolved(tmp_path: Path) -> None:
    """_prune_incomplete keeps only unresolved records."""
    incomplete_path = tmp_path / "incomplete.jsonl"
    records = [
        {"observation_id": "obs1", "ie_name": "baseline", "model_id": "m",
         "prompt_variation": "minimal", "run_index": 0, "temperature": 0.0},
        {"observation_id": "obs2", "ie_name": "baseline", "model_id": "m",
         "prompt_variation": "minimal", "run_index": 0, "temperature": 0.0},
    ]
    incomplete_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    resolved_keys = {("obs1", "baseline", "m", "minimal", 0, 0.0)}
    _prune_incomplete(incomplete_path, resolved_keys)

    assert incomplete_path.exists()
    lines = [ln for ln in incomplete_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["observation_id"] == "obs2"


def test_prune_incomplete_deletes_file_when_all_resolved(tmp_path: Path) -> None:
    """_prune_incomplete deletes incomplete.jsonl when all entries are resolved."""
    incomplete_path = tmp_path / "incomplete.jsonl"
    record = {"observation_id": "obs1", "ie_name": "baseline", "model_id": "m",
              "prompt_variation": "minimal", "run_index": 0, "temperature": 0.0}
    incomplete_path.write_text(json.dumps(record) + "\n")

    resolved_keys = {("obs1", "baseline", "m", "minimal", 0, 0.0)}
    _prune_incomplete(incomplete_path, resolved_keys)

    assert not incomplete_path.exists()


def test_scores_complete_false_when_item_scores_missing(tmp_path: Path) -> None:
    """_scores_complete returns False when item_scores.jsonl is absent."""
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "aggregated_scores.jsonl").write_text('{"x": 1}\n')

    pipeline = EvaluationPipeline(tmp_path)
    assert pipeline._scores_complete(scores_dir) is False


# ---------------------------------------------------------------------------
# Second-pass retry (same-run)
# ---------------------------------------------------------------------------


class _FlipFlopMetric(BaseMetric):
    """Raises PartialPanelResult on the first call per observation, succeeds on the second."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    @property
    def name(self) -> str:
        return "flip_flop"

    @property
    def aggregation_method(self) -> str:
        return "mean"

    @property
    def primary_score_key(self) -> str:
        return "value"

    async def score(self, output, ground_truth) -> dict[str, float]:
        obs_id = output.observation_id
        if obs_id not in self._seen:
            self._seen.add(obs_id)
            raise PartialPanelResult(n_succeeded=1, n_expected=2)
        return {"value": 1.0}

    def aggregate(self, run_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["value"] for s in run_scores]
        return {"value": sum(values) / len(values)}

    def summarize(self, batch_scores: list[dict[str, float]]) -> dict[str, float]:
        return {"value": 1.0}


@pytest.fixture
def register_flip_flop_metric():
    """Register _FlipFlopMetric for second-pass tests, clean up after."""
    METRIC_REGISTRY["flip_flop"] = _FlipFlopMetric
    yield
    METRIC_REGISTRY.pop("flip_flop", None)


async def test_second_pass_retries_incomplete(
    tmp_path: Path, register_flip_flop_metric: None
) -> None:
    """After a PartialPanelResult on pass 1, the second pass should succeed and
    produce an item_score, leaving incomplete.jsonl absent."""
    exp_dir = tmp_path / "exp_second_pass"
    exp_dir.mkdir()

    gt_items = [_make_gt_item("obs1", "de_spd")]
    save_ground_truth(gt_items, exp_dir / "ground_truth.jsonl")
    append_output_to_path(_make_output("obs1", "de_spd"), exp_dir / "outputs.jsonl")

    pipeline = EvaluationPipeline(exp_dir)
    result = await pipeline.run(metric_names=["flip_flop"])

    scores_dir = exp_dir / "scores" / "flip_flop"

    # item_scores.jsonl must have 1 entry (from second pass)
    item_scores = result.metrics["flip_flop"].item_scores
    assert len(item_scores) == 1
    assert item_scores[0].scores["value"] == 1.0

    # incomplete.jsonl must be absent after second pass resolved it
    assert not (scores_dir / "incomplete.jsonl").exists()


async def test_second_pass_not_triggered_when_no_incomplete(
    experiment_dir: Path,
) -> None:
    """When all outputs score cleanly, second_pass_starting must NOT be logged."""
    import structlog.testing

    pipeline = EvaluationPipeline(experiment_dir)

    with structlog.testing.capture_logs() as captured:
        await pipeline.run(metric_names=["dummy"])

    assert not any(entry.get("event") == "second_pass_starting" for entry in captured)
