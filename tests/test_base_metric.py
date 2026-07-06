from __future__ import annotations

import pytest

from src.metrics.base import BaseMetric
from src.metrics.registry import METRIC_REGISTRY

# --- DummyMetric fixture ---

class DummyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def aggregation_method(self) -> str:
        return "mean"

    async def score(self, output, ground_truth) -> dict[str, float]:
        return {"value": 1.0}

    def aggregate(self, run_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["value"] for s in run_scores]
        return {"value": sum(values) / len(values)}

    @property
    def primary_score_key(self) -> str:
        return "mean"

    def summarize(self, batch_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["value"] for s in batch_scores]
        return {"mean": sum(values) / len(values), "count": float(len(values))}


# --- ABC enforcement tests ---

def test_cannot_instantiate_base_metric() -> None:
    """BaseMetric is abstract and must not be instantiable directly."""
    with pytest.raises(TypeError):
        BaseMetric()  # type: ignore[abstract]


def test_incomplete_subclass_raises() -> None:
    """A subclass that omits any abstract member raises TypeError on instantiation."""
    class IncompleteMetric(BaseMetric):
        @property
        def name(self) -> str:
            return "incomplete"
        # aggregation_method, score, aggregate, summarize are missing

    with pytest.raises(TypeError):
        IncompleteMetric()  # type: ignore[abstract]


# --- DummyMetric contract tests ---

def test_complete_subclass_works() -> None:
    """A fully implemented subclass can be instantiated with correct property values."""
    metric = DummyMetric()
    assert metric.name == "dummy"
    assert metric.aggregation_method == "mean"


def test_primary_score_key_property() -> None:
    """primary_score_key returns a string usable as dict key in summarize output."""
    metric = DummyMetric()
    summary = metric.summarize([{"value": 1.0}, {"value": 0.5}])
    assert metric.primary_score_key in summary


async def test_score_returns_dict() -> None:
    """score() returns a dict[str, float] (output/ground_truth args are unused by DummyMetric)."""
    metric = DummyMetric()
    result = await metric.score(None, None)  # type: ignore[arg-type]
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_aggregate_contract() -> None:
    """aggregate() receives list of score dicts and returns a score dict."""
    metric = DummyMetric()
    run_scores = [{"value": 1.0}, {"value": 0.0}, {"value": 1.0}]
    result = metric.aggregate(run_scores)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == pytest.approx(2 / 3)


def test_summarize_contract() -> None:
    """summarize() receives list of aggregated score dicts and returns a summary dict."""
    metric = DummyMetric()
    batch_scores = [{"value": 1.0}, {"value": 0.5}, {"value": 0.0}]
    result = metric.summarize(batch_scores)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "count" in result
    assert result["mean"] == pytest.approx(0.5)
    assert result["count"] == 3.0


# --- Registry tests ---

def test_registry_is_dict() -> None:
    """METRIC_REGISTRY is a plain dict."""
    assert isinstance(METRIC_REGISTRY, dict)


def test_registry_importable_after_import() -> None:
    """Importing src.metrics makes METRIC_REGISTRY available."""
    import src.metrics  # noqa: F401
    from src.metrics.registry import METRIC_REGISTRY
    assert isinstance(METRIC_REGISTRY, dict)
