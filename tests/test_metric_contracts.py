"""Guardrail tests: primary_score_key must exist in aggregate() output for ALL metrics.

Prevents the class of bug where primary_score_key references a summarize()-only key
that doesn't exist in AggregatedScore.scores.
"""
from __future__ import annotations

from src.metrics.base import BaseMetric


class _ContractDummyMetric(BaseMetric):
    """Minimal concrete metric for contract tests."""

    @property
    def name(self) -> str:
        return "contract_dummy"

    @property
    def aggregation_method(self) -> str:
        return "mean"

    @property
    def primary_score_key(self) -> str:
        return "value"

    async def score(self, output, ground_truth) -> dict[str, float]:
        return {"value": 1.0}

    def aggregate(self, run_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["value"] for s in run_scores]
        return {"value": sum(values) / len(values)}

    def summarize(self, batch_scores: list[dict[str, float]]) -> dict[str, float]:
        values = [s["value"] for s in batch_scores]
        return {"value": sum(values) / len(values), "count": float(len(values))}


# --- Guardrail: primary_score_key ∈ aggregate() output ---


async def test_primary_key_in_aggregate() -> None:
    metric = _ContractDummyMetric()
    score = await metric.score(None, None)  # type: ignore[arg-type]
    agg = metric.aggregate([score])
    assert metric.primary_score_key in agg, (
        f"{metric.name}.primary_score_key='{metric.primary_score_key}' "
        f"not in aggregate() output keys: {list(agg.keys())}"
    )
