"""Integration smoke test for FaithfulnessRubric end-to-end dispatch.

Verifies that rubric.score() dispatches correctly for baseline IE:
  - 2 judge sub-questions (position_representation, information_fabrication)
  - pass_if polarity applied correctly in scoring
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import create_model

from src.common.prompts import PromptBuilder
from src.common.schemas import (
    DatasetItem,
    LLMOutput,
    OutputStatus,
    PromptVariation,
    RefusalType,
    StanceLabel,
)
from src.evaluate.panel import PanelResult, SingleJudgeResponse
from src.metrics.faithfulness import FaithfulnessRubric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gt(ie_name: str = "baseline") -> DatasetItem:
    return DatasetItem(
        observation_id="smoke_obs",
        election_id="btw2025",
        party_id="de_spd",
        party_name="SPD",
        party_anonymized="Party 01",
        statement_id="s001",
        statement_number=1,
        statement_text="Germany should expand renewable energy.",
        statement_category="Energie",
        stance_label=StanceLabel.AGREE,
        rationale_text="The party supports solar expansion.",
        has_rationale=True,
        ie_name=ie_name,
        ie_chunks=["SPD supports expanding renewable energy sources."],
    )


def _make_output(ie_name: str = "baseline") -> LLMOutput:
    return LLMOutput(
        observation_id="smoke_obs",
        statement_id="s001",
        party_id="de_spd",
        experiment_id="smoke_test",
        model_id="mock_model",
        prompt_variation=PromptVariation.DEFAULT,
        run_index=0,
        temperature=0.0,
        predicted_stance=StanceLabel.AGREE,
        predicted_explanation="The party agrees with expanding renewable energy.",
        timestamp=datetime.now(UTC),
        latency_ms=100.0,
        tokens_input=50,
        tokens_output=100,
        cost_usd=None,
        status=OutputStatus.SUCCESS,
        error_message=None,
        refusal_type=RefusalType.NONE,
        ie_name=ie_name,
        condition_id="baseline__real__evidence__none",
    )


def _make_mock_panel(verdict_fields: dict[str, bool]) -> MagicMock:
    """Build a mock JudgePanel returning unanimous verdicts for given fields."""
    field_defs = {k: (bool, ...) for k in verdict_fields}
    verdict_type = create_model("MockVerdict", **field_defs)
    verdict = verdict_type(**verdict_fields)

    responses = [
        SingleJudgeResponse(model_id=f"judge_{i}", result=verdict, latency_ms=100.0)
        for i in range(3)
    ]
    panel_result = PanelResult(responses=responses)

    panel = MagicMock()
    panel.evaluate = AsyncMock(return_value=panel_result)
    panel.write_audit = AsyncMock()
    panel.model_ids = ["judge_0", "judge_1", "judge_2"]
    return panel


def _make_mock_prompt_builder() -> MagicMock:
    """Build a mock PromptBuilder returning stub prompts."""
    pb = MagicMock(spec=PromptBuilder)
    pb.build_judge_prompt = MagicMock(
        return_value=("You are an expert evaluator...", "QUERY: ...\nCONTEXT: ...\nANSWER: ...")
    )
    return pb


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_faithfulness_baseline_smoke() -> None:
    """End-to-end: FaithfulnessRubric.score() for baseline IE.

    Baseline activates 2 judge sub-questions:
      - position_representation (pass_if=yes, verdict=True -> 1.0)
      - information_fabrication (pass_if=no, verdict=False -> 1.0)
    """
    panel = _make_mock_panel({
        "q1": True,
        "q2": False,  # no fabrication found = good
    })
    pb = _make_mock_prompt_builder()

    rubric = FaithfulnessRubric(panel=panel, prompt_builder=pb)
    output = _make_output()
    gt = _make_gt()

    scores = await rubric.score(output, gt)

    # Both sub-questions should pass
    assert "position_representation" in scores
    assert "information_fabrication" in scores
    assert len(scores) == 2

    assert scores["position_representation"] == 1.0
    assert scores["information_fabrication"] == 1.0  # pass_if=no, verdict=False -> 1.0

    # Panel should have been called exactly once (both judge Qs in one call)
    panel.evaluate.assert_called_once()

    # Audit write should have been called
    panel.write_audit.assert_called_once()


@pytest.mark.asyncio
async def test_faithfulness_baseline_fabrication_detected() -> None:
    """When fabrication is detected (verdict=True), pass_if=no inverts to 0.0."""
    panel = _make_mock_panel({
        "q1": True,
        "q2": True,  # fabrication found = bad
    })
    pb = _make_mock_prompt_builder()

    rubric = FaithfulnessRubric(panel=panel, prompt_builder=pb)
    output = _make_output()
    gt = _make_gt()

    scores = await rubric.score(output, gt)

    assert scores["position_representation"] == 1.0
    # pass_if=no, verdict=True -> score 0.0 (fabrication detected = fail)
    assert scores["information_fabrication"] == 0.0


@pytest.mark.asyncio
async def test_faithfulness_availability_empty() -> None:
    """Availability IE: no sub-questions active, no panel call, empty scores."""
    panel = _make_mock_panel({})
    pb = _make_mock_prompt_builder()

    rubric = FaithfulnessRubric(panel=panel, prompt_builder=pb)
    output = _make_output(ie_name="availability")
    gt = _make_gt(ie_name="availability")

    scores = await rubric.score(output, gt)

    # Rubric returns None (not empty dict) when no sub-questions are active for this IE.
    assert scores is None

    # No sub-questions -> panel NOT called
    panel.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_faithfulness_consistency_includes_false_synthesis() -> None:
    """Consistency IE: 2 shared + false_synthesis CS, all judge-evaluated."""
    panel = _make_mock_panel({
        "q1": True,
        "q2": False,
        "q3": False,  # no false synthesis = good
    })
    pb = _make_mock_prompt_builder()

    rubric = FaithfulnessRubric(panel=panel, prompt_builder=pb)
    output = _make_output(ie_name="consistency")
    gt = _make_gt(ie_name="consistency")

    scores = await rubric.score(output, gt)

    assert len(scores) == 3
    assert scores["position_representation"] == 1.0
    assert scores["information_fabrication"] == 1.0
    assert scores["false_synthesis"] == 1.0  # pass_if=no, verdict=False -> 1.0
