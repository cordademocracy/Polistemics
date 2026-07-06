"""Tests for the rubric dispatcher."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from src.common.schemas import (
    DatasetItem,
    LLMOutput,
    OutputStatus,
    PromptVariation,
    RefusalType,
    StanceLabel,
)
from src.evaluate.dispatcher import (
    _evaluate_judge_questions,
    _evaluate_programmatic_questions,
    _majority_vote,
    dispatch,
)
from src.evaluate.panel import PanelResult, PartialPanelResult, SingleJudgeResponse
from src.metrics.rubric import BaseRubric, SubQuestion

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gt(
    ie_name: str = "baseline",
    party_name: str = "SPD",
) -> DatasetItem:
    return DatasetItem(
        observation_id="obs1",
        election_id="btw2025",
        party_id="de_spd",
        party_name=party_name,
        party_anonymized="Party 01",
        statement_id="s001",
        statement_number=None,
        statement_text="Test statement.",
        statement_category="Wirtschaft",
        stance_label=StanceLabel.AGREE,
        rationale_text=None,
        has_rationale=False,
        ie_name=ie_name,
        ie_chunks=["Some evidence chunk."],
    )


def _make_output(
    predicted_stance: StanceLabel = StanceLabel.AGREE,
) -> LLMOutput:
    return LLMOutput(
        observation_id="obs1",
        statement_id="s001",
        party_id="de_spd",
        experiment_id="test",
        model_id="model_a",
        prompt_variation=PromptVariation.DEFAULT,
        run_index=0,
        temperature=0.0,
        predicted_stance=predicted_stance,
        predicted_explanation="The party supports this.",
        timestamp=datetime.now(UTC),
        latency_ms=100.0,
        tokens_input=50,
        tokens_output=100,
        cost_usd=0.001,
        status=OutputStatus.SUCCESS,
        error_message=None,
        refusal_type=RefusalType.NONE,
        ie_name="baseline",
        condition_id="baseline__real__de__none",
    )


def _make_verdict(**fields: bool) -> BaseModel:
    """Build a dynamic Pydantic model instance from field name/value pairs."""
    from pydantic import create_model

    field_defs = {k: (bool, ...) for k in fields}
    model_cls = create_model("TestVerdict", **field_defs)
    return model_cls(**fields)


def _make_judge_response(
    model_id: str,
    verdict_fields: dict[str, bool] | None = None,
) -> SingleJudgeResponse:
    result = None
    if verdict_fields is not None:
        result = _make_verdict(**verdict_fields)
    return SingleJudgeResponse(
        model_id=model_id,
        result=result,
        latency_ms=100.0,
    )


def _make_panel_result(
    responses: list[SingleJudgeResponse],
) -> PanelResult:
    return PanelResult(responses=responses)


class _TestRubric(BaseRubric):
    """Test rubric with mixed judge + programmatic sub-questions."""

    name = "test_rubric"
    definition = "Test rubric for dispatcher tests."

    shared_items = [
        SubQuestion(
            id="judge_q1",
            text="Is the answer faithful?",
            eval="judge",
            comparability="shared",
        ),
        SubQuestion(
            id="prog_q1",
            text="Is the stance correct?",
            eval="programmatic",
            comparability="shared",
            programmatic_check="check_stance",
        ),
    ]

    condition_specific = {}

    def check_stance(self, observation: LLMOutput, gt: DatasetItem) -> bool:
        return observation.predicted_stance == gt.stance_label


def _make_mock_panel(
    panel_result: PanelResult,
    model_ids: list[str] | None = None,
) -> MagicMock:
    panel = MagicMock()
    panel.evaluate = AsyncMock(return_value=panel_result)
    panel.write_audit = AsyncMock()
    panel.model_ids = model_ids or ["m1", "m2", "m3"]
    return panel


def _make_mock_prompt_builder() -> MagicMock:
    pb = MagicMock()
    pb.build_judge_prompt = MagicMock(return_value=("system", "user"))
    return pb


# ---------------------------------------------------------------------------
# Unit tests: _majority_vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_all_true(self) -> None:
        assert _majority_vote([True, True, True]) is True

    def test_all_false(self) -> None:
        assert _majority_vote([False, False, False]) is False

    def test_two_out_of_three(self) -> None:
        assert _majority_vote([True, True, False]) is True

    def test_one_out_of_three(self) -> None:
        assert _majority_vote([True, False, False]) is False

    def test_single_true(self) -> None:
        assert _majority_vote([True]) is True

    def test_single_false(self) -> None:
        assert _majority_vote([False]) is False


# ---------------------------------------------------------------------------
# Unit tests: _evaluate_judge_questions
# ---------------------------------------------------------------------------


class TestEvaluateJudgeQuestions:
    def test_unanimous_true(self) -> None:
        judge_qs = [SubQuestion(id="q1", text="t", eval="judge", comparability="shared")]
        responses = [
            _make_judge_response("m1", {"q1": True}),
            _make_judge_response("m2", {"q1": True}),
            _make_judge_response("m3", {"q1": True}),
        ]
        panel_result = _make_panel_result(responses)
        results = _evaluate_judge_questions(judge_qs, panel_result)

        assert len(results) == 1
        assert results[0].verdict is True
        assert results[0].disagreement is False
        assert results[0].source == "judge"

    def test_majority_true_with_disagreement(self) -> None:
        judge_qs = [SubQuestion(id="q1", text="t", eval="judge", comparability="shared")]
        responses = [
            _make_judge_response("m1", {"q1": True}),
            _make_judge_response("m2", {"q1": True}),
            _make_judge_response("m3", {"q1": False}),
        ]
        panel_result = _make_panel_result(responses)
        results = _evaluate_judge_questions(judge_qs, panel_result)

        assert results[0].verdict is True
        assert results[0].disagreement is True

    def test_majority_false(self) -> None:
        judge_qs = [SubQuestion(id="q1", text="t", eval="judge", comparability="shared")]
        responses = [
            _make_judge_response("m1", {"q1": False}),
            _make_judge_response("m2", {"q1": False}),
            _make_judge_response("m3", {"q1": True}),
        ]
        panel_result = _make_panel_result(responses)
        results = _evaluate_judge_questions(judge_qs, panel_result)

        assert results[0].verdict is False
        assert results[0].disagreement is True

    def test_skips_failed_judges(self) -> None:
        judge_qs = [SubQuestion(id="q1", text="t", eval="judge", comparability="shared")]
        responses = [
            _make_judge_response("m1", {"q1": True}),
            _make_judge_response("m2", None),  # failed
            _make_judge_response("m3", {"q1": True}),
        ]
        panel_result = _make_panel_result(responses)
        results = _evaluate_judge_questions(judge_qs, panel_result)

        assert results[0].verdict is True
        assert len(results[0].per_judge) == 2

    def test_multiple_questions(self) -> None:
        judge_qs = [
            SubQuestion(id="q1", text="t1", eval="judge", comparability="shared"),
            SubQuestion(id="q2", text="t2", eval="judge", comparability="shared"),
        ]
        responses = [
            _make_judge_response("m1", {"q1": True, "q2": False}),
            _make_judge_response("m2", {"q1": True, "q2": True}),
            _make_judge_response("m3", {"q1": False, "q2": True}),
        ]
        panel_result = _make_panel_result(responses)
        results = _evaluate_judge_questions(judge_qs, panel_result)

        assert results[0].sub_question_id == "q1"
        assert results[0].verdict is True
        assert results[1].sub_question_id == "q2"
        assert results[1].verdict is True


# ---------------------------------------------------------------------------
# Unit tests: _evaluate_programmatic_questions
# ---------------------------------------------------------------------------


class TestEvaluateProgrammaticQuestions:
    def test_correct_stance(self) -> None:
        rubric = _TestRubric(panel=MagicMock(), prompt_builder=MagicMock())
        prog_qs = [
            SubQuestion(
                id="prog_q1", text="t", eval="programmatic",
                comparability="shared", programmatic_check="check_stance",
            ),
        ]
        gt = _make_gt()
        output = _make_output(predicted_stance=StanceLabel.AGREE)
        results = _evaluate_programmatic_questions(prog_qs, rubric, output, gt)

        assert len(results) == 1
        assert results[0].verdict is True
        assert results[0].source == "programmatic"
        assert results[0].per_judge is None
        assert results[0].disagreement is False

    def test_incorrect_stance(self) -> None:
        rubric = _TestRubric(panel=MagicMock(), prompt_builder=MagicMock())
        prog_qs = [
            SubQuestion(
                id="prog_q1", text="t", eval="programmatic",
                comparability="shared", programmatic_check="check_stance",
            ),
        ]
        gt = _make_gt()
        output = _make_output(predicted_stance=StanceLabel.DISAGREE)
        results = _evaluate_programmatic_questions(prog_qs, rubric, output, gt)

        assert results[0].verdict is False


# ---------------------------------------------------------------------------
# Integration tests: dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    @pytest.mark.asyncio
    async def test_mixed_judge_and_programmatic(self) -> None:
        responses = [
            _make_judge_response("m1", {"q1": True}),
            _make_judge_response("m2", {"q1": True}),
            _make_judge_response("m3", {"q1": False}),
        ]
        panel_result = _make_panel_result(responses)
        panel = _make_mock_panel(panel_result)
        pb = _make_mock_prompt_builder()
        rubric = _TestRubric(panel=panel, prompt_builder=pb)

        gt = _make_gt()
        output = _make_output()
        results = await dispatch(rubric, "baseline", output, gt, panel, pb)

        assert len(results) == 2
        judge_r = next(r for r in results if r.source == "judge")
        prog_r = next(r for r in results if r.source == "programmatic")
        assert judge_r.verdict is True
        assert judge_r.disagreement is True
        assert prog_r.verdict is True
        assert prog_r.disagreement is False

    @pytest.mark.asyncio
    async def test_partial_panel_raises(self) -> None:
        responses = [
            _make_judge_response("m1", {"q1": True}),
            _make_judge_response("m2", None),  # failed
            _make_judge_response("m3", None),  # failed
        ]
        panel_result = _make_panel_result(responses)
        panel = _make_mock_panel(panel_result)
        pb = _make_mock_prompt_builder()
        rubric = _TestRubric(panel=panel, prompt_builder=pb)

        gt = _make_gt()
        output = _make_output()
        with pytest.raises(PartialPanelResult) as exc_info:
            await dispatch(rubric, "baseline", output, gt, panel, pb)

        assert exc_info.value.n_succeeded == 1
        assert exc_info.value.n_expected == 3

    @pytest.mark.asyncio
    async def test_name_pattern_passed_to_prompt_builder(self) -> None:
        """Dispatcher passes name_pattern=None (no name stripping in judge prompt)."""
        responses = [
            _make_judge_response("m1", {"q1": True}),
            _make_judge_response("m2", {"q1": True}),
            _make_judge_response("m3", {"q1": True}),
        ]
        panel_result = _make_panel_result(responses)
        panel = _make_mock_panel(panel_result)
        pb = _make_mock_prompt_builder()
        rubric = _TestRubric(panel=panel, prompt_builder=pb)

        gt = _make_gt(party_name="CDU")
        output = _make_output()
        await dispatch(rubric, "baseline", output, gt, panel, pb)

        call_args = pb.build_judge_prompt.call_args
        assert call_args.kwargs.get("name_pattern") is None

    @pytest.mark.asyncio
    async def test_programmatic_only_no_panel_call(self) -> None:
        """When no judge sub-questions are active, panel.evaluate is not called."""

        class _ProgOnlyRubric(BaseRubric):
            name = "prog_only"
            definition = "Only programmatic."
            shared_items = [
                SubQuestion(
                    id="prog_q", text="t", eval="programmatic",
                    comparability="shared", programmatic_check="check_stance",
                ),
            ]
            condition_specific = {}

            def check_stance(self, observation, gt) -> bool:
                return True

        panel = _make_mock_panel(_make_panel_result([]))
        pb = _make_mock_prompt_builder()
        rubric = _ProgOnlyRubric(panel=panel, prompt_builder=pb)

        gt = _make_gt()
        output = _make_output()
        results = await dispatch(rubric, "baseline", output, gt, panel, pb)

        assert len(results) == 1
        assert results[0].source == "programmatic"
        panel.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_unanimous_panel_no_disagreement(self) -> None:
        responses = [
            _make_judge_response("m1", {"q1": True}),
            _make_judge_response("m2", {"q1": True}),
            _make_judge_response("m3", {"q1": True}),
        ]
        panel_result = _make_panel_result(responses)
        panel = _make_mock_panel(panel_result)
        pb = _make_mock_prompt_builder()
        rubric = _TestRubric(panel=panel, prompt_builder=pb)

        gt = _make_gt()
        output = _make_output()
        results = await dispatch(rubric, "baseline", output, gt, panel, pb)

        judge_r = next(r for r in results if r.source == "judge")
        assert judge_r.disagreement is False
