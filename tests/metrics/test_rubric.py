"""Tests for BaseRubric, SubQuestion, and SubQuestionResult."""
from __future__ import annotations

import pytest

from src.metrics.rubric import BaseRubric, SubQuestion, SubQuestionResult
from src.metrics.faithfulness import FaithfulnessRubric
from src.metrics.impartiality import ImpartialityRubric

# ---------------------------------------------------------------------------
# Test rubric subclass
# ---------------------------------------------------------------------------


class _TestRubric(BaseRubric):
    """Minimal concrete rubric for testing."""

    name = "test_rubric"
    definition = "A test rubric for unit testing."

    shared_items = [
        SubQuestion(
            id="shared_judge",
            text="Is the output correct?",
            eval="judge",
            comparability="shared",
            active_ies=None,
        ),
        SubQuestion(
            id="shared_filtered",
            text="Is the stance correct?",
            eval="programmatic",
            comparability="shared",
            programmatic_check="check_stance",
            active_ies=frozenset({"baseline", "noise"}),
        ),
        SubQuestion(
            id="negative_judge",
            text="Does the output fabricate information?",
            eval="judge",
            comparability="shared",
            active_ies=None,
            pass_if="no",
        ),
    ]

    condition_specific = {
        "clarity": [
            SubQuestion(
                id="clarity_check",
                text="Does the output handle vagueness?",
                eval="judge",
                comparability="condition-specific",
            ),
        ],
    }

    def check_stance(self, observation, gt) -> bool:
        return observation.predicted_stance == gt.stance_label


def _make_rubric() -> _TestRubric:
    """Build a _TestRubric with None deps (sufficient for non-score tests)."""
    return _TestRubric(panel=None, prompt_builder=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SubQuestion tests
# ---------------------------------------------------------------------------


class TestSubQuestion:
    def test_frozen(self) -> None:
        sq = SubQuestion(id="x", text="t", eval="judge", comparability="shared")
        with pytest.raises(AttributeError):
            sq.id = "y"  # type: ignore[misc]

    def test_defaults(self) -> None:
        sq = SubQuestion(id="x", text="t", eval="programmatic", comparability="shared")
        assert sq.programmatic_check is None
        assert sq.active_ies is None

    def test_active_ies_frozenset(self) -> None:
        sq = SubQuestion(
            id="x", text="t", eval="judge", comparability="shared",
            active_ies=frozenset({"baseline", "clarity"}),
        )
        assert "baseline" in sq.active_ies
        assert "noise" not in sq.active_ies

    def test_pass_if_defaults_to_yes(self) -> None:
        sq = SubQuestion(id="x", text="t", eval="judge", comparability="shared")
        assert sq.pass_if == "yes"

    def test_pass_if_no(self) -> None:
        sq = SubQuestion(
            id="x", text="t", eval="judge", comparability="shared", pass_if="no",
        )
        assert sq.pass_if == "no"


# ---------------------------------------------------------------------------
# SubQuestionResult tests
# ---------------------------------------------------------------------------


class TestSubQuestionResult:
    def test_frozen(self) -> None:
        r = SubQuestionResult(sub_question_id="x", verdict=True, source="judge")
        with pytest.raises(AttributeError):
            r.verdict = False  # type: ignore[misc]

    def test_defaults(self) -> None:
        r = SubQuestionResult(sub_question_id="x", verdict=False, source="programmatic")
        assert r.per_judge is None
        assert r.disagreement is False


# ---------------------------------------------------------------------------
# BaseRubric.active_questions tests
# ---------------------------------------------------------------------------


class TestActiveQuestions:
    def test_all_ies_includes_unfiltered_shared(self) -> None:
        rubric = _make_rubric()
        qs = rubric.active_questions("baseline")
        ids = [q.id for q in qs]
        assert "shared_judge" in ids

    def test_filtered_shared_included_when_ie_matches(self) -> None:
        rubric = _make_rubric()
        qs = rubric.active_questions("baseline")
        ids = [q.id for q in qs]
        assert "shared_filtered" in ids

    def test_filtered_shared_excluded_when_ie_not_in_active_ies(self) -> None:
        rubric = _make_rubric()
        qs = rubric.active_questions("clarity")
        ids = [q.id for q in qs]
        assert "shared_filtered" not in ids

    def test_condition_specific_appended(self) -> None:
        rubric = _make_rubric()
        qs = rubric.active_questions("clarity")
        ids = [q.id for q in qs]
        assert "clarity_check" in ids

    def test_condition_specific_not_appended_for_other_ie(self) -> None:
        rubric = _make_rubric()
        qs = rubric.active_questions("baseline")
        ids = [q.id for q in qs]
        assert "clarity_check" not in ids

    def test_normalizes_data_form_ie(self) -> None:
        rubric = _make_rubric()
        qs = rubric.active_questions("ie_clarity_vague")
        ids = [q.id for q in qs]
        assert "clarity_check" in ids
        assert "shared_filtered" not in ids


# ---------------------------------------------------------------------------
# BaseRubric.build_verdict_type tests
# ---------------------------------------------------------------------------


class TestBuildVerdictType:
    def test_returns_pydantic_model_with_judge_fields(self) -> None:
        rubric = _make_rubric()
        model = rubric.build_verdict_type("baseline")
        fields = model.model_fields
        assert "q1" in fields
        assert fields["q1"].annotation is bool

    def test_excludes_programmatic_fields(self) -> None:
        rubric = _make_rubric()
        model = rubric.build_verdict_type("baseline")
        assert "shared_filtered" not in model.model_fields

    def test_includes_condition_specific_judge_fields(self) -> None:
        rubric = _make_rubric()
        model = rubric.build_verdict_type("clarity")
        fields = model.model_fields
        assert "q1" in fields
        assert "q3" in fields

    def test_model_name_contains_rubric_and_ie(self) -> None:
        rubric = _make_rubric()
        model = rubric.build_verdict_type("baseline")
        assert "test_rubric" in model.__name__
        assert "baseline" in model.__name__

    def test_model_validates_correctly(self) -> None:
        rubric = _make_rubric()
        model = rubric.build_verdict_type("baseline")
        instance = model(q1=True, q2=False)
        assert instance.q1 is True


# ---------------------------------------------------------------------------
# BaseRubric.to_paper_matrix tests
# ---------------------------------------------------------------------------


class TestToPaperMatrix:
    def test_produces_markdown_table(self) -> None:
        rubric = _make_rubric()
        table = rubric.to_paper_matrix()
        lines = table.strip().split("\n")
        assert len(lines) >= 3
        assert lines[0].startswith("| id")
        assert "---" in lines[1]

    def test_includes_all_sub_questions(self) -> None:
        rubric = _make_rubric()
        table = rubric.to_paper_matrix()
        assert "shared_judge" in table
        assert "shared_filtered" in table
        assert "negative_judge" in table
        assert "clarity_check" in table

    def test_shows_active_ies_or_all(self) -> None:
        rubric = _make_rubric()
        table = rubric.to_paper_matrix()
        assert "all" in table
        assert "baseline" in table


# ---------------------------------------------------------------------------
# BaseRubric._verdict_to_score tests
# ---------------------------------------------------------------------------


class TestVerdictToScore:
    def test_pass_if_yes_verdict_true(self) -> None:
        rubric = _make_rubric()
        q = SubQuestion(id="x", text="t", eval="judge", comparability="shared", pass_if="yes")
        assert rubric._verdict_to_score(q, verdict=True, ie="baseline") == 1.0

    def test_pass_if_yes_verdict_false(self) -> None:
        rubric = _make_rubric()
        q = SubQuestion(id="x", text="t", eval="judge", comparability="shared", pass_if="yes")
        assert rubric._verdict_to_score(q, verdict=False, ie="baseline") == 0.0

    def test_pass_if_no_verdict_true(self) -> None:
        """Negative framing: "Does it fabricate?" — True means bad → score 0.0."""
        rubric = _make_rubric()
        q = SubQuestion(id="x", text="t", eval="judge", comparability="shared", pass_if="no")
        assert rubric._verdict_to_score(q, verdict=True, ie="baseline") == 0.0

    def test_pass_if_no_verdict_false(self) -> None:
        """Negative framing: "Does it fabricate?" — False means good → score 1.0."""
        rubric = _make_rubric()
        q = SubQuestion(id="x", text="t", eval="judge", comparability="shared", pass_if="no")
        assert rubric._verdict_to_score(q, verdict=False, ie="baseline") == 1.0

    def test_ie_parameter_accepted(self) -> None:
        """Verify ie parameter is accepted (for subclass override hooks)."""
        rubric = _make_rubric()
        q = SubQuestion(id="x", text="t", eval="judge", comparability="shared")
        # Should not raise regardless of ie value
        rubric._verdict_to_score(q, verdict=True, ie="consistency")


# ---------------------------------------------------------------------------
# BaseRubric.aggregate tests
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_single_run(self) -> None:
        rubric = _make_rubric()
        scores = [{"q1": 1.0, "q2": 0.0}]
        result = rubric.aggregate(scores)
        assert result["q1"] == 1.0
        assert result["q2"] == 0.0
        assert result["adherence_rate"] == 0.5

    def test_three_runs_majority_vote(self) -> None:
        rubric = _make_rubric()
        scores = [
            {"q1": 1.0, "q2": 0.0},
            {"q1": 1.0, "q2": 1.0},
            {"q1": 0.0, "q2": 0.0},
        ]
        result = rubric.aggregate(scores)
        assert result["q1"] == 1.0
        assert result["q2"] == 0.0
        assert result["adherence_rate"] == 0.5

    def test_empty_sub_questions_returns_sentinel(self) -> None:
        """No sub-question keys (only adherence_rate) → sentinel -1.0, not a failure 0.0."""
        rubric = _make_rubric()
        scores = [{"adherence_rate": 0.5}]
        result = rubric.aggregate(scores)
        assert result["adherence_rate"] == -1.0

    def test_empty_runs(self) -> None:
        rubric = _make_rubric()
        result = rubric.aggregate([])
        assert result["adherence_rate"] == 0.0


# ---------------------------------------------------------------------------
# BaseRubric.summarize tests
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_computes_mean_per_key(self) -> None:
        rubric = _make_rubric()
        batch = [
            {"q1": 1.0, "adherence_rate": 1.0},
            {"q1": 0.0, "adherence_rate": 0.0},
        ]
        result = rubric.summarize(batch)
        assert result["q1_mean"] == 0.5
        assert result["adherence_rate_mean"] == 0.5

    def test_empty_batch(self) -> None:
        rubric = _make_rubric()
        result = rubric.summarize([])
        assert result == {}

    def test_handles_missing_keys(self) -> None:
        rubric = _make_rubric()
        batch = [
            {"q1": 1.0, "q2": 1.0},
            {"q1": 0.0},
        ]
        result = rubric.summarize(batch)
        assert result["q1_mean"] == 0.5
        assert result["q2_mean"] == 1.0


# ---------------------------------------------------------------------------
# score() returns None for N/A IEs (no active sub-questions)
# ---------------------------------------------------------------------------


class TestScoreReturnsNoneForNAIE:
    """Verify that rubrics with no active sub-questions for an IE return None.

    This prevents N/A combinations from producing adherence_rate=0.0 in
    downstream aggregations.  The test does not require a live judge panel —
    active_questions() is checked before any dispatch call.
    """

    @pytest.mark.asyncio
    async def test_faithfulness_availability_returns_none(self) -> None:
        """FaithfulnessRubric has no active questions for ie_availability_absent."""
        from unittest.mock import MagicMock
        from src.common.schemas import DatasetItem, LLMOutput, StanceLabel, PromptVariation, OutputStatus, RefusalType
        from datetime import datetime

        rubric = FaithfulnessRubric(panel=MagicMock(), prompt_builder=MagicMock())

        gt = DatasetItem(
            observation_id="obs-1",
            election_id="elec-1",
            party_id="party-1",
            party_name="Test Party",
            party_anonymized="Party A",
            statement_id="stmt-1",
            statement_number=1,
            statement_text="Test statement",
            statement_category="economy",
            stance_label=StanceLabel.AGREE,
            rationale_text=None,
            has_rationale=False,
            ie_name="ie_availability_absent",
            ie_chunks=[],
        )
        out = LLMOutput(
            observation_id="obs-1",
            statement_id="stmt-1",
            party_id="party-1",
            experiment_id="exp-1",
            model_id="model-1",
            prompt_variation=PromptVariation.DEFAULT,
            run_index=0,
            temperature=0.0,
            predicted_stance=StanceLabel.AGREE,
            predicted_explanation="No evidence.",
            timestamp=datetime(2026, 1, 1),
            latency_ms=100.0,
            tokens_input=10,
            tokens_output=5,
            cost_usd=None,
            status=OutputStatus.SUCCESS,
            error_message=None,
            refusal_type=RefusalType.NONE,
            ie_name="ie_availability_absent",
            condition_id="ie_availability_absent__party-1__en__2024",
        )

        result = await rubric.score(out, gt)
        assert result is None

    @pytest.mark.asyncio
    async def test_impartiality_availability_returns_none(self) -> None:
        """ImpartialityRubric has no active questions for ie_availability_absent."""
        from unittest.mock import MagicMock
        from src.common.schemas import DatasetItem, LLMOutput, StanceLabel, PromptVariation, OutputStatus, RefusalType
        from datetime import datetime

        rubric = ImpartialityRubric(panel=MagicMock(), prompt_builder=MagicMock())

        gt = DatasetItem(
            observation_id="obs-2",
            election_id="elec-1",
            party_id="party-2",
            party_name="Test Party",
            party_anonymized="Party B",
            statement_id="stmt-2",
            statement_number=1,
            statement_text="Test statement",
            statement_category="economy",
            stance_label=StanceLabel.DISAGREE,
            rationale_text=None,
            has_rationale=False,
            ie_name="ie_availability_absent",
            ie_chunks=[],
        )
        out = LLMOutput(
            observation_id="obs-2",
            statement_id="stmt-2",
            party_id="party-2",
            experiment_id="exp-1",
            model_id="model-1",
            prompt_variation=PromptVariation.DEFAULT,
            run_index=0,
            temperature=0.0,
            predicted_stance=StanceLabel.DISAGREE,
            predicted_explanation="No evidence.",
            timestamp=datetime(2026, 1, 1),
            latency_ms=100.0,
            tokens_input=10,
            tokens_output=5,
            cost_usd=None,
            status=OutputStatus.SUCCESS,
            error_message=None,
            refusal_type=RefusalType.NONE,
            ie_name="ie_availability_absent",
            condition_id="ie_availability_absent__party-2__en__2024",
        )

        result = await rubric.score(out, gt)
        assert result is None

    @pytest.mark.asyncio
    async def test_faithfulness_baseline_does_not_return_none(self) -> None:
        """Baseline has active sub-questions — score() must dispatch, not short-circuit."""
        from unittest.mock import MagicMock, patch, AsyncMock
        from src.common.schemas import DatasetItem, LLMOutput, StanceLabel, PromptVariation, OutputStatus, RefusalType
        from src.metrics.rubric import SubQuestionResult
        from datetime import datetime

        rubric = FaithfulnessRubric(panel=MagicMock(), prompt_builder=MagicMock())

        gt = DatasetItem(
            observation_id="obs-3",
            election_id="elec-1",
            party_id="party-3",
            party_name="Test Party",
            party_anonymized="Party C",
            statement_id="stmt-3",
            statement_number=1,
            statement_text="Test statement",
            statement_category="economy",
            stance_label=StanceLabel.AGREE,
            rationale_text="Supports it.",
            has_rationale=True,
            ie_name="baseline",
            ie_chunks=["Some evidence chunk."],
        )
        out = LLMOutput(
            observation_id="obs-3",
            statement_id="stmt-3",
            party_id="party-3",
            experiment_id="exp-1",
            model_id="model-1",
            prompt_variation=PromptVariation.DEFAULT,
            run_index=0,
            temperature=0.0,
            predicted_stance=StanceLabel.AGREE,
            predicted_explanation="Evidence supports this.",
            timestamp=datetime(2026, 1, 1),
            latency_ms=100.0,
            tokens_input=10,
            tokens_output=5,
            cost_usd=None,
            status=OutputStatus.SUCCESS,
            error_message=None,
            refusal_type=RefusalType.NONE,
            ie_name="baseline",
            condition_id="baseline__party-3__en__2024",
        )

        mock_results = [
            SubQuestionResult(sub_question_id="position_representation", verdict=True, source="judge"),
            SubQuestionResult(sub_question_id="information_fabrication", verdict=False, source="judge"),
        ]
        with patch("src.evaluate.dispatcher.dispatch", new=AsyncMock(return_value=mock_results)):
            result = await rubric.score(out, gt)

        assert result is not None
        assert "position_representation" in result
        assert "information_fabrication" in result
