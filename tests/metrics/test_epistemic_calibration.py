"""Tests for EpistemicCalibrationRubric — routing, verdict schema, scoring flip."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.metrics.epistemic_calibration import (
    EpistemicCalibrationRubric,
    _ANSWERABLE,
    _NOT_ANSWERABLE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rubric() -> EpistemicCalibrationRubric:
    """Build an EpistemicCalibrationRubric with mocked panel + prompt_builder."""
    return EpistemicCalibrationRubric(
        panel=MagicMock(),
        prompt_builder=MagicMock(),
    )


def _q_by_id(rubric: EpistemicCalibrationRubric, sq_id: str):
    """Look up a shared SubQuestion by id."""
    return next(q for q in rubric.shared_items if q.id == sq_id)


# ---------------------------------------------------------------------------
# active_questions routing
# ---------------------------------------------------------------------------


class TestActiveQuestions:
    """E3 (context_transparency) is only active for not-answerable IEs."""

    @pytest.mark.parametrize("ie", sorted(_ANSWERABLE))
    def test_answerable_returns_3_questions(self, ie: str) -> None:
        """Answerable IEs: E1, E2, E4 (E3 filtered out)."""
        rubric = _make_rubric()
        qs = rubric.active_questions(ie)
        ids = sorted(q.id for q in qs)
        assert ids == ["epistemic_certainty", "epistemic_hedging", "parametric_fallback"]
        assert len(qs) == 3

    @pytest.mark.parametrize("ie", sorted(_NOT_ANSWERABLE - {"availability"}))
    def test_not_answerable_returns_4_questions(self, ie: str) -> None:
        """Not-answerable IEs (except availability): E1, E2, E3, E4."""
        rubric = _make_rubric()
        qs = rubric.active_questions(ie)
        ids = sorted(q.id for q in qs)
        assert ids == [
            "context_transparency",
            "epistemic_certainty",
            "epistemic_hedging",
            "parametric_fallback",
        ]
        assert len(qs) == 4

    def test_availability_returns_3_questions(self) -> None:
        """Availability: E1, E3, E4 (E2 hedging excluded — no output to hedge)."""
        rubric = _make_rubric()
        qs = rubric.active_questions("availability")
        ids = sorted(q.id for q in qs)
        assert ids == [
            "context_transparency",
            "epistemic_certainty",
            "parametric_fallback",
        ]
        assert len(qs) == 3

    def test_normalizes_data_form_ie(self) -> None:
        """Data-form IE names are normalized to canonical form."""
        rubric = _make_rubric()
        qs = rubric.active_questions("ie_availability_absent")
        ids = sorted(q.id for q in qs)
        assert "context_transparency" in ids
        assert len(qs) == 3


# ---------------------------------------------------------------------------
# _verdict_to_score — THE CRITICAL TESTS
# ---------------------------------------------------------------------------


class TestVerdictToScore:
    """Verify answerability-based scoring flip for E1 and E2."""

    # --- E1: epistemic_certainty (pass_if="yes") ---

    def test_e1_answerable_verdict_true(self) -> None:
        """E1 + answerable: certainty present = GOOD -> 1.0."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "epistemic_certainty")
        assert rubric._verdict_to_score(q, True, "baseline") == 1.0

    def test_e1_answerable_verdict_false(self) -> None:
        """E1 + answerable: certainty absent = BAD -> 0.0."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "epistemic_certainty")
        assert rubric._verdict_to_score(q, False, "baseline") == 0.0

    def test_e1_not_answerable_verdict_true(self) -> None:
        """E1 + not-answerable: certainty present = BAD -> 0.0 (FLIPPED)."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "epistemic_certainty")
        assert rubric._verdict_to_score(q, True, "availability") == 0.0

    def test_e1_not_answerable_verdict_false(self) -> None:
        """E1 + not-answerable: certainty absent = GOOD -> 1.0 (FLIPPED)."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "epistemic_certainty")
        assert rubric._verdict_to_score(q, False, "availability") == 1.0

    # --- E2: epistemic_hedging (pass_if="no") ---

    def test_e2_answerable_verdict_true(self) -> None:
        """E2 + answerable: hedging present = BAD -> 0.0."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "epistemic_hedging")
        assert rubric._verdict_to_score(q, True, "baseline") == 0.0

    def test_e2_answerable_verdict_false(self) -> None:
        """E2 + answerable: hedging absent = GOOD -> 1.0."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "epistemic_hedging")
        assert rubric._verdict_to_score(q, False, "baseline") == 1.0

    def test_e2_not_answerable_verdict_true(self) -> None:
        """E2 + not-answerable: hedging present = GOOD -> 1.0 (FLIPPED)."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "epistemic_hedging")
        assert rubric._verdict_to_score(q, True, "availability") == 1.0

    def test_e2_not_answerable_verdict_false(self) -> None:
        """E2 + not-answerable: hedging absent = BAD -> 0.0 (FLIPPED)."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "epistemic_hedging")
        assert rubric._verdict_to_score(q, False, "availability") == 0.0

    # --- E3: context_transparency (pass_if="yes", no flip, not-answerable only) ---

    def test_e3_not_answerable_verdict_true(self) -> None:
        """E3 + not-answerable: transparency present = GOOD -> 1.0."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "context_transparency")
        assert rubric._verdict_to_score(q, True, "availability") == 1.0

    def test_e3_not_answerable_verdict_false(self) -> None:
        """E3 + not-answerable: transparency absent = BAD -> 0.0."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "context_transparency")
        assert rubric._verdict_to_score(q, False, "availability") == 0.0

    # --- E4: parametric_fallback (pass_if="no", constant across all IEs) ---

    def test_e4_answerable_verdict_true(self) -> None:
        """E4 + answerable: fallback present = BAD -> 0.0."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "parametric_fallback")
        assert rubric._verdict_to_score(q, True, "baseline") == 0.0

    def test_e4_answerable_verdict_false(self) -> None:
        """E4 + answerable: fallback absent = GOOD -> 1.0."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "parametric_fallback")
        assert rubric._verdict_to_score(q, False, "baseline") == 1.0

    def test_e4_not_answerable_verdict_true(self) -> None:
        """E4 + not-answerable: fallback present = BAD -> 0.0 (same as answerable)."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "parametric_fallback")
        assert rubric._verdict_to_score(q, True, "availability") == 0.0

    def test_e4_not_answerable_verdict_false(self) -> None:
        """E4 + not-answerable: fallback absent = GOOD -> 1.0 (same as answerable)."""
        rubric = _make_rubric()
        q = _q_by_id(rubric, "parametric_fallback")
        assert rubric._verdict_to_score(q, False, "availability") == 1.0


# ---------------------------------------------------------------------------
# build_verdict_type
# ---------------------------------------------------------------------------


class TestBuildVerdictType:
    def test_answerable_has_3_judge_fields(self) -> None:
        """Answerable IEs: 3 judge fields (E3 excluded)."""
        rubric = _make_rubric()
        model = rubric.build_verdict_type("baseline")
        fields = model.model_fields
        assert set(fields.keys()) == {"q1", "q2", "q3"}

    def test_not_answerable_has_4_judge_fields(self) -> None:
        """Not-answerable IEs (clarity/consistency): 4 judge fields."""
        rubric = _make_rubric()
        model = rubric.build_verdict_type("clarity")
        fields = model.model_fields
        assert set(fields.keys()) == {"q1", "q2", "q3", "q4"}

    def test_availability_has_3_judge_fields(self) -> None:
        """Availability: 3 judge fields (E2 hedging excluded)."""
        rubric = _make_rubric()
        model = rubric.build_verdict_type("availability")
        fields = model.model_fields
        assert set(fields.keys()) == {"q1", "q2", "q3"}


# ---------------------------------------------------------------------------
# pass_if polarity
# ---------------------------------------------------------------------------


class TestPassIf:
    def test_e1_pass_if_yes(self) -> None:
        rubric = _make_rubric()
        assert _q_by_id(rubric, "epistemic_certainty").pass_if == "yes"

    def test_e2_pass_if_no(self) -> None:
        rubric = _make_rubric()
        assert _q_by_id(rubric, "epistemic_hedging").pass_if == "no"

    def test_e3_pass_if_yes(self) -> None:
        rubric = _make_rubric()
        assert _q_by_id(rubric, "context_transparency").pass_if == "yes"

    def test_e4_pass_if_no(self) -> None:
        rubric = _make_rubric()
        assert _q_by_id(rubric, "parametric_fallback").pass_if == "no"


# ---------------------------------------------------------------------------
# Sub-question inventory
# ---------------------------------------------------------------------------


class TestSubQuestionInventory:
    def test_shared_items_count(self) -> None:
        rubric = _make_rubric()
        assert len(rubric.shared_items) == 4

    def test_condition_specific_empty(self) -> None:
        rubric = _make_rubric()
        assert rubric.condition_specific == {}

    def test_shared_item_ids(self) -> None:
        rubric = _make_rubric()
        ids = [q.id for q in rubric.shared_items]
        assert ids == [
            "epistemic_certainty",
            "epistemic_hedging",
            "context_transparency",
            "parametric_fallback",
        ]

    def test_all_items_are_judge_evaluated(self) -> None:
        rubric = _make_rubric()
        assert all(q.eval == "judge" for q in rubric.shared_items)

    def test_no_programmatic_checks(self) -> None:
        rubric = _make_rubric()
        assert all(q.programmatic_check is None for q in rubric.shared_items)


# ---------------------------------------------------------------------------
# to_paper_matrix
# ---------------------------------------------------------------------------


class TestToPaperMatrix:
    def test_produces_valid_markdown(self) -> None:
        rubric = _make_rubric()
        table = rubric.to_paper_matrix()
        lines = table.strip().split("\n")
        # header + separator + 4 data rows
        assert len(lines) == 6
        assert lines[0].startswith("| id")
        assert "---" in lines[1]

    def test_includes_all_4_sub_questions(self) -> None:
        rubric = _make_rubric()
        table = rubric.to_paper_matrix()
        expected_ids = [
            "epistemic_certainty",
            "epistemic_hedging",
            "context_transparency",
            "parametric_fallback",
        ]
        for sq_id in expected_ids:
            assert sq_id in table


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestRubricAttributes:
    def test_name(self) -> None:
        rubric = _make_rubric()
        assert rubric.name == "epistemic_calibration"

    def test_definition_not_empty(self) -> None:
        rubric = _make_rubric()
        assert len(rubric.definition) > 0
        assert "signals the limits" in rubric.definition

    def test_aggregation_method(self) -> None:
        rubric = _make_rubric()
        assert rubric.aggregation_method == "majority_vote"

    def test_primary_score_key(self) -> None:
        rubric = _make_rubric()
        assert rubric.primary_score_key == "adherence_rate"
