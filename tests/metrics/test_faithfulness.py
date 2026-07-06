"""Tests for FaithfulnessRubric — sub-question routing, verdict schema, pass_if polarity."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.metrics.faithfulness import FaithfulnessRubric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rubric() -> FaithfulnessRubric:
    """Build a FaithfulnessRubric with mocked panel + prompt_builder."""
    return FaithfulnessRubric(
        panel=MagicMock(),
        prompt_builder=MagicMock(),
    )


# ---------------------------------------------------------------------------
# active_questions routing
# ---------------------------------------------------------------------------


class TestActiveQuestions:
    def test_baseline_returns_2_shared(self) -> None:
        """Baseline: position_representation + information_fabrication."""
        rubric = _make_rubric()
        qs = rubric.active_questions("baseline")
        ids = sorted(q.id for q in qs)
        assert ids == ["information_fabrication", "position_representation"]
        assert len(qs) == 2

    def test_clarity_returns_2_shared(self) -> None:
        """Clarity: position_representation + information_fabrication (no CS)."""
        rubric = _make_rubric()
        qs = rubric.active_questions("clarity")
        ids = sorted(q.id for q in qs)
        assert ids == ["information_fabrication", "position_representation"]
        assert len(qs) == 2

    def test_consistency_returns_3_items(self) -> None:
        """Consistency: 2 shared + false_synthesis CS."""
        rubric = _make_rubric()
        qs = rubric.active_questions("consistency")
        ids = sorted(q.id for q in qs)
        assert ids == ["false_synthesis", "information_fabrication", "position_representation"]
        assert len(qs) == 3

    def test_noise_returns_3_items(self) -> None:
        """Noise: 2 shared + noise_contamination CS."""
        rubric = _make_rubric()
        qs = rubric.active_questions("noise")
        ids = sorted(q.id for q in qs)
        assert ids == ["information_fabrication", "noise_contamination", "position_representation"]
        assert len(qs) == 3

    def test_prior_conflict_returns_2_shared(self) -> None:
        """Prior-conflict: position_representation + information_fabrication (no CS)."""
        rubric = _make_rubric()
        qs = rubric.active_questions("prior_conflict")
        ids = sorted(q.id for q in qs)
        assert ids == ["information_fabrication", "position_representation"]
        assert len(qs) == 2

    def test_availability_returns_empty(self) -> None:
        """Availability is not in active_ies — no sub-questions active."""
        rubric = _make_rubric()
        qs = rubric.active_questions("availability")
        assert len(qs) == 0

    def test_normalizes_data_form_ie(self) -> None:
        """Data-form IE names are normalized to canonical form."""
        rubric = _make_rubric()
        qs = rubric.active_questions("ie_consistency_contradiction")
        ids = sorted(q.id for q in qs)
        assert "false_synthesis" in ids
        assert len(qs) == 3


# ---------------------------------------------------------------------------
# build_verdict_type
# ---------------------------------------------------------------------------


class TestBuildVerdictType:
    def test_baseline_has_2_judge_fields(self) -> None:
        """Baseline verdict: position_representation + information_fabrication."""
        rubric = _make_rubric()
        model = rubric.build_verdict_type("baseline")
        fields = model.model_fields
        assert "q1" in fields
        assert "q2" in fields
        assert len(fields) == 2

    def test_consistency_has_3_judge_fields(self) -> None:
        """Consistency verdict: 2 shared + false_synthesis."""
        rubric = _make_rubric()
        model = rubric.build_verdict_type("consistency")
        fields = model.model_fields
        assert "q1" in fields
        assert "q2" in fields
        assert "q3" in fields
        assert len(fields) == 3

    def test_noise_has_3_judge_fields(self) -> None:
        """Noise verdict: 2 shared + noise_contamination."""
        rubric = _make_rubric()
        model = rubric.build_verdict_type("noise")
        fields = model.model_fields
        assert "q1" in fields
        assert "q2" in fields
        assert "q3" in fields
        assert len(fields) == 3

    def test_availability_has_0_judge_fields(self) -> None:
        """Availability: nothing active -> 0 judge fields."""
        rubric = _make_rubric()
        model = rubric.build_verdict_type("availability")
        assert len(model.model_fields) == 0


# ---------------------------------------------------------------------------
# pass_if polarity
# ---------------------------------------------------------------------------


class TestPassIf:
    def test_position_representation_pass_if_yes(self) -> None:
        rubric = _make_rubric()
        q = next(q for q in rubric.shared_items if q.id == "position_representation")
        assert q.pass_if == "yes"

    def test_information_fabrication_pass_if_no(self) -> None:
        rubric = _make_rubric()
        q = next(q for q in rubric.shared_items if q.id == "information_fabrication")
        assert q.pass_if == "no"

    def test_false_synthesis_pass_if_no(self) -> None:
        rubric = _make_rubric()
        q = rubric.condition_specific["consistency"][0]
        assert q.pass_if == "no"

    def test_noise_contamination_pass_if_no(self) -> None:
        rubric = _make_rubric()
        q = rubric.condition_specific["noise"][0]
        assert q.pass_if == "no"


# ---------------------------------------------------------------------------
# Sub-question inventory
# ---------------------------------------------------------------------------


class TestSubQuestionInventory:
    def test_shared_items_count(self) -> None:
        rubric = _make_rubric()
        assert len(rubric.shared_items) == 2

    def test_shared_item_ids(self) -> None:
        rubric = _make_rubric()
        ids = [q.id for q in rubric.shared_items]
        assert ids == ["position_representation", "information_fabrication"]

    def test_condition_specific_keys(self) -> None:
        rubric = _make_rubric()
        assert sorted(rubric.condition_specific.keys()) == ["consistency", "noise"]

    def test_condition_specific_count(self) -> None:
        rubric = _make_rubric()
        assert len(rubric.condition_specific["consistency"]) == 1
        assert len(rubric.condition_specific["noise"]) == 1

    def test_all_items_are_judge_evaluated(self) -> None:
        """All faithfulness sub-questions use judge evaluation (no programmatic)."""
        rubric = _make_rubric()
        all_qs = list(rubric.shared_items)
        for qs in rubric.condition_specific.values():
            all_qs.extend(qs)
        assert all(q.eval == "judge" for q in all_qs)

    def test_no_programmatic_checks(self) -> None:
        """No sub-questions should have programmatic_check set."""
        rubric = _make_rubric()
        all_qs = list(rubric.shared_items)
        for qs in rubric.condition_specific.values():
            all_qs.extend(qs)
        assert all(q.programmatic_check is None for q in all_qs)


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
            "position_representation",
            "information_fabrication",
            "false_synthesis",
            "noise_contamination",
        ]
        for sq_id in expected_ids:
            assert sq_id in table


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestRubricAttributes:
    def test_name(self) -> None:
        rubric = _make_rubric()
        assert rubric.name == "faithfulness"

    def test_definition_not_stub(self) -> None:
        rubric = _make_rubric()
        assert "STUB" not in rubric.definition
        assert "accurately" in rubric.definition

    def test_aggregation_method(self) -> None:
        rubric = _make_rubric()
        assert rubric.aggregation_method == "majority_vote"

    def test_primary_score_key(self) -> None:
        rubric = _make_rubric()
        assert rubric.primary_score_key == "adherence_rate"
