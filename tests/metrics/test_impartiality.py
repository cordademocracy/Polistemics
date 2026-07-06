"""Tests for ImpartialityRubric — sub-question routing, verdict schema, pass_if polarity."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.metrics.impartiality import ImpartialityRubric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rubric() -> ImpartialityRubric:
    """Build an ImpartialityRubric with mocked panel + prompt_builder."""
    return ImpartialityRubric(
        panel=MagicMock(),
        prompt_builder=MagicMock(),
    )


# ---------------------------------------------------------------------------
# active_questions routing
# ---------------------------------------------------------------------------


class TestActiveQuestions:
    def test_baseline_returns_5_shared(self) -> None:
        """Baseline: all 5 shared items, no CS."""
        rubric = _make_rubric()
        qs = rubric.active_questions("baseline")
        assert len(qs) == 5

    def test_clarity_returns_5_shared(self) -> None:
        """Clarity (Vague): all 5 shared items, no CS."""
        rubric = _make_rubric()
        qs = rubric.active_questions("clarity")
        assert len(qs) == 5

    def test_consistency_returns_6_items(self) -> None:
        """Consistency (Contradictory): 5 shared + structural_balance CS."""
        rubric = _make_rubric()
        qs = rubric.active_questions("consistency")
        ids = [q.id for q in qs]
        assert len(qs) == 6
        assert "structural_balance" in ids

    def test_noise_returns_5_shared(self) -> None:
        """Noise (Noisy): all 5 shared, no CS."""
        rubric = _make_rubric()
        qs = rubric.active_questions("noise")
        assert len(qs) == 5

    def test_prior_conflict_returns_5_shared(self) -> None:
        """Prior-conflict (Counterfactual): all 5 shared, no CS."""
        rubric = _make_rubric()
        qs = rubric.active_questions("prior_conflict")
        assert len(qs) == 5

    def test_availability_returns_empty(self) -> None:
        """Availability (Absent) is not in active_ies — no sub-questions."""
        rubric = _make_rubric()
        qs = rubric.active_questions("availability")
        assert len(qs) == 0

    def test_normalizes_data_form_ie(self) -> None:
        """Data-form IE names are normalized to canonical form."""
        rubric = _make_rubric()
        qs = rubric.active_questions("ie_consistency_contradiction")
        assert len(qs) == 6
        assert any(q.id == "structural_balance" for q in qs)


# ---------------------------------------------------------------------------
# build_verdict_type
# ---------------------------------------------------------------------------


class TestBuildVerdictType:
    def test_baseline_has_5_judge_fields(self) -> None:
        rubric = _make_rubric()
        model = rubric.build_verdict_type("baseline")
        assert len(model.model_fields) == 5

    def test_consistency_has_6_judge_fields(self) -> None:
        rubric = _make_rubric()
        model = rubric.build_verdict_type("consistency")
        assert len(model.model_fields) == 6

    def test_availability_has_0_judge_fields(self) -> None:
        rubric = _make_rubric()
        model = rubric.build_verdict_type("availability")
        assert len(model.model_fields) == 0


# ---------------------------------------------------------------------------
# pass_if polarity
# ---------------------------------------------------------------------------


class TestPassIf:
    def test_all_shared_pass_if_no(self) -> None:
        """All 5 shared items have pass_if='no' (negative framing: absence of bad behavior)."""
        rubric = _make_rubric()
        for q in rubric.shared_items:
            assert q.pass_if == "no", f"{q.id} has pass_if={q.pass_if!r}"

    def test_structural_balance_pass_if_yes(self) -> None:
        rubric = _make_rubric()
        q = rubric.condition_specific["consistency"][0]
        assert q.pass_if == "yes"


# ---------------------------------------------------------------------------
# Sub-question inventory
# ---------------------------------------------------------------------------


class TestSubQuestionInventory:
    def test_shared_items_count(self) -> None:
        rubric = _make_rubric()
        assert len(rubric.shared_items) == 5

    def test_shared_item_ids(self) -> None:
        rubric = _make_rubric()
        ids = [q.id for q in rubric.shared_items]
        assert ids == [
            "endorsement",
            "condemnation",
            "loaded_language",
            "sanitization",
            "attribution_bias",
        ]

    def test_condition_specific_keys(self) -> None:
        rubric = _make_rubric()
        assert list(rubric.condition_specific.keys()) == ["consistency"]

    def test_condition_specific_count(self) -> None:
        rubric = _make_rubric()
        assert len(rubric.condition_specific["consistency"]) == 1

    def test_all_items_are_judge_evaluated(self) -> None:
        rubric = _make_rubric()
        all_qs = list(rubric.shared_items)
        for qs in rubric.condition_specific.values():
            all_qs.extend(qs)
        assert all(q.eval == "judge" for q in all_qs)

    def test_no_programmatic_checks(self) -> None:
        rubric = _make_rubric()
        all_qs = list(rubric.shared_items)
        for qs in rubric.condition_specific.values():
            all_qs.extend(qs)
        assert all(q.programmatic_check is None for q in all_qs)

    def test_target_label_in_evidence_questions(self) -> None:
        """I3, I4, I6 reference [TARGET] since they compare against evidence."""
        rubric = _make_rubric()
        i3 = next(q for q in rubric.shared_items if q.id == "loaded_language")
        i4 = next(q for q in rubric.shared_items if q.id == "sanitization")
        i6 = rubric.condition_specific["consistency"][0]
        assert "[TARGET]" in i3.text
        assert "[TARGET]" in i4.text
        assert "[TARGET]" in i6.text

    def test_no_target_label_in_output_questions(self) -> None:
        """I1, I2, I5 are about output behavior — no [TARGET] reference."""
        rubric = _make_rubric()
        output_qs = [
            q for q in rubric.shared_items
            if q.id in ("endorsement", "condemnation", "attribution_bias")
        ]
        for q in output_qs:
            assert "[TARGET]" not in q.text, f"{q.id} should not reference [TARGET]"


# ---------------------------------------------------------------------------
# to_paper_matrix
# ---------------------------------------------------------------------------


class TestToPaperMatrix:
    def test_produces_valid_markdown(self) -> None:
        rubric = _make_rubric()
        table = rubric.to_paper_matrix()
        lines = table.strip().split("\n")
        # header + separator + 6 data rows
        assert len(lines) == 8
        assert lines[0].startswith("| id")
        assert "---" in lines[1]

    def test_includes_all_6_sub_questions(self) -> None:
        rubric = _make_rubric()
        table = rubric.to_paper_matrix()
        expected_ids = [
            "endorsement",
            "condemnation",
            "loaded_language",
            "sanitization",
            "attribution_bias",
            "structural_balance",
        ]
        for sq_id in expected_ids:
            assert sq_id in table


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestRubricAttributes:
    def test_name(self) -> None:
        rubric = _make_rubric()
        assert rubric.name == "impartiality"

    def test_definition_not_empty(self) -> None:
        rubric = _make_rubric()
        assert len(rubric.definition) > 20

    def test_aggregation_method(self) -> None:
        rubric = _make_rubric()
        assert rubric.aggregation_method == "majority_vote"

    def test_primary_score_key(self) -> None:
        rubric = _make_rubric()
        assert rubric.primary_score_key == "adherence_rate"
