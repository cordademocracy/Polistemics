import pytest

from src.common.context_format import (
    IE_PROPERTY_OF_CONTEXT,
    format_source,
    normalize_ie,
)

SINGLE_CHUNK = ["The party supports renewable energy."]
TWO_CHUNKS = [
    "The party supports renewable energy.",
    "The party opposes renewable energy.",
]
NOISE_CHUNKS = [
    "Distractor about healthcare.",
    "Distractor about education.",
    "The party supports renewable energy.",
    "Distractor about defense.",
    "Distractor about taxes.",
]


class TestNormalizeIe:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("baseline", "baseline"),
            ("ie_availability_absent", "availability"),
            ("ie_clarity_vague", "clarity"),
            ("ie_consistency_contradiction", "consistency"),
            ("ie_noise", "noise"),
            ("ie_prior_conflict", "prior_conflict"),
            ("availability", "availability"),
            ("clarity", "clarity"),
            ("consistency", "consistency"),
            ("noise", "noise"),
            ("prior_conflict", "prior_conflict"),
        ],
    )
    def test_normalizes_both_forms(self, raw: str, expected: str) -> None:
        assert normalize_ie(raw) == expected

    def test_unknown_ie_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown IE name"):
            normalize_ie("ie_unknown_foo")


class TestFormatSource:
    def test_baseline(self) -> None:
        result = format_source("baseline", SINGLE_CHUNK)
        assert result == "[TARGET] The party supports renewable energy."

    def test_baseline_data_form(self) -> None:
        result = format_source("baseline", SINGLE_CHUNK)
        assert result.startswith("[TARGET]")

    def test_availability_ignores_chunks(self) -> None:
        result = format_source("availability", [])
        assert result == "No relevant documents found."

    def test_availability_with_data_form(self) -> None:
        result = format_source("ie_availability_absent", ["should be ignored"])
        assert result == "No relevant documents found."

    def test_clarity(self) -> None:
        result = format_source("clarity", SINGLE_CHUNK)
        assert result == "[TARGET] The party supports renewable energy."

    def test_consistency_both_target(self) -> None:
        result = format_source("consistency", TWO_CHUNKS)
        lines = result.split("\n\n")
        assert len(lines) == 2
        assert all(line.startswith("[TARGET]") for line in lines)
        assert "supports" in lines[0]
        assert "opposes" in lines[1]

    def test_noise_target_and_distractors(self) -> None:
        result = format_source("noise", NOISE_CHUNKS, evidence_index=2)
        parts = result.split("\n\n")
        assert len(parts) == 5
        assert parts[0] == "[TARGET] The party supports renewable energy."
        for part in parts[1:]:
            assert part.startswith("[DISTRACTOR]")

    def test_noise_requires_evidence_index(self) -> None:
        with pytest.raises(ValueError, match="evidence_index is required"):
            format_source("noise", NOISE_CHUNKS)

    def test_noise_data_form(self) -> None:
        result = format_source("ie_noise", NOISE_CHUNKS, evidence_index=2)
        assert "[TARGET]" in result
        assert "[DISTRACTOR]" in result

    def test_prior_conflict(self) -> None:
        result = format_source("prior_conflict", SINGLE_CHUNK)
        assert result == "[TARGET] The party supports renewable energy."

    @pytest.mark.parametrize(
        "ie",
        ["baseline", "availability", "clarity", "consistency", "noise", "prior_conflict"],
    )
    def test_all_ies_covered_in_property_of_context(self, ie: str) -> None:
        assert ie in IE_PROPERTY_OF_CONTEXT


class TestIePropertyOfContext:
    """Tests for IE_PROPERTY_OF_CONTEXT description values."""

    @pytest.mark.parametrize("ie", ["baseline", "noise", "prior_conflict"])
    def test_answerable_conditions_contain_expected_stance_placeholder(
        self, ie: str
    ) -> None:
        assert "{expected_stance}" in IE_PROPERTY_OF_CONTEXT[ie]

    @pytest.mark.parametrize("ie", ["availability", "clarity", "consistency"])
    def test_non_answerable_conditions_lack_expected_stance_placeholder(
        self, ie: str
    ) -> None:
        assert "{expected_stance}" not in IE_PROPERTY_OF_CONTEXT[ie]

    def test_baseline_description(self) -> None:
        text = IE_PROPERTY_OF_CONTEXT["baseline"]
        assert "task-relevant evidence" in text
        assert "stance is determinable" in text

    def test_availability_description(self) -> None:
        text = IE_PROPERTY_OF_CONTEXT["availability"]
        assert "No evidence was provided" in text
        assert "placeholder" in text

    def test_clarity_description(self) -> None:
        text = IE_PROPERTY_OF_CONTEXT["clarity"]
        assert "[TARGET]" in text
        assert "lack a determinable stance" in text

    def test_consistency_description(self) -> None:
        text = IE_PROPERTY_OF_CONTEXT["consistency"]
        assert "equally-weighted" in text
        assert "opposing stances" in text

    def test_noise_description(self) -> None:
        text = IE_PROPERTY_OF_CONTEXT["noise"]
        assert "[TARGET]" in text
        assert "[DISTRACTOR]" in text

    def test_prior_conflict_description(self) -> None:
        text = IE_PROPERTY_OF_CONTEXT["prior_conflict"]
        assert "prior knowledge" in text
        assert "[TARGET]" in text
