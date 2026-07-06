from datetime import UTC, datetime

import pytest

from src.common.schemas import (
    CollectionResult,
    DatasetItem,
    ItemScore,
    LLMOutput,
    OutputStatus,
    PromptVariation,
    RefusalType,
    StanceLabel,
)


def test_stance_label_values():
    assert StanceLabel.AGREE == "Agree"
    assert StanceLabel.DISAGREE == "Disagree"
    assert StanceLabel.NEUTRAL == "Neutral"


def test_prompt_variation_values():
    assert PromptVariation.MINIMAL == "minimal"
    assert PromptVariation.CONTEXTUAL == "contextual"


def test_output_status_values():
    assert OutputStatus.SUCCESS == "success"
    assert OutputStatus.PARSE_ERROR == "parse_error"
    assert OutputStatus.REFUSAL == "refusal"
    assert OutputStatus.API_ERROR == "api_error"


def test_refusal_type_values():
    assert RefusalType.HARD == "hard"
    assert RefusalType.SOFT == "soft"
    assert RefusalType.NONE == "none"


def test_dataset_item_creation():
    item = DatasetItem(
        observation_id="bundestagswahl2025__de_spd__s001",
        election_id="bundestagswahl2025",
        party_id="de_spd",
        party_name="SPD",
        party_anonymized="Party 01",
        statement_id="bundestagswahl2025__s001",
        statement_number=1,
        statement_text="Deutschland soll aus der NATO austreten.",
        statement_category=None,
        stance_label=StanceLabel.DISAGREE,
        rationale_text="Die SPD steht zur NATO.",
        has_rationale=True,
        ie_name="baseline",
        ie_chunks=[],
    )
    assert item.party_name == "SPD"
    assert item.stance_label == StanceLabel.DISAGREE


def test_dataset_item_rejects_invalid_stance():
    with pytest.raises(ValueError):
        DatasetItem(
            observation_id="test", election_id="test", party_id="test",
            party_name="Test", party_anonymized="Party 01",
            statement_id="test", statement_number=None,
            statement_text="test", statement_category=None,
            stance_label="Invalid", rationale_text=None, has_rationale=False,
            ie_name="baseline", ie_chunks=[],
        )


def test_llm_output_creation():
    output = LLMOutput(
        observation_id="test__s001", statement_id="test__s001", party_id="de_spd",
        experiment_id="t0_pilot_v1", model_id="model_a",
        prompt_variation=PromptVariation.MINIMAL, run_index=0, temperature=0.0,
        predicted_stance=StanceLabel.AGREE,
        predicted_explanation="The SPD supports this because...",
        timestamp=datetime.now(UTC), latency_ms=1823.0,
        tokens_input=245, tokens_output=312, cost_usd=0.0042,
        status=OutputStatus.SUCCESS, error_message=None, refusal_type=RefusalType.NONE,
        ie_name="baseline",
        condition_id="baseline__real__evidence__none",
    )
    assert output.predicted_stance == StanceLabel.AGREE
    assert output.status == OutputStatus.SUCCESS


def test_llm_output_allows_none_stance():
    output = LLMOutput(
        observation_id="test__s001", statement_id="test__s001", party_id="de_spd",
        experiment_id="t0_pilot_v1", model_id="model_a",
        prompt_variation=PromptVariation.MINIMAL, run_index=0, temperature=0.0,
        predicted_stance=None, predicted_explanation="",
        timestamp=datetime.now(UTC), latency_ms=100.0,
        tokens_input=0, tokens_output=0, cost_usd=None,
        status=OutputStatus.API_ERROR, error_message="Connection refused",
        refusal_type=None,
        ie_name="baseline",
        condition_id="baseline__real__evidence__none",
    )
    assert output.predicted_stance is None
    assert output.status == OutputStatus.API_ERROR


def test_llm_output_json_roundtrip():
    output = LLMOutput(
        observation_id="test__s001", statement_id="test__s001", party_id="de_spd",
        experiment_id="t0_pilot_v1", model_id="model_a",
        prompt_variation=PromptVariation.MINIMAL, run_index=0, temperature=0.0,
        predicted_stance=StanceLabel.AGREE, predicted_explanation="test explanation",
        timestamp=datetime.now(UTC), latency_ms=100.0,
        tokens_input=100, tokens_output=50, cost_usd=0.001,
        status=OutputStatus.SUCCESS, error_message=None, refusal_type=RefusalType.NONE,
        ie_name="baseline",
        condition_id="baseline__real__evidence__none",
    )
    json_str = output.model_dump_json()
    restored = LLMOutput.model_validate_json(json_str)
    assert restored.predicted_stance == output.predicted_stance
    assert restored.prompt_variation == output.prompt_variation


def test_item_score_flexible_scores():
    score = ItemScore(
        observation_id="test", party_id="de_spd", statement_id="test__s001",
        model_id="model_a", prompt_variation=PromptVariation.MINIMAL,
        run_index=0, temperature=0.0, metric_name="faithfulness",
        scores={"stance_correct": 1.0, "f1_agree": 0.85},
        ie_name="baseline",
    )
    assert score.scores["stance_correct"] == 1.0


def test_collection_result():
    result = CollectionResult(
        experiment_id="t0_pilot_v1", total_items=384,
        successful=380, failed=4, skipped=0,
        total_cost_usd=12.50, total_tokens=150000, duration_seconds=120.5,
    )
    assert result.successful + result.failed + result.skipped == result.total_items
