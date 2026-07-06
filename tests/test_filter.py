from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.schemas import LLMOutput, OutputStatus, PromptVariation, RefusalType, StanceLabel
from src.evaluate.filter import OutputFilter, filter_outputs


def _make_output(
    party_id: str = "de_spd",
    model_id: str = "model_a",
    prompt_variation: PromptVariation = PromptVariation.MINIMAL,
    temperature: float = 0.0,
    status: OutputStatus = OutputStatus.SUCCESS,
) -> LLMOutput:
    return LLMOutput(
        observation_id=f"obs_{party_id}_{model_id}",
        statement_id="s001",
        party_id=party_id,
        experiment_id="test",
        model_id=model_id,
        prompt_variation=prompt_variation,
        run_index=0,
        temperature=temperature,
        predicted_stance=StanceLabel.AGREE,
        predicted_explanation="test",
        timestamp=datetime.now(UTC),
        latency_ms=100.0,
        tokens_input=50,
        tokens_output=100,
        cost_usd=0.001,
        status=status,
        error_message=None,
        refusal_type=RefusalType.NONE,
        ie_name="baseline",
        condition_id="baseline__real__evidence__none",
    )


@pytest.fixture
def sample_outputs() -> list[LLMOutput]:
    return [
        _make_output(party_id="de_spd", model_id="model_a", prompt_variation=PromptVariation.MINIMAL, temperature=0.0, status=OutputStatus.SUCCESS),
        _make_output(party_id="de_cdu", model_id="model_a", prompt_variation=PromptVariation.CONTEXTUAL, temperature=0.7, status=OutputStatus.SUCCESS),
        _make_output(party_id="de_spd", model_id="model_b", prompt_variation=PromptVariation.CONTEXTUAL, temperature=0.0, status=OutputStatus.REFUSAL),
        _make_output(party_id="de_fdp", model_id="model_b", prompt_variation=PromptVariation.MINIMAL, temperature=0.7, status=OutputStatus.PARSE_ERROR),
    ]


def test_filter_by_party(sample_outputs: list[LLMOutput]) -> None:
    result = filter_outputs(sample_outputs, OutputFilter(party_ids=["de_spd"]))
    assert all(o.party_id == "de_spd" for o in result)
    assert len(result) == 2


def test_filter_by_model(sample_outputs: list[LLMOutput]) -> None:
    result = filter_outputs(sample_outputs, OutputFilter(model_ids=["model_a"]))
    assert all(o.model_id == "model_a" for o in result)
    assert len(result) == 2


def test_filter_by_prompt_variation(sample_outputs: list[LLMOutput]) -> None:
    result = filter_outputs(sample_outputs, OutputFilter(prompt_variations=[PromptVariation.MINIMAL]))
    assert all(o.prompt_variation == PromptVariation.MINIMAL for o in result)
    assert len(result) == 2


def test_filter_by_status(sample_outputs: list[LLMOutput]) -> None:
    result = filter_outputs(sample_outputs, OutputFilter(statuses=[OutputStatus.SUCCESS]))
    assert all(o.status == OutputStatus.SUCCESS for o in result)
    assert len(result) == 2


def test_filter_by_temperature(sample_outputs: list[LLMOutput]) -> None:
    result = filter_outputs(sample_outputs, OutputFilter(temperatures=[0.0]))
    assert all(o.temperature == 0.0 for o in result)
    assert len(result) == 2


def test_filter_combined(sample_outputs: list[LLMOutput]) -> None:
    result = filter_outputs(sample_outputs, OutputFilter(party_ids=["de_spd"], model_ids=["model_a"]))
    assert len(result) == 1
    assert result[0].party_id == "de_spd"
    assert result[0].model_id == "model_a"


def test_filter_none_returns_all(sample_outputs: list[LLMOutput]) -> None:
    result = filter_outputs(sample_outputs, OutputFilter())
    assert result == sample_outputs


def test_filter_no_matches(sample_outputs: list[LLMOutput]) -> None:
    result = filter_outputs(sample_outputs, OutputFilter(party_ids=["de_gruene"]))
    assert result == []
