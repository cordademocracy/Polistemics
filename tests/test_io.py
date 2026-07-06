from datetime import UTC, datetime

from src.common.io import append_output_to_path, load_outputs_from_path
from src.common.schemas import (
    LLMOutput,
    OutputStatus,
    PromptVariation,
    RefusalType,
    StanceLabel,
)


def _make_output(obs_id: str = "test__s001") -> LLMOutput:
    return LLMOutput(
        observation_id=obs_id, statement_id="test__s001", party_id="de_spd",
        experiment_id="test_exp", model_id="model_a",
        prompt_variation=PromptVariation.MINIMAL, run_index=0, temperature=0.0,
        predicted_stance=StanceLabel.AGREE, predicted_explanation="test",
        timestamp=datetime.now(UTC), latency_ms=100.0,
        tokens_input=50, tokens_output=30, cost_usd=0.001,
        status=OutputStatus.SUCCESS, error_message=None, refusal_type=RefusalType.NONE,
        ie_name="baseline",
        condition_id="baseline__real__evidence__none",
    )


def test_jsonl_roundtrip(tmp_path):
    output = _make_output()
    path = tmp_path / "outputs.jsonl"
    append_output_to_path(output, path)
    loaded = load_outputs_from_path(path)
    assert len(loaded) == 1
    assert loaded[0].predicted_stance == StanceLabel.AGREE


def test_jsonl_append_multiple(tmp_path):
    path = tmp_path / "outputs.jsonl"
    for i in range(3):
        append_output_to_path(_make_output(f"test__s{i:03d}"), path)
    loaded = load_outputs_from_path(path)
    assert len(loaded) == 3


def test_load_outputs_empty(tmp_path):
    path = tmp_path / "nonexistent.jsonl"
    loaded = load_outputs_from_path(path)
    assert loaded == []
