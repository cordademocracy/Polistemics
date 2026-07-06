"""Tests for JudgePanel — multi-model evaluation infrastructure."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ThinkingPart,
)

from src.evaluate.panel import JudgePanel, PanelResult, SingleJudgeResponse


# ---------------------------------------------------------------------------
# Test output type
# ---------------------------------------------------------------------------


class _MockJudgeOutput(BaseModel):
    label: str
    confidence: float


def _make_run_result(
    label: str,
    confidence: float = 0.9,
    messages: list | None = None,
) -> MagicMock:
    """Create a mock PydanticAI RunResult with structured output."""
    result = MagicMock()
    result.output = _MockJudgeOutput(label=label, confidence=confidence)
    if messages is None:
        messages = [ModelResponse(parts=[TextPart(content="answer")])]
    result.all_messages = MagicMock(return_value=messages)
    return result


# ---------------------------------------------------------------------------
# Panel creation
# ---------------------------------------------------------------------------


def test_panel_stores_model_ids() -> None:
    panel = JudgePanel(model_ids=["a", "b", "c"])
    assert panel._model_ids == ["a", "b", "c"]


def test_panel_creates_semaphores_per_model() -> None:
    panel = JudgePanel(model_ids=["a", "b"], max_concurrent_per_model=3)
    assert "a" in panel._semaphores
    assert "b" in panel._semaphores


# ---------------------------------------------------------------------------
# evaluate() — successful calls
# ---------------------------------------------------------------------------


async def test_evaluate_returns_panel_result() -> None:
    panel = JudgePanel(model_ids=["a", "b", "c"])

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=_make_run_result("economic"))

    # Inject mock agents for all models
    for mid in panel._model_ids:
        key = (mid, "_MockJudgeOutput", "")
        panel._agents[key] = mock_agent

    result = await panel.evaluate("Classify this text", _MockJudgeOutput)

    assert isinstance(result, PanelResult)
    assert result.n_succeeded == 3
    assert result.n_failed == 0
    assert len(result.responses) == 3


async def test_evaluate_parallel_execution() -> None:
    """All panel members should be called (verified by call count)."""
    panel = JudgePanel(model_ids=["a", "b", "c"])

    call_count = 0

    async def _mock_run(prompt: str, **kwargs):
        nonlocal call_count
        call_count += 1
        return _make_run_result("legal")

    mock_agent = MagicMock()
    mock_agent.run = _mock_run

    for mid in panel._model_ids:
        key = (mid, "_MockJudgeOutput", "")
        panel._agents[key] = mock_agent

    await panel.evaluate("text", _MockJudgeOutput)
    assert call_count == 3


# ---------------------------------------------------------------------------
# evaluate() — partial failure
# ---------------------------------------------------------------------------


async def test_partial_failure_continues_with_remaining() -> None:
    panel = JudgePanel(model_ids=["good1", "bad", "good2"])

    good_agent = MagicMock()
    good_agent.run = AsyncMock(return_value=_make_run_result("values"))

    bad_agent = MagicMock()
    bad_agent.run = AsyncMock(side_effect=RuntimeError("API down"))

    panel._agents[("good1", "_MockJudgeOutput", "")] = good_agent
    panel._agents[("bad", "_MockJudgeOutput", "")] = bad_agent
    panel._agents[("good2", "_MockJudgeOutput", "")] = good_agent

    result = await panel.evaluate("text", _MockJudgeOutput)

    assert result.n_succeeded == 2
    assert result.n_failed == 1

    failed = [r for r in result.responses if r.result is None]
    assert len(failed) == 1
    assert "API down" in failed[0].error


async def test_all_fail_returns_all_failed() -> None:
    panel = JudgePanel(model_ids=["a", "b"])

    bad_agent = MagicMock()
    bad_agent.run = AsyncMock(side_effect=RuntimeError("timeout"))

    for mid in panel._model_ids:
        panel._agents[(mid, "_MockJudgeOutput", "")] = bad_agent

    result = await panel.evaluate("text", _MockJudgeOutput)

    assert result.n_succeeded == 0
    assert result.n_failed == 2


# ---------------------------------------------------------------------------
# majority_vote()
# ---------------------------------------------------------------------------


def test_majority_vote_clear_winner() -> None:
    panel = JudgePanel(model_ids=["a", "b", "c"])
    responses = [
        SingleJudgeResponse(model_id="a", result=_MockJudgeOutput(label="economic", confidence=0.9), latency_ms=100),
        SingleJudgeResponse(model_id="b", result=_MockJudgeOutput(label="economic", confidence=0.8), latency_ms=100),
        SingleJudgeResponse(model_id="c", result=_MockJudgeOutput(label="legal", confidence=0.7), latency_ms=100),
    ]
    winner = panel.majority_vote(responses, key_fn=lambda r: r.label)
    assert winner == "economic"


def test_majority_vote_tie_deterministic() -> None:
    panel = JudgePanel(model_ids=["a", "b", "c"])
    responses = [
        SingleJudgeResponse(model_id="a", result=_MockJudgeOutput(label="economic", confidence=0.9), latency_ms=100),
        SingleJudgeResponse(model_id="b", result=_MockJudgeOutput(label="legal", confidence=0.8), latency_ms=100),
        SingleJudgeResponse(model_id="c", result=_MockJudgeOutput(label="values", confidence=0.7), latency_ms=100),
    ]
    # 3-way tie → sorted first = "economic"
    winner = panel.majority_vote(responses, key_fn=lambda r: r.label)
    assert winner == "economic"


def test_majority_vote_skips_failed() -> None:
    panel = JudgePanel(model_ids=["a", "b", "c"])
    responses = [
        SingleJudgeResponse(model_id="a", result=_MockJudgeOutput(label="economic", confidence=0.9), latency_ms=100),
        SingleJudgeResponse(model_id="b", result=None, error="failed", latency_ms=100),
        SingleJudgeResponse(model_id="c", result=_MockJudgeOutput(label="legal", confidence=0.7), latency_ms=100),
    ]
    winner = panel.majority_vote(responses, key_fn=lambda r: r.label)
    # 1-1 tie → sorted first = "economic"
    assert winner == "economic"


def test_majority_vote_all_failed_returns_none() -> None:
    panel = JudgePanel(model_ids=["a"])
    responses = [
        SingleJudgeResponse(model_id="a", result=None, error="failed", latency_ms=100),
    ]
    assert panel.majority_vote(responses, key_fn=lambda r: r.label) is None


# ---------------------------------------------------------------------------
# average()
# ---------------------------------------------------------------------------


def test_average_returns_mean() -> None:
    panel = JudgePanel(model_ids=["a", "b", "c"])
    responses = [
        SingleJudgeResponse(model_id="a", result=_MockJudgeOutput(label="x", confidence=0.9), latency_ms=100),
        SingleJudgeResponse(model_id="b", result=_MockJudgeOutput(label="x", confidence=0.6), latency_ms=100),
        SingleJudgeResponse(model_id="c", result=_MockJudgeOutput(label="x", confidence=0.3), latency_ms=100),
    ]
    avg = panel.average(responses, key_fn=lambda r: r.confidence)
    assert avg == pytest.approx(0.6)


def test_average_skips_failed() -> None:
    panel = JudgePanel(model_ids=["a", "b"])
    responses = [
        SingleJudgeResponse(model_id="a", result=_MockJudgeOutput(label="x", confidence=0.8), latency_ms=100),
        SingleJudgeResponse(model_id="b", result=None, error="failed", latency_ms=100),
    ]
    avg = panel.average(responses, key_fn=lambda r: r.confidence)
    assert avg == pytest.approx(0.8)


def test_average_all_failed_returns_none() -> None:
    panel = JudgePanel(model_ids=["a"])
    responses = [
        SingleJudgeResponse(model_id="a", result=None, error="failed", latency_ms=100),
    ]
    assert panel.average(responses, key_fn=lambda r: r.confidence) is None


# ---------------------------------------------------------------------------
# Audit JSONL
# ---------------------------------------------------------------------------


async def test_write_audit_creates_jsonl(tmp_path: Path) -> None:
    panel = JudgePanel(model_ids=["a"], output_dir=tmp_path)
    responses = [
        SingleJudgeResponse(model_id="a", result=_MockJudgeOutput(label="economic", confidence=0.9), latency_ms=150),
    ]
    await panel.write_audit("frame_classification", "obs1", responses)

    audit_path = tmp_path / "frame_classification.jsonl"
    assert audit_path.exists()

    with audit_path.open() as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    assert lines[0]["observation_id"] == "obs1"
    assert lines[0]["model_id"] == "a"
    assert lines[0]["result"]["label"] == "economic"


async def test_write_audit_skips_when_no_output_dir() -> None:
    panel = JudgePanel(model_ids=["a"], output_dir=None)
    # Should not raise
    await panel.write_audit("metric", "obs1", [])


# ---------------------------------------------------------------------------
# Name-stripping shared utility
# ---------------------------------------------------------------------------


def test_name_stripping_shared_utility() -> None:
    from src.common.text import build_name_pattern, strip_party_names

    pattern = build_name_pattern(["CDU/CSU", "CDU", "SPD", "AfD", "GRÜNE"])
    text = "Die SPD und die CDU/CSU haben unterschiedliche Ansichten."
    stripped = strip_party_names(text, pattern)
    assert "SPD" not in stripped
    assert "CDU/CSU" not in stripped
    assert "Party" in stripped


def test_name_stripping_case_insensitive() -> None:
    from src.common.text import build_name_pattern, strip_party_names

    pattern = build_name_pattern(["SPD"])
    stripped = strip_party_names("Die spd ist aktiv.", pattern)
    assert "spd" not in stripped.lower()
    assert "Party" in stripped


# ---------------------------------------------------------------------------
# Config: JudgesConfig
# ---------------------------------------------------------------------------


def test_get_agent_tool_mode_no_output_marker() -> None:
    """tool mode: output_type passed as plain class, no marker wrapper."""
    from unittest.mock import MagicMock, patch
    from pydantic_ai import NativeOutput, PromptedOutput
    from src.evaluate.panel import JudgePanel
    from src.common.models.config import ModelConfig

    mc = ModelConfig(id="gemini", provider="openrouter", model="google/gemini-3-flash-preview", output_mode="tool")
    panel = JudgePanel(model_ids=["gemini"], judge_model_configs={"gemini": mc})
    captured: dict = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("src.evaluate.panel.Agent", side_effect=fake_agent), \
         patch("src.evaluate.panel.create_model", return_value=MagicMock()):
        panel._get_agent("gemini", _MockJudgeOutput, "sys")

    assert not isinstance(captured["output_type"], NativeOutput)
    assert not isinstance(captured["output_type"], PromptedOutput)


def test_get_agent_native_mode_wraps_in_native_output() -> None:
    """native mode: output_type wrapped in NativeOutput."""
    from unittest.mock import MagicMock, patch
    from pydantic_ai import NativeOutput
    from src.evaluate.panel import JudgePanel
    from src.common.models.config import ModelConfig

    mc = ModelConfig(id="gemini", provider="openrouter", model="google/gemini-3-flash-preview", output_mode="native")
    panel = JudgePanel(model_ids=["gemini"], judge_model_configs={"gemini": mc})
    captured: dict = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("src.evaluate.panel.Agent", side_effect=fake_agent), \
         patch("src.evaluate.panel.create_model", return_value=MagicMock()):
        panel._get_agent("gemini", _MockJudgeOutput, "sys")

    assert isinstance(captured["output_type"], NativeOutput)


def test_get_agent_prompted_mode_wraps_in_prompted_output() -> None:
    """prompted mode: output_type wrapped in PromptedOutput."""
    from unittest.mock import MagicMock, patch
    from pydantic_ai import PromptedOutput
    from src.evaluate.panel import JudgePanel
    from src.common.models.config import ModelConfig

    mc = ModelConfig(id="qwen", provider="openrouter", model="qwen/qwen3.6-plus", output_mode="prompted")
    panel = JudgePanel(model_ids=["qwen"], judge_model_configs={"qwen": mc})
    captured: dict = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("src.evaluate.panel.Agent", side_effect=fake_agent), \
         patch("src.evaluate.panel.create_model", return_value=MagicMock()):
        panel._get_agent("qwen", _MockJudgeOutput, "sys")

    assert isinstance(captured["output_type"], PromptedOutput)


def test_get_agent_default_mode_is_tool() -> None:
    """No model_settings → defaults to tool mode (plain class)."""
    from unittest.mock import MagicMock, patch
    from pydantic_ai import NativeOutput, PromptedOutput
    from src.evaluate.panel import JudgePanel

    panel = JudgePanel(model_ids=["some:model"])
    captured: dict = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("src.evaluate.panel.Agent", side_effect=fake_agent):
        panel._get_agent("some:model", _MockJudgeOutput, "")

    assert not isinstance(captured["output_type"], NativeOutput)
    assert not isinstance(captured["output_type"], PromptedOutput)


# ---------------------------------------------------------------------------
# Config: JudgesConfig
# ---------------------------------------------------------------------------


def test_judges_config_defaults() -> None:
    from src.common.config import JudgesConfig

    config = JudgesConfig()
    assert config.panel == []
    assert config.temperature == 0.0
    assert config.max_concurrent_per_model == 5


def test_judges_config_from_dict() -> None:
    from src.common.config import JudgesConfig

    config = JudgesConfig(
        panel=["openai:gpt-4o-mini", "anthropic:claude-3-haiku"],
        temperature=0.1,
        max_concurrent_per_model=3,
    )
    assert len(config.panel) == 2
    assert config.temperature == 0.1


def test_experiment_config_includes_judges() -> None:
    from src.common.config import ExperimentConfig

    data = {
        "experiment_id": "test",
        "experiment_name": "Test",
        "dataset": {"election_ids": ["bw2025"]},
        "model_ids": ["m1"],
        "prompts": {"levels": ["minimal"], "system_prompt": "sys.txt", "templates": {"minimal": "m.txt"}},
        "runs": [{"temperature": 0.0, "k": 1}],
        "metrics": ["faithfulness"],
        "judges": {
            "panel": ["openai:gpt-4o-mini", "anthropic:claude-3-haiku"],
            "temperature": 0.0,
        },
    }
    config = ExperimentConfig(**data)
    assert len(config.judges.panel) == 2


def test_experiment_config_judges_defaults_to_empty() -> None:
    from src.common.config import ExperimentConfig

    data = {
        "experiment_id": "test",
        "experiment_name": "Test",
        "dataset": {"election_ids": ["bw2025"]},
        "model_ids": ["m1"],
        "prompts": {"levels": ["minimal"], "system_prompt": "sys.txt", "templates": {"minimal": "m.txt"}},
        "runs": [{"temperature": 0.0, "k": 1}],
        "metrics": ["faithfulness"],
    }
    config = ExperimentConfig(**data)
    assert config.judges.panel == []


# ---------------------------------------------------------------------------
# JudgeResponse schema
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Backoff sleep on retry
# ---------------------------------------------------------------------------


async def test_rate_limit_triggers_backoff_sleep() -> None:
    """429 ModelHTTPError must trigger rate-limit (longer) delays before each retry."""
    from unittest.mock import AsyncMock, patch
    from pydantic_ai import ModelHTTPError

    panel = JudgePanel(model_ids=["model-a"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=ModelHTTPError(429, "model-a", {}))
    panel._agents[("model-a", "_MockJudgeOutput", "")] = mock_agent

    # Fix jitter so delay is deterministic
    with patch("src.evaluate.panel.asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("src.common.models.retry.random") as mock_rand:
        mock_rand.random.return_value = 0.0
        result = await panel._run_single("model-a", "prompt", _MockJudgeOutput, "")

    # max_retries=2 → 2 sleeps before giving up
    assert mock_sleep.call_count == 2
    # Rate-limit uses multiplier=10, base=2: attempt 0 → 10.0, attempt 1 → 20.0
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(10.0)
    assert mock_sleep.call_args_list[1].args[0] == pytest.approx(20.0)
    assert result.result is None
    assert "429" in result.error


async def test_other_error_triggers_short_sleep() -> None:
    """Transient network errors must trigger shorter exponential delays."""
    from unittest.mock import AsyncMock, patch

    panel = JudgePanel(model_ids=["model-b"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=ConnectionError("connection refused"))
    panel._agents[("model-b", "_MockJudgeOutput", "")] = mock_agent

    with patch("src.evaluate.panel.asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("src.common.models.retry.random") as mock_rand:
        mock_rand.random.return_value = 0.0
        result = await panel._run_single("model-b", "prompt", _MockJudgeOutput, "")

    # max_retries=2 → 2 sleeps; transient uses multiplier=1, base=2: 1.0, 2.0
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(1.0)
    assert mock_sleep.call_args_list[1].args[0] == pytest.approx(2.0)
    assert result.result is None


async def test_success_on_retry_no_sleep_after() -> None:
    """A 429 on attempt 0 followed by success on attempt 1 sleeps exactly once."""
    from unittest.mock import AsyncMock, patch
    from pydantic_ai import ModelHTTPError

    panel = JudgePanel(model_ids=["model-c"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(
        side_effect=[ModelHTTPError(429, "model-c", {}), _make_run_result("ok")]
    )
    panel._agents[("model-c", "_MockJudgeOutput", "")] = mock_agent

    with patch("src.evaluate.panel.asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("src.common.models.retry.random") as mock_rand:
        mock_rand.random.return_value = 0.0
        result = await panel._run_single("model-c", "prompt", _MockJudgeOutput, "")

    assert mock_sleep.call_count == 1
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(10.0)
    assert result.result is not None


# ---------------------------------------------------------------------------
# JudgeResponse schema
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PartialPanelResult exception
# ---------------------------------------------------------------------------


def test_partial_panel_result_attrs() -> None:
    from src.evaluate.panel import PartialPanelResult

    exc = PartialPanelResult(2, 3)
    assert exc.n_succeeded == 2
    assert exc.n_expected == 3
    assert "2/3" in str(exc)


# ---------------------------------------------------------------------------
# Timeout behaviour
# ---------------------------------------------------------------------------


async def test_timeout_raises_partial_panel_result() -> None:
    """Judge call that hangs past timeout → n_failed > 0 in result."""
    from unittest.mock import AsyncMock, patch

    panel = JudgePanel(model_ids=["slow-model"], judge_timeout_seconds=0.05)

    mock_agent = MagicMock()

    async def _hang(*args, **kwargs):
        await asyncio.sleep(9999)

    mock_agent.run = _hang
    panel._agents[("slow-model", "_MockJudgeOutput", "")] = mock_agent

    with patch("src.evaluate.panel.asyncio.wait_for", side_effect=asyncio.TimeoutError), \
         patch("src.evaluate.panel.asyncio.sleep", new_callable=AsyncMock):
        result = await panel.evaluate("prompt", _MockJudgeOutput)
    assert result.n_failed > 0
    failed = [r for r in result.responses if r.result is None]
    assert len(failed) == 1
    assert "timeout" in (failed[0].error or "")


async def test_no_timeout_when_unset() -> None:
    """judge_timeout_seconds=None → fast mock succeeds normally."""
    panel = JudgePanel(model_ids=["fast-model"], judge_timeout_seconds=None)

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=_make_run_result("ok"))
    panel._agents[("fast-model", "_MockJudgeOutput", "")] = mock_agent

    result = await panel.evaluate("prompt", _MockJudgeOutput)
    assert result.n_succeeded == 1
    assert result.n_failed == 0


async def test_timeout_logs_judge_timeout() -> None:
    """judge_timeout event must be logged when a timeout fires."""
    import structlog.testing
    from unittest.mock import AsyncMock, patch

    panel = JudgePanel(model_ids=["slow-model-2"], judge_timeout_seconds=0.05)

    mock_agent = MagicMock()

    async def _hang(*args, **kwargs):
        await asyncio.sleep(9999)

    mock_agent.run = _hang
    panel._agents[("slow-model-2", "_MockJudgeOutput", "")] = mock_agent

    with patch("src.evaluate.panel.asyncio.wait_for", side_effect=asyncio.TimeoutError), \
         patch("src.evaluate.panel.asyncio.sleep", new_callable=AsyncMock):
        with structlog.testing.capture_logs() as captured:
            await panel.evaluate("prompt", _MockJudgeOutput)

    assert any(entry.get("event") == "judge_timeout" for entry in captured)


async def test_timeout_triggers_other_delays() -> None:
    """asyncio.TimeoutError (from wait_for) must use transient (non-rate-limit) backoff."""
    from unittest.mock import AsyncMock, patch

    panel = JudgePanel(model_ids=["timeout-model"], judge_timeout_seconds=0.05)
    mock_agent = MagicMock()
    panel._agents[("timeout-model", "_MockJudgeOutput", "")] = mock_agent

    with patch("src.evaluate.panel.asyncio.wait_for", side_effect=asyncio.TimeoutError), \
         patch("src.evaluate.panel.asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("src.common.models.retry.random") as mock_rand:
        mock_rand.random.return_value = 0.0
        result = await panel._run_single("timeout-model", "prompt", _MockJudgeOutput, "")

    # Transient backoff: multiplier=1, base=2 → 1.0, 2.0
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(1.0)
    assert mock_sleep.call_args_list[1].args[0] == pytest.approx(2.0)
    assert result.result is None
    assert "timeout" in (result.error or "")


def test_judge_response_schema() -> None:
    from datetime import UTC, datetime

    from src.common.schemas import JudgeResponse, PromptVariation

    resp = JudgeResponse(
        observation_id="obs1",
        metric_name="frame_classification",
        model_id="openai:gpt-4o-mini",
        generator_model_id="anthropic:claude-3-sonnet",
        prompt_variation=PromptVariation.MINIMAL,
        run_index=0,
        raw_response={"label": "economic", "confidence": 0.95},
        latency_ms=150.5,
        timestamp=datetime.now(UTC),
        ie_name="baseline",
    )
    # Roundtrip
    restored = JudgeResponse.model_validate_json(resp.model_dump_json())
    assert restored.observation_id == "obs1"
    assert restored.raw_response["label"] == "economic"


# ---------------------------------------------------------------------------
# Panel transport extensions — audit enrichment fields
# ---------------------------------------------------------------------------


async def test_run_single_populates_prompt() -> None:
    panel = JudgePanel(model_ids=["a"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=_make_run_result("ok"))
    panel._agents[("a", "_MockJudgeOutput", "You are a judge.")] = mock_agent

    resp = await panel._run_single("a", "Rate this.", _MockJudgeOutput, "You are a judge.")

    assert resp.prompt == {"system": "You are a judge.", "user": "Rate this."}


async def test_run_single_extracts_raw_response() -> None:
    model_resp = ModelResponse(parts=[TextPart(content="the answer")])
    run_result = _make_run_result("ok", messages=[model_resp])

    panel = JudgePanel(model_ids=["a"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_result)
    panel._agents[("a", "_MockJudgeOutput", "")] = mock_agent

    resp = await panel._run_single("a", "prompt", _MockJudgeOutput, "")

    assert resp.raw_response is not None
    assert isinstance(resp.raw_response, dict)
    assert any(p["part_kind"] == "text" for p in resp.raw_response["parts"])


async def test_run_single_extracts_native_reasoning() -> None:
    model_resp = ModelResponse(
        parts=[
            ThinkingPart(content="Step 1: analyze"),
            ThinkingPart(content="Step 2: conclude"),
            TextPart(content="result"),
        ]
    )
    run_result = _make_run_result("ok", messages=[model_resp])

    panel = JudgePanel(model_ids=["a"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_result)
    panel._agents[("a", "_MockJudgeOutput", "")] = mock_agent

    resp = await panel._run_single("a", "prompt", _MockJudgeOutput, "")

    assert resp.native_reasoning == "Step 1: analyze\nStep 2: conclude"


async def test_run_single_no_thinking_parts_reasoning_is_none() -> None:
    model_resp = ModelResponse(parts=[TextPart(content="just text")])
    run_result = _make_run_result("ok", messages=[model_resp])

    panel = JudgePanel(model_ids=["a"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_result)
    panel._agents[("a", "_MockJudgeOutput", "")] = mock_agent

    resp = await panel._run_single("a", "prompt", _MockJudgeOutput, "")

    assert resp.native_reasoning is None


async def test_run_single_no_model_response_in_messages() -> None:
    run_result = _make_run_result("ok", messages=[])

    panel = JudgePanel(model_ids=["a"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_result)
    panel._agents[("a", "_MockJudgeOutput", "")] = mock_agent

    resp = await panel._run_single("a", "prompt", _MockJudgeOutput, "")

    assert resp.raw_response is None
    assert resp.native_reasoning is None


async def test_run_single_logprobs_stubbed_none() -> None:
    panel = JudgePanel(model_ids=["a"])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=_make_run_result("ok"))
    panel._agents[("a", "_MockJudgeOutput", "")] = mock_agent

    resp = await panel._run_single("a", "prompt", _MockJudgeOutput, "")

    assert resp.logprobs is None


async def test_write_audit_includes_new_fields(tmp_path: Path) -> None:
    panel = JudgePanel(model_ids=["a"], output_dir=tmp_path)

    responses = [
        SingleJudgeResponse(
            model_id="a",
            result=_MockJudgeOutput(label="economic", confidence=0.9),
            latency_ms=150,
            prompt={"system": "sys", "user": "usr"},
            raw_response={"parts": [{"part_kind": "text", "content": "answer"}]},
            native_reasoning="I thought about it.",
            logprobs=None,
        ),
    ]
    await panel.write_audit("faithfulness", "obs1", responses)

    audit_path = tmp_path / "faithfulness.jsonl"
    with audit_path.open() as f:
        row = json.loads(f.readline())

    assert row["prompt"] == {"system": "sys", "user": "usr"}
    assert row["raw_response"]["parts"][0]["part_kind"] == "text"
    assert row["native_reasoning"] == "I thought about it."
    assert row["logprobs"] is None


async def test_write_audit_new_fields_absent_when_none(tmp_path: Path) -> None:
    panel = JudgePanel(model_ids=["a"], output_dir=tmp_path)

    responses = [
        SingleJudgeResponse(
            model_id="a",
            result=_MockJudgeOutput(label="legal", confidence=0.5),
            latency_ms=100,
        ),
    ]
    await panel.write_audit("metric", "obs2", responses)

    audit_path = tmp_path / "metric.jsonl"
    with audit_path.open() as f:
        row = json.loads(f.readline())

    assert row["prompt"] is None
    assert row["raw_response"] is None
    assert row["native_reasoning"] is None
    assert row["logprobs"] is None


def test_single_judge_response_back_compat() -> None:
    resp = SingleJudgeResponse(
        model_id="m", latency_ms=100.0
    )
    assert resp.prompt is None
    assert resp.raw_response is None
    assert resp.native_reasoning is None
    assert resp.logprobs is None
