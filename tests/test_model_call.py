from __future__ import annotations


from src.common.config import ModelConfig
from src.common.model_call import build_model_settings


def _make_mc(**kwargs) -> ModelConfig:
    """Create a ModelConfig with required fields and given overrides."""
    defaults = {
        "id": "test_model",
        "provider": "openrouter",
        "model": "some/model",
        "max_tokens": 1024,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# max_tokens is always present
# ---------------------------------------------------------------------------


def test_max_tokens_always_included() -> None:
    mc = _make_mc(max_tokens=512)
    settings = build_model_settings(mc)
    assert settings["max_tokens"] == 512


# ---------------------------------------------------------------------------
# No reasoning_effort → no reasoning keys
# ---------------------------------------------------------------------------


def test_no_reasoning_effort_produces_no_reasoning_keys() -> None:
    mc = _make_mc(reasoning_effort=None)
    settings = build_model_settings(mc)
    assert "openai_reasoning_effort" not in settings
    assert "openrouter_reasoning" not in settings


# ---------------------------------------------------------------------------
# OpenRouter provider → openrouter_reasoning
# ---------------------------------------------------------------------------


def test_openrouter_provider_uses_openrouter_reasoning() -> None:
    mc = _make_mc(provider="openrouter", reasoning_effort="low")
    settings = build_model_settings(mc)
    assert "openrouter_reasoning" in settings
    assert settings["openrouter_reasoning"] == {"effort": "low"}
    assert "openai_reasoning_effort" not in settings


def test_openrouter_reasoning_effort_values_forwarded() -> None:
    """Effort value is forwarded verbatim, not validated here."""
    for effort in ("low", "medium", "high"):
        mc = _make_mc(provider="openrouter", reasoning_effort=effort)
        settings = build_model_settings(mc)
        assert settings["openrouter_reasoning"] == {"effort": effort}


def test_openai_model_via_openrouter_uses_openrouter_reasoning() -> None:
    """openai/* models routed through OpenRouter still use openrouter_reasoning."""
    mc = _make_mc(provider="openrouter", model="openai/gpt-5.4", reasoning_effort="low")
    settings = build_model_settings(mc)
    assert "openrouter_reasoning" in settings
    assert "openai_reasoning_effort" not in settings


# ---------------------------------------------------------------------------
# OpenAI native provider → openai_reasoning_effort
# ---------------------------------------------------------------------------


def test_openai_native_provider_uses_openai_reasoning_effort() -> None:
    mc = _make_mc(provider="openai", reasoning_effort="medium")
    settings = build_model_settings(mc)
    assert settings["openai_reasoning_effort"] == "medium"
    assert "openrouter_reasoning" not in settings


# ---------------------------------------------------------------------------
# openrouter_provider forwarding
# ---------------------------------------------------------------------------


def test_openrouter_provider_routing_included_when_set() -> None:
    routing = {"order": ["DeepInfra"]}
    mc = _make_mc(openrouter_provider=routing)
    settings = build_model_settings(mc)
    assert settings["openrouter_provider"] == routing


def test_openrouter_provider_routing_absent_when_none() -> None:
    mc = _make_mc(openrouter_provider=None)
    settings = build_model_settings(mc)
    assert "openrouter_provider" not in settings


# ---------------------------------------------------------------------------
# Temperature is NOT included (injected per-call by the caller)
# ---------------------------------------------------------------------------


def test_temperature_not_included() -> None:
    mc = _make_mc(temperature=0.7)
    settings = build_model_settings(mc)
    assert "temperature" not in settings


# ---------------------------------------------------------------------------
# Combined: all relevant fields together
# ---------------------------------------------------------------------------


def test_full_openrouter_model_config() -> None:
    mc = _make_mc(
        provider="openrouter",
        model="qwen/qwen3.6-flash",
        max_tokens=2048,
        reasoning_effort="low",
        openrouter_provider={"order": ["amazon-bedrock"]},
    )
    settings = build_model_settings(mc)
    assert settings["max_tokens"] == 2048
    assert settings["openrouter_reasoning"] == {"effort": "low"}
    assert settings["openrouter_provider"] == {"order": ["amazon-bedrock"]}
    assert "openai_reasoning_effort" not in settings
    assert "temperature" not in settings
