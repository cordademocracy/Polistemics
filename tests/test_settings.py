from __future__ import annotations

import pytest

from src.common.models import factory as model_factory
from src.common.models.factory import create_model_pool
from src.common.config import ModelConfig
from src.common.settings import get_api_key_from_env_name, get_settings, validate_required_api_keys


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_reads_env(monkeypatch: pytest.MonkeyPatch, isolated_settings) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    settings = get_settings()

    assert settings.openrouter_api_key == "test-openrouter-key"


def test_validate_required_api_keys_fails_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    isolated_settings,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    model_configs = {
        "model_a": ModelConfig(
            id="model_a",
            provider="openrouter",
            model="placeholder",
            api_key_env="OPENROUTER_API_KEY",
        ),
    }

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        validate_required_api_keys(model_configs)


def test_validate_required_api_keys_passes_when_present(
    monkeypatch: pytest.MonkeyPatch,
    isolated_settings,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    model_configs = {
        "model_a": ModelConfig(
            id="model_a",
            provider="openrouter",
            model="placeholder",
            api_key_env="OPENROUTER_API_KEY",
        ),
    }

    validate_required_api_keys(model_configs)


def test_validate_required_api_keys_fails_for_missing_custom_env_name(
    isolated_settings,
) -> None:
    """Custom env var names (e.g. rotation keys) are valid but must be set."""
    model_configs = {
        "model_a": ModelConfig(
            id="model_a",
            provider="openrouter",
            model="placeholder",
            api_key_env="CUSTOM_PROVIDER_API_KEY",
        ),
    }

    with pytest.raises(ValueError, match="CUSTOM_PROVIDER_API_KEY"):
        validate_required_api_keys(model_configs)


def test_create_model_resolves_api_key_from_settings(
    monkeypatch: pytest.MonkeyPatch,
    isolated_settings,
) -> None:
    captured: dict[str, str | None] = {}

    class DummyOpenAIProvider:
        def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
            captured["api_key"] = api_key

    class DummyOpenAIChatModel:
        def __init__(self, model_name: str, provider: object = None) -> None:
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(model_factory, "OpenAIChatModel", DummyOpenAIChatModel)
    monkeypatch.setattr(model_factory, "OpenAIProvider", DummyOpenAIProvider)

    model_config = ModelConfig(
        id="model_a",
        provider="openai",
        model="placeholder",
        api_key_env="OPENAI_API_KEY",
    )
    model_factory.create_model(model_config)

    assert captured["api_key"] == "test-openai-key"


def test_get_api_key_falls_back_to_os_environ(
    monkeypatch: pytest.MonkeyPatch,
    isolated_settings,
) -> None:
    """Arbitrary env var names (not in AppSettings) resolve via os.environ."""
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "rotated-key-value")

    result = get_api_key_from_env_name("OPENROUTER_API_KEY_2")

    assert result == "rotated-key-value"


def test_validate_with_key_list_passes(
    monkeypatch: pytest.MonkeyPatch,
    isolated_settings,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-1")
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "key-2")

    model_configs = {
        "model_a": ModelConfig(
            id="model_a",
            provider="openrouter",
            model="placeholder",
            api_key_env=["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"],
        ),
    }

    # Should not raise
    validate_required_api_keys(model_configs)


def test_validate_with_key_list_fails_on_missing(
    monkeypatch: pytest.MonkeyPatch,
    isolated_settings,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-1")
    # OPENROUTER_API_KEY_2 intentionally NOT set

    model_configs = {
        "model_a": ModelConfig(
            id="model_a",
            provider="openrouter",
            model="placeholder",
            api_key_env=["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"],
        ),
    }

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY_2"):
        validate_required_api_keys(model_configs)


def test_create_model_pool_single_key(
    monkeypatch: pytest.MonkeyPatch,
    isolated_settings,
) -> None:
    class DummyOpenAIProvider:
        def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
            pass

    class DummyOpenAIChatModel:
        def __init__(self, model_name: str, provider: object = None) -> None:
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(model_factory, "OpenAIChatModel", DummyOpenAIChatModel)
    monkeypatch.setattr(model_factory, "OpenAIProvider", DummyOpenAIProvider)

    config = ModelConfig(
        id="single", provider="openai", model="placeholder",
        api_key_env="OPENAI_API_KEY",
    )
    pool = create_model_pool(config)

    assert isinstance(pool, list)
    assert len(pool) == 1


def test_create_model_pool_multiple_keys(
    monkeypatch: pytest.MonkeyPatch,
    isolated_settings,
) -> None:
    captured_keys: list[str | None] = []

    class DummyOpenAIProvider:
        def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
            captured_keys.append(api_key)

    class DummyOpenAIChatModel:
        def __init__(self, model_name: str, provider: object = None) -> None:
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "key-primary")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key-secondary")
    monkeypatch.setattr(model_factory, "OpenAIChatModel", DummyOpenAIChatModel)
    monkeypatch.setattr(model_factory, "OpenAIProvider", DummyOpenAIProvider)

    config = ModelConfig(
        id="rotated", provider="openai", model="placeholder",
        api_key_env=["OPENAI_API_KEY", "OPENAI_API_KEY_2"],
    )
    pool = create_model_pool(config)

    assert isinstance(pool, list)
    assert len(pool) == 2
    assert captured_keys == ["key-primary", "key-secondary"]
