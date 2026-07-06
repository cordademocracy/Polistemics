from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.models.config import ModelConfig

_API_KEY_FIELDS_BY_ENV_VAR: dict[str, str] = {
    "OPENROUTER_API_KEY": "openrouter_api_key",
    "OPENROUTER_API_KEY_2": "openrouter_api_key_2",
    "OPENROUTER_API_KEY_GEMINI": "openrouter_api_key_gemini",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "OPENAI_API_KEY": "openai_api_key",
    "HF_API_KEY": "hf_api_key",
}


class AppSettings(BaseSettings):
    """Environment-backed runtime settings."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        extra="ignore",
        case_sensitive=False,
    )

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_api_key_2: str | None = Field(default=None, alias="OPENROUTER_API_KEY_2")
    openrouter_api_key_gemini: str | None = Field(default=None, alias="OPENROUTER_API_KEY_GEMINI")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    hf_api_key: str | None = Field(default=None, alias="HF_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache app settings from environment sources."""
    return AppSettings()


def get_api_key_from_env_name(env_var_name: str) -> str | None:
    """Resolve an API key by its env-var name.

    Checks pydantic-settings fields first (loaded from .env), then falls
    back to os.environ for rotation keys (e.g. OPENROUTER_API_KEY_2).
    """
    import os

    normalized_name = env_var_name.strip().upper()

    # Try known pydantic-settings fields first
    field_name = _API_KEY_FIELDS_BY_ENV_VAR.get(normalized_name)
    if field_name is not None:
        value = getattr(get_settings(), field_name)
        if value is not None:
            cleaned = value.strip()
            if cleaned:
                return cleaned

    # Fallback: os.environ (for rotation keys not in AppSettings)
    value = os.environ.get(normalized_name)
    if value is not None:
        cleaned = value.strip()
        if cleaned:
            return cleaned

    return None


def export_api_keys_to_env() -> None:
    """Export loaded API keys to os.environ.

    PydanticAI's built-in providers (e.g. OpenRouterProvider used by the
    JudgePanel) read keys directly from os.environ rather than from our
    AppSettings. This bridges the gap so .env keys are available globally.
    """
    import os

    settings = get_settings()
    for env_name, field_name in _API_KEY_FIELDS_BY_ENV_VAR.items():
        value = getattr(settings, field_name, None)
        if value is not None and env_name not in os.environ:
            os.environ[env_name] = value


def _collect_env_names(model_configs: dict[str, ModelConfig]) -> set[str]:
    """Extract all env-var names from model configs (handles str | list[str])."""
    names: set[str] = set()
    for config in model_configs.values():
        if config.api_key_env is None:
            continue
        keys = config.api_key_env if isinstance(config.api_key_env, list) else [config.api_key_env]
        for k in keys:
            stripped = k.strip().upper()
            if stripped:
                names.add(stripped)
    return names


def validate_required_api_keys(model_configs: dict[str, ModelConfig]) -> None:
    """Validate that all configured model API keys are present at startup.

    Supports both single keys and rotation key lists. Keys are resolved
    via AppSettings fields first, then os.environ as fallback.
    """
    required_env_names = sorted(_collect_env_names(model_configs))

    if not required_env_names:
        return

    missing_env_names = [
        env_name for env_name in required_env_names
        if get_api_key_from_env_name(env_name) is None
    ]
    if not missing_env_names:
        return

    missing_names = ", ".join(missing_env_names)
    raise ValueError(
        "Missing required API key env vars: "
        f"{missing_names}. Set them in .env, .env.local, or environment.",
    )
