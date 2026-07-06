from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from src.common.models.config import ModelConfig, WebSearchConfig
from src.common.settings import get_api_key_from_env_name


class WebSearchOpenRouterModel(OpenRouterModel):
    """OpenRouterModel that uses the modern server tools API for web search.

    Pydantic-ai's built-in WebSearchTool converts to the deprecated
    ``plugins: [{id: "web"}]`` format. This subclass injects web search
    as ``{type: "openrouter:web_search", parameters: {...}}`` in the
    tools array — the current OpenRouter-recommended approach.

    Web search is configured at model creation and injected into every
    request's tools array alongside pydantic-ai's function tools.
    """

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenRouterProvider,
        web_search: WebSearchConfig,
    ) -> None:
        super().__init__(model_name, provider=provider)
        self._web_search = web_search

    def _get_tools(self, model_request_parameters: Any) -> list[Any]:
        """Append openrouter:web_search server tool to the function tools list."""
        tools = super()._get_tools(model_request_parameters)
        tools.append({
            "type": "openrouter:web_search",
            "parameters": {
                "search_context_size": self._web_search.search_context_size,
                "max_results": self._web_search.max_results,
            },
        })
        return tools


def _create_openai_compatible_model(config: ModelConfig, api_key: str | None) -> Model:
    if config.provider == "openrouter":
        provider = OpenRouterProvider(api_key=api_key)

        # Patch supports_json_schema_output onto the provider's auto-detected profile
        # when native output mode is requested. The provider's profile lookup is keyed
        # by model name (vendor prefix routing), so DeepSeek/Mimo etc. get their correct
        # base profile — we just flip the one flag PydanticAI gates on before sending.
        profile: ModelProfile | None = None
        if config.output_mode == "native":
            base = OpenRouterProvider.model_profile(config.model) or ModelProfile()
            profile = replace(base, supports_json_schema_output=True)

        # Fix: use OpenRouterModel (not OpenAIChatModel) so that
        # prepare_request() calls _openrouter_settings_to_openai_settings(),
        # which translates openrouter_provider → extra_body['provider'] and
        # openrouter_reasoning → extra_body['reasoning']. OpenAIChatModel
        # silently drops both fields.
        return OpenRouterModel(model_name=config.model, provider=provider, profile=profile)

    provider = OpenAIProvider(
        api_key=api_key,
        base_url=config.base_url,
    )
    return OpenAIChatModel(model_name=config.model, provider=provider)


def _create_model_with_key(config: ModelConfig, api_key: str | None) -> Model:
    """Create a single Pydantic AI model instance with the given API key."""
    if config.provider == "anthropic":
        provider = AnthropicProvider(api_key=api_key, base_url=config.base_url)
        return AnthropicModel(model_name=config.model, provider=provider)

    if config.provider in {"openai", "openrouter", "vllm", "huggingface"}:
        return _create_openai_compatible_model(config, api_key)

    raise ValueError(f"Unknown provider: {config.provider}")


def create_model(config: ModelConfig) -> Model:
    """Create a Pydantic AI model from ModelConfig.

    Routes by config.provider to the appropriate Pydantic AI model class.
    Reads API keys from environment variables specified in config.
    When api_key_env is a list, uses the first key.
    """
    api_key = None
    if config.api_key_env:
        env_name = config.api_key_env[0] if isinstance(config.api_key_env, list) else config.api_key_env
        api_key = get_api_key_from_env_name(env_name)

    return _create_model_with_key(config, api_key)


def create_model_pool(config: ModelConfig) -> list[Model]:
    """Create a list of model instances, one per API key.

    When api_key_env is a single string or None, returns a single-element list.
    When api_key_env is a list, creates one model per key for rotation.
    """
    if config.api_key_env is None:
        return [_create_model_with_key(config, api_key=None)]

    key_envs = config.api_key_env if isinstance(config.api_key_env, list) else [config.api_key_env]
    return [
        _create_model_with_key(config, get_api_key_from_env_name(env))
        for env in key_envs
    ]
