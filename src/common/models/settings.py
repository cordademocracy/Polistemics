from __future__ import annotations

from src.common.models.config import ModelConfig


def build_model_settings(mc: ModelConfig) -> dict[str, object]:
    """Build the model_settings dict for a pydantic_ai agent.run() call.

    Handles reasoning parameter routing correctly per provider:
    - OpenAI-native (provider == "openai"): use openai_reasoning_effort
    - All other providers with reasoning_effort (openrouter, etc.):
      use openrouter_reasoning = {"effort": ...} — the OpenRouter unified
      reasoning API that normalises effort across model families (Qwen3,
      DeepSeek, Gemini, etc.). Do NOT use openai_reasoning_effort here;
      that field targets the OpenAI native API and is ignored by the
      OpenRouter client even when the underlying model is openai/*.
    - No reasoning_effort set: no reasoning params added.

    Also injects openrouter_provider if configured on the ModelConfig.

    Note: temperature is intentionally excluded here because the collect
    pipeline overrides it per work-item *after* calling this function.
    The judge panel sets temperature separately at the JudgePanel level.

    Args:
        mc: The ModelConfig for the model being called.

    Returns:
        A dict suitable for passing as model_settings to agent.run().
    """
    settings: dict[str, object] = {"max_tokens": mc.max_tokens}

    if mc.reasoning_max_tokens is not None:
        # Hard token cap — preferred for Qwen, DeepSeek, Gemini, Anthropic where
        # effort percentages are not reliably honored. Takes priority over effort.
        settings["openrouter_reasoning"] = {"max_tokens": mc.reasoning_max_tokens}
    elif mc.reasoning_effort is not None:
        if mc.provider == "openai":
            settings["openai_reasoning_effort"] = mc.reasoning_effort
        else:
            # Percentage-based effort — reliable for OpenAI/Grok via OpenRouter.
            settings["openrouter_reasoning"] = {"effort": mc.reasoning_effort}

    if mc.openrouter_provider is not None:
        settings["openrouter_provider"] = mc.openrouter_provider

    return settings
