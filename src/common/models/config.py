from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field

from src.common.schemas import PromptVariation


class DatasetFilter(BaseModel):
    path: str = ""
    source: str = "ResPolitica"
    election_ids: list[str]
    party_filter: list[str] | None = None
    statement_filter: list[int] | None = None


class EvaluationConfig(BaseModel):
    bootstrap_resamples: int = 10000
    significance_threshold: float = 0.05


class PromptConfig(BaseModel):
    levels: list[PromptVariation]
    system_prompt: str
    templates: dict[str, str]


class RunConfig(BaseModel):
    temperature: float | None = None  # None = use model default
    k: int
    seed: int | None = None


class ConcurrencyConfig(BaseModel):
    default: int = 5
    per_model: dict[str, int] = Field(default_factory=dict)


class CollectionConfig(BaseModel):
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 60.0
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)


class JudgesConfig(BaseModel):
    """Configuration for the LLM judge panel.

    All model-level settings (max_tokens, reasoning_effort, output_mode,
    api_key_env, etc.) live exclusively in models.yaml under ``judge_models``.
    Only panel-level orchestration settings belong here.
    """
    panel: list[str] = Field(default_factory=list)  # model IDs from models.yaml judge_models
    temperature: float = 0.0
    max_concurrent_per_model: int = 5
    timeout_seconds: float | None = None  # seconds per judge call; None = no timeout
    max_retries: int = 2                  # attempts = max_retries + 1
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 60.0


class ConditionConfig(BaseModel):
    """Experimental conditions applied per work-item or at experiment level.

    ie_levels controls which Information Environment variants are included
    (per work-item selection). All other dimensions are experiment-level constants.
    """

    ie_levels: list[str] | Literal["all"] = "all"  # sentinel = all 6 IE names
    party_label: Literal["real", "anonymized"] = "real"
    response_language: str = ""  # required — e.g. "German", "English", "Dutch"; flagged if unset
    year_mention: Literal["none", "future"] = "none"


class ExperimentConfig(BaseModel):
    experiment_id: str
    experiment_name: str
    description: str | None = None
    dataset: DatasetFilter
    conditions: ConditionConfig = Field(default_factory=ConditionConfig)
    model_ids: list[str]
    prompts: PromptConfig
    runs: list[RunConfig]
    metrics: list[str]
    judges: JudgesConfig = Field(default_factory=JudgesConfig)
    collection: CollectionConfig = Field(default_factory=CollectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class WebSearchConfig(BaseModel):
    """Configuration for OpenRouter server-side web search."""
    enabled: bool = False
    search_context_size: Literal["low", "medium", "high"] = "medium"
    max_results: int = 5  # 1-25, default 5 per OpenRouter docs


class ModelConfig(BaseModel):
    id: str
    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int = 1024
    api_key_env: str | list[str] | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None  # "low", "medium", "high" etc. — percentage-based (OpenAI/Grok only reliable)
    reasoning_max_tokens: int | None = None  # hard token cap for reasoning (Qwen, DeepSeek, Gemini, Anthropic)
    openrouter_provider: dict | None = None  # OpenRouter provider routing (e.g. {"order": ["amazon-bedrock"]})
    web_search: WebSearchConfig | None = None
    # output_mode is only relevant for judge models (controls how structured
    # output is requested from pydantic_ai: tool call, native JSON, or prompted)
    output_mode: Literal["native", "tool", "prompted"] | None = None


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load YAML → ExperimentConfig with validation."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)


def load_model_configs(path: str) -> dict[str, ModelConfig]:
    """Load models.yaml → dict keyed by model id.

    Reads both the ``models`` list and the optional ``judge_models`` list,
    merging them into a single dict. judge_models may carry output_mode.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    configs: dict[str, ModelConfig] = {}
    for entry in data.get("models", []):
        mc = ModelConfig(**entry)
        configs[mc.id] = mc
    for entry in data.get("judge_models", []):
        mc = ModelConfig(**entry)
        configs[mc.id] = mc
    return configs
