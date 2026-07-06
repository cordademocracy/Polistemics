"""Model-related infrastructure: config schema, factory, settings builder, retry utilities."""
from src.common.models.config import (
    CollectionConfig,
    ConcurrencyConfig,
    ConditionConfig,
    DatasetFilter,
    EvaluationConfig,
    ExperimentConfig,
    JudgesConfig,
    ModelConfig,
    PromptConfig,
    RunConfig,
    WebSearchConfig,
    load_experiment_config,
    load_model_configs,
)
from src.common.models.factory import (
    WebSearchOpenRouterModel,
    create_model,
    create_model_pool,
)
from src.common.models.retry import compute_backoff, is_rate_limit_error, parse_retry_after
from src.common.models.settings import build_model_settings

__all__ = [
    # config
    "CollectionConfig",
    "ConcurrencyConfig",
    "ConditionConfig",
    "DatasetFilter",
    "EvaluationConfig",
    "ExperimentConfig",
    "JudgesConfig",
    "ModelConfig",
    "PromptConfig",
    "RunConfig",
    "WebSearchConfig",
    "load_experiment_config",
    "load_model_configs",
    # factory
    "WebSearchOpenRouterModel",
    "create_model",
    "create_model_pool",
    # settings
    "build_model_settings",
    # retry
    "is_rate_limit_error",
    "parse_retry_after",
    "compute_backoff",
]
