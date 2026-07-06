"""Backward-compat shim — config has moved to src.common.models.config."""
from src.common.models.config import (  # noqa: F401
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
