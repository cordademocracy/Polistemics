"""Backward-compat shim — model_factory has moved to src.common.models.factory."""
from src.common.models.factory import (  # noqa: F401
    WebSearchOpenRouterModel,
    create_model,
    create_model_pool,
)
