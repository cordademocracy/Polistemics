from src.metrics.registry import METRIC_REGISTRY as METRIC_REGISTRY

# Register metrics that need no experiment-specific configuration.
# Metrics needing config (e.g., sentiment with party_names) are registered
# at runtime via build_metrics() in the factory module.
