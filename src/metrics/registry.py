from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.metrics.base import BaseMetric

METRIC_REGISTRY: dict[str, type[BaseMetric] | Callable[[], BaseMetric]] = {}
