from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog

from src.metrics.faithfulness import FaithfulnessRubric
from src.metrics.impartiality import ImpartialityRubric
from src.metrics.epistemic_calibration import EpistemicCalibrationRubric

if TYPE_CHECKING:
    from src.common.prompts import PromptBuilder
    from src.common.schemas import DatasetItem
    from src.evaluate.panel import JudgePanel
    from src.metrics.base import BaseMetric

logger = structlog.get_logger(__name__)

_SIMPLE_METRICS: dict[str, type] = {}


def build_metrics(
    metric_names: list[str],
    gt_items: list[DatasetItem],
    panel: JudgePanel | None = None,
    prompt_builder: PromptBuilder | None = None,
) -> dict[str, Callable[[], BaseMetric]]:
    """Construct metric factories from metric names + ground truth data.

    Returns a dict of name -> callable that produces a configured metric instance.
    Unknown metric names are logged and skipped.

    Args:
        metric_names: List of metric identifiers to build.
        gt_items: Ground truth dataset items (used to derive party names).
        panel: Optional JudgePanel instance for LLM-judge-based metrics.
            If None and a judge metric is requested, it is skipped with a warning.
        prompt_builder: Optional PromptBuilder for rubric-based metrics.
            If None and a rubric metric is requested, it is skipped with a warning.
    """
    result: dict[str, Callable[[], BaseMetric]] = {}

    for name in metric_names:
        if name in _SIMPLE_METRICS:
            cls = _SIMPLE_METRICS[name]
            result[name] = cls
        elif name == "faithfulness":
            if panel is None or prompt_builder is None:
                logger.warning(
                    "faithfulness rubric requires judge panel + prompt_builder — skipping",
                )
                continue
            result[name] = lambda p=panel, pb=prompt_builder: FaithfulnessRubric(
                panel=p,
                prompt_builder=pb,
            )
        elif name == "impartiality":
            if panel is None or prompt_builder is None:
                logger.warning(
                    "impartiality rubric requires judge panel + prompt_builder — skipping",
                )
                continue
            result[name] = lambda p=panel, pb=prompt_builder: ImpartialityRubric(
                panel=p,
                prompt_builder=pb,
            )
        elif name == "epistemic_calibration":
            if panel is None or prompt_builder is None:
                logger.warning(
                    "epistemic_calibration rubric requires judge panel + prompt_builder — skipping",
                )
                continue
            result[name] = lambda p=panel, pb=prompt_builder: EpistemicCalibrationRubric(
                panel=p,
                prompt_builder=pb,
            )
        else:
            logger.warning("unknown_metric_skipped", metric=name)

    return result
