from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.schemas import DatasetItem, LLMOutput


class BaseMetric(ABC):
    """Interface for all evaluation metrics.

    Three levels of computation:
    - score(): per-individual-output scoring (called by evaluation pipeline)
    - aggregate(): K-run robustness aggregation (called by evaluation pipeline)
    - summarize(): batch-level statistics (called by analysis module)

    The metric defines HOW to compute at each level.
    The pipeline/analysis decides WHEN and on WHAT.

    Contract: aggregate() must preserve all score keys that summarize() needs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric identifier used in registry and file paths."""
        ...

    @property
    @abstractmethod
    def aggregation_method(self) -> str:
        """Description of K-run aggregation strategy (e.g., 'majority_vote', 'mean')."""
        ...

    @property
    @abstractmethod
    def primary_score_key(self) -> str:
        """Key from aggregate() output used for cross-group comparisons."""
        ...

    @abstractmethod
    async def score(
        self,
        output: LLMOutput,
        ground_truth: DatasetItem,
    ) -> dict[str, float] | None:
        """Score a single LLM output against ground truth.

        Returns metric-specific scores, or None if the metric does not
        apply to this (output, ground_truth) combination (e.g. a rubric
        with no active sub-questions for the IE).  None signals the
        pipeline to skip persisting an ItemScore for this item.

        Must include enough information for aggregate() and summarize()
        to work when not None.
        """
        ...

    @abstractmethod
    def aggregate(
        self,
        run_scores: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate K run scores into a single score for robustness.

        Strategy is metric-specific (majority vote, mean, etc.).
        Must preserve all keys that summarize() needs.
        """
        ...

    @abstractmethod
    def summarize(
        self,
        batch_scores: list[dict[str, float]],
    ) -> dict[str, float]:
        """Compute summary statistics for a batch of aggregated scores.

        Called by analysis module after grouping (e.g., all items for one party).
        The metric doesn't know what the group is — it just summarizes.
        Receives the .scores dicts from AggregatedScore objects.
        """
        ...
