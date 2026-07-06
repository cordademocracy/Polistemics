"""Rubric framework for LLM-judge evaluation metrics.

Defines SubQuestion, SubQuestionResult, and BaseRubric ABC.
Concrete rubrics (e.g. FaithfulnessRubric) subclass BaseRubric and
declare their sub-question inventories.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, Field, create_model

from src.common.schemas import DatasetItem, LLMOutput
from src.common.context_format import normalize_ie
from src.metrics.base import BaseMetric

if TYPE_CHECKING:
    from src.common.prompts import PromptBuilder
    from src.evaluate.panel import JudgePanel, SingleJudgeResponse


# ---------------------------------------------------------------------------
# SubQuestion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubQuestion:
    """A single evaluation question within a rubric.

    Args:
        id: Snake_case identifier, used as verdict field name.
        text: Question text posed to the judge.
        eval: Dispatch routing — "judge" for LLM panel, "programmatic" for code check.
        comparability: Scoring layer tag — "shared" or "condition-specific".
        programmatic_check: Method name on the Rubric (eval="programmatic" only).
        active_ies: If set, this sub-question is only active for these IEs.
            None means active for all IEs.
        pass_if: Whether a "yes" (True) or "no" (False) verdict means the item passes.
            "yes" = verdict True → score 1.0 (default, positive framing).
            "no"  = verdict True → score 0.0 (negative framing, absence of bad behavior).
    """

    id: str
    text: str
    eval: Literal["judge", "programmatic"]
    comparability: Literal["shared", "condition-specific"]
    programmatic_check: str | None = None
    active_ies: frozenset[str] | None = None
    pass_if: Literal["yes", "no"] = "yes"


# ---------------------------------------------------------------------------
# SubQuestionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubQuestionResult:
    """Result of evaluating a single sub-question.

    Args:
        sub_question_id: Matches SubQuestion.id.
        verdict: Final verdict (majority-voted for judge, direct for programmatic).
        source: Whether this came from the judge panel or a programmatic check.
        per_judge: Individual judge responses (None for programmatic).
        disagreement: True if judges disagreed (judge-evaluated only).
    """

    sub_question_id: str
    verdict: bool
    source: Literal["judge", "programmatic"]
    per_judge: list[SingleJudgeResponse] | None = None
    disagreement: bool = False


# ---------------------------------------------------------------------------
# BaseRubric
# ---------------------------------------------------------------------------


class BaseRubric(BaseMetric, ABC):
    """Abstract base for rubric-based LLM-judge metrics.

    Subclasses declare sub-question inventories via `shared_items` and
    `condition_specific` class variables. The rubric framework handles
    dispatch routing, verdict schema generation, and BaseMetric compatibility.
    """

    name: ClassVar[str]
    definition: ClassVar[str]
    shared_items: ClassVar[list[SubQuestion]]
    condition_specific: ClassVar[dict[str, list[SubQuestion]]]

    def __init__(self, panel: JudgePanel, prompt_builder: PromptBuilder) -> None:
        self._panel = panel
        self._prompt_builder = prompt_builder

    # --- Sub-question routing ---

    def active_questions(self, ie: str) -> list[SubQuestion]:
        """Return sub-questions active for this IE.

        Filters shared_items by active_ies membership, then appends
        condition-specific items for the IE.

        Args:
            ie: IE name (short or data form — normalized internally).

        Returns:
            Merged list of active SubQuestion instances.
        """
        ie_canonical = normalize_ie(ie)
        active = [
            q
            for q in self.shared_items
            if q.active_ies is None or ie_canonical in q.active_ies
        ]
        active += self.condition_specific.get(ie_canonical, [])
        return active

    # --- Dynamic verdict schema ---

    def build_verdict_type(self, ie: str) -> type[BaseModel]:
        """Build a dynamic Pydantic model with opaque q1..qN bool fields.

        Field names are positional (q1, q2, ...) rather than sub-question IDs
        so the structured-output schema does not expose semantic labels to the judge.
        The mapping back to sub-question IDs is positional — q{i} corresponds to the
        i-th judge sub-question returned by active_questions(ie).

        Args:
            ie: IE name (short or data form).

        Returns:
            A Pydantic BaseModel subclass with bool fields q1, q2, ..., qN.
        """
        ie_canonical = normalize_ie(ie)
        judge_qs = [q for q in self.active_questions(ie) if q.eval == "judge"]
        fields = {
            f"q{i}": (bool, Field(description=f"Question {i}"))
            for i, _ in enumerate(judge_qs, 1)
        }
        return create_model(f"{self.name}_{ie_canonical}_Verdict", **fields)

    def to_paper_matrix(self) -> str:
        """Generate a markdown table of all sub-questions for paper inclusion.

        Returns:
            Markdown-formatted table with columns: id, text, active_ies, eval, comparability.
        """
        all_qs: list[SubQuestion] = list(self.shared_items)
        for ie_qs in self.condition_specific.values():
            all_qs.extend(ie_qs)

        lines = ["| id | text | active_ies | eval | comparability |"]
        lines.append("|---|---|---|---|---|")
        for q in all_qs:
            ies = ", ".join(sorted(q.active_ies)) if q.active_ies else "all"
            lines.append(f"| {q.id} | {q.text} | {ies} | {q.eval} | {q.comparability} |")
        return "\n".join(lines)

    # --- Verdict scoring ---

    def _verdict_to_score(self, q: SubQuestion, verdict: bool, ie: str) -> float:
        """Convert a raw verdict to a 0.0/1.0 score, respecting pass_if polarity.

        Args:
            q: The sub-question definition.
            verdict: Raw boolean verdict (True = "yes" answer to the question).
            ie: Canonical IE name (unused in default impl; available for subclass override,
                e.g. Epistemic Calibration answerability-based flip).

        Returns:
            1.0 if the item passes, 0.0 otherwise.
        """
        if q.pass_if == "yes":
            return float(verdict)
        return float(not verdict)

    # --- BaseMetric interface ---

    @property
    def aggregation_method(self) -> str:
        return "majority_vote"

    @property
    def primary_score_key(self) -> str:
        return "adherence_rate"

    async def score(
        self,
        output: LLMOutput,
        ground_truth: DatasetItem,
        prior_responses: list | None = None,
    ) -> dict[str, float] | None:
        """Dispatch sub-questions and return per-sub-question verdicts as 0.0/1.0.

        Returns None when no sub-questions are active for this IE (e.g. Faithfulness
        and Impartiality on ie_availability_absent).  None signals the pipeline to
        skip persisting an ItemScore — keeps aggregations clean of non-applicable items.

        Args:
            output: The LLM output being evaluated.
            ground_truth: The ground-truth dataset item.
            prior_responses: Optional pre-existing SingleJudgeResponse objects from a
                prior partial attempt. Passed to dispatch() to skip re-calling judges
                that already succeeded.

        Returns:
            Dict mapping sub_question_id -> 0.0 or 1.0 (respecting pass_if polarity),
            or None if the rubric does not apply for this IE.
        """
        from src.evaluate.dispatcher import dispatch

        ie = ground_truth.ie_name
        # Rubric not applicable for this IE — skip dispatch entirely.
        if not self.active_questions(ie):
            return None
        results = await dispatch(
            self, ie, output, ground_truth, self._panel, self._prompt_builder,
            prior_responses=prior_responses,
        )
        ie_canonical = normalize_ie(ie)
        q_lookup = {q.id: q for q in self.active_questions(ie_canonical)}
        return {
            r.sub_question_id: self._verdict_to_score(
                q_lookup[r.sub_question_id], r.verdict, ie_canonical,
            )
            for r in results
        }

    def aggregate(self, run_scores: list[dict[str, float]]) -> dict[str, float]:
        """K-run majority per sub-question + adherence_rate.

        Args:
            run_scores: List of score dicts from K runs of score().

        Returns:
            Majority-voted dict with adherence_rate appended.
        """
        if not run_scores:
            return {"adherence_rate": 0.0}

        sub_q_ids = [k for k in run_scores[0] if k != "adherence_rate"]
        voted: dict[str, float] = {}
        for sq_id in sub_q_ids:
            values = [s[sq_id] for s in run_scores]
            voted[sq_id] = float(sum(values) >= len(values) / 2)

        if sub_q_ids:
            voted["adherence_rate"] = sum(voted[k] for k in sub_q_ids) / len(sub_q_ids)
        else:
            # No sub-questions fired — rubric not applicable for this IE (e.g. availability).
            # Use -1.0 as a sentinel so callers can distinguish "not evaluated" from "scored 0".
            voted["adherence_rate"] = -1.0
        return voted

    def summarize(self, batch_scores: list[dict[str, float]]) -> dict[str, float]:
        """Per sub-question pass rate + mean adherence_rate across batch.

        Args:
            batch_scores: List of aggregated score dicts from a batch of items.

        Returns:
            Dict with {key}_mean for each key present in the batch.
        """
        if not batch_scores:
            return {}

        all_keys = {k for s in batch_scores for k in s}
        result: dict[str, float] = {}
        for key in sorted(all_keys):
            # Exclude sentinel values (-1.0 = rubric not applicable) from mean computation.
            values = [s[key] for s in batch_scores if key in s and s[key] >= 0.0]
            result[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
        return result
