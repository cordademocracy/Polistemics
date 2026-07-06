"""Dispatcher for rubric-based LLM-judge evaluation.

Routes sub-questions to either the judge panel or programmatic checks,
then collects and returns SubQuestionResults.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.common.text import NamePattern, build_name_pattern
from src.evaluate.panel import PartialPanelResult

if TYPE_CHECKING:
    from src.common.prompts import PromptBuilder
    from src.common.schemas import DatasetItem, LLMOutput
    from src.evaluate.panel import JudgePanel
    from src.metrics.rubric import BaseRubric, SubQuestion, SubQuestionResult

logger = structlog.get_logger(__name__)


def _majority_vote(verdicts: list[bool]) -> bool:
    """Return True if a strict majority of verdicts are True."""
    return sum(verdicts) > len(verdicts) / 2


def _build_name_pattern(gt: DatasetItem) -> NamePattern:
    """Build a name-stripping pattern from the ground-truth party name."""
    return build_name_pattern([gt.party_name])


def _evaluate_judge_questions(
    judge_qs: list[SubQuestion],
    panel_result: object,
) -> list[SubQuestionResult]:
    """Extract per-sub-question verdicts from panel responses via majority vote.

    Args:
        judge_qs: Judge-evaluated sub-questions.
        panel_result: PanelResult from the judge panel.

    Returns:
        List of SubQuestionResult for each judge sub-question.
    """
    from src.metrics.rubric import SubQuestionResult

    succeeded = [r for r in panel_result.responses if r.result is not None]
    results: list[SubQuestionResult] = []

    for i, q in enumerate(judge_qs, 1):
        per_judge_verdicts = [getattr(r.result, f"q{i}") for r in succeeded]
        verdict = _majority_vote(per_judge_verdicts)
        disagreement = len(set(per_judge_verdicts)) > 1

        results.append(
            SubQuestionResult(
                sub_question_id=q.id,
                verdict=verdict,
                source="judge",
                per_judge=list(succeeded),
                disagreement=disagreement,
            )
        )

    return results


def _evaluate_programmatic_questions(
    prog_qs: list[SubQuestion],
    rubric: BaseRubric,
    observation: LLMOutput,
    gt: DatasetItem,
) -> list[SubQuestionResult]:
    """Evaluate programmatic sub-questions by calling rubric methods.

    Args:
        prog_qs: Programmatic sub-questions.
        rubric: The rubric instance with check methods.
        observation: The LLM output being evaluated.
        gt: Ground-truth dataset item.

    Returns:
        List of SubQuestionResult for each programmatic sub-question.
    """
    from src.metrics.rubric import SubQuestionResult

    results: list[SubQuestionResult] = []
    for q in prog_qs:
        check = getattr(rubric, q.programmatic_check)
        verdict = check(observation, gt)
        results.append(
            SubQuestionResult(
                sub_question_id=q.id,
                verdict=verdict,
                source="programmatic",
                per_judge=None,
                disagreement=False,
            )
        )
    return results


async def dispatch(
    rubric: BaseRubric,
    ie: str,
    observation: LLMOutput,
    gt: DatasetItem,
    panel: JudgePanel,
    prompt_builder: PromptBuilder,
    prior_responses: list[SingleJudgeResponse] | None = None,
) -> list[SubQuestionResult]:
    """Dispatch rubric sub-questions to judge panel and/or programmatic checks.

    On a retry pass, pass prior_responses to reuse already-succeeded judge results.
    Only judges not present in prior_responses (or that had errors) are re-called.

    Args:
        rubric: The rubric defining the evaluation dimension.
        ie: IE name (short or data form).
        observation: The LLM output being evaluated.
        gt: The ground-truth dataset item.
        panel: JudgePanel instance for LLM-judge sub-questions.
        prompt_builder: PromptBuilder for assembling judge prompts.
        prior_responses: Pre-existing SingleJudgeResponse objects from a prior partial attempt.

    Returns:
        List of SubQuestionResult — one per active sub-question.

    Raises:
        PartialPanelResult: If fewer judges than expected returned results after merging.
    """
    qs = rubric.active_questions(ie)
    judge_qs = [q for q in qs if q.eval == "judge"]
    prog_qs = [q for q in qs if q.eval == "programmatic"]

    results: list[SubQuestionResult] = []

    if judge_qs:
        verdict_type = rubric.build_verdict_type(ie)
        system, user = prompt_builder.build_judge_prompt(
            rubric, ie, observation, gt, name_pattern=None,
        )

        # Identify judges we already have successful results for.
        succeeded_ids: set[str] = set()
        merged_responses: list[SingleJudgeResponse] = []
        if prior_responses:
            for r in prior_responses:
                if r.result is not None:
                    succeeded_ids.add(r.model_id)
                    merged_responses.append(r)

        # Only call judges we're missing.
        new_result = await panel.evaluate(user, verdict_type, system, skip_model_ids=succeeded_ids)
        merged_responses.extend(new_result.responses)

        # Persist audit for new responses only (prior responses were already audited).
        await panel.write_audit(rubric.name, observation.observation_id, new_result.responses)

        from src.evaluate.panel import PanelResult
        full_panel_result = PanelResult(responses=merged_responses)

        n_expected = len(panel.model_ids)
        if full_panel_result.n_succeeded < n_expected:
            raise PartialPanelResult(full_panel_result.n_succeeded, n_expected, full_panel_result.responses)

        results.extend(_evaluate_judge_questions(judge_qs, full_panel_result))

    results.extend(
        _evaluate_programmatic_questions(prog_qs, rubric, observation, gt)
    )

    return results
