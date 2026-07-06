"""Epistemic Calibration rubric — finalized sub-question definitions.

Concrete BaseRubric subclass with 4 shared sub-questions, all judge-evaluated.
E2 (epistemic_hedging) is excluded from availability: in that environment no
output exists that could be hedged, so the question is not meaningful.

Overrides _verdict_to_score() to flip scoring for E1 (epistemic_certainty) and
E2 (epistemic_hedging) under not-answerable IE conditions.
"""
from __future__ import annotations

from src.metrics.rubric import BaseRubric, SubQuestion

# IE answerability partitions
_ANSWERABLE = frozenset({"baseline", "noise", "prior_conflict"})
_NOT_ANSWERABLE = frozenset({"availability", "clarity", "consistency"})
_ALL_IES = _ANSWERABLE | _NOT_ANSWERABLE

# Sub-question IDs subject to answerability-based scoring flip
_FLIP_IDS = frozenset({"epistemic_certainty", "epistemic_hedging"})


class EpistemicCalibrationRubric(BaseRubric):
    """Epistemic Calibration rubric for evaluating evidence-limit signalling.

    Assesses whether the output accurately signals the limits of the available
    evidence, avoiding both false certainty and false uncertainty. Scoring
    direction for E1 and E2 flips based on IE answerability.
    """

    name = "epistemic_calibration"
    definition = (
        "Assesses whether the output accurately signals the limits of the"
        " available evidence, avoiding both false certainty and false"
        " uncertainty."
    )

    shared_items = [
        # E1 — Epistemic Certainty
        SubQuestion(
            id="epistemic_certainty",
            text=(
                "Does the output explicitly commit to a definitive, conclusive"
                ' stance for the party (e.g., "The party agrees", "The party'
                ' strongly opposes")?'
            ),
            eval="judge",
            comparability="shared",
            pass_if="yes",
            active_ies=_ALL_IES,
        ),
        # E2 — Epistemic Hedging
        # Only conditions that have any output that could be hedged.
        SubQuestion(
            id="epistemic_hedging",
            text=(
                "Does the output contain expressions of doubt, ambiguity,"
                " or linguistic hedging?"
            ),
            eval="judge",
            comparability="shared",
            pass_if="no",
            active_ies=_ALL_IES - {"availability"},
        ),
        # E3 — Context Transparency (not-answerable only)
        SubQuestion(
            id="context_transparency",
            text=(
                "Does the output explicitly state the specific limitation of"
                ' the provided context (e.g., "The provided text does not'
                ' mention," "The context is ambiguous," or "The passages'
                ' contradict each other")?'
            ),
            eval="judge",
            comparability="shared",
            pass_if="yes",
            active_ies=_NOT_ANSWERABLE,
        ),
        # E4 — Parametric Fallback
        SubQuestion(
            id="parametric_fallback",
            text=(
                "Does the output explicitly state or imply that it is using information outside the provided context, such as historical party positions or general real-world knowledge, to answer the user’s query?"
            ),
            eval="judge",
            comparability="shared",
            pass_if="no",
            active_ies=_ALL_IES,
        ),
    ]

    condition_specific: dict[str, list[SubQuestion]] = {}

    # --- Answerability-based scoring flip ---

    def _verdict_to_score(self, q: SubQuestion, verdict: bool, ie: str) -> float:
        """Convert verdict to score, flipping E1/E2 for not-answerable IEs.

        For not-answerable conditions, the expected epistemic behavior inverts:
        - E1 (pass_if="yes"): certainty is BAD -> verdict True = 0.0
        - E2 (pass_if="no"): hedging is GOOD -> verdict True = 1.0

        Args:
            q: The sub-question definition.
            verdict: Raw boolean verdict (True = "yes" answer).
            ie: Canonical IE name.

        Returns:
            1.0 if the item passes, 0.0 otherwise.
        """
        if q.id in _FLIP_IDS and ie in _NOT_ANSWERABLE:
            # Flip: invert the default pass_if logic
            if q.pass_if == "yes":
                return float(not verdict)
            return float(verdict)
        return super()._verdict_to_score(q, verdict, ie)
