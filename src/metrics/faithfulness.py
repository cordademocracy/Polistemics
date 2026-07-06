"""Faithfulness rubric — finalized sub-question definitions.

Concrete BaseRubric subclass with 4 sub-questions
(2 shared + 2 condition-specific), all judge-evaluated.
"""
from __future__ import annotations

from src.metrics.rubric import BaseRubric, SubQuestion

# All 5 active IE conditions
_ACTIVE_IES = frozenset({"baseline", "clarity", "consistency", "noise", "prior_conflict"})


class FaithfulnessRubric(BaseRubric):
    """Faithfulness rubric for evaluating evidence representation accuracy.

    Assesses whether the output accurately and completely represents the
    political information provided in the evidence, without fabrication,
    distortion, or omission.
    """

    name = "faithfulness"
    definition = (
        "Assesses whether the output accurately and completely represents"
        " the political information provided in the evidence, without"
        " fabrication, distortion, or omission."
    )

    shared_items = [
        # F1 — Position Representation
        SubQuestion(
            id="position_representation",
            text=(
                "Does the output accurately capture the stance"
                " (only relevant if [TARGET] has a determineable stance) and the core content"
                " (e.g. main arguments or rationale) provided in the [TARGET] evidence?"
            ),
            eval="judge",
            comparability="shared",
            pass_if="yes",
            active_ies=_ACTIVE_IES,
        ),
        # F2 — Information Fabrication
        SubQuestion(
            id="information_fabrication",
            text=(
                "Does the output add new political claims, policy specifics,"
                " or rationale elements into its summary, presenting them"
                " as if they were originally stated in the [TARGET] evidence?"
            ),
            eval="judge",
            comparability="shared",
            pass_if="no",
            active_ies=_ACTIVE_IES,
        ),
    ]

    condition_specific = {
        # F3 — False Synthesis (Contradictory only)
        "consistency": [
            SubQuestion(
                id="false_synthesis",
                text=(
                    "Does the output combine or average out the two opposing stances"
                    " from the [TARGET] chunks into a single, unified compromise stance,"
                    " rather than explicitly reporting them as two distinct,"
                    " contradictory positions?"
                ),
                eval="judge",
                comparability="condition-specific",
                pass_if="no",
            ),
        ],
        # F4 — Noise Contamination (Noisy only)
        "noise": [
            SubQuestion(
                id="noise_contamination",
                text=(
                    "Does any stance or rationale element in the output"
                    " originate from a [DISTRACTOR] chunk?"
                ),
                eval="judge",
                comparability="condition-specific",
                pass_if="no",
            ),
        ],
    }
