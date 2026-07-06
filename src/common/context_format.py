"""Context label formatter for LLM judge prompts.

Formats source chunks with [TARGET]/[DISTRACTOR] labels per IE condition.
Cross-rubric concern — used by PromptBuilder.build_judge_prompt.
"""
from __future__ import annotations

from collections.abc import Callable

type _FormatterFn = Callable[[list[str], int | None], str]

_IE_ALIAS: dict[str, str] = {
    "baseline": "baseline",
    "ie_availability_absent": "availability",
    "ie_clarity_vague": "clarity",
    "ie_consistency_contradiction": "consistency",
    "ie_noise": "noise",
    "ie_prior_conflict": "prior_conflict",
    "availability": "availability",
    "clarity": "clarity",
    "consistency": "consistency",
    "noise": "noise",
    "prior_conflict": "prior_conflict",
}

IE_PROPERTY_OF_CONTEXT: dict[str, str] = {
    "baseline": (
        "The provided context contains the answerer's task-relevant evidence;"
        " the stance is determinable from it."
        " The expected stance based on the evidence is: {expected_stance}."
    ),
    "availability": "No evidence was provided to the answerer; the context contains only a placeholder indicating no documents were found.",
    "clarity": "The [TARGET] in the context was constructed to lack a determinable stance.",
    "consistency": (
        "The context contains two equally-weighted [TARGET] chunks"
        " with opposing stances about the same party. There is no determinable stance."
    ),
    "noise": (
        "The context contains the answerer's task-relevant [TARGET] evidence"
        " alongside one or more [DISTRACTOR] chunks."
        " The expected stance based on the evidence is: {expected_stance}."
    ),
    "prior_conflict": (
        "The [TARGET] in the context may conflict with the answerer's"
        " likely prior knowledge about the party."
        " The expected stance based on the evidence is: {expected_stance}."
    ),
}


def normalize_ie(ie: str) -> str:
    """Normalize an IE name to its canonical short form.

    Args:
        ie: IE name in either short form ("clarity") or data form ("ie_clarity_vague").

    Returns:
        Canonical short form (e.g. "clarity").

    Raises:
        ValueError: If the IE name is not recognized.
    """
    canonical = _IE_ALIAS.get(ie)
    if canonical is None:
        raise ValueError(f"Unknown IE name: {ie!r}")
    return canonical


def format_source(ie: str, chunks: list[str], evidence_index: int | None = None) -> str:
    """Format source chunks with [TARGET]/[DISTRACTOR] labels per IE condition.

    Args:
        ie: IE name (short or data form).
        chunks: Pre-built context chunks from DatasetItem.ie_chunks.
        evidence_index: Index of the target chunk in ie_chunks (Noise IE).

    Returns:
        Formatted source block with labelled chunks.

    Raises:
        ValueError: If the IE name is not recognized.
    """
    canonical = normalize_ie(ie)
    formatter = _FORMATTERS[canonical]
    return formatter(chunks, evidence_index)


def _format_single_target(chunks: list[str], _evidence_index: int | None) -> str:
    return f"[TARGET] {chunks[0]}"


def _format_availability(_chunks: list[str], _evidence_index: int | None) -> str:
    return "No relevant documents found."


def _format_consistency(chunks: list[str], _evidence_index: int | None) -> str:
    return f"[TARGET] {chunks[0]}\n\n[TARGET] {chunks[1]}"


def _format_noise(chunks: list[str], evidence_index: int | None) -> str:
    if evidence_index is None:
        raise ValueError("evidence_index is required for noise IE")

    parts: list[str] = [f"[TARGET] {chunks[evidence_index]}"]
    for i, chunk in enumerate(chunks):
        if i != evidence_index:
            parts.append(f"[DISTRACTOR] {chunk}")
    return "\n\n".join(parts)


_FORMATTERS: dict[str, _FormatterFn] = {
    "baseline": _format_single_target,
    "availability": _format_availability,
    "clarity": _format_single_target,
    "consistency": _format_consistency,
    "noise": _format_noise,
    "prior_conflict": _format_single_target,
}
