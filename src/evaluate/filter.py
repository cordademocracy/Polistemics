from __future__ import annotations

from pydantic import BaseModel

from src.common.schemas import LLMOutput, OutputStatus, PromptVariation


class OutputFilter(BaseModel):
    """Filter for subsetting LLMOutput records during evaluation."""

    party_ids: list[str] | None = None
    model_ids: list[str] | None = None
    statement_ids: list[str] | None = None
    prompt_variations: list[PromptVariation] | None = None
    temperatures: list[float] | None = None
    statuses: list[OutputStatus] | None = None


def filter_outputs(
    outputs: list[LLMOutput],
    output_filter: OutputFilter,
) -> list[LLMOutput]:
    """Apply filter to a list of LLMOutput records.

    Each non-None filter field narrows the result set. Multiple fields
    combine with AND logic.
    """
    result = outputs
    if output_filter.party_ids is not None:
        party_set = set(output_filter.party_ids)
        result = [o for o in result if o.party_id in party_set]
    if output_filter.model_ids is not None:
        model_set = set(output_filter.model_ids)
        result = [o for o in result if o.model_id in model_set]
    if output_filter.statement_ids is not None:
        stmt_set = set(output_filter.statement_ids)
        result = [o for o in result if o.statement_id in stmt_set]
    if output_filter.prompt_variations is not None:
        pv_set = set(output_filter.prompt_variations)
        result = [o for o in result if o.prompt_variation in pv_set]
    if output_filter.temperatures is not None:
        temp_set = set(output_filter.temperatures)
        result = [o for o in result if o.temperature in temp_set]
    if output_filter.statuses is not None:
        status_set = set(output_filter.statuses)
        result = [o for o in result if o.status in status_set]
    return result
