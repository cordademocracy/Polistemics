from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.common.config import PromptConfig
from src.common.schemas import DatasetItem, PromptVariation
from src.common.text import NamePattern, strip_party_names
from src.common.context_format import IE_PROPERTY_OF_CONTEXT, format_source, normalize_ie

if TYPE_CHECKING:
    from src.common.schemas import LLMOutput
    from src.metrics.rubric import BaseRubric


def _format_chunks(chunks: list[str]) -> str:
    """Render IE chunks as numbered passages [1]...[N]. Empty list → sentinel string."""
    if not chunks:
        return "No relevant documents found."
    return "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1))


class PromptBuilder:
    """Loads prompt templates and fills placeholders from DatasetItem fields."""

    def __init__(
        self,
        config: PromptConfig,
        templates_dir: Path,
        response_language: str = "",
        party_label: str = "real",
    ) -> None:
        self._config = config
        self._templates_dir = templates_dir
        self._response_language = response_language
        self._party_label = party_label
        self._system_template: str | None = None
        self._judge_template: str | None = None
        self._templates: dict[str, str] = {}

    def build_system_prompt(self, item: DatasetItem) -> str:
        """Return the system prompt with IE context and language injected for this item.

        The raw template is read once and cached in ``_system_template``;
        formatting happens on every call.
        """
        if not self._response_language:
            raise ValueError(
                "conditions.response_language is not set. "
                "Set it in your experiment YAML, e.g. response_language: \"German\"."
            )
        if self._system_template is None:
            path = self._templates_dir / self._config.system_prompt
            self._system_template = path.read_text().strip()
        return self._system_template.format(
            context=_format_chunks(item.ie_chunks),
            language=self._response_language,
        )

    def build_user_prompt(self, item: DatasetItem, variation: PromptVariation) -> str:
        template = self._load_template(variation)
        party_name = item.party_anonymized if self._party_label == "anonymized" else item.party_name
        return template.format(
            party_name=party_name,
            statement_text=item.statement_text,
            election_id=item.election_id,
            party_id=item.party_id,
            statement_number=item.statement_number or "",
        )

    def build_judge_prompt(
        self,
        rubric: BaseRubric,
        ie: str,
        observation: LLMOutput,
        gt: DatasetItem,
        name_pattern: NamePattern | None = None,
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for one judge call.

        Args:
            rubric: The rubric defining the evaluation dimension.
            ie: IE name (short or data form).
            observation: The LLM output being evaluated.
            gt: The ground-truth dataset item.
            name_pattern: Compiled party-name pattern for name stripping.
                If None, no name stripping is applied to the answer.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        ie_canonical = normalize_ie(ie)

        property_text = IE_PROPERTY_OF_CONTEXT[ie_canonical]
        if "{expected_stance}" in property_text:
            # For prior_conflict: use the evidence-induced expected_stance (opposite of real-world).
            # For all other IEs (baseline, noise): fall back to the real-world stance_label.
            stance = gt.expected_stance if gt.expected_stance is not None else gt.stance_label
            property_text = property_text.format(expected_stance=stance.value)

        system = self._load_judge_template().format(
            dimension_name=rubric.name,
            dimension_definition=rubric.definition,
            property_of_context=property_text,
            questions=self._render_questions(rubric, ie),
        )

        party_label = gt.party_name if name_pattern is None else gt.party_anonymized
        query = f"What is {party_label}'s position on: {gt.statement_text}"
        source = format_source(ie, gt.ie_chunks, gt.evidence_index)

        answer = observation.predicted_explanation
        if name_pattern is not None:
            source = strip_party_names(source, name_pattern)
            answer = strip_party_names(answer, name_pattern)

        user = f"QUERY: {query}\n\nCONTEXT:\n{source}\n\nANSWER:\n{answer}"

        return system, user

    def _render_questions(self, rubric: BaseRubric, ie: str) -> str:
        """Render judge sub-questions as a numbered bullet list."""
        judge_qs = [q for q in rubric.active_questions(ie) if q.eval == "judge"]
        return "\n".join(f"- Q{i}: {q.text}" for i, q in enumerate(judge_qs, 1))

    def _load_judge_template(self) -> str:
        if self._judge_template is None:
            path = self._templates_dir / "judge.txt"
            self._judge_template = path.read_text().strip()
        return self._judge_template

    def _load_template(self, variation: PromptVariation) -> str:
        if variation.value not in self._templates:
            filename = self._config.templates[variation.value]
            path = self._templates_dir / filename
            self._templates[variation.value] = path.read_text().strip()
        return self._templates[variation.value]
