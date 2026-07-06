from __future__ import annotations

from pathlib import Path
from typing import Protocol

import structlog

from src.common.benchmark_sample import BenchmarkSample, IndexedEnvironment, PriorConflictContext
from src.common.config import ConditionConfig, DatasetFilter
from src.common.schemas import DatasetItem, StanceLabel

logger = structlog.get_logger(__name__)

# Canonical IE field order — deterministic for downstream reproducibility.
IE_NAMES: tuple[str, ...] = (
    "baseline",
    "ie_availability_absent",
    "ie_clarity_vague",
    "ie_consistency_contradiction",
    "ie_noise",
    "ie_prior_conflict",
)


class DatasetAdapter(Protocol):
    """Protocol for dataset loaders — any adapter must implement load()."""

    def load(self, filter: DatasetFilter) -> list[DatasetItem]:
        """Load and filter dataset observations.

        Args:
            filter: Filter criteria (election_ids, party_filter, statement_filter).

        Returns:
            Filtered list of DatasetItem observations.
        """
        ...


class BenchmarkAdapter:
    """Loads vaa_benchmark.jsonl into DatasetItem lists, expanding per IE.

    Each JSONL line is a BenchmarkSample. For each selected IE the sample is
    expanded into one DatasetItem, giving N items per sample where N is the
    number of selected IEs.

    Args:
        jsonl_path: Absolute path to the vaa_benchmark.jsonl file.
        conditions: Experiment-level condition configuration.
    """

    def __init__(self, jsonl_path: Path, conditions: ConditionConfig) -> None:
        self._jsonl_path = jsonl_path
        self._conditions = conditions

    def load(self, filter: DatasetFilter) -> list[DatasetItem]:
        """Load and expand benchmark samples into DatasetItem per IE.

        Args:
            filter: Filter criteria (election_ids, party_filter, statement_filter).

        Returns:
            Expanded, filtered list of DatasetItem observations.

        Raises:
            FileNotFoundError: If the JSONL file does not exist.
        """
        if not self._jsonl_path.exists():
            raise FileNotFoundError(f"Benchmark JSONL not found: {self._jsonl_path}")

        selected_ies = list(IE_NAMES) if self._conditions.ie_levels == "all" else list(self._conditions.ie_levels)

        items: list[DatasetItem] = []

        with self._jsonl_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                sample = BenchmarkSample.model_validate_json(raw_line)

                # --- election_ids filter ---
                if sample.metadata.election_id not in filter.election_ids:
                    continue

                # --- party_filter ---
                if filter.party_filter and sample.metadata.party_name not in filter.party_filter:
                    continue

                # --- statement_filter ---
                # Parse numeric suffix from statement_id (e.g. "bundestagswahl2025__s001" → 1).
                if filter.statement_filter is not None:
                    try:
                        stmt_num = int(sample.metadata.statement_id.rsplit("__s", 1)[-1])
                    except (ValueError, IndexError):
                        continue
                    if stmt_num not in filter.statement_filter:
                        continue

                # Resolve substitution token based on party_label condition.
                party_token: str = (
                    sample.metadata.party_anonymized
                    if self._conditions.party_label == "anonymized"
                    else sample.metadata.party_name
                )

                for ie_name in selected_ies:
                    raw_value = getattr(sample.environments, ie_name)

                    # Resolve chunks or skip.
                    evidence_index: int | None = None
                    expected_stance: StanceLabel | None = None
                    if ie_name == "ie_availability_absent" and raw_value is None:
                        # "no context" is the point — empty list is valid.
                        chunks: list[str] = []
                    elif raw_value is None:
                        # Missing IE for this sample → skip this IE.
                        continue
                    elif ie_name == "ie_prior_conflict":
                        # PriorConflictContext — extract evidence as single-item list.
                        assert isinstance(raw_value, PriorConflictContext)
                        chunks = [raw_value.evidence]
                        expected_stance = StanceLabel(raw_value.expected_stance)
                    elif isinstance(raw_value, IndexedEnvironment):
                        # Consistency / Noise — target chunk embedded among distractors.
                        chunks = list(raw_value.items)
                        evidence_index = raw_value.evidence_index
                    else:
                        chunks = list(raw_value)

                    substituted = [c.replace("[PARTY]", party_token) for c in chunks]

                    items.append(DatasetItem(
                        observation_id=sample.sample_id,
                        election_id=sample.metadata.election_id,
                        party_id=sample.metadata.party_id,
                        party_name=sample.metadata.party_name,
                        party_anonymized=sample.metadata.party_anonymized,
                        statement_id=sample.metadata.statement_id,
                        statement_number=None,
                        statement_text=sample.metadata.statement_text,
                        statement_category=None,
                        stance_label=StanceLabel(sample.party_position.stance),
                        rationale_text=sample.party_position.original_rationale or None,
                        has_rationale=bool(sample.party_position.original_rationale),
                        ie_name=ie_name,
                        ie_chunks=substituted,
                        evidence_index=evidence_index,
                        expected_stance=expected_stance,
                    ))

        return items
