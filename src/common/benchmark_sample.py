"""Local mirror of the BenchmarkSample schema from ResPolitica.

This module mirrors ResPolitica/scripts/benchmark/models.py exactly for the
classes Polistemics consumes. Keep in sync with the upstream source when the
ResPolitica schema evolves.

Mirrored classes: SampleMetadata, PartyPosition, PriorConflictContext,
Environments, BenchmarkSample.

Not mirrored (ResPolitica-internal): StandardizedEvidence, VagueEvidence,
JudgeVerdict.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

_STRICT = ConfigDict(extra="forbid")


class SampleMetadata(BaseModel):
    election_id: str
    country: str
    election_year: int
    party_id: str
    party_name: str
    party_anonymized: str
    statement_id: str
    statement_text: str


class PartyPosition(BaseModel):
    model_config = _STRICT

    stance: Literal["Agree", "Disagree", "Neutral"]
    stance_numeric: Literal[0.0, 0.5, 1.0]
    original_rationale: str


class PriorConflictContext(BaseModel):
    model_config = _STRICT

    evidence: str
    expected_stance: Literal["Agree", "Disagree"]


class IndexedEnvironment(BaseModel):
    """IE variant that embeds the target chunk among distractors."""

    model_config = _STRICT

    items: list[str]
    evidence_index: int


class Environments(BaseModel):
    model_config = _STRICT

    baseline: list[str] | None = None
    ie_availability_absent: list[str] | None = None
    ie_clarity_vague: list[str] | None = None
    ie_clarity_vague_mode: int | None = None
    ie_consistency_contradiction: IndexedEnvironment | None = None
    ie_noise: IndexedEnvironment | None = None
    ie_prior_conflict: PriorConflictContext | None = None


class BenchmarkSample(BaseModel):
    model_config = _STRICT

    sample_id: str
    metadata: SampleMetadata
    party_position: PartyPosition
    environments: Environments
