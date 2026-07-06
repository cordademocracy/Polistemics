from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class StanceLabel(StrEnum):
    AGREE = "Agree"
    DISAGREE = "Disagree"
    NEUTRAL = "Neutral"
    UNCERTAIN = "Uncertain"


class PromptVariation(StrEnum):
    DEFAULT = "default"
    MINIMAL = "minimal"      # legacy — kept for ablation comparison tests
    CONTEXTUAL = "contextual"  # legacy — kept for ablation comparison tests


class OutputStatus(StrEnum):
    SUCCESS = "success"
    PARSE_ERROR = "parse_error"
    REFUSAL = "refusal"
    API_ERROR = "api_error"


class RefusalType(StrEnum):
    HARD = "hard"
    SOFT = "soft"
    NONE = "none"


class DatasetItem(BaseModel):
    """A single party × statement observation from the VAA dataset."""
    observation_id: str
    election_id: str
    party_id: str
    party_name: str
    party_anonymized: str          # anonymized party name (owned by ResPolitica)
    statement_id: str
    statement_number: int | None   # None for BenchmarkSample entries
    statement_text: str
    statement_category: str | None
    stance_label: StanceLabel
    rationale_text: str | None
    has_rationale: bool
    ie_name: str                   # which IE this item represents (e.g. "baseline", "ie_clarity_vague")
    ie_chunks: list[str]           # pre-built context chunks with [PARTY] already substituted; empty for ie_availability_absent
    evidence_index: int | None = None  # index of target chunk in ie_chunks (Noise, Consistency)
    expected_stance: StanceLabel | None = None  # evidence-induced stance (Prior Conflict only); None for all other IEs


class LLMOutput(BaseModel):
    """A single LLM response for a (party, statement, model, prompt_variation, run) tuple."""
    observation_id: str
    statement_id: str
    party_id: str
    experiment_id: str
    model_id: str
    prompt_variation: PromptVariation
    run_index: int
    temperature: float | None
    predicted_stance: StanceLabel | None
    predicted_explanation: str
    timestamp: datetime
    latency_ms: float
    tokens_input: int
    tokens_output: int
    cost_usd: float | None
    status: OutputStatus
    error_message: str | None
    refusal_type: RefusalType | None
    ie_name: str
    condition_id: str              # denormalized: {ie_name}__{party_label}__{response_language}__{year_mention}
    collection_hash: str | None = None  # 8-char hex fingerprint of collection config at generation time


class ItemScore(BaseModel):
    """Score for a single LLM output from a single metric."""
    observation_id: str
    party_id: str
    statement_id: str
    model_id: str
    prompt_variation: PromptVariation
    run_index: int
    temperature: float | None
    metric_name: str
    scores: dict[str, float]
    ie_name: str
    judge_hash: str | None = None  # 8-char hex fingerprint of judge config at scoring time


class AggregatedScore(BaseModel):
    """K-run aggregated score for a single (observation, model, prompt_variation) tuple."""
    observation_id: str
    party_id: str
    statement_id: str
    model_id: str
    prompt_variation: PromptVariation
    temperature: float | None
    metric_name: str
    scores: dict[str, float]
    k_runs: int
    aggregation_method: str
    ie_name: str


class WorkItem(BaseModel):
    """A single unit of work for the collection pipeline."""
    dataset_item: DatasetItem
    model_id: str
    prompt_variation: PromptVariation
    run_index: int
    temperature: float | None


class CollectionResult(BaseModel):
    """Summary of a collection run."""
    experiment_id: str
    total_items: int
    successful: int
    failed: int
    skipped: int
    total_cost_usd: float
    total_tokens: int
    duration_seconds: float


class MetricResult(BaseModel):
    """Result of evaluating one metric on an experiment."""
    item_scores: list[ItemScore]
    aggregated_scores: list[AggregatedScore]


class EvaluationResult(BaseModel):
    """Full result of an evaluation pipeline run."""
    experiment_id: str
    metrics: dict[str, MetricResult]


class JudgeResponse(BaseModel):
    """Raw response from a single panel member — written to judge_outputs JSONL."""

    observation_id: str
    metric_name: str
    model_id: str
    generator_model_id: str
    prompt_variation: PromptVariation
    run_index: int
    raw_response: dict
    latency_ms: float
    timestamp: datetime
    ie_name: str
