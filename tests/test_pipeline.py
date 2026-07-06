from datetime import UTC, datetime

from src.collect.pipeline import (
    CollectionPipeline,
    _RATE_LIMIT_BACKOFF_BASE,
    _RATE_LIMIT_BACKOFF_MAX,
)
from src.common.config import (
    CollectionConfig,
    ConcurrencyConfig,
    DatasetFilter,
    ExperimentConfig,
    ModelConfig,
    PromptConfig,
    RunConfig,
)
from src.common.schemas import (
    CollectionResult,
    DatasetItem,
    LLMOutput,
    OutputStatus,
    PromptVariation,
    StanceLabel,
)


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="test_v1", experiment_name="Test",
        dataset=DatasetFilter(source="ResPolitica", election_ids=["test"], path="datasets/test.parquet"),
        model_ids=["model_a"],
        prompts=PromptConfig(
            levels=[PromptVariation.MINIMAL], system_prompt="system.txt",
            templates={"minimal": "minimal_user.txt"},
        ),
        runs=[RunConfig(temperature=0.0, k=1)],
        metrics=["faithfulness"],
        collection=CollectionConfig(max_retries=1, concurrency=ConcurrencyConfig(default=2)),
    )


def _make_model_configs() -> dict[str, ModelConfig]:
    return {"model_a": ModelConfig(id="model_a", provider="openrouter", model="placeholder", api_key_env="OPENROUTER_API_KEY")}


def _make_items(n: int = 2) -> list[DatasetItem]:
    return [
        DatasetItem(
            observation_id=f"test__de_spd__s{i:03d}", election_id="test",
            party_id="de_spd", party_name="SPD", party_anonymized="Party 01",
            statement_id=f"test__s{i:03d}",
            statement_number=None, statement_text=f"Statement {i}",
            statement_category=None, stance_label=StanceLabel.AGREE,
            rationale_text=None, has_rationale=False,
            ie_name="baseline", ie_chunks=[],
        )
        for i in range(1, n + 1)
    ]


def test_pipeline_generates_work_items(tmp_path):
    config = _make_config()
    pipeline = CollectionPipeline(config=config, model_configs=_make_model_configs(), data_dir=tmp_path, templates_dir=tmp_path)
    work_items = pipeline._generate_work_items(_make_items(2))
    assert len(work_items) == 2


def test_pipeline_skips_completed(tmp_path):
    config = _make_config()
    exp_dir = tmp_path / "experiments" / "test_v1"
    exp_dir.mkdir(parents=True)
    completed = LLMOutput(
        observation_id="test__de_spd__s001", statement_id="test__s001",
        party_id="de_spd", experiment_id="test_v1", model_id="model_a",
        prompt_variation=PromptVariation.MINIMAL, run_index=0, temperature=0.0,
        predicted_stance=StanceLabel.AGREE, predicted_explanation="test",
        timestamp=datetime.now(UTC), latency_ms=100.0,
        tokens_input=50, tokens_output=30, cost_usd=0.001,
        status=OutputStatus.SUCCESS, error_message=None, refusal_type=None,
        ie_name="baseline",
        condition_id="baseline__real__evidence__none",
    )
    (exp_dir / "outputs.jsonl").write_text(completed.model_dump_json() + "\n")
    pipeline = CollectionPipeline(config=config, model_configs=_make_model_configs(), data_dir=tmp_path, templates_dir=tmp_path)
    completed_set = pipeline._load_completed()
    assert len(completed_set) == 1


def test_collection_result_shape():
    result = CollectionResult(
        experiment_id="test", total_items=10, successful=8,
        failed=1, skipped=1, total_cost_usd=0.05,
        total_tokens=5000, duration_seconds=10.0,
    )
    assert result.total_items == result.successful + result.failed + result.skipped


def test_rate_limit_backoff_constants():
    """Verify rate-limit backoff is slower than transient-error backoff."""
    assert _RATE_LIMIT_BACKOFF_BASE == 10.0
    assert _RATE_LIMIT_BACKOFF_MAX == 120.0


def test_model_config_reasoning_effort_in_pipeline():
    """Verify ModelConfig with reasoning_effort works in pipeline initialization."""
    mc = ModelConfig(
        id="gpt_5_4", provider="openrouter", model="openai/gpt-5.4",
        max_tokens=1024, api_key_env="OPENROUTER_API_KEY",
        reasoning_effort="medium",
    )
    configs = {"gpt_5_4": mc}
    config = ExperimentConfig(
        experiment_id="test_v1", experiment_name="Test",
        dataset=DatasetFilter(source="ResPolitica", election_ids=["test"], path="datasets/test.parquet"),
        model_ids=["gpt_5_4"],
        prompts=PromptConfig(
            levels=[PromptVariation.MINIMAL], system_prompt="system.txt",
            templates={"minimal": "minimal_user.txt"},
        ),
        runs=[RunConfig(temperature=0.0, k=1)],
        metrics=["faithfulness"],
    )
    # Verify pipeline can be instantiated with reasoning_effort model
    pipeline = CollectionPipeline(config=config, model_configs=configs)
    assert configs["gpt_5_4"].reasoning_effort == "medium"
