from __future__ import annotations

import asyncio
import random
import time
from datetime import UTC, datetime
from pathlib import Path
import structlog
from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.models import Model

from src.collect.cost_tracker import CostTracker
from src.common.config_hash import compute_collection_hash, read_lock_file, write_lock_file
from src.common.model_factory import create_model_pool
from src.collect.prompts import PromptBuilder
from src.collect.refusal import RefusalDetector
from src.common.config import ExperimentConfig, ModelConfig
from src.common.io import append_output_to_path, load_outputs_from_path, save_ground_truth
from src.common.model_call import build_model_settings
from src.common.progress import ProgressTracker
from src.common.settings import export_api_keys_to_env
from src.common.schemas import (
    CollectionResult,
    DatasetItem,
    LLMOutput,
    OutputStatus,
    WorkItem,
)

logger = structlog.get_logger(__name__)

# Rate-limit backoff parameters (longer than transient-error backoff)
_RATE_LIMIT_BACKOFF_BASE: float = 10.0
_RATE_LIMIT_BACKOFF_MAX: float = 120.0


class CollectionPipeline:
    """Async orchestrator for LLM data collection."""

    def __init__(
        self,
        config: ExperimentConfig,
        model_configs: dict[str, ModelConfig],
        data_dir: Path | None = None,
        templates_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._model_configs = model_configs
        self._base_dir = Path(__file__).parent.parent.parent
        self._data_dir = data_dir or (self._base_dir / "data")
        self._experiment_dir = self._data_dir / "experiments" / config.experiment_id
        self._templates_dir = templates_dir or (self._base_dir / "configs" / "prompts")
        self._prompt_builder: PromptBuilder | None = None
        self._cost_tracker: CostTracker | None = None
        self._refusal_detector = RefusalDetector()
        self._agent_pools: dict[str, list[Agent[None, str]]] = {}
        self._pool_index: dict[str, int] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._collection_hashes: dict[str, str] = {}

    async def run(self, dataset: list[DatasetItem] | None = None, force: bool = False) -> CollectionResult:
        """Main entry point. Runs all combinations, returns summary."""
        start = time.monotonic()
        export_api_keys_to_env()

        cond = self._config.conditions
        self._prompt_builder = PromptBuilder(
            self._config.prompts, self._templates_dir, cond.response_language, cond.party_label,
        )
        # Compute per-model fingerprints — prompt builder must be initialized first so
        # template files are accessible via self._templates_dir.
        self._collection_hashes = {
            model_id: compute_collection_hash(
                model_id=model_id,
                model_config=self._model_configs[model_id],
                prompts_config=self._config.prompts,
                templates_dir=self._templates_dir,
                conditions=self._config.conditions,
            )
            for model_id in self._config.model_ids
        }

        self._cost_tracker = CostTracker(
            self._config.experiment_id,
            self._experiment_dir,
        )

        concurrency = self._config.collection.concurrency
        for model_id in self._config.model_ids:
            limit = concurrency.per_model.get(model_id, concurrency.default)
            self._semaphores[model_id] = asyncio.Semaphore(limit)

        for model_id in self._config.model_ids:
            model_config = self._model_configs[model_id]
            models = create_model_pool(model_config)
            # System prompt is injected per-item at request time via agent.override()
            self._agent_pools[model_id] = [
                self._build_agent(model=m)
                for m in models
            ]
            self._pool_index[model_id] = 0
            if model_config.web_search is not None and model_config.web_search.enabled:
                logger.info(
                    "web_search_enabled",
                    model_id=model_id,
                    search_context_size=model_config.web_search.search_context_size,
                    max_results=model_config.web_search.max_results,
                )
            logger.info("model_initialized", model_id=model_id, api_keys=len(models))

        # Ensure experiment dir exists before writing lock files.
        self._experiment_dir.mkdir(parents=True, exist_ok=True)

        # Check / write per-model lock files to detect config drift on resumption.
        for model_id in self._config.model_ids:
            lock_path = self._experiment_dir / f"collection_config_{model_id}.json"
            existing = read_lock_file(lock_path)

            if existing is None:
                # First run for this model — write lock and proceed.
                write_lock_file(lock_path, {
                    "model_id": model_id,
                    "collection_hash": self._collection_hashes[model_id],
                    "timestamp": datetime.now(UTC).isoformat(),
                })
            elif existing["collection_hash"] != self._collection_hashes[model_id]:
                if not force:
                    raise RuntimeError(
                        f"collection_config_mismatch for model_id={model_id}: "
                        f"stored={existing['collection_hash']}, "
                        f"current={self._collection_hashes[model_id]}. "
                        f"Pass force=True to override."
                    )
                else:
                    logger.warning(
                        "collection_config_mismatch_forced",
                        model_id=model_id,
                        stored_hash=existing["collection_hash"],
                        current_hash=self._collection_hashes[model_id],
                    )
                    # Overwrite lock file with new hash.
                    write_lock_file(lock_path, {
                        "model_id": model_id,
                        "collection_hash": self._collection_hashes[model_id],
                        "timestamp": datetime.now(UTC).isoformat(),
                        "forced": True,
                    })

        if dataset is None:
            from src.common.adapters import BenchmarkAdapter
            adapter = BenchmarkAdapter(Path(self._config.dataset.path), self._config.conditions)
            dataset = adapter.load(self._config.dataset)

        save_ground_truth(dataset, self._experiment_dir / "ground_truth.jsonl")

        all_items = self._generate_work_items(dataset)
        completed = self._load_completed()
        remaining = [
            w for w in all_items
            if self._work_item_key(w, self._collection_hashes.get(w.model_id)) not in completed
        ]
        skipped = len(all_items) - len(remaining)

        logger.info(
            "collection_start",
            remaining=len(remaining),
            skipped=skipped,
            total=len(all_items),
        )

        async with ProgressTracker(len(remaining), "Collecting LLM outputs") as tracker:
            async def _tracked(w: WorkItem) -> LLMOutput:
                try:
                    return await self._collect_single(w)
                finally:
                    tracker.advance(1, model=w.model_id)

            results = await asyncio.gather(
                *[_tracked(w) for w in remaining],
                return_exceptions=True,
            )

        successful = sum(
            1 for r in results
            if isinstance(r, LLMOutput) and r.status == OutputStatus.SUCCESS
        )
        failed = len(remaining) - successful
        total_cost = sum(
            r.cost_usd for r in results
            if isinstance(r, LLMOutput) and r.cost_usd is not None
        )
        total_tokens = sum(
            r.tokens_input + r.tokens_output for r in results
            if isinstance(r, LLMOutput)
        )

        return CollectionResult(
            experiment_id=self._config.experiment_id,
            total_items=len(all_items),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            duration_seconds=time.monotonic() - start,
        )

    def _generate_work_items(self, dataset: list[DatasetItem]) -> list[WorkItem]:
        items = []
        for model_id in self._config.model_ids:
            for item in dataset:
                for variation in self._config.prompts.levels:
                    for run_config in self._config.runs:
                        for run_idx in range(run_config.k):
                            items.append(WorkItem(
                                dataset_item=item,
                                model_id=model_id,
                                prompt_variation=variation,
                                run_index=run_idx,
                                temperature=run_config.temperature,
                            ))
        return items

    def _load_completed(self) -> set[tuple]:
        outputs = load_outputs_from_path(self._experiment_dir / "outputs.jsonl")
        return {
            (o.observation_id, o.ie_name, o.model_id, o.prompt_variation, o.run_index, o.temperature, o.collection_hash)
            for o in outputs
        }

    @staticmethod
    def _work_item_key(w: WorkItem, collection_hash: str | None) -> tuple:
        return (
            w.dataset_item.observation_id,
            w.dataset_item.ie_name,
            w.model_id,
            w.prompt_variation,
            w.run_index,
            w.temperature,
            collection_hash,
        )

    async def _collect_single(self, work_item: WorkItem) -> LLMOutput:
        """Single LLM call with retry + error handling."""
        item = work_item.dataset_item
        semaphore = self._semaphores[work_item.model_id]
        mc = self._model_configs[work_item.model_id]

        async with semaphore:
            if self._prompt_builder is None:
                raise RuntimeError("PromptBuilder is not initialized")
            system_prompt = self._prompt_builder.build_system_prompt(item)
            user_prompt = self._prompt_builder.build_user_prompt(
                item, work_item.prompt_variation,
            )

            start = time.monotonic()
            max_retries = self._config.collection.max_retries
            backoff_base = self._config.collection.retry_backoff_base
            backoff_max = self._config.collection.retry_backoff_max

            last_error: Exception | None = None
            pool = self._agent_pools[work_item.model_id]
            for attempt in range(max_retries + 1):
                agent = pool[self._pool_index[work_item.model_id]]
                try:
                    settings = build_model_settings(mc)
                    if work_item.temperature is not None:
                        settings["temperature"] = work_item.temperature
                    # Web search is handled at model level via
                    # WebSearchOpenRouterModel (modern server tools API)
                    with agent.override(instructions=system_prompt):
                        result = await agent.run(
                            user_prompt,
                            model_settings=settings,
                        )
                    elapsed_ms = (time.monotonic() - start) * 1000
                    usage = result.usage()

                    output = LLMOutput(
                        observation_id=item.observation_id,
                        statement_id=item.statement_id,
                        party_id=item.party_id,
                        experiment_id=self._config.experiment_id,
                        model_id=work_item.model_id,
                        prompt_variation=work_item.prompt_variation,
                        run_index=work_item.run_index,
                        temperature=work_item.temperature,
                        predicted_stance=None,
                        predicted_explanation=result.output,
                        timestamp=datetime.now(UTC),
                        latency_ms=elapsed_ms,
                        tokens_input=usage.input_tokens or 0,
                        tokens_output=usage.output_tokens or 0,
                        cost_usd=None,
                        status=OutputStatus.SUCCESS,
                        error_message=None,
                        refusal_type=self._refusal_detector.detect(result.output),
                        ie_name=work_item.dataset_item.ie_name,
                        condition_id=self._build_condition_id(work_item.dataset_item.ie_name),
                        collection_hash=self._collection_hashes.get(work_item.model_id),
                    )
                    self._persist(output, work_item, usage)
                    return output

                except (TimeoutError, ConnectionError, OSError) as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = min(
                            backoff_base ** attempt + random.random(),
                            backoff_max,
                        )
                        logger.warning(
                            "retryable_error",
                            observation_id=item.observation_id,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                    continue

                except ModelHTTPError as e:
                    if e.status_code != 429:
                        last_error = e
                        break
                    # Rate-limited: rotate API key if pool has alternatives
                    last_error = e
                    if len(pool) > 1:
                        self._pool_index[work_item.model_id] = (
                            self._pool_index[work_item.model_id] + 1
                        ) % len(pool)
                        logger.info(
                            "rotated_api_key",
                            model_id=work_item.model_id,
                            key_index=self._pool_index[work_item.model_id],
                        )
                    if attempt < max_retries:
                        # Prefer the Retry-After header from the upstream response;
                        # fall back to exponential backoff when the header is absent
                        # or unparseable.
                        retry_after: float | None = None
                        try:
                            cause = e.__cause__
                            if cause is not None and hasattr(cause, "response"):
                                raw = cause.response.headers.get("Retry-After")
                                if raw is not None:
                                    retry_after = float(raw)
                        except (AttributeError, TypeError, ValueError):
                            retry_after = None

                        if retry_after is not None:
                            delay = min(retry_after, _RATE_LIMIT_BACKOFF_MAX)
                            delay_source = "retry_after_header"
                        else:
                            delay = min(
                                _RATE_LIMIT_BACKOFF_BASE * (backoff_base ** attempt)
                                + random.random(),
                                _RATE_LIMIT_BACKOFF_MAX,
                            )
                            delay_source = "exponential_backoff"

                        logger.warning(
                            "rate_limited",
                            observation_id=item.observation_id,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait_seconds=round(delay, 1),
                            delay_source=delay_source,
                        )
                        await asyncio.sleep(delay)
                    continue

                except Exception as e:
                    last_error = e
                    break

            elapsed_ms = (time.monotonic() - start) * 1000
            output = LLMOutput(
                observation_id=item.observation_id,
                statement_id=item.statement_id,
                party_id=item.party_id,
                experiment_id=self._config.experiment_id,
                model_id=work_item.model_id,
                prompt_variation=work_item.prompt_variation,
                run_index=work_item.run_index,
                temperature=work_item.temperature,
                predicted_stance=None,
                predicted_explanation="",
                timestamp=datetime.now(UTC),
                latency_ms=elapsed_ms,
                tokens_input=0,
                tokens_output=0,
                cost_usd=None,
                status=OutputStatus.API_ERROR,
                error_message=str(last_error),
                refusal_type=None,
                ie_name=work_item.dataset_item.ie_name,
                condition_id=self._build_condition_id(work_item.dataset_item.ie_name),
                collection_hash=self._collection_hashes.get(work_item.model_id),
            )
            self._persist(output, work_item)
            return output

    def _build_condition_id(self, ie_name: str) -> str:
        """Build denormalized condition identifier for this IE.

        Format: {ie_name}__{party_label}__{response_language}__{year_mention}
        """
        c = self._config.conditions
        return f"{ie_name}__{c.party_label}__{c.response_language}__{c.year_mention}"

    def _persist(self, output: LLMOutput, work_item: WorkItem, usage=None) -> None:
        append_output_to_path(output, self._experiment_dir / "outputs.jsonl")
        tokens_in = usage.input_tokens if usage and usage.input_tokens else 0
        tokens_out = usage.output_tokens if usage and usage.output_tokens else 0
        if self._cost_tracker is None:
            raise RuntimeError("CostTracker is not initialized")
        self._cost_tracker.log(
            work_item.model_id, tokens_in, tokens_out,
            output.cost_usd, output.latency_ms,
        )

    def _build_agent(self, model: Model) -> Agent[None, str]:
        # System prompt is injected per-item at request time via agent.override()
        return Agent(model, output_type=str)
