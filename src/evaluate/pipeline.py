from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.common.io import (
    append_item_score,
    load_aggregated_scores,
    load_ground_truth,
    load_outputs_from_path,
    save_aggregated_scores,
    save_item_scores,
)
from src.common.progress import ProgressTracker
from src.common.schemas import (
    AggregatedScore,
    DatasetItem,
    EvaluationResult,
    ItemScore,
    LLMOutput,
    MetricResult,
    OutputStatus,
)
from src.evaluate.filter import OutputFilter, filter_outputs
from src.evaluate.panel import PartialPanelResult
from src.metrics.base import BaseMetric
from src.metrics.registry import METRIC_REGISTRY
from src.metrics.rubric import BaseRubric

if TYPE_CHECKING:
    from src.common.config import ExperimentConfig, ModelConfig

logger = structlog.get_logger(__name__)

# Max number of concurrent metric.score() coroutines per metric pass.
_MAX_CONCURRENT_SCORES = 10


def _append_incomplete(record: dict, path: Path) -> None:
    """Append one record to incomplete.jsonl (crash-safe)."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _prune_incomplete(path: Path, resolved_keys: set[tuple[str, str, str, str, int, float | None]]) -> None:
    """Remove resolved entries from incomplete.jsonl. Delete file if empty."""
    if not path.exists():
        return
    remaining = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["observation_id"], r["ie_name"], r["model_id"],
                   r["prompt_variation"], r["run_index"], r["temperature"])
            if key not in resolved_keys:
                remaining.append(line)
    if not remaining:
        path.unlink()
    else:
        with path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(remaining) + "\n")


def _load_incomplete_keys(path: Path, metric_name: str) -> set[tuple]:
    """Load the score-key tuples of all entries currently in incomplete.jsonl.

    Used to route prior-incomplete items directly to the retry pass on restart,
    preventing them from entering the main eligible set and being scored twice.
    """
    if not path.exists():
        return set()
    keys: set[tuple] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("metric_name") != metric_name:
                continue
            keys.add((
                r["observation_id"],
                r["ie_name"],
                r["model_id"],
                r["prompt_variation"],
                r["run_index"],
                r["temperature"],
                metric_name,
                None,  # judge_hash — legacy incomplete entries don't carry it
            ))
    return keys


def _load_retry_eligible(
    incomplete_path: Path,
    metric_name: str,
    outputs: list[LLMOutput],
    gt_lookup: dict[tuple[str, str], DatasetItem],
) -> list[tuple[LLMOutput, DatasetItem, list]]:
    """Load outputs matching incomplete.jsonl entries for a given metric.

    Returns tuples of (output, gt_item, prior_responses) where prior_responses
    are the already-succeeded SingleJudgeResponse objects stored in incomplete.jsonl,
    ready to be passed to dispatch() to skip re-calling those judges.

    Args:
        incomplete_path: Path to the incomplete.jsonl file.
        metric_name: The metric name to filter on.
        outputs: All LLMOutput objects from the experiment.
        gt_lookup: Mapping of (observation_id, ie_name) to DatasetItem.

    Returns:
        List of (output, gt_item, prior_responses) tuples eligible for a retry pass.
    """
    if not incomplete_path.exists():
        return []

    # Map from output key → prior partial responses stored in incomplete.jsonl
    incomplete_data: dict[tuple, list] = {}
    with incomplete_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("metric_name") != metric_name:
                continue
            key = (r["observation_id"], r["ie_name"], r["model_id"], r["prompt_variation"], r["run_index"], r["temperature"])
            incomplete_data[key] = r.get("partial_responses", [])

    if not incomplete_data:
        return []

    from src.evaluate.panel import SingleJudgeResponse

    result = []
    for out in outputs:
        key = (out.observation_id, out.ie_name, out.model_id, out.prompt_variation.value, out.run_index, out.temperature)
        if key not in incomplete_data:
            continue
        if out.status != OutputStatus.SUCCESS:
            continue
        if (out.observation_id, out.ie_name) not in gt_lookup:
            continue

        # Reconstruct prior SingleJudgeResponse objects from stored dicts.
        # result is stored as a plain dict (via model_dump()); deserialise it as
        # a SimpleNamespace so getattr(r.result, "q1") works in the retry pass
        # without triggering Pydantic's __private_attributes__ introspection on
        # a dynamically-created verdict model that no longer exists in memory.
        prior: list = []
        for rd in incomplete_data[key]:
            try:
                raw = dict(rd)
                if isinstance(raw.get("result"), dict):
                    from types import SimpleNamespace
                    raw["result"] = SimpleNamespace(**raw["result"])
                prior.append(SingleJudgeResponse.model_validate(raw))
            except Exception:
                pass  # malformed stored response — skip, judge will be re-called

        result.append((out, gt_lookup[(out.observation_id, out.ie_name)], prior))

    return result


def _clear_incomplete_for_obs(path: Path, observation_ids: set[str]) -> None:
    """Remove all incomplete entries for the given observation IDs.

    Args:
        path: Path to the incomplete.jsonl file.
        observation_ids: Set of observation IDs whose entries should be removed.
    """
    if not path.exists():
        return

    remaining = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["observation_id"] not in observation_ids:
                remaining.append(line)

    if not remaining:
        path.unlink()
    else:
        with path.open("w") as fh:
            fh.write("\n".join(remaining) + "\n")


class EvaluationPipeline:
    """Orchestrates metric computation on collected experiment data.

    Loads outputs + ground truth from experiment directory, runs selected
    metrics, aggregates K runs, and persists scores per metric.

    Supports resumability: skips metrics with existing score files
    unless force=True.

    When experiment_config is provided, factory-built metrics are created
    via build_metrics() with the appropriate JudgePanel.
    Metrics already in METRIC_REGISTRY are used as-is.
    """

    def __init__(
        self,
        experiment_dir: Path,
        experiment_config: ExperimentConfig | None = None,
        templates_dir: Path | None = None,
        judge_model_configs: dict[str, ModelConfig] | None = None,
    ) -> None:
        self._experiment_dir = experiment_dir
        self._experiment_config = experiment_config
        self._base_dir = Path(__file__).parent.parent.parent
        self._templates_dir = templates_dir or (self._base_dir / "configs" / "prompts")
        self._judge_model_configs = judge_model_configs or {}

    async def run(
        self,
        metric_names: list[str],
        output_filter: OutputFilter | None = None,
        force: bool = False,
    ) -> EvaluationResult:
        """Run evaluation pipeline on collected experiment data."""
        from src.common.settings import export_api_keys_to_env
        export_api_keys_to_env()

        # 1. Load data
        outputs = load_outputs_from_path(self._experiment_dir / "outputs.jsonl")
        ground_truth = load_ground_truth(self._experiment_dir / "ground_truth.jsonl")

        if not outputs:
            logger.warning("no_outputs_found", path=str(self._experiment_dir))

        # Filter outputs to only those matching the current collection_hash(es).
        # None-hash outputs (legacy, pre-config-hash) are always included.
        from src.common.config_hash import read_lock_file as _read_lock_file
        _collection_hashes: set[str | None] = set()
        for model_id in {o.model_id for o in outputs}:
            lock_path = self._experiment_dir / f"collection_config_{model_id}.json"
            lock = _read_lock_file(lock_path)
            _collection_hashes.add(lock["collection_hash"] if lock else None)
        outputs = [
            o for o in outputs
            if o.collection_hash in _collection_hashes or o.collection_hash is None
        ]

        # 2. Filter
        if output_filter:
            before = len(outputs)
            outputs = filter_outputs(outputs, output_filter)
            logger.info("outputs_filtered", before=before, after=len(outputs))

        # 3. Build GT lookup — keyed by (observation_id, ie_name) to handle multi-IE datasets
        gt_lookup: dict[tuple[str, str], DatasetItem] = {
            (item.observation_id, item.ie_name): item for item in ground_truth
        }

        # 4. Build factory metrics (IFD) from experiment config
        factory_metrics = self._build_factory_metrics(metric_names, ground_truth)

        # 5. Run each metric
        results: dict[str, MetricResult] = {}
        for metric_name in metric_names:
            scores_dir = self._experiment_dir / "scores" / metric_name

            # Two-tier metric resolution: registry first, factory second.
            # Must happen before the skip check so _compute_judge_hash has a metric instance.
            metric = self._resolve_metric(metric_name, factory_metrics)
            if metric is None:
                continue

            judge_hash = self._compute_judge_hash(metric)

            # Resumability check: skip only when scores are complete, judge config unchanged,
            # AND every current success output already has a score entry. The last condition
            # catches the case where new models are added to an existing experiment — without
            # it, the metric is declared complete even though the new model has never been judged.
            if not force and self._scores_complete(scores_dir) and self._judge_config_matches(scores_dir, judge_hash):
                _item_scores_path = scores_dir / "item_scores.jsonl"
                _already_scored = self._load_scored_keys(_item_scores_path, metric_name)
                _has_unscored = any(
                    self._item_score_key(o, metric_name, judge_hash) not in _already_scored
                    for o in outputs
                    if o.status == OutputStatus.SUCCESS
                )
                if not _has_unscored:
                    logger.info(
                        "metric_scores_skipped",
                        metric=metric_name,
                        hint="use force=True to recompute",
                    )
                    results[metric_name] = self._load_existing(scores_dir)
                    continue
                logger.info(
                    "metric_has_unscored_outputs",
                    metric=metric_name,
                    hint="new outputs detected — scoring incrementally",
                )

            logger.info("metric_computing", metric=metric_name, n_outputs=len(outputs))

            # --- Incremental scoring with crash-safe persistence ---
            # Each scored item is appended to item_scores.jsonl immediately.
            # On restart, already-scored items are skipped (resumability).

            item_scores_path = scores_dir / "item_scores.jsonl"
            incomplete_path = scores_dir / "incomplete.jsonl"
            scores_dir.mkdir(parents=True, exist_ok=True)

            # Write lock file so future runs can detect judge config changes.
            from src.common.config_hash import write_lock_file as _write_lock_file
            _write_lock_file(
                scores_dir / "scoring_config.json",
                {
                    "metric_name": metric_name,
                    "judge_hash": judge_hash,
                    "panel": self._experiment_config.judges.panel if self._experiment_config else [],
                    "temperature": self._experiment_config.judges.temperature if self._experiment_config else None,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            # Load already-scored keys for resumability
            already_scored = self._load_scored_keys(item_scores_path, metric_name)
            if already_scored:
                logger.info(
                    "metric_resuming",
                    metric=metric_name,
                    n_already_scored=len(already_scored),
                )

            # Load prior incomplete keys so they go to the retry pass, not the main pass.
            # Without this, a restarted run scores them twice: once in eligible and once in
            # _load_retry_eligible, producing duplicate item_scores.
            prior_incomplete = _load_incomplete_keys(incomplete_path, metric_name)

            # Collect eligible (output, gt) pairs:
            #  - skip already-scored (resumability)
            #  - skip prior-incomplete (routed to retry pass instead)
            #  - skip items the rubric declares inactive for this IE (pre-filter mirrors
            #    the active_questions() gate inside score(), applied here to keep the
            #    progress counter and semaphore load accurate)
            #  - deduplicate by score_key (guard against duplicate outputs.jsonl entries)
            eligible: list[tuple] = []
            seen_eligible: set[tuple] = set()
            for output in outputs:
                if output.status != OutputStatus.SUCCESS:
                    continue
                gt = gt_lookup.get((output.observation_id, output.ie_name))
                if gt is None:
                    logger.warning("ground_truth_missing", observation_id=output.observation_id, ie_name=output.ie_name)
                    continue
                if isinstance(metric, BaseRubric) and not metric.active_questions(output.ie_name):
                    continue
                score_key = self._item_score_key(output, metric_name, judge_hash)
                if score_key in already_scored or score_key in prior_incomplete:
                    continue
                if score_key in seen_eligible:
                    continue
                seen_eligible.add(score_key)
                eligible.append((output, gt))

            # Score remaining outputs concurrently with bounded parallelism.
            # Each result is appended to JSONL immediately via asyncio.Lock.
            write_lock = asyncio.Lock()
            new_item_scores: list[ItemScore] = []

            def _make_score_one(
                sem_: asyncio.Semaphore,
                tracker_: ProgressTracker,
                judge_hash_: str | None,
                *,
                prior_responses_: list | None = None,
            ) -> Callable[[LLMOutput, DatasetItem], Coroutine[Any, Any, ItemScore | None]]:
                """Factory that binds sem, tracker, judge_hash, and prior_responses into a _score_one coroutine."""
                async def _score_one(out: LLMOutput, gt_item: DatasetItem) -> ItemScore | None:
                    async with sem_:
                        try:
                            from src.metrics.rubric import BaseRubric
                            if prior_responses_ is not None and isinstance(metric, BaseRubric):
                                result = await metric.score(out, gt_item, prior_responses=prior_responses_)
                            else:
                                result = await metric.score(out, gt_item)
                        except PartialPanelResult as exc:
                            # Store partial responses so retry can reuse succeeded judges.
                            partial_dicts = [
                                r.model_dump(mode="json") for r in exc.responses
                                if r.result is not None
                            ]
                            record = {
                                "observation_id": out.observation_id,
                                "ie_name": out.ie_name,
                                "model_id": out.model_id,
                                "prompt_variation": out.prompt_variation.value,
                                "run_index": out.run_index,
                                "temperature": out.temperature,
                                "metric_name": metric.name,
                                "n_succeeded": exc.n_succeeded,
                                "n_expected": exc.n_expected,
                                "partial_responses": partial_dicts,
                            }
                            async with write_lock:
                                _append_incomplete(record, incomplete_path)
                            return None
                        except Exception as exc:
                            logger.warning(
                                "metric_score_failed",
                                metric=metric_name,
                                observation_id=out.observation_id,
                                error=str(exc),
                            )
                            return None
                        finally:
                            tracker_.advance(1)

                        # Rubric not applicable for this (output, IE) pair — skip.
                        if result is None:
                            return None

                        item_score = ItemScore(
                            observation_id=out.observation_id,
                            party_id=out.party_id,
                            statement_id=out.statement_id,
                            model_id=out.model_id,
                            prompt_variation=out.prompt_variation,
                            run_index=out.run_index,
                            temperature=out.temperature,
                            metric_name=metric.name,
                            scores=result,
                            ie_name=out.ie_name,
                            judge_hash=judge_hash_,
                        )
                        # Persist immediately — survives crashes
                        async with write_lock:
                            append_item_score(item_score, item_scores_path)
                            new_item_scores.append(item_score)
                        return item_score
                return _score_one

            sem = asyncio.Semaphore(_MAX_CONCURRENT_SCORES)
            async with ProgressTracker(len(eligible), f"Scoring {metric_name}") as tracker:
                await asyncio.gather(
                    *[_make_score_one(sem, tracker, judge_hash)(out, gt_item) for out, gt_item in eligible],
                    return_exceptions=True,
                )

            # Second pass: one retry over incomplete observations (transient failures).
            # prior_responses carries already-succeeded judge results so only the
            # failed judge(s) are re-called, not the full panel.
            _retry_eligible = _load_retry_eligible(incomplete_path, metric_name, outputs, gt_lookup)
            if _retry_eligible:
                logger.info(
                    "second_pass_starting",
                    metric=metric_name,
                    n_incomplete=len(_retry_eligible),
                )
                # Clear incomplete entries for these obs so they can be re-scored clean
                _clear_incomplete_for_obs(incomplete_path, {out.observation_id for out, _, _pr in _retry_eligible})
                sem2 = asyncio.Semaphore(_MAX_CONCURRENT_SCORES)
                async with ProgressTracker(len(_retry_eligible), f"Retry {metric_name}") as tracker2:
                    await asyncio.gather(
                        *[
                            _make_score_one(sem2, tracker2, judge_hash, prior_responses_=prior)(out, gt_item)
                            for out, gt_item, prior in _retry_eligible
                        ],
                        return_exceptions=True,
                    )

            # After gather: clean up resolved incomplete entries
            all_scored_keys = {
                (s.observation_id, s.ie_name, s.model_id, s.prompt_variation.value, s.run_index, s.temperature)
                for s in self._load_all_item_scores(item_scores_path)
            }
            _prune_incomplete(incomplete_path, all_scored_keys)

            # Reload ALL item scores (existing + new) for aggregation
            all_item_scores = self._load_all_item_scores(item_scores_path)

            # Aggregate K runs
            aggregated = self._aggregate_run_groups(all_item_scores, metric)

            # Persist aggregated scores (overwrites — cheap, always recomputed)
            save_aggregated_scores(aggregated, scores_dir / "aggregated_scores.jsonl")
            results[metric_name] = MetricResult(
                item_scores=all_item_scores,
                aggregated_scores=aggregated,
            )
            logger.info(
                "metric_complete",
                metric=metric_name,
                n_item_scores=len(all_item_scores),
                n_new=len(new_item_scores),
                n_aggregated=len(aggregated),
            )

        return EvaluationResult(
            experiment_id=self._experiment_dir.name,
            metrics=results,
        )

    # --- Metric resolution helpers ---

    def _build_factory_metrics(
        self,
        metric_names: list[str],
        ground_truth: list[DatasetItem],
    ) -> dict[str, Callable[[], BaseMetric]]:
        """Build factory metrics from experiment config for names not in registry.

        Returns an empty dict if no experiment_config is available.
        """
        if self._experiment_config is None:
            return {}

        # Only build factories for metrics not already in the registry
        factory_names = [n for n in metric_names if n not in METRIC_REGISTRY]
        if not factory_names:
            return {}

        from src.common.prompts import PromptBuilder
        from src.evaluate.panel import JudgePanel
        from src.metrics.factory import build_metrics

        judges_cfg = self._experiment_config.judges
        panel: JudgePanel | None = None
        if judges_cfg.panel:
            panel = JudgePanel(
                model_ids=judges_cfg.panel,
                judge_model_configs=self._judge_model_configs,
                temperature=judges_cfg.temperature,
                max_concurrent_per_model=judges_cfg.max_concurrent_per_model,
                output_dir=self._experiment_dir / "audit" / "judge",
                judge_timeout_seconds=judges_cfg.timeout_seconds,
                max_retries=judges_cfg.max_retries,
                retry_backoff_base=judges_cfg.retry_backoff_base,
                retry_backoff_max=judges_cfg.retry_backoff_max,
            )

        prompt_builder = PromptBuilder(
            self._experiment_config.prompts, self._templates_dir,
        )

        return build_metrics(
            metric_names=factory_names,
            gt_items=ground_truth,
            panel=panel,
            prompt_builder=prompt_builder,
        )

    def _resolve_metric(
        self,
        metric_name: str,
        factory_metrics: dict[str, Callable[[], BaseMetric]],
    ) -> BaseMetric | None:
        """Resolve a metric by name: registry first, factory second.

        Returns None if the metric cannot be resolved (logged as error).
        """
        if metric_name in METRIC_REGISTRY:
            return METRIC_REGISTRY[metric_name]()

        if metric_name in factory_metrics:
            return factory_metrics[metric_name]()

        logger.error(
            "metric_not_resolved",
            metric=metric_name,
            hint="not in registry and no experiment_config provided for factory",
        )
        return None

    def _compute_judge_hash(self, metric: BaseMetric) -> str | None:
        """Compute judge hash for a single metric. Returns None if no judges configured.

        Only BaseRubric instances carry a rubric structure to fingerprint.
        All other metric types return None.

        Args:
            metric: Resolved metric instance.

        Returns:
            8-char hex hash, or None if metric is not a rubric or no panel is configured.
        """
        from src.metrics.rubric import BaseRubric
        if not self._experiment_config or not self._experiment_config.judges.panel:
            return None
        if not isinstance(metric, BaseRubric):
            return None
        from src.common.config_hash import compute_judge_hash
        return compute_judge_hash(
            judges_config=self._experiment_config.judges,
            judge_model_configs=self._judge_model_configs,
            metrics=[metric],
        )

    def _judge_config_matches(self, scores_dir: Path, judge_hash: str | None) -> bool:
        """Check stored judge hash against current. Returns True if match or legacy (no lock file).

        Args:
            scores_dir: The metric-level scores directory.
            judge_hash: Current judge hash (None if no judges configured).

        Returns:
            True if safe to reuse existing scores, False if config changed (blocks reuse).
        """
        from src.common.config_hash import read_lock_file
        lock_path = scores_dir / "scoring_config.json"
        existing = read_lock_file(lock_path)

        if existing is None:
            if (scores_dir / "item_scores.jsonl").exists():
                logger.warning(
                    "legacy_scores_no_config",
                    scores_dir=str(scores_dir),
                    hint="scores exist but no scoring_config.json — may be stale",
                )
            return True  # legacy: don't block

        stored_hash = existing.get("judge_hash")
        if stored_hash != judge_hash:
            logger.warning(
                "judge_config_mismatch",
                scores_dir=str(scores_dir),
                stored_hash=stored_hash,
                current_hash=judge_hash,
                hint="pass force=True to recompute",
            )
            return False
        return True

    # --- Resumability helpers ---

    @staticmethod
    def _item_score_key(output, metric_name: str, judge_hash: str | None = None) -> tuple:
        """Build a unique key for an item score to detect already-scored items."""
        return (
            output.observation_id,
            output.ie_name,
            output.model_id,
            output.prompt_variation,
            output.run_index,
            output.temperature,
            metric_name,
            judge_hash,
        )

    def _load_scored_keys(self, item_scores_path: Path, metric_name: str) -> set[tuple]:
        """Load already-scored keys from existing item_scores.jsonl."""
        if not item_scores_path.exists():
            return set()
        keys: set[tuple] = set()
        with open(item_scores_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                score = ItemScore.model_validate_json(line)
                keys.add((
                    score.observation_id,
                    score.ie_name,
                    score.model_id,
                    score.prompt_variation,
                    score.run_index,
                    score.temperature,
                    score.metric_name,
                    score.judge_hash,
                ))
        return keys

    def _load_all_item_scores(self, item_scores_path: Path) -> list[ItemScore]:
        """Load all item scores from JSONL for aggregation."""
        if not item_scores_path.exists():
            return []
        scores: list[ItemScore] = []
        with open(item_scores_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    scores.append(ItemScore.model_validate_json(line))
        return scores

    # --- Persistence helpers ---

    def _scores_complete(self, scores_dir: Path) -> bool:
        """Check whether scoring for a metric directory is fully complete.

        Args:
            scores_dir: The metric-level scores directory to inspect.

        Returns:
            True if item_scores.jsonl and aggregated_scores.jsonl both exist
            and incomplete.jsonl is either absent or empty.
        """
        if not (scores_dir / "item_scores.jsonl").exists():
            return False
        if not (scores_dir / "aggregated_scores.jsonl").exists():
            return False
        incomplete_path = scores_dir / "incomplete.jsonl"
        if incomplete_path.exists() and incomplete_path.stat().st_size > 0:
            return False
        return True

    def _load_existing(self, scores_dir: Path) -> MetricResult:
        """Load existing scores from disk."""
        item_path = scores_dir / "item_scores.jsonl"
        agg_path = scores_dir / "aggregated_scores.jsonl"

        item_scores: list[ItemScore] = []
        if item_path.exists():
            with open(item_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item_scores.append(ItemScore.model_validate_json(line))

        aggregated = load_aggregated_scores(agg_path)

        return MetricResult(item_scores=item_scores, aggregated_scores=aggregated)

    def _aggregate_run_groups(
        self,
        item_scores: list[ItemScore],
        metric: BaseMetric,
    ) -> list[AggregatedScore]:
        """Group item scores by run group and aggregate via the metric."""
        # Run group key: (observation_id, ie_name, model_id, prompt_variation, temperature)
        groups: dict[tuple, list[ItemScore]] = defaultdict(list)
        for score in item_scores:
            key = (score.observation_id, score.ie_name, score.model_id, score.prompt_variation, score.temperature)
            groups[key].append(score)

        aggregated: list[AggregatedScore] = []
        for (obs_id, ie_name, model_id, pv, temp), run_scores in groups.items():
            agg_dict = metric.aggregate([s.scores for s in run_scores])
            aggregated.append(
                AggregatedScore(
                    observation_id=obs_id,
                    party_id=run_scores[0].party_id,
                    statement_id=run_scores[0].statement_id,
                    model_id=model_id,
                    prompt_variation=pv,
                    temperature=temp,
                    metric_name=metric.name,
                    scores=agg_dict,
                    k_runs=len(run_scores),
                    aggregation_method=metric.aggregation_method,
                    ie_name=ie_name,
                )
            )
        return aggregated

    def _save_metric_scores(
        self,
        scores_dir: Path,
        item_scores: list[ItemScore],
        aggregated: list[AggregatedScore],
    ) -> None:
        """Persist scores to the metric's scores directory."""
        scores_dir.mkdir(parents=True, exist_ok=True)
        save_item_scores(item_scores, scores_dir / "item_scores.jsonl")
        save_aggregated_scores(aggregated, scores_dir / "aggregated_scores.jsonl")
