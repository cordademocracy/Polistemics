"""Entry point for evaluation + analysis pipeline.

Usage:
    uv run python scripts/evaluate.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml
    uv run python scripts/evaluate.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml --metrics faithfulness
    uv run python scripts/evaluate.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml --filter-parties de_cdu,de_afd --force
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import structlog

import src.metrics  # noqa: F401 — trigger static registration
from src.analysis.tidy import build_tidy
from src.common import tracking
from src.common.config import load_experiment_config, load_model_configs
from src.common.settings import export_api_keys_to_env
from src.common.io import get_experiment_dir, load_ground_truth
from src.common.log import setup_logging
from src.evaluate.filter import OutputFilter
from src.evaluate.pipeline import EvaluationPipeline
from src.metrics.factory import build_metrics
from src.metrics.registry import METRIC_REGISTRY

logger = structlog.get_logger(__name__)


async def async_main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run evaluation + analysis on collected experiment data")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config or experiment ID (with --tidy-only)")
    parser.add_argument("--tidy-only", action="store_true", help="Rebuild tidy scores table for an already-scored experiment; --config accepts an experiment ID directly")
    parser.add_argument("--models", default="configs/models.yaml", help="Path to models YAML")
    parser.add_argument("--metrics", help="Override metrics (comma-separated)")
    parser.add_argument("--filter-parties", help="Filter by party IDs (comma-separated)")
    parser.add_argument("--filter-models", help="Filter by model IDs (comma-separated)")
    parser.add_argument("--filter-statements", help="Filter by statement IDs (comma-separated)")
    parser.add_argument("--filter-status", help="Filter by status (comma-separated)")
    parser.add_argument("--force", action="store_true", help="Recompute all metrics")
    parser.add_argument("--skip-analysis", action="store_true", help="Only run evaluation, skip analysis + report")
    args = parser.parse_args()

    if args.tidy_only:
        setup_logging()
        experiment_dir = get_experiment_dir(args.config)
        if not experiment_dir.exists():
            logger.error("experiment_dir_not_found", path=str(experiment_dir))
            sys.exit(1)
        df = build_tidy(experiment_dir)
        tidy_path = experiment_dir / "scores" / "tidy_scores.parquet"
        logger.info("tidy_built", path=str(tidy_path), n_rows=len(df))
        return

    # Load config
    config = load_experiment_config(args.config)
    logger.info("experiment_loaded", experiment_id=config.experiment_id)

    # Export API keys and load model configs (same as collection pipeline)
    export_api_keys_to_env()
    model_configs = load_model_configs(args.models)

    tracking.check_tracking()  # print banner if unavailable

    # Resolve experiment directory
    experiment_dir = get_experiment_dir(config.experiment_id)
    if not experiment_dir.exists():
        logger.error("experiment_dir_not_found", path=str(experiment_dir))
        logger.error("hint_run_collection_first", config=args.config)
        sys.exit(1)

    # Determine metrics
    raw_metrics = args.metrics.split(",") if args.metrics else config.metrics
    metrics = raw_metrics
    logger.info("metrics_resolved", metrics=metrics)

    # Build metric factories from GT data and register
    gt_path = experiment_dir / "ground_truth.jsonl"
    gt_items = load_ground_truth(gt_path)
    if not gt_items:
        logger.error("ground_truth_not_found", path=str(gt_path))
        sys.exit(1)

    # Create judge panel if configured
    panel = None
    if config.judges.panel:
        from src.evaluate.panel import JudgePanel

        panel = JudgePanel(
            model_ids=config.judges.panel,
            judge_model_configs=model_configs,
            temperature=config.judges.temperature,
            max_concurrent_per_model=config.judges.max_concurrent_per_model,
            output_dir=experiment_dir / "audit" / "judge",
            judge_timeout_seconds=config.judges.timeout_seconds,
            max_retries=config.judges.max_retries,
            retry_backoff_base=config.judges.retry_backoff_base,
            retry_backoff_max=config.judges.retry_backoff_max,
        )
        logger.info("judge_panel_created", n_models=len(config.judges.panel))

    # Build prompt builder for judge-based metrics
    prompt_builder = None
    if panel is not None:
        from src.collect.prompts import PromptBuilder

        base_dir = Path(__file__).parent.parent
        prompt_builder = PromptBuilder(config.prompts, base_dir / "configs" / "prompts")

    metric_factories = build_metrics(
        metrics,
        gt_items,
        panel=panel,
        prompt_builder=prompt_builder,
    )
    METRIC_REGISTRY.update(metric_factories)

    # Build output filter from CLI flags
    output_filter = None
    if any([args.filter_parties, args.filter_models, args.filter_statements, args.filter_status]):
        output_filter = OutputFilter(
            party_ids=args.filter_parties.split(",") if args.filter_parties else None,
            model_ids=args.filter_models.split(",") if args.filter_models else None,
            statement_ids=args.filter_statements.split(",") if args.filter_statements else None,
            statuses=args.filter_status.split(",") if args.filter_status else None,
        )
        logger.info("output_filter_applied", filter=output_filter.model_dump(exclude_none=True))

    run_name = f"{config.experiment_id}_evaluate"
    tags = {
        "experiment_id": config.experiment_id,
        "pass": "evaluate",
        "metrics": ",".join(metrics),
        "election_ids": ",".join(config.dataset.election_ids),
    }

    start = time.monotonic()
    with tracking.start_run(run_name=run_name, tags=tags):
        tracking.log_params({"n_metrics": len(metrics), "force": str(args.force)})

        # Run evaluation
        pipeline = EvaluationPipeline(experiment_dir, judge_model_configs=model_configs, experiment_config=config)
        result = await pipeline.run(
            metric_names=metrics,
            output_filter=output_filter,
            force=args.force,
        )
        duration = time.monotonic() - start

        n_items_total = sum(len(mr.aggregated_scores) for mr in result.metrics.values())
        tracking.log_metrics({
            "n_items_total": float(n_items_total),
            "duration_s": duration,
        })
        tracking.log_artifact(Path(args.config))

        # Print evaluation summary
        print(json.dumps({
            "experiment_id": result.experiment_id,
            "metrics": {
                name: {
                    "item_scores": len(mr.item_scores),
                    "aggregated_scores": len(mr.aggregated_scores),
                }
                for name, mr in result.metrics.items()
            },
        }, indent=2))

        # Materialize the tidy scores table for downstream analysis notebooks.
        if not args.skip_analysis:
            try:
                tidy_df = build_tidy(experiment_dir)
                tidy_path = experiment_dir / "scores" / "tidy_scores.parquet"
                logger.info("tidy_built", path=str(tidy_path), n_rows=len(tidy_df))
            except Exception as exc:  # noqa: BLE001 — scoring already succeeded; never crash on analysis.
                logger.error("tidy_build_failed", error=str(exc))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
