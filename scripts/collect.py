"""Run the Polistemics collection pipeline."""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from src.collect.pipeline import CollectionPipeline
from src.common import tracking
from src.common.config import load_experiment_config, load_model_configs
from src.common.log import setup_logging
from src.common.settings import validate_required_api_keys


async def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Polistemics collection pipeline")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--models", default="configs/models.yaml", help="Path to models YAML")
    args = parser.parse_args()

    tracking.check_tracking()  # print banner if unavailable

    config = load_experiment_config(args.config)
    model_configs = load_model_configs(args.models)
    validate_required_api_keys(model_configs)

    run_name = f"{config.experiment_id}_collect"
    tags = {
        "experiment_id": config.experiment_id,
        "pass": "collect",
        "election_ids": ",".join(config.dataset.election_ids),
        "ie_levels": str(config.conditions.ie_levels),
    }

    with tracking.start_run(run_name=run_name, tags=tags):
        tracking.log_params({
            "n_parties": len(config.dataset.party_filter or []),
            "n_statements": len(config.dataset.statement_filter or []),
            "n_models": len(config.model_ids),
        })

        start = time.monotonic()
        pipeline = CollectionPipeline(config, model_configs)
        result = await pipeline.run()
        duration = time.monotonic() - start

        tracking.log_metrics({
            "n_items_collected": float(result.successful),
            "n_failed": float(result.failed),
            "n_skipped": float(result.skipped),
            "total_cost_usd": result.total_cost_usd or 0.0,
            "duration_s": duration,
        })
        tracking.log_artifact(Path(args.config))

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
