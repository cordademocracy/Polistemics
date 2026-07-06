from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class CostTracker:
    """Appends per-request cost entries to a JSONL file."""

    def __init__(self, experiment_id: str, experiment_dir: Path) -> None:
        self._experiment_id = experiment_id
        self._experiment_dir = experiment_dir
        # Create directory once at construction, not on every log() call.
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._experiment_dir / "costs.jsonl"

    def log(
        self,
        model_id: str,
        tokens_input: int,
        tokens_output: int,
        cost_usd: float | None,
        latency_ms: float,
    ) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "experiment_id": self._experiment_id,
            "model_id": model_id,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }
        with open(self._path, "a") as f:
            # No explicit flush needed — the `with` block guarantees close+flush on exit.
            f.write(json.dumps(entry) + "\n")
