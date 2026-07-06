from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.schemas import AggregatedScore, DatasetItem, ItemScore, LLMOutput


DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Experiment directory helpers
# ---------------------------------------------------------------------------


def get_experiment_dir(
    experiment_id: str,
    base_dir: Path = DEFAULT_DATA_DIR / "experiments",
) -> Path:
    """Resolve experiment directory path."""
    return base_dir / experiment_id


# ---------------------------------------------------------------------------
# Ground truth sidecar I/O
# ---------------------------------------------------------------------------


def save_ground_truth(items: list[DatasetItem], path: Path) -> None:
    """Write DatasetItem list to JSONL (ground truth sidecar)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def load_ground_truth(path: Path) -> list[DatasetItem]:
    """Read ground truth sidecar JSONL → list of DatasetItem."""
    if not path.exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(DatasetItem.model_validate_json(line))
    return items


# ---------------------------------------------------------------------------
# Explicit-path LLMOutput I/O
# ---------------------------------------------------------------------------


def load_outputs_from_path(path: Path) -> list[LLMOutput]:
    """Load LLMOutput records from an explicit JSONL path."""
    if not path.exists():
        return []
    outputs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(LLMOutput.model_validate_json(line))
    return outputs


def append_output_to_path(output: LLMOutput, path: Path) -> None:
    """Append single LLMOutput to explicit JSONL path (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        # No explicit flush needed — the `with` block guarantees close+flush on exit.
        f.write(output.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Score I/O
# ---------------------------------------------------------------------------


def save_item_scores(scores: list[ItemScore], path: Path) -> None:
    """Write ItemScore list to JSONL (overwrites existing file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for score in scores:
            f.write(score.model_dump_json() + "\n")


def append_item_score(score: ItemScore, path: Path) -> None:
    """Append a single ItemScore to JSONL (creates file if needed).

    Used for incremental saving during evaluation — each scored item
    is persisted immediately so progress survives crashes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(score.model_dump_json() + "\n")


def save_aggregated_scores(scores: list[AggregatedScore], path: Path) -> None:
    """Write AggregatedScore list to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for score in scores:
            f.write(score.model_dump_json() + "\n")


def load_aggregated_scores(path: Path) -> list[AggregatedScore]:
    """Read AggregatedScore JSONL."""
    if not path.exists():
        return []
    scores = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(AggregatedScore.model_validate_json(line))
    return scores


def load_item_scores(path: Path) -> list[ItemScore]:
    """Read ItemScore JSONL."""
    if not path.exists():
        return []
    scores = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(ItemScore.model_validate_json(line))
    return scores
