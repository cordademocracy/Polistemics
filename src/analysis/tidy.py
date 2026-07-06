"""Format-aware tidy-table builder for experiment scores.

Reads each metric's post-vote ``aggregated_scores.jsonl``, melts the per-row
``scores`` dict into long form (one row per judged sub-question), maps fields to
the tidy schema in ``analysis/AGGREGATION.md`` §2, and materializes a single
``tidy_scores.parquet`` artifact. This is the only module here that does I/O —
the aggregation functions are pure DataFrame transforms.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from src.common.context_format import normalize_ie

# Default metrics in materialization order.
DEFAULT_METRICS: Final[list[str]] = [
    "faithfulness",
    "impartiality",
    "epistemic_calibration",
]

# Constant columns for the current DE / real / native run (future-proofed schema).
_LABEL_TYPE: Final[str] = "real"
_EVIDENCE_LANGUAGE: Final[str] = "native"

# Aggregate key dropped from the scores dict before melting (recomputed downstream).
_ADHERENCE_RATE_KEY: Final[str] = "adherence_rate"

# Tidy artifact filename, relative to the experiment's ``scores/`` dir.
_TIDY_FILENAME: Final[str] = "tidy_scores.parquet"

# Election prefix (first ``__`` segment of observation_id) -> ISO country code.
_COUNTRY_BY_ELECTION: Final[dict[str, str]] = {
    "bundestagswahl2025": "DE",
    "nl_tk2025": "NL",
}


class TidyColumns:
    """Tidy table column-name constants (avoid stringly-typed downstream bugs)."""

    MODEL: Final[str] = "model"
    COUNTRY: Final[str] = "country"
    IE: Final[str] = "ie"
    LABEL_TYPE: Final[str] = "label_type"
    EVIDENCE_LANGUAGE: Final[str] = "evidence_language"
    PARTY: Final[str] = "party"
    ITEM_ID: Final[str] = "item_id"
    OBSERVATION_ID: Final[str] = "observation_id"
    RUBRIC: Final[str] = "rubric"
    SUBQUESTION: Final[str] = "subquestion"
    PASSED: Final[str] = "passed"


# Final column order for the materialized frame.
_TIDY_COLUMN_ORDER: Final[list[str]] = [
    TidyColumns.MODEL,
    TidyColumns.COUNTRY,
    TidyColumns.IE,
    TidyColumns.LABEL_TYPE,
    TidyColumns.EVIDENCE_LANGUAGE,
    TidyColumns.PARTY,
    TidyColumns.ITEM_ID,
    TidyColumns.OBSERVATION_ID,
    TidyColumns.RUBRIC,
    TidyColumns.SUBQUESTION,
    TidyColumns.PASSED,
]


def _country_from_observation(observation_id: str) -> str:
    """Map the election prefix of an observation_id to its country code.

    Args:
        observation_id: e.g. ``"bundestagswahl2025__de_spd__s001"``.

    Returns:
        ISO country code (e.g. ``"DE"``).

    Raises:
        ValueError: If the election prefix is not in the country map.
    """
    election = observation_id.split("__")[0]
    country = _COUNTRY_BY_ELECTION.get(election)
    if country is None:
        raise ValueError(
            f"Unknown election prefix {election!r} in observation_id {observation_id!r}"
        )
    return country


def _melt_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Melt aggregated-score rows of one metric to long tidy rows.

    Args:
        rows: Parsed ``aggregated_scores.jsonl`` records for a single metric.

    Returns:
        One tidy row per (observation, active sub-question); ``adherence_rate``
        is excluded.
    """
    tidy_rows: list[dict[str, object]] = []
    for row in rows:
        observation_id = str(row["observation_id"])
        scores: dict[str, object] = row["scores"]  # type: ignore[assignment]
        base = {
            TidyColumns.MODEL: row["model_id"],
            TidyColumns.COUNTRY: _country_from_observation(observation_id),
            TidyColumns.IE: normalize_ie(str(row["ie_name"])),
            TidyColumns.LABEL_TYPE: _LABEL_TYPE,
            TidyColumns.EVIDENCE_LANGUAGE: _EVIDENCE_LANGUAGE,
            TidyColumns.PARTY: row["party_id"],
            TidyColumns.ITEM_ID: row["statement_id"],
            TidyColumns.OBSERVATION_ID: observation_id,
            TidyColumns.RUBRIC: row["metric_name"],
        }
        for subquestion, passed in scores.items():
            if subquestion == _ADHERENCE_RATE_KEY:
                continue
            tidy_rows.append(
                {
                    **base,
                    TidyColumns.SUBQUESTION: subquestion,
                    TidyColumns.PASSED: float(passed),  # type: ignore[arg-type]
                }
            )
    return tidy_rows


def _read_metric_jsonl(scores_dir: Path, metric: str) -> list[dict[str, object]]:
    """Read one metric's ``aggregated_scores.jsonl`` into dict rows.

    Args:
        scores_dir: ``<experiment_dir>/scores``.
        metric: Metric name (subdirectory under ``scores/``).

    Returns:
        Parsed JSON records (empty list if the file is absent).
    """
    path = scores_dir / metric / "aggregated_scores.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_tidy(experiment_dir: Path, metrics: list[str] | None = None) -> pd.DataFrame:
    """Build and materialize the tidy scores table for an experiment.

    Reads each metric's ``aggregated_scores.jsonl``, melts every row's ``scores``
    dict (excluding ``adherence_rate``) to one row per (observation,
    sub-question), maps to the tidy schema, concatenates all metrics, and writes
    the result to ``<experiment_dir>/scores/tidy_scores.parquet`` (pyarrow).

    Args:
        experiment_dir: Experiment root directory.
        metrics: Metric names to include; defaults to the three core rubrics.

    Returns:
        The materialized tidy DataFrame.
    """
    scores_dir = experiment_dir / "scores"
    selected = metrics if metrics is not None else DEFAULT_METRICS

    tidy_rows: list[dict[str, object]] = []
    for metric in selected:
        tidy_rows.extend(_melt_metric_rows(_read_metric_jsonl(scores_dir, metric)))

    df = pd.DataFrame(tidy_rows, columns=_TIDY_COLUMN_ORDER)
    df[TidyColumns.PASSED] = df[TidyColumns.PASSED].astype(float)

    output_path = scores_dir / _TIDY_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    return df


def load_tidy(experiment_dir: Path) -> pd.DataFrame:
    """Load the materialized tidy scores table.

    Args:
        experiment_dir: Experiment root directory.

    Returns:
        The tidy DataFrame read from ``scores/tidy_scores.parquet``.

    Raises:
        FileNotFoundError: If the artifact is missing — run ``build_tidy`` or
            ``scripts/build_tidy.py --experiment <id>`` first.
    """
    path = experiment_dir / "scores" / _TIDY_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Tidy artifact not found at {path}. "
            "Run build_tidy() or `python scripts/build_tidy.py --experiment <id>` first."
        )
    return pd.read_parquet(path, engine="pyarrow")


def load_tidy_multi(experiment_dirs: Sequence[Path]) -> pd.DataFrame:
    """Load and vertically concatenate tidy tables from several experiments.

    Each experiment already carries its own ``country`` value (derived per row),
    so concatenation alone yields a correct multi-country frame.

    Args:
        experiment_dirs: Sequence of experiment root directories, each containing
            a ``scores/tidy_scores.parquet`` artifact.

    Returns:
        Concatenated tidy DataFrame with rows from all experiments.
    """
    return pd.concat([load_tidy(d) for d in experiment_dirs], ignore_index=True)
