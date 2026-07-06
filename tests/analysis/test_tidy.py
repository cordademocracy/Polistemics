"""Unit tests for the format-aware tidy-table builder in ``src.analysis.tidy``."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.tidy import TidyColumns, build_tidy, load_tidy


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def fixture_experiment(tmp_path: Path) -> Path:
    """Create a tiny experiment dir with two synthetic aggregated_scores rows."""
    scores_dir = tmp_path / "scores"

    faith_rows = [
        {
            "observation_id": "bundestagswahl2025__de_spd__s001",
            "party_id": "de_spd",
            "statement_id": "bundestagswahl2025__s001",
            "model_id": "gpt_5_4",
            "metric_name": "faithfulness",
            "scores": {
                "position_representation": 1.0,
                "information_fabrication": 0.0,
                "adherence_rate": 0.5,  # must be dropped from the melt
            },
            "ie_name": "ie_clarity_vague",
        },
    ]
    epi_rows = [
        {
            "observation_id": "bundestagswahl2025__de_afd__s002",
            "party_id": "de_afd",
            "statement_id": "bundestagswahl2025__s002",
            "model_id": "qwen3_6_flash",
            "metric_name": "epistemic_calibration",
            "scores": {
                "epistemic_certainty": 1.0,
                "parametric_fallback": 1.0,
                "adherence_rate": 1.0,
            },
            "ie_name": "baseline",
        },
    ]
    _write_jsonl(scores_dir / "faithfulness" / "aggregated_scores.jsonl", faith_rows)
    _write_jsonl(scores_dir / "epistemic_calibration" / "aggregated_scores.jsonl", epi_rows)
    return tmp_path


def test_build_tidy_excludes_adherence_rate_and_melts(fixture_experiment: Path) -> None:
    """adherence_rate is dropped; one row per (observation, subquestion)."""
    df = build_tidy(fixture_experiment, metrics=["faithfulness", "epistemic_calibration"])

    assert TidyColumns.SUBQUESTION in df.columns
    assert "adherence_rate" not in df[TidyColumns.SUBQUESTION].unique()
    # faithfulness: 2 active SQ; epistemic_calibration: 2 active SQ -> 4 rows total.
    assert len(df) == 4


def test_build_tidy_columns_and_constants(fixture_experiment: Path) -> None:
    """Column set, constant columns, and IE normalization are correct."""
    df = build_tidy(fixture_experiment)

    expected_cols = {
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
    }
    assert set(df.columns) == expected_cols

    assert (df[TidyColumns.COUNTRY] == "DE").all()
    assert (df[TidyColumns.LABEL_TYPE] == "real").all()
    assert (df[TidyColumns.EVIDENCE_LANGUAGE] == "native").all()

    # IE normalization: ie_clarity_vague -> clarity, baseline -> baseline.
    assert set(df[TidyColumns.IE].unique()) == {"clarity", "baseline"}

    # passed is float; rubric mirrors metric_name.
    assert df[TidyColumns.PASSED].dtype == float
    assert set(df[TidyColumns.RUBRIC].unique()) == {"faithfulness", "epistemic_calibration"}


def test_build_tidy_round_trip(fixture_experiment: Path) -> None:
    """build_tidy writes parquet; load_tidy reads an identical frame."""
    written = build_tidy(fixture_experiment)
    parquet_path = fixture_experiment / "scores" / "tidy_scores.parquet"
    assert parquet_path.exists()

    loaded = load_tidy(fixture_experiment)
    # Sort + reset index for a deterministic comparison (row order is irrelevant).
    sort_cols = list(written.columns)
    pd.testing.assert_frame_equal(
        written.sort_values(sort_cols).reset_index(drop=True),
        loaded.sort_values(sort_cols).reset_index(drop=True),
    )


def test_load_tidy_missing_raises(tmp_path: Path) -> None:
    """load_tidy raises a clear error when the artifact is missing."""
    with pytest.raises(FileNotFoundError, match="build_tidy"):
        load_tidy(tmp_path)
