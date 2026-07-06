from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.common.io import (
    DEFAULT_DATA_DIR,
    append_output_to_path,
    get_experiment_dir,
    load_ground_truth,
    load_outputs_from_path,
    save_ground_truth,
    save_item_scores,
)
from src.common.schemas import (
    DatasetItem,
    ItemScore,
    LLMOutput,
    OutputStatus,
    PromptVariation,
    RefusalType,
    StanceLabel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset_item(
    obs_id: str,
    party_name: str = "SPD",
    statement_category: str | None = "economy",
) -> DatasetItem:
    return DatasetItem(
        observation_id=obs_id,
        election_id="bundestagswahl2025",
        party_id="de_spd",
        party_name=party_name,
        party_anonymized="Party 01",
        statement_id="bundestagswahl2025__s001",
        statement_number=None,
        statement_text="Test statement",
        statement_category=statement_category,
        stance_label=StanceLabel.AGREE,
        rationale_text="Test rationale",
        has_rationale=True,
        ie_name="baseline",
        ie_chunks=[],
    )


def _make_llm_output(obs_id: str, party_id: str = "de_spd") -> LLMOutput:
    return LLMOutput(
        observation_id=obs_id,
        statement_id="bundestagswahl2025__s001",
        party_id=party_id,
        experiment_id="test",
        model_id="model_a",
        prompt_variation=PromptVariation.MINIMAL,
        run_index=0,
        temperature=0.0,
        predicted_stance=StanceLabel.AGREE,
        predicted_explanation="Test explanation",
        timestamp=datetime.now(UTC),
        latency_ms=100.0,
        tokens_input=50,
        tokens_output=100,
        cost_usd=0.001,
        status=OutputStatus.SUCCESS,
        error_message=None,
        refusal_type=RefusalType.NONE,
        ie_name="baseline",
        condition_id="baseline__real__evidence__none",
    )


def _make_item_score(obs_id: str) -> ItemScore:
    return ItemScore(
        observation_id=obs_id,
        party_id="de_spd",
        statement_id="bundestagswahl2025__s001",
        model_id="model_a",
        prompt_variation=PromptVariation.MINIMAL,
        run_index=0,
        temperature=0.0,
        metric_name="test_metric",
        scores={"value": 1.0},
        ie_name="baseline",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gt_sidecar_roundtrip(tmp_path: Path) -> None:
    """Save 2 items, load, check count and IDs."""
    items = [_make_dataset_item("obs_001"), _make_dataset_item("obs_002")]
    path = tmp_path / "gt.jsonl"

    save_ground_truth(items, path)
    loaded = load_ground_truth(path)

    assert len(loaded) == 2
    assert loaded[0].observation_id == "obs_001"
    assert loaded[1].observation_id == "obs_002"


def test_gt_sidecar_preserves_all_fields(tmp_path: Path) -> None:
    """Save item with party_name and statement_category, load, verify fields preserved."""
    item = _make_dataset_item("obs_003", party_name="CDU", statement_category="migration")
    path = tmp_path / "gt.jsonl"

    save_ground_truth([item], path)
    loaded = load_ground_truth(path)

    assert len(loaded) == 1
    result = loaded[0]
    assert result.party_name == "CDU"
    assert result.statement_category == "migration"
    assert result.stance_label == StanceLabel.AGREE
    assert result.has_rationale is True
    assert result.rationale_text == "Test rationale"


def test_gt_sidecar_empty_file(tmp_path: Path) -> None:
    """Load from nonexistent path returns []."""
    path = tmp_path / "nonexistent.jsonl"
    assert load_ground_truth(path) == []


def test_get_experiment_dir() -> None:
    """Check path construction."""
    result = get_experiment_dir("exp_abc")
    expected = DEFAULT_DATA_DIR / "experiments" / "exp_abc"
    assert result == expected


def test_get_experiment_dir_custom_base(tmp_path: Path) -> None:
    """Custom base_dir is respected."""
    result = get_experiment_dir("exp_xyz", base_dir=tmp_path)
    assert result == tmp_path / "exp_xyz"


def test_load_outputs_from_path(tmp_path: Path) -> None:
    """Append then load from explicit path."""
    path = tmp_path / "outputs.jsonl"
    out1 = _make_llm_output("obs_001")
    out2 = _make_llm_output("obs_002", party_id="de_cdu")

    append_output_to_path(out1, path)
    append_output_to_path(out2, path)

    loaded = load_outputs_from_path(path)
    assert len(loaded) == 2
    assert loaded[0].observation_id == "obs_001"
    assert loaded[1].party_id == "de_cdu"


def test_append_output_to_path_creates_dir(tmp_path: Path) -> None:
    """Verify parent dirs are created automatically."""
    path = tmp_path / "deep" / "nested" / "outputs.jsonl"
    assert not path.parent.exists()

    append_output_to_path(_make_llm_output("obs_001"), path)

    assert path.parent.exists()
    assert path.exists()
    loaded = load_outputs_from_path(path)
    assert len(loaded) == 1


def test_score_io_roundtrip(tmp_path: Path) -> None:
    """Save item scores, verify file exists with correct line count."""
    scores = [_make_item_score("obs_001"), _make_item_score("obs_002"), _make_item_score("obs_003")]
    path = tmp_path / "scores.jsonl"

    save_item_scores(scores, path)

    assert path.exists()
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
