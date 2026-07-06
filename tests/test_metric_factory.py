from __future__ import annotations

from src.common.schemas import DatasetItem, StanceLabel
from src.metrics.factory import build_metrics


def _make_gt_items() -> list[DatasetItem]:
    return [
        DatasetItem(
            observation_id="obs1", election_id="btw2025", party_id="de_spd",
            party_name="SPD", party_anonymized="Party 01", statement_id="s001",
            statement_number=None,
            statement_text="Test", statement_category="Wirtschaft",
            stance_label=StanceLabel.AGREE, rationale_text=None, has_rationale=False,
            ie_name="baseline", ie_chunks=[],
        ),
        DatasetItem(
            observation_id="obs2", election_id="btw2025", party_id="de_cdu",
            party_name="CDU/CSU", party_anonymized="Party 02", statement_id="s001",
            statement_number=None,
            statement_text="Test", statement_category="Wirtschaft",
            stance_label=StanceLabel.DISAGREE, rationale_text=None, has_rationale=False,
            ie_name="baseline", ie_chunks=[],
        ),
    ]


def test_build_unknown_metric_skipped() -> None:
    metrics = build_metrics(["nonexistent"], _make_gt_items())
    assert "nonexistent" not in metrics


def test_build_faithfulness_without_panel_skipped() -> None:
    """faithfulness requires panel + prompt_builder — skipped when absent."""
    metrics = build_metrics(["faithfulness"], _make_gt_items())
    assert "faithfulness" not in metrics


def test_build_impartiality_without_panel_skipped() -> None:
    """impartiality requires panel + prompt_builder — skipped when absent."""
    metrics = build_metrics(["impartiality"], _make_gt_items())
    assert "impartiality" not in metrics


def test_build_epistemic_calibration_without_panel_skipped() -> None:
    """epistemic_calibration requires panel + prompt_builder — skipped when absent."""
    metrics = build_metrics(["epistemic_calibration"], _make_gt_items())
    assert "epistemic_calibration" not in metrics
