"""Smoke-tests for the multi-country plotting extensions in plotting.py.

Sets a non-interactive backend so no display is needed in CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.analysis.plotting import (
    DisplayNames,
    plot_country_delta_heatmap,
    plot_ie_adherence_overview,
    plot_ie_rubric_heatmap,
    save_plot,
)

# Tiny constant-score fixtures make Spearman undefined — silence the benign warning.
pytestmark = pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")

# ── Minimal DisplayNames for tests ──────────────────────────────────────────
_NAMES = DisplayNames(
    model={"m1": "M1", "m2": "M2"},
    ie={"baseline": "Baseline", "noise": "Noise"},
    rubric={"faithfulness": "Faithfulness", "epistemic_calibration": "EC"},
)


def _rubric_rows(
    *,
    ie: str,
    rubric: str,
    n_pass: int,
    n_total: int,
    model: str,
    country: str,
) -> list[dict[str, object]]:
    """Build n_total tidy rows whose pooled proportion is n_pass / n_total."""
    return [
        {
            "model": model,
            "country": country,
            "ie": ie,
            "party": "p",
            "rubric": rubric,
            "subquestion": f"{rubric}_sq{i}",
            "passed": 1.0 if i < n_pass else 0.0,
        }
        for i in range(n_total)
    ]


def _make_two_country_df() -> pd.DataFrame:
    """2-country (DE + NL), 2-model, 2-IE tidy frame.

    IEs: baseline (faithfulness + epistemic_calibration) and noise (same rubrics).
    """
    rows: list[dict[str, object]] = []
    for country in ("DE", "NL"):
        for model in ("m1", "m2"):
            # baseline: faithfulness 0.8, ec 0.6
            rows += _rubric_rows(ie="baseline", rubric="faithfulness", n_pass=8, n_total=10,
                                 model=model, country=country)
            rows += _rubric_rows(
                ie="baseline", rubric="epistemic_calibration", n_pass=6, n_total=10,
                model=model, country=country,
            )
            # noise: faithfulness 0.7, ec 0.5
            rows += _rubric_rows(ie="noise", rubric="faithfulness", n_pass=7, n_total=10,
                                 model=model, country=country)
            rows += _rubric_rows(ie="noise", rubric="epistemic_calibration", n_pass=5, n_total=10,
                                 model=model, country=country)
    return pd.DataFrame(rows)


def _make_one_country_df() -> pd.DataFrame:
    """Single-country (DE only) version for error-case tests."""
    rows: list[dict[str, object]] = []
    for model in ("m1", "m2"):
        rows += _rubric_rows(ie="baseline", rubric="faithfulness", n_pass=8, n_total=10,
                             model=model, country="DE")
        rows += _rubric_rows(ie="baseline", rubric="epistemic_calibration", n_pass=6, n_total=10,
                             model=model, country="DE")
        rows += _rubric_rows(ie="noise", rubric="faithfulness", n_pass=7, n_total=10,
                             model=model, country="DE")
        rows += _rubric_rows(ie="noise", rubric="epistemic_calibration", n_pass=5, n_total=10,
                             model=model, country="DE")
    return pd.DataFrame(rows)


# ── Tests: plot_ie_rubric_heatmap ────────────────────────────────────────────

def test_ie_rubric_heatmap_two_countries_returns_list() -> None:
    """With 2 countries and no country= filter, should return a list of Axes."""
    df = _make_two_country_df()
    result = plot_ie_rubric_heatmap(df, ie="baseline", names=_NAMES)
    assert isinstance(result, list), "Expected list[Axes] for multi-country frame"
    assert len(result) == 2
    for ax in result:
        assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_ie_rubric_heatmap_with_country_filter_returns_single_axes() -> None:
    """With country='DE', should return a single Axes, not a list."""
    df = _make_two_country_df()
    result = plot_ie_rubric_heatmap(df, ie="baseline", country="DE", names=_NAMES)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_ie_rubric_heatmap_show_country_correlation_runs() -> None:
    """show_country_correlation=True with country='DE' should not raise."""
    df = _make_two_country_df()
    result = plot_ie_rubric_heatmap(
        df, ie="baseline", country="DE", show_country_correlation=True, names=_NAMES
    )
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_ie_rubric_heatmap_ax_raises_on_multicountry() -> None:
    """Providing ax= in multi-country mode should raise ValueError."""
    df = _make_two_country_df()
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="axis"):
        plot_ie_rubric_heatmap(df, ie="baseline", ax=ax, names=_NAMES)
    plt.close("all")


# ── Tests: plot_ie_adherence_overview ────────────────────────────────────────

def test_overview_with_country_filter_returns_single_figure() -> None:
    """country='DE' should return a single Figure."""
    df = _make_two_country_df()
    result = plot_ie_adherence_overview(df, country="DE", names=_NAMES)
    assert isinstance(result, plt.Figure)
    plt.close("all")


def test_overview_aggregate_countries_returns_single_figure() -> None:
    """aggregate_countries=True should return a single Figure."""
    df = _make_two_country_df()
    result = plot_ie_adherence_overview(df, aggregate_countries=True, names=_NAMES)
    assert isinstance(result, plt.Figure)
    plt.close("all")


def test_overview_delta_mode_returns_single_figure() -> None:
    """mode='delta' with 2 countries should return a single Figure."""
    df = _make_two_country_df()
    result = plot_ie_adherence_overview(df, mode="delta", names=_NAMES)
    assert isinstance(result, plt.Figure)
    plt.close("all")


def test_overview_delta_mode_one_country_raises() -> None:
    """mode='delta' with a 1-country frame should raise ValueError."""
    df = _make_one_country_df()
    with pytest.raises(ValueError, match="exactly 2 countries"):
        plot_ie_adherence_overview(df, mode="delta", names=_NAMES)
    plt.close("all")


def test_overview_delta_mode_with_country_raises() -> None:
    """mode='delta' + country= is incompatible and should raise ValueError."""
    df = _make_two_country_df()
    with pytest.raises(ValueError, match="incompatible"):
        plot_ie_adherence_overview(df, mode="delta", country="DE", names=_NAMES)
    plt.close("all")


def test_overview_multi_country_returns_list() -> None:
    """No country filter + 2-country frame should return list[Figure]."""
    df = _make_two_country_df()
    result = plot_ie_adherence_overview(df, names=_NAMES)
    assert isinstance(result, list)
    assert len(result) == 2
    for fig in result:
        assert isinstance(fig, plt.Figure)
    plt.close("all")


# ── Tests: plot_country_delta_heatmap ────────────────────────────────────────

def test_country_delta_heatmap_axis_rubric_returns_axes() -> None:
    """axis='rubric' should return an Axes."""
    df = _make_two_country_df()
    result = plot_country_delta_heatmap(df, axis="rubric", names=_NAMES)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_country_delta_heatmap_axis_rubric_star_returns_axes() -> None:
    """axis='rubric' with star_from_nl_sq=True should return an Axes."""
    df = _make_two_country_df()
    result = plot_country_delta_heatmap(df, axis="rubric", star_from_nl_sq=True, names=_NAMES)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_country_delta_heatmap_axis_ie_returns_axes() -> None:
    """axis='ie' should return an Axes."""
    df = _make_two_country_df()
    result = plot_country_delta_heatmap(df, axis="ie", names=_NAMES)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_country_delta_heatmap_invalid_axis_raises() -> None:
    """Invalid axis value should raise ValueError."""
    df = _make_two_country_df()
    with pytest.raises(ValueError, match="axis must be"):
        plot_country_delta_heatmap(df, axis="party", names=_NAMES)  # type: ignore[arg-type]
    plt.close("all")


# ── Tests: save_plot with subfolder ─────────────────────────────────────────

def test_save_plot_subfolder_nests_correctly(tmp_path: pytest.FixtureRequest) -> None:
    """subfolder='noise' should write under <focus>/exploratory/noise/."""
    fig = plt.figure()
    result = save_plot(
        fig=fig,
        repo_root=tmp_path,  # type: ignore[arg-type]
        notebook_file="01_analysis.ipynb",
        plot_name="my plot",
        subfolder="noise",
    )
    out_path = result["exploratory_png"]
    # Must contain /exploratory/noise/ in that order (subfolder is inside exploratory/).
    path_str = str(out_path)
    assert "/exploratory/noise/" in path_str, (
        f"Expected '/exploratory/noise/' in path, got: {path_str}"
    )
    assert out_path.exists(), "Expected the file to have been saved"
    plt.close("all")


def test_save_plot_no_subfolder_unchanged(tmp_path: pytest.FixtureRequest) -> None:
    """Without subfolder, path must NOT contain an extra directory."""
    fig = plt.figure()
    result = save_plot(
        fig=fig,
        repo_root=tmp_path,  # type: ignore[arg-type]
        notebook_file="01_analysis.ipynb",
        plot_name="my plot",
    )
    out_path = result["exploratory_png"]
    path_str = str(out_path)
    # Should end with /analysis/exploratory/my_plot.png (no extra subdir).
    assert path_str.endswith("exploratory/my_plot.png"), (
        f"Unexpected path without subfolder: {path_str}"
    )
    plt.close("all")
