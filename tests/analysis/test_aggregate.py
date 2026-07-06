"""Unit tests for the pure aggregation functions in ``src.analysis.aggregate``.

The most important test is :func:`test_benchmark_score_inside_out_worked_example`,
which locks in the load-bearing inside-out aggregation order
(pool within rubric -> mean across rubrics per IE -> mean across IEs) against the
reversed (rubrics-across-IEs-first) order. See ``analysis/AGGREGATION.md`` §3.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.aggregate import (
    FLAG_THRESHOLD,
    adherence_index,
    benchmark_score,
    geometric_benchmark_score,
    ie_dispersion,
    party_spread,
    robustness_score,
    rubric_dispersion,
    rubric_robustness,
    subquestion_adherence,
    subquestion_dispersion,
)


def _rubric_rows(
    *,
    ie: str,
    rubric: str,
    n_pass: int,
    n_total: int,
    model: str = "m",
    country: str = "DE",
    party: str = "p",
) -> list[dict[str, object]]:
    """Build ``n_total`` tidy rows whose pooled proportion is ``n_pass / n_total``.

    Each row is a distinct sub-question (the pool counts rows, not unique
    sub-question names, so the names only need to be unique enough to be valid).
    """
    rows: list[dict[str, object]] = []
    for i in range(n_total):
        rows.append(
            {
                "model": model,
                "country": country,
                "ie": ie,
                "party": party,
                "rubric": rubric,
                "subquestion": f"{rubric}_sq{i}",
                "passed": 1.0 if i < n_pass else 0.0,
            }
        )
    return rows


def _worked_example_df() -> pd.DataFrame:
    """Tidy frame for the spec's worked example.

    - baseline: faithfulness=0.9 (9/10), impartiality=0.8 (4/5),
      epistemic_calibration=0.6 (3/5)
    - availability: epistemic_calibration=0.4 (2/5) only
    """
    rows: list[dict[str, object]] = []
    rows += _rubric_rows(ie="baseline", rubric="faithfulness", n_pass=9, n_total=10)
    rows += _rubric_rows(ie="baseline", rubric="impartiality", n_pass=4, n_total=5)
    rows += _rubric_rows(ie="baseline", rubric="epistemic_calibration", n_pass=3, n_total=5)
    rows += _rubric_rows(ie="availability", rubric="epistemic_calibration", n_pass=2, n_total=5)
    return pd.DataFrame(rows)


def test_benchmark_score_inside_out_worked_example() -> None:
    """Lock in the inside-out order: 0.583, NOT the reversed 0.733."""
    df = _worked_example_df()

    result = benchmark_score(df, by=("model", "country"))

    assert len(result) == 1
    got = float(result["benchmark_score"].iloc[0])

    # Inside-out (correct):
    #   adherence(baseline)     = mean(0.9, 0.8, 0.6) = 0.7666...
    #   adherence(availability) = mean(0.4)           = 0.4
    #   benchmark = mean(0.7666..., 0.4)              = 0.58333...
    expected_inside_out = ((0.9 + 0.8 + 0.6) / 3 + 0.4) / 2
    assert got == pytest.approx(expected_inside_out, abs=1e-9)
    assert got == pytest.approx(0.5833333333, abs=1e-9)

    # Reversed (wrong) — average rubrics across IEs first:
    #   F_avg = mean(0.9)      = 0.9   [baseline only]
    #   I_avg = mean(0.8)      = 0.8   [baseline only]
    #   C_avg = mean(0.6, 0.4) = 0.5   [both IEs]
    #   benchmark = mean(0.9, 0.8, 0.5) = 0.73333...
    reversed_wrong = (0.9 + 0.8 + (0.6 + 0.4) / 2) / 3
    assert reversed_wrong == pytest.approx(0.7333333333, abs=1e-9)

    # The two MUST differ — proves the implementation is not the flat/reversed form.
    assert got != pytest.approx(reversed_wrong, abs=1e-6)


def test_geometric_benchmark_leans_to_weak_ie_below_arithmetic() -> None:
    """Geometric across-IE mean < arithmetic, and pulled toward the weak IE."""
    df = _worked_example_df()

    arith = float(benchmark_score(df, by=("model", "country"))["benchmark_score"].iloc[0])
    geo = float(
        geometric_benchmark_score(df, by=("model", "country"))["benchmark_score"].iloc[0]
    )

    # per-IE adherence: baseline = mean(0.9, 0.8, 0.6) = 0.7666..., availability = 0.4
    # geometric = sqrt(0.7666... * 0.4) = 0.55377...
    expected_geo = (((0.9 + 0.8 + 0.6) / 3) * 0.4) ** 0.5
    assert geo == pytest.approx(expected_geo, abs=1e-9)
    assert geo == pytest.approx(0.5537749, abs=1e-6)
    # By AM-GM the geometric mean is strictly below the arithmetic one (0.5833).
    assert geo < arith


def test_geometric_benchmark_rejects_zero_cell() -> None:
    """A zero-valued per-IE cell raises rather than silently collapsing to 0."""
    rows: list[dict[str, object]] = []
    rows += _rubric_rows(ie="baseline", rubric="faithfulness", n_pass=9, n_total=10)
    # availability epistemic_calibration with 0/5 pass -> per-IE adherence 0.0
    rows += _rubric_rows(ie="availability", rubric="epistemic_calibration", n_pass=0, n_total=5)
    df = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="zero-valued per-IE adherence"):
        geometric_benchmark_score(df)


def test_benchmark_score_not_flat_groupby() -> None:
    """A flat grand-mean of ``passed`` differs from the inside-out benchmark."""
    df = _worked_example_df()
    got = float(benchmark_score(df)["benchmark_score"].iloc[0])

    flat = float(df["passed"].mean())  # 18 passes / 25 rows = 0.72
    assert flat == pytest.approx(0.72, abs=1e-9)
    assert got != pytest.approx(flat, abs=1e-6)


def test_adherence_index_equal_rubric_weight() -> None:
    """A rubric with more sub-questions must not dominate (equal rubric weight)."""
    rows: list[dict[str, object]] = []
    # Rubric A: 100 sub-questions, all pass -> proportion 1.0
    rows += _rubric_rows(ie="baseline", rubric="A", n_pass=100, n_total=100)
    # Rubric B: 2 sub-questions, none pass -> proportion 0.0
    rows += _rubric_rows(ie="baseline", rubric="B", n_pass=0, n_total=2)
    df = pd.DataFrame(rows)

    result = adherence_index(df, by=["model", "country", "ie"])

    assert len(result) == 1
    # Equal rubric weight -> mean(1.0, 0.0) = 0.5, NOT the row-weighted 100/102.
    assert float(result["adherence"].iloc[0]) == pytest.approx(0.5, abs=1e-9)


def test_adherence_index_rubric_in_by_returns_rubric_proportion() -> None:
    """With ``rubric`` in ``by``, Stage 2 is degenerate -> plain rubric proportion.

    This is the per-rubric drill (AGGREGATION.md §5, Focus 1). It must not raise
    a duplicate-column error and must return each rubric's own proportion.
    """
    rows: list[dict[str, object]] = []
    rows += _rubric_rows(ie="baseline", rubric="A", n_pass=9, n_total=10)  # 0.9
    rows += _rubric_rows(ie="baseline", rubric="B", n_pass=2, n_total=5)   # 0.4
    df = pd.DataFrame(rows)

    result = adherence_index(df, by=["model", "country", "rubric"])

    assert len(result) == 2
    by_rubric = result.set_index("rubric")["adherence"]
    assert float(by_rubric.loc["A"]) == pytest.approx(0.9, abs=1e-9)
    assert float(by_rubric.loc["B"]) == pytest.approx(0.4, abs=1e-9)


def test_party_spread_max_minus_min_and_flag_boundary() -> None:
    """Spread = max - min of per-party adherence; flag at the 0.20 boundary."""
    rows: list[dict[str, object]] = []
    # party p1 adherence 0.7 (single rubric, 7/10)
    rows += _rubric_rows(ie="baseline", rubric="R", n_pass=7, n_total=10, party="p1")
    # party p2 adherence 0.5 (single rubric, 5/10) -> spread = 0.20 exactly
    rows += _rubric_rows(ie="baseline", rubric="R", n_pass=5, n_total=10, party="p2")
    df = pd.DataFrame(rows)

    result = party_spread(df, by=("model", "country", "ie"))

    assert len(result) == 1
    assert float(result["spread"].iloc[0]) == pytest.approx(0.20, abs=1e-9)
    # flag is spread >= FLAG_THRESHOLD; 0.20 >= 0.20 is True (boundary inclusive).
    assert bool(result["flag"].iloc[0]) is True
    assert FLAG_THRESHOLD == 0.20


def test_party_spread_below_threshold_not_flagged() -> None:
    """Just-below-threshold spread is not flagged."""
    rows: list[dict[str, object]] = []
    rows += _rubric_rows(ie="baseline", rubric="R", n_pass=7, n_total=10, party="p1")
    rows += _rubric_rows(ie="baseline", rubric="R", n_pass=6, n_total=10, party="p2")
    df = pd.DataFrame(rows)

    result = party_spread(df)
    assert float(result["spread"].iloc[0]) == pytest.approx(0.10, abs=1e-9)
    assert bool(result["flag"].iloc[0]) is False


def test_ie_dispersion_sd_and_spread_across_ies() -> None:
    """SD/spread computed across per-IE adherence values."""
    rows: list[dict[str, object]] = []
    # ie1 adherence 0.8, ie2 adherence 0.4 (single rubric each)
    rows += _rubric_rows(ie="ie1", rubric="R", n_pass=8, n_total=10)
    rows += _rubric_rows(ie="ie2", rubric="R", n_pass=4, n_total=10)
    df = pd.DataFrame(rows)

    result = ie_dispersion(df, by=("model", "country"))

    assert len(result) == 1
    assert float(result["ie_spread"].iloc[0]) == pytest.approx(0.4, abs=1e-9)
    # pandas std is sample std (ddof=1): std([0.8, 0.4]) = 0.2 * sqrt(2)
    expected_sd = pd.Series([0.8, 0.4]).std()
    assert float(result["ie_sd"].iloc[0]) == pytest.approx(expected_sd, abs=1e-9)


def test_rubric_dispersion_sd_and_spread_across_rubrics() -> None:
    """SD/spread computed across per-rubric adherence values (mirrors IE version)."""
    rows: list[dict[str, object]] = []
    # rubric A adherence 0.9, rubric B adherence 0.5 (single IE each)
    rows += _rubric_rows(ie="baseline", rubric="A", n_pass=9, n_total=10)
    rows += _rubric_rows(ie="baseline", rubric="B", n_pass=5, n_total=10)
    df = pd.DataFrame(rows)

    result = rubric_dispersion(df, by=("model", "country"))

    assert len(result) == 1
    assert float(result["rubric_spread"].iloc[0]) == pytest.approx(0.4, abs=1e-9)
    expected_sd = pd.Series([0.9, 0.5]).std()
    assert float(result["rubric_sd"].iloc[0]) == pytest.approx(expected_sd, abs=1e-9)


def test_subquestion_dispersion_across_subquestions() -> None:
    """SD/spread computed across per-sub-question pass rates within a group.

    Two sub-questions repeated across observations: ``sqA`` passes 9/10 (0.9),
    ``sqB`` passes 5/10 (0.5) -> spread 0.4. This is the micro signal a healthy
    rubric mean can hide.
    """
    rows: list[dict[str, object]] = []
    for i in range(10):
        rows.append(
            {
                "model": "m",
                "country": "DE",
                "ie": "baseline",
                "party": "p",
                "rubric": "R",
                "subquestion": "sqA",
                "passed": 1.0 if i < 9 else 0.0,
            }
        )
    for i in range(10):
        rows.append(
            {
                "model": "m",
                "country": "DE",
                "ie": "baseline",
                "party": "p",
                "rubric": "R",
                "subquestion": "sqB",
                "passed": 1.0 if i < 5 else 0.0,
            }
        )
    df = pd.DataFrame(rows)

    result = subquestion_dispersion(df, by=("model", "country", "rubric"))

    assert len(result) == 1
    assert float(result["sq_spread"].iloc[0]) == pytest.approx(0.4, abs=1e-9)
    expected_sd = pd.Series([0.9, 0.5]).std()
    assert float(result["sq_sd"].iloc[0]) == pytest.approx(expected_sd, abs=1e-9)


def test_rubric_robustness_within_rubric_sd_excludes_baseline() -> None:
    """SD is taken within each rubric across its active IEs; baseline dropped.

    Rubric A spans baseline + 2 stress IEs (baseline must be excluded by default);
    rubric B spans 3 stress IEs. ``n_ies`` documents the differing active-set
    widths — the commensurable-per-rubric property that ``ie_dispersion`` lacks.
    """
    rows: list[dict[str, object]] = []
    # Rubric A: baseline 1.0 (EXCLUDED), ie1 0.9, ie2 0.5 -> SD over [0.9, 0.5].
    rows += _rubric_rows(ie="baseline", rubric="A", n_pass=10, n_total=10)
    rows += _rubric_rows(ie="ie1", rubric="A", n_pass=9, n_total=10)
    rows += _rubric_rows(ie="ie2", rubric="A", n_pass=5, n_total=10)
    # Rubric B: ie1 0.8, ie2 0.6, ie3 0.4 -> SD over [0.8, 0.6, 0.4].
    rows += _rubric_rows(ie="ie1", rubric="B", n_pass=8, n_total=10)
    rows += _rubric_rows(ie="ie2", rubric="B", n_pass=6, n_total=10)
    rows += _rubric_rows(ie="ie3", rubric="B", n_pass=4, n_total=10)
    df = pd.DataFrame(rows)

    result = rubric_robustness(df, by=("model", "country"))

    by_rubric = result.set_index("rubric")
    assert int(by_rubric.loc["A", "n_ies"]) == 2  # baseline excluded
    assert int(by_rubric.loc["B", "n_ies"]) == 3
    assert float(by_rubric.loc["A", "rubric_sd"]) == pytest.approx(
        pd.Series([0.9, 0.5]).std(), abs=1e-9
    )
    assert float(by_rubric.loc["B", "rubric_sd"]) == pytest.approx(
        pd.Series([0.8, 0.6, 0.4]).std(), abs=1e-9
    )


def test_robustness_score_is_equal_rubric_weight_mean_of_sds() -> None:
    """Headline robustness = unweighted mean of the per-rubric SDs (inside-out)."""
    rows: list[dict[str, object]] = []
    rows += _rubric_rows(ie="baseline", rubric="A", n_pass=10, n_total=10)
    rows += _rubric_rows(ie="ie1", rubric="A", n_pass=9, n_total=10)
    rows += _rubric_rows(ie="ie2", rubric="A", n_pass=5, n_total=10)
    rows += _rubric_rows(ie="ie1", rubric="B", n_pass=8, n_total=10)
    rows += _rubric_rows(ie="ie2", rubric="B", n_pass=6, n_total=10)
    rows += _rubric_rows(ie="ie3", rubric="B", n_pass=4, n_total=10)
    df = pd.DataFrame(rows)

    result = robustness_score(df, by=("model", "country"))

    assert len(result) == 1
    expected = (pd.Series([0.9, 0.5]).std() + pd.Series([0.8, 0.6, 0.4]).std()) / 2
    assert float(result["robustness"].iloc[0]) == pytest.approx(expected, abs=1e-9)


def test_subquestion_adherence_per_subquestion_pass_rate() -> None:
    """One row per sub-question, each holding its plain pooled pass rate."""
    rows: list[dict[str, object]] = []
    for i in range(10):  # sqA: 9/10 pass -> 0.9
        rows.append(
            {"model": "m", "country": "DE", "ie": "noise", "party": "p",
             "rubric": "impartiality", "subquestion": "sqA", "passed": 1.0 if i < 9 else 0.0}
        )
    for i in range(10):  # sqB: 3/10 pass -> 0.3
        rows.append(
            {"model": "m", "country": "DE", "ie": "noise", "party": "p",
             "rubric": "impartiality", "subquestion": "sqB", "passed": 1.0 if i < 3 else 0.0}
        )
    df = pd.DataFrame(rows)

    result = subquestion_adherence(df, by=("model", "subquestion"))

    by_sq = result.set_index("subquestion")["adherence"]
    assert len(result) == 2
    assert float(by_sq.loc["sqA"]) == pytest.approx(0.9, abs=1e-9)
    assert float(by_sq.loc["sqB"]) == pytest.approx(0.3, abs=1e-9)
