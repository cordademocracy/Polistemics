"""Unit tests for ``src.analysis.replication`` and ``load_tidy_multi``."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.analysis.replication import (
    RankCorrelation,
    country_delta,
    ie_difficulty_correlation,
    model_rank_agreement,
)
from src.analysis.tidy import load_tidy_multi

# Constant-score fixtures make Spearman undefined (NaN rho) by design — that is
# the asserted behaviour, so silence the benign scipy warning it emits.
pytestmark = pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Build ``n_total`` tidy rows whose pooled proportion is ``n_pass / n_total``."""
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


def _make_two_country_frame(
    ie_scores: dict[str, tuple[float, float]],
    *,
    model: str = "m",
    rubric: str = "faithfulness",
    n_total: int = 10,
) -> pd.DataFrame:
    """Build a tidy frame with DE + NL, each IE at the specified adherence.

    Args:
        ie_scores: ``{ie_name: (de_rate, nl_rate)}``.  Rate in [0, 1].
        model: Model name for all rows.
        rubric: Rubric for all rows.
        n_total: Rows per cell (pass count = round(rate * n_total)).
    """
    rows: list[dict[str, object]] = []
    for ie, (de_rate, nl_rate) in ie_scores.items():
        de_pass = round(de_rate * n_total)
        nl_pass = round(nl_rate * n_total)
        rows += _rubric_rows(
            ie=ie, rubric=rubric, n_pass=de_pass, n_total=n_total, model=model, country="DE"
        )
        rows += _rubric_rows(
            ie=ie, rubric=rubric, n_pass=nl_pass, n_total=n_total, model=model, country="NL"
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ie_difficulty_correlation
# ---------------------------------------------------------------------------

_SIX_IES = ["baseline", "noise", "prior_conflict", "clarity", "consistency", "availability"]


def test_ie_difficulty_correlation_perfect_agreement() -> None:
    """Identical IE-difficulty ordering across countries → rho == 1.0."""
    # DE and NL have the SAME difficulty values for all 6 IEs.
    ie_scores = {ie: (0.9 - 0.1 * i, 0.9 - 0.1 * i) for i, ie in enumerate(_SIX_IES)}
    df = _make_two_country_frame(ie_scores)

    rc = ie_difficulty_correlation(df)

    assert isinstance(rc, RankCorrelation)
    assert rc.rho == pytest.approx(1.0, abs=1e-9)
    assert rc.n == 6
    assert rc.p is not None


def test_ie_difficulty_correlation_perfect_reversal() -> None:
    """Reversed IE-difficulty ordering across countries → rho == -1.0."""
    de_vals = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    nl_vals = de_vals[::-1]  # exact reversal
    ie_scores = {ie: (de_vals[i], nl_vals[i]) for i, ie in enumerate(_SIX_IES)}
    df = _make_two_country_frame(ie_scores)

    rc = ie_difficulty_correlation(df)

    assert rc.rho == pytest.approx(-1.0, abs=1e-9)
    assert rc.n == 6


def test_ie_difficulty_correlation_n_equals_shared_ies() -> None:
    """n is the count of IEs present in both countries."""
    # Only 3 IEs, both countries.
    three_ies = ["baseline", "noise", "prior_conflict"]
    ie_scores = {ie: (0.9 - 0.1 * i, 0.9 - 0.1 * i) for i, ie in enumerate(three_ies)}
    df = _make_two_country_frame(ie_scores)

    rc = ie_difficulty_correlation(df)

    assert rc.n == 3


# ---------------------------------------------------------------------------
# model_rank_agreement — overall level
# ---------------------------------------------------------------------------

def _model_overall_df(
    model_scores: dict[str, tuple[float, float]],
    *,
    rubric: str = "faithfulness",
    ie: str = "baseline",
    n_total: int = 10,
) -> pd.DataFrame:
    """Build a frame with several models, each at a known overall adherence per country.

    ``model_scores`` maps model_name -> (de_rate, nl_rate).
    """
    rows: list[dict[str, object]] = []
    for model, (de_rate, nl_rate) in model_scores.items():
        rows += _rubric_rows(
            ie=ie, rubric=rubric,
            n_pass=round(de_rate * n_total), n_total=n_total,
            model=model, country="DE",
        )
        rows += _rubric_rows(
            ie=ie, rubric=rubric,
            n_pass=round(nl_rate * n_total), n_total=n_total,
            model=model, country="NL",
        )
    return pd.DataFrame(rows)


def test_model_rank_agreement_overall_identical_order() -> None:
    """Same model ranking in both countries → rho == 1.0, p is None."""
    model_scores = {"A": (0.9, 0.8), "B": (0.7, 0.6), "C": (0.5, 0.4)}
    df = _model_overall_df(model_scores)

    result = model_rank_agreement(df, level="overall")

    assert list(result.columns) == ["group", "rho", "n", "order_a", "order_b"]
    assert len(result) == 1
    assert result["group"].iloc[0] == "overall"
    assert float(result["rho"].iloc[0]) == pytest.approx(1.0, abs=1e-9)
    # p is not a column — it is None inside RankCorrelation and not surfaced to the DF


def test_model_rank_agreement_overall_p_is_none() -> None:
    """model_rank_agreement must always set p=None (no p-value for small n)."""
    model_scores = {"A": (0.9, 0.8), "B": (0.7, 0.6), "C": (0.5, 0.4)}
    df = _model_overall_df(model_scores)

    result = model_rank_agreement(df, level="overall")

    # p is never reported for model-rank agreement (n=3 is not inferential): the
    # DataFrame intentionally omits a "p" column.
    assert "p" not in result.columns


def test_model_rank_agreement_overall_schema() -> None:
    """Schema must be exactly [group, rho, n, order_a, order_b]."""
    model_scores = {"A": (0.9, 0.8), "B": (0.7, 0.6)}
    df = _model_overall_df(model_scores)

    result = model_rank_agreement(df, level="overall")

    assert list(result.columns) == ["group", "rho", "n", "order_a", "order_b"]


def test_model_rank_agreement_overall_adjacent_swap() -> None:
    """One adjacent swap in 3 models → spearman rho == 0.5."""
    # DE order: A > B > C; NL order: A > C > B (one adjacent swap).
    # spearman([0.9, 0.7, 0.5], [0.8, 0.4, 0.6]) should give 0.5.
    model_scores = {"A": (0.9, 0.8), "B": (0.7, 0.4), "C": (0.5, 0.6)}
    df = _model_overall_df(model_scores)

    result = model_rank_agreement(df, level="overall")

    assert float(result["rho"].iloc[0]) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# model_rank_agreement — ie and rubric levels
# ---------------------------------------------------------------------------

def _multi_ie_model_df() -> pd.DataFrame:
    """Frame with 3 models × 2 countries × 3 IEs × 2 rubrics."""
    ies = ["baseline", "noise", "prior_conflict"]
    rubrics = ["faithfulness", "impartiality"]
    models = ["A", "B", "C"]
    rows: list[dict[str, object]] = []
    for model in models:
        for ie in ies:
            for rubric in rubrics:
                rows += _rubric_rows(
                    ie=ie, rubric=rubric, n_pass=5, n_total=10, model=model, country="DE"
                )
                rows += _rubric_rows(
                    ie=ie, rubric=rubric, n_pass=5, n_total=10, model=model, country="NL"
                )
    return pd.DataFrame(rows)


def test_model_rank_agreement_ie_one_row_per_ie() -> None:
    """level='ie' must emit one row per IE value."""
    df = _multi_ie_model_df()
    result = model_rank_agreement(df, level="ie")

    assert set(result["group"]) == {"baseline", "noise", "prior_conflict"}
    assert len(result) == 3


def test_model_rank_agreement_rubric_one_row_per_rubric() -> None:
    """level='rubric' must emit one row per rubric value."""
    df = _multi_ie_model_df()
    result = model_rank_agreement(df, level="rubric")

    assert set(result["group"]) == {"faithfulness", "impartiality"}
    assert len(result) == 2


def test_model_rank_agreement_ie_group_holds_ie_values() -> None:
    """group column holds the IE names when level='ie'."""
    df = _multi_ie_model_df()
    result = model_rank_agreement(df, level="ie")

    assert all(g in {"baseline", "noise", "prior_conflict"} for g in result["group"])


def test_model_rank_agreement_rubric_group_holds_rubric_values() -> None:
    """group column holds rubric names when level='rubric'."""
    df = _multi_ie_model_df()
    result = model_rank_agreement(df, level="rubric")

    assert all(g in {"faithfulness", "impartiality"} for g in result["group"])


# ---------------------------------------------------------------------------
# country_delta
# ---------------------------------------------------------------------------

def _delta_df() -> pd.DataFrame:
    """Frame with 2 models × 2 IEs × 1 rubric × 2 countries at known rates."""
    rows: list[dict[str, object]] = []
    # model A: ie1 DE=0.9 NL=0.6, ie2 DE=0.7 NL=0.5
    rows += _rubric_rows(ie="ie1", rubric="R", n_pass=9, n_total=10, model="A", country="DE")
    rows += _rubric_rows(ie="ie1", rubric="R", n_pass=6, n_total=10, model="A", country="NL")
    rows += _rubric_rows(ie="ie2", rubric="R", n_pass=7, n_total=10, model="A", country="DE")
    rows += _rubric_rows(ie="ie2", rubric="R", n_pass=5, n_total=10, model="A", country="NL")
    # model B: ie1 DE=0.4 NL=0.8, ie2 DE=0.6 NL=0.3
    rows += _rubric_rows(ie="ie1", rubric="R", n_pass=4, n_total=10, model="B", country="DE")
    rows += _rubric_rows(ie="ie1", rubric="R", n_pass=8, n_total=10, model="B", country="NL")
    rows += _rubric_rows(ie="ie2", rubric="R", n_pass=6, n_total=10, model="B", country="DE")
    rows += _rubric_rows(ie="ie2", rubric="R", n_pass=3, n_total=10, model="B", country="NL")
    return pd.DataFrame(rows)


def test_country_delta_delta_value() -> None:
    """delta == NL_adherence - DE_adherence; negative = NL underperforms."""
    df = _delta_df()
    result = country_delta(df, by=["model", "ie"])

    # model A, ie1: NL=0.6, DE=0.9 → delta = 0.6 - 0.9 = -0.3 (NL underperforms)
    row_a_ie1 = result.loc[(result["model"] == "A") & (result["ie"] == "ie1")]
    assert float(row_a_ie1["delta"].iloc[0]) == pytest.approx(-0.3, abs=1e-9)

    # model B, ie1: NL=0.8, DE=0.4 → delta = 0.8 - 0.4 = +0.4 (NL outperforms)
    row_b_ie1 = result.loc[(result["model"] == "B") & (result["ie"] == "ie1")]
    assert float(row_b_ie1["delta"].iloc[0]) == pytest.approx(0.4, abs=1e-9)


def test_country_delta_abs_b_value() -> None:
    """abs_b == NL_adherence for each (model, ie) cell."""
    df = _delta_df()
    result = country_delta(df, by=["model", "ie"])

    # model A, ie2: abs_b = NL = 0.5
    row_a_ie2 = result.loc[(result["model"] == "A") & (result["ie"] == "ie2")]
    assert float(row_a_ie2["abs_b"].iloc[0]) == pytest.approx(0.5, abs=1e-9)

    # model B, ie2: abs_b = NL = 0.3
    row_b_ie2 = result.loc[(result["model"] == "B") & (result["ie"] == "ie2")]
    assert float(row_b_ie2["abs_b"].iloc[0]) == pytest.approx(0.3, abs=1e-9)


def test_country_delta_columns() -> None:
    """Result must have exactly the by columns + 'delta' + 'abs_b'."""
    df = _delta_df()
    result = country_delta(df, by=["model", "ie"])

    assert list(result.columns) == ["model", "ie", "delta", "abs_b"]


# ---------------------------------------------------------------------------
# load_tidy_multi
# ---------------------------------------------------------------------------

def test_load_tidy_multi_concatenates_rows() -> None:
    """Monkeypatched load_tidy: concatenated frame has sum of rows from both dirs."""
    rows_de = pd.DataFrame(
        [{"model": "m", "country": "DE", "ie": "baseline", "party": "p",
          "rubric": "faithfulness", "subquestion": "sq1", "passed": 1.0}]
    )
    rows_nl = pd.DataFrame(
        [
            {"model": "m", "country": "NL", "ie": "baseline", "party": "p",
             "rubric": "faithfulness", "subquestion": "sq1", "passed": 0.0},
            {"model": "m", "country": "NL", "ie": "noise", "party": "p",
             "rubric": "faithfulness", "subquestion": "sq1", "passed": 1.0},
        ]
    )
    side_effects = [rows_de, rows_nl]

    with patch("src.analysis.tidy.load_tidy", side_effect=side_effects):
        result = load_tidy_multi([Path("/fake/de"), Path("/fake/nl")])

    assert len(result) == 3  # 1 DE + 2 NL
    assert set(result["country"]) == {"DE", "NL"}


def test_load_tidy_multi_both_countries_present() -> None:
    """Both country codes must appear in the concatenated frame."""
    de = pd.DataFrame([{"country": "DE", "model": "m"}])
    nl = pd.DataFrame([{"country": "NL", "model": "m"}])

    with patch("src.analysis.tidy.load_tidy", side_effect=[de, nl]):
        result = load_tidy_multi([Path("/fake/de"), Path("/fake/nl")])

    assert "DE" in result["country"].values
    assert "NL" in result["country"].values
