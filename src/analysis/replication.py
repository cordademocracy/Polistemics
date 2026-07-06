"""Cross-country (replication) statistics over the tidy scores table."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import spearmanr

from src.analysis.aggregate import adherence_index, geometric_benchmark_score
from src.analysis.tidy import TidyColumns


@dataclass(frozen=True)
class RankCorrelation:
    """Spearman rank correlation result with the two underlying orderings.

    p is None when the correlation is reported descriptively (e.g. n=3 models),
    where a p-value would be misleading.
    """

    rho: float
    p: float | None
    n: int
    order_a: list[str]  # items (IEs or models) best→worst for country_a
    order_b: list[str]  # same for country_b


def _spearman(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    report_p: bool,
) -> RankCorrelation:
    """Compute Spearman rho on the shared index of two Series.

    Aligns on the intersection of both index sets, then runs ``spearmanr``.
    Orderings are the index sorted by value descending (highest/best first).

    Args:
        series_a: Scores indexed by item name (IE or model) for country A.
        series_b: Scores indexed by item name (IE or model) for country B.
        report_p: If True, include the p-value; if False, set ``p=None``
            (appropriate when n is too small for a meaningful p).

    Returns:
        A :class:`RankCorrelation` dataclass.
    """
    shared = series_a.index.intersection(series_b.index)
    a = series_a.loc[shared]
    b = series_b.loc[shared]
    n = len(shared)

    stat, pvalue = spearmanr(a.values, b.values)
    rho = float(stat)
    p = float(pvalue) if report_p else None

    order_a = list(a.sort_values(ascending=False).index)
    order_b = list(b.sort_values(ascending=False).index)

    return RankCorrelation(rho=rho, p=p, n=n, order_a=order_a, order_b=order_b)


def ie_difficulty(
    df: pd.DataFrame,
    by: tuple[str, ...] = ("country", "ie"),
) -> pd.DataFrame:
    """Per-IE difficulty: mean across models of the per-(model,country,ie) adherence.

    Difficulty is computed as the adherence index grouped by
    ``["model","country","ie"]``, then averaged over models within each
    ``by`` group. The result column is named ``"difficulty"``.

    Args:
        df: Tidy scores table.
        by: Grouping columns for the final difficulty frame (default
            ``("country", "ie")``).

    Returns:
        DataFrame with the ``by`` columns plus a ``"difficulty"`` column (float, 0-1).
    """
    per_model = adherence_index(
        df, by=[TidyColumns.MODEL, TidyColumns.COUNTRY, TidyColumns.IE]
    )
    result = (
        per_model.groupby(list(by), observed=True)["adherence"]
        .mean()
        .reset_index()
        .rename(columns={"adherence": "difficulty"})
    )
    return result


def ie_difficulty_correlation(
    df: pd.DataFrame,
    country_a: str = "DE",
    country_b: str = "NL",
) -> RankCorrelation:
    """Spearman rank correlation of IE difficulty between two countries.

    Uses :func:`ie_difficulty` at ``("country","ie")`` grain and aligns on the
    common IE set (n=6 expected). p-value IS reported here — n=6 is sufficient
    to be informative.

    Args:
        df: Tidy scores table.
        country_a: ISO code for country A (default ``"DE"``).
        country_b: ISO code for country B (default ``"NL"``).

    Returns:
        A :class:`RankCorrelation` with ``rho``, ``p``, ``n`` and orderings
        (IEs sorted easiest→hardest, i.e. highest difficulty first).
    """
    diff = ie_difficulty(df, by=(TidyColumns.COUNTRY, TidyColumns.IE))

    a = diff.loc[diff[TidyColumns.COUNTRY] == country_a].set_index(TidyColumns.IE)[
        "difficulty"
    ]
    b = diff.loc[diff[TidyColumns.COUNTRY] == country_b].set_index(TidyColumns.IE)[
        "difficulty"
    ]

    return _spearman(a, b, report_p=True)


def model_rank_agreement(
    df: pd.DataFrame,
    level: str = "overall",
    country_a: str = "DE",
    country_b: str = "NL",
) -> pd.DataFrame:
    """Descriptive Spearman rho on model ordering between two countries.

    p is NOT reported (set to None) — model counts are typically too small for
    a meaningful p-value. One row per group; uniform schema:
    ``["group", "rho", "n", "order_a", "order_b"]``.

    Args:
        df: Tidy scores table.
        level: Granularity — ``"overall"`` (single row), ``"ie"`` (one row per
            IE), or ``"rubric"`` (one row per rubric).
        country_a: ISO code for country A (default ``"DE"``).
        country_b: ISO code for country B (default ``"NL"``).

    Returns:
        DataFrame with columns ``["group", "rho", "n", "order_a", "order_b"]``.
        ``group`` is ``"overall"`` / IE value / rubric value. ``order_a`` and
        ``order_b`` are lists of models sorted by score descending (best first).
        Groups with fewer than 2 shared models still emit a row with
        ``rho=float("nan")``.
    """
    rows: list[dict[str, object]] = []

    if level == "overall":
        scores = geometric_benchmark_score(df, by=[TidyColumns.MODEL, TidyColumns.COUNTRY])
        score_col = "benchmark_score"
        groups = [("overall", scores)]
    elif level == "ie":
        scores = adherence_index(
            df, by=[TidyColumns.MODEL, TidyColumns.COUNTRY, TidyColumns.IE]
        )
        score_col = "adherence"
        groups = [
            (ie_val, g)
            for ie_val, g in scores.groupby(TidyColumns.IE, observed=True)
        ]
    elif level == "rubric":
        scores = adherence_index(
            df, by=[TidyColumns.MODEL, TidyColumns.COUNTRY, TidyColumns.RUBRIC]
        )
        score_col = "adherence"
        groups = [
            (rub_val, g)
            for rub_val, g in scores.groupby(TidyColumns.RUBRIC, observed=True)
        ]
    else:
        raise ValueError(f"level must be 'overall', 'ie', or 'rubric'; got {level!r}")

    for group_val, group_df in groups:
        a_series = (
            group_df.loc[group_df[TidyColumns.COUNTRY] == country_a]
            .set_index(TidyColumns.MODEL)[score_col]
        )
        b_series = (
            group_df.loc[group_df[TidyColumns.COUNTRY] == country_b]
            .set_index(TidyColumns.MODEL)[score_col]
        )

        shared = a_series.index.intersection(b_series.index)
        n = len(shared)

        if n < 2:
            order_a = list(a_series.sort_values(ascending=False).index)
            order_b = list(b_series.sort_values(ascending=False).index)
            rows.append(
                {
                    "group": group_val,
                    "rho": float("nan"),
                    "n": n,
                    "order_a": order_a,
                    "order_b": order_b,
                }
            )
            continue

        rc = _spearman(a_series, b_series, report_p=False)
        rows.append(
            {
                "group": group_val,
                "rho": rc.rho,
                "n": rc.n,
                "order_a": rc.order_a,
                "order_b": rc.order_b,
            }
        )

    return pd.DataFrame(rows, columns=["group", "rho", "n", "order_a", "order_b"])


def country_delta(
    df: pd.DataFrame,
    by: tuple[str, ...] | list[str],
    value: str = "adherence",
    country_a: str = "DE",
    country_b: str = "NL",
) -> pd.DataFrame:
    """Signed delta (country_b − country_a) at the requested granularity.

    Returns the delta (NL − DE by default, so negative = NL underperforms) and
    the country_b absolute value for delta heatmaps. Implemented via
    :func:`adherence_index` at ``by + ["country"]`` grain, then a pivot.

    Args:
        df: Tidy scores table.
        by: Grouping columns for the delta (e.g. ``["model","ie"]`` or
            ``["model","rubric"]``). A list or tuple; ``"country"`` must NOT
            be in ``by`` already.
        value: Reserved; only ``"adherence"`` is supported.
        country_a: ISO code for the reference country (default ``"DE"``).
        country_b: ISO code for the comparison country (default ``"NL"``).

    Returns:
        DataFrame with the ``by`` columns plus ``"delta"``
        (country_b adherence − country_a adherence; negative = country_b
        underperforms) and ``"abs_b"`` (country_b adherence).
    """
    assert value == "adherence", f"Only value='adherence' is supported; got {value!r}"

    by_list = list(by)
    adh = adherence_index(df, by=by_list + [TidyColumns.COUNTRY])

    pivoted = adh.pivot_table(
        index=by_list,
        columns=TidyColumns.COUNTRY,
        values="adherence",
    ).reset_index()
    # Remove the column-level name added by pivot_table.
    pivoted.columns.name = None

    pivoted["delta"] = pivoted[country_b] - pivoted[country_a]
    pivoted["abs_b"] = pivoted[country_b]

    return pivoted[by_list + ["delta", "abs_b"]]
