"""Paired-condition ablation utilities for VAA-Bench evaluation.

Compares ``full`` vs ablation conditions (e.g. ``anon``, ``en``) where pairs
are matched at the item level on ``(country, model, party, item_id,
subquestion)``.  All functions are pure: DataFrame/array in, result out.
No I/O, no side effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.analysis.tidy import TidyColumns

# Merge keys that uniquely identify a paired item across conditions.
_PAIR_KEYS: tuple[str, ...] = (
    TidyColumns.COUNTRY,
    TidyColumns.MODEL,
    TidyColumns.PARTY,
    TidyColumns.ITEM_ID,
    TidyColumns.SUBQUESTION,
)

# Column name suffixes applied to ``passed`` after the inner merge.
_SUFFIX_FULL: str = "_full"
_SUFFIX_OTHER: str = "_other"

_COL_PASSED_FULL: str = TidyColumns.PASSED + _SUFFIX_FULL
_COL_PASSED_OTHER: str = TidyColumns.PASSED + _SUFFIX_OTHER


def paired_delta(
    full: pd.DataFrame,
    other: pd.DataFrame,
    *,
    ie: str,
    by: tuple[str, ...] = ("country", "model", "party", "subquestion"),
) -> pd.DataFrame:
    """Compute per-group paired deltas between a full and an ablation condition.

    Filters both frames to a single ``ie``, inner-joins on item-pair keys to
    align rows, then for every ``by`` group returns the pass-rate difference,
    pair count, McNemar discordant-pair counts, and exact McNemar p-value.

    Args:
        full: Tidy scores table for the full condition (native language, real
            party labels).
        other: Tidy scores table for the ablation condition (e.g. ``anon``,
            ``en``).
        ie: Inspection environment to filter to (e.g. ``"baseline"``).
        by: Grouping columns for the output (must be a subset of the merge
            keys or derivable from them).

    Returns:
        Reset-index DataFrame with columns ``[*by, "rate_full", "rate_other",
        "delta_rate", "n_pairs", "b", "c", "p_mcnemar"]``, sorted by ``by``.

    Raises:
        ValueError: If either frame is empty after filtering to ``ie``.
    """
    full_ie = full[full[TidyColumns.IE] == ie]
    other_ie = other[other[TidyColumns.IE] == ie]

    if full_ie.empty:
        raise ValueError(f"No rows in `full` for ie={ie!r}")
    if other_ie.empty:
        raise ValueError(f"No rows in `other` for ie={ie!r}")

    merged = full_ie.merge(
        other_ie,
        on=list(_PAIR_KEYS),
        how="inner",
        suffixes=(_SUFFIX_FULL, _SUFFIX_OTHER),
    )

    def _group_stats(grp: pd.DataFrame) -> pd.Series:
        p_full = grp[_COL_PASSED_FULL].to_numpy()
        p_other = grp[_COL_PASSED_OTHER].to_numpy()

        rate_full = float(p_full.mean())
        rate_other = float(p_other.mean())
        n_pairs = int(len(grp))

        # Discordant pairs for McNemar: full=1/other=0 (b) and full=0/other=1 (c).
        b = int(((p_full == 1) & (p_other == 0)).sum())
        c = int(((p_full == 0) & (p_other == 1)).sum())

        # Exact McNemar via binomtest; if no discordant pairs → p=1.0.
        if b + c == 0:
            p_mcnemar = 1.0
        else:
            p_mcnemar = float(
                binomtest(b, n=b + c, p=0.5, alternative="two-sided").pvalue
            )

        return pd.Series(
            {
                "rate_full": rate_full,
                "rate_other": rate_other,
                "delta_rate": rate_other - rate_full,
                "n_pairs": n_pairs,
                "b": b,
                "c": c,
                "p_mcnemar": p_mcnemar,
            }
        )

    result = (
        merged.groupby(list(by), observed=True)
        .apply(_group_stats, include_groups=False)
        .reset_index()
        .sort_values(list(by))
        .reset_index(drop=True)
    )
    # Ensure integer dtype for count columns.
    result["n_pairs"] = result["n_pairs"].astype(int)
    result["b"] = result["b"].astype(int)
    result["c"] = result["c"].astype(int)
    return result


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm–Bonferroni step-down adjusted p-values.

    Implements the standard Holm (1979) algorithm directly with numpy.
    NaN values are passed through unchanged and excluded from the correction
    count ``m``.

    Algorithm (on non-NaN subset, 1-indexed ranks):
        1. Sort p ascending.
        2. Adjusted_k = (m − k + 1) · p_k.
        3. Enforce monotone non-decreasing (running max from left).
        4. Clip to 1.0.
        5. Scatter back to the original index order.

    Args:
        pvalues: p-values to adjust (any index).

    Returns:
        Series of adjusted p-values with the same index and order as the
        input.
    """
    original_index = pvalues.index
    arr = pvalues.to_numpy(dtype=float, copy=True)

    nan_mask = np.isnan(arr)
    valid_pos = np.where(~nan_mask)[0]

    if valid_pos.size == 0:
        return pvalues.copy()

    m = int(valid_pos.size)
    valid_p = arr[valid_pos]

    # Sort ascending; ranks are 1-indexed so multiplier at rank k is (m - k + 1).
    sort_order = np.argsort(valid_p, kind="stable")
    sorted_p = valid_p[sort_order]
    multipliers = m - np.arange(m)  # m, m-1, ..., 1
    adjusted = multipliers * sorted_p

    # Enforce monotone non-decreasing (step-down running max).
    np.maximum.accumulate(adjusted, out=adjusted)
    np.clip(adjusted, a_min=None, a_max=1.0, out=adjusted)

    # Scatter back: sort_order[i] = original valid index → adjusted[i] goes there.
    result_valid = np.empty(m, dtype=float)
    result_valid[sort_order] = adjusted
    arr[valid_pos] = result_valid

    return pd.Series(arr, index=original_index)


def bootstrap_ci(
    passed: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of a binary array.

    Args:
        passed: 1-D array of binary (0/1) values.
        n_boot: Number of bootstrap resamples (default 2000).
        seed: Random seed for ``np.random.default_rng`` (default 0).
        alpha: Significance level; CI spans ``[alpha/2, 1−alpha/2]`` percentiles
            (default 0.05 → 95 % CI).

    Returns:
        ``(lower, upper)`` percentile-bootstrap CI bounds.
        Returns ``(nan, nan)`` for an empty array; ``(value, value)`` for a
        single-element array.
    """
    arr = np.asarray(passed, dtype=float)

    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        v = float(arr[0])
        return (v, v)

    rng = np.random.default_rng(seed)
    # Resample with replacement: shape (n_boot, n).
    indices = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[indices].mean(axis=1)

    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)
