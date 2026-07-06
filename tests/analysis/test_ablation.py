"""Unit tests for src.analysis.ablation (paired_delta, holm, bootstrap_ci).

All tests use hand-built frames or arrays so that expected values can be
computed by inspection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.ablation import bootstrap_ci, holm, paired_delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tidy(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a minimal tidy frame from a list of row dicts."""
    return pd.DataFrame(rows)


def _pair_row(
    *,
    ie: str,
    item_id: str,
    subquestion: str,
    passed_full: float,
    passed_other: float,
    model: str = "m1",
    country: str = "DE",
    party: str = "SPD",
) -> tuple[dict[str, object], dict[str, object]]:
    """Return a (full_row, other_row) pair sharing all item-pair keys."""
    base = {
        "model": model,
        "country": country,
        "ie": ie,
        "party": party,
        "item_id": item_id,
        "subquestion": subquestion,
        # Extra tidy columns — present but not used by paired_delta.
        "rubric": "faithfulness",
        "observation_id": f"{item_id}__{subquestion}",
    }
    full_row = {**base, "passed": passed_full}
    other_row = {**base, "passed": passed_other}
    return full_row, other_row


# ---------------------------------------------------------------------------
# paired_delta tests
# ---------------------------------------------------------------------------

class TestPairedDelta:
    """Tests for :func:`paired_delta`."""

    def _two_item_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Two items, one IE.

        Item s001 / sq1: full=1, other=0  → b pair (full passed, other failed)
        Item s002 / sq1: full=0, other=1  → c pair (full failed, other passed)

        Within subquestion="sq1", country="DE", party="SPD":
          rate_full  = 0.5, rate_other = 0.5, delta_rate = 0.0
          b = 1, c = 1  →  b+c=2, p_mcnemar = binomtest(1, 2, 0.5) = 1.0
        """
        full_rows, other_rows = [], []
        for item_id, pf, po in [("s001", 1.0, 0.0), ("s002", 0.0, 1.0)]:
            fr, or_ = _pair_row(
                ie="baseline",
                item_id=item_id,
                subquestion="sq1",
                passed_full=pf,
                passed_other=po,
            )
            full_rows.append(fr)
            other_rows.append(or_)
        return _make_tidy(full_rows), _make_tidy(other_rows)

    def test_delta_rate_and_counts_by_hand(self) -> None:
        """Verify delta_rate, b, c computed correctly on a 2-item frame."""
        full, other = self._two_item_frames()

        result = paired_delta(
            full,
            other,
            ie="baseline",
            by=("country", "party", "subquestion"),
        )

        assert len(result) == 1
        row = result.iloc[0]
        assert float(row["rate_full"]) == pytest.approx(0.5, abs=1e-9)
        assert float(row["rate_other"]) == pytest.approx(0.5, abs=1e-9)
        assert float(row["delta_rate"]) == pytest.approx(0.0, abs=1e-9)
        assert int(row["n_pairs"]) == 2
        assert int(row["b"]) == 1
        assert int(row["c"]) == 1

    def test_p_mcnemar_is_1_when_no_discordant_pairs(self) -> None:
        """b + c == 0 → p_mcnemar == 1.0 (concordant outcomes, no evidence)."""
        # Both items concordant: full=1/other=1 and full=0/other=0.
        full_rows, other_rows = [], []
        for item_id, p in [("s001", 1.0), ("s002", 0.0)]:
            fr, or_ = _pair_row(
                ie="baseline",
                item_id=item_id,
                subquestion="sq1",
                passed_full=p,
                passed_other=p,  # same outcome
            )
            full_rows.append(fr)
            other_rows.append(or_)
        full = _make_tidy(full_rows)
        other = _make_tidy(other_rows)

        result = paired_delta(full, other, ie="baseline", by=("country", "party", "subquestion"))

        assert float(result.iloc[0]["p_mcnemar"]) == pytest.approx(1.0, abs=1e-9)
        assert int(result.iloc[0]["b"]) == 0
        assert int(result.iloc[0]["c"]) == 0

    def test_p_mcnemar_decreases_as_discordance_becomes_one_sided(self) -> None:
        """More one-sided discordance → smaller p_mcnemar.

        Balanced: b=1, c=1  → p=1.0
        One-sided light: b=4, c=0  → p=0.125
        One-sided heavy: b=10, c=0  → p≈0.002
        """
        def _make_frames(
            b_count: int, c_count: int
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            full_rows: list[dict[str, object]] = []
            other_rows: list[dict[str, object]] = []
            idx = 0
            # b pairs: full=1, other=0
            for _ in range(b_count):
                fr, or_ = _pair_row(
                    ie="baseline",
                    item_id=f"s{idx:03d}",
                    subquestion="sq1",
                    passed_full=1.0,
                    passed_other=0.0,
                )
                full_rows.append(fr)
                other_rows.append(or_)
                idx += 1
            # c pairs: full=0, other=1
            for _ in range(c_count):
                fr, or_ = _pair_row(
                    ie="baseline",
                    item_id=f"s{idx:03d}",
                    subquestion="sq1",
                    passed_full=0.0,
                    passed_other=1.0,
                )
                full_rows.append(fr)
                other_rows.append(or_)
                idx += 1
            return _make_tidy(full_rows), _make_tidy(other_rows)

        by = ("country", "party", "subquestion")

        p_balanced = float(
            paired_delta(*_make_frames(1, 1), ie="baseline", by=by).iloc[0]["p_mcnemar"]
        )
        p_light = float(
            paired_delta(*_make_frames(4, 0), ie="baseline", by=by).iloc[0]["p_mcnemar"]
        )
        p_heavy = float(
            paired_delta(*_make_frames(10, 0), ie="baseline", by=by).iloc[0]["p_mcnemar"]
        )

        # Balanced discordance: exact binomtest(1,2,0.5) = 1.0
        assert p_balanced == pytest.approx(1.0, abs=1e-9)
        # One-sided: exact binomtest(4,4,0.5, two-sided) = 0.125
        assert p_light == pytest.approx(0.125, abs=1e-9)
        # More one-sided → p still smaller
        assert p_heavy < p_light < p_balanced

    def test_raises_on_empty_ie_filter(self) -> None:
        """ValueError raised when either frame has no rows for the requested ie."""
        fr_baseline, or_baseline = _pair_row(
            ie="baseline", item_id="s001", subquestion="sq1",
            passed_full=1.0, passed_other=0.0,
        )
        fr_noise, or_noise = _pair_row(
            ie="noise", item_id="s001", subquestion="sq1",
            passed_full=1.0, passed_other=0.0,
        )
        # full has "baseline" only; other has both → filter full to "noise" raises.
        full = _make_tidy([fr_baseline])
        other = _make_tidy([or_baseline, or_noise])

        with pytest.raises(ValueError, match="No rows in `full`"):
            paired_delta(full, other, ie="noise")

        # full has both; other has "baseline" only → filter other to "noise" raises.
        full2 = _make_tidy([fr_baseline, fr_noise])
        other2 = _make_tidy([or_baseline])

        with pytest.raises(ValueError, match="No rows in `other`"):
            paired_delta(full2, other2, ie="noise")

    def test_multi_group_sort_order(self) -> None:
        """Output is sorted by the ``by`` columns."""
        full_rows, other_rows = [], []
        # Two sub-questions in two parties.
        for party in ["CDU", "SPD"]:
            for sq in ["sq2", "sq1"]:
                fr, or_ = _pair_row(
                    ie="baseline", item_id="s001", subquestion=sq,
                    passed_full=1.0, passed_other=0.0, party=party,
                )
                full_rows.append(fr)
                other_rows.append(or_)
        full = _make_tidy(full_rows)
        other = _make_tidy(other_rows)

        result = paired_delta(full, other, ie="baseline", by=("party", "subquestion"))

        parties = result["party"].tolist()
        sqs = result["subquestion"].tolist()
        assert parties == sorted(parties)
        assert sqs == ["sq1", "sq2", "sq1", "sq2"]


# ---------------------------------------------------------------------------
# holm tests
# ---------------------------------------------------------------------------

class TestHolm:
    """Tests for :func:`holm`."""

    def test_three_pvalue_worked_example(self) -> None:
        """Hand-worked 3-p-value example.

        Input (sorted): p=[0.01, 0.02, 0.03], m=3
        Multiplied:     [3*0.01, 2*0.02, 1*0.03] = [0.03, 0.04, 0.03]
        Running max:    [0.03, 0.04, 0.04]   (clip to 1 is a no-op here)
        """
        pvals = pd.Series([0.01, 0.02, 0.03], index=[10, 20, 30])
        result = holm(pvals)

        # Values returned in original input order.
        assert list(result.index) == [10, 20, 30]
        assert float(result.iloc[0]) == pytest.approx(0.03, abs=1e-9)
        assert float(result.iloc[1]) == pytest.approx(0.04, abs=1e-9)
        assert float(result.iloc[2]) == pytest.approx(0.04, abs=1e-9)

    def test_output_is_monotone_non_decreasing(self) -> None:
        """Adjusted p-values sorted by the original p-rank are non-decreasing."""
        raw = pd.Series([0.001, 0.04, 0.012, 0.2, 0.003])
        adjusted = holm(raw)

        # Sort by original p-value to get rank order, then check monotonicity.
        paired = sorted(zip(raw.tolist(), adjusted.tolist()))
        adj_sorted = [a for _, a in paired]
        for i in range(len(adj_sorted) - 1):
            assert adj_sorted[i] <= adj_sorted[i + 1]

    def test_nan_passthrough_and_excluded_from_count(self) -> None:
        """NaN values are returned as NaN; m is the non-NaN count."""
        # 2 real p-values + 1 NaN.
        pvals = pd.Series([0.01, float("nan"), 0.02])
        result = holm(pvals)

        assert np.isnan(float(result.iloc[1]))
        # With m=2: adjusted[0] = 2*0.01 = 0.02, adjusted[2] = 1*0.02 = 0.02;
        # running max of [0.02, 0.02] = [0.02, 0.02].
        assert float(result.iloc[0]) == pytest.approx(0.02, abs=1e-9)
        assert float(result.iloc[2]) == pytest.approx(0.02, abs=1e-9)

    def test_all_nan_returns_unchanged(self) -> None:
        """All-NaN input is returned as-is without error."""
        pvals = pd.Series([float("nan"), float("nan")])
        result = holm(pvals)
        assert all(np.isnan(v) for v in result)

    def test_clip_to_1(self) -> None:
        """Adjusted values are capped at 1.0."""
        # Large p-values force adjusted > 1 before clipping.
        pvals = pd.Series([0.6, 0.7, 0.8])
        result = holm(pvals)
        assert all(v <= 1.0 + 1e-12 for v in result)

    def test_original_index_preserved(self) -> None:
        """Output index matches the input index exactly."""
        idx = ["b", "a", "c"]
        pvals = pd.Series([0.05, 0.01, 0.03], index=idx)
        result = holm(pvals)
        assert list(result.index) == idx


# ---------------------------------------------------------------------------
# bootstrap_ci tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Tests for :func:`bootstrap_ci`."""

    def test_ci_brackets_sample_mean(self) -> None:
        """The CI must contain the sample mean for a stable binary array."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 2, size=200).astype(float)
        sample_mean = float(arr.mean())

        lower, upper = bootstrap_ci(arr, n_boot=2000, seed=0, alpha=0.05)

        assert lower < sample_mean < upper

    def test_deterministic_given_seed(self) -> None:
        """Same seed → identical CI; the function is deterministic given seed."""
        arr = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=float)

        lo1, hi1 = bootstrap_ci(arr, seed=7)
        lo2, hi2 = bootstrap_ci(arr, seed=7)

        assert lo1 == lo2
        assert hi1 == hi2

        # Verify seed is actually used: changing n_boot shifts the distribution.
        lo3, hi3 = bootstrap_ci(arr, seed=7, n_boot=3)
        # 3-resample bootstrap over a 10-element discrete array almost certainly
        # yields a narrower (noisier) range than 2000 resamples.
        assert (lo3, hi3) != (lo1, hi1)

    def test_empty_returns_nan(self) -> None:
        """Empty array → (nan, nan)."""
        lo, hi = bootstrap_ci(np.array([], dtype=float))
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_single_element_returns_value_twice(self) -> None:
        """Single element → (value, value)."""
        lo, hi = bootstrap_ci(np.array([0.75]))
        assert lo == pytest.approx(0.75, abs=1e-9)
        assert hi == pytest.approx(0.75, abs=1e-9)

    def test_all_ones_ci_is_1_1(self) -> None:
        """All-pass array → CI is (1.0, 1.0)."""
        arr = np.ones(50, dtype=float)
        lo, hi = bootstrap_ci(arr, n_boot=500, seed=0)
        assert lo == pytest.approx(1.0, abs=1e-9)
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_all_zeros_ci_is_0_0(self) -> None:
        """All-fail array → CI is (0.0, 0.0)."""
        arr = np.zeros(50, dtype=float)
        lo, hi = bootstrap_ci(arr, n_boot=500, seed=0)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi == pytest.approx(0.0, abs=1e-9)

    def test_lower_less_than_upper(self) -> None:
        """CI lower bound must be strictly below upper for a mixed array."""
        arr = np.array([1, 0] * 30, dtype=float)
        lo, hi = bootstrap_ci(arr, n_boot=2000, seed=0)
        assert lo < hi
