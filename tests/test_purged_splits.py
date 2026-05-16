"""Tests for purge+embargo time-series splits."""

import unittest

import numpy as np
import pandas as pd

from projection.ml_model.splits import (
    chronological_holdout_split,
    gap_timedelta,
    grouped_purged_time_series_splits,
    purged_time_series_splits,
)


class TestPurgedSplits(unittest.TestCase):
    def test_gap_grows_with_horizon(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        g5 = gap_timedelta(5, dates)
        g120 = gap_timedelta(120, dates)
        self.assertLess(g5, g120)

    def test_purged_splits_no_overlap(self):
        dates = pd.Series(pd.date_range("2018-01-01", periods=800, freq="B"))
        for train_idx, val_idx in purged_time_series_splits(dates, n_splits=3, horizon_days=20):
            self.assertTrue(train_idx.max() < val_idx.min())

    def test_holdout_has_gap(self):
        sub = pd.DataFrame({
            "date": pd.date_range("2019-01-01", periods=400, freq="B"),
            "x": np.random.randn(400),
        })
        tr, va = chronological_holdout_split(sub, val_fraction=0.2, horizon_days=5)
        if not va.empty and not tr.empty:
            self.assertLess(tr["date"].max(), va["date"].min())

    def test_grouped_purge_blocks_in_group_leakage(self):
        # Build a panel of 4 tickers x 250 dates so any cross-time fold
        # would otherwise put the same ticker in train and val across the
        # gap. The grouped splitter must drop those training rows.
        n_dates = 250
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        rows = []
        for tk in tickers:
            for d in pd.date_range("2018-01-01", periods=n_dates, freq="B"):
                rows.append({"date": d, "ticker": tk})
        df = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)

        any_yielded = False
        for train_idx, val_idx in grouped_purged_time_series_splits(
            df["date"], df["ticker"], n_splits=3, horizon_days=20,
            embargo_in_group_days=30,
        ):
            any_yielded = True
            v_groups_dates: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
            for i in val_idx:
                tk = df.iloc[i]["ticker"]
                d = df.iloc[i]["date"]
                lo, hi = v_groups_dates.get(tk, (d, d))
                v_groups_dates[tk] = (min(lo, d), max(hi, d))
            for j in train_idx:
                tk = df.iloc[j]["ticker"]
                d = df.iloc[j]["date"]
                if tk in v_groups_dates:
                    lo, hi = v_groups_dates[tk]
                    self.assertFalse(
                        lo - pd.Timedelta(days=30) <= d <= hi + pd.Timedelta(days=30),
                        f"in-group leakage: ticker={tk} train={d} val=[{lo}..{hi}]",
                    )
        self.assertTrue(any_yielded)


if __name__ == "__main__":
    unittest.main()
