"""
Time-series splits with purge + embargo gaps (López de Prado style).

Adapted from StockMarketTool clean_analysis.Model.split_data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gap_timedelta(
    horizon_days: int,
    unique_dates: pd.Series | np.ndarray,
    *,
    embargo_fraction: float = 0.01,
    min_embargo_days: int = 5,
    weekend_buffer_days: int = 2,
) -> pd.Timedelta:
    """Purge (label horizon) + embargo (% of calendar) between train and validation."""
    ud = pd.Series(unique_dates) if not isinstance(unique_dates, pd.Series) else unique_dates
    n_unique = int(ud.nunique())
    embargo_days = max(min_embargo_days, int(n_unique * embargo_fraction))
    total = int(horizon_days) + embargo_days + weekend_buffer_days
    return pd.Timedelta(days=total)


def purged_time_series_splits(
    dates: pd.Series,
    n_splits: int = 5,
    horizon_days: int = 5,
):
    """
    Yield (train_idx, val_idx) positional indices with gaps after each train block.

    dates must be sortable (same order as X/y rows after sort).
    """
    n = len(dates)
    if n < 200 or n_splits < 2:
        return

    dt = pd.to_datetime(dates).reset_index(drop=True)
    gap = gap_timedelta(horizon_days, dt)
    fold_size = n // (n_splits + 1)

    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        val_start_time = dt.iloc[train_end - 1] + gap
        val_end = min(n, fold_size * (k + 1))

        train_mask = np.arange(n) < train_end
        val_mask = (dt >= val_start_time) & (np.arange(n) < val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) < 50 or len(val_idx) < 20:
            continue
        yield train_idx, val_idx


def chronological_holdout_split(
    sub: pd.DataFrame,
    val_fraction: float = 0.15,
    horizon_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train / validation split with purge+embargo before validation start."""
    sub = sub.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(sub["date"].unique())
    if len(unique_dates) < 30:
        return sub, sub.iloc[0:0]

    cut = int(len(unique_dates) * (1.0 - val_fraction))
    train_cutoff = unique_dates[max(0, cut - 1)]
    gap = gap_timedelta(horizon_days, unique_dates)
    val_start = train_cutoff + gap

    train_df = sub[sub["date"] < train_cutoff]
    val_df = sub[sub["date"] >= val_start]
    return train_df, val_df
