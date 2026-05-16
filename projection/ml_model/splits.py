"""
Time-series splits with purge + embargo gaps (Lopez de Prado style).

Provides three primitives:

* ``purged_time_series_splits`` — vanilla purge+embargo on a single date
  series. Adequate when feature autocorrelation between rows is weak.

* ``grouped_purged_time_series_splits`` — same purge+embargo, but with the
  additional constraint that **rows belonging to the same group (e.g.
  ticker) cannot appear in both train and validation across the gap**.
  This is the version institutional teams use on cross-sectional panels;
  technical features (MAs, vol, momentum) are heavily autocorrelated
  along a single ticker, so a 5-day calendar purge is *not* enough — you
  also need to keep AAPL out of train if AAPL is in val.

* ``chronological_holdout_split`` — single train/val cut with the same
  purge+embargo. Used for the Optuna tuning fold.

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


def grouped_purged_time_series_splits(
    dates: pd.Series,
    groups: pd.Series | np.ndarray,
    n_splits: int = 5,
    horizon_days: int = 5,
    *,
    embargo_in_group_days: int | None = None,
):
    """
    Purge+embargo CV with **per-group** leakage control.

    Two leakage paths are blocked:

    1. Cross-time leakage (the standard purge): a training row's date must
       sit at least ``gap`` calendar days before the earliest validation
       date.
    2. Cross-group leakage (the new constraint): if a group (e.g. ticker)
       has a row in validation at time ``t``, all rows for that same group
       within ``[t - in_group_gap, t + in_group_gap]`` are removed from
       the training set. This stops AAPL on Monday from training the model
       that's evaluated on AAPL on Wednesday.

    Args
    ----
    dates : sortable date column aligned to ``groups`` and to X/y rows.
    groups : aligned label per row identifying the autocorrelated unit
             (typically the ticker). Must be hashable.
    n_splits, horizon_days : as in ``purged_time_series_splits``.
    embargo_in_group_days : per-group dwell time. Defaults to
                            ``horizon_days + 5`` — long enough to cover
                            label horizon and label-leak from rolling
                            features.

    Yields
    ------
    (train_idx, val_idx) : positional integer indices.
    """
    n = len(dates)
    if n < 200 or n_splits < 2:
        return

    dt = pd.to_datetime(dates).reset_index(drop=True)
    grp = pd.Series(groups).reset_index(drop=True)
    if len(grp) != n:
        raise ValueError(f"len(groups)={len(grp)} != len(dates)={n}")

    gap = gap_timedelta(horizon_days, dt)
    in_group_gap = pd.Timedelta(
        days=int(embargo_in_group_days if embargo_in_group_days is not None else horizon_days + 5)
    )
    fold_size = n // (n_splits + 1)

    grp_idx_by_value: dict = {}
    for i, g in enumerate(grp.values):
        grp_idx_by_value.setdefault(g, []).append(i)

    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        val_start_time = dt.iloc[train_end - 1] + gap
        val_end = min(n, fold_size * (k + 1))

        # Initial calendar-purged candidate sets.
        train_mask = np.arange(n) < train_end
        val_mask = (dt >= val_start_time) & (np.arange(n) < val_end)

        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        # For every group with at least one validation row, drop training
        # rows for that group that fall within ``in_group_gap`` of any
        # validation row's date.
        val_groups = set(grp.values[val_mask])
        for g in val_groups:
            idxs_g = grp_idx_by_value.get(g, [])
            if not idxs_g:
                continue
            # Validation timestamps for this group.
            v_dates = dt.iloc[[i for i in idxs_g if val_mask[i]]]
            if v_dates.empty:
                continue
            v_min = v_dates.min() - in_group_gap
            v_max = v_dates.max() + in_group_gap
            for i in idxs_g:
                if not train_mask[i]:
                    continue
                if v_min <= dt.iloc[i] <= v_max:
                    train_mask[i] = False

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
