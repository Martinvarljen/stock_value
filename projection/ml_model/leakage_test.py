"""Permutation target test — detect feature leakage before trusting a model.

Ported from StockMarketTool ``clean_analysis.Model.run_permutation_test``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class LeakageTestResult:
    passed: bool
    iterations: list[float]
    threshold: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "iterations": self.iterations,
            "threshold": self.threshold,
            "message": self.message,
        }


def run_permutation_leakage_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_repeats: int = 5,
    ic_threshold: float = 0.02,
    num_boost_round: int = 50,
    seed: int = 42,
) -> LeakageTestResult:
    """Shuffle training labels; any |IC| above threshold suggests leakage."""
    if len(X_train) < 50 or len(X_test) < 20:
        return LeakageTestResult(
            passed=True,
            iterations=[],
            threshold=ic_threshold,
            message="Skipped — insufficient rows for permutation test.",
        )

    original_y = y_train.to_numpy()
    ics: list[float] = []
    leakage = False
    y_test_arr = y_test.to_numpy()

    for i in range(n_repeats):
        y_shuf = np.random.default_rng(seed + i).permutation(original_y)
        params = {
            "objective": "binary",
            "verbosity": -1,
            "seed": seed + i,
            "boosting_type": "gbdt",
        }
        dtrain = lgb.Dataset(X_train, label=y_shuf)
        gbm = lgb.train(params, dtrain, num_boost_round=num_boost_round)
        preds = gbm.predict(X_test)
        ic, _ = spearmanr(y_test_arr, preds)
        ic_f = float(ic) if ic == ic else 0.0
        ics.append(ic_f)
        if abs(ic_f) > ic_threshold:
            leakage = True

    if leakage:
        msg = (
            "Permutation test FAILED: model found signal in shuffled targets. "
            "Check feature engineering for lookahead leakage."
        )
    else:
        msg = "Permutation test passed — no obvious target leakage detected."

    return LeakageTestResult(
        passed=not leakage,
        iterations=ics,
        threshold=ic_threshold,
        message=msg,
    )
