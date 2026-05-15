"""
Optuna hyperparameter search for LightGBM classifiers (StockMarketTool-style).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def tune_lgbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_trials: int = 40,
    n_estimators: int = 1200,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Return sklearn LGBMClassifier kwargs from Optuna study (maximize val AUC).
    Falls back to empty dict if optuna/lightgbm unavailable.
    """
    try:
        import lightgbm as lgb
        import optuna
        from optuna.integration import LightGBMPruningCallback
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "random_state": random_state,
            "n_estimators": n_estimators,
            "class_weight": "balanced",
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                LightGBMPruningCallback(trial, "auc"),
            ],
        )
        prob = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, prob))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    return {
        "learning_rate": best["learning_rate"],
        "num_leaves": best["num_leaves"],
        "max_depth": best["max_depth"],
        "min_child_samples": best["min_child_samples"],
        "reg_alpha": best["reg_alpha"],
        "reg_lambda": best["reg_lambda"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "subsample_freq": best.get("subsample_freq", 1),
        "random_state": random_state,
        "verbose": -1,
        "class_weight": "balanced",
    }
