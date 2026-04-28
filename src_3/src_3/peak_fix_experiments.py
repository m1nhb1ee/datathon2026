from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


def xgb_reg(seed: int = 42, n_estimators: int = 800, objective: str = "reg:squarederror", learning_rate: float = 0.03, quantile_alpha: float = 0.8):
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": 8,
        "learning_rate": float(learning_rate),
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "objective": objective,
        "random_state": int(seed),
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if objective == "reg:quantileerror":
        params["quantile_alpha"] = float(quantile_alpha)
    return XGBRegressor(**params)


def xgb_clf(seed: int = 42):
    return XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.02,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=-1,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
    )


def fit_log_model(model, x_tr: pd.DataFrame, y_tr: pd.Series | np.ndarray):
    y = np.asarray(y_tr, dtype=float)
    model.fit(x_tr, np.log1p(np.clip(y, 0, None)))
    return model


def predict_log_model(model, x: pd.DataFrame):
    return np.clip(np.expm1(model.predict(x)), 0, None)


def peak_weights(y: pd.Series | np.ndarray, dates: pd.Series | np.ndarray | None = None):
    yv = np.asarray(y, dtype=float)
    q80 = np.quantile(yv, 0.80)
    q90 = np.quantile(yv, 0.90)
    q95 = np.quantile(yv, 0.95)
    w = np.ones(len(yv), dtype=float)
    w += (yv >= q80).astype(float) * 0.8
    w += (yv >= q90).astype(float) * 1.2
    w += (yv >= q95).astype(float) * 1.5

    # Optional seasonal reweighting for peaks by month-day prior mean, if dates provided.
    if dates is not None:
        d = pd.to_datetime(pd.Series(dates))
        md = d.dt.strftime("%m-%d")
        md_mean = pd.Series(yv).groupby(md).transform("mean").to_numpy()
        rank = pd.Series(md_mean).rank(pct=True).to_numpy()
        w += np.clip(rank - 0.7, 0, 0.3) * 1.5
    return w
