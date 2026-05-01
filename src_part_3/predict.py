
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap
from scipy.optimize import nnls
from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
#  0.  DATA CONTRACT
#      Tag mỗi feature theo 4 loại để audit leakage.
# ─────────────────────────────────────────────────────────
DATA_CONTRACT = {
    # known_future: có thể tính trước cho kỳ forecast
    "known_future": [
        "dayofweek", "month", "quarter", "is_weekend",
        "is_month_end", "is_year_end", "is_month_start",
        "days_to_tet", "days_after_tet", "tet_decay_pre", "tet_decay_post",
        "days_to_sale_11", "days_to_sale_12", "days_to_bf",
        "sale_decay_11", "sale_decay_12", "sale_decay_bf",
        "is_holiday_30_4", "is_holiday_2_9", "is_holiday_1_5",
        "fourier_w_sin1", "fourier_w_cos1", "fourier_w_sin2", "fourier_w_cos2",
        "fourier_m_sin1", "fourier_m_cos1",
        "fourier_y_sin1", "fourier_y_cos1", "fourier_y_sin2", "fourier_y_cos2",
        "promo_active", "promo_pct_discount", "days_to_promo", "days_after_promo",
        "anchor_same_day_ly", "anchor_monthly_prior", "anchor_global_prior",
    ],
    # lagged_observed: dùng được nếu shift đủ (≥ 1 ngày)
    "lagged_observed": [
        "lag_1", "lag_7", "lag_14", "lag_30", "lag_90", "lag_365",
        "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
        "roll_mean_30", "roll_std_30", "roll_mean_90",
        "roll_min_7", "roll_max_7", "roll_min_30", "roll_max_30",
        "cogs_lag_1", "cogs_lag_7", "cogs_lag_30",
        "cogs_roll_mean_7", "cogs_roll_mean_30",
    ],
    # target_recursive: dùng prediction từ bước trước (recursive forecasting)
    "target_recursive": [],
    # historical_only: KHÔNG dùng trong deploy lane
    "historical_only": [
        "sessions", "unique_visitors", "page_views", "bounce_rate",
        "order_volume", "return_rate", "stock_on_hand", "fill_rate",
        "avg_session_duration_sec",
    ],
}

DEPLOY_FEATURES = DATA_CONTRACT["known_future"] + DATA_CONTRACT["lagged_observed"]


# ─────────────────────────────────────────────────────────
#  1.  LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────
def load_data(train_path: str) -> pd.DataFrame:
    df = pd.read_csv(train_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"Date": "date", "Revenue": "revenue", "COGS": "cogs"})
    # Ensure daily continuity
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_idx).rename_axis("date")
    df[["revenue", "cogs"]] = df[["revenue", "cogs"]].ffill().bfill()
    df = df.reset_index()
    return df


# ─────────────────────────────────────────────────────────
#  2.  FEATURE ENGINEERING — DEPLOY LANE ONLY
# ─────────────────────────────────────────────────────────
def _tet_dates() -> list[pd.Timestamp]:
    """Âm lịch 1/1 -> Dương lịch xấp xỉ cho 2012-2024"""
    return [
        pd.Timestamp("2013-02-10"), pd.Timestamp("2014-01-31"),
        pd.Timestamp("2015-02-19"), pd.Timestamp("2016-02-08"),
        pd.Timestamp("2017-01-28"), pd.Timestamp("2018-02-16"),
        pd.Timestamp("2019-02-05"), pd.Timestamp("2020-01-25"),
        pd.Timestamp("2021-02-12"), pd.Timestamp("2022-02-01"),
        pd.Timestamp("2023-01-22"), pd.Timestamp("2024-02-10"),
    ]

def _sale_day_dates(month: int, day: int, years=range(2012, 2025)) -> list[pd.Timestamp]:
    return [pd.Timestamp(year=y, month=month, day=day) for y in years]

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["dayofweek"]    = d.dt.dayofweek
    df["month"]        = d.dt.month
    df["quarter"]      = d.dt.quarter
    df["is_weekend"]   = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_year_end"]  = ((d.dt.month == 12) & (d.dt.day == 31)).astype(int)
    return df

def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    t = (df["date"] - df["date"].min()).dt.days.values
    # Weekly (period = 7)
    df["fourier_w_sin1"] = np.sin(2 * np.pi * t / 7)
    df["fourier_w_cos1"] = np.cos(2 * np.pi * t / 7)
    df["fourier_w_sin2"] = np.sin(4 * np.pi * t / 7)
    df["fourier_w_cos2"] = np.cos(4 * np.pi * t / 7)
    # Monthly (period = 30.44)
    df["fourier_m_sin1"] = np.sin(2 * np.pi * t / 30.44)
    df["fourier_m_cos1"] = np.cos(2 * np.pi * t / 30.44)
    # Yearly (period = 365.25)
    df["fourier_y_sin1"] = np.sin(2 * np.pi * t / 365.25)
    df["fourier_y_cos1"] = np.cos(2 * np.pi * t / 365.25)
    df["fourier_y_sin2"] = np.sin(4 * np.pi * t / 365.25)
    df["fourier_y_cos2"] = np.cos(4 * np.pi * t / 365.25)
    return df

def add_holiday_decay_features(df: pd.DataFrame) -> pd.DataFrame:
    dates = df["date"]

    def days_to_nearest(target_list):
        arr = np.array([t.value for t in target_list], dtype="int64")
        d_vals = dates.values.astype("int64")
        diffs = np.stack([np.abs(d_vals - a) for a in arr]).min(axis=0)
        return (diffs / 86400 / 1e9).astype(int)

    def signed_days(target_list):
        arr = np.array([t.value for t in target_list], dtype="int64")
        d_vals = dates.values.astype("int64")
        idx = np.abs(np.stack([d_vals - a for a in arr])).argmin(axis=0)
        return ((d_vals - np.array([arr[i] for i in idx])) / 86400 / 1e9).astype(int)

    tets = _tet_dates()
    signed = signed_days(tets)
    df["days_to_tet"]      = np.maximum(signed, 0)
    df["days_after_tet"]   = np.maximum(-signed, 0)
    df["tet_decay_pre"]    = np.exp(-df["days_to_tet"]  / 14).where(df["days_to_tet"]  < 30, 0)
    df["tet_decay_post"]   = np.exp(-df["days_after_tet"] / 7).where(df["days_after_tet"] < 15, 0)

    sale11 = _sale_day_dates(11, 11)
    sale12 = _sale_day_dates(12, 12)
    bf     = [pd.Timestamp(year=y, month=11, day=25) for y in range(2012, 2025)]

    df["days_to_sale_11"] = days_to_nearest(sale11)
    df["days_to_sale_12"] = days_to_nearest(sale12)
    df["days_to_bf"]      = days_to_nearest(bf)
    df["sale_decay_11"]   = np.exp(-df["days_to_sale_11"] / 5).where(df["days_to_sale_11"] < 14, 0)
    df["sale_decay_12"]   = np.exp(-df["days_to_sale_12"] / 5).where(df["days_to_sale_12"] < 14, 0)
    df["sale_decay_bf"]   = np.exp(-df["days_to_bf"]      / 5).where(df["days_to_bf"]      < 14, 0)

    df["is_holiday_30_4"] = ((dates.dt.month == 4) & (dates.dt.day == 30)).astype(int)
    df["is_holiday_1_5"]  = ((dates.dt.month == 5) & (dates.dt.day ==  1)).astype(int)
    df["is_holiday_2_9"]  = ((dates.dt.month == 9) & (dates.dt.day ==  2)).astype(int)
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 7, 14, 30, 90, 365]:
        df[f"lag_{lag}"]      = df["revenue"].shift(lag)
        if lag in [1, 7, 30]:
            df[f"cogs_lag_{lag}"] = df["cogs"].shift(lag)
    for win in [7, 14, 30, 90]:
        df[f"roll_mean_{win}"] = df["revenue"].shift(1).rolling(win, min_periods=1).mean()
        df[f"roll_std_{win}"]  = df["revenue"].shift(1).rolling(win, min_periods=1).std().fillna(0)
    for win in [7, 30]:
        df[f"roll_min_{win}"]  = df["revenue"].shift(1).rolling(win, min_periods=1).min()
        df[f"roll_max_{win}"]  = df["revenue"].shift(1).rolling(win, min_periods=1).max()
    for win in [7, 30]:
        df[f"cogs_roll_mean_{win}"] = df["cogs"].shift(1).rolling(win, min_periods=1).mean()
    df["promo_active"]      = 0
    df["promo_pct_discount"]= 0.0
    df["days_to_promo"]     = 999
    df["days_after_promo"]  = 999
    return df

def add_anchor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical seasonal anchor với recency-weighted YoY growth
    same_day_ly:    trung bình cùng ngày trong tuần, tuần đó, các năm trước (shrinkage)
    monthly_prior:  trung bình cùng tháng, năm trước
    global_prior:   trung bình toàn bộ train
    """
    df["anchor_global_prior"] = df["revenue"].expanding(min_periods=30).mean().shift(1)

    monthly_mean = (
        df.assign(month=df["date"].dt.month, year=df["date"].dt.year)
          .groupby(["year", "month"])["revenue"].transform("mean")
    )
    df["anchor_monthly_prior"] = monthly_mean.shift(365).fillna(df["anchor_global_prior"])

    # same month-day in prior years (recency-weighted)
    df["_mday"] = df["date"].dt.month * 100 + df["date"].dt.day
    smd = df.set_index("date")["revenue"]
    anchors = []
    for i, row in df.iterrows():
        past = smd.loc[:row["date"] - pd.Timedelta(days=1)]
        same = past[past.index.month * 100 + past.index.day == row["_mday"]]
        if len(same) == 0:
            anchors.append(np.nan)
        else:
            weights = np.exp(np.linspace(-1, 0, len(same)))
            anchors.append(np.average(same.values, weights=weights))
    df["anchor_same_day_ly"] = anchors
    df["anchor_same_day_ly"] = df["anchor_same_day_ly"].fillna(df["anchor_monthly_prior"])
    df.drop(columns=["_mday"], inplace=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_fourier_features(df)
    df = add_holiday_decay_features(df)
    df = add_lag_features(df)
    df = add_anchor_features(df)
    df["log_revenue"] = np.log1p(df["revenue"])
    df["ratio_to_anchor"] = df["revenue"] / (df["anchor_same_day_ly"] + 1)
    return df

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    available = [c for c in DEPLOY_FEATURES if c in df.columns]
    return available


# ─────────────────────────────────────────────────────────
#  2.5  GROWTH (v6) & PEAK PATTERN ESTIMATION
# ─────────────────────────────────────────────────────────
def compute_smooth_level(rev: pd.Series, window: int = 60) -> pd.Series:
    rolling_med = rev.rolling(31, center=True, min_periods=7).median()
    rolling_mad = (rev - rolling_med).abs().rolling(31, center=True, min_periods=7).median() + 1
    z = (rev - rolling_med) / (1.4826 * rolling_mad)
    rev_clean = rev.where(z.abs() <= 1.5, np.nan)
    level = rev_clean.rolling(window, center=True, min_periods=window // 3).median()
    return level.interpolate(limit_direction="both")


def estimate_growth_v6(df: pd.DataFrame, level: pd.Series) -> float:
    """
    Median của 3 growth estimators trên yearly smooth level means (v6 approach).
    Cap 25%/year. Dùng làm hệ số nhân vào toàn bộ predictions theo năm.
    """
    yl = df.assign(L=level.values).groupby(df["date"].dt.year)["L"].mean()
    ymeans = {y: yl.loc[y] for y in [2020, 2021, 2022] if y in yl.index}
    if len(ymeans) < 2:
        return 0.05
    g_21_22 = ymeans[2022] / ymeans[2021] - 1
    g_20_22 = (ymeans[2022] / ymeans[2020]) ** 0.5 - 1 if 2020 in ymeans else g_21_22
    years = np.array(sorted(ymeans.keys()))
    slope, _ = np.polyfit(years, np.log([ymeans[y] for y in years]), 1)
    g_loglin = float(np.exp(slope) - 1)
    g = float(np.median([g_21_22, g_20_22, g_loglin]))
    print(f"  CAGR21-22={g_21_22:+.2%}  CAGR20-22={g_20_22:+.2%}  log-lin={g_loglin:+.2%}  median={g:+.2%}")
    return float(np.clip(g, 0.0, 0.25))


# ─────────────────────────────────────────────────────────
#  3.  WALK-FORWARD CV
# ─────────────────────────────────────────────────────────
def walk_forward_splits(df: pd.DataFrame, n_splits: int = 4, min_train_years: int = 3):
    """
    Expanding window splits theo năm.
    Fold i  →  train: [start, cutoff_i)   val: [cutoff_i, cutoff_i+1)
    """
    years = sorted(df["date"].dt.year.unique())
    # Keep last n_splits years as validation candidates
    val_years = years[-(n_splits):]
    splits = []
    for i, vy in enumerate(val_years):
        train_end = pd.Timestamp(year=vy, month=1, day=1)
        val_end   = pd.Timestamp(year=vy + 1, month=1, day=1)
        if val_end > df["date"].max() + pd.Timedelta(days=1):
            val_end = df["date"].max() + pd.Timedelta(days=1)
        train_mask = df["date"] < train_end
        val_mask   = (df["date"] >= train_end) & (df["date"] < val_end)
        if train_mask.sum() < min_train_years * 365:
            continue
        splits.append((df.index[train_mask], df.index[val_mask]))
    return splits

def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    bias = np.mean(y_pred - y_true)
    underpred = np.mean(y_pred < y_true)
    print(f"  [{label}] MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}  Bias={bias:,.0f}  Underpred={underpred:.1%}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias, "underpred": underpred}


# ─────────────────────────────────────────────────────────
#  4.  OPTUNA TUNING
# ─────────────────────────────────────────────────────────
def tune_lgb(X_tr, y_tr, X_val, y_val, n_trials: int = 60, objective: str = "regression") -> dict:
    """
    Optuna + early stopping cho LightGBM.
    objective: 'regression' (log head) | 'quantile' (q25/q75) | 'mape' (ratio head)
    """
    def _objective(trial: optuna.Trial) -> float:
        params = {
            "objective":          objective,
            "metric":             "mae" if objective != "quantile" else "quantile",
            "alpha":              trial.suggest_float("alpha", 0.25, 0.75) if objective == "quantile" else 0.5,
            "verbosity":          -1,
            "seed":               RANDOM_SEED,
            "num_leaves":         trial.suggest_int("num_leaves", 16, 256),
            "max_depth":          trial.suggest_int("max_depth", 4, 12),
            "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "n_estimators":       2000,
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "min_child_samples":  trial.suggest_int("min_child_samples", 5, 100),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"objective": objective, "verbosity": -1, "seed": RANDOM_SEED, "n_estimators": 2000})
    return best


def tune_xgb(X_tr, y_tr, X_val, y_val, n_trials: int = 40) -> dict:
    """Optuna cho XGBoost challenger"""
    def _objective(trial: optuna.Trial) -> float:
        params = {
            "objective":        "reg:absoluteerror",
            "eval_metric":      "mae",
            "seed":             RANDOM_SEED,
            "n_estimators":     2000,
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        }
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return mean_absolute_error(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"objective": "reg:absoluteerror", "seed": RANDOM_SEED,
                 "n_estimators": 2000, "verbosity": 0})
    return best


# ─────────────────────────────────────────────────────────
#  5.  MULTI-HEAD LightGBM
# ─────────────────────────────────────────────────────────
class MultiHeadLGB:
    """
    2 đầu dự báo độc lập:
      log   → log1p(revenue)   → inverse: expm1
      ratio → revenue / anchor → inverse: × anchor
    """

    def __init__(self, log_params, ratio_params, log_weight: float = 0.55, ratio_weight: float = 0.45):
        self.log_params   = {**log_params,   "objective": "regression"}
        self.ratio_params = {**ratio_params, "objective": "regression", "metric": "mape"}
        w_sum = max(log_weight + ratio_weight, 1e-9)
        self.log_weight = log_weight / w_sum
        self.ratio_weight = ratio_weight / w_sum
        self.models = {}

    def _fit_one(self, name, params, X_tr, y_tr, X_val, y_val):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        self.models[name] = m
        return m

    def fit(self, X_tr, y_tr, anchor_tr, X_val, y_val, anchor_val):
        log_y_tr = np.log1p(y_tr)
        log_y_val= np.log1p(y_val)
        ratio_tr  = y_tr  / (anchor_tr  + 1)
        ratio_val = y_val / (anchor_val + 1)

        self._fit_one("log",   self.log_params,   X_tr, log_y_tr,  X_val, log_y_val)
        self._fit_one("ratio", self.ratio_params, X_tr, ratio_tr,  X_val, ratio_val)

    def predict(self, X, anchor):
        pred_log   = np.expm1(self.models["log"].predict(X))
        pred_ratio = self.models["ratio"].predict(X) * (anchor + 1)
        # Weighted blend từ 2 heads
        blended = self.log_weight * pred_log + self.ratio_weight * pred_ratio
        return np.maximum(blended, 0)

    def predict_components(self, X, anchor):
        pred_log   = np.expm1(self.models["log"].predict(X))
        pred_ratio = self.models["ratio"].predict(X) * (anchor + 1)
        return np.maximum(pred_log, 0), np.maximum(pred_ratio, 0)

    def feature_importances(self, feature_names):
        out = {}
        for name, m in self.models.items():
            imp = pd.Series(m.feature_importances_, index=feature_names, name=name)
            out[name] = imp.sort_values(ascending=False)
        return out


# ─────────────────────────────────────────────────────────
#  6.  OOF META-ENSEMBLE (NNLS)
# ─────────────────────────────────────────────────────────
def nnls_blend(oof_preds: dict, y_true: np.ndarray) -> dict:
    """
    Non-negative least squares blend.
    oof_preds: {'lgb': array, 'xgb': array, 'anchor': array, ...}
    Returns: {'weights': {}, 'blend_oof': array}
    """
    names = list(oof_preds.keys())
    A = np.column_stack([oof_preds[n] for n in names])
    coef, _ = nnls(A, y_true)
    coef /= coef.sum() + 1e-9
    blend = A @ coef
    print(f"\n  NNLS weights: { {n: round(w, 3) for n, w in zip(names, coef)} }")
    return {"weights": dict(zip(names, coef)), "blend_oof": blend}


def tune_log_ratio_weights(
    y_true: np.ndarray,
    pred_log: np.ndarray,
    pred_ratio: np.ndarray,
    step: float = 0.02,
) -> dict:
    """
    Tune blend between log-head and ratio-head using OOF MAE.
    ratio_weight = 1 - log_weight.
    """
    best = {"log_weight": 0.55, "ratio_weight": 0.45, "mae": float("inf")}
    w = 0.0
    while w <= 1.000001:
        log_w = float(w)
        ratio_w = 1.0 - log_w
        blend = np.maximum(log_w * pred_log + ratio_w * pred_ratio, 0)
        mae = mean_absolute_error(y_true, blend)
        if mae < best["mae"]:
            best = {"log_weight": log_w, "ratio_weight": ratio_w, "mae": float(mae)}
        w += step
    print(
        f"\n  Best log/ratio weights (OOF MAE): "
        f"log={best['log_weight']:.2f}, ratio={best['ratio_weight']:.2f}, MAE={best['mae']:,.0f}"
    )
    return best


def tune_log_ratio_weights_two_stage(
    y_true: np.ndarray,
    pred_log: np.ndarray,
    pred_ratio: np.ndarray,
) -> dict:  
    coarse = tune_log_ratio_weights(y_true, pred_log, pred_ratio, step=0.02)
    center = coarse["log_weight"]
    lo = max(0.0, center - 0.06)
    hi = min(1.0, center + 0.06)
    best = {"log_weight": center, "ratio_weight": 1.0 - center, "mae": float("inf")}
    w = lo
    while w <= hi + 1e-9:
        lw = float(round(w, 4))
        rw = 1.0 - lw
        blend = np.maximum(lw * pred_log + rw * pred_ratio, 0)
        mae = mean_absolute_error(y_true, blend)
        if mae < best["mae"]:
            best = {"log_weight": lw, "ratio_weight": rw, "mae": float(mae)}
        w += 0.005
    print(
        f"  Refined log/ratio weights: "
        f"log={best['log_weight']:.3f}, ratio={best['ratio_weight']:.3f}, MAE={best['mae']:,.0f}"
    )
    return best


def tune_cogs_params(
    revenue_proxy: np.ndarray,
    cogs_true: np.ndarray,
    ratio_proxy: np.ndarray,
) -> dict:
    best = {"window": 90, "min_ratio": 0.40, "max_ratio": 0.93, "mae": float("inf")}
    windows = [30, 60, 90, 120]
    min_caps = [0.35, 0.40, 0.45]
    max_caps = [0.88, 0.90, 0.93, 0.95]
    for w in windows:
        rp = pd.Series(ratio_proxy).rolling(w, min_periods=7).mean().fillna(method="bfill").fillna(method="ffill").to_numpy()
        for mn in min_caps:
            for mx in max_caps:
                if mn >= mx:
                    continue
                pred = np.maximum(revenue_proxy, 0) * np.clip(rp, mn, mx)
                mae = mean_absolute_error(cogs_true, pred)
                if mae < best["mae"]:
                    best = {"window": w, "min_ratio": mn, "max_ratio": mx, "mae": float(mae)}
    print(
        f"\n  Best COGS clamp (OOF MAE): "
        f"window={best['window']}, min={best['min_ratio']:.2f}, max={best['max_ratio']:.2f}, MAE={best['mae']:,.0f}"
    )
    return best


def tune_smooth_window(
    dates: pd.Series,
    revenue_pred: np.ndarray,
    revenue_true: np.ndarray,
    holiday_dates: list,
) -> dict:
    best = {"window": 1, "mae": float("inf")}
    base_df = pd.DataFrame({"date": pd.to_datetime(dates), "revenue": np.maximum(revenue_pred, 0), "cogs": np.zeros(len(revenue_pred))})
    for w in [1, 3, 5]:
        df_sm = apply_constraints(base_df.copy(), max_cogs_ratio=1.0, smooth_window=w, holiday_dates=holiday_dates)
        mae = mean_absolute_error(revenue_true, df_sm["revenue"].to_numpy())
        if mae < best["mae"]:
            best = {"window": w, "mae": float(mae)}
    print(f"\n  Best smooth window (OOF revenue MAE): window={best['window']}, MAE={best['mae']:,.0f}")
    return best


# ─────────────────────────────────────────────────────────
#  7.  SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "lag_1": "Previous day's value",
    "lag_7": "1 week ago (7-day lag)",
    "lag_14": "2 weeks ago (14-day lag)",
    "lag_30": "1 month ago (30-day lag)",
    "lag_90": "3 months ago (90-day lag)",
    "lag_365": "1 year ago (365-day lag)",
    "roll_mean_7": "7-day rolling average",
    "roll_std_7": "7-day rolling std dev",
    "roll_mean_14": "14-day rolling average",
    "roll_std_14": "14-day rolling std dev",
    "roll_mean_30": "30-day rolling average",
    "roll_std_30": "30-day rolling std dev",
    "roll_mean_90": "90-day rolling average",
    "roll_min_7": "7-day rolling minimum",
    "roll_max_7": "7-day rolling maximum",
    "roll_min_30": "30-day rolling minimum",
    "roll_max_30": "30-day rolling maximum",
    "cogs_lag_1": "COGS from 1 day ago",
    "cogs_lag_7": "COGS from 7 days ago",
    "cogs_lag_30": "COGS from 30 days ago",
    "cogs_roll_mean_7": "COGS 7-day rolling average",
    "cogs_roll_mean_30": "COGS 30-day rolling average",
    "anchor_same_day_ly": "Same day from last year",
    "anchor_monthly_prior": "Monthly historical average",
    "anchor_global_prior": "Global historical average",
    "fourier_w_sin1": "Weekly seasonality (sine 1)",
    "fourier_w_cos1": "Weekly seasonality (cosine 1)",
    "fourier_w_sin2": "Weekly seasonality (sine 2)",
    "fourier_w_cos2": "Weekly seasonality (cosine 2)",
    "fourier_m_sin1": "Monthly seasonality (sine)",
    "fourier_m_cos1": "Monthly seasonality (cosine)",
    "fourier_y_sin1": "Yearly seasonality (sine)",
    "fourier_y_cos1": "Yearly seasonality (cosine)",
    "fourier_y_sin2": "Yearly seasonality (sine 2)",
    "fourier_y_cos2": "Yearly seasonality (cosine 2)",
    "dayofweek": "Day of week (0=Mon, 6=Sun)",
    "month": "Month of year (1-12)",
    "quarter": "Quarter of year (1-4)",
    "is_weekend": "Is weekend (0/1)",
    "is_month_end": "Is month end (0/1)",
    "is_year_end": "Is year end (0/1)",
    "is_month_start": "Is month start (0/1)",
    "days_to_tet": "Days until Tet holiday",
    "days_after_tet": "Days since Tet holiday",
    "tet_decay_pre": "Pre-Tet decay factor",
    "tet_decay_post": "Post-Tet decay factor",
    "days_to_sale_11": "Days until 11/11 sale",
    "days_to_sale_12": "Days until 12/12 sale",
    "days_to_bf": "Days until Black Friday",
    "sale_decay_11": "11/11 sale decay factor",
    "sale_decay_12": "12/12 sale decay factor",
    "sale_decay_bf": "Black Friday decay factor",
    "is_holiday_30_4": "Is 30/4 holiday",
    "is_holiday_2_9": "Is 2/9 holiday",
    "is_holiday_1_5": "Is 1/5 holiday",
    "promo_active": "Is promotion active",
    "promo_pct_discount": "Promotion discount percentage",
    "days_to_promo": "Days until next promotion",
    "days_after_promo": "Days since last promotion",
    "sessions": "Website sessions",
    "unique_visitors": "Unique visitors",
    "page_views": "Page views",
    "bounce_rate": "Bounce rate (%)",
    "order_volume": "Order volume",
    "return_rate": "Return rate (%)",
    "stock_on_hand": "Inventory on hand",
    "fill_rate": "Order fill rate (%)",
    "avg_session_duration_sec": "Avg session duration (seconds)",
}

def compute_shap(model: lgb.LGBMRegressor, X: pd.DataFrame, max_display: int = 20):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
        name="mean_abs_shap"
    ).sort_values(ascending=False)
    print(f"\n  Top-{max_display} SHAP features (log head):")
    top_features = mean_abs.head(max_display)
    for feat, importance in top_features.items():
        description = FEATURE_DESCRIPTIONS.get(feat, feat)
        print(f"{description:45s} {importance:10.6f}")
    return shap_values, mean_abs


# ─────────────────────────────────────────────────────────
#  8.  POST-PROCESSING CONSTRAINTS
# ─────────────────────────────────────────────────────────
def apply_constraints(
    df_pred: pd.DataFrame,
    max_cogs_ratio: float = 0.95,
    smooth_window: int = 3,
    holiday_dates: list = None,
) -> pd.DataFrame:
    df_pred["revenue"] = np.maximum(df_pred["revenue"], 0)
    df_pred["cogs"]    = np.maximum(df_pred["cogs"],    0)
    df_pred["cogs"]    = np.minimum(df_pred["cogs"],    df_pred["revenue"] * max_cogs_ratio)

    # Smooth non-event spikes nhẹ (không smooth holiday)
    if holiday_dates is not None:
        holiday_set = set(pd.to_datetime(holiday_dates).date)
        non_event = ~df_pred["date"].dt.date.isin(holiday_set)
        df_pred.loc[non_event, "revenue"] = (
            df_pred.loc[non_event, "revenue"]
                   .rolling(smooth_window, center=True, min_periods=1)
                   .mean()
        )
    return df_pred


# ─────────────────────────────────────────────────────────
#  9.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────
def run_pipeline(
    train_path: str = "sales.csv",
    output_path: str = "submission.csv",
    forecast_start: str = "2023-01-01",
    forecast_end: str = "2024-07-01",
    n_optuna_trials_lgb: int = 60,
    n_optuna_trials_xgb: int = 40,
):
    print("=" * 60)
    print("  REVENUE FORECAST PIPELINE — Datathon 2026")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────
    print("\n[1] Loading & building features...")
    df = load_data(train_path)
    df = build_features(df)
    df = df.dropna(subset=["lag_365"]).reset_index(drop=True)   # cần đủ history

    feat_cols = get_feature_cols(df)
    print(f"    {len(feat_cols)} deploy-safe features | {len(df)} training days")

    X  = df[feat_cols].values
    y  = df["revenue"].values
    yc = df["cogs"].values
    anchor = df["anchor_same_day_ly"].values

    # ── Walk-forward splits ───────────────────────────────
    # Force 3 recent folds: 2019->2020, 2020->2021, 2021->2022
    splits = walk_forward_splits(df, n_splits=3)
    print(f"\n[2] Walk-forward CV: {len(splits)} folds")

    # ── Tune on last fold (most recent) ──────────────────
    tr_idx, val_idx = splits[-1]
    X_tr,  y_tr   = X[tr_idx],  y[tr_idx]
    X_val, y_val  = X[val_idx], y[val_idx]
    a_tr,  a_val  = anchor[tr_idx], anchor[val_idx]

    print(f"\n[3] Tuning LightGBM (log head) — {n_optuna_trials_lgb} trials...")
    lgb_params = tune_lgb(X_tr, np.log1p(y_tr), X_val, np.log1p(y_val),
                          n_trials=n_optuna_trials_lgb, objective="regression")

    print(f"\n[4] Tuning XGBoost challenger — {n_optuna_trials_xgb} trials...")
    xgb_params = tune_xgb(X_tr, y_tr, X_val, y_val, n_trials=n_optuna_trials_xgb)

    # ── OOF predictions ───────────────────────────────────
    print("\n[5] Generating OOF predictions across all folds...")
    oof_lgb    = np.zeros(len(df))
    oof_lgb_log = np.zeros(len(df))
    oof_lgb_ratio = np.zeros(len(df))
    oof_xgb    = np.zeros(len(df))
    oof_anchor = np.zeros(len(df))

    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]
        ava = anchor[val_idx]

        # LightGBM multi-head
        mh = MultiHeadLGB(lgb_params, lgb_params, log_weight=0.55, ratio_weight=0.45)
        mh.fit(Xtr, ytr, anchor[tr_idx], Xva, yva, ava)
        p_log, p_ratio = mh.predict_components(Xva, ava)
        oof_lgb_log[val_idx] = p_log
        oof_lgb_ratio[val_idx] = p_ratio
        oof_lgb[val_idx] = mh.predict(Xva, ava)

        # XGBoost
        xm = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        xm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_xgb[val_idx] = np.maximum(xm.predict(Xva), 0)

        # Anchor baseline
        oof_anchor[val_idx] = ava

        val_mask = np.zeros(len(df), dtype=bool)
        val_mask[val_idx] = True
        print(f"  Fold {fold_i+1} val={df['date'].iloc[val_idx[0]].date()}..{df['date'].iloc[val_idx[-1]].date()}")
        evaluate(yva, oof_lgb[val_idx],    label="LGB")
        evaluate(yva, oof_xgb[val_idx],    label="XGB")
        evaluate(yva, oof_anchor[val_idx], label="Anchor")

    # ── Prepare holiday list for tuners/post-process ──────
    all_holidays = (
        [t.strftime("%Y-%m-%d") for t in _tet_dates()]
        + [f"{y}-04-30" for y in range(2019, 2025)]
        + [f"{y}-11-11" for y in range(2019, 2025)]
        + [f"{y}-12-12" for y in range(2019, 2025)]
    )

    # ── Tune log/ratio weights on OOF (coarse + fine) ────
    # Chỉ blend trên các val indices
    all_val = np.concatenate([v for _, v in splits])
    lr_best = tune_log_ratio_weights_two_stage(
        y_true=y[all_val],
        pred_log=oof_lgb_log[all_val],
        pred_ratio=oof_lgb_ratio[all_val],
    )
    oof_lgb[all_val] = (
        lr_best["log_weight"] * oof_lgb_log[all_val]
        + lr_best["ratio_weight"] * oof_lgb_ratio[all_val]
    )
    print("  Tuned LGB-head blend performance (all folds):")
    evaluate(y[all_val], oof_lgb[all_val], label="LGB_TUNED")

    # ── NNLS blend ────────────────────────────────────────
    oof_ens = nnls_blend(
        {"lgb": oof_lgb[all_val], "xgb": oof_xgb[all_val], "anchor": oof_anchor[all_val]},
        y[all_val]
    )
    w = oof_ens["weights"]
    print("\n  Final ensemble performance (all folds):")
    evaluate(y[all_val], oof_ens["blend_oof"], label="ENSEMBLE")

    # ── Tune COGS clamp params on OOF (order #1) ─────────
    hist_ratio_proxy = (df["cogs"] / (df["revenue"] + 1)).shift(1).fillna(method="bfill").fillna(method="ffill").to_numpy()
    cogs_best = tune_cogs_params(
        revenue_proxy=oof_ens["blend_oof"],
        cogs_true=yc[all_val],
        ratio_proxy=hist_ratio_proxy[all_val],
    )

    # ── Tune smooth window on OOF (order #2) ─────────────
    smooth_best = tune_smooth_window(
        dates=df.loc[all_val, "date"],
        revenue_pred=oof_ens["blend_oof"],
        revenue_true=y[all_val],
        holiday_dates=all_holidays,
    )

    # ── Retrain final models on ALL train data ────────────
    print("\n[6] Retraining final models on full train set...")
    mh_final = MultiHeadLGB(
        lgb_params,
        lgb_params,
        log_weight=lr_best["log_weight"],
        ratio_weight=lr_best["ratio_weight"],
    )
    # Hold last year as early-stopping set
    last_year = df["date"].dt.year.max()
    final_tr  = df["date"].dt.year < last_year
    mh_final.fit(
        X[final_tr], y[final_tr], anchor[final_tr],
        X[~final_tr], y[~final_tr], anchor[~final_tr]
    )
    xm_final = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xm_final.fit(
        X[final_tr], y[final_tr],
        eval_set=[(X[~final_tr], y[~final_tr])],
        verbose=False
    )

    # ── SHAP (disabled for faster benchmark runs) ────────
    print("\n[7] Computing SHAP values (both heads)...")
    X_val_df = pd.DataFrame(X[~final_tr], columns=feat_cols)
    shap_values_log, shap_importance = compute_shap(
        model=mh_final.models["log"],
        X=X_val_df,
        max_display=20,
    )

    import matplotlib.pyplot as plt

    # Bar plot - mean |SHAP|
    shap.summary_plot(shap_values_log, X_val_df, plot_type="bar", max_display=20)
    plt.tight_layout()
    plt.savefig("shap_bar.png", dpi=150)

    # Beeswarm - phân phối từng feature
    shap.summary_plot(shap_values_log, X_val_df, max_display=20)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150)

    # ── Compute growth coefficient từ smooth level 2020-2022 ─
    print("\n[7.5] Computing growth trend (smooth level 2020-2022)...")
    df_level = compute_smooth_level(df["revenue"], window=60)
    g_trend = estimate_growth_v6(df, df_level)
    print(f"  Growth coefficient: {g_trend:+.2%}/yr | 2023: x{1+g_trend:.4f}, 2024: x{(1+g_trend)**2:.4f}")

    # ── Forecast 2023-01-01 to 2024-07-01 ────────────────
    print(f"\n[8] Forecasting {forecast_start} to {forecast_end}...")
    forecast_dates = pd.date_range(forecast_start, forecast_end, freq="D")

    # Recursive: start từ cuối train, append predictions từng bước
    history = df[["date", "revenue", "cogs"]].copy()

    rows = []
    for fd in forecast_dates:
        tmp = pd.concat([history, pd.DataFrame([{"date": fd, "revenue": np.nan, "cogs": np.nan}])])
        tmp = build_features(tmp)
        row = tmp[tmp["date"] == fd].iloc[0]

        fv = np.array([[row.get(c, 0) for c in feat_cols]])
        anc = np.array([row.get("anchor_same_day_ly", row.get("anchor_global_prior", 0))])

        lgb_pred = mh_final.predict(fv, anc)[0]
        xgb_pred = max(xm_final.predict(fv)[0], 0)
        anc_pred = anc[0]

        rev_pred = (w["lgb"] * lgb_pred + w["xgb"] * xgb_pred + w["anchor"] * anc_pred)
        rev_pred = max(rev_pred, 0)

        # COGS: dùng historical cogs/revenue ratio với smoothing
        cogs_ratio = (history["cogs"] / (history["revenue"] + 1)).tail(int(cogs_best["window"])).mean()
        cogs_ratio = np.clip(cogs_ratio, float(cogs_best["min_ratio"]), float(cogs_best["max_ratio"]))
        cogs_pred  = rev_pred * cogs_ratio

        rows.append({"date": fd, "revenue": rev_pred, "cogs": cogs_pred})
        history = pd.concat([
            history,
            pd.DataFrame([{"date": fd, "revenue": rev_pred, "cogs": cogs_pred}])
        ]).reset_index(drop=True)

    forecast_df = pd.DataFrame(rows)

    # ── Post-processing ───────────────────────────────────
    print("\n[9] Applying post-processing constraints...")
    forecast_df = apply_constraints(
        forecast_df,
        smooth_window=int(smooth_best["window"]),
        holiday_dates=all_holidays,
    )

    # ── Áp hệ số tăng trưởng (post-hoc, sau loop) ────────
    # Apply SAU loop để tránh compounding qua lag features của recursive forecast.
    is_2023 = forecast_df["date"].dt.year == 2023
    is_2024 = forecast_df["date"].dt.year == 2024
    forecast_df.loc[is_2023, ["revenue", "cogs"]] *= (1.0 + g_trend)
    forecast_df.loc[is_2024, ["revenue", "cogs"]] *= (1.0 + g_trend) ** 2
    print(f"  Growth applied: 2023 x{1+g_trend:.4f}, 2024 x{(1+g_trend)**2:.4f}")

    # ── Export ────────────────────────────────────────────
    submission = pd.DataFrame({
        "Date":    forecast_df["date"].dt.strftime("%Y-%m-%d"),
        "Revenue": forecast_df["revenue"].round(2),
        "COGS":    forecast_df["cogs"].round(2),
    })
    # Always export to project root (same directory as test.py)
    root_out = Path(__file__).resolve().parent / Path(output_path).name
    submission.to_csv(root_out, index=False)
    print(f"\n[OK] Saved {len(submission)} rows to {root_out}")
    print(
        "    Tuned params:"
        f" log_w={lr_best['log_weight']:.3f}, ratio_w={lr_best['ratio_weight']:.3f},"
        f" cogs_window={int(cogs_best['window'])}, cogs_min={cogs_best['min_ratio']:.2f},"
        f" cogs_max={cogs_best['max_ratio']:.2f}, smooth_window={int(smooth_best['window'])}"
    )
    print(f"    Revenue: {submission['Revenue'].min():,.2f} – {submission['Revenue'].max():,.2f}")
    print(f"    COGS:    {submission['COGS'].min():,.2f} – {submission['COGS'].max():,.2f}")
    print("=" * 60)

    return submission, shap_importance


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    submission, shap_imp = run_pipeline(
        train_path="C:\\Project\\Personal Project\\Datathon 2026\\data\\sales.csv",
        output_path="submission.csv",
        n_optuna_trials_lgb=60,
        n_optuna_trials_xgb=40,
    )