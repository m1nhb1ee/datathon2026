from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor


SEED = 42
DATE_COL = "Date"
TARGETS = ["Revenue", "COGS"]
DROP_COLS = [DATE_COL, *TARGETS]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs" / "forecast_ver3"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"
for path in [OUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FB = load_module(ROOT / "src_3" / "build_modeling_dataset.py", "feature_builder")


UPPER_RAW_WEIGHT = 0.70
UPPER_Q80_WEIGHT = 0.30
GATE_VALIDATION_MAE_TOLERANCE = 1.03


def choose_device() -> str:
    x = np.random.RandomState(SEED).normal(size=(32, 3))
    y = np.random.RandomState(SEED + 1).normal(size=32)
    try:
        probe = XGBRegressor(
            n_estimators=2,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            random_state=SEED,
            verbosity=0,
        )
        probe.fit(x, y)
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = choose_device()


def xgb_reg(
    *,
    seed: int = SEED,
    n_estimators: int = 900,
    objective: str = "reg:squarederror",
    learning_rate: float = 0.035,
    max_depth: int = 4,
    min_child_weight: float = 4,
    reg_alpha: float = 0.05,
    reg_lambda: float = 0.8,
    quantile_alpha: float | None = None,
) -> XGBRegressor:
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.88,
        colsample_bytree=0.88,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective=objective,
        tree_method="hist",
        device=DEVICE,
        random_state=seed,
        n_jobs=0,
        verbosity=0,
    )
    if quantile_alpha is not None:
        params["quantile_alpha"] = quantile_alpha
    return XGBRegressor(**params)


def xgb_clf(seed: int = SEED) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=650,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.88,
        colsample_bytree=0.88,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=DEVICE,
        random_state=seed,
        n_jobs=0,
        verbosity=0,
    )


def fit_log_model(model: XGBRegressor, x: pd.DataFrame, y: pd.Series, sample_weight=None) -> XGBRegressor:
    model.fit(x, np.log1p(y), sample_weight=sample_weight)
    return model


def predict_log_model(model: XGBRegressor, x: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(x)).clip(min=0)


def peak_weights(y: pd.Series, dates: pd.Series) -> np.ndarray:
    y = y.reset_index(drop=True)
    dates = pd.to_datetime(dates).reset_index(drop=True)
    d1 = y.diff().clip(lower=0)
    w = np.ones(len(y), dtype=float)
    q90, q95, q98 = y.quantile([0.90, 0.95, 0.98])
    dq90, dq95 = d1.quantile([0.90, 0.95])
    w += (y >= q90).to_numpy() * 1.25
    w += (y >= q95).to_numpy() * 1.75
    w += (y >= q98).to_numpy() * 2.50
    w += (d1 >= dq90).to_numpy() * 0.75
    w += (d1 >= dq95).to_numpy() * 1.00
    w += dates.dt.day.isin([28, 29, 30, 31]).to_numpy() * 0.25
    return w


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "R2": float(r2_score(y_true, pred)),
        "bias": float(np.mean(pred - y_true)),
        "pred_over_actual": float(np.mean(pred / y_true)),
        "under_pred_rate": float(np.mean(pred < y_true)),
    }


def make_gate(prob: np.ndarray, threshold: float, gate_weight: float) -> np.ndarray:
    return (prob >= threshold).astype(float) * gate_weight


def apply_ver3_revenue(
    base_pred: np.ndarray,
    raw_pred: np.ndarray,
    q80_pred: np.ndarray,
    peak_prob: np.ndarray,
    gate_threshold: float,
    gate_weight: float,
) -> np.ndarray:
    upper_tail = UPPER_RAW_WEIGHT * raw_pred + UPPER_Q80_WEIGHT * q80_pred
    gate = make_gate(peak_prob, gate_threshold, gate_weight)
    return ((1.0 - gate) * base_pred + gate * upper_tail).clip(min=0)


def segment_masks(df: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    q90, q95, q98 = df[target].quantile([0.90, 0.95, 0.98])
    d1 = df[target].diff()
    local_peak = (
        (df[target] > df[target].shift(1))
        & (df[target] > df[target].shift(-1))
        & (df[target] >= q90)
    ).fillna(False).to_numpy()
    fast_growth = (d1 >= d1[d1 > 0].quantile(0.90)).fillna(False).to_numpy()
    return {
        "all": np.ones(len(df), dtype=bool),
        "top10_actual": (df[target] >= q90).to_numpy(),
        "top5_actual": (df[target] >= q95).to_numpy(),
        "top2_actual": (df[target] >= q98).to_numpy(),
        "local_peak_p90": local_peak,
        "fast_growth_top10": fast_growth,
    }


def add_metric_rows(rows: list[dict], df: pd.DataFrame, target: str, pred_col: str, model: str) -> None:
    y = df[target].to_numpy()
    pred = df[pred_col].to_numpy()
    for segment, mask in segment_masks(df, target).items():
        rows.append(
            {
                "target": target,
                "model": model,
                "segment": segment,
                "n": int(mask.sum()),
                **metrics(y[mask], pred[mask]),
            }
        )


def tune_revenue_gate_2019(train_full: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    tune_train = train_full[train_full[DATE_COL] < "2019-01-01"].copy()
    tune_val = train_full[(train_full[DATE_COL] >= "2019-01-01") & (train_full[DATE_COL] <= "2019-12-31")].copy()

    x_tr, y_tr = tune_train[feature_cols], tune_train["Revenue"]
    x_va, y_va = tune_val[feature_cols], tune_val["Revenue"]
    sample_weight = peak_weights(y_tr, tune_train[DATE_COL])

    base_model = fit_log_model(xgb_reg(seed=SEED + 301, n_estimators=700), x_tr, y_tr)
    raw_model = xgb_reg(seed=SEED + 302, n_estimators=800, objective="reg:squarederror", learning_rate=0.03)
    q80_model = xgb_reg(seed=SEED + 303, n_estimators=800, objective="reg:quantileerror", quantile_alpha=0.80)
    raw_model.fit(x_tr, y_tr, sample_weight=sample_weight)
    q80_model.fit(x_tr, y_tr, sample_weight=sample_weight)

    peak_label = (y_tr >= y_tr.quantile(0.90)).astype(int)
    peak_clf = xgb_clf(seed=SEED + 304)
    peak_clf.set_params(scale_pos_weight=float((peak_label == 0).sum() / max((peak_label == 1).sum(), 1)))
    peak_clf.fit(x_tr, peak_label)

    base_pred = predict_log_model(base_model, x_va)
    raw_pred = raw_model.predict(x_va).clip(min=0)
    q80_pred = q80_model.predict(x_va).clip(min=0)
    peak_prob = peak_clf.predict_proba(x_va)[:, 1]

    y_val = y_va.to_numpy()
    q90 = y_va.quantile(0.90)
    local_peak = (
        (y_va > y_va.shift(1))
        & (y_va > y_va.shift(-1))
        & (y_va >= q90)
    ).fillna(False).to_numpy()
    fast_growth = (y_va.diff() >= y_va.diff()[y_va.diff() > 0].quantile(0.90)).fillna(False).to_numpy()
    base_mae = mean_absolute_error(y_val, base_pred)

    rows = []
    for threshold in np.linspace(0.01, 0.50, 25):
        for gate_weight in np.linspace(0.10, 0.90, 17):
            pred = apply_ver3_revenue(base_pred, raw_pred, q80_pred, peak_prob, threshold, gate_weight)
            active = peak_prob >= threshold
            rows.append(
                {
                    "gate_threshold": float(threshold),
                    "gate_weight": float(gate_weight),
                    "val_MAE": float(mean_absolute_error(y_val, pred)),
                    "val_RMSE": float(np.sqrt(mean_squared_error(y_val, pred))),
                    "val_R2": float(r2_score(y_val, pred)),
                    "val_top10_MAE": float(mean_absolute_error(y_val[y_val >= q90], pred[y_val >= q90])),
                    "val_local_peak_p90_MAE": float(mean_absolute_error(y_val[local_peak], pred[local_peak])),
                    "val_fast_growth_top10_MAE": float(mean_absolute_error(y_val[fast_growth], pred[fast_growth])),
                    "selected_rate": float(active.mean()),
                    "base_val_MAE": float(base_mae),
                }
            )

    grid = pd.DataFrame(rows)
    grid.to_csv(TABLE_DIR / "ver3_revenue_gate_2019_tuning_grid.csv", index=False)
    eligible = grid[grid["val_MAE"] <= base_mae * GATE_VALIDATION_MAE_TOLERANCE]
    if len(eligible):
        chosen = eligible.sort_values(["val_local_peak_p90_MAE", "val_MAE"]).iloc[[0]].copy()
        chosen["selection_rule"] = f"min local peak MAE with val_MAE <= base * {GATE_VALIDATION_MAE_TOLERANCE}"
    else:
        chosen = grid.sort_values("val_MAE").iloc[[0]].copy()
        chosen["selection_rule"] = "min validation MAE fallback"
    chosen.to_csv(TABLE_DIR / "ver3_revenue_gate_chosen_params.csv", index=False)
    return chosen


def train_revenue_models(train_df: pd.DataFrame, feature_cols: list[str], seed_offset: int = 0) -> dict:
    x = train_df[feature_cols]
    y = train_df["Revenue"]
    sample_weight = peak_weights(y, train_df[DATE_COL])

    models = {
        "base": fit_log_model(xgb_reg(seed=SEED + seed_offset + 1, n_estimators=900), x, y),
        "raw": xgb_reg(seed=SEED + seed_offset + 2, n_estimators=950, objective="reg:squarederror", learning_rate=0.03),
        "q80": xgb_reg(seed=SEED + seed_offset + 3, n_estimators=900, objective="reg:quantileerror", quantile_alpha=0.80),
    }
    models["raw"].fit(x, y, sample_weight=sample_weight)
    models["q80"].fit(x, y, sample_weight=sample_weight)

    peak_label = (y >= y.quantile(0.90)).astype(int)
    peak_clf = xgb_clf(seed=SEED + seed_offset + 4)
    peak_clf.set_params(scale_pos_weight=float((peak_label == 0).sum() / max((peak_label == 1).sum(), 1)))
    peak_clf.fit(x, peak_label)
    models["peak_clf"] = peak_clf
    return models


def train_cogs_models(train_df: pd.DataFrame, feature_cols: list[str], seed_offset: int = 100) -> dict:
    x = train_df[feature_cols]
    y = train_df["COGS"]
    sample_weight = peak_weights(train_df["Revenue"], train_df[DATE_COL])

    models = {
        "base": fit_log_model(xgb_reg(seed=SEED + seed_offset + 1, n_estimators=900), x, y),
        "raw": xgb_reg(seed=SEED + seed_offset + 2, n_estimators=900, objective="reg:squarederror", learning_rate=0.03),
        "q80": xgb_reg(seed=SEED + seed_offset + 3, n_estimators=850, objective="reg:quantileerror", quantile_alpha=0.80),
    }
    models["raw"].fit(x, y, sample_weight=sample_weight)
    models["q80"].fit(x, y, sample_weight=sample_weight)
    return models


def predict_ver3_models(
    x: pd.DataFrame,
    revenue_models: dict,
    cogs_models: dict,
    gate_threshold: float,
    gate_weight: float,
) -> dict[str, np.ndarray]:
    rev_base = predict_log_model(revenue_models["base"], x)
    rev_raw = revenue_models["raw"].predict(x).clip(min=0)
    rev_q80 = revenue_models["q80"].predict(x).clip(min=0)
    peak_prob = revenue_models["peak_clf"].predict_proba(x)[:, 1]
    rev_final = apply_ver3_revenue(rev_base, rev_raw, rev_q80, peak_prob, gate_threshold, gate_weight)

    cogs_base = predict_log_model(cogs_models["base"], x)
    cogs_raw = cogs_models["raw"].predict(x).clip(min=0)
    cogs_q80 = cogs_models["q80"].predict(x).clip(min=0)
    cogs_upper = UPPER_RAW_WEIGHT * cogs_raw + UPPER_Q80_WEIGHT * cogs_q80
    gate = make_gate(peak_prob, gate_threshold, gate_weight)
    cogs_final = ((1.0 - gate) * cogs_base + gate * cogs_upper).clip(min=0)

    return {
        "Revenue_base_log_xgb": rev_base,
        "Revenue_raw_weighted": rev_raw,
        "Revenue_q80": rev_q80,
        "Revenue_peak_probability": peak_prob,
        "Revenue_pred": rev_final,
        "COGS_base_log_xgb": cogs_base,
        "COGS_raw_weighted": cogs_raw,
        "COGS_q80": cogs_q80,
        "COGS_pred": cogs_final,
    }


def make_seasonal_lookup(real_history: pd.DataFrame) -> dict:
    temp = real_history[[DATE_COL, *TARGETS]].copy()
    temp["season_month"] = temp[DATE_COL].dt.month
    temp["season_day"] = temp[DATE_COL].dt.day
    lookup = {}
    global_stats = temp[TARGETS].agg(["mean", "std"]).to_dict()
    for target in TARGETS:
        stats = temp.groupby(["season_month", "season_day"])[target].agg(["mean", "std"])
        lookup[target] = {"stats": stats, "global": global_stats[target]}
    return lookup


def add_fixed_seasonal_priors(row: pd.DataFrame, seasonal_lookup: dict) -> pd.DataFrame:
    month = int(row["month"].iloc[0])
    day = int(row["dayofmonth"].iloc[0])
    for target in TARGETS:
        stats = seasonal_lookup[target]["stats"]
        global_stats = seasonal_lookup[target]["global"]
        if (month, day) in stats.index:
            mean_value = stats.loc[(month, day), "mean"]
            std_value = stats.loc[(month, day), "std"]
        else:
            mean_value = global_stats["mean"]
            std_value = global_stats["std"]
        row[f"{target}_md_prior_mean"] = mean_value if pd.notna(mean_value) else global_stats["mean"]
        row[f"{target}_md_prior_std"] = std_value if pd.notna(std_value) else global_stats["std"]
    return row


def make_recursive_row(
    buffer: pd.DataFrame,
    date: pd.Timestamp,
    feature_cols: list[str],
    seasonal_lookup: dict,
    base_year: int,
) -> pd.DataFrame:
    tail = buffer.tail(500).copy()
    tmp = pd.concat(
        [
            tail[[DATE_COL, *TARGETS]],
            pd.DataFrame({DATE_COL: [date], "Revenue": [np.nan], "COGS": [np.nan]}),
        ],
        ignore_index=True,
    ).sort_values(DATE_COL).reset_index(drop=True)
    tmp = FB.add_calendar_features(tmp, base_year=base_year)
    tmp = FB.add_holiday_features(tmp)
    tmp = FB.add_target_lag_features(tmp)
    row = tmp[tmp[DATE_COL].eq(date)].tail(1).copy()
    row = add_fixed_seasonal_priors(row, seasonal_lookup)
    for col in feature_cols:
        if col not in row.columns:
            row[col] = np.nan
    return row[feature_cols]


def predict_future_recursive(
    real_history: pd.DataFrame,
    future_dates: pd.Series,
    feature_cols: list[str],
    revenue_models: dict,
    cogs_models: dict,
    gate_threshold: float,
    gate_weight: float,
) -> pd.DataFrame:
    buffer = real_history[[DATE_COL, *TARGETS]].copy().sort_values(DATE_COL).reset_index(drop=True)
    base_year = int(buffer[DATE_COL].dt.year.min())
    seasonal_lookup = make_seasonal_lookup(buffer)
    rows = []
    for date in pd.to_datetime(future_dates):
        x_row = make_recursive_row(buffer, date, feature_cols, seasonal_lookup, base_year)
        pred = predict_ver3_models(x_row, revenue_models, cogs_models, gate_threshold, gate_weight)
        revenue = float(pred["Revenue_pred"][0])
        cogs = float(pred["COGS_pred"][0])
        peak_prob = float(pred["Revenue_peak_probability"][0])
        gate_active = int(peak_prob >= gate_threshold)
        rows.append(
            {
                DATE_COL: date,
                "Revenue": revenue,
                "COGS": cogs,
                "Revenue_base_log_xgb": float(pred["Revenue_base_log_xgb"][0]),
                "Revenue_raw_weighted": float(pred["Revenue_raw_weighted"][0]),
                "Revenue_q80": float(pred["Revenue_q80"][0]),
                "Revenue_peak_probability": peak_prob,
                "peak_gate_active": gate_active,
            }
        )
        buffer = pd.concat(
            [buffer, pd.DataFrame([{DATE_COL: date, "Revenue": revenue, "COGS": cogs}])],
            ignore_index=True,
        )
    return pd.DataFrame(rows)


def save_figures(test_pred: pd.DataFrame, future_pred: pd.DataFrame, history: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(test_pred[DATE_COL], test_pred["Revenue"], lw=0.9, label="Actual Revenue")
    plt.plot(test_pred[DATE_COL], test_pred["Revenue_pred"], lw=0.85, label="Ver3 Revenue")
    plt.plot(test_pred[DATE_COL], test_pred["Revenue_base_log_xgb"], lw=0.75, alpha=0.75, label="Base log-XGB")
    plt.title("Forecast Ver3 - Test Revenue Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "test_revenue_prediction_ver3.png", dpi=150)
    plt.close()

    ver2_path = ROOT / "outputs" / "forecast_ver2" / "tables" / "test_predictions_ver2_gpu.csv"
    if ver2_path.exists():
        ver2 = pd.read_csv(ver2_path, parse_dates=[DATE_COL])
        merged = test_pred.merge(ver2[[DATE_COL, "Revenue_pred"]], on=DATE_COL, how="left", suffixes=("_ver3", "_ver2"))
        plt.figure(figsize=(16, 6))
        plt.plot(merged[DATE_COL], merged["Revenue"], lw=0.9, label="Actual Revenue")
        plt.plot(merged[DATE_COL], merged["Revenue_pred_ver2"], lw=0.8, label="Ver2")
        plt.plot(merged[DATE_COL], merged["Revenue_pred_ver3"], lw=0.8, label="Ver3")
        plt.title("Revenue Test Comparison - Ver2 vs Ver3")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "test_revenue_ver2_vs_ver3.png", dpi=150)
        plt.close()

    local = metrics_df[(metrics_df["target"].eq("Revenue")) & (metrics_df["segment"].isin(["all", "local_peak_p90", "fast_growth_top10"]))]
    pivot = local.pivot_table(index="model", columns="segment", values="MAE", aggfunc="first")
    order = [m for m in ["ver2_reference", "base_log_xgb", "raw_weighted", "q80", "ver3_final"] if m in pivot.index]
    pivot.loc[order].plot(kind="bar", figsize=(12, 5))
    plt.title("Revenue MAE by Segment - Ver3")
    plt.ylabel("MAE")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "revenue_mae_by_segment_ver3.png", dpi=150)
    plt.close()

    hist_tail = history[history[DATE_COL] >= history[DATE_COL].max() - pd.Timedelta(days=365 * 3)]
    plt.figure(figsize=(16, 6))
    plt.plot(hist_tail[DATE_COL], hist_tail["Revenue"], lw=0.8, label="Historical Revenue")
    plt.plot(future_pred[DATE_COL], future_pred["Revenue"], lw=0.9, label="Ver3 Future Revenue")
    plt.title("Forecast Ver3 - Future Revenue 2023-2024")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "future_revenue_forecast_ver3.png", dpi=150)
    plt.close()


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "model_core_train.parquet")
    valid = pd.read_parquet(DATA_DIR / "model_core_valid.parquet")
    test = pd.read_parquet(DATA_DIR / "model_core_test.parquet")
    history = pd.read_parquet(DATA_DIR / "model_core_history.parquet")
    future_template = pd.read_parquet(DATA_DIR / "model_core_future.parquet")

    train_full = pd.concat([train, valid], ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    for frame in [train_full, test, history, future_template]:
        frame[DATE_COL] = pd.to_datetime(frame[DATE_COL])

    feature_cols = [c for c in train_full.columns if c not in DROP_COLS]
    print(f"Device: {DEVICE}")
    print(f"Train fit: {len(train_full):,} rows | {train_full[DATE_COL].min().date()} -> {train_full[DATE_COL].max().date()}")
    print(f"Test     : {len(test):,} rows | {test[DATE_COL].min().date()} -> {test[DATE_COL].max().date()}")
    print(f"Future   : {len(future_template):,} rows | {future_template[DATE_COL].min().date()} -> {future_template[DATE_COL].max().date()}")
    print(f"Features : {len(feature_cols)}")

    chosen_gate = tune_revenue_gate_2019(train_full, feature_cols)
    gate_threshold = float(chosen_gate["gate_threshold"].iloc[0])
    gate_weight = float(chosen_gate["gate_weight"].iloc[0])
    print(f"Chosen gate: threshold={gate_threshold:.6f}, gate_weight={gate_weight:.3f}")

    revenue_models = train_revenue_models(train_full, feature_cols, seed_offset=0)
    cogs_models = train_cogs_models(train_full, feature_cols, seed_offset=100)
    test_preds = predict_ver3_models(test[feature_cols], revenue_models, cogs_models, gate_threshold, gate_weight)

    test_pred = test[[DATE_COL, *TARGETS]].copy()
    for key, value in test_preds.items():
        test_pred[key] = value
    test_pred["peak_gate_active"] = (test_pred["Revenue_peak_probability"] >= gate_threshold).astype(int)
    test_pred.to_csv(TABLE_DIR / "test_predictions_ver3.csv", index=False)

    metric_rows: list[dict] = []
    for col, name in [
        ("Revenue_base_log_xgb", "base_log_xgb"),
        ("Revenue_raw_weighted", "raw_weighted"),
        ("Revenue_q80", "q80"),
        ("Revenue_pred", "ver3_final"),
    ]:
        add_metric_rows(metric_rows, test_pred.rename(columns={col: "tmp_pred"}), "Revenue", "tmp_pred", name)
    for col, name in [
        ("COGS_base_log_xgb", "cogs_base_log_xgb"),
        ("COGS_raw_weighted", "cogs_raw_weighted"),
        ("COGS_q80", "cogs_q80"),
        ("COGS_pred", "cogs_ver3_final"),
    ]:
        add_metric_rows(metric_rows, test_pred.rename(columns={col: "tmp_pred"}), "COGS", "tmp_pred", name)

    ver2_path = ROOT / "outputs" / "forecast_ver2" / "tables" / "test_predictions_ver2_gpu.csv"
    if ver2_path.exists():
        ver2 = pd.read_csv(ver2_path, parse_dates=[DATE_COL])
        tmp = test[[DATE_COL, *TARGETS]].merge(ver2[[DATE_COL, "Revenue_pred", "COGS_pred"]], on=DATE_COL, how="left")
        add_metric_rows(metric_rows, tmp.rename(columns={"Revenue_pred": "tmp_pred"}), "Revenue", "tmp_pred", "ver2_reference")
        add_metric_rows(metric_rows, tmp.rename(columns={"COGS_pred": "tmp_pred"}), "COGS", "tmp_pred", "ver2_reference")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(TABLE_DIR / "metrics_ver3.csv", index=False)

    # Refit final models on full known history for future submission.
    full_feature_cols = [c for c in history.columns if c not in DROP_COLS]
    missing = [c for c in feature_cols if c not in full_feature_cols]
    if missing:
        raise ValueError(f"History is missing expected feature columns: {missing[:10]}")
    history = history.sort_values(DATE_COL).reset_index(drop=True)
    revenue_models_full = train_revenue_models(history, feature_cols, seed_offset=500)
    cogs_models_full = train_cogs_models(history, feature_cols, seed_offset=600)
    future_pred = predict_future_recursive(
        real_history=history,
        future_dates=future_template[DATE_COL],
        feature_cols=feature_cols,
        revenue_models=revenue_models_full,
        cogs_models=cogs_models_full,
        gate_threshold=gate_threshold,
        gate_weight=gate_weight,
    )
    future_pred.to_csv(TABLE_DIR / "future_predictions_ver3.csv", index=False)

    submission = future_pred[[DATE_COL, "Revenue", "COGS"]].copy()
    submission["Revenue"] = submission["Revenue"].clip(lower=0).round(2)
    submission["COGS"] = submission["COGS"].clip(lower=0).round(2)
    submission_out = submission.copy()
    submission_out[DATE_COL] = submission_out[DATE_COL].dt.strftime("%Y-%m-%d")
    submission_out.to_csv(OUT_DIR / "submission_ver3.csv", index=False)

    save_figures(test_pred, future_pred, history[[DATE_COL, *TARGETS]], metrics_df)

    for name, model in revenue_models.items():
        joblib.dump(model, MODEL_DIR / f"revenue_{name}.joblib")
    for name, model in cogs_models.items():
        joblib.dump(model, MODEL_DIR / f"cogs_{name}.joblib")

    summary = {
        "device": DEVICE,
        "train_rows": int(len(train_full)),
        "test_rows": int(len(test)),
        "future_rows": int(len(future_pred)),
        "feature_count": int(len(feature_cols)),
        "upper_tail": {
            "raw_weighted": UPPER_RAW_WEIGHT,
            "q80": UPPER_Q80_WEIGHT,
        },
        "gate": {
            "threshold": gate_threshold,
            "gate_weight": gate_weight,
            "selected_rate_test": float(test_pred["peak_gate_active"].mean()),
            "selected_rate_future": float(future_pred["peak_gate_active"].mean()),
            "selection": chosen_gate.to_dict(orient="records")[0],
        },
        "outputs": {
            "metrics": str(TABLE_DIR / "metrics_ver3.csv"),
            "test_predictions": str(TABLE_DIR / "test_predictions_ver3.csv"),
            "future_predictions": str(TABLE_DIR / "future_predictions_ver3.csv"),
            "submission": str(OUT_DIR / "submission_ver3.csv"),
        },
    }
    (OUT_DIR / "run_summary_ver3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    display_cols = ["target", "model", "segment", "n", "MAE", "RMSE", "R2", "bias", "pred_over_actual"]
    print("\nRevenue metrics:")
    print(
        metrics_df[
            (metrics_df["target"].eq("Revenue"))
            & (metrics_df["segment"].isin(["all", "local_peak_p90", "fast_growth_top10"]))
        ][display_cols].to_string(index=False)
    )
    print("\nCOGS metrics:")
    print(
        metrics_df[
            (metrics_df["target"].eq("COGS"))
            & (metrics_df["segment"].isin(["all", "local_peak_p90", "fast_growth_top10"]))
        ][display_cols].to_string(index=False)
    )
    print(f"\nSaved Ver3 outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
