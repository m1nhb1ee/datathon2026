from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

import build_modeling_dataset as FB


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "forecast_ver3"
TABLE_DIR = OUT_DIR / "tables"

DATE_COL = "Date"
TARGETS = ["Revenue", "COGS"]


def xgb_reg(
    *,
    seed: int,
    n_estimators: int = 900,
    objective: str = "reg:squarederror",
    learning_rate: float = 0.035,
    quantile_alpha: float | None = None,
) -> XGBRegressor:
    params = dict(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=learning_rate,
        subsample=0.88,
        colsample_bytree=0.88,
        min_child_weight=4,
        reg_alpha=0.05,
        reg_lambda=0.8,
        objective=objective,
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=0,
        verbosity=0,
    )
    if quantile_alpha is not None:
        params["quantile_alpha"] = quantile_alpha
    return XGBRegressor(**params)


def xgb_clf(seed: int) -> XGBClassifier:
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
        device="cuda",
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


def make_gate(prob: np.ndarray, threshold: float, gate_weight: float) -> np.ndarray:
    return (prob >= threshold).astype(float) * gate_weight


def apply_ver3(base_pred, raw_pred, q80_pred, peak_prob, gate_threshold, gate_weight, raw_weight=0.7):
    upper = raw_weight * raw_pred + (1.0 - raw_weight) * q80_pred
    gate = make_gate(peak_prob, gate_threshold, gate_weight)
    return ((1.0 - gate) * base_pred + gate * upper).clip(min=0)


def train_target_models(history: pd.DataFrame, feature_cols: list[str], target: str, seed_offset: int):
    x = history[feature_cols]
    y = history[target]
    sample_w = peak_weights(y if target == "Revenue" else history["Revenue"], history[DATE_COL])

    base = fit_log_model(xgb_reg(seed=SEED + seed_offset + 1, n_estimators=900), x, y)
    raw = xgb_reg(seed=SEED + seed_offset + 2, n_estimators=950, objective="reg:squarederror", learning_rate=0.03)
    q80 = xgb_reg(seed=SEED + seed_offset + 3, n_estimators=900, objective="reg:quantileerror", quantile_alpha=0.80)
    raw.fit(x, y, sample_weight=sample_w)
    q80.fit(x, y, sample_weight=sample_w)

    return {"base": base, "raw": raw, "q80": q80}


def train_gate(history: pd.DataFrame, feature_cols: list[str], seed_offset: int):
    x = history[feature_cols]
    y = history["Revenue"]
    peak_label = (y >= y.quantile(0.90)).astype(int)
    clf = xgb_clf(seed=SEED + seed_offset + 4)
    clf.set_params(scale_pos_weight=float((peak_label == 0).sum() / max((peak_label == 1).sum(), 1)))
    clf.fit(x, peak_label)
    return clf


def get_feature_cols(history: pd.DataFrame):
    return [c for c in history.columns if c not in [DATE_COL, "Revenue", "COGS"]]


def make_recursive_row(buffer: pd.DataFrame, date: pd.Timestamp, feature_cols: list[str], base_year: int):
    tail = buffer.tail(500).copy()
    tmp = pd.concat(
        [tail[[DATE_COL, "Revenue", "COGS"]], pd.DataFrame({DATE_COL: [date], "Revenue": [np.nan], "COGS": [np.nan]})],
        ignore_index=True,
    ).sort_values(DATE_COL).reset_index(drop=True)
    tmp = FB.add_calendar_features(tmp, base_year=base_year)
    tmp = FB.add_holiday_features(tmp)
    tmp = FB.add_target_lag_features(tmp)
    row = tmp[tmp[DATE_COL].eq(date)].tail(1).copy()
    # Seasonal priors fallback from history mean/std by month-day
    md = buffer.copy()
    md["m"] = md[DATE_COL].dt.month
    md["d"] = md[DATE_COL].dt.day
    m = int(date.month)
    d = int(date.day)
    sub = md[(md["m"] == m) & (md["d"] == d)]
    rev_mean = sub["Revenue"].mean() if len(sub) else md["Revenue"].mean()
    rev_std = sub["Revenue"].std() if len(sub) else md["Revenue"].std()
    cogs_mean = sub["COGS"].mean() if len(sub) else md["COGS"].mean()
    cogs_std = sub["COGS"].std() if len(sub) else md["COGS"].std()
    row["Revenue_md_prior_mean"] = rev_mean
    row["Revenue_md_prior_std"] = rev_std
    row["COGS_md_prior_mean"] = cogs_mean
    row["COGS_md_prior_std"] = cogs_std
    for c in feature_cols:
        if c not in row.columns:
            row[c] = np.nan
    return row[feature_cols]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    base_year = int(sales[DATE_COL].dt.year.min())
    history = FB.add_calendar_features(sales.copy(), base_year=base_year)
    history = FB.add_holiday_features(history)
    history = FB.add_target_lag_features(history)

    # add seasonal priors from full history
    hist_md = sales.copy()
    hist_md["m"] = hist_md[DATE_COL].dt.month
    hist_md["d"] = hist_md[DATE_COL].dt.day
    for t in TARGETS:
        stats = hist_md.groupby(["m", "d"])[t].agg(["mean", "std"]).reset_index().rename(
            columns={"mean": f"{t}_md_prior_mean", "std": f"{t}_md_prior_std"}
        )
        history["m"] = history[DATE_COL].dt.month
        history["d"] = history[DATE_COL].dt.day
        history = history.merge(stats, on=["m", "d"], how="left")
        history = history.drop(columns=["m", "d"])

    feature_cols = get_feature_cols(history)
    train_hist = history.dropna(subset=[c for c in feature_cols if c.endswith("_lag_365")] + TARGETS).reset_index(drop=True)

    # Use best gate config from previous ver3 run if available
    gate_threshold = 0.050833333333333335
    gate_weight = 0.55
    summary_path = OUT_DIR / "run_summary_ver3.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        gate_threshold = float(s.get("gate", {}).get("threshold", gate_threshold))
        gate_weight = float(s.get("gate", {}).get("gate_weight", gate_weight))

    rev_models = train_target_models(train_hist, feature_cols, "Revenue", seed_offset=0)
    cogs_models = train_target_models(train_hist, feature_cols, "COGS", seed_offset=100)
    gate_clf = train_gate(train_hist, feature_cols, seed_offset=0)

    # recursive future prediction
    buffer = sales[[DATE_COL, "Revenue", "COGS"]].copy().sort_values(DATE_COL).reset_index(drop=True)
    rows = []
    for date in sample[DATE_COL]:
        x_row = make_recursive_row(buffer, date, feature_cols, base_year)
        rev_base = predict_log_model(rev_models["base"], x_row)[0]
        rev_raw = float(rev_models["raw"].predict(x_row)[0])
        rev_q80 = float(rev_models["q80"].predict(x_row)[0])
        p_gate = float(gate_clf.predict_proba(x_row)[:, 1][0])
        rev = float(apply_ver3(np.array([rev_base]), np.array([rev_raw]), np.array([rev_q80]), np.array([p_gate]), gate_threshold, gate_weight)[0])

        cogs_base = predict_log_model(cogs_models["base"], x_row)[0]
        cogs_raw = float(cogs_models["raw"].predict(x_row)[0])
        cogs_q80 = float(cogs_models["q80"].predict(x_row)[0])
        cogs = float(apply_ver3(np.array([cogs_base]), np.array([cogs_raw]), np.array([cogs_q80]), np.array([p_gate]), gate_threshold, gate_weight)[0])
        cogs = min(cogs, rev * 0.98)

        rows.append(
            {
                DATE_COL: date,
                "Revenue": max(rev, 0.0),
                "COGS": max(cogs, 0.0),
                "Revenue_peak_probability": p_gate,
                "peak_gate_active": int(p_gate >= gate_threshold),
            }
        )
        buffer = pd.concat([buffer, pd.DataFrame([{DATE_COL: date, "Revenue": rev, "COGS": cogs}])], ignore_index=True)

    future_pred = pd.DataFrame(rows)
    future_pred.to_csv(TABLE_DIR / "future_predictions_full_sales_2023_2024.csv", index=False)

    submission = future_pred[[DATE_COL, "Revenue", "COGS"]].copy()
    submission[DATE_COL] = submission[DATE_COL].dt.strftime("%Y-%m-%d")
    submission["Revenue"] = submission["Revenue"].round(2)
    submission["COGS"] = submission["COGS"].round(2)
    submission.to_csv(OUT_DIR / "submission_full_sales_2023_2024.csv", index=False)

    run = {
        "mode": "train_full_sales_only",
        "sales_rows": int(len(sales)),
        "future_rows": int(len(future_pred)),
        "train_start": str(sales[DATE_COL].min().date()),
        "train_end": str(sales[DATE_COL].max().date()),
        "future_start": str(sample[DATE_COL].min().date()),
        "future_end": str(sample[DATE_COL].max().date()),
        "feature_count": int(len(feature_cols)),
        "gate_threshold": gate_threshold,
        "gate_weight": gate_weight,
        "outputs": {
            "future_predictions": str(TABLE_DIR / "future_predictions_full_sales_2023_2024.csv"),
            "submission": str(OUT_DIR / "submission_full_sales_2023_2024.csv"),
        },
    }
    (OUT_DIR / "run_summary_full_sales_2023_2024.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")
    print(future_pred.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
