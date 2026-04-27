from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs" / "forecast_baseline"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

for path in [OUT_DIR, FIG_DIR, TABLE_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "Revenue", "COGS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[required].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    return out


def complete_years(df: pd.DataFrame) -> list[int]:
    years: list[int] = []
    for year, g in df.groupby(df["Date"].dt.year):
        start = pd.Timestamp(year=int(year), month=1, day=1)
        end = pd.Timestamp(year=int(year), month=12, day=31)
        expected_days = 366 if pd.Timestamp(year=int(year), month=12, day=31).is_leap_year else 365
        if g["Date"].min() <= start and g["Date"].max() >= end and g["Date"].nunique() >= expected_days:
            years.append(int(year))
    return years


def fit_seasonal_growth_baseline(history: pd.DataFrame) -> dict:
    hist = prepare_daily(history)
    hist["year"] = hist["Date"].dt.year
    hist["month"] = hist["Date"].dt.month
    hist["day"] = hist["Date"].dt.day

    years = complete_years(hist)
    if len(years) < 2:
        raise ValueError("Need at least two complete years to estimate YoY growth.")

    annual = hist.groupby("year")[["Revenue", "COGS"]].sum()
    daily_counts = hist.groupby("year")["Date"].nunique()
    annual_complete = annual.loc[years]

    base_year = int(years[-1])
    span = len(years) - 1
    growth_rev = float((annual_complete["Revenue"].iloc[-1] / annual_complete["Revenue"].iloc[0]) ** (1 / span))
    growth_cogs = float((annual_complete["COGS"].iloc[-1] / annual_complete["COGS"].iloc[0]) ** (1 / span))

    annual_means = hist.groupby("year")[["Revenue", "COGS"]].transform("mean")
    hist["rev_norm"] = hist["Revenue"] / annual_means["Revenue"]
    hist["cogs_norm"] = hist["COGS"] / annual_means["COGS"]

    seasonal = (
        hist.groupby(["month", "day"], as_index=False)[["rev_norm", "cogs_norm"]]
        .mean()
        .sort_values(["month", "day"])
        .reset_index(drop=True)
    )

    return {
        "base_year": base_year,
        "base_rev": float(annual.loc[base_year, "Revenue"] / daily_counts.loc[base_year]),
        "base_cogs": float(annual.loc[base_year, "COGS"] / daily_counts.loc[base_year]),
        "growth_rev": growth_rev,
        "growth_cogs": growth_cogs,
        "complete_years": years,
        "seasonal": seasonal,
    }


def predict_seasonal_growth(params: dict, dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    pred = pd.DataFrame({"Date": pd.to_datetime(dates)})
    pred["year"] = pred["Date"].dt.year
    pred["month"] = pred["Date"].dt.month
    pred["day"] = pred["Date"].dt.day
    pred["years_ahead"] = pred["year"] - int(params["base_year"])
    pred = pred.merge(params["seasonal"], on=["month", "day"], how="left")
    pred["rev_norm"] = pred["rev_norm"].fillna(1.0)
    pred["cogs_norm"] = pred["cogs_norm"].fillna(1.0)

    pred["Revenue_pred"] = params["base_rev"] * (params["growth_rev"] ** pred["years_ahead"]) * pred["rev_norm"]
    pred["COGS_pred_raw"] = params["base_cogs"] * (params["growth_cogs"] ** pred["years_ahead"]) * pred["cogs_norm"]

    pred["Revenue_pred"] = pred["Revenue_pred"].clip(lower=0)
    pred["COGS_pred"] = pred["COGS_pred_raw"].clip(lower=0, upper=pred["Revenue_pred"] * 0.995)
    return pred[["Date", "Revenue_pred", "COGS_pred"]]


def score_predictions(actual: pd.DataFrame, pred: pd.DataFrame, split: str, model_name: str) -> pd.DataFrame:
    merged = prepare_daily(actual).merge(pred, on="Date", how="inner")
    rows = []
    for target in ["Revenue", "COGS"]:
        y_true = merged[target].to_numpy()
        y_pred = merged[f"{target}_pred"].to_numpy()
        rows.append(
            {
                "split": split,
                "target": target,
                "model": model_name,
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "R2": r2_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def save_actual_vs_pred(test_pred: pd.DataFrame, target: str, path: Path) -> None:
    plt.figure(figsize=(16, 5))
    plt.plot(test_pred["Date"], test_pred[target], lw=0.8, label=f"Actual {target}")
    plt.plot(test_pred["Date"], test_pred[f"{target}_pred"], lw=0.9, label=f"Baseline {target}")
    plt.title(f"Baseline test prediction - {target}")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_model_comparison(test_pred: pd.DataFrame, path: Path) -> pd.DataFrame | None:
    ver2_path = ROOT / "outputs" / "forecast_ver2" / "tables" / "test_predictions_ver2_gpu.csv"
    if not ver2_path.exists():
        return None

    ver2 = pd.read_csv(ver2_path, parse_dates=["Date"])
    merged = test_pred.merge(ver2[["Date", "Revenue_pred", "COGS_pred"]], on="Date", suffixes=("_baseline", "_ver2"))

    plt.figure(figsize=(16, 5))
    plt.plot(merged["Date"], merged["Revenue"], lw=0.8, label="Actual Revenue")
    plt.plot(merged["Date"], merged["Revenue_pred_baseline"], lw=0.9, label="Baseline")
    plt.plot(merged["Date"], merged["Revenue_pred_ver2"], lw=0.9, label="Ver2 GPU")
    plt.title("Test Revenue comparison: actual vs baseline vs Ver2")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return merged


def save_metric_comparison(metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = [metrics]
    ver2_metrics_path = ROOT / "outputs" / "forecast_ver2" / "tables" / "metrics_ver2_gpu.csv"
    if ver2_metrics_path.exists():
        ver2_metrics = pd.read_csv(ver2_metrics_path)
        best_ver2 = ver2_metrics.loc[ver2_metrics.groupby("target")["MAE"].idxmin()].copy()
        best_ver2["model"] = "ver2_best"
        rows.append(best_ver2[metrics.columns])

    combined = pd.concat(rows, ignore_index=True)
    plot_df = combined.pivot_table(index="target", columns="model", values="MAE", aggfunc="first")

    ax = plot_df.plot(kind="bar", figsize=(9, 4))
    ax.set_title("Test MAE comparison")
    ax.set_ylabel("MAE")
    ax.set_xlabel("Target")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return combined


def save_future_plot(history: pd.DataFrame, future_pred: pd.DataFrame, path: Path) -> None:
    hist = prepare_daily(history)
    tail = hist[hist["Date"] >= hist["Date"].max() - pd.Timedelta(days=365 * 3)]

    plt.figure(figsize=(16, 5))
    plt.plot(tail["Date"], tail["Revenue"], lw=0.8, label="Historical Revenue")
    plt.plot(future_pred["Date"], future_pred["Revenue_pred"], lw=0.9, label="Baseline future forecast")
    plt.title("Future Revenue forecast: 2023-2024")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    valid = pd.read_parquet(DATA_DIR / "valid.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    train_fit = pd.concat([train, valid], ignore_index=True)
    train_fit = prepare_daily(train_fit)
    test = prepare_daily(test)

    params_test = fit_seasonal_growth_baseline(train_fit)
    test_forecast = predict_seasonal_growth(params_test, test["Date"])
    test_pred = test.merge(test_forecast, on="Date", how="left")

    metrics = score_predictions(test, test_forecast, split="test", model_name="seasonal_growth_baseline")
    metrics.to_csv(TABLE_DIR / "metrics_baseline_trainvalid_test.csv", index=False)
    test_pred.to_csv(TABLE_DIR / "test_predictions_baseline.csv", index=False)

    save_actual_vs_pred(test_pred, "Revenue", FIG_DIR / "test_revenue_actual_vs_baseline.png")
    save_actual_vs_pred(test_pred, "COGS", FIG_DIR / "test_cogs_actual_vs_baseline.png")
    save_model_comparison(test_pred, FIG_DIR / "test_revenue_baseline_vs_ver2.png")
    combined_metrics = save_metric_comparison(metrics, FIG_DIR / "test_mae_baseline_vs_ver2.png")
    combined_metrics.to_csv(TABLE_DIR / "metrics_baseline_vs_ver2.csv", index=False)

    history = pd.read_parquet(DATA_DIR / "model_core_history.parquet")[["Date", "Revenue", "COGS"]]
    future_dates = pd.read_parquet(DATA_DIR / "model_core_future.parquet")["Date"]
    history = prepare_daily(history)

    params_future = fit_seasonal_growth_baseline(history)
    future_forecast = predict_seasonal_growth(params_future, future_dates)
    submission = future_forecast.rename(columns={"Revenue_pred": "Revenue", "COGS_pred": "COGS"})
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    submission[["Date", "Revenue", "COGS"]].round({"Revenue": 2, "COGS": 2}).to_csv(
        OUT_DIR / "submission_baseline.csv", index=False
    )
    future_forecast.to_csv(TABLE_DIR / "future_predictions_baseline.csv", index=False)
    save_future_plot(history, future_forecast, FIG_DIR / "future_revenue_forecast_baseline.png")

    summary = {
        "train_fit_rows": int(len(train_fit)),
        "train_fit_min_date": str(train_fit["Date"].min().date()),
        "train_fit_max_date": str(train_fit["Date"].max().date()),
        "test_rows": int(len(test)),
        "test_min_date": str(test["Date"].min().date()),
        "test_max_date": str(test["Date"].max().date()),
        "future_rows": int(len(future_forecast)),
        "future_min_date": str(future_forecast["Date"].min().date()),
        "future_max_date": str(future_forecast["Date"].max().date()),
        "test_baseline_params": {
            "base_year": params_test["base_year"],
            "complete_years": params_test["complete_years"],
            "growth_rev": params_test["growth_rev"],
            "growth_cogs": params_test["growth_cogs"],
        },
        "future_baseline_params": {
            "base_year": params_future["base_year"],
            "complete_years": params_future["complete_years"],
            "growth_rev": params_future["growth_rev"],
            "growth_cogs": params_future["growth_cogs"],
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT_DIR / "run_summary_baseline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Baseline completed.")
    print(f"Train fit: {len(train_fit):,} rows | {train_fit['Date'].min().date()} -> {train_fit['Date'].max().date()}")
    print(f"Test     : {len(test):,} rows | {test['Date'].min().date()} -> {test['Date'].max().date()}")
    print(f"Future   : {len(future_forecast):,} rows | {future_forecast['Date'].min().date()} -> {future_forecast['Date'].max().date()}")
    print()
    print("Test baseline params:")
    print({k: summary["test_baseline_params"][k] for k in ["base_year", "growth_rev", "growth_cogs"]})
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
