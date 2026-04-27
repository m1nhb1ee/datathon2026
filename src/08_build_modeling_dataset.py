"""
Build leakage-safe modeling datasets for daily Revenue/COGS forecasting.

Outputs
-------
Primary, future-safe core datasets:
    dataset/model_core_history.parquet
    dataset/model_core_train.parquet
    dataset/model_core_valid.parquet
    dataset/model_core_test.parquet
    dataset/model_core_future.parquet

Historical experiment datasets with lagged exogenous signals:
    dataset/model_lagged_exog_history.parquet
    dataset/model_lagged_exog_train.parquet
    dataset/model_lagged_exog_valid.parquet
    dataset/model_lagged_exog_test.parquet

Metadata:
    dataset/feature_manifest.json

Leakage policy
--------------
For a row dated t, every non-calendar feature must be observable no later than
the end of day t-1. Future submission rows intentionally contain only known
calendar/seasonal fields; lag/rolling fields must be filled recursively during
forecasting after each predicted day is appended to history.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "dataset"

TARGETS = ["Revenue", "COGS"]
DATE_COL = "Date"

TRAIN_END = pd.Timestamp("2017-12-31")
VALID_START = pd.Timestamp("2018-01-01")
VALID_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2022-12-31")

LAGS = [1, 2, 3, 7, 14, 21, 28, 30, 56, 90, 180, 364, 365]
ROLL_WINDOWS = [3, 7, 14, 28, 30, 56, 90, 180, 365]
EWMA_SPANS = [7, 14, 30, 90]

TET_DATES = {
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
}
FIXED_HOLIDAYS = {(1, 1), (4, 30), (5, 1), (9, 2)}
SALE_DAYS = {(11, 11), (12, 12)}

warnings.simplefilter("ignore", PerformanceWarning)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(RAW_DATA_DIR / path, **kwargs)


def add_calendar_features(df: pd.DataFrame, base_year: int | None = None) -> pd.DataFrame:
    """Add features known for any future date."""
    out = df.copy()
    date = out[DATE_COL]
    iso = date.dt.isocalendar()
    if base_year is None:
        base_year = int(date.dt.year.min())

    out["year"] = date.dt.year
    out["year_index"] = out["year"] - base_year
    out["quarter"] = date.dt.quarter
    out["month"] = date.dt.month
    out["weekofyear"] = iso.week.astype("int16")
    out["dayofyear"] = date.dt.dayofyear
    out["dayofmonth"] = date.dt.day
    out["dayofweek"] = date.dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype("int8")
    out["is_month_start"] = date.dt.is_month_start.astype("int8")
    out["is_month_end"] = date.dt.is_month_end.astype("int8")
    out["is_quarter_start"] = date.dt.is_quarter_start.astype("int8")
    out["is_quarter_end"] = date.dt.is_quarter_end.astype("int8")
    out["is_year_start"] = date.dt.is_year_start.astype("int8")
    out["is_year_end"] = date.dt.is_year_end.astype("int8")

    for period, col in [(7, "dow"), (12, "month"), (365.25, "year")]:
        source = out["dayofweek"] if col == "dow" else out["month"] if col == "month" else out["dayofyear"]
        out[f"{col}_sin"] = np.sin(2 * np.pi * source / period)
        out[f"{col}_cos"] = np.cos(2 * np.pi * source / period)

    return out


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic Vietnamese calendar and retail sale-day features."""
    out = df.copy()
    dates = out[DATE_COL]
    tet_dates = pd.to_datetime(list(TET_DATES.values()))

    def tet_distance(date: pd.Timestamp) -> tuple[int, int]:
        deltas = (tet_dates - date).days
        future = deltas[deltas >= 0]
        past = deltas[deltas < 0]
        days_to = int(future.min()) if len(future) else 999
        days_after = int(-past.max()) if len(past) else 999
        return days_to, days_after

    proximity = dates.apply(tet_distance)
    out["days_to_tet"] = [item[0] for item in proximity]
    out["days_after_tet"] = [item[1] for item in proximity]
    out["tet_proximity"] = np.minimum(out["days_to_tet"], out["days_after_tet"])
    out["is_tet_week"] = (
        (out["days_to_tet"].le(7)) | (out["days_after_tet"].le(7))
    ).astype("int8")
    out["is_pre_tet_2w"] = (
        out["days_to_tet"].between(1, 14, inclusive="both")
    ).astype("int8")
    out["is_pre_tet_month"] = (
        out["days_to_tet"].between(1, 30, inclusive="both")
    ).astype("int8")
    out["is_fixed_holiday"] = dates.apply(
        lambda date: int((date.month, date.day) in FIXED_HOLIDAYS)
    ).astype("int8")
    out["is_sale_day"] = dates.apply(
        lambda date: int((date.month, date.day) in SALE_DAYS)
    ).astype("int8")
    out["is_1111"] = ((out["month"] == 11) & (out["dayofmonth"] == 11)).astype("int8")
    out["is_1212"] = ((out["month"] == 12) & (out["dayofmonth"] == 12)).astype("int8")
    out["is_black_friday"] = dates.apply(
        lambda date: int(date.month == 11 and date.weekday() == 4 and (date.day - 1) // 7 == 3)
    ).astype("int8")
    return out


def add_target_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target lags and rolling statistics using shifted history only."""
    out = df.copy()

    for target in TARGETS:
        shifted = out[target].shift(1)

        for lag in LAGS:
            out[f"{target}_lag_{lag}"] = out[target].shift(lag)

        for window in ROLL_WINDOWS:
            roll = shifted.rolling(window=window, min_periods=max(2, min(7, window)))
            out[f"{target}_roll{window}_mean"] = roll.mean()
            out[f"{target}_roll{window}_std"] = roll.std()
            out[f"{target}_roll{window}_min"] = roll.min()
            out[f"{target}_roll{window}_max"] = roll.max()
            out[f"{target}_roll{window}_median"] = roll.median()

        for span in EWMA_SPANS:
            out[f"{target}_ewm{span}_mean"] = shifted.ewm(span=span, adjust=False, min_periods=2).mean()

        out[f"{target}_lag1_vs_roll7"] = out[f"{target}_lag_1"] / out[f"{target}_roll7_mean"]
        out[f"{target}_roll7_vs_roll30"] = out[f"{target}_roll7_mean"] / out[f"{target}_roll30_mean"]
        out[f"{target}_roll30_vs_roll90"] = out[f"{target}_roll30_mean"] / out[f"{target}_roll90_mean"]
        out[f"{target}_roll90_vs_roll365"] = out[f"{target}_roll90_mean"] / out[f"{target}_roll365_mean"]

    out["gross_margin_lag_1"] = out["Revenue_lag_1"] - out["COGS_lag_1"]
    out["gross_margin_rate_lag_1"] = out["gross_margin_lag_1"] / out["Revenue_lag_1"]
    out["revenue_cogs_ratio_lag_1"] = out["Revenue_lag_1"] / out["COGS_lag_1"]

    return out.replace([np.inf, -np.inf], np.nan)


def add_prior_seasonal_profile(history: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add same-month/day priors without using future rows.

    Historical rows get expanding statistics from previous years only.
    Future rows get statistics from all historical rows through 2022-12-31.
    """
    hist = history.copy()
    fut = future.copy()

    for frame in (hist, fut):
        frame["season_month"] = frame[DATE_COL].dt.month
        frame["season_day"] = frame[DATE_COL].dt.day

    future_global_stats = hist[TARGETS].agg(["mean", "std"]).to_dict()

    for target in TARGETS:
        group = hist.groupby(["season_month", "season_day"], group_keys=False)[target]
        hist[f"{target}_md_prior_mean"] = group.transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        hist[f"{target}_md_prior_std"] = group.transform(
            lambda s: s.shift(1).expanding(min_periods=2).std()
        )
        hist_global_mean_prior = hist[target].shift(1).expanding(min_periods=1).mean()
        hist_global_std_prior = hist[target].shift(1).expanding(min_periods=2).std()
        hist[f"{target}_md_prior_mean"] = hist[f"{target}_md_prior_mean"].fillna(
            hist_global_mean_prior
        )
        hist[f"{target}_md_prior_std"] = hist[f"{target}_md_prior_std"].fillna(
            hist_global_std_prior
        )

        md_stats = (
            hist.groupby(["season_month", "season_day"])[target]
            .agg(["mean", "std"])
            .rename(
                columns={
                    "mean": f"{target}_md_prior_mean",
                    "std": f"{target}_md_prior_std",
                }
            )
            .reset_index()
        )
        fut = fut.merge(md_stats, on=["season_month", "season_day"], how="left")

        fut[f"{target}_md_prior_mean"] = fut[f"{target}_md_prior_mean"].fillna(
            future_global_stats[target]["mean"]
        )
        fut[f"{target}_md_prior_std"] = fut[f"{target}_md_prior_std"].fillna(
            future_global_stats[target]["std"]
        )

    hist = hist.drop(columns=["season_month", "season_day"])
    fut = fut.drop(columns=["season_month", "season_day"])
    return hist, fut


def build_core_datasets() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sales = read_csv("sales.csv", parse_dates=[DATE_COL]).sort_values(DATE_COL)
    sales = sales.reset_index(drop=True)

    future = read_csv("sample_submission.csv", parse_dates=[DATE_COL])[[DATE_COL]]
    future = future.sort_values(DATE_COL).reset_index(drop=True)
    for target in TARGETS:
        future[target] = np.nan

    base_year = int(sales[DATE_COL].dt.year.min())
    history = add_calendar_features(sales, base_year=base_year)
    future = add_calendar_features(future, base_year=base_year)
    history = add_holiday_features(history)
    future = add_holiday_features(future)

    history = add_target_lag_features(history)
    history, future = add_prior_seasonal_profile(history, future)

    # Keep the same columns in the future template. Lag/rolling columns stay
    # empty and are meant to be populated by a recursive forecaster.
    for col in history.columns:
        if col not in future.columns:
            future[col] = np.nan
    future = future[history.columns]

    feature_cols = [
        col
        for col in history.columns
        if col not in {DATE_COL, *TARGETS}
    ]

    return history, future, feature_cols


def one_hot_counts(
    df: pd.DataFrame,
    date_col: str,
    cat_col: str,
    prefix: str,
    normalize: bool = False,
) -> pd.DataFrame:
    pivot = pd.crosstab(df[date_col], df[cat_col], normalize="index" if normalize else False)
    pivot.columns = [f"{prefix}_{str(col).lower()}_{'share' if normalize else 'count'}" for col in pivot.columns]
    return pivot.reset_index().rename(columns={date_col: DATE_COL})


def build_lagged_exogenous_features(date_index: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build historical exogenous signals shifted by one day.

    These are useful for backtests and optional experiments. They are not part
    of the primary final-forecast dataset because 2023-2024 values are unknown.
    """
    daily = date_index[[DATE_COL]].copy()

    orders = read_csv("orders.csv", parse_dates=["order_date"])
    orders = orders.rename(columns={"order_date": DATE_COL})
    order_daily = (
        orders.groupby(DATE_COL)
        .agg(
            exog_orders_count=("order_id", "count"),
            exog_unique_customers=("customer_id", "nunique"),
            exog_unique_zips=("zip", "nunique"),
        )
        .reset_index()
    )
    for col in ["order_status", "payment_method", "device_type", "order_source"]:
        order_daily = order_daily.merge(
            one_hot_counts(orders, DATE_COL, col, f"exog_order_{col}", normalize=True),
            on=DATE_COL,
            how="outer",
        )

    order_items = read_csv("order_items.csv")
    products = read_csv("products.csv")
    items = order_items.merge(orders[[DATE_COL, "order_id"]], on="order_id", how="left")
    items = items.merge(products[["product_id", "category", "segment"]], on="product_id", how="left")
    items["gross_sales"] = items["quantity"] * items["unit_price"]
    items["has_promo"] = items["promo_id"].notna().astype("int8")
    items["has_second_promo"] = items["promo_id_2"].notna().astype("int8")
    item_daily = (
        items.groupby(DATE_COL)
        .agg(
            exog_item_lines=("order_id", "size"),
            exog_units=("quantity", "sum"),
            exog_gross_sales=("gross_sales", "sum"),
            exog_discount_sum=("discount_amount", "sum"),
            exog_avg_unit_price=("unit_price", "mean"),
            exog_unique_products=("product_id", "nunique"),
            exog_promo_line_share=("has_promo", "mean"),
            exog_second_promo_line_share=("has_second_promo", "mean"),
        )
        .reset_index()
    )

    payments = read_csv("payments.csv")
    payments = payments.merge(orders[[DATE_COL, "order_id"]], on="order_id", how="left")
    payment_daily = (
        payments.groupby(DATE_COL)
        .agg(
            exog_payment_value_sum=("payment_value", "sum"),
            exog_payment_value_mean=("payment_value", "mean"),
            exog_installments_mean=("installments", "mean"),
        )
        .reset_index()
    )

    traffic = read_csv("web_traffic.csv", parse_dates=["date"]).rename(columns={"date": DATE_COL})
    traffic_daily = (
        traffic.groupby(DATE_COL)
        .agg(
            exog_sessions=("sessions", "sum"),
            exog_unique_visitors=("unique_visitors", "sum"),
            exog_page_views=("page_views", "sum"),
            exog_bounce_rate=("bounce_rate", "mean"),
            exog_avg_session_duration_sec=("avg_session_duration_sec", "mean"),
        )
        .reset_index()
    )
    traffic_daily = traffic_daily.merge(
        one_hot_counts(traffic, DATE_COL, "traffic_source", "exog_traffic_source", normalize=True),
        on=DATE_COL,
        how="outer",
    )

    returns = read_csv("returns.csv", parse_dates=["return_date"]).rename(columns={"return_date": DATE_COL})
    return_daily = (
        returns.groupby(DATE_COL)
        .agg(
            exog_return_count=("return_id", "count"),
            exog_return_quantity=("return_quantity", "sum"),
            exog_refund_amount=("refund_amount", "sum"),
        )
        .reset_index()
    )
    return_daily = return_daily.merge(
        one_hot_counts(returns, DATE_COL, "return_reason", "exog_return_reason", normalize=True),
        on=DATE_COL,
        how="outer",
    )

    reviews = read_csv("reviews.csv", parse_dates=["review_date"]).rename(columns={"review_date": DATE_COL})
    reviews["is_low_rating"] = reviews["rating"].le(2).astype("int8")
    review_daily = (
        reviews.groupby(DATE_COL)
        .agg(
            exog_review_count=("review_id", "count"),
            exog_rating_mean=("rating", "mean"),
            exog_low_rating_share=("is_low_rating", "mean"),
        )
        .reset_index()
    )

    shipments = read_csv("shipments.csv", parse_dates=["ship_date", "delivery_date"])
    ship_daily = (
        shipments.groupby("ship_date")
        .agg(
            exog_shipped_orders=("order_id", "count"),
            exog_shipping_fee_sum=("shipping_fee", "sum"),
            exog_shipping_fee_mean=("shipping_fee", "mean"),
        )
        .reset_index()
        .rename(columns={"ship_date": DATE_COL})
    )
    delivery_daily = (
        shipments.groupby("delivery_date")
        .agg(exog_delivered_orders=("order_id", "count"))
        .reset_index()
        .rename(columns={"delivery_date": DATE_COL})
    )

    inventory = read_csv("inventory.csv", parse_dates=["snapshot_date"])
    inventory_daily = (
        inventory.groupby("snapshot_date")
        .agg(
            exog_stock_on_hand=("stock_on_hand", "sum"),
            exog_units_received=("units_received", "sum"),
            exog_units_sold_inventory=("units_sold", "sum"),
            exog_stockout_days=("stockout_days", "sum"),
            exog_days_of_supply_mean=("days_of_supply", "mean"),
            exog_fill_rate_mean=("fill_rate", "mean"),
            exog_stockout_flag_rate=("stockout_flag", "mean"),
            exog_overstock_flag_rate=("overstock_flag", "mean"),
            exog_reorder_flag_rate=("reorder_flag", "mean"),
            exog_sell_through_rate_mean=("sell_through_rate", "mean"),
        )
        .reset_index()
        .rename(columns={"snapshot_date": DATE_COL})
    )

    components = [
        order_daily,
        item_daily,
        payment_daily,
        traffic_daily,
        return_daily,
        review_daily,
        ship_daily,
        delivery_daily,
    ]
    for component in components:
        daily = daily.merge(component, on=DATE_COL, how="left")

    # Inventory snapshots are month-end. For date t, use the latest snapshot
    # known through t-1, implemented by forward fill followed by one-day shift.
    inv_daily = date_index[[DATE_COL]].merge(inventory_daily, on=DATE_COL, how="left")
    inv_feature_cols = [col for col in inv_daily.columns if col != DATE_COL]
    inv_daily[inv_feature_cols] = inv_daily[inv_feature_cols].ffill().shift(1)
    daily = daily.merge(inv_daily, on=DATE_COL, how="left")

    exog_cols = [col for col in daily.columns if col != DATE_COL]
    count_like = [col for col in exog_cols if col.endswith("_count") or col in {
        "exog_orders_count",
        "exog_unique_customers",
        "exog_unique_zips",
        "exog_item_lines",
        "exog_units",
        "exog_unique_products",
        "exog_return_count",
        "exog_return_quantity",
        "exog_review_count",
        "exog_shipped_orders",
        "exog_delivered_orders",
    }]
    daily[count_like] = daily[count_like].fillna(0)

    # Every exogenous signal is shifted so the t row sees at most t-1.
    daily[exog_cols] = daily[exog_cols].shift(1)

    # Add short rolling summaries over shifted exogenous signals.
    for col in list(exog_cols):
        if daily[col].dtype.kind not in "biufc":
            continue
        shifted = daily[col]
        for window in [7, 30]:
            daily[f"{col}_roll{window}_mean"] = shifted.rolling(window, min_periods=2).mean()

    exog_cols = [col for col in daily.columns if col != DATE_COL]
    return daily, exog_cols


def split_named(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = df[df[DATE_COL].le(TRAIN_END)].copy()
    valid = df[df[DATE_COL].between(VALID_START, VALID_END, inclusive="both")].copy()
    test = df[df[DATE_COL].between(TEST_START, TEST_END, inclusive="both")].copy()

    return {
        "history": df.copy(),
        "train": train,
        "valid": valid,
        "test": test,
    }


def validate_time_splits(name: str, splits: dict[str, pd.DataFrame]) -> None:
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    assert train[DATE_COL].max() <= TRAIN_END
    assert valid[DATE_COL].min() >= VALID_START
    assert valid[DATE_COL].max() <= VALID_END
    assert test[DATE_COL].min() >= TEST_START
    assert test[DATE_COL].max() <= TEST_END
    assert set(train[DATE_COL]).isdisjoint(set(valid[DATE_COL]))
    assert set(train[DATE_COL]).isdisjoint(set(test[DATE_COL]))
    assert set(valid[DATE_COL]).isdisjoint(set(test[DATE_COL]))

    print(f"{name} split validation:")
    print(f"- history rows   : {len(splits['history']):,}")
    print(f"- train rows     : {len(train):,} ({train[DATE_COL].min().date()} -> {train[DATE_COL].max().date()})")
    print(f"- valid rows     : {len(valid):,} ({valid[DATE_COL].min().date()} -> {valid[DATE_COL].max().date()})")
    print(f"- test rows      : {len(test):,} ({test[DATE_COL].min().date()} -> {test[DATE_COL].max().date()})")
    print("- Assertions passed: no overlap and strict time ordering.\n")


def save_split_family(prefix: str, splits: dict[str, pd.DataFrame]) -> None:
    for split_name, frame in splits.items():
        frame.to_parquet(OUTPUT_DIR / f"{prefix}_{split_name}.parquet", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    core_history, core_future, core_features = build_core_datasets()
    core_splits = split_named(core_history)
    validate_time_splits("Core future-safe", core_splits)
    save_split_family("model_core", core_splits)
    core_future.to_parquet(OUTPUT_DIR / "model_core_future.parquet", index=False)

    exog_daily, exog_features = build_lagged_exogenous_features(core_history[[DATE_COL]])
    exog_history = core_history.merge(exog_daily, on=DATE_COL, how="left")
    exog_splits = split_named(exog_history)
    validate_time_splits("Lagged exogenous historical", exog_splits)
    save_split_family("model_lagged_exog", exog_splits)

    manifest = {
        "date_column": DATE_COL,
        "targets": TARGETS,
        "primary_recommended_dataset": "model_core_train.parquet",
        "primary_future_template": "model_core_future.parquet",
        "core_feature_count": len(core_features),
        "core_features": core_features,
        "lagged_exog_feature_count": len(exog_features),
        "lagged_exog_features": exog_features,
        "leakage_policy": {
            "core": "Calendar features are known in advance; target lag/rolling features use shifted history only; seasonal profile for historical rows uses prior years only.",
            "lagged_exog": "Transaction, traffic, returns, reviews, shipments, and inventory signals are shifted so row t only sees data observable through t-1.",
            "future": "Exogenous future values are not available for 2023-2024. Train final model on core features unless a separate recursive exogenous forecasting layer is added.",
        },
        "split_rule": {
            "train": "beginning -> 2017-12-31 inclusive",
            "valid": "2018-01-01 -> 2019-12-31 inclusive",
            "test": "2020-01-01 -> 2022-12-31 inclusive",
        },
    }
    (OUTPUT_DIR / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Saved modeling datasets to:")
    print(f"- {OUTPUT_DIR / 'model_core_train.parquet'}")
    print(f"- {OUTPUT_DIR / 'model_core_test.parquet'}")
    print(f"- {OUTPUT_DIR / 'model_core_future.parquet'}")
    print(f"- {OUTPUT_DIR / 'model_lagged_exog_train.parquet'}")
    print(f"- {OUTPUT_DIR / 'feature_manifest.json'}")


if __name__ == "__main__":
    main()
