# Datathon 2026 - Leakage-Safe Sales Forecasting Architecture

## 0. Problem

Forecast daily `Revenue` and `COGS` for the dates in `sample_submission.csv`
(`2023-01-01` to `2024-07-01`). The submission format is:

```text
Date,Revenue,COGS
```

Metrics to report: MAE, RMSE, and R2. All models must use a fixed random seed
and provide SHAP or feature-importance based explanations.

## 1. Time Split

Use strict time splits with no overlap:

```text
train : 2012-07-04 -> 2017-12-31  (2,007 rows)
valid : 2018-01-01 -> 2019-12-31  (730 rows)
test  : 2020-01-01 -> 2022-12-31  (1,096 rows, final benchmark)
future: 2023-01-01 -> 2024-07-01  (548 rows, submission target)
```

The `test` split is a benchmark split. Do not tune hyperparameters on it.

## 2. Leakage Policy

For a row dated `t`, every non-calendar feature must be available no later than
the end of day `t-1`.

Allowed for final future forecasting:

- deterministic calendar features for date `t`
- deterministic holiday/sale-day features for date `t`
- shifted `Revenue` and `COGS` lag features
- rolling statistics computed on `Revenue.shift(1)` and `COGS.shift(1)`
- same-month/day historical priors computed without future target values

Not allowed directly for future forecasting:

- `orders`, `order_items`, `payments`, or `web_traffic` observed on date `t`
- `returns`, `reviews`, `shipments`, or inventory snapshots observed on or after `t`
- any unshifted target or rolling feature
- future auxiliary data that is not present in the competition input

## 3. Dataset Families

All generated files live in `dataset/`.

### Core Dataset

Use this branch for the primary model and final submission:

```text
dataset/model_core_train.parquet
dataset/model_core_valid.parquet
dataset/model_core_test.parquet
dataset/model_core_future.parquet
```

Core features include:

- calendar fields: `year`, `year_index`, `quarter`, `month`, `weekofyear`,
  `dayofyear`, `dayofmonth`, `dayofweek`, month/quarter/year boundary flags
- cyclic fields: `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `year_sin`,
  `year_cos`
- deterministic holiday fields: `days_to_tet`, `days_after_tet`,
  `tet_proximity`, `is_tet_week`, `is_pre_tet_2w`, `is_pre_tet_month`,
  `is_fixed_holiday`, `is_sale_day`, `is_1111`, `is_1212`,
  `is_black_friday`
- target lags: `Revenue_lag_*`, `COGS_lag_*`
- target rolling stats: `roll*_mean`, `roll*_std`, `roll*_min`,
  `roll*_max`, `roll*_median`
- target EWMA features
- lag/rolling ratios
- gross margin lag fields
- same-month/day seasonal priors

`model_core_future.parquet` is not for training. It is a future-date template
for recursive prediction.

### Lagged Exogenous Dataset

Use this branch only for experiments:

```text
dataset/model_lagged_exog_train.parquet
dataset/model_lagged_exog_valid.parquet
dataset/model_lagged_exog_test.parquet
```

It adds prior-day aggregates from orders, order items, payments, web traffic,
returns, reviews, shipments, and inventory. These features are shifted to avoid
historical leakage, but they are not directly available for 2023-2024. Do not
use them for the final submission unless a separate future exogenous forecasting
layer is built.

## 4. Primary Model

Train two separate models:

```text
model_revenue: features -> log1p(Revenue)
model_cogs   : features -> log1p(COGS)
```

Preferred estimator:

```text
LightGBM LGBMRegressor
```

Fallback if LightGBM is unavailable:

```text
sklearn HistGradientBoostingRegressor
```

Use validation for early stopping and tuning. After benchmarking, retrain final
models on the full historical period through `2022-12-31`, using the selected
iteration count from validation.

## 5. Evaluation

Evaluate both targets on `valid` while tuning. Evaluate once on `test` after the
model configuration is finalized.

Report:

```text
MAE
RMSE
R2
```

Use clipped non-negative predictions after inverting `log1p`.

## 6. Future Forecast

Use recursive one-step forecasting:

```text
1. Start with true history through 2022-12-31.
2. Predict 2023-01-01.
3. Append predicted Revenue and COGS to the history buffer.
4. Recompute lag/rolling features for 2023-01-02.
5. Repeat through 2024-07-01.
```

After prediction, post-process COGS:

```text
COGS = min(COGS, Revenue * 0.95)
COGS = max(COGS, 0)
Revenue = max(Revenue, 0)
```

## 7. SHAP and Feature Importance

For LightGBM, generate:

- SHAP summary plot
- SHAP importance bar plot
- SHAP waterfall plot for a notable date if available

Always save feature importance CSVs, even if SHAP is unavailable.

## 8. Execution Order

```text
1. Rebuild core and exogenous datasets:
   python src/08_build_modeling_dataset.py

2. Run Kaggle training notebook:
   notebooks/03_lgb_core_kaggle.ipynb

3. Inspect valid metrics.

4. After model configuration is fixed, inspect test benchmark metrics once.

5. Submit outputs/submission.csv.
```

## 9. Requirements

Core:

```text
pandas
numpy
scikit-learn
lightgbm
pyarrow
matplotlib
shap
joblib
```

Prophet is optional and should only be tried after the LightGBM recursive model
is benchmarked.
