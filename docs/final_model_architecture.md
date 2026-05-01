# Final Model Architecture

## Decision

Use `baseline_plus_anchor_lgb` as the final deploy-safe model.

## Pipeline

1. Build leakage-safe `model_core` datasets.
2. Audit split dates, target sanity, and deploy feature list.
3. Build no-lag calendar, holiday, seasonal prior, and hierarchical anchor features.
4. Train LightGBM log-target models for Revenue and COGS.
5. Tune anchor/LGB blend weights on walk-forward CV.
6. Evaluate once on 2020-2022 test.
7. Retrain on history through 2022-12-31 and forecast 2023-01-01 to 2024-07-01.

## Exogenous Policy

The lagged exogenous dataset remains useful for diagnostics and feature discovery, but is excluded from final submission because future orders, traffic, returns, reviews, shipments, and inventory are not available for 2023-2024.

## Post-processing

Revenue and COGS are clipped at zero. COGS is not capped by Revenue because historical target data contains valid COGS > Revenue rows.
