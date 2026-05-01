# Datathon 2026: Vietnamese Fashion E-commerce Revenue Forecasting

**A comprehensive machine learning solution for revenue and COGS prediction in Vietnamese fashion e-commerce.**

---

## 📋 Project Overview

This project forecasts daily revenue and cost of goods sold (COGS) for a Vietnamese fashion e-commerce business from January 1, 2023 to July 1, 2024. The challenge uses historical data from July 4, 2012 to December 31, 2022 to build a robust predictive model.

**Key Challenge:** Build a production-safe forecasting model that handles:
- Seasonal patterns (Vietnamese holidays, Tết, 11/11, Black Friday, etc.)
- Autoregressive dependencies
- Strict no-leakage cross-validation using expanding-window walk-forward CV
- Business constraints (non-negative revenue, COGS consistency)

---

## 🎯 Model Architecture

### Core Components

**Ensemble Strategy:** Three-way blend with NNLS (Non-Negative Least Squares)
1. **LightGBM Dual-Head Model (55% weight)**
   - Log head: Predicts `log1p(revenue)` for stability
   - Ratio head: Predicts `revenue / anchor` for adaptivity
   
2. **XGBoost Challenger (weight tuned)**
   - Alternative gradient boosting for diversity
   - Uses `reg:absoluteerror` objective
   
3. **Anchor Baseline (weight tuned)**
   - Same-day year-over-year historical reference
   - Provides domain-driven consistency

### Cross-Validation
- **Method:** Expanding-window walk-forward (3 annual folds)
- **Folds:** 2020 train/test, 2021 train/test, 2022 train/test
- **Protection:** No shuffling, no future leakage
- **Training growth:** ~2,737 → ~3,468 rows per fold

### Post-Processing Pipeline
```
Raw Predictions
    ↓
1. Hard clip: Revenue ≥ 0, COGS ≥ 0
    ↓
2. Smooth: Rolling mean (window tuned: 1-5, excluding holidays)
    ↓
3. Ratio constraint: COGS ≤ Revenue (tuned window: 30-120 days)
    ↓
4. Growth envelope: Multiply trend (CAGR capped at +25%/year)
    ↓
Final Submission
```

---

## 📊 Feature Engineering (56 Deployed Features)

### Feature Categories

| Category | Count | Type | Examples |
|----------|-------|------|----------|
| **Calendar & Seasonal** | 8 | Safe Future | Day-of-week, month, quarter, is_weekend |
| **Fourier Harmonics** | 6 | Safe Future | Weekly/monthly/yearly sine-cosine pairs |
| **Holiday Decay** | 10 | Safe Future | Tết, 11/11, 12/12, Black Friday exponential decay |
| **Lag Features** | 6 | Observed | lag_{1,7,14,30,90,365} |
| **Rolling Stats** | 12 | Observed | mean, std, min, max over windows {7,14,30,90} |
| **Anchor Features** | 8 | Safe Future | Same month-day YoY, monthly prior-year mean |

### Top-10 Most Important Features (SHAP)

| Rank | Feature | Type | Meaning |
|------|---------|------|---------|
| 1 | `lag_1` | Autoregressive | Previous day memory (AR(1)) |
| 2 | `roll_mean_7` | Seasonal | 7-day trend smoothing |
| 3 | `anchor_same_day_ly` | Anchor | Same-day year-over-year momentum |
| 4 | `lag_7` | Autoregressive | Weekly seasonality (direct) |
| 5 | `tet_decay_pre` | Holiday | Pre-Tết surge (largest spike) |
| 6 | `lag_365` | Autoregressive | Annual seasonality |
| 7 | `fourier_y_sin1` | Seasonal | Annual pattern (smooth sine) |
| 8 | `roll_mean_30` | Baseline | Monthly baseline |
| 9 | `sale_decay_11` | Holiday | 11/11 event proximity |
| 10 | `days_to_tet` | Holiday | Countdown to Lunar New Year |

---

## 📁 Project Structure

```
Datathon 2026/
│
├── data/                          # Input datasets
│   ├── Master Layer (lookups)
│   │   ├── products.csv
│   │   ├── customers.csv
│   │   ├── promotions.csv
│   │   └── geography.csv
│   │
│   ├── Transaction Layer
│   │   ├── orders.csv
│   │   ├── order_items.csv
│   │   ├── payments.csv
│   │   ├── shipments.csv
│   │   ├── returns.csv
│   │   └── reviews.csv
│   │
│   └── Analytical/Operational Layer
│       ├── sales.csv              # Training target (2012-07-04 to 2022-12-31)
│       ├── inventory.csv          # Monthly end-of-month snapshots
│       ├── web_traffic.csv        # Daily website sessions/page views
│       └── sample_submission.csv  # Required output format
│
├── src_part_2/                    # Exploratory & feature analysis
│   ├── 00_build_oat.py            # Build outer-aligned table (OAT)
│   ├── 01_act1_revenue_anatomy.ipynb       # Revenue decomposition analysis
│   ├── 02_act2_promo_economics.ipynb       # Promotional campaign analysis
│   ├── 03_act3_margin_trajectory.ipynb     # Margin & COGS trends
│   ├── 04_act4_promo_stockout.ipynb        # Promotion × stockout interaction
│   ├── 05_act5_promo_triage.ipynb          # Promotion triage & ROI
│   ├── calculate.py               # Feature calculation utilities
│   ├── report.ipynb               # Summary report notebook
│   └── oat.parquet                # Cached outer-aligned table
│
├── src_part_3/                    # Production pipeline
│   ├── predict.py                 # Final prediction & submission pipeline
│   ├── compare_benchmarks.py      # Model comparison utilities
│   └── submission.csv             # Final predictions for 2023-01-01 to 2024-07-01
│
├── outputs/                       # Generated artifacts
│   ├── charts/                    # SHAP visualizations & plots
│   │   ├── shap_bar.png           # Feature importance (mean |SHAP|)
│   │   └── shap_beeswarm.png      # SHAP force plot for feature directions
│   │
│   └── tables/                    # Analysis tables (CSV)
│       ├── promo_triage_table.csv           # Promotion ROI analysis
│       ├── margin_forecast_table.csv        # Margin trajectory forecast
│       ├── stockout_covid_period_summary.csv # Inventory risk during COVID
│       └── [13 other diagnostic tables]
│
├── dataset/                       # Metadata
│   └── feature_manifest.json      # Feature data contract (171 core features)
│
├── docs/                          # Documentation
│   ├── de_thi.md                  # Competition requirements & data spec
│   └── final_model_architecture.md # Model design decisions
│
├── DATATHON2026_TomTat.md         # Detailed technical summary
├── requirements.txt               # Python dependencies
├── submission.csv                 # Final submission file
└── README.md                      # This file

```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pandas>=2.0.0` — Data manipulation
- `numpy>=1.24.0` — Numerical computing
- `scikit-learn>=1.3.0` — ML utilities
- `lightgbm>=4.0.0` — Primary model
- `xgboost>=2.0.0` — Ensemble component
- `shap>=0.44.0` — Model explainability
- `matplotlib>=3.7.0` — Visualization

### 2. Run the Prediction Pipeline

```bash
python ./src_part_3/predict.py
```

This script:
1. Loads and cleans raw data from `data/`
2. Builds 56 safe-future features
3. Runs walk-forward cross-validation
4. Tunes model weights via Optuna
5. Retrains on full historical dataset (2012-2022)
6. Generates daily forecasts for 2023-01-01 to 2024-07-01
7. Applies business constraints (non-negativity, growth envelope, ratio consistency)
8. Outputs `submission.csv` in required format

### 3. Generate Analysis Reports

Exploratory analysis notebooks (Part 2):
```bash
jupyter notebook ./src_part_2/
```

Key notebooks:
- `01_act1_revenue_anatomy.ipynb` — Revenue decomposition by product, region, customer
- `02_act2_promo_economics.ipynb` — Promotional lift & ROI analysis
- `03_act3_margin_trajectory.ipynb` — COGS trends & margin compression
- `04_act4_promo_stockout.ipynb` — Stockout risk during promotions
- `05_act5_promo_triage.ipynb` — Promotion performance ranking

---

## 📈 Model Performance & Explainability

### Cross-Validation Results
- **Walk-forward folds:** 3 expanding annual windows (2020, 2021, 2022)
- **Out-of-fold (OOF) predictions:** Used to tune blend weights
- **NNLS optimization:** Solves `min ||Aw - y||²` subject to `w ≥ 0, Σw = 1`

### SHAP Explainability
- **Tree Explainer:** TreeExplainer on 2022 test data
- **Outputs:** 
  - `shap_bar.png` — Mean absolute SHAP per feature
  - `shap_beeswarm.png` — Distribution of feature impacts

---

## 📋 Data Specification

### Training Target: `sales.csv`
- **Columns:** Date, Revenue, COGS
- **Period:** 2012-07-04 to 2022-12-31 (3,737 days)
- **Granularity:** Daily aggregates

### Input Features
- **Master data:** Products (categories, sizes, colors), Customers, Promotions, Geography
- **Transactions:** Orders, Order items, Payments, Shipments, Returns, Reviews
- **Operational data:** Inventory (monthly snapshots), Web traffic (daily)

### Output Format: `submission.csv`
```
Date,Revenue,COGS
2023-01-01,value,value
2023-01-02,value,value
...
2024-07-01,value,value
```
- **Row count:** 548 days (2023-01-01 to 2024-07-01)
- **Revenue & COGS:** Float values ≥ 0

---

## 🔒 Design Decisions & Constraints

### Why No Lag-based Exogenous Features in Production?

Future order counts, traffic, returns, reviews, shipments, and inventory are **not available** for 2023-2024. Lagged versions (t-1, t-7) would leak information from the prediction period. ✅ Safe calendar, holiday, and anchor features only.

### Why Dual-Head LightGBM?
- **Log head:** Stabilizes high-volatility days, captures multiplicative patterns
- **Ratio head:** Adapts to scale, exploits anchor baseline
- **Result:** Stronger OOF predictions → better NNLS blend

### Why Growth Envelope?
- Prevents aggressive extrapolation
- Median of 3 CAGR estimators (robust to outliers)
- Capped at +25%/year (conservative, business-safe)
- Applied to trend component independently

### Why Expanding-Window CV?
- Simulates real deployment scenario (train on past, test on future)
- Prevents look-ahead bias & data leakage
- Reflects actual forecast difficulty over time

---

## 🛠️ Key Utilities

### `src_part_2/calculate.py`
Helper functions for feature engineering:
- Calendar features (day-of-week, month, holidays)
- Lag & rolling statistics
- Anchor features (YoY same-day, monthly priors)
- Holiday decay exponential functions

### `src_part_2/00_build_oat.py`
Constructs outer-aligned table (OAT) from raw transaction data:
- Aggregates orders, items, payments at daily level
- Computes daily revenue & COGS
- Handles missing dates (fills forward or clips)

### `src_part_3/compare_benchmarks.py`
Baseline & ablation experiments:
- Naive anchor model (same-day YoY)
- Seasonal exponential smoothing
- Individual LGB & XGB performance

---

## 📊 Output Artifacts

### Diagnostic Tables (in `outputs/tables/`)
- `promo_triage_table.csv` — Top promotional campaigns by revenue impact
- `margin_forecast_table.csv` — Predicted COGS, margin, and growth rates
- `stockout_covid_period_summary.csv` — Inventory constraints during demand shocks
- `order_status_profit_audit.csv` — Profit by order fulfillment status
- [13 additional diagnostic tables for deeper insights]

### Visualizations (in `outputs/charts/`)
- `shap_bar.png` — Mean absolute SHAP feature importance
- `shap_beeswarm.png` — Feature contribution direction (positive/negative)

---

## 📖 Technical Documentation

- **[DATATHON2026_TomTat.md](DATATHON2026_TomTat.md)** — Detailed technical report
  - Pipeline steps, cross-validation methodology
  - Feature engineering rationale
  - SHAP explainability results
  - Constraint post-processing

- **[docs/final_model_architecture.md](docs/final_model_architecture.md)** — Model architecture decisions
  - Choice of `baseline_plus_anchor_lgb` for deployment
  - Exogenous policy (why lagged features excluded)
  - Post-processing constraints

- **[docs/de_thi.md](docs/de_thi.md)** — Competition brief
  - Data schema and table descriptions
  - Submission requirements
  - Evaluation metrics

---

## ✅ Validation & Compliance

### No-Leakage Audit
- ✅ Calendar features: Known future (day-of-week, month)
- ✅ Fourier terms: Deterministic, no data leak
- ✅ Holiday decay: Published Vietnamese calendar
- ✅ Lag features: All from training period (t-1, t-7, etc.)
- ✅ Anchor features: Computed from historical data only
- ❌ Excluded: Web traffic, orders, returns, reviews, inventory (not available in future)

### Business Constraints
- ✅ Revenue ≥ 0 (hard clipped)
- ✅ COGS ≥ 0 (hard clipped)
- ✅ COGS ≤ Revenue (rolling ratio, tuned window)
- ✅ Smooth revenue (rolling mean, tuned window)
- ✅ Growth capped (+25%/year, to prevent extrapolation)

---

## 📝 License & Attribution

**Competition:** Datathon 2026: The GridBreaker  
**Host:** VinTelligence — VinUniversity Data Science & AI Club  
**Submission Date:** May 1, 2026  

---

## 📞 Support & Contact

For questions about the model, features, or results, refer to:
- `DATATHON2026_TomTat.md` — Detailed technical breakdown
- `docs/final_model_architecture.md` — Architecture decisions
- Jupyter notebooks in `src_part_2/` — Interactive analysis

---

**Last Updated:** May 1, 2026  
**Status:** ✅ Final Submission
