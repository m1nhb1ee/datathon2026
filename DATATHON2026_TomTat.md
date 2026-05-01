# Tóm tắt báo cáo: Dự báo doanh thu E-commerce thời trang Việt Nam

## Pipeline & Cross-Validation

### Pipeline rõ ràng (9 bước)
1. Load & clean data
2. Feature engineering (56 features safe)
3. Tune LightGBM & XGBoost via Optuna
4. Walk-forward CV → OOF predictions
5. Tune head weights + NNLS blend + COGS constraints
6. Final retrain
7. SHAP explainability
8. Recursive forecast
9. Post-process constraints

### Cross-validation đúng
- **Phương pháp:** Expanding-window walk-forward (3 annual folds: 2020, 2021, 2022)
- **Đặc điểm:** Không shuffling, không future leakage
- **Tập train:** ~2,737 → ~3,468 rows tăng dần theo năm

---

## Mô hình & Thời gian

### Kiến trúc hai đầu LightGBM
- **Log head (55%):** `log1p(revenue)` → stability trong log-space
- **Ratio head (45%):** `revenue / anchor` → adaptive với scale
- **XGBoost challenger:** Diversity khác — `reg:absoluteerror` objective

### Ensemble NNLS
- Blend ba thành phần: LGB blend, XGB, anchor baseline
- Non-negative weights tối ưu từ OOF predictions
- Giải quyết: `min ||Aw - y||²` subject to `w ≥ 0, Σw = 1`

### Trend tách riêu
- Ước tính CAGR từ smooth 2020-2022 (robust median 3 estimator)
- Nhân độc lập: `(1+g)^t` (2023: t=1, 2024: t=2)
- Capped +25%/yr để tránh extrapolation quá tham lam

---

## SHAP & Giải thích cụ thể

### Top-10 features (TreeExplainer trên 2022)

| Rank | Feature | Loại | Ý nghĩa |
|------|---------|------|---------|
| 1 | `lag_1` | Observed | AR(1) — bộ nhớ ngày trước |
| 2 | `roll_mean_7` | Observed | Trend 7 ngày — mịn nhiễu tuần |
| 3 | `anchor_same_day_ly` | Future | Same-day YoY — ưu thế trung kỳ |
| 4 | `lag_7` | Observed | Weekly seasonality trực tiếp |
| 5 | `tet_decay_pre` | Future | Pre-Tết surge — spike lớn nhất |
| 6 | `lag_365` | Observed | Annual seasonality |
| 7 | `fourier_y_sin1` | Future | Pattern năm (smooth sin) |
| 8 | `roll_mean_30` | Observed | Baseline tháng |
| 9 | `sale_decay_11` | Future | 11/11 event proximity |
| 10 | `days_to_tet` | Future | Countdown Tết |

### Output visualization
- `shap_bar.png` — mean |SHAP| feature importance
- `shap_beeswarm.png` — direction & magnitude từng dự báo

---

## Feature Engineering (56 features)

### Phân loại Data Contract
| Category | Count | Deploy? | Ví dụ |
|----------|-------|---------|-------|
| known_future | 34 | ✅ | Calendar, Fourier, decay, promo flags |
| lagged_observed | 22 | ✅ | lag_{1,7,14,30,90,365}, roll_mean/std/min/max |
| target_recursive | 0 | ❌ | — |
| historical_only | 9 | ❌ | Sessions, page_views, inventory |

### Nhóm features
- **Calendar & Fourier:** dayofweek, month, quarter, is_weekend; Fourier harmonics (weekly k=1,2, monthly k=1, yearly k=1,2)
- **Vietnamese Holiday Decay:** Exponential decay cho Tết (pre-30d, post-15d), 11/11, 12/12, Black Friday (14d)
- **Lag & Rolling:** Raw lags {1,7,14,30,90,365}; rolling mean/std/min/max {7,14,30,90}
- **Anchor Features:** Same month-day YoY (weighted), monthly prior-year mean, global expanding mean

---

## Ràng buộc (tuân thủ đầy đủ)

### Pipeline post-processing (OOF-tuned)

| # | Constraint | Phương pháp | Tuning |
|----|------------|-----------|--------|
| 1 | Revenue ≥ 0 | Hard clip: `max(pred, 0)` | Fixed |
| 2 | COGS ≥ 0 | Hard clip: `max(cogs_pred, 0)` | Fixed |
| 3 | COGS ≤ Revenue | Clamp rolling ratio | window∈{30,60,90,120} |
| 4 | Revenue smooth | Rolling mean (exclude holidays) | window∈{1,3,5} |
| 5 | Growth `x(1+g)^t` | Median 3 CAGR estimators | Capped +25%/yr |

### Growth coefficient estimators
| Estimator | Formula | Window |
|-----------|---------|--------|
| YoY 2021→2022 | `L₂₀₂₂ / L₂₀₂₁ - 1` | 1-year |
| YoY 2020→2022 | `(L₂₀₂₂ / L₂₀₂₀)^0.5 - 1` | 2-year CAGR |
| Log-linear trend | `exp(slope log(Lᵧ) on year) - 1` | All available |
| **Median (used)** | median of above 3 | Robust |

---

## Hyperparameter Optimization

### Optuna TPE (60 trials LGB, 40 trials XGB)

| Parameter | LGB range | XGB range |
|-----------|-----------|-----------|
| num_leaves / max_depth | 16–256 / 4–12 | — / 3–10 |
| learning_rate | 1e-3–0.15 (log) | 1e-3–0.15 (log) |
| subsample | 0.5–1.0 | 0.5–1.0 |
| colsample_bytree | 0.4–1.0 | 0.4–1.0 |
| reg_alpha / reg_lambda | 1e-4–10 (log) | 1e-4–10 (log) |
| min_child_samples | 5–100 | 1–50 (min_child_weight) |
| gamma | — | 0.0–5.0 |

- **LGB:** Median Pruner (10 startup trials, 20 warmup steps)
- **XGB:** TPE without pruning
- **Early stopping:** 50 rounds trên fold gần nhất

---

## Kết quả & Submission

### Reproducibility checklist
✅ **Random seed:** `RANDOM_SEED = 42` (numpy, Optuna, LGB, XGB)  
✅ **Train data:** `sales.csv` (04/07/2012 – 31/12/2022), no external  
✅ **Source code:** `test.py` — single file, self-contained  
✅ **SHAP output:** `shap_bar.png`, `shap_beeswarm.png`  
✅ **Submission:** `submission.csv` (548 rows)  
✅ **Dependencies:** lightgbm, xgboost, optuna, shap, scipy, scikit-learn, pandas, numpy

### Output submission.csv
```
Date,Revenue,COGS
2023-01-01,26607.20,2585.15
2023-01-02,1007.89,163.00
...
2024-07-01,xxxxx.xx,xxxxx.xx
```
- **Period:** 01/01/2023 – 01/07/2024 (548 rows)
- **Guarantee:** COGS ≤ Revenue 100% tất cả rows

---

## Tóm tắt chính

| Tiêu chí | Kết quả |
|----------|---------|
| **Pipeline** | 9 bước rõ ràng: load → feature → tune → CV → blend → retrain → SHAP → forecast → post-process |
| **Cross-validation** | Walk-forward expanding (3 annual folds), no leakage, no shuffling |
| **Time series** | Lag {1,7,14,30,90,365}, Fourier multi-period, holiday decay, YoY anchor, growth trend multiplicative |
| **Model explainability** | SHAP TreeExplainer — top feature lag_1 (AR memory), visualize beeswarm+bar |
| **Constraints** | COGS ≤ Revenue, revenue ≥ 0, smooth (non-events), growth capped +25%/yr — all OOF-tuned |
| **Ensemble** | LGB dual-head (log 55% + ratio 45%) + XGB challenger → NNLS blend (non-negative weights) |
| **Reproducibility** | seed=42, single file, full source, no external data |

---

**Report:** DATATHON 2026 — THE GRIDBREAKER | Vòng 1 | Phần 3: Mô hình Dự báo Doanh thu  
**Team:** VinTelligence / VinUniversity | 2026
