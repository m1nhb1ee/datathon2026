# Model V2

Train = train + valid: 2012-07-04 -> 2019-12-31, 2,737 rows  

Test benchmark = 2020-01-01 -> 2022-12-31, 1,096 rows  

Features: 171  
Model chính final: XGBoost GPU  
Model phụ để so sánh: LightGBM  

Có thêm COGS ratio model để diagnostic, nhưng final tốt nhất vẫn là XGBoost direct.  


# Metrics V2

| Target  | Model final  | MAE        | RMSE       | R2     |
|--------|-------------|------------|------------|--------|
| Revenue | XGBoost GPU | 523,688.63 | 715,713.61 | 0.8136 |
| COGS    | XGBoost GPU | 454,452.55 | 626,159.57 | 0.8077 |


# So sánh với V1

| Target  | V1 MAE      | V2 MAE      | Cải thiện               |
|--------|-------------|-------------|--------------------------|
| Revenue | 592,084.57 | 523,688.63 | -68,395.95 (~11.6%)       |
| COGS    | 560,273.42 | 454,452.55 | -105,820.88 (~18.9%)      |


# R2 cải thiện

- Revenue: 0.7736 -> 0.8136  
- COGS   : 0.7391 -> 0.8077  