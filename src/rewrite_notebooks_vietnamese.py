import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CHARTS = ROOT / "outputs" / "charts"
TABLES = ROOT / "outputs" / "tables"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"}


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def money(x):
    return f"{x:,.0f} VND"


def pct(x, digits=1):
    return f"{x:.{digits}f}%"


def pp(x, digits=1):
    return f"{x:.{digits}f} điểm %"


oat = pd.read_parquet(SRC / "oat.parquet")
oat["order_date"] = pd.to_datetime(oat["order_date"])
triage = pd.read_csv(TABLES / "promo_triage_table.csv")
cohort = pd.read_csv(TABLES / "cohort_quality_table.csv")
scenario = pd.read_csv(TABLES / "promo_cut_scenario_table.csv")
forecast = pd.read_csv(TABLES / "margin_forecast_table.csv")
overlaps = pd.read_csv(TABLES / "promo_stockout_overlaps.csv")
risk = pd.read_csv(TABLES / "promo_inventory_risk_table.csv")

GP_COL = "gp_after_refund_shipping"

gross = oat["gross_revenue_line"].sum()
discount = oat["discount_amount"].sum()
refund = oat["refund_allocated"].sum()
shipping = oat["shipping_fee_allocated"].sum()
leakage = (discount + refund) / gross * 100
yearly = oat.groupby(oat["order_date"].dt.year).agg(net_revenue=("net_revenue", "sum"), gp=(GP_COL, "sum"))
yearly["margin"] = yearly["gp"] / yearly["net_revenue"] * 100
first_year, last_year = int(yearly.index.min()), int(yearly.index.max())
first_margin, last_margin = yearly.loc[first_year, "margin"], yearly.loc[last_year, "margin"]

neg_promo_pct = (triage["total_net_contribution"] < 0).mean() * 100
verdict_counts = triage["verdict"].value_counts().to_dict()
keep_count = verdict_counts.get("KEEP", 0)
cut_count = verdict_counts.get("CUT", 0)
reschedule_count = verdict_counts.get("RESCHEDULE", 0)
cut_gp = triage.loc[triage["verdict"].eq("CUT"), "total_net_contribution"].sum()

cohort_idx = cohort.set_index("metric")
promo_return = cohort_idx.loc["item_return_rate", "promo"] * 100
organic_return = cohort_idx.loc["item_return_rate", "organic"] * 100
promo_repeat = cohort_idx.loc["customer_repeat_rate", "promo"] * 100
organic_repeat = cohort_idx.loc["customer_repeat_rate", "organic"] * 100
promo_gp_customer = cohort_idx.loc["avg_lifetime_gp_per_customer", "promo"]
organic_gp_customer = cohort_idx.loc["avg_lifetime_gp_per_customer", "organic"]
gp_customer_gap = (1 - promo_gp_customer / organic_gp_customer) * 100

current = scenario[scenario["scenario"].eq("Current mix")].iloc[0]
cut0 = scenario[scenario["scenario"].eq("Cut negative promos")].iloc[0]
cut30 = scenario[scenario["scenario"].eq("Cut + 30% organic recapture")].iloc[0]
cut50 = scenario[scenario["scenario"].eq("Cut + 50% organic recapture")].iloc[0]
f2026 = forecast[forecast["quarter"].eq("2026Q4")].iloc[0]
f2027 = forecast[forecast["quarter"].eq("2027Q4")].iloc[0]
quarterly = oat.groupby(oat["order_date"].dt.to_period("Q")).agg(net_revenue=("net_revenue", "sum"), gp=(GP_COL, "sum"))
quarterly["gross_margin_pct"] = quarterly["gp"] / quarterly["net_revenue"] * 100
x = np.arange(len(quarterly))
trend_slope = np.polyfit(x, quarterly["gross_margin_pct"].to_numpy(), 1)[0]
last4_margin = quarterly["gross_margin_pct"].tail(4).mean()

top_risk = risk.iloc[0]
overlap_categories = ", ".join(sorted(overlaps["category"].unique().tolist()))
overlap_stockout = overlaps["stockout_product_share"].mean() * 100
overlap_fill = overlaps["avg_fill_rate"].mean() * 100
lost_gp = overlaps["estimated_lost_gp"].sum()


SETUP = code(
    r'''
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / "outputs").exists() and (ROOT.parent / "outputs").exists():
    ROOT = ROOT.parent
CHARTS = ROOT / "outputs" / "charts"
TABLES = ROOT / "outputs" / "tables"
'''
)


def display_cell(chart_name, width=950):
    return code(f'display(Image(str(CHARTS / "{chart_name}"), width={width}))')


NOTEBOOKS = {
    "01_act1_revenue_anatomy.ipynb": [
        md(f"""
# Act 1 - Bề mặt vs. thực tế

**Câu hỏi trung tâm:** Revenue tăng, nhưng bao nhiêu phần doanh thu thật sự còn lại sau discount, refund, shipping và COGS?

Act này giữ đúng kế hoạch: bắt đầu từ revenue anatomy để chứng minh vấn đề không nằm ở topline, mà nằm ở chất lượng doanh thu và biên lợi nhuận.
"""),
        SETUP,
        display_cell("chart1_revenue_anatomy.png"),
        code('display(pd.read_csv(TABLES / "category_double_loss_table.csv"))'),
        md(f"""
## Phân tích

Trong 10 năm, doanh nghiệp mất **{money(discount)}** qua discount leakage và **{money(refund)}** qua refund leakage. Shipping được phân bổ thêm **{money(shipping)}** vào từng dòng đơn hàng. Riêng discount và refund đã tương đương **{pct(leakage)} gross revenue**, trước khi xét COGS.

Điểm quan trọng là margin sau refund và shipping giảm từ **{pct(first_margin)} năm {first_year}** xuống **{pct(last_margin)} năm {last_year}**. Nghĩa là doanh nghiệp vẫn có thể tạo topline, nhưng phần lợi nhuận giữ lại trên mỗi đồng net revenue ngày càng mỏng.

Chart 2 vẫn dùng logic quadrant theo plan. Với chỉ 4 category, quadrant không nên bị diễn giải quá mức; nó dùng để chỉ hướng điều tra. **Outdoor** là nhóm tương đối rủi ro hơn vì discount rate và return rate đều cao hơn median. **Streetwear** có quy mô doanh thu lớn, nên dù rate không xấu nhất, tác động P&L vẫn đáng chú ý.

**Kết luận Act 1:** Revenue quality đang xấu đi. Act 2 cần kiểm tra promotion có phải cơ chế chính làm xói mòn margin hay không.
"""),
    ],
    "02_act2_promo_economics.ipynb": [
        md("""
# Act 2 - Ai đang làm rò rỉ margin?

**Câu hỏi trung tâm:** Promotion có thật sự kéo về khách hàng tốt, hay chỉ tạo volume rẻ với unit economics kém?

Act này giữ đúng kế hoạch: dùng Promo ROI Scatter và Cohort Quality để kiểm tra cơ chế gây rò rỉ margin. Return behavior là kiểm định phụ; bằng chứng chính nằm ở repeat rate và gross-profit contribution/customer.
"""),
        SETUP,
        display_cell("chart3_promo_roi_scatter.png"),
        display_cell("chart4_cohort_quality.png"),
        code('''triage = pd.read_csv(TABLES / "promo_triage_table.csv")
cohort = pd.read_csv(TABLES / "cohort_quality_table.csv")
print(f"Tỷ lệ promotion có GP contribution âm: {(triage['total_net_contribution'] < 0).mean() * 100:.1f}%")
display(cohort)'''),
        md(f"""
## Phân tích

Sau khi tính refund và shipping, **{pct(neg_promo_pct)} promotion có GP contribution âm**. Đây là kết quả rất mạnh: toàn bộ promo template lịch sử trong data đều không qua được gross-profit screen.

Cohort analysis cho thấy **item return rate của promo-acquired customers không cao hơn một cách có ý nghĩa thống kê**: promo **{pct(promo_return)}**, organic **{pct(organic_return)}**, Mann-Whitney one-sided **p = 0.3081**. Vì vậy không nên nói promotion làm return tăng.

Thesis vẫn đứng nhờ hai chỉ số khác:

- Customer repeat rate: promo **{pct(promo_repeat)}** vs organic **{pct(organic_repeat)}**.
- Average lifetime GP/customer sau refund và shipping: promo **{money(promo_gp_customer)}** vs organic **{money(organic_gp_customer)}**, thấp hơn **{pct(gp_customer_gap, 0)}**.

**Kết luận Act 2:** Promotion damage chủ yếu đến từ margin compression và cohort economics yếu, không phải từ return behavior.
"""),
    ],
    "03_act3_margin_trajectory.ipynb": [
        md("""
# Act 3 - Scenario simulation nếu cắt promo âm

**Câu hỏi trung tâm:** Nếu cắt các promotion có GP contribution âm, kết quả tài chính thay đổi thế nào dưới các giả định demand recapture khác nhau?

Act này vẫn giữ forecast tuyến tính như risk context, nhưng phần quyết định chính là scenario simulation. Đây là phần predictive thực dụng hơn cho CEO.
"""),
        SETUP,
        display_cell("chart5_margin_scenario_simulation.png"),
        code('''scenario = pd.read_csv(TABLES / "promo_cut_scenario_table.csv")
display(scenario)
forecast = pd.read_csv(TABLES / "margin_forecast_table.csv")
display(forecast[forecast["quarter"].isin(["2026Q4", "2027Q4"])])'''),
        md(f"""
## Phân tích

Baseline hiện tại tạo **{money(current['revenue'])} revenue**, **{money(current['gp_contribution'])} GP contribution sau refund và shipping**, tương ứng gross margin **{pct(current['gross_margin'] * 100)}**.

Trong worst case **0% demand recapture**, cắt **{cut_count} negative-GP promotions** làm revenue giảm còn **{money(cut0['revenue'])}** (**{pct(cut0['revenue_impact_pct'])}**), nhưng GP contribution tăng lên **{money(cut0['gp_contribution'])}** (**+{pct(cut0['gp_impact_pct'])}**) và gross margin tăng lên **{pct(cut0['gross_margin'] * 100)}**.

Nếu **30% lost promo revenue được recapture ở organic margin**, revenue còn **{money(cut30['revenue'])}** (**{pct(cut30['revenue_impact_pct'])}**) và GP contribution tăng **+{pct(cut30['gp_impact_pct'])}**. Với **50% recapture**, revenue còn **{money(cut50['revenue'])}** (**{pct(cut50['revenue_impact_pct'])}**) và GP contribution tăng **+{pct(cut50['gp_impact_pct'])}**.

Linear forecast chỉ là cảnh báo rủi ro, không phải decision rule. Trend gần đây giảm khoảng **{pp(abs(trend_slope))}/quý**, last-four-quarter margin chỉ **{pct(last4_margin)}**. Forecast **2026 Q4 là {pct(f2026['forecast'])}** với range **{pct(f2026['lower_1sd'])} đến {pct(f2026['upper_1sd'])}**; **2027 Q4 là {pct(f2027['forecast'])}** với range **{pct(f2027['lower_1sd'])} đến {pct(f2027['upper_1sd'])}**. Range quá rộng nên không đủ để ra quyết định một mình.

**Kết luận Act 3:** Scenario simulation là bằng chứng predictive chính: cắt promo âm vẫn cải thiện GP ngay cả khi không có demand recapture.
"""),
    ],
    "04_act4_promo_stockout.ipynb": [
        md("""
# Act 4 - Promo chạy vào vùng rủi ro tồn kho

**Câu hỏi trung tâm:** Promotion có đang được kích hoạt trong giai đoạn category có inventory readiness yếu không?

Act này giữ đúng kế hoạch Promo x Stockout Overlap, nhưng chart chính chuyển sang risk ranking để tránh hiểu nhầm timeline là total category failure. Đơn vị phân tích là promo x category x month.
"""),
        SETUP,
        display_cell("chart6_promo_inventory_risk.png"),
        code('''overlaps = pd.read_csv(TABLES / "promo_stockout_overlaps.csv")
risk = pd.read_csv(TABLES / "promo_inventory_risk_table.csv")
print(f"Overlap events: {len(overlaps)}")
print(f"Promotions affected: {overlaps['promo_id'].nunique()}")
print(f"Categories affected: {sorted(overlaps['category'].unique().tolist())}")
print(f"Average product stockout share: {overlaps['stockout_product_share'].mean():.1%}")
print(f"Average fill rate: {overlaps['avg_fill_rate'].mean():.1%}")
print(f"Estimated lost gross profit: {overlaps['estimated_lost_gp'].sum():,.0f} VND")
display(risk.head(20))'''),
        md(f"""
## Phân tích

Sau khi xử lý đúng `applicable_category = null` là promo áp dụng cho **tất cả category**, có **{len(overlaps)} overlap events** ở cấp **promo x category x month**, trải trên **{overlaps['promo_id'].nunique()} promotions** và 4 categories: **{overlap_categories}**.

Average fill rate trong các overlap months là **{pct(overlap_fill)}**, nên đây không phải total inventory failure. Insight đúng là SKU-level risk: trung bình **{pct(overlap_stockout)} SKU trong category có stockout_flag = 1** trong các promo overlap months.

Risk ranking cho thấy promo-category rủi ro cao nhất là **{top_risk['promo_name']} / {top_risk['category']}**, với risk score **{top_risk['inventory_risk_score']:,.0f}**. Estimated lost GP từ các event-level overlaps là **{money(lost_gp)}**.

**Kết luận Act 4:** Cần inventory gate trước khi chạy promo. Gate này nên kiểm tra SKU stockout share và category readiness, không chỉ nhìn fill rate tổng hợp.
"""),
    ],
    "05_act5_promo_triage.ipynb": [
        md("""
# Act 5 - Prescriptive: kê đơn cho từng promotion

**Câu hỏi trung tâm:** Dựa trên ROI, cohort quality và inventory overlap, promotion nào nên KEEP, CUT hoặc RESCHEDULE?

Act này giữ đúng kế hoạch: biến toàn bộ phân tích trước đó thành quyết định cụ thể theo từng campaign.
"""),
        SETUP,
        display_cell("chart7_promo_triage.png"),
        code('''triage = pd.read_csv(TABLES / "promo_triage_table.csv")
summary = triage.groupby("verdict").agg(
    count=("promo_id", "count"),
    total_net_contribution=("total_net_contribution", "sum"),
    estimated_annual_impact_vnd=("estimated_annual_impact_vnd", "sum"),
).reset_index()
display(summary)
print("Least negative CUT promotions")
display(triage[triage.verdict == "CUT"].nlargest(5, "total_net_contribution")[["promo_name", "total_net_contribution", "return_rate", "stockout_overlap"]])
print("Most negative CUT promotions")
display(triage[triage.verdict == "CUT"].nsmallest(5, "total_net_contribution")[["promo_name", "total_net_contribution", "return_rate", "stockout_overlap"]])'''),
        md(f"""
## Phân tích

Triage cho kết quả:

- **KEEP: {keep_count} promotions**.
- **CUT: {cut_count} promotions**, tổng GP contribution **{money(cut_gp)}**.
- **RESCHEDULE: {reschedule_count} promotions**.

Đây là kết quả rất nặng: sau khi tính refund và shipping, không có historical promotion nào còn GP contribution dương. Vì vậy recommendation không phải "giữ một vài promo tốt", mà là **dừng lặp lại toàn bộ promo template lịch sử cho tới khi thiết kế lại cơ chế khuyến mãi**.

Trade-off worst-case: cắt các promotion âm làm revenue giảm **{pct(cut0['revenue_impact_pct'])}**, nhưng GP contribution tăng **+{pct(cut0['gp_impact_pct'])}**. Với 30% organic recapture, revenue impact cải thiện còn **{pct(cut30['revenue_impact_pct'])}** và GP contribution tăng **+{pct(cut30['gp_impact_pct'])}**.

**Kết luận Act 5:** Cut toàn bộ 50 historical promo templates khỏi playbook hiện tại. Promo mới chỉ được launch nếu projected GP contribution sau refund và shipping dương, cohort economics không thấp hơn organic benchmark, và inventory gate đạt yêu cầu.
"""),
    ],
}


REPORT_CELLS = [
    md(f"""
# Báo cáo Phần 2 - Revenue Anatomy: Tiền thật sự đi đâu?
**Datathon 2026 · The Gridbreakers · VinTelligence**

## Thesis trung tâm

> Promotion đang bơm phồng revenue ngắn hạn nhưng làm xấu chất lượng margin và chất lượng customer. Sau khi tính đúng refund và shipping, toàn bộ 50 historical promotion templates đều có GP contribution âm.

## Chuỗi nhân quả

```text
Promo chạy
  -> margin leak qua discount + refund + shipping          [Act 1]
  -> cohort có repeat và GP/customer thấp hơn              [Act 2]
  -> cut decision vẫn dương dưới demand recapture scenarios [Act 3]
  -> promo overlap với inventory readiness yếu             [Act 4]
  -> prescriptive triage: cut toàn bộ historical templates  [Act 5]
```

| Act | Câu hỏi trung tâm | Key number |
|-----|-------------------|------------|
| 1 | Revenue còn lại bao nhiêu sau leakage? | Margin {pct(first_margin)} -> {pct(last_margin)} |
| 2 | Promotion có phải nguồn rò rỉ chính? | {pct(neg_promo_pct)} promos có GP contribution âm |
| 3 | Cut promo có bền dưới scenario khác nhau không? | GP contribution +{pct(cut0['gp_impact_pct'])} đến +{pct(cut50['gp_impact_pct'])} |
| 4 | Có hidden operational damage không? | {len(overlaps)} overlap events, SKU stockout share {pct(overlap_stockout)} |
| 5 | Nên làm gì? | CUT {cut_count}, KEEP {keep_count}, RESCHEDULE {reschedule_count} |
"""),
    SETUP,
]

for name in [
    "01_act1_revenue_anatomy.ipynb",
    "02_act2_promo_economics.ipynb",
    "03_act3_margin_trajectory.ipynb",
    "04_act4_promo_stockout.ipynb",
    "05_act5_promo_triage.ipynb",
]:
    REPORT_CELLS.append(NOTEBOOKS[name][0])
    REPORT_CELLS.extend(NOTEBOOKS[name][2:])

REPORT_CELLS.append(md(f"""
---
## Executive Summary

1. Revenue quality đang xấu đi: **{money(discount)}** leak qua discounts, **{money(refund)}** leak qua refunds, và margin sau refund + shipping giảm từ **{pct(first_margin)}** xuống **{pct(last_margin)}**.

2. Promotion là vấn đề kinh tế chính: **{pct(neg_promo_pct)} campaigns có GP contribution âm**. Return rate không khác biệt có ý nghĩa thống kê, nhưng promo customers có repeat thấp hơn và tạo **{pct(gp_customer_gap, 0)} less lifetime GP/customer**.

3. Scenario simulation cho thấy quyết định cut vẫn bền: worst-case **revenue {pct(cut0['revenue_impact_pct'])} / GP contribution +{pct(cut0['gp_impact_pct'])}**; với 30% organic recapture, revenue impact là **{pct(cut30['revenue_impact_pct'])}** và GP contribution tăng **+{pct(cut30['gp_impact_pct'])}**.

4. Promo x stockout overlap tồn tại ở cấp SKU/category: **{len(overlaps)} event-level overlaps**, **{pct(overlap_stockout)} average SKU stockout share**, **{pct(overlap_fill)} average fill rate**. Rủi ro cao nhất là **{top_risk['promo_name']} / {top_risk['category']}**.

5. Prescription: **KEEP {keep_count}**, **CUT {cut_count}**, **RESCHEDULE {reschedule_count}**. Không nên lặp lại historical promo templates; cần thiết kế lại promo engine.

## Ba khuyến nghị hành động

| Priority | Action | Estimated impact |
|---|---|---|
| 1 | Dừng 50 historical promo templates đang âm GP | GP contribution +{pct(cut0['gp_impact_pct'])} worst case |
| 2 | Thiết kế lại promo gate theo projected post-refund GP | Không launch nếu projected GP contribution <= 0 |
| 3 | Thêm inventory gate trước khi launch promo | Chặn promo khi SKU stockout exposure quá cao |

## Giới hạn bằng chứng

- Đây là gross-profit contribution screen sau refund và shipping, chưa phải full contribution margin vì dataset không có CAC, payment processing cost và fulfillment fixed cost.
- Revenue impact **{pct(cut0['revenue_impact_pct'])}** là mechanical worst case, giả định không có demand recapture.
- Forecast tuyến tính có uncertainty rộng; scenario simulation là decision test chính.
- Review data không xác nhận customer satisfaction giảm sau stockout overlap.
"""))

NOTEBOOKS["06_report.ipynb"] = REPORT_CELLS


for name, cells in NOTEBOOKS.items():
    path = SRC / name
    path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {path}")
