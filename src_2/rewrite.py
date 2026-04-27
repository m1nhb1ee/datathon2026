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

**Câu hỏi trung tâm:** Doanh thu tăng, nhưng bao nhiêu phần doanh thu thật sự còn lại sau chiết khấu, hoàn tiền, phí vận chuyển và giá vốn?

Phân tích bắt đầu từ cấu trúc doanh thu để tách phần bị bào mòn khỏi phần lợi nhuận còn giữ lại.
"""),
        SETUP,
        display_cell("chart1_revenue_anatomy.png"),
        code('''category = pd.read_csv(TABLES / "category_double_loss_table.csv")
category_display = category.rename(columns={
    "category": "nganh_hang",
    "discount_rate": "ty_le_chiet_khau",
    "return_rate": "ty_le_tra_hang",
    "gross_revenue": "doanh_thu_gop",
    "rows": "so_dong",
    "quadrant": "nhom",
})
display(category_display)'''),
        md(f"""
## Phân tích

Trong 10 năm, doanh nghiệp mất **{money(discount)}** qua chiết khấu và **{money(refund)}** qua hoàn tiền. Phí vận chuyển được phân bổ thêm **{money(shipping)}** vào từng dòng đơn hàng. Riêng chiết khấu và hoàn tiền đã tương đương **{pct(leakage)} doanh thu gộp**, trước khi xét giá vốn.

Điểm quan trọng là biên lợi nhuận sau hoàn tiền và phí vận chuyển giảm từ **{pct(first_margin)} năm {first_year}** xuống **{pct(last_margin)} năm {last_year}**. Nghĩa là doanh nghiệp vẫn có thể tạo doanh thu, nhưng phần lợi nhuận giữ lại trên mỗi đồng doanh thu thuần ngày càng mỏng.

Biểu đồ phân nhóm ngành hàng chỉ có 4 điểm dữ liệu, nên nên dùng như bản đồ ưu tiên điều tra thay vì bằng chứng tuyệt đối. **Outdoor** là nhóm tương đối rủi ro hơn vì tỷ lệ chiết khấu và tỷ lệ trả hàng đều cao hơn trung vị. **Streetwear** có quy mô doanh thu lớn, nên dù tỷ lệ không xấu nhất, tác động tới lãi lỗ vẫn đáng chú ý.

**Kết luận Act 1:** Chất lượng doanh thu đang xấu đi. Act 2 kiểm tra liệu khuyến mãi có phải cơ chế chính làm xói mòn biên lợi nhuận hay không.
"""),
    ],
    "02_act2_promo_economics.ipynb": [
        md("""
# Act 2 - Ai đang làm rò rỉ biên lợi nhuận?

**Câu hỏi trung tâm:** Khuyến mãi có thật sự kéo về khách hàng tốt, hay chỉ tạo sản lượng rẻ với hiệu quả kinh tế kém?

Trọng tâm là kiểm tra hiệu quả từng chương trình khuyến mãi và chất lượng nhóm khách hàng đi kèm. Hành vi trả hàng là kiểm định phụ; bằng chứng chính nằm ở tỷ lệ mua lại và lợi nhuận gộp trên mỗi khách hàng.
"""),
        SETUP,
        display_cell("chart3_promo_roi_scatter.png"),
        display_cell("chart4_cohort_quality.png"),
        code('''triage = pd.read_csv(TABLES / "promo_triage_table.csv")
cohort = pd.read_csv(TABLES / "cohort_quality_table.csv")
print(f"Tỷ lệ chương trình có đóng góp lợi nhuận gộp âm: {(triage['total_net_contribution'] < 0).mean() * 100:.1f}%")
cohort_display = cohort.copy()
cohort_display["metric"] = cohort_display["metric"].replace({
    "item_return_rate": "ty_le_tra_hang",
    "customer_repeat_rate": "ty_le_mua_lai",
    "avg_lifetime_gp_per_customer": "loi_nhuan_gop_vong_doi_moi_khach",
})
cohort_display = cohort_display.rename(columns={
    "metric": "chi_so",
    "promo": "nhom_khuyen_mai",
    "organic": "nhom_tu_nhien",
    "difference": "chenh_lech",
})
display(cohort_display)'''),
        md(f"""
## Phân tích

Sau khi tính hoàn tiền và phí vận chuyển, **{pct(neg_promo_pct)} chương trình khuyến mãi có đóng góp lợi nhuận gộp âm**. Đây là kết quả rất mạnh: toàn bộ mẫu khuyến mãi lịch sử trong dữ liệu đều không qua được ngưỡng lợi nhuận gộp.

Phân tích nhóm khách hàng cho thấy **tỷ lệ trả hàng của khách đến từ khuyến mãi không cao hơn một cách có ý nghĩa thống kê**: nhóm khuyến mãi **{pct(promo_return)}**, nhóm tự nhiên **{pct(organic_return)}**, Mann-Whitney một phía **p = 0.3081**. Vì vậy không nên nói khuyến mãi làm tỷ lệ trả hàng tăng.

Luận điểm chính đến từ hai chỉ số khác:

- Tỷ lệ mua lại của khách hàng: nhóm khuyến mãi **{pct(promo_repeat)}** so với nhóm tự nhiên **{pct(organic_repeat)}**.
- Lợi nhuận gộp vòng đời trên mỗi khách hàng sau hoàn tiền và phí vận chuyển: nhóm khuyến mãi **{money(promo_gp_customer)}** so với nhóm tự nhiên **{money(organic_gp_customer)}**, thấp hơn **{pct(gp_customer_gap, 0)}**.

**Kết luận Act 2:** Tác hại của khuyến mãi chủ yếu đến từ biên lợi nhuận bị nén và hiệu quả kinh tế khách hàng yếu, không phải từ hành vi trả hàng.
"""),
    ],
    "03_act3_margin_trajectory.ipynb": [
        md("""
# Act 3 - Mô phỏng kịch bản nếu cắt khuyến mãi âm

**Câu hỏi trung tâm:** Nếu cắt các chương trình có đóng góp lợi nhuận gộp âm, kết quả tài chính thay đổi thế nào dưới các giả định nhu cầu quay lại khác nhau?

Dự báo tuyến tính chỉ đóng vai trò cảnh báo rủi ro; phần quyết định chính là mô phỏng kịch bản tài chính khi cắt các chương trình âm lợi nhuận.
"""),
        SETUP,
        display_cell("chart5_margin_scenario_simulation.png"),
        code('''scenario = pd.read_csv(TABLES / "promo_cut_scenario_table.csv")
scenario_display = scenario.copy()
scenario_display["assumption"] = scenario_display["assumption"].replace({
    "All promotions continue": "Giữ nguyên toàn bộ khuyến mãi",
    "0% of removed promo revenue returns at organic margin": "0% doanh thu khuyến mãi bị cắt quay lại ở biên tự nhiên",
    "30% of removed promo revenue returns at organic margin": "30% doanh thu khuyến mãi bị cắt quay lại ở biên tự nhiên",
    "50% of removed promo revenue returns at organic margin": "50% doanh thu khuyến mãi bị cắt quay lại ở biên tự nhiên",
})
scenario_display = scenario_display.rename(columns={
    "scenario": "kich_ban",
    "recapture_rate": "ty_le_nhu_cau_quay_lai",
    "revenue": "doanh_thu",
    "gp_contribution": "dong_gop_loi_nhuan_gop",
    "gross_margin": "bien_loi_nhuan_gop",
    "revenue_impact_pct": "tac_dong_doanh_thu_pct",
    "gp_impact_pct": "tac_dong_loi_nhuan_gop_pct",
    "assumption": "gia_dinh",
})
display(scenario_display)
forecast = pd.read_csv(TABLES / "margin_forecast_table.csv")
forecast_display = forecast[forecast["quarter"].isin(["2026Q4", "2027Q4"])].rename(columns={
    "quarter": "quy",
    "date": "ngay",
    "forecast": "du_bao",
    "lower_1sd": "can_duoi_1sd",
    "upper_1sd": "can_tren_1sd",
})
display(forecast_display)'''),
        md(f"""
## Phân tích

Hiện trạng tạo **{money(current['revenue'])} doanh thu**, **{money(current['gp_contribution'])} đóng góp lợi nhuận gộp sau hoàn tiền và phí vận chuyển**, tương ứng biên lợi nhuận gộp **{pct(current['gross_margin'] * 100)}**.

Trong kịch bản thận trọng nhất **0% nhu cầu quay lại**, cắt **{cut_count} chương trình âm lợi nhuận gộp** làm doanh thu giảm còn **{money(cut0['revenue'])}** (**{pct(cut0['revenue_impact_pct'])}**), nhưng đóng góp lợi nhuận gộp tăng lên **{money(cut0['gp_contribution'])}** (**+{pct(cut0['gp_impact_pct'])}**) và biên lợi nhuận gộp tăng lên **{pct(cut0['gross_margin'] * 100)}**.

Nếu **30% doanh thu khuyến mãi bị mất quay lại ở biên lợi nhuận tự nhiên**, doanh thu còn **{money(cut30['revenue'])}** (**{pct(cut30['revenue_impact_pct'])}**) và đóng góp lợi nhuận gộp tăng **+{pct(cut30['gp_impact_pct'])}**. Với **50% nhu cầu quay lại**, doanh thu còn **{money(cut50['revenue'])}** (**{pct(cut50['revenue_impact_pct'])}**) và đóng góp lợi nhuận gộp tăng **+{pct(cut50['gp_impact_pct'])}**.

Dự báo tuyến tính chỉ là cảnh báo rủi ro, không phải quy tắc ra quyết định. Xu hướng gần đây giảm khoảng **{pp(abs(trend_slope))}/quý**, biên lợi nhuận bốn quý gần nhất chỉ **{pct(last4_margin)}**. Dự báo **Q4/2026 là {pct(f2026['forecast'])}** với khoảng dao động **{pct(f2026['lower_1sd'])} đến {pct(f2026['upper_1sd'])}**; **Q4/2027 là {pct(f2027['forecast'])}** với khoảng dao động **{pct(f2027['lower_1sd'])} đến {pct(f2027['upper_1sd'])}**. Khoảng dao động quá rộng nên không đủ để ra quyết định một mình.

**Kết luận Act 3:** Mô phỏng kịch bản là bằng chứng dự báo chính: cắt khuyến mãi âm vẫn cải thiện lợi nhuận gộp ngay cả khi không có nhu cầu quay lại.
"""),
    ],
    "04_act4_promo_stockout.ipynb": [
        md("""
# Act 4 - Khuyến mãi chạy vào vùng rủi ro tồn kho

**Câu hỏi trung tâm:** Khuyến mãi có đang được kích hoạt trong giai đoạn ngành hàng có mức sẵn sàng tồn kho yếu không?

Phân tích tập trung vào giao điểm giữa lịch khuyến mãi và tồn kho yếu ở cấp chương trình x ngành hàng x tháng.
"""),
        SETUP,
        display_cell("chart6_promo_inventory_risk.png"),
        code('''overlaps = pd.read_csv(TABLES / "promo_stockout_overlaps.csv")
risk = pd.read_csv(TABLES / "promo_inventory_risk_table.csv")
print(f"Số event chồng lấn: {len(overlaps)}")
print(f"Số chương trình bị ảnh hưởng: {overlaps['promo_id'].nunique()}")
print(f"Ngành hàng bị ảnh hưởng: {sorted(overlaps['category'].unique().tolist())}")
print(f"Tỷ lệ SKU thiếu hàng trung bình: {overlaps['stockout_product_share'].mean():.1%}")
print(f"Tỷ lệ đáp ứng đơn trung bình: {overlaps['avg_fill_rate'].mean():.1%}")
print(f"Lợi nhuận gộp ước tính bị mất: {overlaps['estimated_lost_gp'].sum():,.0f} VND")
risk_display = risk.head(20).rename(columns={
    "promo_id": "ma_khuyen_mai",
    "promo_name": "ten_khuyen_mai",
    "category": "nganh_hang",
    "overlap_months": "so_thang_chong_lan",
    "avg_sku_stockout_share": "ty_le_sku_thieu_hang_tb",
    "avg_fill_rate": "ty_le_dap_ung_don_tb",
    "estimated_lost_gp": "loi_nhuan_gop_uoc_tinh_mat",
    "promo_gp_contribution": "dong_gop_loi_nhuan_gop",
    "promo_net_revenue": "doanh_thu_thuan_khuyen_mai",
    "inventory_risk_score": "diem_rui_ro_ton_kho",
})
display(risk_display)'''),
        md(f"""
## Phân tích

Sau khi xử lý đúng `applicable_category = null` là khuyến mãi áp dụng cho **tất cả ngành hàng**, có **{len(overlaps)} điểm chồng lấn** ở cấp **chương trình x ngành hàng x tháng**, trải trên **{overlaps['promo_id'].nunique()} chương trình** và 4 ngành hàng: **{overlap_categories}**.

Tỷ lệ đáp ứng đơn trung bình trong các tháng chồng lấn là **{pct(overlap_fill)}**, nên đây không phải thất bại tồn kho toàn ngành hàng. Điểm rủi ro nằm ở cấp SKU: trung bình **{pct(overlap_stockout)} SKU trong ngành hàng có `stockout_flag = 1`** trong các tháng khuyến mãi chồng lấn.

Bảng xếp hạng rủi ro cho thấy cặp chương trình-ngành hàng rủi ro cao nhất là **{top_risk['promo_name']} / {top_risk['category']}**, với điểm rủi ro **{top_risk['inventory_risk_score']:,.0f}**. Lợi nhuận gộp ước tính bị mất từ các điểm chồng lấn là **{money(lost_gp)}**.

**Kết luận Act 4:** Cần cổng kiểm tra tồn kho trước khi chạy khuyến mãi. Cổng này nên kiểm tra tỷ lệ SKU thiếu hàng và mức sẵn sàng của ngành hàng, không chỉ nhìn tỷ lệ đáp ứng đơn tổng hợp.
"""),
    ],
    "05_act5_promo_triage.ipynb": [
        md("""
# Act 5 - Kê đơn hành động cho từng chương trình

**Câu hỏi trung tâm:** Dựa trên hiệu quả tài chính, chất lượng khách hàng và rủi ro tồn kho, chương trình nào nên KEEP, CUT hoặc RESCHEDULE?

Phần cuối chuyển toàn bộ kết quả phân tích thành quyết định cụ thể cho từng chương trình.
"""),
        SETUP,
        display_cell("chart7_promo_triage.png"),
        code('''triage = pd.read_csv(TABLES / "promo_triage_table.csv")
summary = triage.groupby("verdict").agg(
    count=("promo_id", "count"),
    total_net_contribution=("total_net_contribution", "sum"),
    estimated_annual_impact_vnd=("estimated_annual_impact_vnd", "sum"),
).reset_index()
summary_display = summary.rename(columns={
    "verdict": "quyet_dinh",
    "count": "so_chuong_trinh",
    "total_net_contribution": "tong_dong_gop_loi_nhuan_gop",
    "estimated_annual_impact_vnd": "tac_dong_uoc_tinh_nam_vnd",
})
display(summary_display)
print("Các chương trình CUT ít âm nhất")
cols = ["promo_name", "total_net_contribution", "return_rate", "stockout_overlap"]
display(triage[triage.verdict == "CUT"].nlargest(5, "total_net_contribution")[cols].rename(columns={
    "promo_name": "ten_khuyen_mai",
    "total_net_contribution": "dong_gop_loi_nhuan_gop",
    "return_rate": "ty_le_tra_hang",
    "stockout_overlap": "chong_lan_thieu_hang",
}))
print("Các chương trình CUT âm nặng nhất")
display(triage[triage.verdict == "CUT"].nsmallest(5, "total_net_contribution")[cols].rename(columns={
    "promo_name": "ten_khuyen_mai",
    "total_net_contribution": "dong_gop_loi_nhuan_gop",
    "return_rate": "ty_le_tra_hang",
    "stockout_overlap": "chong_lan_thieu_hang",
}))'''),
        md(f"""
## Phân tích

Triage cho kết quả:

- **KEEP: {keep_count} chương trình**.
- **CUT: {cut_count} chương trình**, tổng đóng góp lợi nhuận gộp **{money(cut_gp)}**.
- **RESCHEDULE: {reschedule_count} chương trình**.

Đây là kết quả rất nặng: sau khi tính hoàn tiền và phí vận chuyển, không có chương trình khuyến mãi lịch sử nào còn đóng góp lợi nhuận gộp dương. Vì vậy khuyến nghị không phải "giữ một vài khuyến mãi tốt", mà là **dừng lặp lại toàn bộ mẫu khuyến mãi lịch sử cho tới khi thiết kế lại cơ chế khuyến mãi**.

Đánh đổi trong kịch bản thận trọng nhất: cắt các chương trình âm làm doanh thu giảm **{pct(cut0['revenue_impact_pct'])}**, nhưng đóng góp lợi nhuận gộp tăng **+{pct(cut0['gp_impact_pct'])}**. Với 30% nhu cầu quay lại tự nhiên, tác động doanh thu cải thiện còn **{pct(cut30['revenue_impact_pct'])}** và đóng góp lợi nhuận gộp tăng **+{pct(cut30['gp_impact_pct'])}**.

**Kết luận Act 5:** Loại toàn bộ 50 mẫu khuyến mãi lịch sử khỏi sổ tay vận hành hiện tại. Khuyến mãi mới chỉ nên triển khai nếu đóng góp lợi nhuận gộp dự kiến sau hoàn tiền và phí vận chuyển dương, hiệu quả kinh tế khách hàng không thấp hơn nhóm tự nhiên, và cổng kiểm tra tồn kho đạt yêu cầu.
"""),
    ],
}


REPORT_CELLS = [
    md(f"""
# Báo cáo Phần 2 - Cấu trúc doanh thu: Tiền thật sự đi đâu?
**Datathon 2026 · The Gridbreakers · VinTelligence**

## Luận điểm trung tâm

> Khuyến mãi đang bơm phồng doanh thu ngắn hạn nhưng làm xấu chất lượng biên lợi nhuận và chất lượng khách hàng. Sau khi tính đúng hoàn tiền và phí vận chuyển, toàn bộ 50 mẫu khuyến mãi lịch sử đều có đóng góp lợi nhuận gộp âm.

## Chuỗi nhân quả

```text
Khuyến mãi chạy
  -> rò rỉ biên lợi nhuận qua chiết khấu, hoàn tiền, phí vận chuyển [Act 1]
  -> nhóm khách hàng có mua lại và lợi nhuận/khách thấp hơn          [Act 2]
  -> quyết định cắt vẫn dương dưới các kịch bản nhu cầu quay lại     [Act 3]
  -> khuyến mãi chồng lấn với tồn kho sẵn sàng yếu                   [Act 4]
  -> kê đơn hành động: cắt toàn bộ mẫu khuyến mãi lịch sử            [Act 5]
```

| Act | Câu hỏi trung tâm | Số liệu chính |
|-----|-------------------|------------|
| 1 | Doanh thu còn lại bao nhiêu sau rò rỉ? | Biên lợi nhuận {pct(first_margin)} -> {pct(last_margin)} |
| 2 | Khuyến mãi có phải nguồn rò rỉ chính? | {pct(neg_promo_pct)} chương trình có đóng góp lợi nhuận gộp âm |
| 3 | Cắt khuyến mãi có bền dưới các kịch bản khác nhau không? | Đóng góp lợi nhuận gộp +{pct(cut0['gp_impact_pct'])} đến +{pct(cut50['gp_impact_pct'])} |
| 4 | Có thiệt hại vận hành ẩn không? | {len(overlaps)} điểm chồng lấn, tỷ lệ SKU thiếu hàng {pct(overlap_stockout)} |
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
## Tóm tắt điều hành

1. Chất lượng doanh thu đang xấu đi: **{money(discount)}** rò rỉ qua chiết khấu, **{money(refund)}** rò rỉ qua hoàn tiền, và biên lợi nhuận sau hoàn tiền + phí vận chuyển giảm từ **{pct(first_margin)}** xuống **{pct(last_margin)}**.

2. Khuyến mãi là vấn đề kinh tế chính: **{pct(neg_promo_pct)} chương trình có đóng góp lợi nhuận gộp âm**. Tỷ lệ trả hàng không khác biệt có ý nghĩa thống kê, nhưng khách hàng từ khuyến mãi có tỷ lệ mua lại thấp hơn và tạo **ít hơn {pct(gp_customer_gap, 0)} lợi nhuận gộp vòng đời trên mỗi khách hàng**.

3. Mô phỏng kịch bản cho thấy quyết định cắt vẫn bền: kịch bản thận trọng nhất **doanh thu {pct(cut0['revenue_impact_pct'])} / đóng góp lợi nhuận gộp +{pct(cut0['gp_impact_pct'])}**; với 30% nhu cầu quay lại tự nhiên, tác động doanh thu là **{pct(cut30['revenue_impact_pct'])}** và đóng góp lợi nhuận gộp tăng **+{pct(cut30['gp_impact_pct'])}**.

4. Chồng lấn giữa khuyến mãi và thiếu hàng tồn tại ở cấp SKU/ngành hàng: **{len(overlaps)} điểm chồng lấn**, **{pct(overlap_stockout)} tỷ lệ SKU thiếu hàng trung bình**, **{pct(overlap_fill)} tỷ lệ đáp ứng đơn trung bình**. Rủi ro cao nhất là **{top_risk['promo_name']} / {top_risk['category']}**.

5. Kê đơn hành động: **KEEP {keep_count}**, **CUT {cut_count}**, **RESCHEDULE {reschedule_count}**. Không nên lặp lại các mẫu khuyến mãi lịch sử; cần thiết kế lại cơ chế khuyến mãi.

## Ba khuyến nghị hành động

| Ưu tiên | Hành động | Tác động ước tính |
|---|---|---|
| 1 | Dừng 50 mẫu khuyến mãi lịch sử đang âm lợi nhuận gộp | Đóng góp lợi nhuận gộp +{pct(cut0['gp_impact_pct'])} trong kịch bản thận trọng nhất |
| 2 | Thiết kế lại cổng duyệt khuyến mãi theo lợi nhuận gộp dự kiến sau hoàn tiền | Không triển khai nếu đóng góp lợi nhuận gộp dự kiến <= 0 |
| 3 | Thêm cổng kiểm tra tồn kho trước khi triển khai khuyến mãi | Chặn khuyến mãi khi mức thiếu hàng SKU quá cao |

## Giới hạn bằng chứng

- Đây là phép lọc theo đóng góp lợi nhuận gộp sau hoàn tiền và phí vận chuyển, chưa phải biên đóng góp đầy đủ vì dữ liệu không có chi phí thu hút khách hàng, chi phí xử lý thanh toán và chi phí vận hành cố định.
- Phân tích khuyến mãi dùng `discount_amount` thực tế trong `order_items` vì cột này khớp với `payments.payment_value`; các cột `promo_channel` và `min_order_value` trong bảng master được dùng như metadata, không dùng để lọc lại các đơn hàng lịch sử đã có chiết khấu.
- Tác động doanh thu **{pct(cut0['revenue_impact_pct'])}** là kịch bản cơ học thận trọng nhất, giả định không có nhu cầu quay lại.
- Dự báo tuyến tính có độ bất định rộng; mô phỏng kịch bản là kiểm định quyết định chính.
- Dữ liệu đánh giá không xác nhận mức hài lòng khách hàng giảm sau các điểm chồng lấn với thiếu hàng.
"""))

NOTEBOOKS["06_report.ipynb"] = REPORT_CELLS


for name, cells in NOTEBOOKS.items():
    path = SRC / name
    path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {path}")
