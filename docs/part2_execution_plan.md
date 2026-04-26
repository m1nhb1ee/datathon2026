# Part 2 — Execution Plan
**Datathon 2026 · The Gridbreakers · VinTelligence**

---

## Thesis

> **Promo đang bơm phồng doanh thu ngắn hạn trong khi phá hủy chất lượng khách hàng và margin dài hạn — và bằng chứng nằm trong return behavior, cohort retention, và phân bổ net contribution theo từng chiến dịch.**

Mọi chart trong bài đều trả lời một trong ba câu hỏi:
1. Margin thực sự là bao nhiêu?
2. Promo phá hủy giá trị như thế nào?
3. Cụ thể nên làm gì với promo nào?

---

## Bước 0 — OAT (One Analytics Table)

Build một lần, dùng xuyên suốt. Không join lại trong từng notebook.

### Join logic

```
orders
  ├── order_items  →  products        (category, segment, size, price, cogs)
  ├── order_items  →  promotions      (promo_type, discount_value, promo_name)
  ├── returns                         (return_reason, return_quantity, refund_amount)
  ├── reviews                         (rating)
  ├── customers                       (age_group, gender, acquisition_channel, signup_date)
  ├── geography                       (city, region, district)
  └── payments                        (payment_value, installments)
```

### Derived columns — tính sẵn trong OAT

| Column | Formula |
|---|---|
| `net_revenue` | `unit_price × quantity − discount_amount` |
| `gross_profit` | `net_revenue − cogs × quantity` |
| `has_promo` | `promo_id IS NOT NULL` |
| `is_returned` | `order_id IN returns` |
| `discount_rate` | `discount_amount / (unit_price × quantity)` |
| `customer_order_seq` | `RANK() OVER (PARTITION BY customer_id ORDER BY order_date)` |
| `is_repeat` | `customer_order_seq > 1` |

### Output
- `oat.parquet` — lưu file, không recompute
- Validate: row counts, NULL audit trên join keys

---

## Act 1 — Descriptive Foundation
**Thời gian: 2 giờ**

**Câu hỏi trung tâm:** Revenue tăng như thế nào trong 10 năm, nhưng margin thực sự đang đi đâu?

### Chart 1 — Revenue Waterfall

**Câu hỏi nó trả lời:** Gross revenue tăng trưởng như thế nào, và bao nhiêu bị ăn mòn bởi discount và COGS?

- **Chart type:** Stacked area chart theo tháng (hoặc waterfall bar theo năm)
- **Layers:** Gross Revenue → (−) Discount Leakage → Net Revenue → (−) COGS → Gross Profit
- **Nguồn dữ liệu:** `sales.csv` + `order_items` + `products`
- **Annotation bắt buộc:**
  - Tổng discount leakage tích lũy 10 năm = **X tỷ đồng** (chữ to, màu đỏ)
  - Gross margin đầu kỳ vs cuối kỳ: **Y% → Z%**

> **Headline số — đặt đầu report, không được thiếu:**
> "Discount leakage tích lũy 10 năm = X tỷ đồng. Gross margin giảm từ Y% xuống Z% dù revenue tăng."

**Không cần thêm chart nào trong Act 1.** Seasonality heatmap là optional, chỉ thêm sau khi Act 2 và Act 3 hoàn toàn xong.

---

## Act 2 — Promo Creates Bad Economics
**Thời gian: 4 giờ — không được rush — đây là 35–40/60 điểm**

**Câu hỏi trung tâm:** Promo có thật sự tạo ra giá trị, hay chỉ kéo khách chất lượng thấp và đẩy return rate lên?

### Chart 2 — Promo ROI Scatter

**Câu hỏi nó trả lời:** Promo nào có ROI thực dương và promo nào đang phá hủy margin?

- **Chart type:** Bubble scatter
- **Trục:** X = tổng discount cost của promo, Y = tổng net contribution của orders dùng promo, Size = số orders, Color = return rate
- **Nguồn dữ liệu:** `order_items` + `promotions` + `products` (để tính cogs)
- **Annotation bắt buộc:** Highlight top 3 promo tốt nhất và top 3 tệ nhất
- **Kết luận bắt buộc:** "X% promotions có net contribution âm."

---

### Chart 3 — Cohort Quality Comparison

**Câu hỏi nó trả lời:** Khách hàng đến qua promo có hành vi khác gì so với khách organic?

- **Chart type:** Grouped bar (3 metrics, 2 groups)
- **Groups:** Promo cohort vs Organic cohort
- **Metrics:**
  1. Return rate
  2. Repeat rate (`customer_order_seq > 1`)
  3. Avg gross profit per customer
- **Nguồn dữ liệu:** `orders` + `order_items` + `returns` + `customers`
- **Statistical test:** Mann-Whitney U cho return rate — hiển thị dưới dạng:
  > "Promo customers return **X lần** nhiều hơn (p < 0.01)" — không cần giải thích test
- **Kết luận bắt buộc:** "Promo customers return X lần nhiều hơn, repeat rate thấp hơn Y%, gross profit per customer thấp hơn Z%. Đây không phải tăng trưởng — đây là debt."

---

### Chart 4 — Geographic Promo Efficiency *(Conditional)*

**Câu hỏi nó trả lời:** Promo có hiệu quả khác nhau theo vùng địa lý không — và nếu có, promo đang deploy sai target ở đâu?

- **Chart type:** Bar chart, phân theo region
- **Metric:** Return rate của promo orders vs organic orders theo từng region
- **Nguồn dữ liệu:** `orders` + `returns` + `geography`
- **Nguồn dữ liệu:** Giữ nếu sự khác biệt > 3 percentage points giữa các vùng. **Cắt ngay nếu signal không rõ.** Không ép narrative.

---

> **Headline bắt buộc cuối Act 2:**
> "Promo customers return X lần nhiều hơn, repeat rate thấp hơn Y%, và gross profit per customer thấp hơn Z%. Đây không phải tăng trưởng — đây là debt."

---

## Act 3 — Predictive + Prescriptive
**Thời gian: 2 giờ**

**Câu hỏi trung tâm:** Nếu không thay đổi gì, điều gì sẽ xảy ra? Và cụ thể nên làm gì với từng promo?

### Chart 5 — Margin Trajectory

**Câu hỏi nó trả lời:** Nếu discount rate tiếp tục tăng, margin sẽ chạm mức nào và khi nào?

- **Chart type:** Line chart với trend line extrapolated
- **Dữ liệu:** Gross margin % theo quý 2012–2022, extrapolate đến 2026
- **Nguồn dữ liệu:** `sales.csv` + `order_items` + `products`
- **Không cần ML** — linear fit đủ để visual impact
- **Annotation bắt buộc:** "Với tốc độ hiện tại, gross margin sẽ chạm X% vào năm Y."

---

### Chart 6 — Promo Triage

**Câu hỏi nó trả lời:** Nên giữ, cắt, hay điều chỉnh từng promo nào — và impact tài chính là bao nhiêu?

- **Chart type:** Horizontal ranked bar chart
- **Metric:** Net contribution per promo — xanh top 5 (keep), đỏ bottom 5 (cut)
- **Nguồn dữ liệu:** `order_items` + `promotions` + `products` + `returns`
- **Kèm prescriptive table:**

| promo_id | verdict | estimated annual impact (VND) |
|---|---|---|
| PROMO_X | ✅ KEEP | +X tỷ |
| PROMO_Y | ❌ CUT | −Y tỷ (saving) |
| PROMO_Z | ⚠️ RESCHEDULE | +Z tỷ nếu dịch sang Q4 |

- **Estimated impact** = avg net_contribution × số orders affected — số thực từ data, không dùng proxy
- **Trade-off bắt buộc:** "Cắt bottom 5 promo → revenue giảm X% nhưng gross profit tăng Y%."

---

## Danh sách CẮT — không thêm dù còn thời gian

| Phân tích | Lý do cắt |
|---|---|
| Traffic funnel (web_traffic vs order_source) | Mapping `traffic_source → order_source` không được dataset guarantee. Không defend được nếu bị hỏi. |
| Inventory analysis | Không nối trực tiếp vào thesis promo. |
| Acquisition channel standalone | Chỉ giữ nếu có evidence promo kéo acquisition từ kênh chất lượng thấp. Nếu không: cắt. |
| Review sentiment chi tiết | ROI thấp. Rating aggregate đã đủ trong Chart 3. |
| Seasonality heatmap | Optional. Chỉ thêm sau khi Act 2 + Act 3 xong hoàn toàn. |

---

## Phân bổ thời gian

| Bước | Nội dung | Giờ |
|---|---|---|
| 0 | Build OAT, validate joins, NULL audit | 2h |
| Act 1 | Chart 1 + headline number | 2h |
| Act 2 | Chart 2 + 3 (+ Chart 4 conditional) | 4h |
| Act 3 | Chart 5 + Chart 6 + prescriptive table | 2h |
| Polish | Annotation, headline numbers, narrative linkage | 2h |
| Buffer | Fix issues, Chart 4 nếu signal rõ | 1h |
| **Tổng** | | **13h** |

**Rule tuyệt đối:** Nếu Act 2 chưa xong, không bắt đầu Act 3. Nếu Act 3 chưa xong, không polish. Không thêm chart nào ngoài danh sách.

---

## Checklist trước khi nộp

- [ ] Mỗi chart có: title, axis labels, data source annotation, headline number
- [ ] Mỗi chart theo sau bởi đúng hai câu: (1) Findings — con số cụ thể. (2) Implication — business action
- [ ] Prescriptive table có cột "Estimated Annual Impact (VND)" với số tính từ data thực
- [ ] Trade-off được phát biểu rõ: "Nếu cắt → revenue −X%, profit +Y%"
- [ ] Narrative linkage: mỗi act kết thúc bằng câu dẫn sang act tiếp theo
- [ ] OAT được save thành file parquet — không join lại ở notebook sau
- [ ] Random seed set nếu có bất kỳ sampling nào
- [ ] GitHub push trước deadline — verify link trong report
