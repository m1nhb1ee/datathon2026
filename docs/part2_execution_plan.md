# Part 2 — Final Execution Plan
**Datathon 2026 · The Gridbreakers · VinTelligence**

---

## Thesis

> **Promo đang bơm phồng doanh thu ngắn hạn trong khi phá hủy customer quality, margin dài hạn, và customer experience — bằng chứng nằm trong return behavior, cohort retention, net contribution, và promo×stockout overlap.**

**Chuỗi nhân quả:**
```
Promo chạy
  → margin leak (discount + return leakage)
  → kéo khách chất lượng thấp (return rate cao, repeat rate thấp)
  → projected margin tiếp tục giảm nếu không thay đổi
  → promo chạy đúng lúc hết hàng → double damage (lost revenue + review rating giảm)
  → prescriptive triage: promo nào keep / cut / reschedule
```

Mọi chart phục vụ chuỗi nhân quả này. Không có chart nào đứng ngoài.

---

## Bước 0 — OAT (One Analytics Table)

Build một lần, save thành `oat.parquet`. Mọi bước sau load từ file này, không join lại.

### Join logic

```
orders
  ├── LEFT JOIN order_items       ON order_id
  │     ├── LEFT JOIN products    ON product_id   → category, segment, size, price, cogs
  │     └── LEFT JOIN promotions  ON promo_id     → promo_type, discount_value, promo_name
  ├── LEFT JOIN customers         ON customer_id  → age_group, gender, acquisition_channel
  ├── LEFT JOIN geography         ON zip          → city, region, district
  ├── LEFT JOIN payments          ON order_id     → payment_value, installments
  ├── LEFT JOIN returns           ON order_id     → is_returned, refund_amount, return_reason
  └── LEFT JOIN reviews           ON order_id     → rating
```

### Derived columns — tính sẵn trong OAT

| Column | Formula |
|---|---|
| `gross_revenue_line` | `unit_price × quantity` |
| `net_revenue` | `unit_price × quantity − discount_amount` |
| `gross_profit` | `net_revenue − cogs × quantity` |
| `has_promo` | `promo_id IS NOT NULL` |
| `is_returned` | `order_id IN returns` |
| `discount_rate` | `discount_amount / (unit_price × quantity)` |
| `return_leakage` | `refund_amount` (0 nếu không trả hàng) |
| `customer_order_seq` | `RANK() OVER (PARTITION BY customer_id ORDER BY order_date)` |
| `is_repeat` | `customer_order_seq > 1` |

### Validation bắt buộc trước khi tiếp tục

- Row counts từng bảng nguồn vs OAT
- NULL audit: `order_id`, `customer_id`, `product_id`, `net_revenue`, `gross_profit`
- Assert: `cogs < price` cho mọi sản phẩm

---

## Act 1 — Bề mặt vs. Thực tế
**Câu hỏi trung tâm:** Revenue nhìn bề mặt có vẻ ổn — nhưng bao nhiêu trong số đó thực sự còn lại?

---

### Chart 1 — Revenue Anatomy (Stacked Area)

**Câu hỏi nó trả lời:** Gross revenue tăng trưởng như thế nào trong 10 năm, và bao nhiêu bị ăn mòn bởi discount, return, và COGS?

- **Chart type:** Stacked area chart theo tháng
- **Layers theo thứ tự từ trên xuống:**
  1. Gross Revenue (`unit_price × quantity`)
  2. (−) Discount Leakage (`discount_amount`)
  3. (−) Return Leakage (`refund_amount`)
  4. Net Revenue (phần còn lại sau discount + return)
  5. (−) COGS (`cogs × quantity`)
  6. Gross Profit (phần còn lại cuối cùng)
- **Nguồn dữ liệu:** `order_items` + `products` + `returns` (join qua OAT)
- **Annotation bắt buộc:**
  - Tổng discount leakage tích lũy 10 năm
  - Tổng return leakage tích lũy 10 năm
  - Gross margin % năm 2012 vs năm 2022

> **Tại sao chart này không hiển nhiên:** Hầu hết đội sẽ plot `Revenue` từ `sales.csv` rồi dừng. Nhưng `sales.csv` chỉ có tổng hợp — muốn tách discount leakage và return leakage riêng, phải join ngược từ `order_items` + `returns`. Đây là cross-table join tạo ra insight mà nhìn một bảng không thấy.

---

### Chart 2 — Double Loss Category (Quadrant)

**Câu hỏi nó trả lời:** Danh mục sản phẩm nào vừa bị discount cao vừa bị return cao — tức là đang bị rò rỉ từ cả hai đầu?

- **Chart type:** Quadrant scatter chart
- **Trục:** X = avg discount_rate per category, Y = avg return_rate per category
- **Size của bubble:** tổng gross_revenue_line của category đó
- **Quadrant label:**
  - Góc trên phải: "Double loss" → ưu tiên can thiệp nhất
  - Góc dưới trái: "Healthy" → benchmark
- **Nguồn dữ liệu:** OAT, group by `category`
- **Annotation bắt buộc:** Gắn nhãn tên category cho từng bubble

---

## Act 2 — Ai Đang Làm Rò Rỉ?
**Câu hỏi trung tâm:** Promo có thật sự tạo ra khách hàng tốt, hay chỉ kéo về khách chất lượng thấp rồi để họ trả hàng?

---

### Chart 3 — Promo ROI Scatter

**Câu hỏi nó trả lời:** Promo nào có ROI thực dương, và promo nào đang tốn tiền discount nhiều hơn net contribution mà nó tạo ra?

- **Chart type:** Bubble scatter
- **Trục:** X = tổng discount cost, Y = tổng net contribution
- **Size:** số orders
- **Color:** return rate (continuous colormap, xanh→đỏ)
- **Đường tham chiếu:** horizontal line tại Y = 0
- **Nguồn dữ liệu:** OAT, group by `promo_id`
- **Annotation bắt buộc:**
  - Top 3 promo net contribution cao nhất → label "KEEP"
  - Top 3 promo net contribution thấp nhất (âm) → label "CUT"
  - % promotions có net contribution âm

---

### Chart 4 — Cohort Quality Comparison

**Câu hỏi nó trả lời:** Khách đến qua promo có hành vi khác gì so với khách organic — và sự khác biệt đó đủ lớn để kết luận promo đang kéo sai khách không?

- **Chart type:** Grouped bar (3 metric groups, 2 bars mỗi group)
- **Groups:** Promo cohort vs Organic cohort
- **Metrics:**
  1. Return rate
  2. Repeat rate (`is_repeat = True`)
  3. Avg gross profit per customer
- **Nguồn dữ liệu:** OAT, cohort định nghĩa theo first order của customer
- **Statistical test:**
  - Mann-Whitney U cho return rate: promo vs organic
  - Hiển thị trên chart: `"Promo customers return X lần nhiều hơn (p < 0.01)"`
  - Không cần giải thích test — chỉ cần effect size và p-value

> **Headline bắt buộc cuối Act 2:** "Promo customers return X lần nhiều hơn, repeat rate thấp hơn Y%, gross profit per customer thấp hơn Z%. Đây không phải tăng trưởng — đây là debt."

---

## Act 3 — Projected Loss Nếu Không Thay Đổi
**Câu hỏi trung tâm:** Nếu xu hướng hiện tại tiếp tục, margin sẽ đi đến đâu?

---

### Chart 5 — Margin Trajectory

**Câu hỏi nó trả lời:** Nếu discount rate tiếp tục tăng theo tốc độ hiện tại, gross margin sẽ chạm mức nguy hiểm khi nào?

- **Chart type:** Line chart + trend extrapolation
- **Dữ liệu:** Gross margin % theo quý, 2012–2022 (actual) + extrapolate đến 2027 (dashed)
- **Nguồn dữ liệu:** OAT, group by quarter
- **Method:** Linear trend fit trên gross_margin_pct — không cần ML
- **Annotation bắt buộc:**
  - Vertical line tại 2023-01-01: "Forecast start"
  - Projected gross margin tại 2026 Q4 và 2027 Q4
  - Shaded confidence band quanh phần extrapolated

---

## Act 4 — Insight Độc Đáo: Promo Chạy Đúng Lúc Hết Hàng
**Câu hỏi trung tâm:** Ngoài việc kéo khách tệ, promo còn gây hại thêm bằng cách nào?

---

### Chart 6 — Promo × Stockout Overlap (Timeline)

**Câu hỏi nó trả lời:** Promotion có đang chạy đúng vào những tháng mà inventory của category đó đang hết hàng không — và nếu có, thiệt hại kép là bao nhiêu?

- **Chart type:** Timeline chart — trục X là thời gian, mỗi row là một category
  - Màu nền đỏ = tháng có `stockout_flag = 1`
  - Overlay bar xanh = khoảng thời gian promotion đang chạy cho category đó
  - Điểm giao nhau đỏ × xanh = "double damage event"
- **Nguồn dữ liệu:** `promotions` JOIN `inventory` ON `applicable_category` và thời gian overlap
- **Logic join:**
  ```sql
  SELECT p.promo_id, p.promo_name, p.applicable_category,
         i.snapshot_date, i.stockout_days, i.fill_rate
  FROM promotions p
  JOIN inventory i
    ON i.category = p.applicable_category
    AND i.snapshot_date BETWEEN p.start_date AND p.end_date
  WHERE i.stockout_flag = 1
  ```
- **Estimated lost revenue per overlap event:**
  ```
  stockout_days × avg_daily_gross_profit_trong_promo_period
  ```
- **Kết nối với reviews:** Kiểm tra avg rating trong 7–14 ngày sau mỗi double damage event. Nếu rating giảm, đây là chuỗi nhân quả 4 bảng được chứng minh bằng data.

> **Tại sao chart này không ai nghĩ tới:** Đây là join giữa `promotions` và `inventory` — hai bảng không có FK trực tiếp. Join phải thực hiện qua `applicable_category` và time overlap. Insight: công ty đang tốn tiền chạy promo để tăng demand đúng lúc supply bằng không.

---

## Act 5 — Prescriptive: Kê Đơn
**Câu hỏi trung tâm:** Cụ thể nên làm gì, với promo nào, và impact tài chính là bao nhiêu?

---

### Chart 7 — Promo Triage

**Câu hỏi nó trả lời:** Với từng promo cụ thể, verdict là gì và estimated annual impact của việc thực thi verdict đó là bao nhiêu VND?

- **Chart type:** Horizontal ranked bar chart
- **Metric:** Net contribution per promo — xanh nếu dương, đỏ nếu âm
- **Label:** Top 5 bar gắn "KEEP", Bottom 5 gắn "CUT"
- **Nguồn dữ liệu:** OAT, group by `promo_id`
- **Kèm prescriptive table** (save thành `promo_triage_table.csv`):

| promo_id | promo_name | verdict | net_contribution | estimated_annual_impact_vnd | return_rate | stockout_overlap |
|---|---|---|---|---|---|---|
| P001 | ... | KEEP | +X | +X/năm | 8% | Không |
| P002 | ... | CUT | −Y | −Y saved/năm | 24% | Có |
| P003 | ... | RESCHEDULE | +Z | +Z nếu dịch Q4 | 12% | Có |

- **Verdict logic:**
  - `KEEP`: net_contribution > 0 AND return_rate < 15% AND không có stockout overlap
  - `CUT`: net_contribution < 0
  - `RESCHEDULE`: net_contribution > 0 BUT có stockout overlap hoặc return_rate > 15%

- **Trade-off bắt buộc:** "Cắt bottom 5 promo → revenue giảm X%, gross profit tăng Y%."

---

## Chuỗi nhân quả hoàn chỉnh — narrative linkage

```
Chart 1+2 (Act 1)
"Revenue tăng nhưng X tỷ đang rò rỉ qua discount và return — margin thực giảm từ Y% xuống Z%."
    ↓
Chart 3+4 (Act 2)
"Nguyên nhân một phần: promo kéo khách chất lượng thấp — return X lần nhiều hơn, profit per customer thấp hơn Z%."
    ↓
Chart 5 (Act 3)
"Nếu không thay đổi, gross margin sẽ chạm [X]% vào năm [Y]."
    ↓
Chart 6 (Act 4)
"Promo còn gây hại thêm: chạy đúng lúc hết hàng — estimated lost revenue [X] tỷ từ double damage events."
    ↓
Chart 7 (Act 5)
"Verdict cụ thể: keep [N] promo, cut [M] promo, reschedule [K] promo. Net impact: gross profit tăng [X]% nếu thực thi."
```

---

## Danh sách CẮT — không thêm dù còn thời gian

| Phân tích | Lý do cắt |
|---|---|
| Traffic funnel (web_traffic × order_source) | `traffic_source` và `order_source` không guarantee cùng attribution system. Không defend được nếu bị hỏi methodology. |
| Web traffic lag cross-correlation | Trả lời câu hỏi của Part 3 (forecasting), không Part 2. Làm gãy narrative. |
| Acquisition channel standalone | Chỉ giữ nếu có evidence promo kéo acquisition từ kênh LTV thấp — nếu không rõ trong data: cắt. |
| Review sentiment chi tiết | ROI thấp. Rating aggregate trong Chart 4 và Chart 6 đã đủ. |
| Seasonality heatmap | Optional. Chỉ thêm sau khi tất cả 7 charts xong hoàn toàn. |

---

## Checklist trước khi nộp

- [ ] OAT được save thành `oat.parquet` — không join lại ở bất kỳ notebook nào
- [ ] Mỗi chart có: title, axis labels, data source annotation, headline number
- [ ] Mỗi chart theo sau bởi đúng hai câu: **(1) Finding** — con số cụ thể. **(2) Implication** — business action
- [ ] Headline numbers được đặt nổi bật, không để người đọc tự tính
- [ ] Prescriptive table có cột `estimated_annual_impact_vnd` với số tính từ data thực
- [ ] Cột `stockout_overlap` trong triage table được điền từ kết quả Chart 6
- [ ] Trade-off được phát biểu rõ: "Cắt → revenue −X%, profit +Y%"
- [ ] Narrative linkage: mỗi act kết thúc bằng một câu dẫn sang act tiếp theo
- [ ] Random seed set nếu có bất kỳ sampling nào: `np.random.seed(42)`