# Đề Thi Vòng 1 — DATATHON 2026: THE GRIDBREAKER
**Hosted by VinTelligence — VinUniversity Data Science & AI Club**

---

## 1. Mô tả Dữ liệu

### 1.1 Giới thiệu

Bộ dữ liệu mô phỏng hoạt động của một doanh nghiệp thời trang thương mại điện tử tại Việt Nam trong giai đoạn từ 04/07/2012 đến 31/12/2022. Dữ liệu bao gồm các file CSV, được chia thành 4 lớp: Master, Transaction, Analytical, Operational.

**Phân chia dữ liệu cho bài toán dự báo:**
- `sales.csv` (train): 04/07/2012 → 31/12/2022
- `sales_test.csv` (test): 01/01/2023 → 01/07/2024

---

### 1.2 Danh sách file dữ liệu

| # | File | Lớp | Mô tả |
|---|---|---|---|
| 1 | products.csv | Master | Danh mục sản phẩm |
| 2 | customers.csv | Master | Thông tin khách hàng |
| 3 | promotions.csv | Master | Các chiến dịch khuyến mãi |
| 4 | geography.csv | Master | Danh sách mã bưu chính các vùng |
| 5 | orders.csv | Transaction | Thông tin đơn hàng |
| 6 | order_items.csv | Transaction | Chi tiết từng dòng sản phẩm trong đơn |
| 7 | payments.csv | Transaction | Thông tin thanh toán (1:1 với đơn hàng) |
| 8 | shipments.csv | Transaction | Thông tin vận chuyển |
| 9 | returns.csv | Transaction | Các sản phẩm bị trả lại |
| 10 | reviews.csv | Transaction | Đánh giá sản phẩm sau giao hàng |
| 11 | sales.csv | Analytical | Dữ liệu doanh thu huấn luyện |
| 12 | sample_submission.csv | Analytical | Định dạng file nộp bài |
| 13 | inventory.csv | Operational | Ảnh chụp tồn kho cuối tháng |
| 14 | web_traffic.csv | Operational | Lưu lượng truy cập website hàng ngày |

---

### 1.3 Bảng Master

#### products.csv — Danh mục sản phẩm

| Cột | Kiểu | Mô tả |
|---|---|---|
| product_id | int | Khoá chính |
| product_name | str | Tên sản phẩm |
| category | str | Danh mục sản phẩm |
| segment | str | Phân khúc thị trường |
| size | str | Kích cỡ sản phẩm |
| color | str | Nhãn màu sản phẩm |
| price | float | Giá bán lẻ |
| cogs | float | Giá vốn hàng bán |

**Ràng buộc:** `cogs < price` với mọi sản phẩm.

---

#### customers.csv — Khách hàng

| Cột | Kiểu | Mô tả |
|---|---|---|
| customer_id | int | Khoá chính |
| zip | int | Mã bưu chính (FK → geography.zip) |
| city | str | Tên thành phố |
| signup_date | date | Ngày đăng ký tài khoản |
| gender | str | Giới tính (nullable) |
| age_group | str | Nhóm tuổi (nullable) |
| acquisition_channel | str | Kênh tiếp thị (nullable) |

---

#### promotions.csv — Chương trình khuyến mãi

| Cột | Kiểu | Mô tả |
|---|---|---|
| promo_id | str | Khoá chính |
| promo_name | str | Tên chiến dịch kèm năm |
| promo_type | str | Loại giảm giá: `percentage` hoặc `fixed` |
| discount_value | float | Giá trị giảm |
| start_date | date | Ngày bắt đầu chiến dịch |
| end_date | date | Ngày kết thúc chiến dịch |
| applicable_category | str | Danh mục áp dụng (null = tất cả) |
| promo_channel | str | Kênh phân phối (nullable) |
| stackable_flag | int | Cho phép áp dụng đồng thời nhiều khuyến mãi |
| min_order_value | float | Giá trị đơn hàng tối thiểu (nullable) |

**Công thức giảm giá:**
- `percentage`: `discount_amount = quantity × unit_price × (discount_value / 100)`
- `fixed`: `discount_amount = quantity × discount_value`

---

#### geography.csv — Địa lý

| Cột | Kiểu | Mô tả |
|---|---|---|
| zip | int | Khoá chính |
| city | str | Tên thành phố |
| region | str | Vùng địa lý |
| district | str | Tên quận/huyện |

---

### 1.4 Bảng Transaction

#### orders.csv — Đơn hàng

| Cột | Kiểu | Mô tả |
|---|---|---|
| order_id | int | Khoá chính |
| order_date | date | Ngày đặt hàng |
| customer_id | int | FK → customers.customer_id |
| zip | int | Mã bưu chính giao hàng (FK → geography.zip) |
| order_status | str | Trạng thái đơn hàng |
| payment_method | str | Phương thức thanh toán |
| device_type | str | Thiết bị khách hàng dùng |
| order_source | str | Kênh marketing dẫn đến đơn hàng |

---

#### order_items.csv — Chi tiết đơn hàng

| Cột | Kiểu | Mô tả |
|---|---|---|
| order_id | int | FK → orders.order_id |
| product_id | int | FK → products.product_id |
| quantity | int | Số lượng sản phẩm |
| unit_price | float | Đơn giá |
| discount_amount | float | Tổng tiền giảm giá cho dòng này |
| promo_id | str | FK → promotions.promo_id (nullable) |
| promo_id_2 | str | FK → promotions.promo_id, khuyến mãi thứ hai (nullable) |

---

#### payments.csv — Thanh toán

| Cột | Kiểu | Mô tả |
|---|---|---|
| order_id | int | FK → orders.order_id (quan hệ 1:1) |
| payment_method | str | Phương thức thanh toán |
| payment_value | float | Tổng giá trị thanh toán |
| installments | int | Số kỳ trả góp |

---

#### shipments.csv — Vận chuyển

| Cột | Kiểu | Mô tả |
|---|---|---|
| order_id | int | FK → orders.order_id |
| ship_date | date | Ngày gửi hàng |
| delivery_date | date | Ngày giao hàng |
| shipping_fee | float | Phí vận chuyển |

**Lưu ý:** Chỉ tồn tại cho đơn hàng có status = `shipped`, `delivered`, hoặc `returned`.

---

#### returns.csv — Trả hàng

| Cột | Kiểu | Mô tả |
|---|---|---|
| return_id | str | Khoá chính |
| order_id | int | FK → orders.order_id |
| product_id | int | FK → products.product_id |
| return_date | date | Ngày khách gửi trả |
| return_reason | str | Lý do trả hàng |
| return_quantity | int | Số lượng trả |
| refund_amount | float | Số tiền hoàn lại |

---

#### reviews.csv — Đánh giá

| Cột | Kiểu | Mô tả |
|---|---|---|
| review_id | str | Khoá chính |
| order_id | int | FK → orders.order_id |
| product_id | int | FK → products.product_id |
| customer_id | int | FK → customers.customer_id |
| review_date | date | Ngày gửi đánh giá |
| rating | int | Điểm từ 1 đến 5 |
| review_title | str | Tiêu đề đánh giá |

---

### 1.5 Bảng Analytical

#### sales.csv — Dữ liệu doanh thu

| Cột | Kiểu | Mô tả |
|---|---|---|
| Date | date | Ngày đặt hàng |
| Revenue | float | Tổng doanh thu thuần |
| COGS | float | Tổng giá vốn hàng bán |

| Split | File | Khoảng thời gian |
|---|---|---|
| Train | sales.csv | 04/07/2012 – 31/12/2022 |
| Test | sales_test.csv | 01/01/2023 – 01/07/2024 |

---

### 1.6 Bảng Operational

#### inventory.csv — Tồn kho

| Cột | Kiểu | Mô tả |
|---|---|---|
| snapshot_date | date | Ngày chụp tồn kho (cuối tháng) |
| product_id | int | FK → products.product_id |
| stock_on_hand | int | Số lượng tồn kho cuối tháng |
| units_received | int | Số lượng nhập kho trong tháng |
| units_sold | int | Số lượng bán ra trong tháng |
| stockout_days | int | Số ngày hết hàng trong tháng |
| days_of_supply | float | Số ngày tồn kho có thể đáp ứng |
| fill_rate | float | Tỷ lệ đơn hàng được đáp ứng đủ |
| stockout_flag | int | 1 nếu tháng có xảy ra hết hàng |
| overstock_flag | int | 1 nếu tồn kho vượt mức cần thiết |
| reorder_flag | int | 1 nếu cần tái đặt hàng sớm |
| sell_through_rate | float | Tỷ lệ hàng đã bán / tổng hàng sẵn có |
| product_name | str | Tên sản phẩm |
| category | str | Danh mục sản phẩm |
| segment | str | Phân khúc sản phẩm |
| year | int | Năm trích từ snapshot_date |
| month | int | Tháng trích từ snapshot_date |

---

#### web_traffic.csv — Lưu lượng truy cập

| Cột | Kiểu | Mô tả |
|---|---|---|
| date | date | Ngày ghi nhận |
| sessions | int | Tổng số phiên truy cập |
| unique_visitors | int | Số lượt khách duy nhất |
| page_views | int | Tổng số lượt xem trang |
| bounce_rate | float | Tỷ lệ thoát |
| avg_session_duration_sec | float | Thời gian trung bình mỗi phiên (giây) |
| traffic_source | str | Kênh nguồn traffic |

---

### 1.7 Quan hệ giữa các bảng (Cardinality)

| Quan hệ | Cardinality |
|---|---|
| orders ↔ payments | 1 : 1 |
| orders ↔ shipments | 1 : 0 hoặc 1 (status shipped/delivered/returned) |
| orders ↔ returns | 1 : 0 hoặc nhiều (status returned) |
| orders ↔ reviews | 1 : 0 hoặc nhiều (status delivered, ~20%) |
| order_items ↔ promotions | nhiều : 0 hoặc 1 |
| products ↔ inventory | 1 : nhiều (1 dòng/sản phẩm/tháng) |

---

## 2. Đề Bài

### 2.1 Phần 1 — Câu hỏi Trắc nghiệm (20 điểm)

Mỗi câu đúng 2 điểm. Không trừ điểm sai.

**Q1.** Trong số các khách hàng có nhiều hơn một đơn hàng, trung vị số ngày giữa hai lần mua liên tiếp (inter-order gap) xấp xỉ là bao nhiêu?
- A) 30 ngày / B) 90 ngày / C) 144 ngày / D) 365 ngày

**Q2.** Phân khúc sản phẩm (segment) nào có tỷ suất lợi nhuận gộp trung bình cao nhất, với công thức `(price − cogs) / price`?
- A) Premium / B) Performance / C) Activewear / D) Standard

**Q3.** Trong các bản ghi trả hàng thuộc danh mục Streetwear, lý do trả hàng nào xuất hiện nhiều nhất?
- A) defective / B) wrong_size / C) changed_mind / D) not_as_described

**Q4.** Nguồn truy cập (traffic_source) nào có tỷ lệ thoát trung bình thấp nhất?
- A) organic_search / B) paid_search / C) email_campaign / D) social_media

**Q5.** Tỷ lệ phần trăm các dòng trong order_items.csv có áp dụng khuyến mãi (promo_id không null)?
- A) 12% / B) 25% / C) 39% / D) 54%

**Q6.** Nhóm tuổi nào có số đơn hàng trung bình trên mỗi khách hàng cao nhất?
- A) 55+ / B) 25–34 / C) 35–44 / D) 45–54

**Q7.** Vùng (region) nào tạo ra tổng doanh thu cao nhất trong sales_train.csv?
- A) West / B) Central / C) East / D) Cả ba xấp xỉ bằng nhau

**Q8.** Trong các đơn hàng bị cancelled, phương thức thanh toán nào được dùng nhiều nhất?
- A) credit_card / B) cod / C) paypal / D) bank_transfer

**Q9.** Kích thước sản phẩm nào có tỷ lệ trả hàng cao nhất (returns / order_items)?
- A) S / B) M / C) L / D) XL

**Q10.** Kế hoạch trả góp nào có giá trị thanh toán trung bình cao nhất?
- A) 1 kỳ / B) 3 kỳ / C) 6 kỳ / D) 12 kỳ

---

### 2.2 Phần 2 — Trực quan hoá và Phân tích (60 điểm)

#### Tiêu chí chấm

| Tiêu chí | Điểm tối đa | Mô tả |
|---|---|---|
| Chất lượng trực quan hoá | 15 | Biểu đồ có tiêu đề, nhãn trục, chú thích; loại biểu đồ phù hợp |
| Chiều sâu phân tích | 25 | Bao phủ Descriptive → Diagnostic → Predictive → Prescriptive |
| Insight kinh doanh | 15 | Đề xuất hành động khả thi, liên kết dữ liệu với quyết định |
| Tính sáng tạo & kể chuyện | 5 | Góc nhìn độc đáo, kết hợp nhiều bảng có chủ đích |

**4 cấp độ phân tích:**

| Cấp độ | Câu hỏi | Đánh giá |
|---|---|---|
| Descriptive | What happened? | Thống kê tổng hợp chính xác, biểu đồ rõ ràng |
| Diagnostic | Why did it happen? | Giả thuyết nhân quả, so sánh phân khúc |
| Predictive | What is likely to happen? | Ngoại suy xu hướng, phân tích tính mùa vụ |
| Prescriptive | What should we do? | Đề xuất hành động được hỗ trợ bởi dữ liệu |

---

### 2.3 Phần 3 — Mô hình Dự báo Doanh thu (20 điểm)

**Bài toán:** Dự báo cột `Revenue` trong khoảng 01/01/2023 – 01/07/2024.

**Chỉ số đánh giá:** MAE, RMSE, R²

**Định dạng file nộp:**
```
Date,Revenue,COGS
2023-01-01,26607.2,2585.15
2023-01-02,1007.89,163.0
...
```

**Ràng buộc:**
1. Không dùng dữ liệu ngoài
2. Đính kèm toàn bộ mã nguồn, set random seed
3. Giải thích model bằng SHAP values hoặc feature importances

---

## 3. Thang điểm

| Phần | Nội dung | Điểm | Tỷ trọng |
|---|---|---|---|
| 1 | Câu hỏi Trắc nghiệm | 20 | 20% |
| 2 | Trực quan hoá & Phân tích | 60 | 60% |
| 3 | Mô hình Dự báo | 20 | 20% |
| | **Tổng** | **100** | **100%** |

---

## 4. Hướng dẫn Nộp bài

- Nộp kết quả dự báo: https://www.kaggle.com/competitions/datathon-2026-round-1
- Báo cáo: dùng NeurIPS LaTeX template, tối đa 4 trang
- Form nộp bài: điền đáp án trắc nghiệm, upload PDF, link GitHub, link Kaggle
- GitHub phải public hoặc cấp quyền truy cập cho ban tổ chức
- Ít nhất 1 thành viên tham gia trực tiếp Vòng Chung kết ngày 23/05/2026 tại VinUniversity, Hà Nội
