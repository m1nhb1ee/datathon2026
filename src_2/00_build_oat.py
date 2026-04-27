import numpy as np
import random
import pandas as pd
import os

np.random.seed(42)
random.seed(42)

PALETTE = {
    "positive": "#2d6a4f",
    "negative": "#c1121f",
    "promo":    "#e76f51",
    "organic":  "#457b9d",
    "neutral":  "#adb5bd",
    "warning":  "#e9c46a",
}

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD ALL CSVs
# ---------------------------------------------------------------------------
print("Loading CSVs...")

orders      = pd.read_csv("data/orders.csv",      parse_dates=["order_date"],
                           dtype={"zip": str}, low_memory=False)
order_items = pd.read_csv("data/order_items.csv", low_memory=False)
products    = pd.read_csv("data/products.csv",    low_memory=False)
promotions  = pd.read_csv("data/promotions.csv",  parse_dates=["start_date", "end_date"])
customers   = pd.read_csv("data/customers.csv",   parse_dates=["signup_date"],
                           dtype={"zip": str})
geography   = pd.read_csv("data/geography.csv",   dtype={"zip": str}, low_memory=False)
payments    = pd.read_csv("data/payments.csv",    low_memory=False)
shipments   = pd.read_csv("data/shipments.csv",   parse_dates=["ship_date", "delivery_date"])
returns_df  = pd.read_csv("data/returns.csv",     parse_dates=["return_date"])
reviews     = pd.read_csv("data/reviews.csv",     parse_dates=["review_date"])
sales       = pd.read_csv("data/sales.csv",       parse_dates=["Date"])
inventory   = pd.read_csv("data/inventory.csv",   parse_dates=["snapshot_date"])

print("\n--- SOURCE ROW COUNTS ---")
for name, df in [
    ("orders",      orders),
    ("order_items", order_items),
    ("products",    products),
    ("promotions",  promotions),
    ("customers",   customers),
    ("geography",   geography),
    ("payments",    payments),
    ("shipments",   shipments),
    ("returns",     returns_df),
    ("reviews",     reviews),
    ("sales",       sales),
    ("inventory",   inventory),
]:
    print(f"  {name:15s}: {len(df):>9,} rows")

# ---------------------------------------------------------------------------
# 2. VALIDATION — products: cogs < price
# ---------------------------------------------------------------------------
invalid_cogs = products[products["cogs"] >= products["price"]]
if len(invalid_cogs) > 0:
    print(f"\nWARNING: {len(invalid_cogs)} products have cogs >= price")
    print(invalid_cogs[["product_id", "product_name", "price", "cogs"]].head())
else:
    print("\nPASS: cogs < price for all products")

# ---------------------------------------------------------------------------
# 3. BUILD OAT — join order: order_items → orders → products → promotions
#                             → customers → geography → payments → returns → reviews
# ---------------------------------------------------------------------------
print("\nBuilding OAT...")

oat = order_items.copy()

def normalize_promo_id(series):
    values = series.astype("string").str.strip()
    is_null = values.isna() | values.eq("") | values.str.lower().eq("nan")
    numeric = pd.to_numeric(values.str.replace(r"^PROMO-", "", regex=True), errors="coerce")
    normalized = "PROMO-" + numeric.astype("Int64").astype("string").str.zfill(4)
    return normalized.mask(is_null)

oat["promo_id"] = normalize_promo_id(oat["promo_id"])
oat["promo_id_2"] = normalize_promo_id(oat["promo_id_2"])
promotions["promo_id"] = normalize_promo_id(promotions["promo_id"])

# (1) orders — get order_date, customer_id, zip, etc.
oat = oat.merge(
    orders[["order_id", "order_date", "customer_id", "zip",
            "order_status", "payment_method", "device_type", "order_source"]],
    on="order_id", how="left"
)

# (2) products — category, cogs, price
oat = oat.merge(
    products[["product_id", "product_name", "category", "segment",
              "size", "color", "price", "cogs"]],
    on="product_id", how="left"
)

# (3) promotions — order_items.discount_amount is the applied discount
# ground truth. Master promotion columns are joined as metadata and validation
# context, not used to recalculate historical discount amounts.
oat["has_promo"] = oat["promo_id"].notna() | oat["promo_id_2"].notna()
promo_cols = [
    "promo_id", "promo_name", "promo_type", "discount_value",
    "start_date", "end_date", "applicable_category", "promo_channel",
    "stackable_flag", "min_order_value",
]
oat = oat.merge(
    promotions[promo_cols],
    on="promo_id", how="left"
)
promo2 = promotions[promo_cols].rename(columns={
    col: f"{col}_2" for col in promo_cols if col != "promo_id"
}).rename(columns={"promo_id": "promo_id_2"})
oat = oat.merge(promo2, on="promo_id_2", how="left")

# (4) customers
oat = oat.merge(
    customers[["customer_id", "signup_date", "gender", "age_group", "acquisition_channel"]],
    on="customer_id", how="left"
)

# (5) geography via orders.zip — deduplicate on zip
geo = geography[["zip", "region", "district"]].drop_duplicates("zip")
oat = oat.merge(geo, on="zip", how="left")

# (6) payments
oat = oat.merge(
    payments[["order_id", "payment_value", "installments"]],
    on="order_id", how="left"
)

# (7) shipments
oat = oat.merge(
    shipments[["order_id", "shipping_fee"]],
    on="order_id", how="left"
)
oat["shipping_fee"] = oat["shipping_fee"].fillna(0.0)

# (8) returns — aggregate by (order_id, product_id) then left join
returns_agg = returns_df.groupby(["order_id", "product_id"]).agg(
    refund_amount=("refund_amount", "sum"),
    return_quantity=("return_quantity", "sum"),
    return_reason=("return_reason", "first"),
).reset_index()
returns_agg["is_returned"] = True

oat = oat.merge(
    returns_agg[["order_id", "product_id", "is_returned", "refund_amount", "return_quantity"]],
    on=["order_id", "product_id"], how="left"
)
oat["is_returned"]   = oat["is_returned"].fillna(False)
oat["refund_amount"] = oat["refund_amount"].fillna(0.0)
oat["return_quantity"] = oat["return_quantity"].fillna(0.0)

# (9) reviews — first review per (order_id, product_id) by date
rev = (reviews.sort_values("review_date")
              .groupby(["order_id", "product_id"])
              .first()
              .reset_index())
oat = oat.merge(rev[["order_id", "product_id", "rating"]],
                on=["order_id", "product_id"], how="left")

# ---------------------------------------------------------------------------
# 4. DERIVED COLUMNS
# ---------------------------------------------------------------------------
oat["gross_revenue_line"] = oat["unit_price"] * oat["quantity"]
oat["net_revenue"]        = oat["gross_revenue_line"] - oat["discount_amount"]
oat["cogs_quantity"]      = oat["cogs"] * oat["quantity"]
oat["gp_before_refund"]   = oat["net_revenue"] - oat["cogs_quantity"]
oat["discount_rate"]      = np.where(
    oat["gross_revenue_line"] > 0,
    oat["discount_amount"] / oat["gross_revenue_line"],
    0.0
)

# Allocate refund and shipping at line level. Returns are keyed by
# (order_id, product_id), while a few order-product pairs have multiple item
# rows; allocation prevents refund overcounting.
pair_net = oat.groupby(["order_id", "product_id"])["net_revenue"].transform("sum")
pair_qty = oat.groupby(["order_id", "product_id"])["quantity"].transform("sum")
oat["refund_allocated"] = np.where(
    pair_net > 0,
    oat["refund_amount"] * oat["net_revenue"] / pair_net,
    np.where(pair_qty > 0, oat["refund_amount"] * oat["quantity"] / pair_qty, 0.0)
)

order_net = oat.groupby("order_id")["net_revenue"].transform("sum")
oat["shipping_fee_allocated"] = np.where(
    order_net > 0,
    oat["shipping_fee"] * oat["net_revenue"] / order_net,
    0.0
)

oat["gp_after_refund"] = oat["gp_before_refund"] - oat["refund_allocated"]
oat["gp_after_refund_shipping"] = oat["gp_after_refund"] - oat["shipping_fee_allocated"]

# Backward-compatible alias used by older notebooks: pre-refund GP.
oat["gross_profit"] = oat["gp_before_refund"]
oat["return_leakage"] = oat["refund_allocated"]

oat["primary_promo_date_ok"] = np.where(
    oat["promo_id"].notna(),
    (oat["order_date"] >= oat["start_date"]) & (oat["order_date"] <= oat["end_date"]),
    np.nan,
)
oat["primary_promo_category_ok"] = np.where(
    oat["promo_id"].notna(),
    oat["applicable_category"].isna() | (oat["category"] == oat["applicable_category"]),
    np.nan,
)
oat["secondary_promo_date_ok"] = np.where(
    oat["promo_id_2"].notna(),
    (oat["order_date"] >= oat["start_date_2"]) & (oat["order_date"] <= oat["end_date_2"]),
    np.nan,
)
oat["secondary_promo_category_ok"] = np.where(
    oat["promo_id_2"].notna(),
    oat["applicable_category_2"].isna() | (oat["category"] == oat["applicable_category_2"]),
    np.nan,
)

# customer_order_seq at order level → join back to items
order_seq = orders[["customer_id", "order_id", "order_date"]].copy()
order_seq["customer_order_seq"] = (
    order_seq.groupby("customer_id")["order_date"]
             .rank(method="first")
             .astype(int)
)
oat = oat.merge(order_seq[["order_id", "customer_order_seq"]], on="order_id", how="left")
oat["is_repeat"] = oat["customer_order_seq"] > 1

# ---------------------------------------------------------------------------
# 5. VALIDATION
# ---------------------------------------------------------------------------
print("\n--- NULL COUNTS (key columns) ---")
for col in ["order_id", "customer_id", "product_id", "net_revenue", "gross_profit"]:
    print(f"  {col:20s}: {oat[col].isna().sum():,} NULLs")

neg_nr = (oat["net_revenue"] < 0).sum()
if neg_nr > 0:
    print(f"\nWARNING: {neg_nr:,} rows have net_revenue < 0 — clipping to 0")
    oat["net_revenue"]  = oat["net_revenue"].clip(lower=0)
    oat["gp_before_refund"] = oat["net_revenue"] - oat["cogs_quantity"]
    oat["gp_after_refund"] = oat["gp_before_refund"] - oat["refund_allocated"]
    oat["gp_after_refund_shipping"] = oat["gp_after_refund"] - oat["shipping_fee_allocated"]
    oat["gross_profit"] = oat["gp_before_refund"]
else:
    print("\nPASS: net_revenue >= 0 for all rows")

print("\n--- SUMMARY STATS ---")
print(f"  gp_before_refund — min: {oat['gp_before_refund'].min():>15,.0f}  "
      f"max: {oat['gp_before_refund'].max():>15,.0f}  "
      f"mean: {oat['gp_before_refund'].mean():>12,.0f}")
print(f"  gp_after_refund  — min: {oat['gp_after_refund'].min():>15,.0f}  "
      f"max: {oat['gp_after_refund'].max():>15,.0f}  "
      f"mean: {oat['gp_after_refund'].mean():>12,.0f}")
print(f"  gp_after_ref_ship— min: {oat['gp_after_refund_shipping'].min():>15,.0f}  "
      f"max: {oat['gp_after_refund_shipping'].max():>15,.0f}  "
      f"mean: {oat['gp_after_refund_shipping'].mean():>12,.0f}")
print(f"  refund_allocated — total: {oat['refund_allocated'].sum():>15,.0f}")
print(f"  shipping_alloc   — total: {oat['shipping_fee_allocated'].sum():>15,.0f}")
print(f"  discount_rate — min: {oat['discount_rate'].min():.4f}  "
      f"max: {oat['discount_rate'].max():.4f}  "
      f"mean: {oat['discount_rate'].mean():.4f}")
print(f"  return_rate   — {oat['is_returned'].mean():.4f}  "
      f"({oat['is_returned'].sum():,} returned items)")

print("\n--- PROMO RULE AUDIT ---")
primary = oat["promo_id"].notna()
secondary = oat["promo_id_2"].notna()
print(f"  primary promo rows       : {primary.sum():,}")
print(f"  secondary promo rows     : {secondary.sum():,}")
print(f"  primary date mismatches  : {(primary & ~oat['primary_promo_date_ok'].astype(bool)).sum():,}")
print(f"  primary category mismatch: {(primary & ~oat['primary_promo_category_ok'].astype(bool)).sum():,}")
print(f"  secondary date mismatches: {(secondary & ~oat['secondary_promo_date_ok'].astype(bool)).sum():,}")
print(f"  secondary category mismatch: {(secondary & ~oat['secondary_promo_category_ok'].astype(bool)).sum():,}")

# ---------------------------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------------------------
out_path = "src/oat.parquet"
oat.to_parquet(out_path, index=False)
print(f"\nOAT BUILD COMPLETE — {len(oat):,} rows saved to {out_path}")
