import numpy as np
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

# ---------------------------------------------------------------------------
# LOAD OAT
# ---------------------------------------------------------------------------
print("Loading oat.parquet...")
oat = pd.read_parquet("oat.parquet")
oat["order_date"] = pd.to_datetime(oat["order_date"])
print(f"  {len(oat):,} rows loaded")

# ===========================================================================
# CHART 1 — Revenue Anatomy (Stacked Area)
# ===========================================================================
print("\nBuilding Chart 1 — Revenue Anatomy...")

try:
    monthly = oat.groupby(pd.Grouper(key="order_date", freq="ME")).agg(
        gross_revenue=("gross_revenue_line", "sum"),
        discount_leakage=("discount_amount", "sum"),
        return_leakage=("refund_amount", "sum"),
        cogs_total=("cogs_quantity", "sum"),
        gross_profit=("gross_profit", "sum"),
    )
except Exception:
    monthly = oat.groupby(pd.Grouper(key="order_date", freq="M")).agg(
        gross_revenue=("gross_revenue_line", "sum"),
        discount_leakage=("discount_amount", "sum"),
        return_leakage=("refund_amount", "sum"),
        cogs_total=("cogs_quantity", "sum"),
        gross_profit=("gross_profit", "sum"),
    )

# Chart net_revenue = gross − discount − return (as specified)
monthly["net_revenue_chart"] = (
    monthly["gross_revenue"] - monthly["discount_leakage"] - monthly["return_leakage"]
)
# Real gross profit accounting for returns
monthly["gp_real"] = monthly["net_revenue_chart"] - monthly["cogs_total"]
monthly["gp_real"] = monthly["gp_real"].clip(lower=0)

# Cumulative annotation numbers
total_discount = monthly["discount_leakage"].sum()
total_return   = monthly["return_leakage"].sum()

# Margin by year
oat["year"] = oat["order_date"].dt.year
yr_2012 = oat[oat["year"] == oat["year"].min()]
yr_2022 = oat[oat["year"] == oat["year"].max()]
margin_2012 = yr_2012["gross_profit"].sum() / yr_2012["net_revenue"].sum() * 100
margin_2022 = yr_2022["gross_profit"].sum() / yr_2022["net_revenue"].sum() * 100

fig, ax = plt.subplots(figsize=(16, 8))

x = monthly.index

# Stack layers from bottom: gp_real, cogs, return_leakage, discount_leakage
ax.stackplot(
    x,
    monthly["gp_real"],
    monthly["cogs_total"],
    monthly["return_leakage"],
    monthly["discount_leakage"],
    labels=["Gross Profit", "COGS", "Return Leakage", "Discount Leakage"],
    colors=[PALETTE["positive"], PALETTE["neutral"], PALETTE["warning"], PALETTE["negative"]],
    alpha=0.85,
)

# Overlay gross_revenue line
ax.plot(x, monthly["gross_revenue"], color="navy", linewidth=1.5,
        linestyle="--", label="Gross Revenue (top line)", alpha=0.7)

# Annotations
ymax = monthly["gross_revenue"].max()
ax.annotate(
    f"Total discount leakage (10yr):\n{total_discount:,.0f} VND",
    xy=(x[len(x)//2], ymax * 0.92),
    fontsize=10, color=PALETTE["negative"],
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["negative"], alpha=0.85),
)
ax.annotate(
    f"Total return leakage (10yr):\n{total_return:,.0f} VND",
    xy=(x[len(x)//2], ymax * 0.75),
    fontsize=10, color=PALETTE["warning"],
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["warning"], alpha=0.85),
)
ax.annotate(
    f"Gross margin: {margin_2012:.1f}% (first yr) → {margin_2022:.1f}% (last yr)",
    xy=(x[len(x)//4], ymax * 0.58),
    fontsize=10, color=PALETTE["positive"],
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["positive"], alpha=0.85),
)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
ax.set_title("Revenue Anatomy — 10-Year Gross Revenue Erosion",
             fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("VND (millions)", fontsize=12)
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
fig.text(0.5, 0.01,
         "Data sources: orders.csv · order_items.csv · products.csv · returns.csv",
         ha="center", fontsize=9, color="grey")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("outputs/charts/chart1_revenue_anatomy.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: outputs/charts/chart1_revenue_anatomy.png")

# ===========================================================================
# CHART 2 — Double Loss Quadrant (Bubble Scatter)
# ===========================================================================
print("\nBuilding Chart 2 — Double Loss Quadrant...")

try:
    cat = oat.groupby("category").agg(
        avg_discount_rate=("discount_rate", "mean"),
        avg_return_rate=("is_returned", "mean"),
        total_gross_revenue=("gross_revenue_line", "sum"),
    ).reset_index().dropna(subset=["category"])

    med_x = cat["avg_discount_rate"].median()
    med_y = cat["avg_return_rate"].median()

    # Normalize bubble size
    sz_min, sz_max = 200, 3000
    rev_min, rev_max = cat["total_gross_revenue"].min(), cat["total_gross_revenue"].max()
    cat["bubble_size"] = (
        (cat["total_gross_revenue"] - rev_min) / (rev_max - rev_min + 1e-9)
        * (sz_max - sz_min) + sz_min
    )

    fig, ax = plt.subplots(figsize=(13, 9))

    scatter = ax.scatter(
        cat["avg_discount_rate"],
        cat["avg_return_rate"],
        s=cat["bubble_size"],
        c=[PALETTE["negative"] if (r["avg_discount_rate"] >= med_x and r["avg_return_rate"] >= med_y)
           else PALETTE["positive"] if (r["avg_discount_rate"] < med_x and r["avg_return_rate"] < med_y)
           else PALETTE["warning"]
           for _, r in cat.iterrows()],
        alpha=0.75, edgecolors="white", linewidth=1.2,
    )

    # Quadrant lines
    ax.axvline(med_x, color="grey", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(med_y, color="grey", linewidth=1.2, linestyle="--", alpha=0.7)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    pad_x = (xlim[1] - xlim[0]) * 0.03
    pad_y = (ylim[1] - ylim[0]) * 0.03

    ax.text(xlim[1] - pad_x, ylim[1] - pad_y, "DOUBLE LOSS",
            ha="right", va="top", fontsize=11, fontweight="bold", color=PALETTE["negative"])
    ax.text(xlim[0] + pad_x, ylim[0] + pad_y, "HEALTHY",
            ha="left", va="bottom", fontsize=11, fontweight="bold", color=PALETTE["positive"])
    ax.text(xlim[0] + pad_x, ylim[1] - pad_y, "DISCOUNT HEAVY",
            ha="left", va="top", fontsize=10, color=PALETTE["warning"])
    ax.text(xlim[1] - pad_x, ylim[0] + pad_y, "RETURN HEAVY",
            ha="right", va="bottom", fontsize=10, color=PALETTE["warning"])

    # Annotate each bubble
    for _, row in cat.iterrows():
        ax.annotate(
            row["category"],
            (row["avg_discount_rate"], row["avg_return_rate"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, alpha=0.9,
        )

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title("Double Loss Quadrant — Discount Rate vs. Return Rate by Category",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Average Discount Rate", fontsize=12)
    ax.set_ylabel("Average Return Rate", fontsize=12)
    fig.text(0.5, 0.01,
             "Data sources: orders.csv · order_items.csv · products.csv · returns.csv",
             ha="center", fontsize=9, color="grey")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("outputs/charts/chart2_double_loss_quadrant.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/charts/chart2_double_loss_quadrant.png")

    # Double-loss categories
    double_loss = cat[
        (cat["avg_discount_rate"] >= med_x) & (cat["avg_return_rate"] >= med_y)
    ]["category"].tolist()

except Exception as e:
    print(f"  ERROR in Chart 2: {e}")
    double_loss = []

# ---------------------------------------------------------------------------
# FINDINGS — ACT 1
# ---------------------------------------------------------------------------
total_gross = oat["gross_revenue_line"].sum()
erosion_pct = (total_discount + total_return) / total_gross * 100

print("""
==================================================
ACT 1 FINDINGS:
==================================================""")
print(f"- Total 10-year discount leakage : {total_discount:>20,.0f} VND")
print(f"- Total 10-year return leakage   : {total_return:>20,.0f} VND")
print(f"- Gross margin first year        : {margin_2012:.1f}%")
print(f"- Gross margin last year         : {margin_2022:.1f}%")
print(f"- 'Double Loss' categories       : {double_loss}")
print(f"- Narrative: \"Revenue grew but {erosion_pct:.1f}% is being eroded "
      f"before reaching gross profit.\"")
print("==================================================")
