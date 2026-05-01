import random
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


np.random.seed(42)
random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = ROOT / "src"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
TABLES = OUT / "tables"
CHARTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "positive": "#2d6a4f",
    "negative": "#c1121f",
    "promo": "#e76f51",
    "organic": "#457b9d",
    "neutral": "#6c757d",
    "muted": "#adb5bd",
    "warning": "#e9c46a",
    "gp": "#2d6a4f",
    "cogs": "#adb5bd",
    "discount": "#c1121f",
    "returns": "#e9c46a",
    "overlap": "#7d0000",
}

GP_COL = "gp_after_refund_shipping"
GP_LABEL = "GP contribution after refunds + shipping"

plt.rcParams.update({
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def money_m(v, _pos=None):
    return f"{v / 1e6:.0f}M"


def pct1(v, _pos=None):
    return f"{v:.1f}%"


def save_chart(fig, name):
    path = CHARTS / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


def load_inputs():
    oat_candidates = [
        ROOT / "src_2" / "oat.parquet",
        SRC / "oat.parquet",
        ROOT / "oat.parquet",
    ]
    oat_path = next((path for path in oat_candidates if path.exists()), oat_candidates[0])
    oat = pd.read_parquet(oat_path)
    oat["order_date"] = pd.to_datetime(oat["order_date"])

    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"])
    promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    inventory = pd.read_csv(DATA / "inventory.csv", parse_dates=["snapshot_date"])
    reviews = pd.read_csv(DATA / "reviews.csv", parse_dates=["review_date"])
    return oat, sales, promotions, inventory, reviews


def promo_id_mask(oat, promo_ids):
    promo_ids = set(pd.Series(promo_ids).dropna())
    if not promo_ids:
        return pd.Series(False, index=oat.index)
    return oat["promo_id"].isin(promo_ids) | oat["promo_id_2"].isin(promo_ids)


def covid_period_from_year(year):
    if year <= 2019:
        return "Pre-COVID"
    if year <= 2021:
        return "COVID"
    return "Post-COVID"


def covid_period_diagnostics(oat, promotions, inventory):
    df = oat.copy()
    df["year"] = df["order_date"].dt.year
    df["period"] = df["year"].map(covid_period_from_year)

    period_order = ["Pre-COVID", "COVID", "Post-COVID"]
    margin = df.groupby("period").agg(
        net_revenue=("net_revenue", "sum"),
        gp_contribution=(GP_COL, "sum"),
        discount_amount=("discount_amount", "sum"),
        rows=("order_id", "size"),
    ).reindex(period_order).reset_index()
    margin["gross_margin"] = margin["gp_contribution"] / margin["net_revenue"]
    margin["discount_rate_on_net_revenue"] = margin["discount_amount"] / margin["net_revenue"]
    margin.to_csv(TABLES / "covid_period_margin_table.csv", index=False)

    yearly = df.groupby("year").agg(
        net_revenue=("net_revenue", "sum"),
        gp_contribution=(GP_COL, "sum"),
        discount_amount=("discount_amount", "sum"),
    ).reset_index()
    yearly["gross_margin"] = yearly["gp_contribution"] / yearly["net_revenue"]
    yearly["period"] = yearly["year"].map(covid_period_from_year)
    yearly.to_csv(TABLES / "covid_yearly_margin_table.csv", index=False)

    promo = df[df["has_promo"]].groupby("promo_id").agg(
        promo_name=("promo_name", "first"),
        gp_contribution=(GP_COL, "sum"),
        net_revenue=("net_revenue", "sum"),
        rows=("order_id", "size"),
    ).reset_index()
    promo = promo.merge(promotions[["promo_id", "start_date", "end_date"]], on="promo_id", how="left")
    promo["period"] = promo["start_date"].dt.year.map(covid_period_from_year)
    promo["gross_margin"] = promo["gp_contribution"] / promo["net_revenue"]
    promo.to_csv(TABLES / "promo_covid_classification_table.csv", index=False)

    promo_period = promo.groupby("period").agg(
        promotions=("promo_id", "nunique"),
        negative_gp_promotions=("gp_contribution", lambda s: int((s < 0).sum())),
        gp_contribution=("gp_contribution", "sum"),
        net_revenue=("net_revenue", "sum"),
        rows=("rows", "sum"),
    ).reindex(period_order).reset_index()
    promo_period["negative_gp_rate"] = promo_period["negative_gp_promotions"] / promo_period["promotions"]
    promo_period["gross_margin"] = promo_period["gp_contribution"] / promo_period["net_revenue"]

    cf_path = TABLES / "promo_counterfactual_proxy_table.csv"
    if cf_path.exists():
        cf = pd.read_csv(cf_path).merge(promotions[["promo_id", "start_date"]], on="promo_id", how="left")
        cf["period"] = cf["start_date"].dt.year.map(covid_period_from_year)
        gap = cf.groupby("period")["gp_gap_vs_matched_organic_margin"].sum().rename("gp_gap_vs_matched_organic_margin")
        promo_period = promo_period.merge(gap, on="period", how="left")

    promo_period.to_csv(TABLES / "promo_covid_period_summary.csv", index=False)

    inv = inventory.copy()
    inv["year"] = inv["snapshot_date"].dt.year
    inv["period"] = inv["year"].map(covid_period_from_year)
    stockout_period = inv.groupby("period").agg(
        stockout_product_share=("stockout_flag", "mean"),
        avg_stockout_days=("stockout_days", "mean"),
        avg_fill_rate=("fill_rate", "mean"),
        rows=("product_id", "size"),
    ).reindex(period_order).reset_index()
    stockout_period.to_csv(TABLES / "stockout_covid_period_summary.csv", index=False)

    stockout_year = inv.groupby("year").agg(
        stockout_product_share=("stockout_flag", "mean"),
        avg_stockout_days=("stockout_days", "mean"),
        avg_fill_rate=("fill_rate", "mean"),
    ).reset_index()
    stockout_year["period"] = stockout_year["year"].map(covid_period_from_year)
    stockout_year.to_csv(TABLES / "stockout_yearly_summary.csv", index=False)

    return {
        "margin": margin,
        "promo": promo_period,
        "stockout": stockout_period,
    }


def chart1_revenue_anatomy(oat):
    monthly = oat.set_index("order_date").resample("ME").agg(
        gross_revenue=("gross_revenue_line", "sum"),
        discount=("discount_amount", "sum"),
        returns=("refund_allocated", "sum"),
        cogs=("cogs_quantity", "sum"),
        shipping=("shipping_fee_allocated", "sum"),
        gross_profit=(GP_COL, "sum"),
        net_revenue=("net_revenue", "sum"),
    )
    monthly["after_discount_return"] = monthly["gross_revenue"] - monthly["discount"] - monthly["returns"]

    by_year = oat.groupby(oat["order_date"].dt.year).agg(
        gross_profit=(GP_COL, "sum"),
        net_revenue=("net_revenue", "sum"),
    )
    by_year["gross_margin_pct"] = by_year["gross_profit"] / by_year["net_revenue"] * 100
    first_year = int(by_year.index.min())
    last_year = int(by_year.index.max())
    margin_first = by_year.loc[first_year, "gross_margin_pct"]
    margin_last = by_year.loc[last_year, "gross_margin_pct"]

    total_discount = oat["discount_amount"].sum()
    total_returns = oat["refund_allocated"].sum()
    total_shipping = oat["shipping_fee_allocated"].sum()
    leakage_pct = (total_discount + total_returns) / oat["gross_revenue_line"].sum() * 100

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(monthly.index, monthly["gross_revenue"], color="#1d3557", linewidth=2.2, label="Gross revenue")
    ax.fill_between(monthly.index, monthly["net_revenue"], monthly["gross_revenue"],
                    color=PALETTE["discount"], alpha=0.25, label="Discount leakage")
    ax.fill_between(monthly.index, monthly["after_discount_return"], monthly["net_revenue"],
                    color=PALETTE["returns"], alpha=0.35, label="Return leakage")
    ax.fill_between(monthly.index, monthly["gross_profit"], monthly["after_discount_return"],
                    color=PALETTE["cogs"], alpha=0.35, label="COGS + shipping")
    ax.fill_between(monthly.index, 0, monthly["gross_profit"],
                    color=PALETTE["gp"], alpha=0.42, label="Gross profit")
    ax.set_title("Revenue Anatomy - what survives after discounts, returns, and COGS",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("VND")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(money_m))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(ncol=3, loc="upper left")
    ax.annotate(
        f"10-year leakage: {(total_discount + total_returns) / 1e6:,.0f}M VND\n"
        f"Discount: {total_discount / 1e6:,.0f}M | Refunds: {total_returns / 1e6:,.0f}M | Shipping: {total_shipping / 1e6:,.1f}M\n"
        f"Post-refund margin: {margin_first:.1f}% ({first_year}) -> {margin_last:.1f}% ({last_year})",
        xy=(0.02, 0.82), xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=PALETTE["neutral"], alpha=0.92),
        fontsize=10,
    )
    fig.text(0.5, 0.01, "Source: oat.parquet (orders, order_items, products, returns)",
             ha="center", fontsize=9, color="grey")
    save_chart(fig, "chart1_revenue_anatomy.png")
    return {
        "total_discount": total_discount,
        "total_returns": total_returns,
        "total_shipping": total_shipping,
        "leakage_pct": leakage_pct,
        "first_year": first_year,
        "last_year": last_year,
        "margin_first": margin_first,
        "margin_last": margin_last,
    }


def order_status_profit_audit(oat):
    status = oat.groupby("order_status", dropna=False).agg(
        rows=("order_id", "size"),
        orders=("order_id", "nunique"),
        net_revenue=("net_revenue", "sum"),
        gp_after_refund_shipping=(GP_COL, "sum"),
        gross_revenue=("gross_revenue_line", "sum"),
    ).reset_index()
    status["net_revenue_share"] = status["net_revenue"] / status["net_revenue"].sum()
    status["gp_share"] = (
        status["gp_after_refund_shipping"] / status["gp_after_refund_shipping"].sum()
    )
    status = status.sort_values("net_revenue", ascending=False)
    status.to_csv(TABLES / "order_status_profit_audit.csv", index=False)
    return status


def chart2_double_loss(oat):
    cat = oat.groupby("category").agg(
        discount_rate=("discount_rate", "mean"),
        return_rate=("is_returned", "mean"),
        gross_revenue=("gross_revenue_line", "sum"),
        rows=("order_id", "size"),
    ).reset_index()
    x_med = cat["discount_rate"].median()
    y_med = cat["return_rate"].median()
    cat["quadrant"] = np.select(
        [
            (cat["discount_rate"] >= x_med) & (cat["return_rate"] >= y_med),
            (cat["discount_rate"] < x_med) & (cat["return_rate"] < y_med),
        ],
        ["Double loss", "Healthy"],
        default="Mixed",
    )

    sizes = 400 + 2600 * cat["gross_revenue"] / cat["gross_revenue"].max()
    colors = [PALETTE["negative"] if q == "Double loss" else PALETTE["positive"] if q == "Healthy" else PALETTE["warning"]
              for q in cat["quadrant"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(cat["discount_rate"] * 100, cat["return_rate"] * 100,
               s=sizes, c=colors, alpha=0.78, edgecolor="white", linewidth=1.2)
    ax.axvline(x_med * 100, color=PALETTE["neutral"], linestyle="--", linewidth=1)
    ax.axhline(y_med * 100, color=PALETTE["neutral"], linestyle="--", linewidth=1)
    for _, r in cat.iterrows():
        ax.annotate(
            f"{r['category']}\n{r['gross_revenue'] / 1e9:.1f}B VND",
            (r["discount_rate"] * 100, r["return_rate"] * 100),
            xytext=(8, 6), textcoords="offset points", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.9),
        )
    ax.set_title("Double Loss Category Map - discount pressure vs. return pressure",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Average discount rate")
    ax.set_ylabel("Item return rate")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    fig.text(0.5, 0.01, "Bubble size = gross revenue. Median lines define relative category risk.",
             ha="center", fontsize=9, color="grey")
    save_chart(fig, "chart2_double_loss_quadrant.png")
    cat.to_csv(TABLES / "category_double_loss_table.csv", index=False)
    return cat


def chart3_promo_roi(oat):
    promo_df = oat[oat["has_promo"]].copy()
    promo = promo_df.groupby("promo_id").agg(
        total_discount_cost=("discount_amount", "sum"),
        total_net_contribution=(GP_COL, "sum"),
        order_count=("order_id", "nunique"),
        return_rate=("is_returned", "mean"),
        promo_name=("promo_name", "first"),
    ).reset_index()
    promo["marker_size"] = 80 + 900 * promo["order_count"] / promo["order_count"].max()
    norm = mcolors.Normalize(vmin=promo["return_rate"].min(), vmax=promo["return_rate"].max())

    fig, ax = plt.subplots(figsize=(14, 9))
    sc = ax.scatter(promo["total_discount_cost"], promo["total_net_contribution"],
                    s=promo["marker_size"], c=promo["return_rate"], cmap=cm.RdYlGn_r,
                    norm=norm, alpha=0.82, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1.4, linestyle="--", alpha=0.75)
    top_frame = promo.nlargest(3, "total_net_contribution")
    bottom_frame = promo.nsmallest(3, "total_net_contribution")
    for frame_name, frame, color, dy in [
        ("top", top_frame, PALETTE["neutral"], 8),
        ("bottom", bottom_frame, PALETTE["negative"], -28),
    ]:
        for _, r in frame.iterrows():
            if frame_name == "top":
                label = "KEEP" if r["total_net_contribution"] > 0 else "LEAST NEGATIVE"
                label_color = PALETTE["positive"] if r["total_net_contribution"] > 0 else "#5c677d"
            else:
                label = "CUT"
                label_color = color
            ax.annotate(
                f"{label}\n{r['promo_name']}",
                (r["total_discount_cost"], r["total_net_contribution"]),
                xytext=(8, dy), textcoords="offset points", fontsize=8, fontweight="bold",
                color=label_color, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=label_color, alpha=0.9),
            )
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Item return rate")
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(money_m))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(money_m))
    ax.set_title("Promo GP Contribution Map - discount cost vs. post-refund GP contribution",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Total discount cost (VND)")
    ax.set_ylabel("GP contribution after refunds + shipping (VND)")
    pct_negative = (promo["total_net_contribution"] < 0).mean() * 100
    ax.annotate(f"{pct_negative:.1f}% of promotions are negative-GP",
                xy=(0.03, 0.94), xycoords="axes fraction", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["negative"], alpha=0.9))
    fig.text(0.5, 0.01, "Source: oat.parquet (order_items, promotions, products, returns)",
             ha="center", fontsize=9, color="grey")
    save_chart(fig, "chart3_promo_roi_scatter.png")
    return promo


def promo_counterfactual_proxy(oat):
    df = oat.copy()
    df["quarter"] = df["order_date"].dt.to_period("Q")
    nonpromo = df[~df["has_promo"]].copy()

    matched_margin = (
        nonpromo.groupby(["category", "quarter"])
        .agg(organic_gp=(GP_COL, "sum"), organic_net_revenue=("net_revenue", "sum"))
        .reset_index()
    )
    matched_margin["matched_organic_margin"] = (
        matched_margin["organic_gp"] / matched_margin["organic_net_revenue"]
    )
    category_margin = (
        nonpromo.groupby("category")
        .agg(category_gp=(GP_COL, "sum"), category_net_revenue=("net_revenue", "sum"))
        .reset_index()
    )
    category_margin["category_organic_margin"] = (
        category_margin["category_gp"] / category_margin["category_net_revenue"]
    )
    global_margin = nonpromo[GP_COL].sum() / nonpromo["net_revenue"].sum()

    promo = df[df["has_promo"]].merge(
        matched_margin[["category", "quarter", "matched_organic_margin"]],
        on=["category", "quarter"], how="left"
    ).merge(
        category_margin[["category", "category_organic_margin"]],
        on="category", how="left"
    )
    promo["counterfactual_margin"] = (
        promo["matched_organic_margin"]
        .fillna(promo["category_organic_margin"])
        .fillna(global_margin)
    )
    promo["expected_gp_at_matched_organic_margin"] = (
        promo["net_revenue"] * promo["counterfactual_margin"]
    )
    promo["gp_gap_vs_matched_organic_margin"] = (
        promo[GP_COL] - promo["expected_gp_at_matched_organic_margin"]
    )

    proxy = promo.groupby("promo_id").agg(
        promo_name=("promo_name", "first"),
        actual_gp=(GP_COL, "sum"),
        expected_gp_at_matched_organic_margin=("expected_gp_at_matched_organic_margin", "sum"),
        gp_gap_vs_matched_organic_margin=("gp_gap_vs_matched_organic_margin", "sum"),
        net_revenue=("net_revenue", "sum"),
        matched_margin_coverage=("matched_organic_margin", lambda s: s.notna().mean()),
        rows=("order_id", "size"),
    ).reset_index()
    proxy["actual_margin"] = proxy["actual_gp"] / proxy["net_revenue"]
    proxy["expected_margin"] = (
        proxy["expected_gp_at_matched_organic_margin"] / proxy["net_revenue"]
    )
    proxy = proxy.sort_values("gp_gap_vs_matched_organic_margin")
    proxy.to_csv(TABLES / "promo_counterfactual_proxy_table.csv", index=False)
    return proxy


def chart4_cohort_quality(oat):
    first_orders = oat.sort_values("order_date").groupby("customer_id").first().reset_index()
    first_orders["cohort"] = first_orders["has_promo"].map({True: "Promo", False: "Organic"})
    oatc = oat.merge(first_orders[["customer_id", "cohort"]], on="customer_id", how="left")

    promo_returns = oatc[oatc["cohort"] == "Promo"]["is_returned"].astype(int)
    organic_returns = oatc[oatc["cohort"] == "Organic"]["is_returned"].astype(int)
    _, p_greater = mannwhitneyu(promo_returns, organic_returns, alternative="greater")

    line_stats = oatc.groupby("cohort").agg(item_return_rate=("is_returned", "mean"))
    cust_orders = oat[["customer_id", "order_id"]].drop_duplicates().groupby("customer_id").size().rename("n_orders").reset_index()
    cust_gp = oat.groupby("customer_id")[GP_COL].sum().rename("lifetime_gp").reset_index()
    cust = first_orders[["customer_id", "cohort"]].merge(cust_orders).merge(cust_gp)
    cust["is_repeat_customer"] = cust["n_orders"] > 1
    cust_stats = cust.groupby("cohort").agg(
        customer_repeat_rate=("is_repeat_customer", "mean"),
        avg_gp_per_customer=("lifetime_gp", "mean"),
        customers=("customer_id", "size"),
    )

    rr_promo = line_stats.loc["Promo", "item_return_rate"]
    rr_org = line_stats.loc["Organic", "item_return_rate"]
    rep_promo = cust_stats.loc["Promo", "customer_repeat_rate"]
    rep_org = cust_stats.loc["Organic", "customer_repeat_rate"]
    gp_promo = cust_stats.loc["Promo", "avg_gp_per_customer"]
    gp_org = cust_stats.loc["Organic", "avg_gp_per_customer"]
    gp_gap = 1 - gp_promo / gp_org

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.set_facecolor("#fafafa")
    labels = ["Promo", "Organic"]
    colors = [PALETTE["promo"], PALETTE["organic"]]

    def bar(ax, vals, title, ylabel, formatter, ylim_pad=0.2):
        bars = ax.bar(labels, vals, color=colors, alpha=0.86, edgecolor="white", width=0.52)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        top = max(vals) * (1 + ylim_pad)
        ax.set_ylim(0, top if top else 1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.025,
                    formatter(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    bar(axes[0], [rr_promo, rr_org], "Item return rate", "Rate", lambda v: f"{v:.1%}", 0.35)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    axes[0].annotate(
        f"No significant promo lift\nratio {rr_promo / rr_org:.2f}x, p={p_greater:.4f}",
        xy=(0.5, 0.82), xycoords="axes fraction", ha="center", fontsize=9,
        color=PALETTE["neutral"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["neutral"], alpha=0.9),
    )

    bar(axes[1], [rep_promo, rep_org], "Customer repeat rate", "Rate", lambda v: f"{v:.1%}", 0.22)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    axes[1].annotate(
        f"Promo cohort: {(rep_promo - rep_org) * 100:.1f} pp lower",
        xy=(0.5, 0.82), xycoords="axes fraction", ha="center", fontsize=9,
        color=PALETTE["negative"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["negative"], alpha=0.9),
    )

    bar(axes[2], [gp_promo, gp_org], "Avg lifetime GP / customer", "VND",
        lambda v: f"{v:,.0f}", 0.30)
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}K"))
    axes[2].annotate(
        f"Promo cohort: {gp_gap:.0%} lower GP/customer",
        xy=(0.5, 0.82), xycoords="axes fraction", ha="center", fontsize=9,
        color=PALETTE["negative"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["negative"], alpha=0.9),
    )

    fig.suptitle("Cohort Quality - promo-acquired vs. organic customers",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Return rate is item-level; repeat rate and gross profit are customer-level.",
             ha="center", fontsize=9, color="grey")
    save_chart(fig, "chart4_cohort_quality.png")

    table = pd.DataFrame({
        "metric": ["item_return_rate", "customer_repeat_rate", "avg_lifetime_gp_per_customer"],
        "promo": [rr_promo, rep_promo, gp_promo],
        "organic": [rr_org, rep_org, gp_org],
        "difference": [rr_promo - rr_org, rep_promo - rep_org, gp_promo - gp_org],
    })
    table.to_csv(TABLES / "cohort_quality_table.csv", index=False)
    return {
        "rr_promo": rr_promo,
        "rr_org": rr_org,
        "p_greater": p_greater,
        "rep_promo": rep_promo,
        "rep_org": rep_org,
        "gp_promo": gp_promo,
        "gp_org": gp_org,
        "gp_gap": gp_gap,
    }


def chart5_margin_trajectory(oat, sales):
    oat_q = oat.copy()
    oat_q["quarter"] = oat_q["order_date"].dt.to_period("Q")
    quarterly = oat_q.groupby("quarter").agg(
        revenue=("net_revenue", "sum"),
        gross_profit=(GP_COL, "sum"),
    ).reset_index()
    quarterly["gross_margin_pct"] = quarterly["gross_profit"] / quarterly["revenue"] * 100
    quarterly = quarterly[quarterly["revenue"] > 0].copy()
    quarterly["date"] = quarterly["quarter"].dt.to_timestamp()

    x = np.arange(len(quarterly))
    y = quarterly["gross_margin_pct"].to_numpy()
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    residuals = y - trend(x)
    sd = np.std(residuals)

    quarterly["period"] = quarterly["quarter"].dt.year.map(covid_period_from_year)
    pre_mask = quarterly["period"].eq("Pre-COVID")
    post_mask = quarterly["period"].eq("Post-COVID")
    pre_coeffs = np.polyfit(x[pre_mask], y[pre_mask], 1) if pre_mask.sum() >= 2 else None
    post_coeffs = np.polyfit(x[post_mask], y[post_mask], 1) if post_mask.sum() >= 2 else None

    n_extra = 20
    xf = np.arange(len(quarterly), len(quarterly) + n_extra)
    yf = trend(xf)
    last_q = quarterly["quarter"].iloc[-1]
    future_q = [last_q + i + 1 for i in range(n_extra)]
    future_dates = pd.DatetimeIndex([q.to_timestamp() for q in future_q])
    future = pd.DataFrame({"quarter": future_q, "date": future_dates, "forecast": yf})
    future["lower_1sd"] = future["forecast"] - sd
    future["upper_1sd"] = future["forecast"] + sd

    def qval(year, q):
        row = future[(future["quarter"].dt.year == year) & (future["quarter"].dt.quarter == q)]
        if row.empty:
            return None
        return row.iloc[0]

    q2026 = qval(2026, 4)
    q2027 = qval(2027, 4)
    current = quarterly["gross_margin_pct"].iloc[-4:].mean()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01"),
               color="#9aa0a6", alpha=0.18, label="COVID shock window")
    ax.plot(quarterly["date"], quarterly["gross_margin_pct"], color=PALETTE["positive"],
            linewidth=2.4, label="Actual gross margin")
    ax.plot(quarterly["date"], trend(x), color=PALETTE["neutral"], linestyle=":",
            linewidth=1.3, label="Linear trend on actual period")
    if pre_coeffs is not None:
        pre_trend = np.poly1d(pre_coeffs)
        ax.plot(quarterly.loc[pre_mask, "date"], pre_trend(x[pre_mask]),
                color="#1d3557", linewidth=2.0, linestyle="-.",
                label=f"Pre-COVID trend ({pre_coeffs[0]:+.2f} pp/q)")
    if post_coeffs is not None:
        post_trend = np.poly1d(post_coeffs)
        ax.plot(quarterly.loc[post_mask, "date"], post_trend(x[post_mask]),
                color="#f77f00", linewidth=2.0, linestyle="-.",
                label=f"Post-COVID trend ({post_coeffs[0]:+.2f} pp/q)")
    ax.plot(future["date"], future["forecast"], color=PALETTE["negative"],
            linestyle="--", linewidth=2.3, label="Linear extrapolation")
    ax.fill_between(future["date"], future["lower_1sd"], future["upper_1sd"],
                    color=PALETTE["negative"], alpha=0.12, label=f"+/- 1 SD band ({sd:.2f} pp)")
    ax.axvline(pd.Timestamp("2023-01-01"), color="grey", linestyle="--", linewidth=1.3)
    ax.text(pd.Timestamp("2023-01-01"), max(y), "  Forecast start", fontsize=9, color="grey", va="top")
    ax.annotate(
        "COVID window: demand shock, defensive promo, and logistics disruption may distort margin",
        xy=(pd.Timestamp("2020-06-30"), np.nanpercentile(y, 88)),
        xytext=(pd.Timestamp("2016-01-01"), np.nanpercentile(y, 96)),
        fontsize=9, color="#5c677d",
        arrowprops=dict(arrowstyle="->", color="#5c677d"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9aa0a6", alpha=0.92),
    )
    for row in [q2026, q2027]:
        if row is not None:
            ax.annotate(f"{row['quarter']}: {row['forecast']:.1f}%\nrange {row['lower_1sd']:.1f}-{row['upper_1sd']:.1f}%",
                        xy=(row["date"], row["forecast"]), xytext=(26, 12),
                        textcoords="offset points", fontsize=10, fontweight="bold",
                        color=PALETTE["negative"],
                        arrowprops=dict(arrowstyle="->", color=PALETTE["negative"]),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["negative"], alpha=0.9))
    ax.set_title("Post-refund GP Margin Trajectory - actual trend and uncertainty band",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Gross margin")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct1))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right")
    fig.text(0.5, 0.01, "Source: oat.parquet. Margin uses net revenue and GP after refunds + shipping.",
             ha="center", fontsize=9, color="grey")
    save_chart(fig, "chart5_margin_trajectory.png")

    future.to_csv(TABLES / "margin_forecast_table.csv", index=False)
    return {
        "slope": coeffs[0],
        "sd": sd,
        "current": current,
        "pre_slope": pre_coeffs[0] if pre_coeffs is not None else np.nan,
        "post_slope": post_coeffs[0] if post_coeffs is not None else np.nan,
        "q2026": q2026,
        "q2027": q2027,
    }


def build_promo_cut_scenarios(oat, scope):
    promo = oat[oat["has_promo"]].groupby("promo_id").agg(
        promo_name=("promo_name", "first"),
        gp_contribution=(GP_COL, "sum"),
        net_revenue=("net_revenue", "sum"),
    ).reset_index()
    cut_ids = promo.loc[promo["gp_contribution"] < 0, "promo_id"]
    cut_mask = promo_id_mask(oat, cut_ids)

    total_revenue = oat["net_revenue"].sum()
    total_gp = oat[GP_COL].sum()
    cut_revenue = oat.loc[cut_mask, "net_revenue"].sum()
    cut_gp = oat.loc[cut_mask, GP_COL].sum()

    organic = oat[~oat["has_promo"]]
    organic_margin = organic[GP_COL].sum() / organic["net_revenue"].sum()

    rows = [{
        "scenario": "Current mix",
        "recapture_rate": np.nan,
        "revenue": total_revenue,
        "gp_contribution": total_gp,
        "gross_margin": total_gp / total_revenue,
        "revenue_impact_pct": 0.0,
        "gp_impact_pct": 0.0,
        "assumption": "All promotions continue",
    }]

    for rate, name in [
        (0.0, "Cut negative promos"),
        (0.3, "Cut + 30% organic recapture"),
        (0.5, "Cut + 50% organic recapture"),
    ]:
        recaptured_revenue = cut_revenue * rate
        recaptured_gp = recaptured_revenue * organic_margin
        revenue = total_revenue - cut_revenue + recaptured_revenue
        gp = total_gp - cut_gp + recaptured_gp
        rows.append({
            "scenario": name,
            "recapture_rate": rate,
            "revenue": revenue,
            "gp_contribution": gp,
            "gross_margin": gp / revenue,
            "revenue_impact_pct": (revenue / total_revenue - 1) * 100,
            "gp_impact_pct": (gp / total_gp - 1) * 100,
            "assumption": f"{rate:.0%} of removed promo revenue returns at organic margin",
        })

    scenarios = pd.DataFrame(rows)
    scenarios.insert(0, "scope", scope)
    scenarios["cut_promo_count"] = len(cut_ids)
    scenarios["organic_margin"] = organic_margin
    return scenarios, cut_ids, organic_margin, cut_revenue, cut_gp


def chart5_scenario_simulation(oat):
    scenarios, cut_ids, organic_margin, cut_revenue, cut_gp = build_promo_cut_scenarios(
        oat, "Full history 2012-2022"
    )
    scenarios.to_csv(TABLES / "promo_cut_scenario_table.csv", index=False)

    post_oat = oat[oat["order_date"].dt.year == 2022].copy()
    post_scenarios, post_cut_ids, _, _, _ = build_promo_cut_scenarios(
        post_oat, "Post-COVID normalized 2022"
    )
    scenario_compare = pd.concat([scenarios, post_scenarios], ignore_index=True)
    scenario_compare.to_csv(TABLES / "promo_cut_scenario_by_period_table.csv", index=False)

    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.set_facecolor("#fafafa")
    colors = [PALETTE["neutral"], PALETTE["negative"], PALETTE["warning"], PALETTE["positive"]]

    axes[0].bar(x, scenarios["revenue_impact_pct"], color=colors, alpha=0.86)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Revenue impact", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Change vs. current")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    axes[1].bar(x, scenarios["gp_impact_pct"], color=colors, alpha=0.86)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("GP contribution impact", fontsize=12, fontweight="bold")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    axes[2].bar(x, scenarios["gross_margin"] * 100, color=colors, alpha=0.86)
    axes[2].set_title("Gross margin after action", fontsize=12, fontweight="bold")
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    short_labels = ["Current\nmix", "Cut negative\npromos", "30% organic\nrecapture", "50% organic\nrecapture"]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=0, ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    for ax, col, fmt in [
        (axes[0], "revenue_impact_pct", "{:+.1f}%"),
        (axes[1], "gp_impact_pct", "{:+.1f}%"),
        (axes[2], "gross_margin", "{:.1%}"),
    ]:
        vals = scenarios[col]
        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        for i, v in enumerate(vals):
            label_val = v if col != "gross_margin" else v
            ax.text(i, v + span * 0.025 if v >= 0 else v + span * 0.04,
                    fmt.format(label_val), ha="center",
                    va="bottom" if v >= 0 else "bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Promo Cut Scenario Simulation - demand recapture sensitivity",
                 fontsize=15, fontweight="bold", y=1.04)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.text(0.5, 0.01,
             "Recaptured revenue is assumed to return at observed organic gross margin. CAC and payment costs are unavailable.",
             ha="center", fontsize=9, color="grey")
    save_chart(fig, "chart5_margin_scenario_simulation.png")

    return {
        "cut_promo_count": len(cut_ids),
        "post_cut_promo_count": len(post_cut_ids),
        "organic_margin": organic_margin,
        "cut_revenue": cut_revenue,
        "cut_gp": cut_gp,
        "zero_recapture": scenarios.loc[scenarios["scenario"] == "Cut negative promos"].iloc[0],
        "recapture_30": scenarios.loc[scenarios["scenario"] == "Cut + 30% organic recapture"].iloc[0],
        "recapture_50": scenarios.loc[scenarios["scenario"] == "Cut + 50% organic recapture"].iloc[0],
        "post_zero_recapture": post_scenarios.loc[post_scenarios["scenario"] == "Cut negative promos"].iloc[0],
    }


def chart6_stockout_overlap(oat, promotions, inventory, reviews):
    inv = inventory.copy()
    inv["year_month"] = inv["snapshot_date"].dt.to_period("M")
    cat_month = inv.groupby(["category", "year_month"]).agg(
        category_stockout_flag=("stockout_flag", "max"),
        stockout_product_share=("stockout_flag", "mean"),
        avg_stockout_days=("stockout_days", "mean"),
        max_stockout_days=("stockout_days", "max"),
        avg_fill_rate=("fill_rate", "mean"),
        min_fill_rate=("fill_rate", "min"),
        product_count=("product_id", "nunique"),
    ).reset_index()
    cat_month["snapshot_date"] = cat_month["year_month"].dt.to_timestamp("M")

    oat_month = oat.copy()
    oat_month["year_month"] = oat_month["order_date"].dt.to_period("M")
    monthly_gp = oat_month.groupby(["category", "year_month"]).agg(monthly_gp=(GP_COL, "sum")).reset_index()
    monthly_gp["daily_gp"] = monthly_gp["monthly_gp"] / 30.0

    events = []
    all_categories = sorted(cat_month["category"].dropna().unique().tolist())
    for _, promo in promotions.iterrows():
        cats = all_categories if pd.isna(promo["applicable_category"]) else [promo["applicable_category"]]
        start_m = promo["start_date"].to_period("M")
        end_m = promo["end_date"].to_period("M")
        rows = cat_month[
            (cat_month["category"].isin(cats))
            & (cat_month["year_month"] >= start_m)
            & (cat_month["year_month"] <= end_m)
            & (cat_month["category_stockout_flag"] == 1)
        ].copy()
        if rows.empty:
            continue
        rows["promo_id"] = promo["promo_id"]
        rows["promo_name"] = promo["promo_name"]
        rows["promo_start"] = promo["start_date"]
        rows["promo_end"] = promo["end_date"]
        events.append(rows)

    if events:
        event_df = pd.concat(events, ignore_index=True)
    else:
        event_df = pd.DataFrame()

    if not event_df.empty:
        def estimate(row):
            prev_months = [row["year_month"] - i for i in range(1, 4)]
            base = monthly_gp[(monthly_gp["category"] == row["category"]) & (monthly_gp["year_month"].isin(prev_months))]
            if base.empty:
                return 0.0
            return base["daily_gp"].mean() * row["avg_stockout_days"] * row["stockout_product_share"]

        event_df["estimated_lost_gp"] = event_df.apply(estimate, axis=1)
        event_df = event_df[[
            "promo_id", "promo_name", "category", "snapshot_date", "year_month",
            "stockout_product_share", "avg_stockout_days", "max_stockout_days",
            "avg_fill_rate", "min_fill_rate", "product_count", "estimated_lost_gp",
        ]].sort_values(["category", "snapshot_date", "promo_id"])
    else:
        event_df["estimated_lost_gp"] = []

    event_df.to_csv(TABLES / "promo_stockout_overlaps.csv", index=False)

    # Review check: category ratings before vs. 7-14 days after event month end.
    rating_delta = np.nan
    pre_mean = np.nan
    post_mean = np.nan
    try:
        reviews_cat = reviews.merge(
            oat[["order_id", "product_id", "category"]].drop_duplicates(),
            on=["order_id", "product_id"], how="left",
        )
        pre_ratings = []
        post_ratings = []
        for _, ev in event_df.iterrows():
            snap = ev["snapshot_date"]
            cat = ev["category"]
            pre = reviews_cat[
                (reviews_cat["category"] == cat)
                & (reviews_cat["review_date"] >= snap - pd.Timedelta(days=30))
                & (reviews_cat["review_date"] < snap)
            ]["rating"]
            post = reviews_cat[
                (reviews_cat["category"] == cat)
                & (reviews_cat["review_date"] >= snap + pd.Timedelta(days=7))
                & (reviews_cat["review_date"] <= snap + pd.Timedelta(days=14))
            ]["rating"]
            pre_ratings.extend(pre.tolist())
            post_ratings.extend(post.tolist())
        if pre_ratings and post_ratings:
            pre_mean = float(np.mean(pre_ratings))
            post_mean = float(np.mean(post_ratings))
            rating_delta = post_mean - pre_mean
    except Exception as exc:
        print(f"Review rating check skipped: {exc}")

    promo_gp = oat[oat["has_promo"]].groupby("promo_id").agg(
        promo_gp_contribution=(GP_COL, "sum"),
        promo_net_revenue=("net_revenue", "sum"),
    ).reset_index()

    if not event_df.empty:
        risk = event_df.groupby(["promo_id", "promo_name", "category"]).agg(
            overlap_months=("year_month", "nunique"),
            avg_sku_stockout_share=("stockout_product_share", "mean"),
            max_sku_stockout_share=("stockout_product_share", "max"),
            avg_stockout_days=("avg_stockout_days", "mean"),
            max_stockout_days=("max_stockout_days", "max"),
            avg_fill_rate=("avg_fill_rate", "mean"),
            min_fill_rate=("min_fill_rate", "min"),
            estimated_lost_gp=("estimated_lost_gp", "sum"),
        ).reset_index()
        risk = risk.merge(promo_gp, on="promo_id", how="left")
        risk["negative_gp_exposure"] = (-risk["promo_gp_contribution"]).clip(lower=0)
        risk["inventory_risk_score"] = (
            risk["avg_sku_stockout_share"]
            * risk["avg_stockout_days"]
            * risk["promo_net_revenue"]
        )
        risk = risk.sort_values("inventory_risk_score", ascending=False)
    else:
        risk = pd.DataFrame()

    risk.to_csv(TABLES / "promo_inventory_risk_table.csv", index=False)

    if not risk.empty:
        plot_df = risk.head(15).copy()
        plot_df["plot_label"] = plot_df["promo_name"] + " / " + plot_df["category"]
        plot_df = plot_df.sort_values("inventory_risk_score")
        norm = mcolors.Normalize(
            vmin=plot_df["avg_sku_stockout_share"].min(),
            vmax=plot_df["avg_sku_stockout_share"].max(),
        )
        colors = cm.Reds(norm(plot_df["avg_sku_stockout_share"]))
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.barh(plot_df["plot_label"], plot_df["inventory_risk_score"], color=colors, alpha=0.9)
        max_score = plot_df["inventory_risk_score"].max()
        ax.set_xlim(0, max_score * 1.52)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(money_m))
        ax.set_title("Promo Inventory Risk Ranking - SKU stockout exposure during promo windows",
                     fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Inventory risk score (stockout share x stockout days x promo revenue exposure)")
        ax.set_ylabel("")
        for _, row in plot_df.iterrows():
            ax.text(
                row["inventory_risk_score"],
                row["plot_label"],
                f" SKU stockout {row['avg_sku_stockout_share']:.0%} | fill {row['avg_fill_rate']:.0%}",
                va="center", ha="left", fontsize=8, color="#333333",
            )
        sm = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Avg SKU stockout share")
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        fig.text(0.5, 0.01,
                 "Source: promotions.csv, inventory.csv, oat.parquet. Chart shows top 15 promo-category windows by SKU-level inventory risk.",
                 ha="center", fontsize=9, color="grey")
        save_chart(fig, "chart6_promo_inventory_risk.png")

    return {
        "event_count": len(event_df),
        "promo_count": event_df["promo_id"].nunique() if not event_df.empty else 0,
        "categories": sorted(event_df["category"].unique().tolist()) if not event_df.empty else [],
        "avg_stockout_share": event_df["stockout_product_share"].mean() if not event_df.empty else 0,
        "avg_fill_rate": event_df["avg_fill_rate"].mean() if not event_df.empty else 0,
        "total_lost_gp": event_df["estimated_lost_gp"].sum() if not event_df.empty else 0,
        "risk_top_promo": risk.iloc[0]["promo_name"] if not risk.empty else "",
        "risk_top_score": risk.iloc[0]["inventory_risk_score"] if not risk.empty else 0,
        "risk_table_rows": len(risk),
        "rating_delta": rating_delta,
        "pre_rating": pre_mean,
        "post_rating": post_mean,
    }


def chart7_triage(oat, overlap_events):
    promo = oat[oat["has_promo"]].groupby("promo_id").agg(
        promo_name=("promo_name", "first"),
        total_discount_cost=("discount_amount", "sum"),
        total_net_contribution=(GP_COL, "sum"),
        order_count=("order_id", "nunique"),
        return_rate=("is_returned", "mean"),
        net_revenue=("net_revenue", "sum"),
    ).reset_index()
    overlap_ids = set(overlap_events["promo_id"].unique()) if not overlap_events.empty else set()
    promo["stockout_overlap"] = np.where(promo["promo_id"].isin(overlap_ids), "Yes", "No")

    def verdict(row):
        if row["total_net_contribution"] < 0:
            return "CUT"
        if row["stockout_overlap"] == "Yes" or row["return_rate"] > 0.15:
            return "RESCHEDULE"
        return "KEEP"

    promo["verdict"] = promo.apply(verdict, axis=1)
    promo["estimated_annual_impact_vnd"] = np.where(
        promo["verdict"] == "CUT", -promo["total_net_contribution"], promo["total_net_contribution"]
    )
    promo = promo.sort_values("total_net_contribution", ascending=False)
    promo.to_csv(TABLES / "promo_triage_table.csv", index=False)

    total_revenue = oat["net_revenue"].sum()
    total_profit = oat[GP_COL].sum()
    cut_ids = promo.loc[promo["verdict"] == "CUT", "promo_id"]
    cut_mask = promo_id_mask(oat, cut_ids)
    rev_impact_pct = oat.loc[cut_mask, "net_revenue"].sum() / total_revenue * 100
    profit_impact_pct = -oat.loc[cut_mask, GP_COL].sum() / total_profit * 100

    plot_df = promo.sort_values("total_net_contribution")
    colors = np.where(plot_df["verdict"].eq("CUT"), PALETTE["negative"],
                      np.where(plot_df["verdict"].eq("RESCHEDULE"), PALETTE["warning"], PALETTE["positive"]))
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.barh(plot_df["promo_name"], plot_df["total_net_contribution"], color=colors, alpha=0.86)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(money_m))
    ax.set_title("Promo Triage - action verdict by GP contribution",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Gross-profit contribution (VND)")
    ax.set_ylabel("")
    for _, r in plot_df.head(5).iterrows():
        ax.text(r["total_net_contribution"], r["promo_name"], " CUT", va="center",
                ha="right", color="white", fontsize=8, fontweight="bold")
    for _, r in plot_df.tail(5).iterrows():
        if r["verdict"] == "KEEP" and r["total_net_contribution"] > 0:
            tag = " KEEP"
            tag_color = PALETTE["positive"]
        elif r["verdict"] == "RESCHEDULE":
            tag = " RESCHEDULE"
            tag_color = PALETTE["warning"]
        else:
            tag = " LEAST NEGATIVE"
            tag_color = "#5c677d"
        ax.text(r["total_net_contribution"], r["promo_name"], tag, va="center",
                ha="left", color=tag_color, fontsize=8, fontweight="bold")
    ax.annotate(
        f"Cut {len(cut_ids)} negative promos:\nRevenue -{rev_impact_pct:.1f}% | Gross profit +{profit_impact_pct:.1f}%",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["negative"], alpha=0.92),
        fontsize=11, fontweight="bold",
    )
    fig.text(0.5, 0.01, "Source: oat.parquet and promo_stockout_overlaps.csv",
             ha="center", fontsize=9, color="grey")
    save_chart(fig, "chart7_promo_triage.png")

    return {
        "keep_count": int((promo["verdict"] == "KEEP").sum()),
        "cut_count": int((promo["verdict"] == "CUT").sum()),
        "reschedule_count": int((promo["verdict"] == "RESCHEDULE").sum()),
        "keep_net": promo.loc[promo["verdict"] == "KEEP", "total_net_contribution"].sum(),
        "cut_net": promo.loc[promo["verdict"] == "CUT", "total_net_contribution"].sum(),
        "rev_impact_pct": rev_impact_pct,
        "profit_impact_pct": profit_impact_pct,
    }


def write_summary(metrics):
    q2026 = metrics["margin"]["q2026"]
    q2027 = metrics["margin"]["q2027"]
    summary = f"""# Corrected Part 2 Results

## Act 1 - Revenue Anatomy
- Discount leakage: {metrics['act1']['total_discount']:,.0f} VND.
- Return leakage: {metrics['act1']['total_returns']:,.0f} VND.
- Combined leakage: {metrics['act1']['leakage_pct']:.1f}% of gross revenue.
- Gross margin moved from {metrics['act1']['margin_first']:.1f}% in {metrics['act1']['first_year']} to {metrics['act1']['margin_last']:.1f}% in {metrics['act1']['last_year']}.

## Act 2 - Promo Economics and Cohort Quality
- Net-negative promotions: {(metrics['promo']['total_net_contribution'] < 0).mean() * 100:.1f}%.
- Matched organic-margin proxy: promo rows underperform matched non-promo category-quarter margin by {metrics['counterfactual']['gp_gap_vs_matched_organic_margin'].sum():,.0f} VND. This is a margin benchmark, not a causal uplift estimate.
- Item return rate is not meaningfully higher for promo-acquired customers: {metrics['cohort']['rr_promo']:.1%} vs {metrics['cohort']['rr_org']:.1%}, Mann-Whitney one-sided p={metrics['cohort']['p_greater']:.4f}.
- Customer repeat rate is lower for promo-acquired customers: {metrics['cohort']['rep_promo']:.1%} vs {metrics['cohort']['rep_org']:.1%}.
- Average lifetime gross profit per customer is {metrics['cohort']['gp_promo']:,.0f} VND for promo vs {metrics['cohort']['gp_org']:,.0f} VND for organic, a {metrics['cohort']['gp_gap']:.0%} gap.

## Act 3 - Margin Trajectory
- Linear trend slope: {metrics['margin']['slope']:+.3f} percentage points per quarter.
- Last-four-quarter gross margin: {metrics['margin']['current']:.1f}%.
- 2026 Q4 forecast: {q2026['forecast']:.1f}% with +/-1 SD range {q2026['lower_1sd']:.1f}% to {q2026['upper_1sd']:.1f}%.
- 2027 Q4 forecast: {q2027['forecast']:.1f}% with +/-1 SD range {q2027['lower_1sd']:.1f}% to {q2027['upper_1sd']:.1f}%.
- Scenario simulation: cutting {metrics['scenario']['cut_promo_count']} negative-GP promotions gives GP contribution {metrics['scenario']['zero_recapture']['gp_impact_pct']:+.1f}% under 0% demand recapture.
- With 30% organic recapture, revenue impact is {metrics['scenario']['recapture_30']['revenue_impact_pct']:+.1f}% and GP contribution impact is {metrics['scenario']['recapture_30']['gp_impact_pct']:+.1f}%.

## Act 4 - Promo x Stockout Overlap
- Event unit: promo x category x month.
- Overlap events: {metrics['stockout']['event_count']} across {metrics['stockout']['promo_count']} promotions and {', '.join(metrics['stockout']['categories'])}.
- Average product stockout share during overlap months: {metrics['stockout']['avg_stockout_share']:.1%}.
- Average fill rate during overlap months: {metrics['stockout']['avg_fill_rate']:.1%}.
- Estimated lost gross profit from these event-level overlaps: {metrics['stockout']['total_lost_gp']:,.0f} VND.
- Highest inventory-risk promo: {metrics['stockout']['risk_top_promo']} with risk score {metrics['stockout']['risk_top_score']:,.0f}.
- Review rating delta after overlap windows: {metrics['stockout']['rating_delta']:+.2f} stars.

## Act 5 - Promo Triage
- KEEP: {metrics['triage']['keep_count']} promotions, total GP contribution {metrics['triage']['keep_net']:,.0f} VND.
- CUT: {metrics['triage']['cut_count']} promotions, total GP contribution {metrics['triage']['cut_net']:,.0f} VND.
- RESCHEDULE: {metrics['triage']['reschedule_count']} promotions.
- Cutting negative-GP promotions implies revenue -{metrics['triage']['rev_impact_pct']:.1f}% and GP contribution +{metrics['triage']['profit_impact_pct']:.1f}% under zero demand recapture.

## Method Notes
- Contribution metrics use all order statuses; see `order_status_profit_audit.csv` for sensitivity and status mix transparency.
- `Promo GP Contribution Map` is contribution accounting, not true ROI. True incremental ROI would require a counterfactual demand model or experiment.
"""
    path = OUT / "part2_corrected_results.md"
    path.write_text(summary, encoding="utf-8")
    print(f"Saved {path.relative_to(ROOT)}")


def main():
    oat, sales, promotions, inventory, reviews = load_inputs()
    metrics = {}
    metrics["act1"] = chart1_revenue_anatomy(oat)
    metrics["status"] = order_status_profit_audit(oat)
    metrics["category"] = chart2_double_loss(oat)
    metrics["promo"] = chart3_promo_roi(oat)
    metrics["counterfactual"] = promo_counterfactual_proxy(oat)
    metrics["covid_periods"] = covid_period_diagnostics(oat, promotions, inventory)
    metrics["cohort"] = chart4_cohort_quality(oat)
    metrics["margin"] = chart5_margin_trajectory(oat, sales)
    metrics["scenario"] = chart5_scenario_simulation(oat)
    metrics["stockout"] = chart6_stockout_overlap(oat, promotions, inventory, reviews)
    overlap_events = pd.read_csv(TABLES / "promo_stockout_overlaps.csv")
    metrics["triage"] = chart7_triage(oat, overlap_events)
    write_summary(metrics)


if __name__ == "__main__":
    main()
