import numpy as np
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu

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
# CHART 3 — Promo ROI Scatter
# ===========================================================================
print("\nBuilding Chart 3 — Promo ROI Scatter...")

try:
    promo_df = oat[oat["has_promo"]].copy()

    promo_grp = promo_df.groupby("promo_id").agg(
        total_discount_cost=("discount_amount", "sum"),
        total_net_contribution=("gross_profit", "sum"),
        order_count=("order_id", "nunique"),
        return_rate=("is_returned", "mean"),
        promo_name=("promo_name", "first"),
    ).reset_index()

    # Normalize marker size to 20–400
    oc = promo_grp["order_count"]
    promo_grp["marker_size"] = (
        (oc - oc.min()) / (oc.max() - oc.min() + 1e-9) * (400 - 20) + 20
    )

    # Color by return_rate via continuous colormap
    norm = mcolors.Normalize(
        vmin=promo_grp["return_rate"].min(),
        vmax=promo_grp["return_rate"].max(),
    )
    cmap = cm.RdYlGn_r  # green=low return, red=high return

    fig, ax = plt.subplots(figsize=(14, 9))

    sc = ax.scatter(
        promo_grp["total_discount_cost"],
        promo_grp["total_net_contribution"],
        s=promo_grp["marker_size"],
        c=promo_grp["return_rate"],
        cmap=cmap, norm=norm,
        alpha=0.80, edgecolors="white", linewidth=0.8,
    )

    # Break-even line
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7, label="Break-even (Y=0)")

    # Annotate top 3 and bottom 3
    top3 = promo_grp.nlargest(3, "total_net_contribution")
    bot3 = promo_grp.nsmallest(3, "total_net_contribution")

    for _, r in top3.iterrows():
        ax.annotate(
            f"KEEP\n{r['promo_name'] if pd.notna(r['promo_name']) else r['promo_id']}",
            (r["total_discount_cost"], r["total_net_contribution"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8, fontweight="bold", color=PALETTE["positive"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["positive"], alpha=0.85),
        )

    for _, r in bot3.iterrows():
        ax.annotate(
            f"CUT\n{r['promo_name'] if pd.notna(r['promo_name']) else r['promo_id']}",
            (r["total_discount_cost"], r["total_net_contribution"]),
            textcoords="offset points", xytext=(8, -20),
            fontsize=8, fontweight="bold", color=PALETTE["negative"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["negative"], alpha=0.85),
        )

    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Return Rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    ax.set_title("Promo ROI Scatter — Discount Cost vs. Net Contribution per Promotion",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Total Discount Cost (VND millions)", fontsize=12)
    ax.set_ylabel("Total Net Contribution / Gross Profit (VND millions)", fontsize=12)
    ax.legend(fontsize=10)
    fig.text(0.5, 0.01,
             "Data sources: oat.parquet (order_items · promotions · products · returns)",
             ha="center", fontsize=9, color="grey")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("outputs/charts/chart3_promo_roi_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/charts/chart3_promo_roi_scatter.png")

    pct_neg = (promo_grp["total_net_contribution"] < 0).mean() * 100
    worst   = promo_grp.loc[promo_grp["total_net_contribution"].idxmin()]

except Exception as e:
    print(f"  ERROR in Chart 3: {e}")
    import traceback; traceback.print_exc()
    promo_grp = pd.DataFrame()
    pct_neg = 0
    worst = pd.Series({"promo_name": "N/A", "total_net_contribution": 0})

# ===========================================================================
# CHART 4 — Cohort Quality Comparison
# ===========================================================================
print("\nBuilding Chart 4 — Cohort Quality Comparison...")

try:
    # Assign cohort based on FIRST order per customer
    first_orders = (
        oat.sort_values("order_date")
           .groupby("customer_id")
           .first()
           .reset_index()
    )
    first_orders["cohort"] = first_orders["has_promo"].map({True: "Promo", False: "Organic"})

    oat_cohort = oat.merge(first_orders[["customer_id", "cohort"]], on="customer_id", how="left")

    cohort_stats = oat_cohort.groupby("cohort").agg(
        return_rate=("is_returned", "mean"),
        repeat_rate=("is_repeat", "mean"),
    )

    gp_per_cust = (
        oat_cohort.groupby(["cohort", "customer_id"])["gross_profit"]
                  .sum()
                  .reset_index()
    )
    avg_gp = gp_per_cust.groupby("cohort")["gross_profit"].mean()

    # Mann-Whitney U test on return rates
    promo_r   = oat_cohort[oat_cohort["cohort"] == "Promo"]["is_returned"].astype(int)
    organic_r = oat_cohort[oat_cohort["cohort"] == "Organic"]["is_returned"].astype(int)
    stat, p   = mannwhitneyu(promo_r, organic_r, alternative="greater")

    rr_promo   = cohort_stats.loc["Promo",   "return_rate"]
    rr_organic = cohort_stats.loc["Organic", "return_rate"]
    rep_promo  = cohort_stats.loc["Promo",   "repeat_rate"]
    rep_org    = cohort_stats.loc["Organic", "repeat_rate"]
    gp_promo   = avg_gp.get("Promo",   0)
    gp_org     = avg_gp.get("Organic", 0)
    ratio      = rr_promo / rr_organic if rr_organic > 0 else float("inf")

    # Build grouped bar chart — 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    def _bar(ax, vals, labels, colors, title, ylabel, fmt="{:.1%}"):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    fmt.format(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    colors2 = [PALETTE["promo"], PALETTE["organic"]]

    _bar(axes[0],
         [rr_promo, rr_organic],
         ["Promo", "Organic"], colors2,
         "Return Rate", "Rate", fmt="{:.1%}")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    _bar(axes[1],
         [rep_promo, rep_org],
         ["Promo", "Organic"], colors2,
         "Repeat Rate", "Rate", fmt="{:.1%}")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    _bar(axes[2],
         [gp_promo, gp_org],
         ["Promo", "Organic"], colors2,
         "Avg Gross Profit per Customer", "VND", fmt="{:,.0f}")
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))

    # Statistical annotation
    axes[0].annotate(
        f"Promo customers return {ratio:.1f}x more often\n(p={p:.4f})",
        xy=(0.5, 0.88), xycoords="axes fraction",
        ha="center", fontsize=9,
        color=PALETTE["negative"],
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["negative"], alpha=0.85),
    )

    fig.suptitle("Cohort Quality — Promo-Acquired vs. Organic Customers",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02,
             "Data sources: oat.parquet (orders · order_items · customers · returns)",
             ha="center", fontsize=9, color="grey")
    plt.tight_layout()
    plt.savefig("outputs/charts/chart4_cohort_quality.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/charts/chart4_cohort_quality.png")

except Exception as e:
    print(f"  ERROR in Chart 4: {e}")
    import traceback; traceback.print_exc()
    rr_promo = rr_organic = rep_promo = rep_org = gp_promo = gp_org = ratio = p = 0

# ---------------------------------------------------------------------------
# FINDINGS — ACT 2
# ---------------------------------------------------------------------------
print("""
==================================================
ACT 2 FINDINGS:
==================================================""")
print(f"- % promotions with negative net contribution : {pct_neg:.1f}%")
wname = worst.get('promo_name', 'N/A') if not isinstance(worst, float) else 'N/A'
wval  = worst.get('total_net_contribution', 0) if not isinstance(worst, float) else 0
print(f"- Worst promo : {wname} — net contribution = {wval:,.0f} VND")
print(f"- Return rate : promo {rr_promo:.1%} vs organic {rr_organic:.1%} "
      f"— ratio {ratio:.1f}x (p={p:.4f})")
print(f"- Repeat rate : promo {rep_promo:.1%} vs organic {rep_org:.1%}")
print(f"- Avg gross profit per customer : promo {gp_promo:,.0f} vs organic {gp_org:,.0f} VND")
print(f"- Narrative: \"Promo customers return {ratio:.1f}x more, repeat less, and generate "
      f"less profit — this is debt disguised as growth.\"")
print("==================================================")
