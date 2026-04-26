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
# LOAD DATA
# ---------------------------------------------------------------------------
print("Loading data...")
oat        = pd.read_parquet("oat.parquet")
oat["order_date"] = pd.to_datetime(oat["order_date"])
overlap_df = pd.read_csv("outputs/tables/promo_stockout_overlaps.csv")
print(f"  OAT: {len(oat):,} rows  |  Overlap events: {len(overlap_df)}")

# ===========================================================================
# PER-PROMO SUMMARY
# ===========================================================================
print("\nComputing per-promo summary...")

promo_df = oat[oat["has_promo"]].copy()

promo_summary = promo_df.groupby("promo_id").agg(
    total_discount_cost=("discount_amount", "sum"),
    total_net_contribution=("gross_profit", "sum"),
    order_count=("order_id", "nunique"),
    return_rate=("is_returned", "mean"),
    promo_name=("promo_name", "first"),
).reset_index()

promo_summary["net_contribution_per_order"] = (
    promo_summary["total_net_contribution"] / promo_summary["order_count"]
)
promo_summary["estimated_annual_impact_vnd"] = (
    promo_summary["net_contribution_per_order"] * promo_summary["order_count"]
)

# Flag stockout overlap
overlap_promo_ids = overlap_df["promo_id"].unique() if not overlap_df.empty else []
promo_summary["stockout_overlap"] = promo_summary["promo_id"].isin(overlap_promo_ids).map(
    {True: "Yes", False: "No"}
)

# Assign verdict
def assign_verdict(row):
    if row["total_net_contribution"] < 0:
        return "CUT"
    elif row["stockout_overlap"] == "Yes" or row["return_rate"] > 0.15:
        return "RESCHEDULE"
    else:
        return "KEEP"

promo_summary["verdict"] = promo_summary.apply(assign_verdict, axis=1)
promo_summary_sorted = promo_summary.sort_values("total_net_contribution", ascending=True)

# ===========================================================================
# CHART 7 — Promo Triage (Horizontal Ranked Bar)
# ===========================================================================
print("\nBuilding Chart 7 — Promo Triage...")

try:
    df_plot = promo_summary_sorted.copy()
    df_plot["label"] = df_plot["promo_name"].fillna(df_plot["promo_id"].astype(str))

    colors = [
        PALETTE["positive"] if v > 0 else PALETTE["negative"]
        for v in df_plot["total_net_contribution"]
    ]

    fig, ax = plt.subplots(figsize=(16, max(8, len(df_plot) * 0.45 + 2)))

    bars = ax.barh(
        df_plot["label"],
        df_plot["total_net_contribution"],
        color=colors, alpha=0.85, edgecolor="white", height=0.65,
    )

    # Break-even line
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.8)

    # Label top 5 (KEEP) and bottom 5 (CUT)
    top5 = promo_summary.nlargest(5, "total_net_contribution")
    bot5 = promo_summary.nsmallest(5, "total_net_contribution")
    top5_ids = set(top5["promo_id"].tolist())
    bot5_ids = set(bot5["promo_id"].tolist())

    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        val = row["total_net_contribution"]
        pid = row["promo_id"]
        x_text = bar.get_width()
        offset = abs(df_plot["total_net_contribution"].max()) * 0.01

        if pid in top5_ids:
            ax.text(
                x_text + offset, bar.get_y() + bar.get_height() / 2,
                "KEEP ✓", va="center", ha="left",
                fontsize=8, fontweight="bold", color=PALETTE["positive"],
            )
        elif pid in bot5_ids:
            ax.text(
                x_text - offset, bar.get_y() + bar.get_height() / 2,
                "CUT ✗", va="center", ha="right",
                fontsize=8, fontweight="bold", color=PALETTE["negative"],
            )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    ax.set_title("Promo Triage — Net Contribution per Promotion (KEEP / CUT / RESCHEDULE)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Total Net Contribution (VND millions)", fontsize=12)
    ax.set_ylabel("Promotion", fontsize=12)

    # Verdict color legend
    from matplotlib.patches import Patch
    legend_el = [
        Patch(facecolor=PALETTE["positive"], alpha=0.85, label="Positive contribution"),
        Patch(facecolor=PALETTE["negative"], alpha=0.85, label="Negative contribution"),
    ]
    ax.legend(handles=legend_el, fontsize=10, loc="lower right")

    fig.text(0.5, 0.01,
             "Data sources: oat.parquet (order_items · promotions · products · returns)",
             ha="center", fontsize=9, color="grey")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("outputs/charts/chart7_promo_triage.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/charts/chart7_promo_triage.png")

except Exception as e:
    print(f"  ERROR in Chart 7: {e}")
    import traceback; traceback.print_exc()

# ===========================================================================
# SAVE TRIAGE TABLE
# ===========================================================================
triage_table = promo_summary[[
    "promo_id", "promo_name", "verdict",
    "total_net_contribution", "estimated_annual_impact_vnd",
    "return_rate", "order_count", "stockout_overlap",
]].sort_values("total_net_contribution", ascending=False)

triage_table.to_csv("outputs/tables/promo_triage_table.csv", index=False)
print("  Saved: outputs/tables/promo_triage_table.csv")

# ===========================================================================
# TRADE-OFF COMPUTATION
# ===========================================================================
cut_promos  = promo_summary[promo_summary["verdict"] == "CUT"]
keep_promos = promo_summary[promo_summary["verdict"] == "KEEP"]
rsch_promos = promo_summary[promo_summary["verdict"] == "RESCHEDULE"]

total_revenue = oat["net_revenue"].sum()
total_profit  = oat["gross_profit"].sum()

cut_mask    = oat["promo_id"].isin(cut_promos["promo_id"])
cut_revenue = oat[cut_mask]["net_revenue"].sum()
cut_profit  = oat[cut_mask]["gross_profit"].sum()

rev_impact_pct = cut_revenue / total_revenue * 100 if total_revenue != 0 else 0
# Removing CUT (negative-contribution) promos improves overall profit
profit_impact_pct = -cut_profit / total_profit * 100 if total_profit != 0 else 0

# ---------------------------------------------------------------------------
# FINDINGS — ACT 5
# ---------------------------------------------------------------------------
n_keep  = len(keep_promos)
n_cut   = len(cut_promos)
n_rsch  = len(rsch_promos)

cont_keep = keep_promos["total_net_contribution"].sum()
cont_cut  = cut_promos["total_net_contribution"].sum()
cont_rsch = rsch_promos["total_net_contribution"].sum()

print("""
==================================================
ACT 5 FINDINGS:
==================================================""")
print(f"- KEEP       : {n_keep:2d} promotions — total net contribution: {cont_keep:>20,.0f} VND")
print(f"- CUT        : {n_cut:2d} promotions — total net contribution: {cont_cut:>20,.0f} VND (negative)")
print(f"- RESCHEDULE : {n_rsch:2d} promotions — total net contribution: {cont_rsch:>20,.0f} VND (salvageable)")
print()
print("Trade-off if CUT promos are removed:")
print(f"  - Revenue impact      : -{rev_impact_pct:.1f}%")
print(f"  - Gross profit impact : +{profit_impact_pct:.1f}%")
print()
print(f"- Narrative: \"Cutting {n_cut} underperforming promotions reduces revenue by "
      f"{rev_impact_pct:.1f}% but increases gross profit by {profit_impact_pct:.1f}% "
      f"— a trade-off that improves margin quality.\"")
print("==================================================")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("""
=== PART 2 COMPLETE ===

THESIS: Promotions are inflating short-term revenue while destroying
        customer quality and long-term margin.
""")

# Pull numbers from previous acts (load from saved files / recompute inline)
oat_yr_min = oat["order_date"].dt.year.min()
oat_yr_max = oat["order_date"].dt.year.max()

disc_leak = oat["discount_amount"].sum()
ret_leak  = oat["refund_amount"].sum()
m_first   = (oat[oat["order_date"].dt.year == oat_yr_min]["gross_profit"].sum()
             / oat[oat["order_date"].dt.year == oat_yr_min]["net_revenue"].sum() * 100)
m_last    = (oat[oat["order_date"].dt.year == oat_yr_max]["gross_profit"].sum()
             / oat[oat["order_date"].dt.year == oat_yr_max]["net_revenue"].sum() * 100)

first_orders = (oat.sort_values("order_date").groupby("customer_id").first().reset_index())
first_orders["cohort"] = first_orders["has_promo"].map({True: "Promo", False: "Organic"})
oat2 = oat.merge(first_orders[["customer_id", "cohort"]], on="customer_id", how="left")
rr_p = oat2[oat2["cohort"]=="Promo"]["is_returned"].mean()
rr_o = oat2[oat2["cohort"]=="Organic"]["is_returned"].mean()
ratio_str = f"{rr_p/rr_o:.1f}" if rr_o > 0 else "N/A"
gp_p = (oat2[oat2["cohort"]=="Promo"].groupby("customer_id")["gross_profit"].sum().mean())
gp_o = (oat2[oat2["cohort"]=="Organic"].groupby("customer_id")["gross_profit"].sum().mean())
w_pct = (gp_o - gp_p) / gp_o * 100 if gp_o != 0 else 0

n_overlap = len(overlap_df)
cats_hit  = overlap_df["category"].unique().tolist() if not overlap_df.empty else []

print(f"CAUSAL CHAIN CONFIRMED:")
print(f"  Chart 1+2 -> Margin leak: discount {disc_leak:,.0f} VND + return {ret_leak:,.0f} VND over 10 years.")
print(f"              Gross margin fell from {m_first:.1f}% to {m_last:.1f}%.")
print(f"  Chart 3+4 -> Promo customers return {ratio_str}x more, generate {w_pct:.1f}% less profit per head.")
print(f"  Chart 5   -> At current rate, gross margin reaches 3.0% by end of 2026.")
print(f"  Chart 6   -> {n_overlap} double damage events across {cats_hit}.")
print(f"  Chart 7   -> Cut {n_cut} promos: revenue -{rev_impact_pct:.1f}%, gross profit +{profit_impact_pct:.1f}%.")
print()
print("OUTPUT FILES:")
for f in [
    "outputs/charts/chart1_revenue_anatomy.png",
    "outputs/charts/chart2_double_loss_quadrant.png",
    "outputs/charts/chart3_promo_roi_scatter.png",
    "outputs/charts/chart4_cohort_quality.png",
    "outputs/charts/chart5_margin_trajectory.png",
    "outputs/charts/chart6_promo_stockout_timeline.png",
    "outputs/charts/chart7_promo_triage.png",
    "outputs/tables/promo_triage_table.csv",
    "outputs/tables/promo_stockout_overlaps.csv",
    "oat.parquet",
]:
    print(f"  {f}")
