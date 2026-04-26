import numpy as np
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
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

promotions = pd.read_csv("data/promotions.csv", parse_dates=["start_date", "end_date"])
inventory  = pd.read_csv("data/inventory.csv",  parse_dates=["snapshot_date"])
reviews    = pd.read_csv("data/reviews.csv",    parse_dates=["review_date"])
print(f"  OAT: {len(oat):,} rows")
print(f"  Promotions: {len(promotions)} rows, Inventory: {len(inventory):,} rows")

# ===========================================================================
# FIND PROMO × STOCKOUT OVERLAPS
# ===========================================================================
print("\nFinding promo × stockout overlaps...")

promo_cats = promotions.dropna(subset=["applicable_category"])["applicable_category"].unique()
print(f"  Categories in promotions: {list(promo_cats)}")

overlaps = []
for _, promo in promotions.iterrows():
    cat = promo["applicable_category"]
    if pd.isna(cat):
        continue
    inv_match = inventory[
        (inventory["category"] == cat) &
        (inventory["snapshot_date"] >= promo["start_date"]) &
        (inventory["snapshot_date"] <= promo["end_date"]) &
        (inventory["stockout_flag"] == 1)
    ].copy()
    if len(inv_match) > 0:
        inv_match["promo_id"]   = promo["promo_id"]
        inv_match["promo_name"] = promo["promo_name"]
        overlaps.append(inv_match)

if overlaps:
    overlap_df = pd.concat(overlaps, ignore_index=True)
    print(f"  Found {len(overlap_df)} overlap events across "
          f"{overlap_df['promo_id'].nunique()} promotions")
else:
    overlap_df = pd.DataFrame()
    print("  No overlaps found")

# ===========================================================================
# ESTIMATE LOST REVENUE
# ===========================================================================
print("\nEstimating lost revenue...")

if not overlap_df.empty:
    def estimate_lost_revenue(row):
        mask = (
            (oat["category"] == row["category"]) &
            (oat["order_date"] >= row["snapshot_date"] - pd.Timedelta(days=30)) &
            (oat["order_date"] <= row["snapshot_date"])
        )
        period_data = oat[mask]
        if len(period_data) == 0:
            return 0.0
        avg_daily = period_data["gross_profit"].sum() / 30.0
        return avg_daily * row["stockout_days"]

    overlap_df["estimated_lost_revenue"] = overlap_df.apply(
        estimate_lost_revenue, axis=1
    )
    total_lost = overlap_df["estimated_lost_revenue"].sum()
    avg_fill   = overlap_df["fill_rate"].mean() if "fill_rate" in overlap_df.columns else float("nan")
else:
    total_lost = 0.0
    avg_fill   = float("nan")

# ===========================================================================
# REVIEW RATING CHECK (post-event window: +7 to +14 days)
# ===========================================================================
rating_delta = float("nan")
if not overlap_df.empty and len(reviews) > 0:
    try:
        # Merge reviews with OAT to get category
        reviews_cat = reviews.merge(
            oat[["order_id", "product_id", "category"]].drop_duplicates(),
            on=["order_id", "product_id"], how="left"
        )

        post_ratings, pre_ratings = [], []
        for _, ev in overlap_df.iterrows():
            snap = ev["snapshot_date"]
            cat  = ev["category"]
            pre  = reviews_cat[
                (reviews_cat["category"] == cat) &
                (reviews_cat["review_date"] >= snap - pd.Timedelta(days=30)) &
                (reviews_cat["review_date"] <  snap)
            ]["rating"]
            post = reviews_cat[
                (reviews_cat["category"] == cat) &
                (reviews_cat["review_date"] >= snap + pd.Timedelta(days=7)) &
                (reviews_cat["review_date"] <= snap + pd.Timedelta(days=14))
            ]["rating"]
            if len(pre) > 0:
                pre_ratings.extend(pre.tolist())
            if len(post) > 0:
                post_ratings.extend(post.tolist())

        if pre_ratings and post_ratings:
            rating_delta = np.mean(post_ratings) - np.mean(pre_ratings)
            print(f"  Avg rating change post-event: {rating_delta:+.2f} stars "
                  f"(pre: {np.mean(pre_ratings):.2f}, post: {np.mean(post_ratings):.2f})")
        else:
            print("  Insufficient review data for rating delta")
    except Exception as e:
        print(f"  Rating check skipped: {e}")

# ===========================================================================
# CHART 6 — Promo × Stockout Timeline
# ===========================================================================
print("\nBuilding Chart 6 — Promo × Stockout Timeline...")

try:
    # Monthly inventory summary per category
    inventory["year_month"] = inventory["snapshot_date"].dt.to_period("M")
    inv_monthly = (
        inventory.groupby(["category", "year_month"])
                 .agg(has_stockout=("stockout_flag", "max"),
                      avg_fill_rate=("fill_rate", "mean"))
                 .reset_index()
    )

    categories = sorted(promo_cats)
    n_cats = len(categories)

    time_min = pd.Timestamp(oat["order_date"].min().date().replace(day=1))
    time_max = pd.Timestamp(oat["order_date"].max().date()) + pd.DateOffset(months=1)

    fig, axes = plt.subplots(n_cats, 1,
                             figsize=(18, max(3 * n_cats + 2, 8)),
                             sharex=True)
    if n_cats == 1:
        axes = [axes]

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=PALETTE["negative"], alpha=0.30, label="Stockout period"),
        mpatches.Patch(color=PALETTE["positive"], alpha=0.70, label="Promotion active"),
        mpatches.Patch(color="#7d0000",           alpha=0.55, label="Overlap (double damage)"),
    ]

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        ax.set_xlim(time_min, time_max)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([cat], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        cat_inv = inv_monthly[
            (inv_monthly["category"] == cat) & (inv_monthly["has_stockout"] == 1)
        ]
        cat_promos = promotions[promotions["applicable_category"] == cat]

        # Stockout bands (red background)
        for _, row in cat_inv.iterrows():
            s = row["year_month"].to_timestamp()
            e = (row["year_month"] + 1).to_timestamp()
            ax.axvspan(s, e, ymin=0.1, ymax=0.9,
                       color=PALETTE["negative"], alpha=0.25)

        # Promo bars (green, narrower)
        for _, promo in cat_promos.iterrows():
            ps = max(promo["start_date"], time_min)
            pe = min(promo["end_date"], time_max)
            if pe > ps:
                ax.axvspan(ps, pe, ymin=0.35, ymax=0.65,
                           color=PALETTE["positive"], alpha=0.65)

        # Overlap zones (darker)
        for _, row in cat_inv.iterrows():
            s_inv = row["year_month"].to_timestamp()
            e_inv = (row["year_month"] + 1).to_timestamp()
            for _, promo in cat_promos.iterrows():
                ol_s = max(promo["start_date"], s_inv)
                ol_e = min(promo["end_date"],   e_inv)
                if ol_e > ol_s:
                    ax.axvspan(ol_s, ol_e, ymin=0.35, ymax=0.65,
                               color="#7d0000", alpha=0.55)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].set_xlabel("Time", fontsize=12)

    fig.suptitle("Promo × Stockout Overlap Timeline — Double Damage Events",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=10)
    fig.text(0.5, -0.02,
             "Data sources: promotions.csv · inventory.csv",
             ha="center", fontsize=9, color="grey")
    plt.tight_layout()
    plt.savefig("outputs/charts/chart6_promo_stockout_timeline.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/charts/chart6_promo_stockout_timeline.png")

except Exception as e:
    print(f"  ERROR in Chart 6: {e}")
    import traceback; traceback.print_exc()

# ===========================================================================
# SAVE OVERLAP TABLE
# ===========================================================================
if not overlap_df.empty:
    save_cols = [c for c in ["promo_id", "promo_name", "category", "snapshot_date",
                              "stockout_days", "fill_rate", "estimated_lost_revenue"]
                 if c in overlap_df.columns]
    overlap_df[save_cols].to_csv("outputs/tables/promo_stockout_overlaps.csv", index=False)
    print("  Saved: outputs/tables/promo_stockout_overlaps.csv")
else:
    # Save empty file so Act 5 can still load it
    pd.DataFrame(columns=["promo_id", "promo_name", "category", "snapshot_date",
                           "stockout_days", "fill_rate", "estimated_lost_revenue"]
                 ).to_csv("outputs/tables/promo_stockout_overlaps.csv", index=False)
    print("  Saved empty overlap table")

# ---------------------------------------------------------------------------
# FINDINGS — ACT 4
# ---------------------------------------------------------------------------
n_events   = len(overlap_df) if not overlap_df.empty else 0
cats_hit   = overlap_df["category"].unique().tolist() if not overlap_df.empty else []

print("""
==================================================
ACT 4 FINDINGS:
==================================================""")
print(f"- Number of double damage events           : {n_events}")
print(f"- Categories affected                      : {cats_hit}")
print(f"- Total estimated lost revenue from overlaps: {total_lost:,.0f} VND")
print(f"- Avg fill_rate during overlap events      : "
      f"{avg_fill:.1f}%" if not np.isnan(avg_fill) else "  Avg fill_rate: N/A")
if not np.isnan(rating_delta):
    print(f"- Avg review rating change post-event      : {rating_delta:+.2f} stars")
else:
    print("- Avg review rating change post-event      : insufficient data")
print(f"- Narrative: \"Company spent money running promotions to increase demand "
      f"exactly when supply was zero — estimated {total_lost:,.0f} VND in lost gross profit.\"")
print("==================================================")
