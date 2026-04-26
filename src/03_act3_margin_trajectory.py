import numpy as np
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

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
# CHART 5 — Margin Trajectory
# ===========================================================================
print("\nBuilding Chart 5 — Margin Trajectory...")

try:
    # Quarterly aggregation
    try:
        quarterly = oat.groupby(pd.Grouper(key="order_date", freq="QE")).agg(
            gross_profit=("gross_profit", "sum"),
            net_revenue=("net_revenue", "sum"),
        )
    except Exception:
        quarterly = oat.groupby(pd.Grouper(key="order_date", freq="Q")).agg(
            gross_profit=("gross_profit", "sum"),
            net_revenue=("net_revenue", "sum"),
        )

    quarterly = quarterly[quarterly["net_revenue"] > 0].copy()
    quarterly["gross_margin_pct"] = quarterly["gross_profit"] / quarterly["net_revenue"] * 100

    # Linear trend fit on actual data
    x_actual = np.arange(len(quarterly))
    y_actual = quarterly["gross_margin_pct"].values
    coeffs   = np.polyfit(x_actual, y_actual, 1)
    trend_fn = np.poly1d(coeffs)
    slope    = coeffs[0]

    # Residuals & std
    residuals = y_actual - trend_fn(x_actual)
    std_resid = np.std(residuals)

    # Extrapolate to 2027 Q4 (20 quarters beyond last actual quarter)
    n_extra  = 20
    x_future = np.arange(len(quarterly), len(quarterly) + n_extra)
    future_margin = trend_fn(x_future)

    # Build future date index
    last_q = quarterly.index[-1]
    future_dates = pd.date_range(
        start=last_q + pd.DateOffset(months=3),
        periods=n_extra,
        freq="QE" if "QE" in str(quarterly.index.freq) else "Q",
    )
    if len(future_dates) == 0:
        future_dates = [last_q + pd.DateOffset(months=3*i) for i in range(1, n_extra+1)]

    # Identify specific forecast quarters for annotation
    future_series = pd.Series(future_margin, index=pd.DatetimeIndex(future_dates))

    # Find 2026 Q4 and 2027 Q4
    def get_quarter_value(series, year, q):
        matches = [(d, v) for d, v in series.items()
                   if d.year == year and d.quarter == q]
        return matches[0] if matches else (None, None)

    d_2026q4, v_2026q4 = get_quarter_value(future_series, 2026, 4)
    d_2027q4, v_2027q4 = get_quarter_value(future_series, 2027, 4)

    # Fallback: use last available projected values
    if v_2026q4 is None:
        v_2026q4 = future_margin[min(15, n_extra-1)]
    if v_2027q4 is None:
        v_2027q4 = future_margin[min(19, n_extra-1)]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(16, 8))

    # Actual solid line
    ax.plot(quarterly.index, quarterly["gross_margin_pct"],
            color=PALETTE["positive"], linewidth=2.2, label="Actual Quarterly Gross Margin %")

    # Trend line over actual
    ax.plot(quarterly.index, trend_fn(x_actual),
            color=PALETTE["neutral"], linewidth=1.2, linestyle=":", alpha=0.7,
            label="Trend (actual period)")

    # Forecast dashed line
    ax.plot(future_dates, future_margin,
            color=PALETTE["negative"], linewidth=2.2, linestyle="--",
            label="Extrapolated Trend (forecast)")

    # Confidence band ±1 std
    ax.fill_between(
        future_dates,
        future_margin - std_resid,
        future_margin + std_resid,
        color=PALETTE["negative"], alpha=0.12, label=f"±1 SD band ({std_resid:.2f}%)",
    )

    # Forecast start vertical line
    forecast_start = pd.Timestamp("2023-01-01")
    ax.axvline(forecast_start, color="grey", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(forecast_start, ax.get_ylim()[1] if ax.get_ylim()[1] else 30,
            "  Forecast start", fontsize=9, color="grey", va="top")

    # Annotations for 2026 Q4 and 2027 Q4
    if d_2026q4 is not None:
        ax.annotate(
            f"2026 Q4: {v_2026q4:.1f}%",
            xy=(d_2026q4, v_2026q4),
            xytext=(30, 10), textcoords="offset points",
            fontsize=10, fontweight="bold", color=PALETTE["negative"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["negative"]),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["negative"]),
        )
    if d_2027q4 is not None:
        ax.annotate(
            f"2027 Q4: {v_2027q4:.1f}%",
            xy=(d_2027q4, v_2027q4),
            xytext=(30, -20), textcoords="offset points",
            fontsize=10, fontweight="bold", color=PALETTE["negative"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["negative"]),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["negative"]),
        )

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_title("Gross Margin Trajectory — Actual (2012–2022) & Extrapolated (2023–2027)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Gross Margin %", fontsize=12)
    ax.legend(fontsize=10)
    fig.text(0.5, 0.01,
             "Data sources: oat.parquet (order_items · products)",
             ha="center", fontsize=9, color="grey")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("outputs/charts/chart5_margin_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/charts/chart5_margin_trajectory.png")

    # Current margin (last year in data)
    last_year = oat["order_date"].dt.year.max()
    margin_current = (
        oat[oat["order_date"].dt.year == last_year]["gross_profit"].sum()
        / oat[oat["order_date"].dt.year == last_year]["net_revenue"].sum() * 100
    )

except Exception as e:
    print(f"  ERROR in Chart 5: {e}")
    import traceback; traceback.print_exc()
    slope = margin_current = v_2026q4 = v_2027q4 = 0

# ---------------------------------------------------------------------------
# FINDINGS — ACT 3
# ---------------------------------------------------------------------------
print("""
==================================================
ACT 3 FINDINGS:
==================================================""")
print(f"- Current gross margin (last yr)   : {margin_current:.1f}%")
print(f"- Projected gross margin (2026 Q4) : {v_2026q4:.1f}%")
print(f"- Projected gross margin (2027 Q4) : {v_2027q4:.1f}%")
print(f"- Trend slope                      : {slope:+.3f}% per quarter")
print(f"- Narrative: \"At current rate, gross margin reaches {v_2026q4:.1f}% by end of 2026.\"")
print("==================================================")
