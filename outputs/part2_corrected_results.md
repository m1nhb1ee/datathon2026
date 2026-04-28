# Corrected Part 2 Results

## Act 1 - Revenue Anatomy
- Discount leakage: 749,607,320 VND.
- Return leakage: 510,598,507 VND.
- Combined leakage: 7.7% of gross revenue.
- Gross margin moved from 17.6% in 2012 to 5.4% in 2022.

## Act 2 - Promo Economics and Cohort Quality
- Net-negative promotions: 100.0%.
- Matched organic-margin proxy: promo rows underperform matched non-promo category-quarter margin by -1,633,470,867 VND. This is a margin benchmark, not a causal uplift estimate.
- Item return rate is not meaningfully higher for promo-acquired customers: 5.6% vs 5.6%, Mann-Whitney one-sided p=0.3081.
- Customer repeat rate is lower for promo-acquired customers: 68.6% vs 78.1%.
- Average lifetime gross profit per customer is 2,159 VND for promo vs 14,985 VND for organic, a 86% gap.

## Act 3 - Margin Trajectory
- Linear trend slope: -0.141 percentage points per quarter.
- Last-four-quarter gross margin: 4.4%.
- 2026 Q4 forecast: -0.1% with +/-1 SD range -9.0% to 8.7%.
- 2027 Q4 forecast: -0.7% with +/-1 SD range -9.5% to 8.2%.
- Scenario simulation: cutting 50 negative-GP promotions gives GP contribution +84.3% under 0% demand recapture.
- With 30% organic recapture, revenue impact is -20.9% and GP contribution impact is +107.9%.

## Act 4 - Promo x Stockout Overlap
- Event unit: promo x category x month.
- Overlap events: 426 across 50 promotions and Casual, GenZ, Outdoor, Streetwear.
- Average product stockout share during overlap months: 67.5%.
- Average fill rate during overlap months: 96.2%.
- Estimated lost gross profit from these event-level overlaps: 23,547,465 VND.
- Highest inventory-risk promo: Mid-Year Sale 2018 with risk score 286,573,111.
- Review rating delta after overlap windows: +0.00 stars.

## Act 5 - Promo Triage
- KEEP: 0 promotions, total GP contribution 0 VND.
- CUT: 50 promotions, total GP contribution -846,879,462 VND.
- RESCHEDULE: 0 promotions.
- Cutting negative-GP promotions implies revenue -29.9% and GP contribution +84.3% under zero demand recapture.

## Method Notes
- Contribution metrics use all order statuses; see `order_status_profit_audit.csv` for sensitivity and status mix transparency.
- `Promo GP Contribution Map` is contribution accounting, not true ROI. True incremental ROI would require a counterfactual demand model or experiment.
