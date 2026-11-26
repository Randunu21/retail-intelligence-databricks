"""Product segmentation specifics (Phase 4): who gets clustered, on which features.

The clustering machinery itself (preprocessing, sweep, stability, final model, naming) is the
Phase 3 code in `segmentation.*`, reused with `stock_code` as the id. This module holds only
what is different for products.

**The long tail (decided from the 4.1 numbers).** 4,725 products; the top 1,030 make 80 % of
revenue, while 978 products have fewer than 20 invoices and together make 1.7 % of revenue.
For a product with a handful of orders, share-type features (Q4 share, active weeks, return
rate) jump in big steps with every single order, so clustering them clusters noise. Rule:

| eligibility | condition | segment |
|---|---|---|
| `new_product` | months_on_sale < MIN_MONTHS (CV undefined, too new to judge) | rule-based "New products" |
| `low_volume` | n_invoices < MIN_INVOICES | rule-based "Long tail" |
| `clustered` | everything else | from the clustering model |

Alternatives (interview point): cluster everything (noise drives the clusters, many tiny
segments), weight products by revenue (big sellers dominate the centroids), or a fixed
"top N products" cut (arbitrary, ignores how much history a product has).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

MIN_INVOICES = 20
MIN_MONTHS = 3
ELIGIBILITY_COL = "cluster_eligibility"
CLUSTERED, NEW_PRODUCT, LOW_VOLUME = "clustered", "new_product", "low_volume"

# Candidates in PRIORITY order: when two are near-duplicates (|r| > 0.9 after the log), the
# earlier one is kept. Velocity and price first (the two axes a merchandiser thinks in), then
# seasonality, lumpiness, lifecycle, international appeal, regularity, reach.
PRODUCT_CANDIDATE_FEATURES: list[str] = [
    "units_per_week",
    "median_unit_price",
    "seasonal_revenue_share",
    "monthly_units_cv",
    "days_since_last_sale",
    "non_uk_revenue_share",
    "active_week_share",
    "unique_customers",
]

# In the feature table and used to PROFILE segments, but not clustered on:
PRODUCT_PROFILE_ONLY_FEATURES: dict[str, str] = {
    "revenue": "log(revenue) ~ log(price) + log(velocity) + log(weeks): no new direction",
    "revenue_share": "revenue / constant: same as revenue",
    "avg_unit_price": "near-copy of median_unit_price (differs only by bulk discounts)",
    "n_invoices": "volume again (overlaps velocity and unique_customers)",
    "avg_quantity_per_line": "bulk-buying indicator; extreme tail from a few wholesale lines",
    "return_rate": "zero-inflated (p25 = 0): would carve a 'returned at all' split",
    "n_countries": "scales with volume; non_uk_revenue_share measures spread independently",
    "days_on_sale": "almost constant for eligible products (most on sale since 2009-2010)",
}

# Rule-based segments for products that are not clustered.
RULE_SEGMENTS: dict[str, dict[str, str]] = {
    NEW_PRODUCT: {
        "segment_name": "New Products",
        "recommended_action": "Give launch visibility (homepage, new-in email); review after "
        "3 months of sales",
    },
    LOW_VOLUME: {
        "segment_name": "Long Tail",
        "recommended_action": "Keep online only if cheap to hold; candidates for clearance or "
        "delisting; bundle with best-sellers",
    },
}


def add_eligibility(
    features: DataFrame, min_invoices: int = MIN_INVOICES, min_months: int = MIN_MONTHS
) -> DataFrame:
    """Add `cluster_eligibility` (new_product is checked first: a new product with many
    orders is still too new for velocity/seasonality to mean much)."""
    return features.withColumn(
        ELIGIBILITY_COL,
        F.when(F.col("months_on_sale") < min_months, F.lit(NEW_PRODUCT))
        .when(F.col("n_invoices") < min_invoices, F.lit(LOW_VOLUME))
        .otherwise(F.lit(CLUSTERED)),
    )


def eligibility_summary(df: DataFrame) -> list[dict]:
    """Products and revenue per eligibility class (df must have the eligibility column)."""
    total = df.agg(F.count(F.lit(1)).alias("n"), F.sum("revenue").alias("rev")).first()
    rows = (
        df.groupBy(ELIGIBILITY_COL)
        .agg(F.count(F.lit(1)).alias("products"), F.sum("revenue").alias("revenue"))
        .collect()
    )
    out = [
        {
            "eligibility": r[ELIGIBILITY_COL],
            "products": int(r["products"]),
            "product_share": r["products"] / total["n"],
            "revenue_share": r["revenue"] / total["rev"],
        }
        for r in rows
    ]
    order = {CLUSTERED: 0, LOW_VOLUME: 1, NEW_PRODUCT: 2}
    return sorted(out, key=lambda r: order.get(r["eligibility"], 9))


# Stability candidates, stated BEFORE seeing the sweep: per algorithm, the top-n k by
# silhouette among k >= k_min with the smallest segment >= min_share.
DEFAULT_CANDIDATES_PER_ALGORITHM: dict[str, int] = {
    "kmeans": 4,
    "bisecting_kmeans": 2,
    "gmm": 1,
}


def stability_candidates(
    sweep_results: Sequence[Mapping],
    per_algorithm: Mapping[str, int] = DEFAULT_CANDIDATES_PER_ALGORITHM,
    min_share: float = 0.03,
    k_min: int = 3,
) -> list[tuple[str, int]]:
    """(algorithm, k) pairs to test for stability, chosen by a fixed rule (no hand-picking)."""
    out: list[tuple[str, int]] = []
    for algorithm, n in per_algorithm.items():
        ok = [
            r
            for r in sweep_results
            if r["algorithm"] == algorithm
            and not r.get("error")
            and r["k"] >= k_min
            and r["metrics"].get("silhouette") is not None
            and r["metrics"]["min_share"] >= min_share
        ]
        ok.sort(key=lambda r: (-r["metrics"]["silhouette"], r["k"]))
        out += [(algorithm, r["k"]) for r in ok[:n]]
    return out
