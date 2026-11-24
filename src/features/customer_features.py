"""Customer feature table for segmentation (Phase 3) and the propensity model (Phase 5).

One row per known customer, computed **as of** a snapshot date: only sales and returns dated
strictly before `as_of` are used. Phase 3 uses one snapshot (the day after the last sale);
Phase 5 calls the same function with an earlier cutoff, so features never see the label
window (no leakage).

Inputs are retail.silver.sales and retail.silver.returns (same columns; returns have negative
quantity and line_revenue). Lines without a customer id are excluded: they can't be tied to a
customer (226,761 sales lines in Phase 1, kept in Silver for product analysis only).

Feature definitions (all money in GBP, all day counts in whole days):

| Feature | Definition |
|---|---|
| recency_days | days from the last purchase day to `as_of` (1 = bought on the last day of data) |
| frequency | distinct invoices (orders) |
| monetary | gross sales value (sum of line_revenue), before returns |
| avg_basket_value | monetary / frequency |
| distinct_products | distinct stock codes bought |
| returned_value | value of returned goods (positive number) |
| n_return_invoices | distinct cancellation invoices |
| return_rate | returned_value / monetary, capped at 1 (see below) |
| tenure_days | days from the first purchase day to `as_of` |
| n_purchase_days | distinct days with a purchase |
| mean_days_between_purchases | mean gap between consecutive purchase days; NULL if only one day |
| preferred_country | country with the most revenue (ties: more invoices, then name) |
| n_countries | distinct countries ordered to |
| seasonal_revenue_share | share of monetary in the seasonal months (default Q4: Oct-Dec) |
| is_tenure_censored | first purchase falls in the first month of data, so true tenure is unknown |

Notes on the less obvious choices:
- **Mean purchase interval without a window function.** The mean of consecutive gaps between
  sorted days d1 < d2 < ... < dn telescopes: ((d2-d1) + ... + (dn-d(n-1))) / (n-1) =
  (dn - d1) / (n-1). So it comes out of the same single aggregation as RFM (no sort, no lag).
- **Return rate on value, capped at 1.** Value (not quantity) because a returned GBP 200
  mirror matters more than 12 returned pencils. Some customers return goods bought before the
  data starts (Dec 2009), which can push the raw ratio above 1; capping keeps the feature
  in [0, 1] and `returned_value` keeps the raw amount.
- **Customers who only appear in returns** (no sale before `as_of`) are excluded: every
  feature above is undefined for them.
- **Seasonal share = Q4 by default.** This is a gift retailer; Oct-Dec is the Christmas peak.
  Caveat: the data covers Dec 2009, Q4 2010 and Oct to 9 Dec 2011, and a customer whose whole
  history lies in Q4 2011 gets share 1. Read it together with tenure.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Months counted as "seasonal" (Q4, the Christmas gifting peak).
SEASONAL_MONTHS: tuple[int, ...] = (10, 11, 12)

# Output schema: (column, Spark SQL type, comment). Order = column order of the table.
# `computed_at` is added when the snapshot is written (it isn't a function of the data).
FEATURE_SCHEMA: list[tuple[str, str, str]] = [
    ("customer_id", "INT", "Customer id (primary key, with as_of_date)"),
    ("as_of_date", "DATE", "Snapshot date: only activity strictly before this date is used"),
    ("recency_days", "INT", "Days from the last purchase day to as_of_date"),
    ("frequency", "INT", "Distinct invoices (orders)"),
    ("monetary", "DOUBLE", "Gross sales value in GBP (before returns)"),
    ("avg_basket_value", "DOUBLE", "monetary / frequency"),
    ("distinct_products", "INT", "Distinct stock codes bought"),
    ("returned_value", "DOUBLE", "Value of returned goods in GBP (positive)"),
    ("n_return_invoices", "INT", "Distinct cancellation invoices"),
    ("return_rate", "DOUBLE", "returned_value / monetary, capped at 1"),
    ("tenure_days", "INT", "Days from the first purchase day to as_of_date"),
    ("n_purchase_days", "INT", "Distinct days with a purchase"),
    (
        "mean_days_between_purchases",
        "DOUBLE",
        "Mean gap between consecutive purchase days; NULL with one purchase day",
    ),
    ("preferred_country", "STRING", "Country with the most revenue"),
    ("n_countries", "INT", "Distinct countries ordered to"),
    ("seasonal_revenue_share", "DOUBLE", "Share of monetary in Oct-Dec (Q4)"),
    ("first_purchase_date", "DATE", "First purchase day"),
    ("last_purchase_date", "DATE", "Last purchase day"),
    (
        "is_tenure_censored",
        "BOOLEAN",
        "First purchase in the first month of data: true tenure may be longer",
    ),
]
FEATURE_COLUMNS: list[str] = [name for name, _, _ in FEATURE_SCHEMA]
KEY_COLUMNS: tuple[str, str] = ("customer_id", "as_of_date")


def _as_date_col(as_of: date | str):
    return F.to_date(F.lit(str(as_of)))


def default_as_of(sales: DataFrame) -> date:
    """Snapshot date for 'today' in this dataset: the day after the last sale."""
    last = sales.agg(F.max("invoice_date").alias("d")).first()["d"]
    return last + timedelta(days=1)


def known_customer_lines(df: DataFrame, as_of: date | str) -> DataFrame:
    """Lines with a customer id, dated strictly before `as_of`."""
    return df.where(
        F.col("customer_id").isNotNull() & (F.col("invoice_date") < _as_date_col(as_of))
    )


def sales_aggregates(
    sales: DataFrame, as_of: date | str, seasonal_months: Sequence[int] = SEASONAL_MONTHS
) -> DataFrame:
    """RFM, basket, tenure, interval and seasonality features in ONE aggregation per customer."""
    as_of_col = _as_date_col(as_of)
    revenue = F.col("line_revenue").cast("double")
    is_seasonal = F.month("invoice_date").isin(list(seasonal_months))

    agg = (
        known_customer_lines(sales, as_of)
        .groupBy("customer_id")
        .agg(
            F.min("invoice_date").alias("first_purchase_date"),
            F.max("invoice_date").alias("last_purchase_date"),
            F.countDistinct("invoice").cast("int").alias("frequency"),
            F.sum(revenue).alias("monetary"),
            F.countDistinct("stock_code").cast("int").alias("distinct_products"),
            F.countDistinct("invoice_date").cast("int").alias("n_purchase_days"),
            F.countDistinct("country").cast("int").alias("n_countries"),
            F.sum(F.when(is_seasonal, revenue).otherwise(F.lit(0.0))).alias("_seasonal_revenue"),
        )
    )
    span = F.datediff("last_purchase_date", "first_purchase_date")
    return (
        agg.withColumn("recency_days", F.datediff(as_of_col, "last_purchase_date"))
        .withColumn("tenure_days", F.datediff(as_of_col, "first_purchase_date"))
        .withColumn("avg_basket_value", F.col("monetary") / F.col("frequency"))
        .withColumn(
            "mean_days_between_purchases",
            F.when(F.col("n_purchase_days") > 1, span / (F.col("n_purchase_days") - 1)),
        )
        .withColumn("seasonal_revenue_share", F.col("_seasonal_revenue") / F.col("monetary"))
        .drop("_seasonal_revenue")
    )


def return_aggregates(returns: DataFrame, as_of: date | str) -> DataFrame:
    """Returned value (positive) and number of cancellation invoices per customer."""
    return (
        known_customer_lines(returns, as_of)
        .groupBy("customer_id")
        .agg(
            (-F.sum(F.col("line_revenue").cast("double"))).alias("returned_value"),
            F.countDistinct("invoice").cast("int").alias("n_return_invoices"),
        )
    )


def preferred_country(sales: DataFrame, as_of: date | str) -> DataFrame:
    """Country with the most revenue per customer (ties: more invoices, then alphabetical)."""
    per_country = (
        known_customer_lines(sales, as_of)
        .groupBy("customer_id", "country")
        .agg(
            F.sum(F.col("line_revenue").cast("double")).alias("_rev"),
            F.countDistinct("invoice").alias("_inv"),
        )
    )
    w = Window.partitionBy("customer_id").orderBy(
        F.col("_rev").desc(), F.col("_inv").desc(), F.col("country").asc()
    )
    return (
        per_country.withColumn("_rn", F.row_number().over(w))
        .where(F.col("_rn") == 1)
        .select("customer_id", F.col("country").alias("preferred_country"))
    )


def build_customer_features(
    sales: DataFrame,
    returns: DataFrame,
    as_of: date | str,
    data_start: date | str | None = None,
    seasonal_months: Sequence[int] = SEASONAL_MONTHS,
) -> DataFrame:
    """Customer feature snapshot as of `as_of` (pure: DataFrames in, DataFrame out).

    `data_start` is the first day of data (used for `is_tenure_censored`); by default the
    earliest invoice_date in `sales` (all lines, incl. unknown customers).
    """
    if data_start is None:
        data_start = sales.agg(F.min("invoice_date").alias("d")).first()["d"]
    censor_limit = F.add_months(_as_date_col(data_start), 1)

    features = (
        sales_aggregates(sales, as_of, seasonal_months)
        .join(return_aggregates(returns, as_of), "customer_id", "left")
        .join(preferred_country(sales, as_of), "customer_id", "left")
        .fillna({"returned_value": 0.0, "n_return_invoices": 0})
        .withColumn("return_rate", F.least(F.col("returned_value") / F.col("monetary"), F.lit(1.0)))
        .withColumn("is_tenure_censored", F.col("first_purchase_date") < censor_limit)
        .withColumn("as_of_date", _as_date_col(as_of))
    )
    types = {name: dtype for name, dtype, _ in FEATURE_SCHEMA}
    return features.select([F.col(c).cast(types[c]).alias(c) for c in FEATURE_COLUMNS])


# --------------------------------------------------------------------------- checks / summary


def reconcile_features(
    features: DataFrame, sales: DataFrame, returns: DataFrame, as_of: date | str
) -> dict:
    """Independent recounts that the notebook asserts on (one small aggregation per check).

    - customers: rows in `features` vs distinct known customers with a sale before `as_of`
    - duplicate_keys: rows minus distinct customer_id (must be 0)
    - monetary_diff: total monetary vs total line_revenue of the same sales lines (~0)
    - single_day_customers vs null_intervals: must be equal (interval undefined for one day)
    - returns_only_customers: customers with returns but no sale before `as_of` (excluded)
    - return_rate_capped: customers whose raw returned_value / monetary exceeded 1
    """
    known_sales = known_customer_lines(sales, as_of)
    sales_customers = known_sales.select("customer_id").distinct()
    f = features.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("customer_id").alias("distinct_customers"),
        F.sum("monetary").alias("monetary"),
        F.sum(F.when(F.col("n_purchase_days") == 1, 1).otherwise(0)).alias("single_day"),
        F.sum(F.when(F.col("mean_days_between_purchases").isNull(), 1).otherwise(0)).alias(
            "null_intervals"
        ),
        F.sum(F.when(F.col("returned_value") > F.col("monetary"), 1).otherwise(0)).alias("capped"),
        F.min("return_rate").alias("min_rr"),
        F.max("return_rate").alias("max_rr"),
    ).first()
    expected_monetary = known_sales.agg(
        F.sum(F.col("line_revenue").cast("double")).alias("m")
    ).first()["m"]
    returns_only = (
        known_customer_lines(returns, as_of)
        .select("customer_id")
        .distinct()
        .join(sales_customers, "customer_id", "left_anti")
        .count()
    )
    return {
        "customers": f["rows"],
        "expected_customers": sales_customers.count(),
        "duplicate_keys": f["rows"] - f["distinct_customers"],
        "monetary_diff": round((f["monetary"] or 0.0) - (expected_monetary or 0.0), 6),
        "single_day_customers": f["single_day"],
        "null_intervals": f["null_intervals"],
        "returns_only_customers": returns_only,
        "return_rate_capped": f["capped"],
        "return_rate_range": (f["min_rr"], f["max_rr"]),
    }


def describe_features(features: DataFrame, columns: Sequence[str]) -> DataFrame:
    """Long table, one row per feature: nulls, mean, quantiles, max and skewness.

    Used to decide preprocessing (Phase 3.2): skewness well above 1 means a long right tail,
    which distance-based clustering would otherwise let dominate.
    """
    probs = [0.01, 0.25, 0.5, 0.75, 0.95, 0.99]
    rows = None
    for c in columns:
        x = F.col(c).cast("double")
        q = F.percentile_approx(x, probs, 10000)
        one = features.agg(
            F.lit(c).alias("feature"),
            F.sum(F.when(x.isNull(), 1).otherwise(0)).alias("n_null"),
            F.min(x).alias("min"),
            *[q[i].alias(f"p{int(p * 100)}") for i, p in enumerate(probs)],
            F.max(x).alias("max"),
            F.mean(x).alias("mean"),
            F.skewness(x).alias("skewness"),
        )
        rows = one if rows is None else rows.unionByName(one)
    return rows
