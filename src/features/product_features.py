"""Product feature table for product segmentation (Phase 4).

One row per product (`stock_code`) computed **as of** a snapshot date: only sales and returns
dated strictly before `as_of` are used, the same rule and table pattern as the customer
features (Phase 3), so later snapshots can be added side by side.

Inputs are retail.silver.sales and retail.silver.returns. Unlike the customer features,
**all sales lines are used, including lines without a customer id**: that revenue is real
demand for the product (226,761 lines in Phase 1). Only `unique_customers` counts known
customers.

Feature definitions (money in GBP, quantities in units):

| Feature | Business question | Definition |
|---|---|---|
| n_invoices | how often is it bought? | distinct invoices (orders) containing the product |
| n_lines | | sales lines (= n_invoices unless an invoice lists it twice) |
| units_sold | | sum of quantity |
| revenue | | sum of line_revenue |
| revenue_share | revenue contribution | revenue / total revenue of all products (same window) |
| median_unit_price | price level | median of the line price (the list price, not discounted by volume) |
| avg_unit_price | | revenue / units_sold (quantity-weighted; lower when bought in bulk) |
| avg_quantity_per_line | bulk vs single item | units_sold / n_lines |
| unique_customers | reach | distinct known customer ids |
| n_countries | country spread | distinct countries ordered to |
| non_uk_revenue_share | country spread | share of revenue from outside the home country (UK) |
| units_per_week | sales velocity | units_sold / weeks_on_sale |
| active_week_share | regularity | weeks with a sale / weeks_on_sale |
| seasonal_revenue_share | seasonality | share of revenue in Oct-Dec (Q4) |
| monthly_units_cv | seasonality / lumpiness | coefficient of variation of monthly units over months_on_sale (zero months included); NULL with < 3 months |
| months_on_sale, weeks_on_sale | | calendar months / weeks from the first sale to the last day before as_of |
| units_returned, returned_value | | returned units / value (positive numbers) |
| return_rate | quality | units_returned / units_sold, capped at 1 |
| days_on_sale | age in catalogue | days from the first sale to as_of |
| days_since_last_sale | discontinued? | days from the last sale to as_of |
| is_history_censored | | first sale in the first month of data: the product may be older |

Notes on the less obvious choices:
- **Velocity and regularity are measured "since first sale"**, not between first and last
  sale, so a product that stopped selling a year ago scores low (slow mover / discontinued)
  instead of looking fast because it sold well while it existed.
- **Monthly CV without a calendar grid.** With n = months_on_sale months (zero months
  included) and monthly units x_i, mean = sum(x)/n and var = sum(x^2)/n - mean^2. Months with
  no sales add 0 to both sums, so one product-month aggregation is enough. Caveat: the last
  month of data (Dec 2011, 9 days) is partial, which raises every product's CV a little.
- **Q4 share baseline.** The data has Dec 2009, Oct-Dec 2010 and Oct-9 Dec 2011, so a
  perfectly flat product sold over the whole period has a Q4 share of roughly 25 %, not 0.
  Read seasonal share relative to that baseline (and together with monthly_units_cv).
- **Return rate on units, capped at 1**, matching retail.gold.product_performance (which is
  not capped). A few products have returns of goods sold before Dec 2009, so the raw ratio
  can exceed 1; `units_returned` keeps the raw amount.
- **Products that only appear in returns** (no sale before `as_of`) are excluded.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from features.customer_features import SEASONAL_MONTHS

HOME_COUNTRY = "United Kingdom"
MIN_MONTHS_FOR_CV = 3

PRODUCT_FEATURE_SCHEMA: list[tuple[str, str, str]] = [
    ("stock_code", "STRING", "Product code (primary key, with as_of_date)"),
    ("as_of_date", "DATE", "Snapshot date: only activity strictly before this date is used"),
    ("n_invoices", "INT", "Distinct invoices (orders) containing the product"),
    ("n_lines", "INT", "Sales lines"),
    ("units_sold", "BIGINT", "Units sold"),
    ("revenue", "DOUBLE", "Sales revenue in GBP"),
    ("revenue_share", "DOUBLE", "Share of total revenue of all products in the same window"),
    ("median_unit_price", "DOUBLE", "Median line price in GBP (price level)"),
    ("avg_unit_price", "DOUBLE", "revenue / units_sold (quantity-weighted price)"),
    ("avg_quantity_per_line", "DOUBLE", "units_sold / n_lines"),
    ("unique_customers", "INT", "Distinct known customers"),
    ("n_countries", "INT", "Distinct countries ordered to"),
    ("non_uk_revenue_share", "DOUBLE", "Share of revenue from outside the United Kingdom"),
    ("units_per_week", "DOUBLE", "units_sold / weeks_on_sale (sales velocity)"),
    ("active_week_share", "DOUBLE", "Weeks with a sale / weeks_on_sale"),
    ("seasonal_revenue_share", "DOUBLE", "Share of revenue in Oct-Dec (Q4)"),
    (
        "monthly_units_cv",
        "DOUBLE",
        "Coefficient of variation of monthly units (zero months included); NULL with < 3 months",
    ),
    ("months_on_sale", "INT", "Calendar months from the first sale month to the last data month"),
    ("weeks_on_sale", "INT", "Calendar weeks from the first sale week to the last data week"),
    ("units_returned", "BIGINT", "Units returned (positive)"),
    ("returned_value", "DOUBLE", "Value of returned units in GBP (positive)"),
    ("return_rate", "DOUBLE", "units_returned / units_sold, capped at 1"),
    ("days_on_sale", "INT", "Days from the first sale to as_of_date"),
    ("days_since_last_sale", "INT", "Days from the last sale to as_of_date"),
    ("first_sale_date", "DATE", "First sale day"),
    ("last_sale_date", "DATE", "Last sale day"),
    (
        "is_history_censored",
        "BOOLEAN",
        "First sale in the first month of data: the product may have been sold before",
    ),
]
PRODUCT_FEATURE_COLUMNS: list[str] = [name for name, _, _ in PRODUCT_FEATURE_SCHEMA]
PRODUCT_KEY_COLUMNS: tuple[str, str] = ("stock_code", "as_of_date")
PRODUCT_TABLE_COMMENT = (
    "Product features (price level, velocity, revenue contribution, seasonality, returns, "
    "reach, country spread) per product and snapshot date. Feature table: PK (stock_code, "
    "as_of_date TIMESERIES). Built by src/features/product_features.py (Phase 4)."
)


def _date(value: date | str):
    return F.to_date(F.lit(str(value)))


def _last_data_day(as_of: date | str) -> date:
    d = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of))
    return d - timedelta(days=1)


def lines_before(df: DataFrame, as_of: date | str) -> DataFrame:
    """All lines (with or without customer id) dated strictly before `as_of`."""
    return df.where(F.col("invoice_date") < _date(as_of))


def _week_start(col):
    return F.date_trunc("week", col).cast("date")  # Monday of the week


def product_sales_aggregates(
    sales: DataFrame,
    as_of: date | str,
    seasonal_months: Sequence[int] = SEASONAL_MONTHS,
    home_country: str = HOME_COUNTRY,
) -> DataFrame:
    """Everything that comes from one groupBy(stock_code) over the sales lines."""
    revenue = F.col("line_revenue").cast("double")
    is_seasonal = F.month("invoice_date").isin(list(seasonal_months))
    is_home = F.col("country") == F.lit(home_country)
    last_day = _date(_last_data_day(as_of))

    agg = (
        lines_before(sales, as_of)
        .groupBy("stock_code")
        .agg(
            F.min("invoice_date").alias("first_sale_date"),
            F.max("invoice_date").alias("last_sale_date"),
            F.countDistinct("invoice").cast("int").alias("n_invoices"),
            F.count(F.lit(1)).cast("int").alias("n_lines"),
            F.sum("quantity").cast("bigint").alias("units_sold"),
            F.sum(revenue).alias("revenue"),
            F.percentile_approx(F.col("price").cast("double"), 0.5, 10000).alias(
                "median_unit_price"
            ),
            F.countDistinct("customer_id").cast("int").alias("unique_customers"),
            F.countDistinct("country").cast("int").alias("n_countries"),
            F.sum(F.when(is_home, F.lit(0.0)).otherwise(revenue)).alias("_non_home_revenue"),
            F.sum(F.when(is_seasonal, revenue).otherwise(F.lit(0.0))).alias("_seasonal_revenue"),
            F.countDistinct(_week_start(F.col("invoice_date"))).alias("_sale_weeks"),
        )
    )
    weeks_on_sale = (
        F.datediff(_week_start(last_day), _week_start(F.col("first_sale_date"))) / 7 + 1
    ).cast("int")
    months_on_sale = (
        F.months_between(F.trunc(last_day, "month"), F.trunc(F.col("first_sale_date"), "month")) + 1
    ).cast("int")
    return (
        agg.withColumn("weeks_on_sale", weeks_on_sale)
        .withColumn("months_on_sale", months_on_sale)
        .withColumn("avg_unit_price", F.col("revenue") / F.col("units_sold"))
        .withColumn("avg_quantity_per_line", F.col("units_sold") / F.col("n_lines"))
        .withColumn("units_per_week", F.col("units_sold") / F.col("weeks_on_sale"))
        .withColumn("active_week_share", F.col("_sale_weeks") / F.col("weeks_on_sale"))
        .withColumn("non_uk_revenue_share", F.col("_non_home_revenue") / F.col("revenue"))
        .withColumn("seasonal_revenue_share", F.col("_seasonal_revenue") / F.col("revenue"))
        .withColumn("days_on_sale", F.datediff(_date(as_of), F.col("first_sale_date")))
        .withColumn("days_since_last_sale", F.datediff(_date(as_of), F.col("last_sale_date")))
        .drop("_non_home_revenue", "_seasonal_revenue", "_sale_weeks")
    )


def monthly_unit_moments(sales: DataFrame, as_of: date | str) -> DataFrame:
    """Per product: sum of monthly units and sum of squared monthly units (months with sales).

    Months without sales contribute 0 to both sums, so the CV can be computed over the full
    months_on_sale without building a product x month grid.
    """
    monthly = (
        lines_before(sales, as_of)
        .groupBy("stock_code", F.trunc("invoice_date", "month").alias("_month"))
        .agg(F.sum("quantity").cast("double").alias("_units"))
    )
    return monthly.groupBy("stock_code").agg(
        F.sum("_units").alias("_sum_units"),
        F.sum(F.col("_units") * F.col("_units")).alias("_sum_sq_units"),
    )


def coefficient_of_variation(sum_x, sum_sq, n, min_n: int = MIN_MONTHS_FOR_CV):
    """Population CV from sums: sqrt(sum_sq/n - mean^2) / mean; NULL when n < min_n."""
    mean = sum_x / n
    var = F.greatest(sum_sq / n - mean * mean, F.lit(0.0))  # guard tiny negative rounding
    return F.when((n >= min_n) & (mean > 0), F.sqrt(var) / mean)


def product_return_aggregates(returns: DataFrame, as_of: date | str) -> DataFrame:
    """Returned units and value per product (positive numbers)."""
    return (
        lines_before(returns, as_of)
        .groupBy("stock_code")
        .agg(
            (-F.sum("quantity")).cast("bigint").alias("units_returned"),
            (-F.sum(F.col("line_revenue").cast("double"))).alias("returned_value"),
        )
    )


def build_product_features(
    sales: DataFrame,
    returns: DataFrame,
    as_of: date | str,
    data_start: date | str | None = None,
    seasonal_months: Sequence[int] = SEASONAL_MONTHS,
    home_country: str = HOME_COUNTRY,
    min_months_for_cv: int = MIN_MONTHS_FOR_CV,
) -> DataFrame:
    """Product feature snapshot as of `as_of` (pure: DataFrames in, DataFrame out).

    `data_start` = first day of data (for `is_history_censored`); by default the earliest
    invoice_date in `sales`.
    """
    if data_start is None:
        data_start = sales.agg(F.min("invoice_date").alias("d")).first()["d"]
    censor_limit = F.add_months(_date(data_start), 1)

    base = product_sales_aggregates(sales, as_of, seasonal_months, home_country)
    total = base.agg(F.sum("revenue").alias("_total_revenue"))
    features = (
        base.crossJoin(F.broadcast(total))
        .join(monthly_unit_moments(sales, as_of), "stock_code", "left")
        .join(product_return_aggregates(returns, as_of), "stock_code", "left")
        .fillna({"units_returned": 0, "returned_value": 0.0})
        .withColumn("revenue_share", F.col("revenue") / F.col("_total_revenue"))
        .withColumn(
            "monthly_units_cv",
            coefficient_of_variation(
                F.col("_sum_units"),
                F.col("_sum_sq_units"),
                F.col("months_on_sale"),
                min_months_for_cv,
            ),
        )
        .withColumn(
            "return_rate",
            F.least(F.col("units_returned") / F.col("units_sold"), F.lit(1.0)),
        )
        .withColumn("is_history_censored", F.col("first_sale_date") < censor_limit)
        .withColumn("as_of_date", _date(as_of))
    )
    types = {name: dtype for name, dtype, _ in PRODUCT_FEATURE_SCHEMA}
    return features.select([F.col(c).cast(types[c]).alias(c) for c in PRODUCT_FEATURE_COLUMNS])


# --------------------------------------------------------------------------- checks


def reconcile_product_features(
    features: DataFrame,
    sales: DataFrame,
    returns: DataFrame,
    as_of: date | str,
    min_months_for_cv: int = MIN_MONTHS_FOR_CV,
) -> dict:
    """Independent recounts that the notebook asserts on.

    - products vs distinct stock codes sold before `as_of`; duplicate keys must be 0
    - revenue_diff / units_diff: feature totals vs the same Silver lines (~0)
    - revenue_share_sum: must be 1
    - short_history vs null_cv: products with months_on_sale < min are exactly the NULL CVs
    - returns_only_products: returned but never sold before `as_of` (excluded)
    - return_rate_capped: products whose raw units_returned / units_sold exceeded 1
    """
    sold = lines_before(sales, as_of)
    f = features.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("stock_code").alias("distinct_products"),
        F.sum("revenue").alias("revenue"),
        F.sum("units_sold").alias("units"),
        F.sum("revenue_share").alias("share_sum"),
        F.sum(F.when(F.col("months_on_sale") < min_months_for_cv, 1).otherwise(0)).alias(
            "short_history"
        ),
        F.sum(F.when(F.col("monthly_units_cv").isNull(), 1).otherwise(0)).alias("null_cv"),
        F.sum(F.when(F.col("units_returned") > F.col("units_sold"), 1).otherwise(0)).alias(
            "capped"
        ),
        F.min("return_rate").alias("min_rr"),
        F.max("return_rate").alias("max_rr"),
    ).first()
    expected = sold.agg(
        F.countDistinct("stock_code").alias("products"),
        F.sum(F.col("line_revenue").cast("double")).alias("revenue"),
        F.sum("quantity").alias("units"),
    ).first()
    returns_only = (
        lines_before(returns, as_of)
        .select("stock_code")
        .distinct()
        .join(sold.select("stock_code").distinct(), "stock_code", "left_anti")
        .count()
    )
    return {
        "products": f["rows"],
        "expected_products": expected["products"],
        "duplicate_keys": f["rows"] - f["distinct_products"],
        "revenue_diff": round((f["revenue"] or 0.0) - (expected["revenue"] or 0.0), 6),
        "units_diff": int((f["units"] or 0) - (expected["units"] or 0)),
        "revenue_share_sum": round(f["share_sum"] or 0.0, 9),
        "short_history_products": f["short_history"],
        "null_cv": f["null_cv"],
        "returns_only_products": returns_only,
        "return_rate_capped": f["capped"],
        "return_rate_range": (f["min_rr"], f["max_rr"]),
    }


# --------------------------------------------------------------------------- long tail


def long_tail_table(
    features: DataFrame, min_invoices: Sequence[int] = (1, 2, 5, 10, 20, 50, 100)
) -> list[dict]:
    """For each threshold t: how many products have n_invoices >= t, and what share of
    products and revenue they hold. One aggregation; used to choose the clustering cut-off
    (products below it get a rule-based "long tail" segment instead of a cluster)."""
    aggs = [F.count(F.lit(1)).alias("_n"), F.sum("revenue").alias("_rev")]
    for t in min_invoices:
        keep = F.col("n_invoices") >= t
        aggs += [
            F.sum(keep.cast("int")).alias(f"n_{t}"),
            F.sum(F.when(keep, F.col("revenue")).otherwise(0.0)).alias(f"rev_{t}"),
        ]
    row = features.agg(*aggs).first()
    return [
        {
            "min_invoices": t,
            "products": int(row[f"n_{t}"]),
            "product_share": row[f"n_{t}"] / row["_n"],
            "revenue_share": row[f"rev_{t}"] / row["_rev"],
        }
        for t in min_invoices
    ]


def pareto_counts(
    features: DataFrame, revenue_shares: Sequence[float] = (0.5, 0.8, 0.9, 0.95)
) -> dict[float, int]:
    """Smallest number of top-revenue products that together reach each revenue share
    (e.g. {0.8: 812} = the top 812 products make 80 % of revenue).

    A product is needed for share s if the products ranked above it make less than s.
    One small window over ~4.7k rows (a single partition is fine at this size).
    """
    from pyspark.sql import Window

    order = Window.orderBy(F.col("revenue").desc(), F.col("stock_code"))
    above = F.sum("revenue").over(order.rowsBetween(Window.unboundedPreceding, -1))
    total = F.sum("revenue").over(Window.partitionBy(F.lit(1)))
    ranked = features.select((F.coalesce(above, F.lit(0.0)) / total).alias("_share_above"))
    row = ranked.agg(
        *[
            F.sum((F.col("_share_above") < F.lit(s)).cast("int")).alias(f"s{i}")
            for i, s in enumerate(revenue_shares)
        ]
    ).first()
    return {s: int(row[f"s{i}"]) for i, s in enumerate(revenue_shares)}
