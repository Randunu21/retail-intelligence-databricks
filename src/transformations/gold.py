"""Gold layer (DataFrame API part): dimensions whose logic needs window functions + tests.

Star schema (Phase 1.4):

                    dim_date (SQL)
                        │ date_key
    dim_customer ── fact_sales (SQL) ── dim_product
     customer_key       │                stock_code
                        └─ grain: one row per de-duplicated invoice line of silver.sales

Split between SQL and the DataFrame API (both run in the same Lakeflow pipeline):
- SQL (`pipelines/.../gold_fact_sales.sql`, `gold_dim_date.sql`): straight projections and
  calendar generation. SQL is the most readable form for those.
- DataFrame API (this module): `dim_product` and `dim_customer`, which pick ONE value out of
  many (which description? which country?) with ranked windows and explicit tie-breaks.
  That logic has edge cases, so it lives in tested Python functions.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# Gold tables carry TIMESTAMP_NTZ columns too -> same Delta table feature as Silver.
GOLD_TABLE_PROPERTIES: dict[str, str] = {"delta.feature.timestampNtz": "supported"}

# Sales without a customer_id point to this "Unknown customer" row instead of NULL,
# so every fact row joins to dim_customer (no NULL foreign keys in the star schema).
UNKNOWN_CUSTOMER_KEY = -1
UNKNOWN_COUNTRY = "Unknown"

DIM_PRODUCT_COLUMNS: list[str] = [
    "stock_code",
    "description",
    "n_description_variants",
    "median_unit_price",
    "first_sold_date",
    "last_sold_date",
]

DIM_CUSTOMER_COLUMNS: list[str] = [
    "customer_key",
    "customer_id",
    "country",
    "n_countries",
    "first_purchase_ts",
    "last_purchase_ts",
    "cohort_month",
    "is_unknown",
]


def _pick_most_frequent(
    df: DataFrame, key: str, value: str, out_name: str, extra_order: list[Column] | None = None
) -> DataFrame:
    """For each `key`, the `value` that occurs most often (NULL values ignored).

    Ties are broken by the most recent `invoice_ts`, then alphabetically, so the result is
    deterministic. Returns columns (key, out_name).
    """
    counts = (
        df.filter(F.col(value).isNotNull())
        .groupBy(key, value)
        .agg(F.count(F.lit(1)).alias("_n"), F.max("invoice_ts").alias("_last_seen"))
    )
    order = [F.desc("_n"), F.desc("_last_seen"), *(extra_order or []), F.asc(value)]
    w = Window.partitionBy(key).orderBy(*order)
    return (
        counts.withColumn("_rank", F.row_number().over(w))
        .filter("_rank = 1")
        .select(key, F.col(value).alias(out_name))
    )


def build_dim_product(sales: DataFrame) -> DataFrame:
    """One row per stock_code.

    The same stock_code appears with several descriptions (typos, renames, case changes).
    Resolution rule: the **most frequent** description across sales lines; ties -> the most
    recently used one. `n_description_variants` records how many variants existed.
    """
    description = _pick_most_frequent(sales, "stock_code", "description", "description")
    stats = sales.groupBy("stock_code").agg(
        F.countDistinct("description").alias("n_description_variants"),
        F.percentile_approx("price", 0.5).alias("median_unit_price"),
        F.min("invoice_date").alias("first_sold_date"),
        F.max("invoice_date").alias("last_sold_date"),
    )
    return stats.join(description, "stock_code", "left").select(*DIM_PRODUCT_COLUMNS)


def build_dim_customer(sales: DataFrame) -> DataFrame:
    """One row per known customer + one 'Unknown' row (customer_key = -1).

    `country`: a few customers bought from more than one country; we keep the most frequent
    one (ties -> most recent) and record `n_countries`. The sale's own country stays on
    fact_sales, so nothing is lost.
    `cohort_month`: first day of the month of the first purchase (used for retention).
    """
    known = sales.filter(F.col("customer_id").isNotNull())
    country = _pick_most_frequent(known, "customer_id", "country", "country")
    stats = known.groupBy("customer_id").agg(
        F.countDistinct("country").alias("n_countries"),
        F.min("invoice_ts").alias("first_purchase_ts"),
        F.max("invoice_ts").alias("last_purchase_ts"),
    )
    customers = (
        stats.join(country, "customer_id", "left")
        .withColumn("customer_key", F.col("customer_id"))
        .withColumn("cohort_month", F.trunc(F.to_date("first_purchase_ts"), "month"))
        .withColumn("is_unknown", F.lit(False))
        .select(*DIM_CUSTOMER_COLUMNS)
    )
    unknown = sales.sparkSession.range(1).select(
        F.lit(UNKNOWN_CUSTOMER_KEY).cast("int").alias("customer_key"),
        F.lit(None).cast("int").alias("customer_id"),
        F.lit(UNKNOWN_COUNTRY).alias("country"),
        F.lit(None).cast("bigint").alias("n_countries"),
        F.lit(None).cast("timestamp_ntz").alias("first_purchase_ts"),
        F.lit(None).cast("timestamp_ntz").alias("last_purchase_ts"),
        F.lit(None).cast("date").alias("cohort_month"),
        F.lit(True).alias("is_unknown"),
    )
    return customers.unionByName(unknown)
