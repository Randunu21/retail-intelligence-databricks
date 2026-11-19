"""Fully qualified Unity Catalog names for everything the pipeline reads or writes.

One place to change names (e.g. a different catalog for the Azure phase), and no string
literals scattered through pipeline files, notebooks and tests.
"""

from __future__ import annotations

CATALOG = "retail"

# Raw files in the Volume ------------------------------------------------------------
RAW_DATASET_DIR = f"/Volumes/{CATALOG}/bronze/raw/online_retail_ii"
RAW_CSV_DIR = f"{RAW_DATASET_DIR}/csv/"
RAW_MANIFEST_PATH = f"{RAW_DATASET_DIR}/manifest.json"


def fq(schema: str, table: str, catalog: str = CATALOG) -> str:
    """Return the three-level name `catalog.schema.table`."""
    return f"{catalog}.{schema}.{table}"


# Bronze -----------------------------------------------------------------------------
BRONZE_TRANSACTIONS = fq("bronze", "online_retail_transactions")

# Silver -----------------------------------------------------------------------------
SILVER_SALES = fq("silver", "sales")
SILVER_RETURNS = fq("silver", "returns")
SILVER_DQ_LOG = fq("silver", "dq_log")
PIPELINE_EVENT_LOG = fq("silver", "medallion_event_log")  # published by the pipeline

# Gold -------------------------------------------------------------------------------
GOLD_FACT_SALES = fq("gold", "fact_sales")
GOLD_DIM_CUSTOMER = fq("gold", "dim_customer")
GOLD_DIM_PRODUCT = fq("gold", "dim_product")
GOLD_DIM_DATE = fq("gold", "dim_date")
GOLD_DAILY_SALES = fq("gold", "daily_sales")
GOLD_MONTHLY_COUNTRY_SALES = fq("gold", "monthly_country_sales")
GOLD_PRODUCT_PERFORMANCE = fq("gold", "product_performance")
GOLD_CUSTOMER_COHORT_RETENTION = fq("gold", "customer_cohort_retention")

# Benchmarks (Phase 2) -----------------------------------------------------------------
# A separate scratch schema keeps experiment tables out of the production layers: every
# benchmark copy can be dropped with one `DROP SCHEMA retail.bench CASCADE` at the end.
BENCH_SCHEMA = "bench"
BENCHMARK_RUNS = fq(BENCH_SCHEMA, "benchmark_runs")


def bench_table(prefix: str, logical_name: str) -> str:
    """Benchmark copy of a table, e.g. bench_table("x1_baseline", "silver_sales")
    -> retail.bench.x1_baseline__silver_sales (double underscore separates the two parts)."""
    return fq(BENCH_SCHEMA, f"{prefix}__{logical_name}")
