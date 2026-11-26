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

# ML (Phase 3+) ------------------------------------------------------------------------
# Customer feature table: one row per (customer_id, as_of_date). The primary key makes it a
# feature table in Unity Catalog; as_of_date is a TIMESERIES key so several snapshots can live
# side by side (Phase 5 needs features "as of" a cutoff date, without leakage).
ML_CUSTOMER_FEATURES = fq("ml", "customer_features")
GOLD_CUSTOMER_SEGMENTS = fq("gold", "customer_segments")

# MLflow experiment for customer segmentation, created under the user's workspace folder.
SEGMENTATION_EXPERIMENT_NAME = "retail_customer_segmentation"


def user_experiment_path(user_email: str, name: str = SEGMENTATION_EXPERIMENT_NAME) -> str:
    """Workspace path of an MLflow experiment in the user's home folder,
    e.g. /Users/me@example.com/retail_customer_segmentation."""
    return f"/Users/{user_email}/{name}"


# Products (Phase 4) --------------------------------------------------------------------
# Product feature table: one row per (stock_code, as_of_date), same feature-table pattern as
# customers (PK with a TIMESERIES key, one snapshot per as_of_date).
ML_PRODUCT_FEATURES = fq("ml", "product_features")
GOLD_PRODUCT_SEGMENTS = fq("gold", "product_segments")
GOLD_PRODUCT_ASSOCIATIONS = fq("gold", "product_associations")

# One MLflow experiment for everything about products: clustering sweep/stability/final model
# and the market-basket runs (thresholds, itemset counts, rules).
PRODUCT_EXPERIMENT_NAME = "retail_product_intelligence"


# Benchmarks (Phase 2) -----------------------------------------------------------------
# A separate scratch schema keeps experiment tables out of the production layers: every
# benchmark copy can be dropped with one `DROP SCHEMA retail.bench CASCADE` at the end.
BENCH_SCHEMA = "bench"
BENCHMARK_RUNS = fq(BENCH_SCHEMA, "benchmark_runs")
# Per-execution I/O of read benchmarks, copied from system.query.history (files read/pruned).
BENCHMARK_QUERY_METRICS = fq(BENCH_SCHEMA, "benchmark_query_metrics")
# One isolated execution per (variant, query) with its time window and query tag.
BENCHMARK_IO_PROBES = fq(BENCH_SCHEMA, "benchmark_io_probes")


def bench_table(prefix: str, logical_name: str) -> str:
    """Benchmark copy of a table, e.g. bench_table("x1_baseline", "silver_sales")
    -> retail.bench.x1_baseline__silver_sales (double underscore separates the two parts)."""
    return fq(BENCH_SCHEMA, f"{prefix}__{logical_name}")


def bench_bronze(factor: int) -> str:
    """Synthetic scaled-up Bronze copy, e.g. retail.bench.bronze_x10 (see optimisation.scaling)."""
    return fq(BENCH_SCHEMA, f"bronze_x{factor}")
