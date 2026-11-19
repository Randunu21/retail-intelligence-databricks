"""Synthetic scale-up of Bronze for benchmarks (Phase 2.2).

Why: at x1 every Silver/Gold table is 1–2 files, so partitioning, clustering and file
skipping have nothing to skip, and each build is dominated by a fixed ~2.5 s overhead
(the baseline's 739-row dim_date takes as long as the 1M-row fact table). A bigger copy
shows how the same pipeline behaves when the data stops fitting in a single file.

How: every Bronze row is copied `factor` times. Copy 0 is the real data, untouched. Copies
1..factor-1 get shifted keys so they are NEW customers and NEW invoices, not duplicates
(otherwise Silver's de-dup would just delete them):

- `Customer ID` + replica * 100,000 (real IDs are 12346..18287, so ranges never overlap;
  the largest value at x20 is 1,918,287, well inside INT). Empty IDs stay empty.
- `Invoice` gets "R<nn>-" after its letter prefix: 489434 -> R03-489434,
  C489449 -> CR03-489449. The leading "C" (cancellation) and "A" (adjustment) are kept,
  so Silver routes the rows exactly like the originals.
- Everything else (stock codes, prices, dates, countries, `_source_file`) is unchanged,
  so the copy has the same products, calendar and data-quality problems as the real data.

Consequences we can check (notebook §4a): every Silver/fact row count is exactly
factor × x1, customers are factor × 5,852 (+ Unknown), while products (4,725), days (739),
monthly-country rows (562) and cohort months (325 grid rows) stay the same.

Honesty rule: anything measured on this table is labelled "synthetic xN replica". It is
the real data repeated with relabelled keys, not new behaviour: no new products, seasons or
customer patterns.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

REPLICA_CUSTOMER_OFFSET = 100_000
MAX_FACTOR = 20  # keeps shifted IDs < 2,000,000 and the run inside Free Edition quotas


def _shift_invoice(invoice: Column, replica: Column) -> Column:
    prefix = F.regexp_extract(invoice, r"^([A-Za-z]*)", 1)
    rest = F.regexp_extract(invoice, r"^[A-Za-z]*(.*)$", 1)
    tag = F.concat(F.lit("R"), F.lpad(replica.cast("string"), 2, "0"), F.lit("-"))
    return F.when(replica == 0, invoice).otherwise(F.concat(prefix, tag, rest))


def _shift_customer(customer_id: Column, replica: Column) -> Column:
    as_int = F.trim(customer_id).try_cast("int")
    shifted = (as_int + replica * F.lit(REPLICA_CUSTOMER_OFFSET)).cast("string")
    # Replica 0, empty and non-numeric IDs are left exactly as they were.
    return F.when((replica == 0) | as_int.isNull(), customer_id).otherwise(shifted)


def replicate_bronze(bronze: DataFrame, factor: int) -> DataFrame:
    """Return `factor` copies of Bronze with shifted keys (copy 0 = the original rows).

    Output has the Bronze columns in the same order plus `_replica` (0..factor-1), which
    Silver ignores (it selects its columns explicitly) but keeps the copy traceable.
    """
    if not 1 <= factor <= MAX_FACTOR:
        raise ValueError(f"factor must be between 1 and {MAX_FACTOR}")
    replica = F.col("_replica")
    exploded = bronze.withColumn("_replica", F.explode(F.sequence(F.lit(0), F.lit(factor - 1))))
    return exploded.select(
        *[
            _shift_invoice(F.col(c), replica).alias(c)
            if c == "Invoice"
            else _shift_customer(F.col(f"`{c}`"), replica).alias(c)
            if c == "Customer ID"
            else F.col(f"`{c}`")
            for c in bronze.columns
        ],
        replica,
    )


# Expected x-factor of each table's row count after scaling (used by the notebook check).
# "factor": grows linearly; 1: unchanged; "customers": factor * known customers + Unknown.
SCALING_EXPECTATIONS: dict[str, str] = {
    "silver_sales": "factor",
    "silver_returns": "factor",
    "gold_fact_sales": "factor",
    "gold_dim_customer": "customers",
    "gold_dim_date": "same",
    "gold_dim_product": "same",
    "gold_daily_sales": "same",
    "gold_monthly_country_sales": "same",
    "gold_product_performance": "same",
    "gold_customer_cohort_retention": "same",
}


def expected_rows(table: str, x1_rows: int, factor: int) -> int:
    """Row count a scaled table must have, given its x1 row count."""
    rule = SCALING_EXPECTATIONS[table]
    if rule == "factor":
        return x1_rows * factor
    if rule == "customers":  # x1_rows includes the single Unknown (-1) row
        return (x1_rows - 1) * factor + 1
    return x1_rows
