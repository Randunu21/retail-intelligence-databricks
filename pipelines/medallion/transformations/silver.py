"""Lakeflow pipeline source: Silver layer (typed, cleaned, de-duplicated) with expectations.

Datasets defined here:
- silver_staged       temporary view: Bronze typed + flags + _dup_rank (nothing removed)
- sales_validated     temporary view: SALES_DROP_RULES (drop) + SALES_WARN_RULES (warn)
- returns_validated   temporary view: cancellations only, RETURNS_* rules
- retail.silver.sales / retail.silver.returns   materialized views (final columns only)

Expectations sit on the `*_validated` views because they are evaluated on the dataset's
output, and the helper columns they need (_dup_rank, is_cancellation, ...) must not end up
in the final tables. Each rule's pass/fail counts go to the pipeline event log, which is
what retail.silver.dq_log is built from (step 1.3).
"""

import sys

from pyspark import pipelines as dp

_SRC = spark.conf.get("retail.src_path")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from common.tables import BRONZE_TRANSACTIONS, SILVER_RETURNS, SILVER_SALES
from transformations.silver import (
    RETURNS_DROP_RULES,
    RETURNS_WARN_RULES,
    SALES_DROP_RULES,
    SALES_WARN_RULES,
    SILVER_TABLE_PROPERTIES,
    stage_silver,
    to_silver_output,
)


@dp.temporary_view(comment="Bronze rows typed + snake_case + flags + _dup_rank. Nothing removed.")
def silver_staged():
    # Batch read of the Bronze streaming table: de-duplication needs to see all rows.
    return stage_silver(spark.read.table(BRONZE_TRANSACTIONS))


@dp.temporary_view(comment="Sales candidates after the Silver drop rules.")
@dp.expect_all_or_drop(SALES_DROP_RULES)
@dp.expect_all(SALES_WARN_RULES)
def sales_validated():
    return spark.read.table("silver_staged")


@dp.materialized_view(
    name=SILVER_SALES,
    comment="Clean product sales lines (no cancellations, duplicates, non-products or invalid "
    "price/quantity). Rows without customer_id are kept with is_missing_customer = true.",
    table_properties=SILVER_TABLE_PROPERTIES,
)
def sales():
    return to_silver_output(spark.read.table("sales_validated"))


@dp.temporary_view(comment="Cancellation lines (invoice starts with 'C') after the drop rules.")
@dp.expect_all_or_drop(RETURNS_DROP_RULES)
@dp.expect_all(RETURNS_WARN_RULES)
def returns_validated():
    return spark.read.table("silver_staged").filter("is_cancellation")


@dp.materialized_view(
    name=SILVER_RETURNS,
    comment="Cancelled/returned product lines. quantity and line_revenue are negative.",
    table_properties=SILVER_TABLE_PROPERTIES,
)
def returns():
    return to_silver_output(spark.read.table("returns_validated"))
