"""Silver layer: typed, cleaned, de-duplicated Online Retail II rows.

Everything here is a pure function (DataFrame in -> DataFrame out) or a constant, so it is
unit-tested locally with pytest and reused unchanged by the Lakeflow pipeline file
`pipelines/medallion/transformations/silver.py`.

Flow inside the pipeline:

    bronze ──stage_silver()──> silver_staged (temporary view: typed + flags + _dup_rank)
                                   │
              ┌────────────────────┴────────────────────┐
     sales_validated (view)                    returns_validated (view, cancellations only)
     expect_all_or_drop(SALES_DROP_RULES)      expect_all_or_drop(RETURNS_DROP_RULES)
     expect_all(SALES_WARN_RULES)              expect_all(RETURNS_WARN_RULES)
              │                                         │
     retail.silver.sales (MV)                  retail.silver.returns (MV)

The rules are plain SQL strings in dicts, the same format as Lakeflow expectations, so the
tests apply exactly the conditions the pipeline uses (see `apply_drop_rules`).
"""

from __future__ import annotations

from collections.abc import Mapping

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# ----------------------------------------------------------------------------------------
# Non-product stock codes, found by profiling the data (see `stock_code_profile` and the
# notebook): codes that don't follow the normal product pattern of 5 digits + optional
# letters were inspected one by one. Matching is done AFTER trim + upper-case.
#
# Removed (not product sales):
#   POST, DOT, C2, C3      postage / carriage           BANK CHARGES, AMAZONFEE   fees
#   M (and lowercase m)    manual adjustments           CRUK                      charity commission
#   D                      discount                     B                         "Adjust bad debt" (A-invoices)
#   S                      samples                      GIFT, GIFT_0001_<n>       gift vouchers
#   ADJUST, ADJUST2        manual adjustments           TEST001, TEST002          test products
# Kept although non-standard (real products): DCGS* (dotcomgiftshop items), PADS, SP1002.
# ----------------------------------------------------------------------------------------
NON_PRODUCT_CODES: frozenset[str] = frozenset(
    {"POST", "DOT", "C2", "C3", "M", "D", "S", "B", "BANK CHARGES", "AMAZONFEE", "CRUK", "GIFT"}
)
NON_PRODUCT_PATTERNS: tuple[str, ...] = (r"^GIFT_0001_\d+$", r"^TEST\d+$", r"^ADJUST\d*$")

# Normal product code: 5 digits + up to 2 letters (e.g. 85123A, 15056BL).
STANDARD_STOCK_CODE_PATTERN = r"^\d{5}[A-Z]{0,2}$"

# Source timestamps are UK wall-clock times with no zone -> TIMESTAMP_NTZ (no tz shifts).
INVOICE_TS_FORMAT = "yyyy-MM-dd HH:mm:ss"

# Columns that identify a sales line. Two rows equal on all of these are duplicates.
BUSINESS_COLUMNS: list[str] = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_ts",
    "price",
    "customer_id",
    "country",
]

# Final columns of retail.silver.sales and retail.silver.returns.
SILVER_OUTPUT_COLUMNS: list[str] = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "price",
    "line_revenue",
    "invoice_ts",
    "invoice_date",
    "customer_id",
    "is_missing_customer",
    "country",
    "_source_file",
    "_ingested_at",
]

# ----------------------------------------------------------------------------------------
# Data-quality rules (Lakeflow expectation format: {name: SQL condition that must be TRUE}).
# Conditions are written NULL-safe ("x IS NOT NULL AND ...") so a NULL never slips through.
# ----------------------------------------------------------------------------------------
SALES_DROP_RULES: dict[str, str] = {
    "not_duplicate": "_dup_rank = 1",
    "not_cancellation": "NOT is_cancellation",  # routed to silver.returns instead
    "is_product": "NOT is_non_product",
    "valid_invoice_ts": "invoice_ts IS NOT NULL",
    "positive_quantity": "quantity IS NOT NULL AND quantity > 0",
    "positive_price": "price IS NOT NULL AND price > 0",
}
SALES_WARN_RULES: dict[str, str] = {
    "has_customer_id": "customer_id IS NOT NULL",  # kept, flagged by is_missing_customer
}

RETURNS_DROP_RULES: dict[str, str] = {
    "not_duplicate": "_dup_rank = 1",
    "is_product": "NOT is_non_product",
    "valid_invoice_ts": "invoice_ts IS NOT NULL",
    "valid_quantity": "quantity IS NOT NULL",
    "positive_price": "price IS NOT NULL AND price > 0",
}
RETURNS_WARN_RULES: dict[str, str] = {
    "negative_quantity": "quantity < 0",
    "has_customer_id": "customer_id IS NOT NULL",
}


# ----------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------
def _clean_text(col: Column) -> Column:
    """Trim, collapse repeated whitespace, and turn empty strings into NULL."""
    cleaned = F.trim(F.regexp_replace(col, r"\s+", " "))
    return F.when(cleaned == "", F.lit(None)).otherwise(cleaned)


def is_non_product_code(stock_code: Column) -> Column:
    """TRUE for non-product codes (expects a trimmed, upper-cased code). NULL code -> TRUE."""
    pattern = "|".join(NON_PRODUCT_PATTERNS)
    flag = stock_code.isin(*sorted(NON_PRODUCT_CODES)) | stock_code.rlike(pattern)
    return F.coalesce(flag, F.lit(True))


def standardise(bronze: DataFrame) -> DataFrame:
    """Rename to snake_case and cast to proper types.

    `try_cast` / `try_to_timestamp` return NULL instead of failing on bad values (serverless
    runs with ANSI mode on, where a plain cast would fail the whole query). The NULLs are
    then caught and counted by the `valid_*` / `positive_*` expectations.
    """
    return bronze.select(
        F.trim("Invoice").alias("invoice"),
        F.upper(F.trim("StockCode")).alias("stock_code"),
        _clean_text(F.col("Description")).alias("description"),
        F.trim("Quantity").try_cast("int").alias("quantity"),
        F.try_to_timestamp(F.trim("InvoiceDate"), F.lit(INVOICE_TS_FORMAT))
        .cast("timestamp_ntz")
        .alias("invoice_ts"),
        F.trim("Price").try_cast("decimal(10,3)").alias("price"),
        F.trim("`Customer ID`").try_cast("int").alias("customer_id"),
        _clean_text(F.col("Country")).alias("country"),
        "_source_file",
        "_ingested_at",
    )


def add_derived_columns(df: DataFrame) -> DataFrame:
    """Add flags and computed columns used by the rules and the output tables."""
    return (
        df.withColumn("is_cancellation", F.coalesce(F.col("invoice").startswith("C"), F.lit(False)))
        .withColumn("is_non_product", is_non_product_code(F.col("stock_code")))
        .withColumn("is_missing_customer", F.col("customer_id").isNull())
        .withColumn("invoice_date", F.to_date("invoice_ts"))
        .withColumn("line_revenue", (F.col("quantity") * F.col("price")).cast("decimal(18,3)"))
    )


def add_duplicate_rank(df: DataFrame) -> DataFrame:
    """Number identical rows: `_dup_rank = 1` is kept, later copies are duplicates.

    Equality is on the business columns only (not `_source_file`), so the 2010-12-01..09
    rows that appear in both source sheets are caught. Ordering by `_source_file` keeps the
    copy from the earlier sheet, which makes the result deterministic.
    """
    w = Window.partitionBy(*BUSINESS_COLUMNS).orderBy("_source_file")
    return df.withColumn("_dup_rank", F.row_number().over(w))


def stage_silver(bronze: DataFrame) -> DataFrame:
    """Bronze -> typed rows with all flags and `_dup_rank` (nothing removed yet)."""
    return add_duplicate_rank(add_derived_columns(standardise(bronze)))


def to_silver_output(df: DataFrame) -> DataFrame:
    """Select the final Silver columns (drops helper columns such as `_dup_rank`)."""
    return df.select(*SILVER_OUTPUT_COLUMNS)


def rules_condition(rules: Mapping[str, str]) -> str:
    """Combine rule conditions into one SQL predicate: (c1) AND (c2) AND ..."""
    return " AND ".join(f"({cond})" for cond in rules.values())


def apply_drop_rules(df: DataFrame, rules: Mapping[str, str]) -> DataFrame:
    """Keep only rows that pass every rule (what `expect_all_or_drop` does in the pipeline)."""
    return df.filter(rules_condition(rules))


def count_rule_failures(df: DataFrame, rules: Mapping[str, str]) -> DataFrame:
    """One row with, per rule, the number of rows that FAIL it (counted independently).

    Same semantics as the pipeline's expectation metrics; handy for checks and tests.
    """
    return df.agg(
        F.count(F.lit(1)).alias("input_rows"),
        *[
            F.sum(F.when(F.expr(f"coalesce({cond}, false)"), 0).otherwise(1)).alias(name)
            for name, cond in rules.items()
        ],
    )


def stock_code_profile(bronze: DataFrame) -> DataFrame:
    """Profile stock codes that don't match the standard product pattern.

    Used to FIND the non-product list from the data instead of guessing it: one row per
    non-standard code with its row count, most common description, price range, how many
    rows have a customer, and whether our rules treat it as a non-product.
    """
    code = F.upper(F.trim("StockCode"))
    price = F.trim("Price").try_cast("decimal(10,3)")
    return (
        bronze.withColumn("stock_code", code)
        .filter(~F.col("stock_code").rlike(STANDARD_STOCK_CODE_PATTERN))
        .groupBy("stock_code")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            F.mode(_clean_text(F.col("Description"))).alias("top_description"),
            F.min(price).alias("min_price"),
            F.max(price).alias("max_price"),
            F.count("Customer ID").alias("rows_with_customer"),
        )
        .withColumn("treated_as_non_product", is_non_product_code(F.col("stock_code")))
        .orderBy(F.desc("rows"))
    )
