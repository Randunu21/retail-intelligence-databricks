# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Medallion pipeline: checks, DQ log and explanations
# MAGIC
# MAGIC The tables are built by the **Lakeflow pipeline** `retail_medallion`
# MAGIC (source: `pipelines/medallion/transformations/`). This notebook runs **after** the
# MAGIC pipeline and:
# MAGIC 1. reconciles each layer against the expected numbers,
# MAGIC 2. builds `retail.silver.dq_log` from the pipeline event log (added in a later step),
# MAGIC 3. explains what each layer does and why.
# MAGIC
# MAGIC | Layer | Purpose | Tables |
# MAGIC |---|---|---|
# MAGIC | **Bronze** | Raw rows as delivered + lineage. Never filtered. | `retail.bronze.online_retail_transactions` |
# MAGIC | **Silver** | Typed, cleaned, de-duplicated. Every removed/flagged row is counted. | `retail.silver.sales`, `retail.silver.returns`, `retail.silver.dq_log` |
# MAGIC | **Gold** | Business-ready star schema + aggregates. | `retail.gold.*` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Make `src/` importable

# COMMAND ----------

import sys
from pathlib import Path

SRC_DIR = str(Path.cwd().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
print("src on path:", SRC_DIR)

# COMMAND ----------

import json

from pyspark.sql import functions as F

from common.tables import (
    BRONZE_TRANSACTIONS,
    PIPELINE_EVENT_LOG,
    RAW_MANIFEST_PATH,
    SILVER_DQ_LOG,
    SILVER_RETURNS,
    SILVER_SALES,
)
from ingestion.raw_reader import RAW_COLUMNS
from transformations.dq_log import build_dq_log, output_rows_per_flow, rules_frame
from transformations.silver import (
    BUSINESS_COLUMNS,
    RETURNS_DROP_RULES,
    RETURNS_WARN_RULES,
    SALES_DROP_RULES,
    SALES_WARN_RULES,
    count_rule_failures,
    stage_silver,
    stock_code_profile,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze
# MAGIC
# MAGIC **What the pipeline does:** Auto Loader (`cloudFiles`) reads every CSV in the Volume with
# MAGIC the fixed all-string schema, adds `_source_file` and `_ingested_at`, and appends to a
# MAGIC **streaming table**.
# MAGIC
# MAGIC **Why keep Bronze raw?** It is the replayable copy of the source. If a Silver rule is
# MAGIC wrong, we fix the rule and rebuild Silver from Bronze; we never re-convert the source.
# MAGIC Nothing is de-duplicated here on purpose: the two sheets **overlap** (2010-12-01 to
# MAGIC 2010-12-09 is in both files). Silver fixes that and counts it.
# MAGIC
# MAGIC **Why Auto Loader + streaming table?** Auto Loader keeps a checkpoint of the files it has
# MAGIC already loaded, so re-running the pipeline does **not** load them again (idempotent), and a
# MAGIC new file (e.g. a 2012 sheet) would be loaded on its own. That's incremental ingestion.
# MAGIC
# MAGIC **Delta detail:** `Customer ID` contains a space, which Delta only allows with
# MAGIC **column mapping** (`delta.columnMapping.mode = name`).

# COMMAND ----------

bronze = spark.table(BRONZE_TRANSACTIONS)
bronze.printSchema()

expected_cols = [*RAW_COLUMNS, "_source_file", "_ingested_at"]
extra_cols = [c for c in bronze.columns if c not in expected_cols]
print("Unexpected columns (should be empty):", extra_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bronze reconciliation against the manifest
# MAGIC Bronze must hold exactly the rows in the source files, per file and in total.

# COMMAND ----------

with open(RAW_MANIFEST_PATH) as fh:
    manifest = json.load(fh)
expected_per_file = {s["csv_file"]: s["rows"] for s in manifest["sheets"]}

bronze_counts_df = (
    bronze.groupBy("_source_file")
    .agg(
        F.count(F.lit(1)).alias("rows"),
        F.min("_ingested_at").alias("first_ingested_at"),
        F.max("_ingested_at").alias("last_ingested_at"),
    )
    .orderBy("_source_file")
)
display(bronze_counts_df)

bronze_counts = {r["_source_file"]: r["rows"] for r in bronze_counts_df.collect()}
assert bronze_counts == expected_per_file, f"{bronze_counts} != {expected_per_file}"
print(f"✅ Bronze rows: {sum(bronze_counts.values()):,} (manifest: {manifest['total_rows']:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Silver
# MAGIC
# MAGIC **What belongs in Silver:** one clean, typed, trustworthy version of each record:
# MAGIC snake_case names, real types, no duplicates, no rows that aren't product sales. No business
# MAGIC aggregation yet (that's Gold).
# MAGIC
# MAGIC **How the pipeline builds it** (`pipelines/medallion/transformations/silver.py`):
# MAGIC
# MAGIC | Dataset | Type | What it does |
# MAGIC |---|---|---|
# MAGIC | `silver_staged` | temporary view | Bronze → types, cleaning, flags, `_dup_rank`. Nothing removed. |
# MAGIC | `sales_validated` | temporary view | **expectations**: drop rules + warn rules |
# MAGIC | `retail.silver.sales` | materialized view | final columns only |
# MAGIC | `returns_validated` | temporary view | cancellations only + their expectations |
# MAGIC | `retail.silver.returns` | materialized view | final columns only |
# MAGIC
# MAGIC **Key decisions (interview talking points)**
# MAGIC - **Cancellations are kept, in their own table**, not dropped: return rates per product/customer
# MAGIC   are needed in Phases 3–4, and dropping them would silently lose that signal. They are
# MAGIC   excluded from `sales` so revenue isn't netted in an untraceable way.
# MAGIC - **Missing `customer_id` is flagged, not dropped** (`is_missing_customer`, warn expectation):
# MAGIC   those rows are real product sales (needed for product analysis and revenue totals) but are
# MAGIC   excluded later from customer analysis.
# MAGIC - **De-duplication on business columns, not on all columns:** the two source sheets overlap
# MAGIC   (2010-12-01..09), so the same line appears with a different `_source_file`.
# MAGIC - **`try_cast` instead of `cast`:** serverless runs in ANSI mode, where one bad value would
# MAGIC   fail the whole run. `try_cast` gives NULL, and a NULL-safe expectation drops and *counts* it.
# MAGIC - **`TIMESTAMP_NTZ`:** the source times are UK wall-clock times without a zone; NTZ stores
# MAGIC   them as-is, so dates never shift with the session time zone.
# MAGIC - **`DECIMAL` for money:** exact arithmetic for `line_revenue = quantity * price` (no float
# MAGIC   rounding when summing ~1M lines).

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. How the non-product stock-code list was found
# MAGIC Normal product codes are 5 digits + up to 2 letters. Everything else is listed below with
# MAGIC its description, price range and row count, and was classified by hand. Some odd-looking
# MAGIC codes are real products (`DCGS*` = dotcomgiftshop items, `PADS`, `SP1002`), so a simple
# MAGIC "not 5 digits → drop" rule would have been wrong.

# COMMAND ----------

display(stock_code_profile(bronze))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. The rules (exactly what the pipeline's expectations use)

# COMMAND ----------

for title, rules in [
    ("sales: DROP", SALES_DROP_RULES),
    ("sales: WARN (kept, counted)", SALES_WARN_RULES),
    ("returns: DROP", RETURNS_DROP_RULES),
    ("returns: WARN (kept, counted)", RETURNS_WARN_RULES),
]:
    print(f"\n{title}")
    for name, cond in rules.items():
        print(f"  {name:<20} {cond}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c. Silver row counts

# COMMAND ----------

sales = spark.table(SILVER_SALES)
returns = spark.table(SILVER_RETURNS)

silver_counts = sales.agg(
    F.count(F.lit(1)).alias("sales_rows"),
    F.sum(F.col("is_missing_customer").cast("int")).alias("sales_missing_customer"),
    F.sum("line_revenue").alias("sales_revenue"),
).crossJoin(
    returns.agg(
        F.count(F.lit(1)).alias("returns_rows"),
        F.sum("line_revenue").alias("returns_value"),
    )
)
display(silver_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d. Invariants: things that must be true of Silver
# MAGIC If any of these fail, a rule is wrong. The pipeline would still "succeed", which is
# MAGIC exactly why we check.

# COMMAND ----------

checks = (
    sales.agg(
        F.sum(F.col("invoice").startswith("C").cast("int")).alias("cancellations_in_sales"),
        F.sum((F.col("price") <= 0).cast("int")).alias("non_positive_price"),
        F.sum((F.col("quantity") <= 0).cast("int")).alias("non_positive_quantity"),
        F.sum(F.col("invoice_ts").isNull().cast("int")).alias("null_invoice_ts"),
    )
    .collect()[0]
    .asDict()
)
checks["duplicate_groups_in_sales"] = (
    sales.groupBy(*BUSINESS_COLUMNS).count().filter("count > 1").count()
)
checks["non_cancellations_in_returns"] = returns.filter(~F.col("invoice").startswith("C")).count()
print(checks)
assert all(v == 0 for v in checks.values()), checks
print("✅ Silver invariants hold")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2e. Recomputed rule counts (cross-check for the DQ log)
# MAGIC The same rules applied here, outside the pipeline, to the Bronze rows. Each rule is
# MAGIC counted **independently** (a POST line with price 0 fails `is_product` *and*
# MAGIC `positive_price`), the same way Lakeflow counts expectations. In step 1.3 these
# MAGIC numbers must match the pipeline's event log.

# COMMAND ----------

staged = stage_silver(bronze)
display(count_rule_failures(staged, {**SALES_DROP_RULES, **SALES_WARN_RULES}))

# COMMAND ----------

display(
    count_rule_failures(
        staged.filter("is_cancellation"), {**RETURNS_DROP_RULES, **RETURNS_WARN_RULES}
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data-quality log: `retail.silver.dq_log`
# MAGIC
# MAGIC **Where the numbers come from:** every pipeline run writes `flow_progress` events to the
# MAGIC event log (published as `retail.silver.medallion_event_log`). For each expectation they
# MAGIC hold `passed_records` and `failed_records`. `build_dq_log` turns them into one row per
# MAGIC **run × dataset × rule** and adds the rule's **action** (`drop` / `warn`) and SQL
# MAGIC **condition** from the same Python rule dictionaries the pipeline uses.
# MAGIC
# MAGIC **Why rebuilt from the full event log every time (`CREATE OR REPLACE`)?** The event log is
# MAGIC the source of truth and already keeps every run, so rebuilding is idempotent and keeps
# MAGIC the full history. Re-running this cell never double-counts.
# MAGIC
# MAGIC **Reading the numbers:** Lakeflow evaluates each expectation **independently** on the
# MAGIC same input, so a row that fails two rules is counted under both. That's why the
# MAGIC per-rule numbers can add up to more than the rows actually removed. Section 3c shows
# MAGIC the real rows in → rows out.

# COMMAND ----------

event_log = spark.table(PIPELINE_EVENT_LOG)
dq_log = build_dq_log(event_log, rules_frame(spark))
dq_log.writeTo(SILVER_DQ_LOG).using("delta").createOrReplace()

dq_log = spark.table(SILVER_DQ_LOG)
latest_run = dq_log.agg(F.max("run_ts")).collect()[0][0]
latest = dq_log.filter(F.col("run_ts") == latest_run)
print("Latest pipeline run with expectations:", latest_run)
display(latest.orderBy("dataset", "action", F.desc("failed_records")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Reconciliation: event log vs. independent recount
# MAGIC The pipeline's numbers must equal the counts computed in section 2e directly from
# MAGIC Bronze with the same rules. If they match, the DQ log can be trusted.

# COMMAND ----------

recount_sales = count_rule_failures(staged, {**SALES_DROP_RULES, **SALES_WARN_RULES})
recount_returns = count_rule_failures(
    staged.filter("is_cancellation"), {**RETURNS_DROP_RULES, **RETURNS_WARN_RULES}
)
expected = {}
for dataset, df in [("sales_validated", recount_sales), ("returns_validated", recount_returns)]:
    counts = df.collect()[0].asDict()
    input_rows = counts.pop("input_rows")
    for rule, failed in counts.items():
        expected[(dataset, rule)] = (failed, input_rows)

actual = {
    (r["dataset"], r["rule"]): (r["failed_records"], r["passed_records"] + r["failed_records"])
    for r in latest.collect()
}
mismatches = {
    k: {"recount": v, "event_log": actual.get(k)} for k, v in expected.items() if actual.get(k) != v
}
print(f"Rules checked: {len(expected)} | mismatches: {len(mismatches)}")
assert not mismatches, mismatches
print("✅ DQ log matches the independent recount for every rule (failed and total records)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. DQ summary for the README / PROGRESS_LOG

# COMMAND ----------

display(
    latest.select("dataset", "rule", "action", "failed_records", "failed_pct", "condition").orderBy(
        "dataset", F.desc("action"), F.desc("failed_records")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. Rows in → rows out (actual effect of the rules)
# MAGIC Bronze rows → Silver `sales` + `returns`. The difference is what the drop rules removed
# MAGIC in total (each removed row counted once).

# COMMAND ----------

bronze_rows = bronze.count()
sales_rows = sales.count()
returns_rows = returns.count()
cancellation_rows = staged.filter("is_cancellation").count()
flow_summary = spark.createDataFrame(
    [
        ("bronze rows", bronze_rows),
        ("sales candidates (non-cancellations)", bronze_rows - cancellation_rows),
        ("sales rows kept", sales_rows),
        ("sales rows removed by drop rules", bronze_rows - cancellation_rows - sales_rows),
        ("returns candidates (cancellations)", cancellation_rows),
        ("returns rows kept", returns_rows),
        ("returns rows removed by drop rules", cancellation_rows - returns_rows),
    ],
    "step STRING, rows BIGINT",
)
display(flow_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC Rows written per flow in the latest run, straight from the event log (the pipeline UI
# MAGIC shows the same numbers rounded).

# COMMAND ----------

latest_update_id = latest.select("update_id").first()[0]
display(output_rows_per_flow(event_log).filter(F.col("update_id") == latest_update_id))
