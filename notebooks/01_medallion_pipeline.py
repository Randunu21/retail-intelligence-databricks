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

from common.tables import BRONZE_TRANSACTIONS, RAW_MANIFEST_PATH
from ingestion.raw_reader import RAW_COLUMNS

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
