# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup check: read the raw Online Retail II files with PySpark
# MAGIC
# MAGIC **Phase 0 goal:** prove that the workspace, the Git folder, Unity Catalog and the raw
# MAGIC Volume all work together, before any pipeline code is written.
# MAGIC
# MAGIC | What | Where |
# MAGIC |---|---|
# MAGIC | Raw CSVs (one per source sheet) | `/Volumes/retail/bronze/raw/online_retail_ii/csv/` |
# MAGIC | Original workbook (lineage) | `/Volumes/retail/bronze/raw/online_retail_ii/xlsx/` |
# MAGIC | Conversion manifest (expected counts) | `/Volumes/retail/bronze/raw/online_retail_ii/manifest.json` |
# MAGIC
# MAGIC **Checks in this notebook**
# MAGIC 1. The files are visible in the Volume.
# MAGIC 2. PySpark reads them with the expected all-string schema.
# MAGIC 3. Row counts (total and per sheet) **exactly match** the manifest written by the local
# MAGIC    conversion script, so nothing was lost or duplicated between the Mac and Databricks.
# MAGIC 4. Record the serverless Spark/Python versions for the log.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Make `src/` importable
# MAGIC In a Git folder the notebook's working directory is its own folder (`notebooks/`),
# MAGIC so the repo root is one level up. Adding `<repo>/src` to `sys.path` lets us import the
# MAGIC same modules that the local pytest suite tests. Same code, two environments.

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

from ingestion.raw_reader import read_raw_csv, with_source_file

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC Unity Catalog names follow `catalog.schema.object`. A Volume's files live under
# MAGIC `/Volumes/<catalog>/<schema>/<volume>/`.

# COMMAND ----------

CATALOG, SCHEMA, VOLUME = "retail", "bronze", "raw"
DATASET_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/online_retail_ii"
CSV_GLOB = f"{DATASET_DIR}/csv/*.csv"
MANIFEST_PATH = f"{DATASET_DIR}/manifest.json"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Files visible in the Volume?

# COMMAND ----------

for sub in ["", "csv", "xlsx"]:
    for f in dbutils.fs.ls(f"{DATASET_DIR}/{sub}"):
        print(f"{f.size / 1e6:10.2f} MB  {f.path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Expected numbers from the manifest
# MAGIC Volumes are also mounted as a normal file path, so plain Python `open()` works.

# COMMAND ----------

with open(MANIFEST_PATH) as fh:
    manifest = json.load(fh)

expected_per_file = {s["csv_file"]: s["rows"] for s in manifest["sheets"]}
expected_null_cust = {s["csv_file"]: s["null_customer_id"] for s in manifest["sheets"]}
print("Source:", manifest["source_file"], "| sha256:", manifest["source_sha256"][:16], "...")
print("Expected rows per file:", expected_per_file, "| total:", manifest["total_rows"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Read with PySpark
# MAGIC `read_raw_csv` applies a fixed **all-string schema** (no `inferSchema`, so no extra pass
# MAGIC and no type guessing) and `escape='"'` so quoted descriptions with commas parse correctly.
# MAGIC `with_source_file` adds the file name from Spark's hidden `_metadata` column. That's
# MAGIC how we know which sheet/year each row came from.

# COMMAND ----------

raw_df = with_source_file(read_raw_csv(spark, CSV_GLOB))
raw_df.printSchema()

# COMMAND ----------

display(raw_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Row counts: total and per sheet, compared with the manifest
# MAGIC One `groupBy` computes both the row count and the missing-Customer-ID count per file in a
# MAGIC single pass over the data. (Serverless doesn't support `.cache()`, so we avoid scanning twice.)

# COMMAND ----------

counts_df = (
    raw_df.groupBy("_source_file")
    .agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.col("Customer ID").isNull().cast("int")).alias("null_customer_id"),
        F.min("InvoiceDate").alias("min_invoice_date_str"),
        F.max("InvoiceDate").alias("max_invoice_date_str"),
    )
    .orderBy("_source_file")
)

counts = {r["_source_file"]: r.asDict() for r in counts_df.collect()}
display(counts_df)

# COMMAND ----------

total_rows = sum(r["rows"] for r in counts.values())
print(f"Spark total rows: {total_rows:,} | manifest total: {manifest['total_rows']:,}")

assert set(counts) == set(expected_per_file), (
    f"File mismatch: {set(counts)} vs {set(expected_per_file)}"
)
for fname, exp in expected_per_file.items():
    got = counts[fname]["rows"]
    assert got == exp, f"{fname}: Spark read {got:,} rows, manifest says {exp:,}"
    got_null, exp_null = counts[fname]["null_customer_id"], expected_null_cust[fname]
    assert got_null == exp_null, f"{fname}: null Customer ID {got_null:,} vs manifest {exp_null:,}"
assert total_rows == manifest["total_rows"]
print("✅ Row counts and missing-Customer-ID counts match the manifest for every file.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Environment versions (copy into PROGRESS_LOG.md → Conventions)
# MAGIC The local `pyspark` pin in `requirements-dev.txt` should match this Spark version line.

# COMMAND ----------

print("Spark version :", spark.version)
print("Python version:", sys.version.split()[0])
