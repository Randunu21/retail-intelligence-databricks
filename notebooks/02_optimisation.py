# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Pipeline optimisation, measured
# MAGIC
# MAGIC Goal: make the Silver + Gold builds faster or cheaper to query, and **prove it with
# MAGIC numbers from real runs**. Every number in `docs/optimisation_report.md` comes from the
# MAGIC Delta table `retail.bench.benchmark_runs`, which this notebook fills.
# MAGIC
# MAGIC | Section | What it does |
# MAGIC |---|---|
# MAGIC | 1 | **Free Edition probe**: checks which optimisation features actually work on this workspace |
# MAGIC | 2 | **Benchmark method**: what is timed, and how noise is handled |
# MAGIC | 3 | **Baseline (x1)**: rebuild Silver + Gold exactly as Phase 1 does, time every step, prove the copies equal production |
# MAGIC | 4+ | Scale-up and the optimisations, one at a time (added in later steps) |
# MAGIC
# MAGIC All experiment tables live in the scratch schema **`retail.bench`**: production tables are
# MAGIC never modified here, and the whole schema can be dropped when Phase 2 is done.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Make `src/` importable

# COMMAND ----------

import sys
from pathlib import Path

REPO_DIR = Path.cwd().parent
SRC_DIR = str(REPO_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
PIPELINE_DIR = REPO_DIR / "pipelines" / "medallion" / "transformations"
print("src on path:", SRC_DIR)
print("pipeline SQL:", PIPELINE_DIR, "exists:", PIPELINE_DIR.exists())

# COMMAND ----------

from pyspark.sql import functions as F

from common.tables import (
    BENCH_SCHEMA,
    BENCHMARK_RUNS,
    BRONZE_TRANSACTIONS,
    CATALOG,
    bench_bronze,
    fq,
)
from optimisation.benchmark import (
    Variant,
    WriteSpec,
    compare_scales,
    describe_detail,
    fingerprint,
    latest_run_id,
    markdown_table,
    pipeline_totals,
    run_benchmark,
    summarise,
    write_table,
)
from optimisation.builds import BASELINE_STEPS, PROD_NAMES, bench_names
from optimisation.scaling import expected_rows, replicate_bronze

spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BENCH_SCHEMA} "
    "COMMENT 'Phase 2 scratch schema: benchmark copies and benchmark_runs. Safe to drop.'"
)
print("Spark", spark.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Free Edition probe: what can we actually use?
# MAGIC
# MAGIC The docs say (Nov 2026): serverless supports **only** a handful of Spark configs
# MAGIC (`spark.sql.shuffle.partitions`, `spark.sql.ansi.enabled`, `spark.sql.session.timeZone`,
# MAGIC `spark.sql.files.maxPartitionBytes`, `spark.sql.legacy.timeParserPolicy`,
# MAGIC `spark.databricks.execution.timeout`), and **`cache()` / `persist()` / `CACHE TABLE`
# MAGIC raise an error**. Liquid clustering and Change Data Feed are Delta features and should
# MAGIC work. Rather than trust that, each cell below *tries* the feature and records the result.
# MAGIC
# MAGIC Why it matters: an optimisation report that recommends `df.cache()` or
# MAGIC `autoBroadcastJoinThreshold` on serverless would be wrong. Interview point: *know what
# MAGIC your platform lets you tune.*

# COMMAND ----------

probe = []


def check(feature, fn):
    """Run fn(); record 'works' + its return value, or 'error' + the error class."""
    try:
        detail = fn()
        probe.append({"feature": feature, "result": "works", "detail": str(detail)[:150]})
    except Exception as e:  # record any failure, don't stop the probe
        msg = getattr(e, "getCondition", lambda: None)() or type(e).__name__
        probe.append({"feature": feature, "result": "error", "detail": f"{msg}: {str(e)[:120]}"})


PROBE_TABLE = fq(BENCH_SCHEMA, "_probe")
CDF_TABLE = fq(BENCH_SCHEMA, "_probe_cdf")

# DataFrame cache: expected to fail on serverless.
check("df.cache()", lambda: spark.range(10).cache().count())
check("CACHE TABLE (SQL)", lambda: spark.sql("CACHE TABLE _probe_cache AS SELECT 1 AS x").collect())


# Spark configs: one documented as supported, two classic-cluster favourites.
def try_conf(key, value):
    old = spark.conf.get(key, None)
    spark.conf.set(key, value)
    new = spark.conf.get(key)
    if old is not None:
        spark.conf.set(key, old)
    else:
        spark.conf.unset(key)
    return f"was {old}, set to {new}, restored"


check("conf spark.sql.shuffle.partitions", lambda: try_conf("spark.sql.shuffle.partitions", "8"))
check(
    "conf spark.sql.autoBroadcastJoinThreshold",
    lambda: try_conf("spark.sql.autoBroadcastJoinThreshold", "-1"),
)
check("conf spark.sql.adaptive.enabled", lambda: try_conf("spark.sql.adaptive.enabled", "false"))


# Liquid clustering + OPTIMIZE on a tiny table.
def clustered():
    df = spark.range(1000).withColumn("k", F.col("id") % 10)
    write_table(df, PROBE_TABLE, WriteSpec(cluster_by=("k",)))
    spark.sql(f"OPTIMIZE {PROBE_TABLE}").collect()
    return describe_detail(spark, PROBE_TABLE)


check("liquid clustering + OPTIMIZE", clustered)


# Partitioned write (for the partitioning-vs-clustering comparison).
def partitioned():
    write_table(
        spark.range(100).withColumn("k", F.col("id") % 3),
        PROBE_TABLE + "_p",
        WriteSpec(partition_by=("k",)),
    )
    return describe_detail(spark, PROBE_TABLE + "_p")


check("partitioned table", partitioned)


# Change Data Feed: enable, change a row, read the change rows back.
def cdf():
    spark.sql(
        f"CREATE OR REPLACE TABLE {CDF_TABLE} (id INT, v STRING) "
        "TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    spark.sql(f"INSERT INTO {CDF_TABLE} VALUES (1, 'a'), (2, 'b')")
    spark.sql(f"UPDATE {CDF_TABLE} SET v = 'B' WHERE id = 2")
    changes = spark.sql(
        f"SELECT _change_type, count(*) n FROM table_changes('{CDF_TABLE}', 0) GROUP BY 1"
    )
    return sorted((r[0], r[1]) for r in changes.collect())


check("Change Data Feed (table_changes)", cdf)

# 'noop' sink: runs a query to the end without writing anything (for read benchmarks).
check("noop write format", lambda: spark.range(10).write.format("noop").mode("overwrite").save())

# Query history system table: files read / pruned per query (needs system-table access).
check(
    "system.query.history",
    lambda: spark.sql(
        "SELECT count(*) FROM system.query.history "
        "WHERE start_time > current_timestamp() - INTERVAL 1 DAY"
    ).first()[0],
)

# Predictive optimisation can run OPTIMIZE / VACUUM on managed tables in the background,
# which would change file counts between benchmark runs. Show whether it's on for retail.
check(
    "predictive optimisation (schema setting)",
    lambda: [
        r.asDict()
        for r in spark.sql(f"DESCRIBE SCHEMA EXTENDED {CATALOG}.{BENCH_SCHEMA}").collect()
        if "redictive" in str(r.asDict())
    ],
)

for t in (PROBE_TABLE, PROBE_TABLE + "_p", CDF_TABLE):
    spark.sql(f"DROP TABLE IF EXISTS {t}")

print(markdown_table(probe, ["feature", "result", "detail"]))
display(spark.createDataFrame(probe))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Benchmark method
# MAGIC
# MAGIC **What is timed.** Each Silver/Gold table is rebuilt in this notebook with the *same code*
# MAGIC the pipeline uses (Silver/Gold functions from `src/`, Gold SQL read from the pipeline's
# MAGIC `.sql` files) and written with `CREATE OR REPLACE` to `retail.bench.<variant>__<table>`.
# MAGIC The timer wraps the write, because a write forces the full plan to execute and can't be
# MAGIC served from the query-result cache. (`count()` is a trap: Delta often answers it from file
# MAGIC statistics without reading data.)
# MAGIC
# MAGIC **Why not time the Lakeflow pipeline directly?** A pipeline update has a large, variable
# MAGIC start-up cost, gives one sample per update, and can't change one thing at a time. The
# MAGIC pipeline is re-checked at the end (step 2.6) with the event log, as confirmation.
# MAGIC
# MAGIC **Noise on serverless.**
# MAGIC - 1 warm-up run per step, excluded (`is_warmup = true`): absorbs scale-up, imports, plan
# MAGIC   caches and the first read into the local disk cache. All variants are then measured in
# MAGIC   the same warm state. Serverless doesn't let us clear the disk cache, so "cold-cache"
# MAGIC   numbers aren't reproducible here; we report warm numbers and say so.
# MAGIC - Several measured runs; we report the **median** plus min/max.
# MAGIC - Variants run **interleaved** (A, B, A, B, ...) so platform drift affects both equally.
# MAGIC - A change counts only if the min–max ranges **don't overlap**; otherwise "within noise".
# MAGIC
# MAGIC **What else is recorded** (`DESCRIBE DETAIL` after each build): number of files, size,
# MAGIC partition and clustering columns, and the output row count.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline at x1 (real data)
# MAGIC
# MAGIC The baseline is Phase 1's logic **unchanged**, including one known inefficiency: in the
# MAGIC pipeline `silver_staged` is a temporary *view* read by both `sales` and `returns`, so the
# MAGIC staging (typing + window de-dup over all Bronze rows) runs twice. We keep it in the
# MAGIC baseline on purpose; step 2.4 measures fixing it.
# MAGIC
# MAGIC Runtime: 10 steps × (1 warm-up + 5 measured) builds, about 4 minutes of build time.
# MAGIC Already recorded, so `RUN_BASELINE_X1 = False` reuses the stored run.

# COMMAND ----------

# Recorded on 2026-11-19 (run 1dea80f3-...). Set to True only to re-time the baseline.
RUN_BASELINE_X1 = False
REPETITIONS_X1 = 5
WARMUP = 1

baseline_x1 = Variant("baseline", bench_names("x1_baseline"), notes="Phase 1 logic as-is")

if RUN_BASELINE_X1:
    records_x1 = run_benchmark(
        spark,
        BASELINE_STEPS,
        [baseline_x1],
        scale="x1",
        pipeline_dir=PIPELINE_DIR,
        repetitions=REPETITIONS_X1,
        warmup=WARMUP,
        runs_table=BENCHMARK_RUNS,
    )
BASELINE_X1_RUN_ID = latest_run_id(spark.read.table(BENCHMARK_RUNS), "baseline", "x1")
print("x1 baseline run_id:", BASELINE_X1_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Do the benchmark copies equal production?
# MAGIC
# MAGIC If the notebook rebuild didn't produce exactly the production tables, its timings would
# MAGIC say nothing about the pipeline. `fingerprint` = row count + an order-independent
# MAGIC checksum of every value. Every row below must show `match = true`.

# COMMAND ----------

checks = []
for step in BASELINE_STEPS:
    prod = fingerprint(spark.read.table(PROD_NAMES[step.name])).first()
    bench = fingerprint(spark.read.table(baseline_x1.names[step.name])).first()
    checks.append(
        {
            "table": step.name,
            "prod_rows": prod.rows,
            "bench_rows": bench.rows,
            "match": prod == bench,
        }
    )
print(markdown_table(checks, ["table", "prod_rows", "bench_rows", "match"]))
assert all(c["match"] for c in checks), (
    "A benchmark copy differs from production: investigate first"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Baseline results
# MAGIC
# MAGIC `median_s` per step, plus files and size of each output. The printed Markdown is what
# MAGIC goes into the report (paste it back into the chat).

# COMMAND ----------

runs = spark.read.table(BENCHMARK_RUNS).filter(F.col("run_id") == BASELINE_X1_RUN_ID)
summary_x1 = summarise(runs)
totals_x1 = pipeline_totals(summary_x1)
display(summary_x1)
display(totals_x1)

SUMMARY_COLS = [
    "step",
    "n_runs",
    "median_s",
    "min_s",
    "max_s",
    "stddev_s",
    "output_rows",
    "num_files",
    "size_mb",
]
print("Spark version of these runs:", runs.select("spark_version").first()[0])
print(markdown_table([r.asDict() for r in summary_x1.collect()], SUMMARY_COLS))
print()
print(
    markdown_table(
        [r.asDict() for r in totals_x1.collect()],
        [
            "experiment",
            "scale",
            "steps",
            "total_median_s",
            "total_min_s",
            "total_max_s",
            "total_files",
            "total_size_mb",
        ],
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. What the x1 baseline already tells us
# MAGIC
# MAGIC - Every Gold step takes **2.5–3.6 s, whatever its size**: the 739-row `dim_date` (2.6 s)
# MAGIC   costs about the same as the 1M-row `fact_sales` (2.8 s). That floor is per-write
# MAGIC   overhead (planning, Delta commit, metadata), not data processing.
# MAGIC - Each table is **1 file** (dim_customer: 2). There is nothing for partitioning, clustering
# MAGIC   or file skipping to skip.
# MAGIC - `silver_returns` (17,915 rows) takes 3.2 s, more than most Gold steps, because it re-runs
# MAGIC   the whole Bronze staging (typing + window de-dup) that `silver_sales` already did.
# MAGIC
# MAGIC So at x1 most layout tricks *can't* show a gain. That's the reason for section 4.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Scaled-up copy (synthetic x10) and its baseline
# MAGIC
# MAGIC **What it is:** Bronze replicated 10× (`optimisation.scaling.replicate_bronze`). Copy 0 is
# MAGIC the real data; copies 1–9 get shifted customer IDs (+100,000 per copy) and invoice numbers
# MAGIC (`489434` → `R03-489434`, `C489449` → `CR03-489449`) so they are *new* customers and
# MAGIC invoices instead of duplicates. Products, prices, dates and countries are unchanged.
# MAGIC
# MAGIC **What it is not:** new behaviour. It's the same purchase patterns repeated with relabelled
# MAGIC keys. Every number measured on it is reported as **"synthetic x10 replica"**, next to the
# MAGIC x1 number, never on its own.
# MAGIC
# MAGIC **Why x10, not x20:** about 10.7M Bronze rows is enough for multi-file tables, and keeps a
# MAGIC full baseline (10 steps × 4 builds) well inside Free Edition's daily fair-use quota.
# MAGIC
# MAGIC **Interview point:** benchmarking on toy data hides or exaggerates effects. Scaling a real
# MAGIC dataset with key-shifting keeps realistic distributions (skew, nulls, duplicates), unlike
# MAGIC random synthetic data, but must be disclosed.

# COMMAND ----------

SCALE_FACTOR = 10
SCALE = f"x{SCALE_FACTOR}"
BRONZE_XN = bench_bronze(SCALE_FACTOR)
REBUILD_BRONZE_XN = False  # True only to regenerate the replica

if REBUILD_BRONZE_XN or not spark.catalog.tableExists(BRONZE_XN):
    scaled = replicate_bronze(spark.read.table(BRONZE_TRANSACTIONS), SCALE_FACTOR)
    # Bronze keeps the source column name "Customer ID" -> needs column mapping, like prod.
    write_table(scaled, BRONZE_XN, WriteSpec(properties={"delta.columnMapping.mode": "name"}))

bronze_info = []
for name, table in (("x1 (prod)", BRONZE_TRANSACTIONS), (SCALE, BRONZE_XN)):
    d = describe_detail(spark, table)
    bronze_info.append(
        {
            "bronze": name,
            "rows": spark.read.table(table).count(),
            "files": d.num_files,
            "size_mb": round(d.size_bytes / 2**20, 2),
        }
    )
print(markdown_table(bronze_info, ["bronze", "rows", "files", "size_mb"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Baseline at x10
# MAGIC
# MAGIC Same Phase 1 logic, reading the replica. 3 measured runs + 1 warm-up per step (fewer
# MAGIC than x1 to save quota; at this size run-to-run noise is smaller relative to the runtime).
# MAGIC Expect roughly 10–20 minutes. Results are saved after every step, so if the daily quota
# MAGIC stops compute, the steps already measured are kept: re-run with `ONLY_STEPS` set to the
# MAGIC missing ones.

# COMMAND ----------

RUN_BASELINE_XN = True
REPETITIONS_XN = 3
ONLY_STEPS = None  # e.g. ["gold_product_performance", "gold_customer_cohort_retention"]

baseline_xn = Variant(
    "baseline",
    bench_names(f"{SCALE}_baseline", bronze_table=BRONZE_XN),
    notes=f"Phase 1 logic as-is; synthetic {SCALE} replica",
)

if RUN_BASELINE_XN:
    run_benchmark(
        spark,
        BASELINE_STEPS,
        [baseline_xn],
        scale=SCALE,
        pipeline_dir=PIPELINE_DIR,
        repetitions=REPETITIONS_XN,
        warmup=WARMUP,
        only_steps=ONLY_STEPS,
        runs_table=BENCHMARK_RUNS,
    )
BASELINE_XN_RUN_ID = latest_run_id(spark.read.table(BENCHMARK_RUNS), "baseline", SCALE)
print(f"{SCALE} baseline run_id:", BASELINE_XN_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Did the replica scale the way it should?
# MAGIC
# MAGIC Because copies 1–9 are new customers/invoices with the same products and dates:
# MAGIC Silver and fact rows must be exactly 10× x1, customers 10 × 5,852 + Unknown, and the
# MAGIC product, date, monthly-country and cohort tables must keep their x1 row counts. Revenue
# MAGIC must be exactly 10× too. Any `ok = false` means the replica isn't what we claim it is.

# COMMAND ----------

scale_checks = []
for step in BASELINE_STEPS:
    n1 = spark.read.table(baseline_x1.names[step.name]).count()
    nn = spark.read.table(baseline_xn.names[step.name]).count()
    exp = expected_rows(step.name, n1, SCALE_FACTOR)
    scale_checks.append(
        {"table": step.name, "x1_rows": n1, f"{SCALE}_rows": nn, "expected": exp, "ok": nn == exp}
    )

rev = {
    s: spark.read.table(v.names["gold_fact_sales"]).agg(F.sum("line_revenue")).first()[0]
    for s, v in (("x1", baseline_x1), (SCALE, baseline_xn))
}
scale_checks.append(
    {
        "table": "fact revenue",
        "x1_rows": rev["x1"],
        f"{SCALE}_rows": rev[SCALE],
        "expected": rev["x1"] * SCALE_FACTOR,
        "ok": rev[SCALE] == rev["x1"] * SCALE_FACTOR,
    }
)
print(markdown_table(scale_checks, ["table", "x1_rows", f"{SCALE}_rows", "expected", "ok"]))
assert all(c["ok"] for c in scale_checks), "Replica doesn't scale as designed: investigate first"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4c. x1 vs x10 baseline
# MAGIC
# MAGIC `time_ratio` = how much longer a step takes with 10× the rows. A ratio near 1 means the
# MAGIC step is dominated by fixed overhead; near 10 means it scales with the data.

# COMMAND ----------

runs_all = spark.read.table(BENCHMARK_RUNS).filter(
    F.col("run_id").isin(BASELINE_X1_RUN_ID, BASELINE_XN_RUN_ID)
)
summary_base = summarise(runs_all)
scales = compare_scales(summary_base, "x1", SCALE)
display(scales)
print(
    markdown_table(
        [r.asDict() for r in summary_base.filter(F.col("scale") == SCALE).collect()], SUMMARY_COLS
    )
)
print()
print(
    markdown_table(
        [r.asDict() for r in scales.collect()],
        [
            "step",
            "median_s_x1",
            f"median_s_{SCALE}",
            "time_ratio",
            "rows_ratio",
            f"num_files_{SCALE}",
            f"size_mb_{SCALE}",
        ],
    )
)
print()
print(
    markdown_table(
        [r.asDict() for r in pipeline_totals(summary_base).collect()],
        [
            "experiment",
            "scale",
            "steps",
            "total_median_s",
            "total_min_s",
            "total_max_s",
            "total_files",
            "total_size_mb",
        ],
    )
)
