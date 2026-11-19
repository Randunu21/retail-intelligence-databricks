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

from common.tables import BENCH_SCHEMA, BENCHMARK_RUNS, CATALOG, fq
from optimisation.benchmark import (
    Variant,
    WriteSpec,
    describe_detail,
    fingerprint,
    markdown_table,
    pipeline_totals,
    run_benchmark,
    summarise,
    write_table,
)
from optimisation.builds import BASELINE_STEPS, PROD_NAMES, bench_names

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
        msg = getattr(e, "getErrorClass", lambda: None)() or type(e).__name__
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
# MAGIC Runtime: 10 steps × (1 warm-up + 5 measured) builds. Expect roughly 10–20 minutes.

# COMMAND ----------

REPETITIONS_X1 = 5
WARMUP = 1

baseline_x1 = Variant("baseline", bench_names("x1_baseline"), notes="Phase 1 logic as-is")

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
BASELINE_X1_RUN_ID = records_x1[0]["run_id"]
print("run_id:", BASELINE_X1_RUN_ID)

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
