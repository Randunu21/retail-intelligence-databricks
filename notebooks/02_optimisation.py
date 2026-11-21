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
import uuid
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
    BENCHMARK_IO_PROBES,
    BENCHMARK_QUERY_METRICS,
    BENCHMARK_RUNS,
    BRONZE_TRANSACTIONS,
    CATALOG,
    bench_bronze,
    fq,
)
from optimisation.benchmark import (
    POST_SUFFIX,
    Variant,
    WriteSpec,
    compare_scales,
    describe_detail,
    fingerprint,
    latest_run_id,
    markdown_table,
    pipeline_totals,
    history_between,
    history_diagnosis,
    match_history,
    probes_to_df,
    run_benchmark,
    run_io_probe,
    run_read_benchmark,
    summarise,
    summarise_io,
    write_table,
)
from optimisation.builds import BASELINE_STEPS, PROD_NAMES, BuildContext, bench_names
from optimisation.compute import (
    GOLD_JOIN_STEPS,
    SILVER_COMBINED_STEPS,
    gold_variants,
    plan_summary,
    plan_text,
    silver_variants,
)
from optimisation.layout import FACT_STEP, LAYOUT_QUERIES, layout_variants
from optimisation.scaling import expected_rows, replicate_bronze
from transformations.silver import stage_silver

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4d. What x10 tells us (run 39753a7e, 2026-11-20)
# MAGIC
# MAGIC - Whole build: **33.9 s → 77.0 s** for 10× the rows (×2.27): much of the x1 time was overhead.
# MAGIC - **Silver is where the time goes:** `silver_sales` 37.6 s (49 % of the total, time ×5.64)
# MAGIC   and `silver_returns` 8.3 s, together 60 %. Both re-run the same typing + window de-dup
# MAGIC   over 10.7M Bronze rows; returns needs it for 179,150 rows. → targets for step 2.4
# MAGIC   (stage once) and 2.5 (incremental instead of full rebuild).
# MAGIC - `fact_sales` (10M rows) takes 5.9 s and is **one 136.84 MB file**: Databricks sized the
# MAGIC   file from the table size. One file can't be skipped, which matters for reads (section 5).
# MAGIC - Dimensions and marts stay at 2.6–4.7 s: still mostly fixed per-write overhead.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data layout of `fact_sales`: partitioning vs Z-order vs liquid clustering
# MAGIC
# MAGIC Layout doesn't make a table faster to *build*; it makes *queries* read less. Delta keeps
# MAGIC min/max statistics per file, so a filter like `date_key BETWEEN ...` can skip every file
# MAGIC whose range doesn't overlap (**data skipping**). That only works if (a) similar values sit
# MAGIC in the same files and (b) there is more than one file.
# MAGIC
# MAGIC | Option | How it lays out data | Strengths | Weaknesses |
# MAGIC |---|---|---|---|
# MAGIC | **Partitioning** (`PARTITIONED BY date_key`) | One folder per value | Perfect pruning on that column | Fixed at creation; one key; high-cardinality keys → thousands of tiny files (here ~600 days); no help for other columns |
# MAGIC | **Z-order** (`OPTIMIZE ZORDER BY`) | Sorts files along a space-filling curve of several columns | Multi-column skipping | Each OPTIMIZE rewrites the whole table/partition; keys not stored with the table; must be re-run after writes |
# MAGIC | **Liquid clustering** (`CLUSTER BY`) | Clustering keys stored in the table; OPTIMIZE clusters incrementally | Multi-column, keys changeable without rewrite, incremental, row-level concurrency, recommended for new tables | Needs OPTIMIZE (or predictive optimisation) to cluster; little effect on tiny tables |
# MAGIC
# MAGIC **Why over-partitioning small data hurts:** every partition value gets its own files, so
# MAGIC ~600 days × a few hundred KB each = hundreds of tiny files. Each file costs a listing,
# MAGIC an open and a metadata entry, which dominates the actual read; and any query that doesn't
# MAGIC filter on the partition column (a customer lookup) must open all of them. The usual rule
# MAGIC of thumb: only partition tables above ~1 TB, with partitions of at least ~1 GB.
# MAGIC
# MAGIC Five variants of `fact_sales` (see `optimisation.layout`), all built from the same
# MAGIC baseline Silver: `baseline`, `part_date`, `zorder_32mb`, `lc_auto`, `lc_32mb`. Measured:
# MAGIC 1. **Build cost** (5a): the write, plus `OPTIMIZE` timed separately (`__post`).
# MAGIC 2. **Read cost** (5c–5d): three queries: one month by country (date filter), one customer
# MAGIC    (customer filter) and all countries (no filter, a control that layout can't help).
# MAGIC
# MAGIC Run at x1 *and* x10: at x1 the honest expectation is "no difference".

# COMMAND ----------

LAYOUT_SCALES = {"x1": baseline_x1, SCALE: baseline_xn}
RUN_LAYOUT_BUILD = True
LAYOUT_REPS = 3

layout = {s: layout_variants(base.names, s) for s, base in LAYOUT_SCALES.items()}

if RUN_LAYOUT_BUILD:
    for s, variants in layout.items():
        run_benchmark(
            spark,
            BASELINE_STEPS,
            variants,
            scale=s,
            pipeline_dir=PIPELINE_DIR,
            repetitions=LAYOUT_REPS,
            warmup=WARMUP,
            only_steps=[FACT_STEP],
            runs_table=BENCHMARK_RUNS,
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5a. Same data in every layout? And what each layout costs to build
# MAGIC
# MAGIC First the fingerprint check (a layout must never change the data), then per variant:
# MAGIC write time, OPTIMIZE time, files and size after OPTIMIZE.

# COMMAND ----------

all_runs = spark.read.table(BENCHMARK_RUNS)
layout_build = []
for s, variants in layout.items():
    ref = fingerprint(spark.read.table(LAYOUT_SCALES[s].names[FACT_STEP])).first()
    run_id = latest_run_id(all_runs, "lc_32mb", s, layer="gold")
    summ = {
        r.step + "|" + r.experiment: r
        for r in summarise(all_runs.filter(F.col("run_id") == run_id)).collect()
    }
    for v in variants:
        same = fingerprint(spark.read.table(v.names[FACT_STEP])).first() == ref
        w = summ[f"{FACT_STEP}|{v.experiment}"]
        post = summ.get(f"{FACT_STEP}{POST_SUFFIX}|{v.experiment}")
        layout_build.append(
            {
                "scale": s,
                "experiment": v.experiment,
                "same_data": same,
                "write_s": w.median_s,
                "write_range": f"{w.min_s}-{w.max_s}",
                "optimize_s": post.median_s if post else 0.0,
                "total_s": round(w.median_s + (post.median_s if post else 0.0), 3),
                "files": w.num_files,
                "size_mb": w.size_mb,
                "layout": w.clustering_columns or w.partition_columns or "-",
            }
        )
print(
    markdown_table(
        layout_build,
        [
            "scale",
            "experiment",
            "same_data",
            "write_s",
            "write_range",
            "optimize_s",
            "total_s",
            "files",
            "size_mb",
            "layout",
        ],
    )
)
assert all(r["same_data"] for r in layout_build), "A layout variant changed the data"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. Read benchmark
# MAGIC
# MAGIC Each query runs 1 warm-up + 5 measured times per variant, interleaved. Every execution's
# MAGIC SQL carries a unique tag (`tagged_sql`): no result cache can answer it, and section 5c
# MAGIC can find that exact execution in `system.query.history`.

# COMMAND ----------

RUN_LAYOUT_READS = True
READ_REPS = 5

if RUN_LAYOUT_READS:
    for s, variants in layout.items():
        run_read_benchmark(
            spark,
            LAYOUT_QUERIES,
            variants,
            scale=s,
            repetitions=READ_REPS,
            warmup=WARMUP,
            runs_table=BENCHMARK_RUNS,
        )

all_runs = spark.read.table(BENCHMARK_RUNS)
READ_RUN_IDS = {s: latest_run_id(all_runs, "lc_32mb", s, layer="read") for s in layout}
print(READ_RUN_IDS)
reads_time = summarise(all_runs.filter(F.col("run_id").isin(*READ_RUN_IDS.values())))
print(
    markdown_table(
        [r.asDict() for r in reads_time.collect()],
        ["scale", "step", "experiment", "median_s", "min_s", "max_s", "output_rows"],
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c. What do the timings say? (x1 run ae73c091, x10 run ccb0aacb, 2026-11-20)
# MAGIC
# MAGIC **Build cost (5a):**
# MAGIC - `part_date` writes **604 files** (one per sales day): x1 16.3 s vs 2.7 s baseline (×6),
# MAGIC   x10 15.8 s vs 5.6 s. Ranges don't overlap. Over-partitioning, measured.
# MAGIC - Liquid clustering at x10: write 5.3 s + OPTIMIZE 4.0 s = 9.3 s → **4 files** instead of 1.
# MAGIC   Z-order: write 5.6 s + OPTIMIZE 7.6 s = 13.2 s, also 4 files: clustering's OPTIMIZE is
# MAGIC   about half as long as Z-order's here.
# MAGIC - `lc_32mb` ended with the same 4 files as `lc_auto`: OPTIMIZE already produced ~35 MB files,
# MAGIC   so the 32 MB target changed nothing at this size.
# MAGIC - At x1 clustering costs ~4 s of OPTIMIZE and still leaves 1 file: pure cost, no layout gain.
# MAGIC
# MAGIC **Read time (5b):** every query takes 0.5–0.8 s on the client, mostly fixed round-trip cost,
# MAGIC so differences are small and mostly inside noise. The ones whose min–max ranges don't overlap:
# MAGIC - x1: `part_date` is **slower** on the full scan (0.734 s vs 0.581 s): 604 files to open.
# MAGIC - x10 `one_customer`: `lc_auto` 0.532 s and `zorder_32mb` 0.532 s vs baseline 0.624 s (−15 %).
# MAGIC - x10 `all_by_country` (the control, no filter): `lc_auto` 0.657 s vs baseline 0.743 s.
# MAGIC   Layout can't skip anything in a full scan, so this is most likely **parallelism**: 4 files
# MAGIC   are read by more tasks than one 137 MB file. To be confirmed with the I/O numbers below.
# MAGIC - x10 `month_by_country`: no difference beyond noise in wall-clock time.
# MAGIC
# MAGIC Client time can't show *how much was read*. That needs `system.query.history` (5d).

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5d. Files and bytes actually read (`system.query.history`)
# MAGIC
# MAGIC **First attempt matched 0 of 180.** Cause (seen in the history table below): for queries
# MAGIC sent from a notebook over Spark Connect, `statement_text` holds the **Python line** that
# MAGIC ran the query (`writer.createOrReplace()`, `spark.sql(f"OPTIMIZE {table}").collect()`),
# MAGIC not the SQL. The tag inside the SQL string never reaches the history. Fix:
# MAGIC 1. **Query tags** (`SET QUERY_TAGS['bench'] = ...`): every following statement carries the
# MAGIC    tag in the `query_tags` column. A tag match is accepted at any time.
# MAGIC 2. **Time window** as a fallback: each probe query runs alone after a 3 s gap; the history
# MAGIC    row that starts inside its window (and reads the most bytes) is ours.
# MAGIC
# MAGIC File and byte counts are deterministic, so one isolated execution per (variant, query) is
# MAGIC enough: 2 scales × 5 variants × 3 queries = 30 probes, about 2 minutes.
# MAGIC
# MAGIC First, a look at what the history shows for your recent statements (is the text
# MAGIC redacted? is `query_tags` there?).

# COMMAND ----------

hist_cols = spark.read.table("system.query.history").columns
print("query_tags column present:", "query_tags" in hist_cols)
recent = spark.sql(
    "SELECT start_time, statement_type, left(statement_text, 60) AS text, produced_rows, read_files, "
    + ("query_tags " if "query_tags" in hist_cols else "NULL AS query_tags ")
    + "FROM system.query.history WHERE executed_by = current_user() ORDER BY start_time DESC LIMIT 8"
)
display(recent)

# COMMAND ----------

RUN_IO_PROBE = True

if RUN_IO_PROBE:
    probe_id = str(uuid.uuid4())  # one id for both scales
    probe_records = []
    for s, variants in layout.items():
        probe_records += run_io_probe(spark, LAYOUT_QUERIES, variants, scale=s, probe_id=probe_id)
    probes_to_df(spark, probe_records).write.mode("append").saveAsTable(BENCHMARK_IO_PROBES)

probes = spark.read.table(BENCHMARK_IO_PROBES)
last_probe = probes.orderBy(F.desc("started_at")).first().probe_id
probes = probes.filter(F.col("probe_id") == last_probe)
print(
    "probe_id:",
    last_probe,
    "| probes:",
    probes.count(),
    "| query tags accepted:",
    probes.agg(F.min("tags_set")).first()[0],
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Wait before running the next cell.** The history table usually takes a few minutes
# MAGIC (up to an hour) to receive new rows. Re-run it until `matched` = 30; nothing is re-timed.
# MAGIC
# MAGIC Columns: `files_read` / `files_pruned` (file-level data skipping), `read_mb` / `rows_read`
# MAGIC (also drop when Parquet row groups inside a file are skipped), `match_method` (tag or
# MAGIC time window), `candidates` (history rows in the window; 1–2 is clean), `result_cache`
# MAGIC (must be 0).

# COMMAND ----------

bounds = probes.agg(F.min("started_at").alias("a"), F.max("ended_at").alias("b")).first()
history = history_between(spark, bounds.a, bounds.b)
matched_io = match_history(history, probes)
n_matched = matched_io.filter(F.col("match_method") != "none").count()
print(f"matched {n_matched} of {probes.count()} probes")
if n_matched < probes.count():
    # Not everything matched: is the history simply behind, or are tags missing?
    for k, v in history_diagnosis(history, probes).items():
        print(f"  {k}: {v}")
    print(
        "  -> history_caught_up = False: wait and re-run this cell (ingestion can take up to 1 h)."
    )
    print("  -> caught up but history_rows_with_bench_tag = 0: tags don't reach the history from")
    print("     notebooks; the time-window fallback is then the only match method.")

if n_matched:
    matched_io.withColumn("captured_at", F.current_timestamp()).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(BENCHMARK_QUERY_METRICS)
    io_rows = [
        r.asDict() for r in summarise_io(spark.read.table(BENCHMARK_QUERY_METRICS)).collect()
    ]
    print(
        markdown_table(
            io_rows,
            [
                "scale",
                "query",
                "experiment",
                "match_method",
                "candidates",
                "server_ms",
                "files_read",
                "files_pruned",
                "read_mb",
                "rows_read",
                "result_cache",
            ],
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5e. Layout result (x1 + x10 I/O, 2026-11-20)
# MAGIC
# MAGIC - **x1:** every non-partitioned variant is 1 file → nothing to skip, same bytes (1.77 MB full
# MAGIC   scan). `part_date` reads 604 files and **9×** the bytes for the full scan (16.03 MB), 3.6×
# MAGIC   for the customer lookup; only the month query reads less (1.19 vs 1.77 MB).
# MAGIC - **x10:** clustering / Z-order skip 2 of 4 files for both filtered queries: month −40 to −48 %
# MAGIC   bytes, customer −45 to −56 % (ranges over two probes). `part_date`: month −61 %, but customer
# MAGIC   +119 % and full scan +603 % bytes.
# MAGIC - File counts were identical in both probes; MB read varied up to ~20 % between probes.
# MAGIC - **Decision:** liquid clustering on `(date_key, customer_key)`, no partitioning (step 2.6).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compute: shuffles, joins, and an alternative to caching
# MAGIC
# MAGIC ### Caching on serverless
# MAGIC The probe (§1) showed `df.cache()` is *accepted* but `CACHE TABLE` is not. Accepted isn't the
# MAGIC same as useful, so we measure it. The alternatives when reuse matters:
# MAGIC - **Materialise the intermediate as a Delta table** (`staged_table` below): survives the
# MAGIC   session, can be reused by other jobs, costs one extra write.
# MAGIC - **Restructure so the result is computed once** (one query producing both outputs).
# MAGIC - Rely on the **disk (IO) cache**: serverless keeps recently read Parquet on local SSD
# MAGIC   automatically; repeated reads of the same table are already faster (that's why we warm up).
# MAGIC
# MAGIC ### 6a. Silver: prepare Bronze once instead of twice
# MAGIC One combined step writes `silver_sales` **and** `silver_returns`, timed together, in four ways
# MAGIC (`optimisation.compute`): `baseline` (staging computed twice, like the pipeline),
# MAGIC `staged_table` (staging written once to Delta), `cached` (staging `cache()`d once),
# MAGIC `agg_dedup` (de-dup by GROUP BY instead of a window). Run interleaved at x1 and x10.
# MAGIC At x10: 4 variants × 4 runs × ~45 s ≈ 12 minutes.

# COMMAND ----------

# Does cache() do anything here? storageLevel / is_cached show what the session reports.
probe_df = stage_silver(spark.read.table(BRONZE_TRANSACTIONS)).cache()
print("is_cached:", probe_df.is_cached, "| storageLevel:", probe_df.storageLevel)
probe_df.unpersist()

# COMMAND ----------

RUN_SILVER_COMPUTE = True
SILVER_SCALES = {"x1": (BRONZE_TRANSACTIONS, baseline_x1), SCALE: (BRONZE_XN, baseline_xn)}
silver_vs = {s: silver_variants(bronze, s) for s, (bronze, _) in SILVER_SCALES.items()}

if RUN_SILVER_COMPUTE:
    for s, variants in silver_vs.items():
        run_benchmark(
            spark,
            SILVER_COMBINED_STEPS,
            variants,
            scale=s,
            pipeline_dir=PIPELINE_DIR,
            repetitions=3,
            warmup=WARMUP,
            runs_table=BENCHMARK_RUNS,
        )

# COMMAND ----------

all_runs = spark.read.table(BENCHMARK_RUNS)
silver_rows = []
for s, variants in silver_vs.items():
    base = SILVER_SCALES[s][1]
    ref = {
        t: fingerprint(spark.read.table(base.names[t])).first()
        for t in ("silver_sales", "silver_returns")
    }
    rid = latest_run_id(all_runs, "agg_dedup", s, layer="silver")
    summ = summarise(all_runs.filter(F.col("run_id") == rid))
    stats = {r.experiment: r for r in summ.collect()}
    b = stats["baseline"]
    for v in variants:
        r = stats[v.experiment]
        overlap = r.min_s <= b.max_s and b.min_s <= r.max_s
        silver_rows.append(
            {
                "scale": s,
                "experiment": v.experiment,
                "median_s": r.median_s,
                "range": f"{r.min_s}-{r.max_s}",
                "pct_vs_base": round((r.median_s - b.median_s) / b.median_s * 100, 1),
                "beyond_noise": not overlap if v.experiment != "baseline" else "-",
                "same_data": all(
                    fingerprint(spark.read.table(v.names[t])).first() == ref[t] for t in ref
                ),
            }
        )
print(
    markdown_table(
        silver_rows,
        ["scale", "experiment", "median_s", "range", "pct_vs_base", "beyond_noise", "same_data"],
    )
)
assert all(r["same_data"] for r in silver_rows), "A Silver variant changed the data"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. Gold joins: which strategy does Spark pick?
# MAGIC
# MAGIC Before timing, read the plans. `plan_summary` counts join types, shuffles (Exchange) and
# MAGIC broadcasts in each query's physical plan at x10. With AQE the plan can still change at
# MAGIC run time (`AdaptiveSparkPlan isFinalPlan=false`); the query profile shows the final one (2.7).
# MAGIC Variants: `baseline` (optimiser decides), `broadcast` (BROADCAST hints), `no_broadcast`
# MAGIC (MERGE hints: what auto-broadcast saves), `rewrite` (aggregate before joining / drop a join).

# COMMAND ----------

GOLD_SCALES = {"x1": baseline_x1, SCALE: baseline_xn}
gold_vs = {s: gold_variants(base.names, s) for s, base in GOLD_SCALES.items()}
steps_by_name = {s.name: s for s in BASELINE_STEPS}

plans = []
for v in gold_vs[SCALE]:
    ctx = BuildContext(spark, v.names, PIPELINE_DIR)
    for step in GOLD_JOIN_STEPS:
        df = v.step_overrides.get(step, steps_by_name[step].build)(ctx)
        plans.append({"step": step, "experiment": v.experiment, **plan_summary(plan_text(df))})
print(markdown_table(plans, ["step", "experiment", "joins", "shuffles", "broadcasts"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6c. Gold joins: timing (x1 and x10)
# MAGIC 4 variants × 4 steps × 4 runs per scale, interleaved; ~5 minutes per scale.

# COMMAND ----------

RUN_GOLD_COMPUTE = True

if RUN_GOLD_COMPUTE:
    for s, variants in gold_vs.items():
        run_benchmark(
            spark,
            BASELINE_STEPS,
            variants,
            scale=s,
            pipeline_dir=PIPELINE_DIR,
            repetitions=3,
            warmup=WARMUP,
            only_steps=list(GOLD_JOIN_STEPS),
            runs_table=BENCHMARK_RUNS,
        )

all_runs = spark.read.table(BENCHMARK_RUNS)
gold_rows = []
for s, variants in gold_vs.items():
    base = GOLD_SCALES[s]
    rid = latest_run_id(all_runs, "no_broadcast", s, layer="gold")
    summ = summarise(all_runs.filter(F.col("run_id") == rid))
    stats = {(r.step, r.experiment): r for r in summ.collect()}
    for step in GOLD_JOIN_STEPS:
        ref = fingerprint(spark.read.table(base.names[step])).first()
        b = stats[(step, "baseline")]
        for v in variants:
            r = stats[(step, v.experiment)]
            overlap = r.min_s <= b.max_s and b.min_s <= r.max_s
            gold_rows.append(
                {
                    "scale": s,
                    "step": step,
                    "experiment": v.experiment,
                    "median_s": r.median_s,
                    "range": f"{r.min_s}-{r.max_s}",
                    "pct_vs_base": round((r.median_s - b.median_s) / b.median_s * 100, 1),
                    "beyond_noise": not overlap if v.experiment != "baseline" else "-",
                    "same_data": fingerprint(spark.read.table(v.names[step])).first() == ref,
                }
            )
print(
    markdown_table(
        gold_rows,
        [
            "scale",
            "step",
            "experiment",
            "median_s",
            "range",
            "pct_vs_base",
            "beyond_noise",
            "same_data",
        ],
    )
)
assert all(r["same_data"] for r in gold_rows), "A Gold variant changed the data"
