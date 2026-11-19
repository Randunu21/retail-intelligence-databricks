"""Benchmark harness: time each table build, repeat it, and store every run in Delta.

How a measurement works (and why)
---------------------------------
- **What is timed:** the wall-clock time of the write action (`createOrReplace`), measured
  on the driver with `time.perf_counter()`. A write forces the whole plan to run and its
  result can't come from the query-result cache, unlike `display()`, `count()` (often
  answered from Delta file statistics without reading data) or `collect()`.
- **Warm-up:** the first run(s) of each step are recorded with `is_warmup = true` and
  excluded from the statistics. They absorb one-off costs: serverless scaling up, Python
  imports, plan/metadata caches, and the first read of files into the local disk cache.
  After that, every variant is measured in the same "warm" state, which is the fair
  comparison (serverless gives us no way to clear the disk cache between runs).
- **Repeats + median:** serverless shares and rescales hardware, so single timings are
  noisy. We take the median (robust to one slow outlier) and also keep min / max / stdev.
- **Interleaving:** when comparing variants, runs alternate A, B, A, B, ... for each step
  instead of A, A, A, B, B, B. Any drift in the platform (busier at minute 10 than at
  minute 2) then hits both variants equally instead of biasing one of them.
- **Noise rule:** a change only counts as real if the two min-max ranges don't overlap
  (`ranges_overlap = false` in `compare`). Otherwise the report says "within noise".
- **Table facts** after the build (`DESCRIBE DETAIL`): number of files, size, partition and
  clustering columns. Layout changes show up here even when time doesn't move.

Everything that doesn't need Databricks (timing loop, statistics, comparison) is a pure
function tested locally; only `write_table`, `describe_detail` and `save_runs` touch Delta.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from optimisation.builds import BuildContext, BuildStep, check_step_order

# Same table feature as Silver/Gold: their outputs have TIMESTAMP_NTZ columns.
DEFAULT_TABLE_PROPERTIES: dict[str, str] = {"delta.feature.timestampNtz": "supported"}

RUNS_COLUMNS: list[tuple[str, str]] = [
    ("run_id", "STRING"),  # one id per call of run_benchmark (groups a session's runs)
    ("experiment", "STRING"),  # variant name, e.g. "baseline", "liquid_clustering"
    ("scale", "STRING"),  # "x1" (real data) or "x10" (synthetic replica, labelled)
    ("step", "STRING"),  # logical output table, e.g. "silver_sales"
    ("step_order", "INT"),
    ("layer", "STRING"),
    ("repetition", "INT"),  # 1..warmup+repetitions
    ("is_warmup", "BOOLEAN"),
    ("duration_s", "DOUBLE"),
    ("output_table", "STRING"),
    ("output_rows", "BIGINT"),
    ("num_files", "BIGINT"),
    ("size_bytes", "BIGINT"),
    ("partition_columns", "STRING"),
    ("clustering_columns", "STRING"),
    ("spark_version", "STRING"),
    ("notes", "STRING"),
    ("started_at", "TIMESTAMP"),
]
RUNS_SCHEMA = ", ".join(f"{name} {dtype}" for name, dtype in RUNS_COLUMNS)


# ----------------------------------------------------------------------------------------
# Pure helpers (tested locally)
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Timing:
    repetition: int
    is_warmup: bool
    seconds: float
    started_at: datetime


def repeat_timed(
    action: Callable[[], object],
    repetitions: int,
    warmup: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[Timing]:
    """Run `action` warmup + repetitions times and time each run with `clock`."""
    if repetitions < 1 or warmup < 0:
        raise ValueError("need repetitions >= 1 and warmup >= 0")
    timings = []
    for rep in range(1, warmup + repetitions + 1):
        started = now()
        t0 = clock()
        action()
        timings.append(Timing(rep, rep <= warmup, clock() - t0, started))
    return timings


@dataclass(frozen=True)
class TableDetail:
    num_files: int | None
    size_bytes: int | None
    partition_columns: str
    clustering_columns: str


def parse_detail(row: Mapping[str, object]) -> TableDetail:
    """Pick the fields we track out of one `DESCRIBE DETAIL` row (as a dict)."""

    def _join(value: object) -> str:
        return ",".join(value) if value else ""

    return TableDetail(
        num_files=row.get("numFiles"),
        size_bytes=row.get("sizeInBytes"),
        partition_columns=_join(row.get("partitionColumns")),
        clustering_columns=_join(row.get("clusteringColumns")),
    )


def pct_change(before: float, after: float) -> float:
    """Percentage change from before to after (negative = faster / smaller)."""
    if before == 0:
        raise ValueError("before must be non-zero")
    return (after - before) / before * 100


@dataclass(frozen=True)
class WriteSpec:
    """How a step's output is laid out. Empty spec = the baseline (Delta defaults)."""

    partition_by: tuple[str, ...] = ()
    cluster_by: tuple[str, ...] = ()
    properties: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_TABLE_PROPERTIES))

    def __post_init__(self) -> None:
        if self.partition_by and self.cluster_by:
            raise ValueError("Delta tables can't be partitioned AND liquid-clustered")

    def describe(self) -> str:
        if self.cluster_by:
            return f"cluster_by({', '.join(self.cluster_by)})"
        if self.partition_by:
            return f"partition_by({', '.join(self.partition_by)})"
        return "default layout"


@dataclass(frozen=True)
class Variant:
    """One thing to measure: a name, where its tables live, and how each step is written.

    `step_overrides` swaps the build logic of individual steps (e.g. a broadcast-join
    version of one Gold query); `write_specs` changes only the layout of a step's output.
    """

    experiment: str
    names: Mapping[str, str]
    write_specs: Mapping[str, WriteSpec] = field(default_factory=dict)
    step_overrides: Mapping[str, Callable[[BuildContext], DataFrame]] = field(default_factory=dict)
    # Maintenance run right after a step's write (e.g. OPTIMIZE). Timed separately and
    # recorded as step "<step>__post" so write cost and maintenance cost stay visible.
    post_write: Mapping[str, Callable[[SparkSession, str], object]] = field(default_factory=dict)
    notes: str = ""

    def spec_for(self, step: str) -> WriteSpec:
        return self.write_specs.get(step, WriteSpec())


def records_to_df(spark: SparkSession, records: Sequence[Mapping[str, object]]) -> DataFrame:
    """Benchmark records (dicts with the RUNS_COLUMNS keys) -> DataFrame with RUNS_SCHEMA."""
    names = [name for name, _ in RUNS_COLUMNS]
    rows = [tuple(r.get(n) for n in names) for r in records]
    return spark.createDataFrame(rows, RUNS_SCHEMA)


def summarise(runs: DataFrame) -> DataFrame:
    """One row per (run_id, experiment, scale, step): statistics over the measured runs."""
    measured = runs.filter(~F.col("is_warmup"))
    return (
        measured.groupBy("run_id", "experiment", "scale", "step", "step_order", "layer")
        .agg(
            F.count(F.lit(1)).alias("n_runs"),
            F.round(F.median("duration_s"), 3).alias("median_s"),
            F.round(F.min("duration_s"), 3).alias("min_s"),
            F.round(F.max("duration_s"), 3).alias("max_s"),
            F.round(F.stddev("duration_s"), 3).alias("stddev_s"),
            F.max("output_rows").alias("output_rows"),
            F.max("num_files").alias("num_files"),
            F.round(F.max("size_bytes") / (1024 * 1024), 2).alias("size_mb"),
            F.max("partition_columns").alias("partition_columns"),
            F.max("clustering_columns").alias("clustering_columns"),
        )
        .orderBy("scale", "step_order", "experiment")
    )


def pipeline_totals(summary: DataFrame) -> DataFrame:
    """Whole Silver+Gold build per experiment: sum of the per-step medians (and min/max)."""
    return (
        summary.groupBy("run_id", "experiment", "scale")
        .agg(
            F.count(F.lit(1)).alias("steps"),
            F.round(F.sum("median_s"), 3).alias("total_median_s"),
            F.round(F.sum("min_s"), 3).alias("total_min_s"),
            F.round(F.sum("max_s"), 3).alias("total_max_s"),
            F.sum("num_files").alias("total_files"),
            F.round(F.sum("size_mb"), 2).alias("total_size_mb"),
        )
        .orderBy("scale", "experiment")
    )


def compare(summary: DataFrame, before: str, after: str) -> DataFrame:
    """Before/after per (scale, step): medians, % change, and whether ranges overlap.

    `ranges_overlap = true` means the fastest "before" run and the slowest "after" run (or
    the other way round) cross, i.e. the difference is inside the run-to-run noise.
    """
    cols = ["scale", "step", "step_order", "median_s", "min_s", "max_s", "num_files", "size_mb"]
    b = summary.filter(F.col("experiment") == before).select(*cols)
    a = summary.filter(F.col("experiment") == after).select(*cols)
    for c in cols[3:]:
        b = b.withColumnRenamed(c, f"{c}_before")
        a = a.withColumnRenamed(c, f"{c}_after")
    joined = b.join(a.drop("step_order"), ["scale", "step"])
    return (
        joined.withColumn(
            "pct_change",
            F.round(
                (F.col("median_s_after") - F.col("median_s_before"))
                / F.col("median_s_before")
                * 100,
                1,
            ),
        )
        .withColumn(
            "ranges_overlap",
            (F.col("min_s_after") <= F.col("max_s_before"))
            & (F.col("min_s_before") <= F.col("max_s_after")),
        )
        .orderBy("scale", "step_order")
    )


def latest_run_id(
    runs: DataFrame, experiment: str, scale: str, layer: str | None = None
) -> str | None:
    """run_id of the most recent run of (experiment, scale) that covered the most steps.

    Lets the notebook reuse a recorded run instead of re-timing it on every visit.
    Runs that only re-timed a few steps (`only_steps`) lose to a full run. `layer`
    ("silver", "gold", "read", ...) separates build runs from read-query runs.
    """
    selected = runs.filter((F.col("experiment") == experiment) & (F.col("scale") == scale))
    if layer is not None:
        selected = selected.filter(F.col("layer") == layer)
    row = (
        selected.groupBy("run_id")
        .agg(F.countDistinct("step").alias("steps"), F.max("started_at").alias("last"))
        .orderBy(F.desc("steps"), F.desc("last"))
        .first()
    )
    return row.run_id if row else None


def compare_scales(summary: DataFrame, small: str, large: str) -> DataFrame:
    """Same experiment at two scales: medians side by side and how much each step grew."""
    cols = ["experiment", "step", "step_order", "median_s", "output_rows", "num_files", "size_mb"]
    a = summary.filter(F.col("scale") == small).select(*cols)
    b = summary.filter(F.col("scale") == large).select(*cols)
    for c in cols[3:]:
        a = a.withColumnRenamed(c, f"{c}_{small}")
        b = b.withColumnRenamed(c, f"{c}_{large}")
    return (
        a.join(b.drop("step_order"), ["experiment", "step"])
        .withColumn(
            "time_ratio", F.round(F.col(f"median_s_{large}") / F.col(f"median_s_{small}"), 2)
        )
        .withColumn(
            "rows_ratio", F.round(F.col(f"output_rows_{large}") / F.col(f"output_rows_{small}"), 2)
        )
        .orderBy("experiment", "step_order")
    )


def fingerprint(df: DataFrame) -> DataFrame:
    """Row count + an order-independent checksum of every value in `df`.

    Two tables with the same fingerprint hold the same multiset of rows (barring a hash
    collision), whatever their file layout or row order. Used to prove that a benchmark copy
    or an optimised variant produces exactly what production produces. Columns are hashed in
    name order so column order doesn't matter; the sum is DECIMAL to avoid BIGINT overflow
    (serverless runs in ANSI mode, where overflow is an error).
    """
    cols = sorted(df.columns)
    return df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.xxhash64(*cols).cast("decimal(38,0)")).alias("checksum"),
    )


def markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    """Render rows as a Markdown table (for pasting results into the report / chat)."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append(
            "| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in columns) + " |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Delta / Databricks side (not unit-tested locally: needs Delta tables)
# ----------------------------------------------------------------------------------------
def write_table(df: DataFrame, table: str, spec: WriteSpec) -> None:
    """CREATE OR REPLACE the table from `df` with the layout in `spec` (DataFrameWriterV2)."""
    writer = df.writeTo(table).using("delta")
    for key, value in spec.properties.items():
        writer = writer.tableProperty(key, value)
    if spec.partition_by:
        writer = writer.partitionedBy(*[F.col(c) for c in spec.partition_by])
    if spec.cluster_by:
        writer = writer.clusterBy(*spec.cluster_by)
    writer.createOrReplace()


def describe_detail(spark: SparkSession, table: str) -> TableDetail:
    row = spark.sql(f"DESCRIBE DETAIL {table}").first()
    return parse_detail(row.asDict())


def ensure_runs_table(spark: SparkSession, table: str) -> None:
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table} ({RUNS_SCHEMA}) "
        "COMMENT 'Phase 2 benchmark runs: one row per timed table build.'"
    )


def save_runs(spark: SparkSession, records: Sequence[Mapping[str, object]], table: str) -> None:
    """Append records to the benchmark_runs Delta table (history is kept, never overwritten)."""
    ensure_runs_table(spark, table)
    records_to_df(spark, records).write.mode("append").saveAsTable(table)


def _record(run_id, v, scale, step, order, layer, t, target, rows, detail, version, notes):
    return {
        "run_id": run_id,
        "experiment": v.experiment,
        "scale": scale,
        "step": step,
        "step_order": order,
        "layer": layer,
        "repetition": t.repetition,
        "is_warmup": t.is_warmup,
        "duration_s": round(t.seconds, 4),
        "output_table": target,
        "output_rows": rows,
        **asdict(detail),
        "spark_version": version,
        "notes": notes,
        "started_at": t.started_at,
    }


POST_SUFFIX = "__post"


def run_benchmark(
    spark: SparkSession,
    steps: Sequence[BuildStep],
    variants: Sequence[Variant],
    *,
    scale: str,
    pipeline_dir: str | Path,
    repetitions: int = 3,
    warmup: int = 1,
    only_steps: Sequence[str] | None = None,
    runs_table: str | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Build every step for every variant, interleaved, and return (and save) one record per run.

    Loop order: step -> repetition -> variant. Each step is finished for all variants before
    the next step starts, so downstream steps always read upstream tables that exist.
    `only_steps` re-times a subset (their inputs must already exist in each variant's names).
    A variant's `post_write` action (e.g. OPTIMIZE) runs right after each write and is
    recorded as a separate step `<step>__post`.
    """
    check_step_order(steps)
    run_id = str(uuid.uuid4())
    version = spark.version
    selected = {s.name for s in steps if only_steps is None or s.name in only_steps}
    records: list[dict[str, object]] = []

    for order, step in enumerate(steps, start=1):
        if step.name not in selected:
            continue
        writes: dict[str, list[Timing]] = {v.experiment: [] for v in variants}
        posts: dict[str, list[Timing]] = {v.experiment: [] for v in variants}
        for rep in range(1, warmup + repetitions + 1):
            for v in variants:  # interleaved: A, B, A, B, ...
                ctx = BuildContext(spark, v.names, Path(pipeline_dir))
                build = v.step_overrides.get(step.name, step.build)
                target = v.names[step.name]
                spec = v.spec_for(step.name)
                t = repeat_timed(
                    lambda b=build, c=ctx, tg=target, s=spec: write_table(b(c), tg, s), 1, 0
                )[0]
                writes[v.experiment].append(Timing(rep, rep <= warmup, t.seconds, t.started_at))
                msg = f"[{scale}] {step.name:<32} {v.experiment:<24} rep {rep}: {t.seconds:7.2f}s"
                post = v.post_write.get(step.name)
                if post:
                    p = repeat_timed(lambda f=post, tg=target: f(spark, tg), 1, 0)[0]
                    posts[v.experiment].append(Timing(rep, rep <= warmup, p.seconds, p.started_at))
                    msg += f"  + post {p.seconds:6.2f}s"
                log(msg + (" (warm-up)" if rep <= warmup else ""))

        for v in variants:
            target = v.names[step.name]
            detail = describe_detail(spark, target)  # after the last write (+ post action)
            rows = spark.read.table(target).count()
            notes = f"{v.spec_for(step.name).describe()}; {v.notes}".strip("; ")
            for t in writes[v.experiment]:
                records.append(_record(run_id, v, scale, step.name, order, step.layer, t,
                                       target, rows, detail, version, notes))  # fmt: skip
            for t in posts[v.experiment]:
                records.append(_record(run_id, v, scale, step.name + POST_SUFFIX, order, "post",
                                       t, target, rows, detail, version, notes))  # fmt: skip
        if runs_table:  # save after each step: a quota stop mid-run keeps what was measured
            save_runs(spark, [r for r in records if r["step"].startswith(step.name)], runs_table)
    return records


# ----------------------------------------------------------------------------------------
# Read benchmarks: how fast are typical queries against each layout variant?
# ----------------------------------------------------------------------------------------
READ_PREFIX = "read:"
_TAG_PATTERN = r"bench:([0-9a-f-]{36}):([A-Za-z0-9_]+):([A-Za-z0-9_]+):([0-9]+)"


def query_tag(run_id: str, experiment: str, query: str, repetition: int) -> str:
    return f"bench:{run_id}:{experiment}:{query}:{repetition}"


def tagged_sql(sql: str, tag: str) -> str:
    """Wrap a query so its text carries a unique tag without changing its result.

    The tag (1) makes every execution's SQL text unique, so no result cache can answer it,
    and (2) lets us find this exact execution in `system.query.history` afterwards. It sits
    in a string literal (always kept in the statement text) in a predicate that is always
    true; the optimiser folds it away, so the plan is the same as the untagged query.
    """
    return f"SELECT * FROM (\n{sql.strip()}\n) WHERE '{tag}' <> ''"


def run_read_benchmark(
    spark: SparkSession,
    queries: Mapping[str, str],
    variants: Sequence[Variant],
    *,
    scale: str,
    table_step: str = "gold_fact_sales",
    repetitions: int = 5,
    warmup: int = 1,
    runs_table: str | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Time each query (SQL with a `{table}` placeholder) against each variant's table.

    Execution = `collect()` of the tagged query: the results are small aggregates or one
    customer's lines, so collecting is cheap and runs the whole plan. Interleaved like the
    build benchmark. Recorded as step `read:<query>`, layer "read".
    """
    run_id = str(uuid.uuid4())
    version = spark.version
    records: list[dict[str, object]] = []
    for order, (qname, sql) in enumerate(queries.items(), start=1):
        timings: dict[str, list[Timing]] = {v.experiment: [] for v in variants}
        for rep in range(1, warmup + repetitions + 1):
            for v in variants:
                q = tagged_sql(
                    sql.format(table=v.names[table_step]),
                    query_tag(run_id, v.experiment, qname, rep),
                )
                t = repeat_timed(lambda q=q: spark.sql(q).collect(), 1, 0)[0]
                timings[v.experiment].append(Timing(rep, rep <= warmup, t.seconds, t.started_at))
                warm = " (warm-up)" if rep <= warmup else ""
                log(
                    f"[{scale}] {READ_PREFIX}{qname:<24} {v.experiment:<16} rep {rep}: {t.seconds:6.2f}s{warm}"
                )
        for v in variants:
            table = v.names[table_step]
            detail = describe_detail(spark, table)
            rows = spark.sql(sql.format(table=table)).count()  # result rows, equal across variants
            for t in timings[v.experiment]:
                records.append(_record(run_id, v, scale, READ_PREFIX + qname, order, "read", t,
                                       table, rows, detail, version, v.spec_for(table_step).describe()))  # fmt: skip
        if runs_table:
            save_runs(spark, [r for r in records if r["step"] == READ_PREFIX + qname], runs_table)
    return records


QUERY_HISTORY_COLUMNS = [
    "statement_id",
    "statement_text",
    "start_time",
    "total_duration_ms",
    "read_files",
    "pruned_files",
    "read_bytes",
    "read_rows",
    "from_result_cache",
    "read_io_cache_percent",
]


def with_tag_columns(history: DataFrame) -> DataFrame:
    """Parse run_id / experiment / query / repetition out of tagged statement texts."""
    text = F.col("statement_text")
    return (
        history.withColumn("run_id", F.regexp_extract(text, _TAG_PATTERN, 1))
        .withColumn("experiment", F.regexp_extract(text, _TAG_PATTERN, 2))
        .withColumn("query", F.regexp_extract(text, _TAG_PATTERN, 3))
        .withColumn("repetition", F.regexp_extract(text, _TAG_PATTERN, 4).cast("int"))
        .filter(F.col("run_id") != "")
    )


def query_history_for_run(spark: SparkSession, run_id: str) -> DataFrame:
    """Per-execution I/O metrics of one read benchmark run, from system.query.history.

    The system table is filled with a delay (usually a few minutes), so run this a little
    after the benchmark. Only finished statements of this run are returned.
    """
    cols = ", ".join(QUERY_HISTORY_COLUMNS)
    history = spark.sql(
        f"SELECT {cols} FROM system.query.history "
        f"WHERE statement_text LIKE '%bench:{run_id}:%' AND execution_status = 'FINISHED' "
        "AND start_time > current_timestamp() - INTERVAL 7 DAYS"
    )
    return with_tag_columns(history)


def summarise_reads(history: DataFrame, warmup: int = 1) -> DataFrame:
    """Median I/O per (experiment, query) over the measured (non-warm-up) executions.

    files_read / files_pruned come from Delta data skipping (min/max statistics per file);
    read_mb and rows_read also drop when Parquet row groups inside a file are skipped.
    """
    measured = history.filter(F.col("repetition") > warmup)
    return (
        measured.groupBy("experiment", "query")
        .agg(
            F.count(F.lit(1)).alias("n"),
            F.median("total_duration_ms").alias("server_ms"),
            F.median("read_files").alias("files_read"),
            F.median("pruned_files").alias("files_pruned"),
            F.round(F.median("read_bytes") / (1024 * 1024), 2).alias("read_mb"),
            F.median("read_rows").alias("rows_read"),
            F.max(F.col("from_result_cache").cast("int")).alias("any_result_cache"),
        )
        .orderBy("query", "experiment")
    )
