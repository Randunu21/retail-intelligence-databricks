"""Phase 2.5: incremental loading with MERGE INTO + Change Data Feed, instead of full rebuilds.

The idea
--------
Today every pipeline run rebuilds Silver and Gold from ALL of Bronze, even if only one
day of new sales arrived. Incremental processing touches only what changed:

    new Bronze rows (one month)
      │ stage_silver() on the batch only (typing + de-dup within the batch)
      ▼
    MERGE INTO silver_sales  ON all business columns   WHEN NOT MATCHED THEN INSERT
      │ Change Data Feed: "which rows were inserted since version N?"
      ▼
    MERGE INTO fact_sales    ON line_id                WHEN NOT MATCHED THEN INSERT
      │ affected days = days with new fact rows + new calendar days
      ▼
    MERGE INTO daily_sales   ON date_key   WHEN MATCHED UPDATE / NOT MATCHED INSERT
    (dim_date: 739 rows, simply rebuilt)

Why it's safe to re-run (idempotent)
------------------------------------
- Silver/fact MERGEs only INSERT rows whose key isn't there yet. Re-running a batch
  finds every key already present -> inserts nothing.
- The fact step reads Silver's Change Data Feed from a stored **checkpoint** (last Silver
  version processed). After a successful run the checkpoint moves forward, so the same
  changes are never applied twice; if a run fails before saving it, the next run re-reads
  them, and the insert-only MERGE makes that harmless.
- daily_sales rows are recomputed from the fact table and UPSERTed, so recomputing a day
  twice gives the same row.

Why results equal a full rebuild
--------------------------------
Duplicates (same business columns) share the same invoice timestamp, so both copies always
land in the same monthly batch, where the window de-dup keeps the earliest file's copy,
exactly like the full rebuild. Days are recomputed from the full fact table, so a day's
row is the same whichever batch touched it. Zero-sales days that become part of the
calendar (dim_date grows) are added explicitly; a test caught that case.

Not covered (and why): dim_product / dim_customer pick "most frequent" values over all
history, so a new batch can change an existing row's description or country. That needs
either a full recompute of those small tables or SCD handling; out of scope here.

Backends
--------
`IncrementalLoader` holds the logic. A backend does the storage operations:
`DeltaBackend` (Databricks: real MERGE, CDF, a checkpoint table) and, in the tests,
an in-memory imitation. The same loader code runs on both.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from common.tables import GOLD_DIM_DATE, GOLD_FACT_SALES, SILVER_SALES
from optimisation.benchmark import DEFAULT_TABLE_PROPERTIES, WriteSpec, write_table
from optimisation.pipeline_sql import load_query
from transformations.silver import (
    BUSINESS_COLUMNS,
    RETURNS_DROP_RULES,
    SALES_DROP_RULES,
    apply_drop_rules,
    stage_silver,
    to_silver_output,
)

# Monthly batches over Bronze's InvoiceDate text ("yyyy-MM-dd HH:mm:ss" sorts correctly as
# text). Initial load = everything before October 2011; then three monthly increments.
Batch = tuple[str, str | None, str | None]  # (name, from incl., to excl.) as 'yyyy-MM-dd'
DEFAULT_BATCHES: tuple[Batch, ...] = (
    ("initial", None, "2011-10-01"),
    ("2011-10", "2011-10-01", "2011-11-01"),
    ("2011-11", "2011-11-01", "2011-12-01"),
    ("2011-12", "2011-12-01", None),
)
CDF_PROPERTIES = {**DEFAULT_TABLE_PROPERTIES, "delta.enableChangeDataFeed": "true"}
INCREMENTAL_TABLES = ("silver_sales", "silver_returns", "gold_fact_sales",
                      "gold_dim_date", "gold_daily_sales")  # fmt: skip


LOG_COLUMNS: list[tuple[str, str]] = [
    ("batch", "STRING"),
    ("silver_rows_inserted", "BIGINT"),
    ("return_rows_inserted", "BIGINT"),
    ("fact_rows_inserted", "BIGINT"),
    ("days_recomputed", "BIGINT"),
    ("daily_rows_upserted", "BIGINT"),
    ("silver_s", "DOUBLE"),
    ("fact_s", "DOUBLE"),
    ("gold_s", "DOUBLE"),
    ("total_s", "DOUBLE"),
]
LOG_SCHEMA = ", ".join(f"{n} {d}" for n, d in LOG_COLUMNS)


# ----------------------------------------------------------------------------------------
# Pure helpers (tested locally)
# ----------------------------------------------------------------------------------------
def log_rows(log: Sequence[Mapping[str, object]]) -> list[tuple]:
    """Loader log entries -> tuples matching LOG_SCHEMA (missing values become NULL; the
    initial load has only a total time)."""
    out = []
    for entry in log:
        row = []
        for name, dtype in LOG_COLUMNS:
            v = entry.get(name)
            row.append(None if v is None else float(v) if dtype == "DOUBLE"
                       else int(v) if dtype == "BIGINT" else str(v))  # fmt: skip
        out.append(tuple(row))
    return out


def bronze_batch(bronze: DataFrame, lo: str | None, hi: str | None) -> DataFrame:
    """Bronze rows whose InvoiceDate is in [lo, hi) (either bound may be open)."""
    ts = F.trim(F.col("InvoiceDate"))
    cond = F.lit(True)
    if lo:
        cond = cond & (ts >= F.lit(lo))
    if hi:
        cond = cond & (ts < F.lit(hi))
    return bronze.filter(cond)


def silver_outputs(bronze_rows: DataFrame) -> tuple[DataFrame, DataFrame]:
    """(sales, returns) for a set of Bronze rows: the pipeline's Silver logic, staged once."""
    staged = stage_silver(bronze_rows)
    sales = to_silver_output(apply_drop_rules(staged, SALES_DROP_RULES))
    returns = to_silver_output(
        apply_drop_rules(staged.filter("is_cancellation"), RETURNS_DROP_RULES)
    )
    return sales, returns


def merge_insert_sql(
    target: str, source: str, keys: Sequence[str], target_filter: str | None = None
) -> str:
    """Insert-only MERGE. Keys compared NULL-safe (<=>): customer_id/description can be NULL,
    and a plain `=` would never match NULLs, so re-runs would insert them again.
    `target_filter` (on alias t) limits which target files must be read (file pruning)."""
    on = " AND ".join(f"t.{k} <=> s.{k}" for k in keys)
    if target_filter:
        on = f"({target_filter}) AND {on}"
    return f"MERGE INTO {target} AS t USING {source} AS s ON {on} WHEN NOT MATCHED THEN INSERT *"


def merge_upsert_sql(target: str, source: str, keys: Sequence[str]) -> str:
    """Update matching rows, insert new ones (for recomputed aggregates)."""
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    return (
        f"MERGE INTO {target} AS t USING {source} AS s ON {on} "
        "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"
    )


def silver_bounds_filter(lo: str | None, hi: str | None) -> str | None:
    parts = ([f"t.invoice_date >= DATE'{lo}'"] if lo else []) + (
        [f"t.invoice_date < DATE'{hi}'"] if hi else []
    )
    return " AND ".join(parts) or None


def fact_bounds_filter(lo: str | None, hi: str | None) -> str | None:
    parts = ([f"t.date_key >= {lo.replace('-', '')}"] if lo else []) + (
        [f"t.date_key < {hi.replace('-', '')}"] if hi else []
    )
    return " AND ".join(parts) or None


def affected_days(new_fact_rows: DataFrame, dim_date: DataFrame, daily: DataFrame) -> DataFrame:
    """date_keys whose daily_sales row must be (re)computed: days with new fact rows, plus
    calendar days not in daily_sales yet (new zero-sales days inside a grown date range)."""
    changed = new_fact_rows.select("date_key")
    missing = dim_date.select("date_key").join(daily.select("date_key"), "date_key", "left_anti")
    return changed.union(missing).distinct()


def restrict_to_keys(query: str, key: str, keys_view: str) -> str:
    """Wrap a query so only rows whose `key` is in `keys_view` come out. The filter is on a
    GROUP BY key, so Spark pushes it down below the aggregation into the scans."""
    return f"SELECT * FROM (\n{query}\n) WHERE {key} IN (SELECT {key} FROM {keys_view})"


# ----------------------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------------------
class Backend(Protocol):
    spark: SparkSession

    def create(self, table: str, df: DataFrame, spec: WriteSpec) -> None: ...
    def insert_new(self, table: str, source: DataFrame, keys: Sequence[str],
                   target_filter: str | None = None) -> int: ...  # fmt: skip
    def upsert(self, table: str, source: DataFrame, keys: Sequence[str]) -> int: ...
    def version(self, table: str) -> int: ...
    def changes_since(self, table: str, version: int) -> DataFrame: ...
    def get_checkpoint(self, name: str) -> int | None: ...
    def set_checkpoint(self, name: str, version: int) -> None: ...


@dataclass
class DeltaBackend:
    """Databricks: real MERGE INTO, Change Data Feed and a checkpoint table."""

    spark: SparkSession
    checkpoint_table: str
    _views: int = 0

    def _view(self, df: DataFrame) -> str:
        self._views += 1
        name = f"_inc_src_{self._views}"
        df.createOrReplaceTempView(name)
        return name

    def create(self, table: str, df: DataFrame, spec: WriteSpec) -> None:
        write_table(df, table, spec)

    def _merge(self, sql: str, metric: str) -> int:
        row = self.spark.sql(sql).first()
        return int((row.asDict() if row else {}).get(metric) or 0)

    def insert_new(self, table, source, keys, target_filter=None) -> int:
        sql = merge_insert_sql(table, self._view(source), keys, target_filter)
        return self._merge(sql, "num_inserted_rows")

    def upsert(self, table, source, keys) -> int:
        return self._merge(merge_upsert_sql(table, self._view(source), keys), "num_affected_rows")

    def version(self, table: str) -> int:
        return int(self.spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first().version)

    def changes_since(self, table: str, version: int) -> DataFrame:
        """Rows inserted/updated after `version` (CDF), without the CDF metadata columns."""
        if self.version(table) <= version:  # nothing new: CDF would reject the start version
            return self.spark.read.table(table).limit(0)
        cdf = (
            self.spark.read.option("readChangeFeed", "true")
            .option("startingVersion", version + 1)
            .table(table)
        )
        return cdf.filter(F.col("_change_type").isin("insert", "update_postimage")).drop(
            "_change_type", "_commit_version", "_commit_timestamp"
        )

    def _ensure_checkpoints(self) -> None:
        self.spark.sql(
            f"CREATE TABLE IF NOT EXISTS {self.checkpoint_table} "
            "(name STRING, last_version BIGINT, updated_at TIMESTAMP)"
        )

    def get_checkpoint(self, name: str) -> int | None:
        self._ensure_checkpoints()
        row = self.spark.read.table(self.checkpoint_table).filter(F.col("name") == name).first()
        return None if row is None else int(row.last_version)

    def set_checkpoint(self, name: str, version: int) -> None:
        self._ensure_checkpoints()
        src = self.spark.createDataFrame([(name, version)], "name STRING, last_version BIGINT")
        self.upsert(
            self.checkpoint_table, src.withColumn("updated_at", F.current_timestamp()), ["name"]
        )


# ----------------------------------------------------------------------------------------
# The loader
# ----------------------------------------------------------------------------------------
FACT_CHECKPOINT = "gold_fact_sales<-silver_sales"


@dataclass
class IncrementalLoader:
    """Initial load + monthly increments for silver_sales/returns, fact_sales, dim_date,
    daily_sales. `names` maps those logical names (+ "bronze") to tables."""

    backend: Backend
    names: Mapping[str, str]
    pipeline_dir: Path
    log: list[dict[str, object]] = field(default_factory=list)

    @property
    def spark(self) -> SparkSession:
        return self.backend.spark

    def _query(self, file_name: str, mapping: Mapping[str, str]) -> DataFrame:
        return self.spark.sql(load_query(self.pipeline_dir, file_name, mapping))

    def _fact_from(self, silver_rows: DataFrame) -> DataFrame:
        view = "_inc_silver_rows"
        silver_rows.createOrReplaceTempView(view)
        return self._query("gold_fact_sales.sql", {SILVER_SALES: view})

    def _dim_date(self) -> DataFrame:
        return self._query("gold_dim_date.sql", {SILVER_SALES: self.names["silver_sales"]})

    def _daily_query(self) -> str:
        mapping = {GOLD_DIM_DATE: self.names["gold_dim_date"],
                   GOLD_FACT_SALES: self.names["gold_fact_sales"]}  # fmt: skip
        return load_query(self.pipeline_dir, "gold_daily_sales.sql", mapping)

    def initial_load(self, hi: str | None) -> dict[str, object]:
        """Create every table from the Bronze rows before `hi` (a normal full build)."""
        n, t0 = self.names, time.perf_counter()
        sales, returns = silver_outputs(bronze_batch(self.spark.read.table(n["bronze"]), None, hi))
        silver_spec = WriteSpec(cluster_by=("invoice_date",), properties=CDF_PROPERTIES)
        self.backend.create(n["silver_sales"], sales, silver_spec)
        self.backend.create(n["silver_returns"], returns, silver_spec)
        self.backend.create(n["gold_fact_sales"], self._fact_from(self.spark.read.table(n["silver_sales"])),
                            WriteSpec(cluster_by=("date_key", "customer_key")))  # fmt: skip
        self.backend.set_checkpoint(FACT_CHECKPOINT, self.backend.version(n["silver_sales"]))
        self.backend.create(n["gold_dim_date"], self._dim_date(), WriteSpec())
        self.backend.create(n["gold_daily_sales"], self.spark.sql(self._daily_query()), WriteSpec())
        entry = {"batch": "initial", "total_s": round(time.perf_counter() - t0, 3)}
        self.log.append(entry)
        return entry

    def apply_batch(self, name: str, lo: str | None, hi: str | None) -> dict[str, object]:
        """Load one increment. Safe to re-run: a repeated batch changes nothing."""
        n, b, timings = self.names, self.backend, {}
        clock = time.perf_counter

        t = clock()  # 1. Silver: stage the batch only, insert rows not seen before
        sales, returns = silver_outputs(bronze_batch(self.spark.read.table(n["bronze"]), lo, hi))
        bounds = silver_bounds_filter(lo, hi)
        sales_new = b.insert_new(n["silver_sales"], sales, BUSINESS_COLUMNS, bounds)
        returns_new = b.insert_new(n["silver_returns"], returns, BUSINESS_COLUMNS, bounds)
        timings["silver_s"] = clock() - t

        t = clock()  # 2. Fact: only Silver rows inserted since the checkpoint (CDF)
        last = b.get_checkpoint(FACT_CHECKPOINT)
        now = b.version(n["silver_sales"])
        new_fact = self._fact_from(b.changes_since(n["silver_sales"], last))
        fact_new = b.insert_new(
            n["gold_fact_sales"], new_fact, ["line_id"], fact_bounds_filter(lo, hi)
        )
        b.set_checkpoint(FACT_CHECKPOINT, now)
        timings["fact_s"] = clock() - t

        t = clock()  # 3. Calendar (tiny) rebuilt; daily rows recomputed for affected days only
        b.create(n["gold_dim_date"], self._dim_date(), WriteSpec())
        days = affected_days(new_fact, self.spark.read.table(n["gold_dim_date"]),
                             self.spark.read.table(n["gold_daily_sales"]))  # fmt: skip
        days.createOrReplaceTempView("_inc_days")
        n_days = days.count()
        daily = self.spark.sql(restrict_to_keys(self._daily_query(), "date_key", "_inc_days"))
        days_upserted = b.upsert(n["gold_daily_sales"], daily, ["date_key"]) if n_days else 0
        timings["gold_s"] = clock() - t

        entry = {
            "batch": name,
            "silver_rows_inserted": sales_new,
            "return_rows_inserted": returns_new,
            "fact_rows_inserted": fact_new,
            "days_recomputed": n_days,
            "daily_rows_upserted": days_upserted,
            **{k: round(v, 3) for k, v in timings.items()},
            "total_s": round(sum(timings.values()), 3),
        }
        self.log.append(entry)
        return entry
