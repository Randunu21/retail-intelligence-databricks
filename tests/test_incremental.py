"""Tests for optimisation.incremental: initial load + increments == full rebuild, re-runs are no-ops.

`MemoryBackend` imitates the Delta operations the loader needs (insert-only MERGE, upsert,
table versions, Change Data Feed, checkpoints) with in-memory DataFrames + temp views, so
the SAME loader code that runs on Databricks runs here on a handful of rows.
"""

from datetime import datetime
from functools import reduce
from pathlib import Path

import pytest
from pyspark.sql import functions as F

from optimisation import incremental as inc
from optimisation.benchmark import fingerprint
from optimisation.builds import BASELINE_STEPS, PROD_NAMES, BuildContext

pytestmark = pytest.mark.spark

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipelines" / "medallion" / "transformations"
COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
        "Customer ID", "Country", "_source_file"]  # fmt: skip
F1, F2 = "online_retail_II_2009_2010.csv", "online_retail_II_2010_2011.csv"
BATCHES = (
    ("initial", None, "2009-12-03"),
    ("dec-2009", "2009-12-03", "2010-01-01"),
    ("jan-2010", "2010-01-01", None),
)


def rows():
    def r(inv, code, qty, cust, ts, src=F1, price="2.50", desc="PRODUCT"):
        return (inv, code, desc, qty, ts, price, cust, "United Kingdom", src)

    return [
        # initial batch (before 2009-12-03)
        r("489434", "85048", "12", "13085", "2009-12-01 07:45:00"),
        r("489434", "85048", "12", "13085", "2009-12-01 07:45:00", src=F2),  # cross-file dup
        r("489435", "22350", "2", "", "2009-12-01 08:00:00"),  # no customer
        r("C489449", "22087", "-12", "16321", "2009-12-02 10:33:00"),  # return
        # dec-2009 increment: 12-03 and 12-04 have no sales -> zero-sales days in daily
        r("489500", "21232", "4", "16000", "2009-12-05 11:00:00"),
        r("489500", "21232", "4", "16000", "2009-12-05 11:00:00"),  # same-file dup
        # jan-2010 increment
        r("491000", "85048", "6", "13085", "2010-01-15 10:00:00"),
        r("C491001", "85048", "-1", "13085", "2010-01-16 09:00:00"),
        r("491002", "POST", "1", "13085", "2010-01-16 09:05:00"),  # non-product
    ]


@pytest.fixture
def bronze(spark):
    schema = ", ".join(f"`{c}` STRING" for c in COLS)
    return spark.createDataFrame(rows(), schema).withColumn(
        "_ingested_at", F.lit(datetime(2026, 11, 22))
    )


class MemoryBackend:
    """In-memory stand-in for DeltaBackend (tables = materialised temp views)."""

    def __init__(self, spark):
        self.spark = spark
        self.versions: dict[str, int] = {}
        self.change_log: dict[str, list[tuple[int, object]]] = {}
        self.checkpoints: dict[str, int] = {}

    def _store(self, table, df):
        frozen = self.spark.createDataFrame(df.collect(), df.schema)  # no lazy self-reference
        frozen.createOrReplaceTempView(table)
        self.versions[table] = self.versions.get(table, -1) + 1
        return frozen

    def create(self, table, df, spec):
        self._store(table, df)

    def insert_new(self, table, source, keys, target_filter=None):
        target = self.spark.table(table)
        cond = reduce(lambda a, b: a & b,
                      [F.col(f"s.{k}").eqNullSafe(F.col(f"t.{k}")) for k in keys])  # fmt: skip
        new = source.alias("s").join(target.alias("t"), cond, "left_anti")
        new = self.spark.createDataFrame(new.collect(), source.schema)
        self._store(table, target.unionByName(new))
        self.change_log.setdefault(table, []).append((self.versions[table], new))
        return new.count()

    def upsert(self, table, source, keys):
        target = self.spark.table(table)
        kept = target.join(source.select(*keys), list(keys), "left_anti")
        self._store(table, kept.unionByName(source))
        return source.count()

    def version(self, table):
        return self.versions[table]

    def changes_since(self, table, version):
        parts = [df for v, df in self.change_log.get(table, []) if v > version]
        empty = self.spark.table(table).limit(0)
        return reduce(lambda a, b: a.unionByName(b), parts, empty)

    def get_checkpoint(self, name):
        return self.checkpoints.get(name)

    def set_checkpoint(self, name, version):
        self.checkpoints[name] = version


def full_rebuild(spark, bronze):
    names = {logical: f"full_{logical}" for logical in PROD_NAMES}
    bronze.createOrReplaceTempView(names["bronze"])
    ctx = BuildContext(spark, names, PIPELINE_DIR)
    for step in BASELINE_STEPS:
        step.build(ctx).createOrReplaceTempView(names[step.name])
    return names


def incremental_load(spark, bronze):
    names = {logical: f"inc_{logical}" for logical in (*inc.INCREMENTAL_TABLES, "bronze")}
    bronze.createOrReplaceTempView(names["bronze"])
    loader = inc.IncrementalLoader(MemoryBackend(spark), names, PIPELINE_DIR)
    loader.initial_load(BATCHES[0][2])
    for name, lo, hi in BATCHES[1:]:
        loader.apply_batch(name, lo, hi)
    return loader


def fp(spark, table):
    return fingerprint(spark.table(table)).first()


def test_incremental_equals_full_rebuild(spark, bronze):
    full = full_rebuild(spark, bronze)
    loader = incremental_load(spark, bronze)
    for table in inc.INCREMENTAL_TABLES:
        assert fp(spark, loader.names[table]) == fp(spark, full[table]), table
    # sanity: the zero-sales days 12-02..12-04 exist in daily_sales with revenue 0
    zero = spark.table(loader.names["gold_daily_sales"]).filter(
        "date_key BETWEEN 20091202 AND 20091204"
    )
    assert [r.revenue for r in zero.collect()] == [0, 0, 0]


def test_increment_counts(spark, bronze):
    log = {e["batch"]: e for e in incremental_load(spark, bronze).log}
    assert log["dec-2009"]["silver_rows_inserted"] == 1  # the same-file dup is removed
    assert log["dec-2009"]["fact_rows_inserted"] == 1
    # 12-05 (new sale) + 12-02..12-04 (calendar grew; 12-02 only had a return, which isn't a
    # sale, so the initial calendar ended on 12-01)
    assert log["dec-2009"]["days_recomputed"] == 4
    assert log["jan-2010"]["silver_rows_inserted"] == 1  # POST dropped
    assert log["jan-2010"]["return_rows_inserted"] == 1


def test_rerunning_a_batch_changes_nothing(spark, bronze):
    loader = incremental_load(spark, bronze)
    before = {t: fp(spark, loader.names[t]) for t in inc.INCREMENTAL_TABLES}
    again = loader.apply_batch(*BATCHES[-1])
    assert (again["silver_rows_inserted"], again["return_rows_inserted"],
            again["fact_rows_inserted"], again["days_recomputed"]) == (0, 0, 0, 0)  # fmt: skip
    assert {t: fp(spark, loader.names[t]) for t in inc.INCREMENTAL_TABLES} == before


def test_bronze_batch_bounds(bronze):
    counts = [inc.bronze_batch(bronze, lo, hi).count() for _, lo, hi in BATCHES]
    assert counts == [4, 2, 3]
    assert sum(counts) == bronze.count()


def test_affected_days(spark):
    fact = spark.createDataFrame([(20091205,), (20091205,)], "date_key INT")
    dim = spark.createDataFrame([(d,) for d in (20091201, 20091203, 20091205)], "date_key INT")
    daily = spark.createDataFrame([(20091201,), (20091205,)], "date_key INT")
    assert sorted(r.date_key for r in inc.affected_days(fact, dim, daily).collect()) == [
        20091203, 20091205]  # fmt: skip


def test_merge_sql_text():
    sql = inc.merge_insert_sql("db.t", "src", ["a", "b"], "t.d >= DATE'2011-10-01'")
    assert sql == (
        "MERGE INTO db.t AS t USING src AS s ON (t.d >= DATE'2011-10-01') AND "
        "t.a <=> s.a AND t.b <=> s.b WHEN NOT MATCHED THEN INSERT *"
    )
    up = inc.merge_upsert_sql("db.d", "src", ["date_key"])
    assert "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *" in up
    assert inc.silver_bounds_filter("2011-10-01", None) == "t.invoice_date >= DATE'2011-10-01'"
    assert inc.fact_bounds_filter("2011-10-01", "2011-11-01") == (
        "t.date_key >= 20111001 AND t.date_key < 20111101"
    )
    assert inc.fact_bounds_filter(None, None) is None


def test_default_batches_cover_everything_once():
    his = [hi for _, _, hi in inc.DEFAULT_BATCHES[:-1]]
    los = [lo for _, lo, _ in inc.DEFAULT_BATCHES[1:]]
    assert his == los  # contiguous, no gaps or overlaps
    assert inc.DEFAULT_BATCHES[0][1] is None and inc.DEFAULT_BATCHES[-1][2] is None


def test_log_rows_fit_schema(spark, bronze):
    loader = incremental_load(spark, bronze)
    df = spark.createDataFrame(inc.log_rows(loader.log), inc.LOG_SCHEMA)
    first = df.first()
    assert first.batch == "initial" and first.silver_s is None and first.total_s >= 0
    assert df.count() == len(BATCHES)
