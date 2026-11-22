"""Tests for optimisation.compute: every variant must produce exactly the baseline's data.

Delta writes are replaced by temp views (monkeypatched `write_table`), so the Silver variants
(staged table, cache, GROUP BY de-dup) and the Gold join variants (hints, rewrites) run
locally end to end and are compared with `fingerprint`.
"""

from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import functions as F

from optimisation import compute
from optimisation.benchmark import fingerprint
from optimisation.builds import BASELINE_STEPS, PROD_NAMES, BuildContext
from transformations.silver import stage_silver

pytestmark = pytest.mark.spark

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipelines" / "medallion" / "transformations"
COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
        "Customer ID", "Country", "_source_file"]  # fmt: skip
F1, F2 = "online_retail_II_2009_2010.csv", "online_retail_II_2010_2011.csv"


def rows():
    def r(inv, code, qty, cust, ts, src=F1, price="2.50", desc="PRODUCT"):
        return (inv, code, desc, qty, ts, price, cust, "United Kingdom", src)

    return [
        r("489434", "85048", "12", "13085", "2009-12-01 07:45:00"),
        r("489434", "85048", "12", "13085", "2009-12-01 07:45:00", src=F2),  # cross-file dup
        r("489435", "22350", "2", "", "2009-12-01 08:00:00"),
        r("489435", "22350", "2", "", "2009-12-01 08:00:00"),  # same-file dup, no customer
        r("491000", "85048", "6", "13085", "2010-01-15 10:00:00"),
        r("C489449", "22087", "-12", "16321", "2009-12-01 10:33:00"),
        r("C489449", "22087", "-12", "16321", "2009-12-01 10:33:00", src=F2),  # dup return
        r("489436", "POST", "1", "13078", "2009-12-01 09:06:00"),
        r("489437", "22350", "0", "13078", "2009-12-01 09:10:00"),  # zero quantity
        r("489438", "21232", "4", "13078", "2010-01-02 11:00:00", desc=None),
    ]


@pytest.fixture
def bronze(spark):
    schema = ", ".join(f"`{c}` STRING" for c in COLS)
    df = spark.createDataFrame(rows(), schema)
    ts = F.when(F.col("_source_file") == F1, F.lit(datetime(2026, 11, 18))).otherwise(
        F.lit(datetime(2026, 11, 19))
    )
    return df.withColumn("_ingested_at", ts)


@pytest.fixture
def local_writes(monkeypatch):
    monkeypatch.setattr(compute, "write_table", lambda df, t, s: df.createOrReplaceTempView(t))


def same(a, b):
    return fingerprint(a).first() == fingerprint(b).first()


def test_agg_dedup_equals_window_dedup(bronze):
    window = stage_silver(bronze).filter("_dup_rank = 1")
    agg = compute.stage_silver_agg(bronze)
    assert sorted(agg.columns) == sorted(window.columns)
    assert agg.count() == window.count() == 7
    assert same(agg.select(*window.columns), window)
    # the kept copy is the earliest file's, with that copy's _ingested_at
    kept = agg.filter("invoice = '489434'").first()
    assert kept._source_file == F1 and kept._ingested_at == datetime(2026, 11, 18)


@pytest.mark.parametrize("experiment", ["staged_table", "cached", "agg_dedup"])
def test_silver_variants_match_baseline(spark, bronze, local_writes, experiment):
    def run(exp):
        names = {"bronze": "s_bronze", **{k: f"s_{exp}_{k}" for k in
                 ("silver_sales", "silver_returns", "silver_staged")}}  # fmt: skip
        bronze.createOrReplaceTempView("s_bronze")
        compute.SILVER_BUILDS[exp](BuildContext(spark, names, PIPELINE_DIR))
        return spark.table(names["silver_sales"]), spark.table(names["silver_returns"])

    base_sales, base_returns = run("baseline")
    sales, returns = run(experiment)
    assert base_sales.count() == 4 and base_returns.count() == 1
    assert same(sales, base_sales) and same(returns, base_returns)


def test_silver_variants_have_own_tables():
    vs = compute.silver_variants("retail.bench.bronze_x10", "x10")
    assert [v.experiment for v in vs] == ["baseline", "staged_table", "cached", "agg_dedup"]
    assert len({v.names["silver_sales"] for v in vs}) == 4
    for v in vs:
        assert v.names[compute.SILVER_STEP] == v.names["silver_sales"]
        assert v.step_overrides[compute.SILVER_STEP] is compute.SILVER_BUILDS[v.experiment]


def test_add_hint():
    q = "WITH a AS (SELECT 1) SELECT x FROM t AS d JOIN a"
    assert compute.add_hint(q, "BROADCAST(d)", 2) == (
        "WITH a AS (SELECT 1) SELECT /*+ BROADCAST(d) */ x FROM t AS d JOIN a"
    )
    with pytest.raises(ValueError):
        compute.add_hint(q, "MERGE(d)", 3)


@pytest.fixture
def gold_base(spark, bronze):
    """Build the full baseline chain into temp views; return its names."""
    names = {logical: f"g_{logical}" for logical in PROD_NAMES}
    bronze.createOrReplaceTempView(names["bronze"])
    ctx = BuildContext(spark, names, PIPELINE_DIR)
    for step in BASELINE_STEPS:
        step.build(ctx).createOrReplaceTempView(names[step.name])
    return names


@pytest.mark.parametrize("experiment", ["broadcast", "no_broadcast", "rewrite"])
def test_gold_variants_match_baseline(spark, gold_base, experiment):
    variant = {v.experiment: v for v in compute.gold_variants(gold_base, "x1")}[experiment]
    ctx = BuildContext(spark, gold_base, PIPELINE_DIR)
    steps = {s.name: s for s in BASELINE_STEPS}
    for step in compute.GOLD_JOIN_STEPS:
        build = variant.step_overrides.get(step, steps[step].build)
        out, base = build(ctx), spark.table(gold_base[step])
        assert out.columns == base.columns, step
        # names + types must match; nullability may differ (the rewrite computes year_month
        # from invoice_ts, which Spark can't prove non-null; dim_date's column is NOT NULL)
        assert [(f.name, f.dataType) for f in out.schema] == [
            (f.name, f.dataType) for f in base.schema
        ], step
        assert same(out, base), step


def test_gold_variants_names():
    base = {k: f"base_{k}" for k in PROD_NAMES}
    vs = compute.gold_variants(base, "x10")
    assert [v.experiment for v in vs] == ["baseline", "broadcast", "no_broadcast", "rewrite"]
    for v in vs:
        assert v.names["gold_fact_sales"] == "base_gold_fact_sales"  # upstream shared
        assert v.names["gold_daily_sales"].startswith(f"retail.bench.x10_gold_{v.experiment}")


def test_plan_summary_sees_join_strategy(spark):
    # repartition: with 1 local partition Spark needs no shuffle at all
    big = spark.range(1000).withColumn("k", F.col("id") % 10).repartition(4)
    small = spark.range(10).withColumnRenamed("id", "k").repartition(3)
    bhj = compute.plan_summary(compute.plan_text(big.join(small.hint("broadcast"), "k")))
    smj = compute.plan_summary(compute.plan_text(big.join(small.hint("merge"), "k")))
    assert "BroadcastHashJoin×1" in bhj["joins"] and bhj["broadcasts"] == 1
    assert "SortMergeJoin×1" in smj["joins"] and smj["shuffles"] >= 2
    assert smj["broadcasts"] == 0


# Condensed from a real serverless plan (gold_monthly_country_sales, x10, 2026-11-22):
# 1 broadcast (dim_date, EXECUTOR_BROADCAST) + 3 shuffles (ENSURE_REQUIREMENTS).
PHOTON_PLAN = """== Physical Plan ==
AdaptiveSparkPlan (26)

(1) PhotonScan parquet retail.bench.x10_baseline__gold_fact_sales
OptionalDataFilters: [hashedrelationcontains(date_key#475589)]

(3) PhotonShuffleExchangeSink
Arguments: SinglePartition

(4) PhotonShuffleMapStage
Arguments: EXECUTOR_BROADCAST, [id=#562344]

(5) PhotonShuffleExchangeSource
Arguments: false

(6) PhotonBroadcastHashJoin
Join type: Inner

(10) PhotonShuffleExchangeSink
Arguments: hashpartitioning(year_month#475604, 218)

(11) PhotonShuffleMapStage
Arguments: ENSURE_REQUIREMENTS, [id=#562358]

(12) PhotonShuffleExchangeSource
Arguments: false

(16) PhotonShuffleMapStage
Arguments: ENSURE_REQUIREMENTS, [id=#562366]

(20) PhotonShuffleMapStage
Arguments: ENSURE_REQUIREMENTS, [id=#562372]

(26) AdaptiveSparkPlan
Arguments: isFinalPlan=false
"""


def test_plan_summary_reads_photon_plan():
    s = compute.plan_summary(PHOTON_PLAN)
    assert s["joins"] == "BroadcastHashJoin×1"
    assert (s["shuffles"], s["broadcasts"]) == (3, 1)
    assert s["runtime_filter"] is True and s["adaptive"] is True
