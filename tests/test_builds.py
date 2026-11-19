"""Tests for optimisation.builds: the baseline steps reproduce the pipeline end to end.

Each step's output is registered as a temp view under the logical name, exactly as the
harness would write it to a Delta table, so the whole Silver -> Gold chain (including the
Gold SQL read from the pipeline files) runs locally on a few Bronze rows.
"""

from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import functions as F

from common.tables import BRONZE_TRANSACTIONS, bench_table
from optimisation.builds import (
    BASELINE_STEPS,
    PROD_NAMES,
    BuildContext,
    BuildStep,
    bench_names,
    check_step_order,
)

pytestmark = pytest.mark.spark

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipelines" / "medallion" / "transformations"
BRONZE_COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
               "Customer ID", "Country", "_source_file"]  # fmt: skip
F1, F2 = "online_retail_II_2009_2010.csv", "online_retail_II_2010_2011.csv"


def bronze_rows():
    def r(inv, code, qty, price, cust, ts, src=F1, desc="PRODUCT A"):
        return (inv, code, desc, qty, ts, price, cust, "United Kingdom", src)

    return [
        r("489434", "85048", "12", "6.95", "13085", "2009-12-01 07:45:00"),
        r("489434", "79323P", "12", "6.75", "13085", "2009-12-01 07:45:00"),
        r("489435", "22350", "2", "2.55", "", "2009-12-01 08:00:00"),  # no customer
        r("491000", "85048", "6", "6.95", "13085", "2010-01-15 10:00:00"),  # month 1
        r("C489449", "22087", "-12", "2.95", "16321", "2009-12-01 10:33:00"),  # return
        r("489436", "POST", "1", "18.00", "13078", "2009-12-01 09:06:00"),  # non-product
        r("489437", "22350", "2", "2.55", "13078", "2009-12-01 09:10:00"),
        r("489437", "22350", "2", "2.55", "13078", "2009-12-01 09:10:00", src=F2),  # dup
    ]


@pytest.fixture
def ctx(spark):
    schema = ", ".join(f"`{c}` STRING" for c in BRONZE_COLS)
    bronze = spark.createDataFrame(bronze_rows(), schema).withColumn(
        "_ingested_at", F.lit(datetime(2026, 11, 19))
    )
    names = {logical: f"t_{logical}" for logical in PROD_NAMES}
    bronze.createOrReplaceTempView(names["bronze"])
    return BuildContext(spark, names, PIPELINE_DIR)


def build_all(ctx):
    out = {}
    for step in BASELINE_STEPS:
        df = step.build(ctx)
        df.createOrReplaceTempView(ctx.names[step.name])  # stands in for the Delta write
        out[step.name] = df
    return out


def test_baseline_steps_are_in_dependency_order():
    check_step_order(BASELINE_STEPS)
    assert {s.name for s in BASELINE_STEPS} == set(PROD_NAMES) - {"bronze"}


def test_check_step_order_detects_missing_input():
    bad = (BuildStep("gold_fact_sales", "gold", lambda c: None, ("silver_sales",)),)
    with pytest.raises(ValueError, match="silver_sales"):
        check_step_order(bad)


def test_bench_names_uses_prefix_and_keeps_bronze():
    names = bench_names("x1_baseline")
    assert names["bronze"] == BRONZE_TRANSACTIONS
    assert names["silver_sales"] == bench_table("x1_baseline", "silver_sales")
    assert names["silver_sales"] == "retail.bench.x1_baseline__silver_sales"
    assert bench_names("x10", bronze_table="b")["bronze"] == "b"


def test_baseline_chain_runs_locally_with_expected_counts(ctx):
    out = build_all(ctx)
    # 7 non-return rows: minus POST, minus 1 duplicate -> 5 sales lines
    assert out["silver_sales"].count() == 5
    assert out["silver_returns"].count() == 1
    assert out["gold_fact_sales"].count() == 5
    assert out["gold_dim_customer"].count() == 2 + 1  # 13085, 13078 + Unknown
    assert out["gold_dim_product"].count() == 3
    # dim_date covers every day from 2009-12-01 to 2010-01-15
    assert out["gold_dim_date"].count() == 46
    assert out["gold_daily_sales"].count() == 46
    revenue = out["gold_daily_sales"].agg(F.sum("revenue")).first()[0]
    fact_revenue = out["gold_fact_sales"].agg(F.sum("line_revenue")).first()[0]
    assert revenue == fact_revenue
    retention = out["gold_customer_cohort_retention"].filter("months_since_first_purchase = 1")
    assert retention.filter("active_customers = 1").count() == 1  # 13085 came back


def test_sql_steps_read_only_mapped_tables(ctx, spark):
    """If retargeting missed a name, the query would hit retail.* (which doesn't exist locally)."""
    build_all(ctx)
    for name in ["gold_daily_sales", "gold_product_performance", "gold_monthly_country_sales"]:
        assert spark.table(ctx.names[name]).count() > 0
