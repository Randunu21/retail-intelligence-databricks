"""Tests for optimisation.scaling: synthetic replicas behave like new customers/invoices."""

from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import functions as F

from optimisation.builds import BASELINE_STEPS, PROD_NAMES, BuildContext
from optimisation.scaling import (
    REPLICA_CUSTOMER_OFFSET,
    SCALING_EXPECTATIONS,
    expected_rows,
    replicate_bronze,
)

pytestmark = pytest.mark.spark

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipelines" / "medallion" / "transformations"
COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
        "Customer ID", "Country", "_source_file"]  # fmt: skip
F1, F2 = "online_retail_II_2009_2010.csv", "online_retail_II_2010_2011.csv"


def rows():
    def r(inv, code, qty, cust, ts, src=F1, price="2.50"):
        return (inv, code, "PRODUCT", qty, ts, price, cust, "United Kingdom", src)

    return [
        r("489434", "85048", "12", "13085", "2009-12-01 07:45:00"),
        r("489435", "22350", "2", "", "2009-12-01 08:00:00"),  # no customer
        r("491000", "85048", "6", "13085", "2010-01-15 10:00:00"),
        r("C489449", "22087", "-12", "16321", "2009-12-01 10:33:00"),  # cancellation
        r("A506401", "B", "1", "", "2010-04-29 13:36:00", price="-53594.36"),  # adjustment
        r("489437", "22350", "2", "13078", "2009-12-01 09:10:00"),
        r("489437", "22350", "2", "13078", "2009-12-01 09:10:00", src=F2),  # cross-file dup
    ]


@pytest.fixture
def bronze(spark):
    schema = ", ".join(f"`{c}` STRING" for c in COLS)
    return spark.createDataFrame(rows(), schema).withColumn(
        "_ingested_at", F.lit(datetime(2026, 11, 20))
    )


def test_replica_zero_is_the_original(bronze):
    x3 = replicate_bronze(bronze, 3)
    original = x3.filter("_replica = 0").drop("_replica")
    assert original.columns == bronze.columns
    assert sorted(original.collect()) == sorted(bronze.collect())
    assert x3.count() == 3 * bronze.count()


def test_keys_are_shifted_and_prefixes_kept(bronze):
    r2 = {
        (r.Invoice, r["Customer ID"])
        for r in replicate_bronze(bronze, 3).filter("_replica = 2").collect()
    }
    assert ("R02-489434", str(13085 + 2 * REPLICA_CUSTOMER_OFFSET)) in r2
    assert ("CR02-489449", str(16321 + 2 * REPLICA_CUSTOMER_OFFSET)) in r2
    assert ("AR02-506401", "") in r2  # empty customer stays empty
    assert ("R02-489435", "") in r2


@pytest.mark.parametrize("factor", [0, 21])
def test_factor_bounds(bronze, factor):
    with pytest.raises(ValueError):
        replicate_bronze(bronze, factor)


def test_expected_rows():
    assert expected_rows("silver_sales", 1_003_357, 10) == 10_033_570
    assert expected_rows("gold_dim_customer", 5_853, 10) == 58_521
    assert expected_rows("gold_dim_date", 739, 10) == 739
    assert set(SCALING_EXPECTATIONS) == set(PROD_NAMES) - {"bronze"}


def _build_all(spark, bronze_df, prefix):
    names = {logical: f"{prefix}_{logical}" for logical in PROD_NAMES}
    bronze_df.createOrReplaceTempView(names["bronze"])
    ctx = BuildContext(spark, names, PIPELINE_DIR)
    counts = {}
    for step in BASELINE_STEPS:
        df = step.build(ctx)
        df.createOrReplaceTempView(names[step.name])
        counts[step.name] = df.count()
    return counts


def test_scaled_pipeline_meets_scaling_expectations(spark, bronze):
    """The whole Silver -> Gold chain on x3 matches expected_rows for every table."""
    x1 = _build_all(spark, bronze, "s1")
    x3 = _build_all(spark, replicate_bronze(bronze, 3), "s3")
    for table, n in x1.items():
        assert x3[table] == expected_rows(table, n, 3), table
