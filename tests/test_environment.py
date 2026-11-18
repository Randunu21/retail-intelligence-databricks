"""Phase 0 sanity checks: the local toolchain can run PySpark and import project modules."""

import pytest


@pytest.mark.spark
def test_local_spark_runs(spark):
    df = spark.createDataFrame([("A", 2, 1.5), ("B", 3, 2.0)], ["code", "qty", "price"])
    total = df.selectExpr("sum(qty * price) AS revenue").first()["revenue"]
    assert total == pytest.approx(9.0)


def test_src_packages_importable():
    import features  # noqa: F401
    import ingestion  # noqa: F401
    import transformations  # noqa: F401
