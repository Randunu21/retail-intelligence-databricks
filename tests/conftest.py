"""Shared pytest fixtures.

A single local SparkSession is created once per test run (session scope) because starting
the JVM takes a few seconds. The config keeps it small and deterministic:
- local[1]: one core -> stable row ordering in small tests
- shuffle partitions = 1: tiny DataFrames don't need 200 partitions
- UI off and UTC timezone: quieter, reproducible timestamps
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[1]")
        .appName("retail-intel-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
