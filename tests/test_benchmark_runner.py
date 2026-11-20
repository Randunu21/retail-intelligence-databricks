"""Run the benchmark loops end to end locally, with Delta-only calls swapped for temp views.

Checks the loop logic: interleaving, warm-up flags, separate `__post` records for
maintenance actions, `only_steps`, and the read benchmark's tagging and result counts.
"""

from pathlib import Path

import pytest
from pyspark.sql import functions as F

from optimisation import benchmark as bm
from optimisation.builds import BuildStep

pytestmark = pytest.mark.spark


@pytest.fixture
def local_delta(monkeypatch):
    """write_table -> temp view; describe_detail -> fixed numbers. Records the call order."""
    calls = []

    def fake_write(df, table, spec):
        calls.append(("write", table))
        df.createOrReplaceTempView(table)

    monkeypatch.setattr(bm, "write_table", fake_write)
    monkeypatch.setattr(bm, "describe_detail", lambda spark, t: bm.TableDetail(1, 10, "", ""))
    return calls


def steps():
    return (
        BuildStep("a", "silver", lambda c: c.spark.range(10).withColumn("k", F.col("id") % 2)),
        BuildStep("b", "gold", lambda c: c.read("a").groupBy("k").count(), ("a",)),
    )


def variants(calls):
    def post(spark, table):
        calls.append(("post", table))

    return [
        bm.Variant("v1", {"a": "v1_a", "b": "v1_b"}),
        bm.Variant("v2", {"a": "v2_a", "b": "v2_b"}, post_write={"b": post}),
    ]


def test_run_benchmark_interleaves_and_records_post(spark, local_delta):
    recs = bm.run_benchmark(spark, steps(), variants(local_delta), scale="x1",
                            pipeline_dir=Path("."), repetitions=2, warmup=1, log=lambda m: None)  # fmt: skip
    writes_b = [c for c in local_delta if c[1].endswith("_b")]
    # step b: rep1 v1, rep1 v2 (+post), rep2 v1, rep2 v2 (+post), ...
    assert writes_b[:3] == [("write", "v1_b"), ("write", "v2_b"), ("post", "v2_b")]
    by_step = {}
    for r in recs:
        by_step.setdefault((r["experiment"], r["step"]), []).append(r)
    assert len(by_step[("v1", "a")]) == 3  # 1 warm-up + 2 measured
    assert [r["is_warmup"] for r in by_step[("v1", "a")]] == [True, False, False]
    assert ("v1", "b__post") not in by_step
    post = by_step[("v2", "b__post")]
    assert len(post) == 3 and post[0]["layer"] == "post" and post[0]["output_rows"] == 2
    assert {r["run_id"] for r in recs} == {recs[0]["run_id"]}
    # records fit the Delta schema
    assert bm.records_to_df(spark, recs).count() == len(recs)


def test_run_benchmark_only_steps(spark, local_delta):
    vs = variants(local_delta)
    bm.run_benchmark(spark, steps(), vs, scale="x1", pipeline_dir=Path("."),
                     repetitions=1, warmup=0, log=lambda m: None)  # fmt: skip
    local_delta.clear()
    recs = bm.run_benchmark(spark, steps(), vs, scale="x1", pipeline_dir=Path("."),
                            repetitions=1, warmup=0, only_steps=["b"], log=lambda m: None)  # fmt: skip
    assert {r["step"] for r in recs} == {"b", "b__post"}
    assert all(c[1].endswith("_b") for c in local_delta)


def test_run_read_benchmark_tags_and_counts(spark, local_delta):
    spark.range(6).withColumn("k", F.col("id") % 3).createOrReplaceTempView("fact_1")
    spark.range(6).withColumn("k", F.col("id") % 3).createOrReplaceTempView("fact_2")
    vs = [bm.Variant("one", {"f": "fact_1"}), bm.Variant("two", {"f": "fact_2"})]
    queries = {"by_k": "SELECT k, count(*) n FROM {table} GROUP BY k",
               "k0": "SELECT id FROM {table} WHERE k = 0"}  # fmt: skip
    recs = bm.run_read_benchmark(spark, queries, vs, scale="x1", table_step="f",
                                 repetitions=2, warmup=1, log=lambda m: None)  # fmt: skip
    assert len(recs) == 2 * 2 * 3
    rows = {(r["experiment"], r["step"]): r["output_rows"] for r in recs}
    assert rows[("one", "read:by_k")] == 3 and rows[("two", "read:k0")] == 2
    assert all(r["layer"] == "read" for r in recs)


def test_run_io_probe_records_windows_and_falls_back_without_tags(spark):
    spark.range(6).withColumn("k", F.col("id") % 3).createOrReplaceTempView("io_fact")
    slept = []
    recs = bm.run_io_probe(
        spark,
        {"k0": "SELECT id FROM {table} WHERE k = 0"},
        [bm.Variant("one", {"f": "io_fact"}), bm.Variant("two", {"f": "io_fact"})],
        scale="x1",
        table_step="f",
        probe_id="fixed",
        sleep=slept.append,
        log=lambda m: None,
    )
    assert slept == [3.0, 3.0]  # a gap before every probe query
    assert [r["experiment"] for r in recs] == ["one", "two"]
    assert all(r["probe_id"] == "fixed" and r["result_rows"] == 2 for r in recs)
    assert all(r["started_at"] <= r["ended_at"] for r in recs)
    assert all(r["tags_set"] is False for r in recs)  # SET QUERY_TAGS doesn't exist locally
    assert bm.probes_to_df(spark, recs).count() == 2
