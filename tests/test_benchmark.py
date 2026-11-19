"""Tests for optimisation.benchmark: timing loop, DESCRIBE DETAIL parsing, statistics."""

from datetime import UTC, datetime

import pytest

from optimisation.benchmark import (
    RUNS_COLUMNS,
    Variant,
    WriteSpec,
    compare,
    compare_scales,
    fingerprint,
    latest_run_id,
    markdown_table,
    parse_detail,
    pct_change,
    pipeline_totals,
    records_to_df,
    repeat_timed,
    summarise,
)


def fake_clock(*ticks):
    """A clock that returns the given values in order (each run uses two ticks)."""
    it = iter(ticks)
    return lambda: next(it)


def test_repeat_timed_marks_warmup_and_measures_each_run():
    calls = []
    timings = repeat_timed(
        lambda: calls.append(1),
        repetitions=2,
        warmup=1,
        clock=fake_clock(0.0, 5.0, 10.0, 12.0, 20.0, 23.0),
        now=lambda: datetime(2026, 11, 19, tzinfo=UTC),
    )
    assert len(calls) == 3
    assert [t.is_warmup for t in timings] == [True, False, False]
    assert [t.seconds for t in timings] == [5.0, 2.0, 3.0]
    assert [t.repetition for t in timings] == [1, 2, 3]


@pytest.mark.parametrize("reps, warmup", [(0, 1), (1, -1)])
def test_repeat_timed_rejects_bad_counts(reps, warmup):
    with pytest.raises(ValueError):
        repeat_timed(lambda: None, repetitions=reps, warmup=warmup)


def test_parse_detail_picks_tracked_fields():
    d = parse_detail(
        {
            "numFiles": 4,
            "sizeInBytes": 1024,
            "partitionColumns": [],
            "clusteringColumns": ["invoice_date", "customer_id"],
            "format": "delta",
        }
    )
    assert (d.num_files, d.size_bytes) == (4, 1024)
    assert d.partition_columns == ""
    assert d.clustering_columns == "invoice_date,customer_id"


def test_pct_change():
    assert pct_change(10.0, 7.5) == -25.0
    with pytest.raises(ValueError):
        pct_change(0, 1)


def test_write_spec_rejects_partition_and_cluster_together():
    with pytest.raises(ValueError):
        WriteSpec(partition_by=("invoice_date",), cluster_by=("customer_id",))
    assert WriteSpec().describe() == "default layout"
    assert WriteSpec(cluster_by=("a", "b")).describe() == "cluster_by(a, b)"
    assert "delta.feature.timestampNtz" in WriteSpec().properties


def test_variant_default_spec():
    v = Variant("baseline", {"silver_sales": "t"}, {"x": WriteSpec(cluster_by=("a",))})
    assert v.spec_for("silver_sales") == WriteSpec()
    assert v.spec_for("x").cluster_by == ("a",)


def _record(experiment, step, order, rep, seconds, warmup=False, files=1):
    return {
        "run_id": "r1",
        "experiment": experiment,
        "scale": "x1",
        "step": step,
        "step_order": order,
        "layer": "silver",
        "repetition": rep,
        "is_warmup": warmup,
        "duration_s": seconds,
        "output_table": f"t_{experiment}_{step}",
        "output_rows": 100,
        "num_files": files,
        "size_bytes": 2 * 1024 * 1024,
        "partition_columns": "",
        "clustering_columns": "",
        "spark_version": "4.2.1",
        "notes": "",
        "started_at": datetime(2026, 11, 19, tzinfo=UTC),
    }


@pytest.fixture
def runs(spark):
    recs = [
        # baseline: warm-up 30s is ignored; measured 10, 12, 11 -> median 11
        _record("baseline", "silver_sales", 1, 1, 30.0, warmup=True),
        _record("baseline", "silver_sales", 1, 2, 10.0),
        _record("baseline", "silver_sales", 1, 3, 12.0),
        _record("baseline", "silver_sales", 1, 4, 11.0),
        _record("baseline", "gold_fact_sales", 2, 2, 4.0),
        _record("baseline", "gold_fact_sales", 2, 3, 5.0),
        # variant: clearly faster on silver, overlapping on gold
        _record("fast", "silver_sales", 1, 2, 6.0, files=3),
        _record("fast", "silver_sales", 1, 3, 5.0, files=3),
        _record("fast", "silver_sales", 1, 4, 7.0, files=3),
        _record("fast", "gold_fact_sales", 2, 2, 4.5),
        _record("fast", "gold_fact_sales", 2, 3, 4.8),
    ]
    return records_to_df(spark, recs)


def test_records_to_df_schema(runs):
    assert [f.name for f in runs.schema.fields] == [n for n, _ in RUNS_COLUMNS]


def test_summarise_excludes_warmup_and_takes_median(runs):
    s = {(r.experiment, r.step): r for r in summarise(runs).collect()}
    base = s[("baseline", "silver_sales")]
    assert base.n_runs == 3
    assert base.median_s == 11.0
    assert (base.min_s, base.max_s) == (10.0, 12.0)
    assert base.size_mb == 2.0
    assert s[("fast", "silver_sales")].num_files == 3


def test_compare_pct_change_and_noise_flag(runs):
    rows = {r.step: r for r in compare(summarise(runs), "baseline", "fast").collect()}
    silver = rows["silver_sales"]
    assert silver.median_s_before == 11.0 and silver.median_s_after == 6.0
    assert silver.pct_change == pytest.approx(-45.5)
    assert silver.ranges_overlap is False  # 5..7 vs 10..12: a real difference
    assert rows["gold_fact_sales"].ranges_overlap is True  # 4..5 vs 4.5..4.8: noise


def test_pipeline_totals_sums_medians(runs):
    totals = {r.experiment: r for r in pipeline_totals(summarise(runs)).collect()}
    assert totals["baseline"].total_median_s == pytest.approx(11.0 + 4.5)
    assert totals["baseline"].steps == 2


def test_fingerprint_ignores_row_and_column_order(spark):
    a = spark.createDataFrame([(1, "x"), (2, "y")], "id INT, v STRING")
    b = spark.createDataFrame([("y", 2), ("x", 1)], "v STRING, id INT")
    c = spark.createDataFrame([(1, "x"), (2, "z")], "id INT, v STRING")
    fa, fb, fc = (fingerprint(d).first() for d in (a, b, c))
    assert fa == fb
    assert fa.rows == fc.rows and fa.checksum != fc.checksum


def test_markdown_table():
    md = markdown_table(
        [{"step": "silver_sales", "median_s": 1.5, "x": None}], ["step", "median_s", "x"]
    )
    assert md.splitlines() == [
        "| step | median_s | x |",
        "|---|---|---|",
        "| silver_sales | 1.5 |  |",
    ]


def test_latest_run_id_prefers_full_then_recent(spark):
    def rec(run, step, day):
        r = _record("baseline", step, 1, 2, 1.0)
        return {**r, "run_id": run, "started_at": datetime(2026, 11, day, tzinfo=UTC)}

    runs = records_to_df(
        spark,
        [rec("old_full", "a", 19), rec("old_full", "b", 19), rec("new_partial", "a", 20)],
    )
    assert latest_run_id(runs, "baseline", "x1") == "old_full"
    assert latest_run_id(runs, "baseline", "x10") is None


def test_compare_scales_ratios(spark):
    recs = []
    for scale, secs, rows in (("x1", 2.0, 100), ("x10", 9.0, 1000)):
        r = {**_record("baseline", "silver_sales", 1, 2, secs), "scale": scale, "output_rows": rows}
        recs.append(r)
    out = compare_scales(summarise(records_to_df(spark, recs)), "x1", "x10").first()
    assert (out.time_ratio, out.rows_ratio) == (4.5, 10.0)
    assert out.median_s_x1 == 2.0 and out.median_s_x10 == 9.0
