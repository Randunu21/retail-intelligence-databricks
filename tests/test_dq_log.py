"""Tests for transformations.dq_log on hand-made pipeline event-log rows."""

import json
from datetime import datetime

import pytest

from transformations.dq_log import (
    DQ_LOG_COLUMNS,
    build_dq_log,
    expectation_metrics,
    output_rows_per_flow,
    rules_frame,
    update_start_times,
)
from transformations.silver import SALES_DROP_RULES, SALES_WARN_RULES

pytestmark = pytest.mark.spark

T0 = datetime(2026, 11, 19, 7, 38, 0)
T1 = datetime(2026, 11, 19, 7, 38, 30)
T2 = datetime(2026, 11, 20, 9, 0, 0)


def flow_details(expectations=None, output_rows=None, dataset="sales_validated"):
    fp = {"status": "COMPLETED"}
    if output_rows is not None:
        fp["metrics"] = {"num_output_rows": output_rows}
    if expectations is not None:
        fp["data_quality"] = {
            "dropped_records": 0,
            "expectations": [
                {"name": n, "dataset": dataset, "passed_records": p, "failed_records": f}
                for n, p, f in expectations
            ],
        }
    return json.dumps({"flow_progress": fp})


@pytest.fixture
def event_log(spark):
    schema = (
        "event_type STRING, timestamp TIMESTAMP, "
        "origin STRUCT<update_id: STRING, flow_name: STRING>, details STRING"
    )
    rows = [
        # update u1: create event, two flow_progress events for sales (summed), one for returns
        ("create_update", T0, ("u1", None), "{}"),
        (
            "flow_progress",
            T1,
            ("u1", "sales_validated"),
            flow_details([("not_duplicate", 600, 400), ("has_customer_id", 900, 100)]),
        ),
        (
            "flow_progress",
            T1,
            ("u1", "sales_validated"),
            flow_details([("not_duplicate", 400, 0), ("has_customer_id", 300, 100)]),
        ),
        (
            "flow_progress",
            T1,
            ("u1", "returns_validated"),
            flow_details([("is_product", 90, 10)], dataset="returns_validated"),
        ),
        # a flow_progress without data_quality (e.g. a table) -> no dq rows, but output rows
        ("flow_progress", T1, ("u1", "`retail`.`silver`.`sales`"), flow_details(output_rows=950)),
        # update u2 (a later run)
        ("create_update", T2, ("u2", None), "{}"),
        ("flow_progress", T2, ("u2", "sales_validated"), flow_details([("not_duplicate", 5, 1)])),
        # unrelated event type
        ("user_action", T0, (None, None), "{}"),
    ]
    return spark.createDataFrame(rows, schema)


def test_expectation_metrics_sums_events_per_update(event_log):
    got = {
        (r["update_id"], r["dataset"], r["rule"]): (r["passed_records"], r["failed_records"])
        for r in expectation_metrics(event_log).collect()
    }
    assert got == {
        ("u1", "sales_validated", "not_duplicate"): (1000, 400),
        ("u1", "sales_validated", "has_customer_id"): (1200, 200),
        ("u1", "returns_validated", "is_product"): (90, 10),
        ("u2", "sales_validated", "not_duplicate"): (5, 1),
    }


def test_update_start_time_is_first_event(event_log):
    got = {r["update_id"]: r["run_ts"] for r in update_start_times(event_log).collect()}
    assert got == {"u1": T0, "u2": T2}


def test_rules_frame_has_actions(spark):
    rules = {(r["dataset"], r["rule"]): r["action"] for r in rules_frame(spark).collect()}
    for name in SALES_DROP_RULES:
        assert rules[("sales_validated", name)] == "drop"
    for name in SALES_WARN_RULES:
        assert rules[("sales_validated", name)] == "warn"
    assert rules[("returns_validated", "negative_quantity")] == "warn"


def test_build_dq_log(spark, event_log):
    dq = build_dq_log(event_log, rules_frame(spark))
    assert dq.columns == DQ_LOG_COLUMNS
    rows = {(r["update_id"], r["dataset"], r["rule"]): r for r in dq.collect()}
    assert len(rows) == 4

    nd = rows[("u1", "sales_validated", "not_duplicate")]
    assert nd["action"] == "drop"
    assert nd["condition"] == SALES_DROP_RULES["not_duplicate"]
    assert nd["run_ts"] == T0
    assert nd["failed_pct"] == pytest.approx(28.571, abs=1e-3)  # 400 / 1400

    assert rows[("u1", "sales_validated", "has_customer_id")]["action"] == "warn"
    assert rows[("u2", "sales_validated", "not_duplicate")]["run_ts"] == T2


def test_output_rows_per_flow(event_log):
    got = {
        (r["update_id"], r["flow"]): r["output_rows"]
        for r in output_rows_per_flow(event_log).collect()
    }
    assert got[("u1", "sales")] == 950  # backticked full name -> short name


def test_origin_and_details_as_json_strings(spark):
    """The published event log may store origin as JSON text instead of a struct."""
    schema = "event_type STRING, timestamp TIMESTAMP, origin STRING, details STRING"
    rows = [
        (
            "flow_progress",
            T1,
            json.dumps({"update_id": "u9", "flow_name": "sales_validated"}),
            flow_details([("positive_price", 7, 3)]),
        )
    ]
    dq = build_dq_log(spark.createDataFrame(rows, schema), rules_frame(spark)).collect()
    assert [(r["update_id"], r["rule"], r["failed_records"], r["action"]) for r in dq] == [
        ("u9", "positive_price", 3, "drop")
    ]
