"""Tests for optimisation.pipeline_runs: update / flow durations from a pipeline event log."""

import json
from datetime import datetime, timedelta

import pytest

from optimisation.pipeline_runs import before_after, flow_summary, update_summary

pytestmark = pytest.mark.spark

T0 = datetime(2026, 11, 22, 10, 0, 0)
SCHEMA = (
    "timestamp TIMESTAMP, event_type STRING, "
    "origin STRUCT<update_id: STRING, flow_name: STRING>, details STRING"
)


def ev(sec, etype, update, flow=None, **details):
    return (T0 + timedelta(seconds=sec), etype, (update, flow), json.dumps(details))


def update_events(update, start, full_refresh, flows, state="COMPLETED"):
    """create_update at `start`, RUNNING at +10 s, flows (name, run seconds) one after another."""
    rows = [
        ev(start, "create_update", update, create_update={"full_refresh": full_refresh}),
        ev(start + 1, "update_progress", update, update_progress={"state": "INITIALIZING"}),
        ev(start + 10, "update_progress", update, update_progress={"state": "RUNNING"}),
    ]
    t = start + 10
    for name, secs in flows:
        rows += [
            ev(t, "flow_progress", update, name, flow_progress={"status": "QUEUED"}),
            ev(t + 1, "flow_progress", update, name, flow_progress={"status": "STARTING"}),
            ev(t + 2, "flow_progress", update, name, flow_progress={"status": "RUNNING"}),
            ev(t + 1 + secs, "flow_progress", update, name, flow_progress={"status": "COMPLETED"}),
        ]
        t += secs + 2
    rows.append(ev(t, "update_progress", update, update_progress={"state": state}))
    return rows


@pytest.fixture
def log(spark):
    rows = (
        update_events(
            "u1", 0, True, [("retail.silver.sales", 10), ("`retail`.`gold`.`fact_sales`", 4)]
        )
        + update_events(
            "u2", 100, True, [("retail.silver.sales", 12), ("retail.gold.fact_sales", 4)]
        )
        + update_events("u3", 200, False, [("retail.silver.sales", 1)])  # not a full refresh
        + update_events(
            "u4",
            300,
            True,
            [("silver_staged", 5), ("retail.silver.sales", 3), ("retail.gold.fact_sales", 4)],
        )  # fmt: skip
        + update_events("u5", 400, True, [("retail.silver.sales", 50)], state="FAILED")
        + [ev(305, "flow_progress", "u4", "missingFlowName", flow_progress={"status": "RUNNING"})]
    )
    return spark.createDataFrame(rows, SCHEMA)


def test_update_summary(log):
    u = {r.update_id: r for r in update_summary(log).collect()}
    assert u["u1"].full_refresh is True and u["u3"].full_refresh is False
    assert u["u1"].final_state == "COMPLETED" and u["u5"].final_state == "FAILED"
    # u1: create at 0, RUNNING at 10, flows 10+2 and 4+2 -> COMPLETED at 28
    assert (u["u1"].total_s, u["u1"].run_s) == (28.0, 18.0)


def test_flow_summary_strips_names_and_times_from_start(log):
    f = {(r.update_id, r.flow): r for r in flow_summary(log).collect()}
    assert f[("u1", "sales")].duration_s == 10.0  # STARTING (t+1) -> COMPLETED (t+1+10)
    assert f[("u1", "fact_sales")].duration_s == 4.0  # backticks removed
    assert f[("u4", "silver_staged")].final_status == "COMPLETED"


def test_before_after_uses_completed_full_refreshes_only(log):
    change = T0 + timedelta(seconds=250)
    rows = {
        r.flow: r for r in before_after(update_summary(log), flow_summary(log), change).collect()
    }
    sales = rows["sales"]
    assert (sales.n_before, sales.before_s) == (2, 11.0)  # u1, u2 (u3 not full, u5 failed)
    assert (sales.n_after, sales.after_s) == (1, 3.0)  # u4
    assert sales.pct_change == pytest.approx(-72.7)
    assert rows["silver_staged"].n_before == 0 and rows["silver_staged"].n_after == 1
    assert rows["(update run_s)"].n_before == 2
    assert "missingFlowName" not in rows  # no duration -> not compared
