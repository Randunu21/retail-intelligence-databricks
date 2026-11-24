"""Phase 2.6: compare real pipeline updates before and after the optimisations.

The pipeline's event log (published as retail.silver.medallion_event_log) records every
update and every flow (one flow per table/view):
- `create_update` event: how the update was started, incl. `full_refresh` (true when you
  click "Full refresh all"),
- `update_progress` events: the update's state over time (... RUNNING ... COMPLETED),
- `flow_progress` events: each flow's status over time (QUEUED, STARTING, RUNNING, COMPLETED).

From the timestamps we get per update: total time (first event -> COMPLETED) and run time
(RUNNING -> COMPLETED, i.e. without cluster start-up / planning), and per flow: how long it
took (first STARTING/RUNNING -> COMPLETED).

Why only FULL refreshes are compared: on a normal update a materialized view whose inputs
didn't change is skipped ("No change"), so its time says nothing about its query. A full
refresh recomputes everything, before and after alike. Each update is one sample and the
platform adds start-up noise, so this is a confirmation, not the measurement: the
measurements are the interleaved benchmarks in notebooks/02_optimisation.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from transformations.dq_log import _as_json_struct, _origin_update_id, _short_name

_DETAILS = (
    "struct<update_progress: struct<state: string>, "
    "create_update: struct<full_refresh: boolean, cause: string>, "
    "flow_progress: struct<status: string>>"
)
_END_STATES = ("COMPLETED", "FAILED", "CANCELED")


def _seconds(start: Column, end: Column) -> Column:
    return F.round((F.unix_micros(end) - F.unix_micros(start)) / 1_000_000, 3)


def _events(event_log: DataFrame) -> DataFrame:
    origin_flow = (
        F.get_json_object("origin", "$.flow_name")
        if dict(event_log.dtypes)["origin"] == "string"
        else F.col("origin.flow_name")
    )
    return event_log.select(
        "timestamp",
        "event_type",
        _origin_update_id(event_log).alias("update_id"),
        _short_name(origin_flow).alias("flow"),
        _as_json_struct(event_log, "details", _DETAILS).alias("d"),
    ).filter(F.col("update_id").isNotNull())


def update_summary(event_log: DataFrame) -> DataFrame:
    """One row per update: started_at, full_refresh, final_state, total_s, run_s."""
    e = _events(event_log)
    state = F.col("d.update_progress.state")
    agg = e.groupBy("update_id").agg(
        F.min("timestamp").alias("started_at"),
        F.max(F.col("d.create_update.full_refresh")).alias("full_refresh"),
        F.min(F.when(state == "RUNNING", F.col("timestamp"))).alias("_running_at"),
        F.max(F.when(state.isin(*_END_STATES), F.col("timestamp"))).alias("_ended_at"),
        F.max_by(state, F.when(state.isNotNull(), F.col("timestamp"))).alias("final_state"),
    )
    return (
        agg.withColumn("full_refresh", F.coalesce("full_refresh", F.lit(False)))
        .withColumn("total_s", _seconds(F.col("started_at"), F.col("_ended_at")))
        .withColumn("run_s", _seconds(F.col("_running_at"), F.col("_ended_at")))
        .drop("_running_at", "_ended_at")
        .orderBy(F.desc("started_at"))
    )


def flow_summary(event_log: DataFrame) -> DataFrame:
    """One row per (update, flow): final_status and duration_s (first STARTING/RUNNING ->
    COMPLETED)."""
    e = _events(event_log).filter(F.col("event_type") == "flow_progress")
    status = F.col("d.flow_progress.status")
    agg = e.groupBy("update_id", "flow").agg(
        F.min(F.when(status.isin("STARTING", "RUNNING"), F.col("timestamp"))).alias("_start"),
        F.max(F.when(status == "COMPLETED", F.col("timestamp"))).alias("_end"),
        F.max_by(status, F.when(status.isNotNull(), F.col("timestamp"))).alias("final_status"),
    )
    return agg.withColumn("duration_s", _seconds(F.col("_start"), F.col("_end"))).drop(
        "_start", "_end"
    )


def before_after(updates: DataFrame, flows: DataFrame, change_ts: datetime | str) -> DataFrame:
    """Median duration per flow (and for the whole update) before vs after `change_ts`.

    Only COMPLETED full-refresh updates count. Columns: flow, n_before, before_s, n_after,
    after_s, pct_change. The row `(update run_s)` is the update's run time.
    """
    ok = updates.filter((F.col("final_state") == "COMPLETED") & F.col("full_refresh"))
    period = F.when(F.col("started_at") < F.lit(change_ts).cast("timestamp"), "before").otherwise(
        "after"
    )
    labelled = ok.withColumn("period", period).select("update_id", "period", "run_s")
    # Events not tied to a dataset (shown as "missingFlowName") have no duration: skip them.
    per_flow = (
        flows.filter(F.col("duration_s").isNotNull())
        .join(labelled, "update_id")
        .select("flow", "period", "duration_s")
    )
    whole = labelled.select(F.lit("(update run_s)").alias("flow"), "period",
                            F.col("run_s").alias("duration_s"))  # fmt: skip
    stats = (
        per_flow.unionByName(whole)
        .groupBy("flow")
        .agg(
            F.count(F.when(F.col("period") == "before", 1)).alias("n_before"),
            F.median(F.when(F.col("period") == "before", F.col("duration_s"))).alias("before_s"),
            F.count(F.when(F.col("period") == "after", 1)).alias("n_after"),
            F.median(F.when(F.col("period") == "after", F.col("duration_s"))).alias("after_s"),
        )
    )
    return stats.withColumn(
        "pct_change", F.round((F.col("after_s") - F.col("before_s")) / F.col("before_s") * 100, 1)
    ).orderBy("flow")
