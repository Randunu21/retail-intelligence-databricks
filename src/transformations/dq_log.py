"""Build `retail.silver.dq_log` from the Lakeflow pipeline event log.

The pipeline records, for every expectation on every update (run), how many records passed
and failed. That lives in `flow_progress` events as JSON inside the `details` column. This
module turns it into one tidy row per (run, dataset, rule) and adds what the event log does
not know: the rule's ACTION (drop / warn) and its SQL condition, taken from the same rule
dictionaries the pipeline uses.

Why Python and not a SQL file: the rule actions come from Python constants
(`SALES_DROP_RULES`, ...), and the parsing logic is unit-tested on hand-made event rows.

dq_log columns:
    run_ts, update_id, dataset, rule, action, condition,
    passed_records, failed_records, failed_pct, logged_at
"""

from __future__ import annotations

from collections.abc import Mapping

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from transformations.silver import (
    RETURNS_DROP_RULES,
    RETURNS_WARN_RULES,
    SALES_DROP_RULES,
    SALES_WARN_RULES,
)

# dataset name in the pipeline -> (drop rules, warn rules)
RULE_SETS: dict[str, tuple[Mapping[str, str], Mapping[str, str]]] = {
    "sales_validated": (SALES_DROP_RULES, SALES_WARN_RULES),
    "returns_validated": (RETURNS_DROP_RULES, RETURNS_WARN_RULES),
}

# Only the parts of `details` we need. Unknown JSON fields are ignored by from_json.
_DETAILS_SCHEMA = (
    "struct<flow_progress: struct<"
    "status: string, "
    "metrics: struct<num_output_rows: bigint>, "
    "data_quality: struct<dropped_records: bigint, "
    "expectations: array<struct<name: string, dataset: string, "
    "passed_records: bigint, failed_records: bigint>>>>>"
)

DQ_LOG_COLUMNS: list[str] = [
    "run_ts",
    "update_id",
    "dataset",
    "rule",
    "action",
    "condition",
    "passed_records",
    "failed_records",
    "failed_pct",
    "logged_at",
]


def _as_json_struct(df: DataFrame, col_name: str, schema: str) -> Column:
    """Parse `col_name` with `schema` whether it is stored as a JSON string or a struct."""
    col = F.col(col_name)
    if dict(df.dtypes)[col_name] != "string":
        col = F.to_json(col)
    return F.from_json(col, schema)


def _origin_update_id(df: DataFrame) -> Column:
    """`origin.update_id`, whether `origin` is a struct or a JSON string."""
    if dict(df.dtypes)["origin"] == "string":
        return F.get_json_object("origin", "$.update_id")
    return F.col("origin.update_id")


def _short_name(col: Column) -> Column:
    """`retail.silver.sales` / `sales_validated` -> last part, without backticks."""
    return F.element_at(F.split(F.regexp_replace(col, "`", ""), r"\."), -1)


def update_start_times(event_log: DataFrame) -> DataFrame:
    """One row per pipeline update: (update_id, run_ts = first event of that update)."""
    return (
        event_log.withColumn("update_id", _origin_update_id(event_log))
        .filter(F.col("update_id").isNotNull())
        .groupBy("update_id")
        .agg(F.min("timestamp").alias("run_ts"))
    )


def expectation_metrics(event_log: DataFrame) -> DataFrame:
    """Passed/failed records per (update_id, dataset, rule), summed over flow_progress events.

    Metrics can be reported in several flow_progress events per flow (e.g. per micro-batch),
    so they are summed, which is the aggregation the Databricks docs use.
    """
    progress = event_log.filter(F.col("event_type") == "flow_progress").select(
        _origin_update_id(event_log).alias("update_id"),
        _as_json_struct(event_log, "details", _DETAILS_SCHEMA).alias("d"),
    )
    exploded = progress.select(
        "update_id",
        F.explode("d.flow_progress.data_quality.expectations").alias("e"),  # NULL -> no rows
    )
    return exploded.groupBy(
        "update_id",
        _short_name(F.col("e.dataset")).alias("dataset"),
        F.col("e.name").alias("rule"),
    ).agg(
        F.sum("e.passed_records").alias("passed_records"),
        F.sum("e.failed_records").alias("failed_records"),
    )


def output_rows_per_flow(event_log: DataFrame) -> DataFrame:
    """Rows written per (update_id, flow), from flow_progress metrics. For the notebook."""
    origin_flow = (
        F.get_json_object("origin", "$.flow_name")
        if dict(event_log.dtypes)["origin"] == "string"
        else F.col("origin.flow_name")
    )
    return (
        event_log.filter(F.col("event_type") == "flow_progress")
        .select(
            _origin_update_id(event_log).alias("update_id"),
            _short_name(origin_flow).alias("flow"),
            _as_json_struct(event_log, "details", _DETAILS_SCHEMA).alias("d"),
        )
        .groupBy("update_id", "flow")
        .agg(F.sum("d.flow_progress.metrics.num_output_rows").alias("output_rows"))
    )


def rules_frame(spark: SparkSession) -> DataFrame:
    """All rules as rows: (dataset, rule, action, condition). `action` is drop or warn."""
    rows = [
        (dataset, name, action, cond)
        for dataset, (drop, warn) in RULE_SETS.items()
        for action, rules in (("drop", drop), ("warn", warn))
        for name, cond in rules.items()
    ]
    return spark.createDataFrame(
        rows, "dataset STRING, rule STRING, action STRING, condition STRING"
    )


def build_dq_log(event_log: DataFrame, rules: DataFrame) -> DataFrame:
    """Event log + rule definitions -> dq_log rows (one per run, dataset and rule)."""
    metrics = expectation_metrics(event_log)
    total = F.col("passed_records") + F.col("failed_records")
    return (
        metrics.join(update_start_times(event_log), "update_id", "left")
        .join(rules, ["dataset", "rule"], "left")
        .withColumn(
            "failed_pct",
            F.when(total > 0, F.round(F.col("failed_records") * 100.0 / total, 3)),
        )
        .withColumn("logged_at", F.current_timestamp())
        .select(*DQ_LOG_COLUMNS)
        .orderBy(F.desc("run_ts"), "dataset", "action", F.desc("failed_records"))
    )
