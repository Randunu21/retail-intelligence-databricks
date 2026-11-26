"""Writing feature snapshots to Unity Catalog as a feature table.

In Unity Catalog, **any Delta table with a PRIMARY KEY constraint is a feature table**: it
shows up in the Features UI, and FeatureEngineeringClient can look features up from it
(training sets, lineage from feature table to model). So the table is created with plain SQL
DDL (no client needed) and written with the normal Delta writer. The client
(`databricks-feature-engineering`, a %pip install on serverless) is optional: the notebook
uses it only to confirm that UC sees the table as a feature table.

Key design: PRIMARY KEY (<entity key>, as_of_date TIMESERIES).
- `as_of_date` as a TIMESERIES key lets several snapshots live in one table, and gives
  point-in-time lookups in Phase 5 (features as of each customer's cutoff, no leakage).
- Each run overwrites only its own snapshot (`replaceWhere as_of_date = ...`), so re-running
  is idempotent and older snapshots are kept.

Phase 3 wrote the customer table (the defaults below). Phase 4 reuses the same functions for
products by passing the product schema, keys and comment (`features.product_features`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from features.customer_features import FEATURE_COLUMNS, FEATURE_SCHEMA, KEY_COLUMNS

COMPUTED_AT_COLUMN = ("computed_at", "TIMESTAMP", "When this snapshot was written")
TABLE_COMMENT = (
    "Customer features (RFM, basket, returns, tenure, purchase interval, country, seasonality) "
    "per customer and snapshot date. Feature table: PK (customer_id, as_of_date TIMESERIES). "
    "Built by src/features/customer_features.py (Phase 3)."
)


def _sql_string(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def constraint_name(table_name: str) -> str:
    """customer_features -> customer_features_pk (constraint names are unique per schema)."""
    return f"{table_name.split('.')[-1]}_pk"


def feature_table_ddl(
    table_name: str,
    schema: Sequence[tuple[str, str, str]] = FEATURE_SCHEMA,
    key_columns: tuple[str, str] = KEY_COLUMNS,
    comment: str | None = None,
) -> str:
    """CREATE TABLE IF NOT EXISTS with NOT NULL keys, column comments and the primary key.

    `schema` = [(column, Spark SQL type, comment)], `key_columns` = (entity key, time key).
    Defaults = the Phase 3 customer feature table (`comment=None` -> TABLE_COMMENT, read at
    call time).
    """
    comment = TABLE_COMMENT if comment is None else comment
    columns = [*schema, COMPUTED_AT_COLUMN]
    lines = []
    for name, dtype, col_comment in columns:
        not_null = " NOT NULL" if name in key_columns else ""
        lines.append(f"  {name} {dtype}{not_null} COMMENT {_sql_string(col_comment)}")
    entity_key, time_key = key_columns
    lines.append(
        f"  CONSTRAINT {constraint_name(table_name)} "
        f"PRIMARY KEY ({entity_key}, {time_key} TIMESERIES)"
    )
    body = ",\n".join(lines)
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n{body}\n)\n"
        f"USING DELTA\nCOMMENT {_sql_string(comment)}"
    )


def snapshot_predicate(as_of: date | str) -> str:
    return f"as_of_date = DATE'{as_of}'"


def write_snapshot(
    features: DataFrame,
    table_name: str,
    as_of: date | str,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> None:
    """Replace one snapshot (rows with this as_of_date) and leave the others untouched."""
    out = features.select(*columns).withColumn("computed_at", F.current_timestamp())
    (
        out.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", snapshot_predicate(as_of))
        .saveAsTable(table_name)
    )


def feature_engineering_client_status() -> dict:
    """Is the Feature Engineering client importable here? Never raises (used by the probe)."""
    try:
        from importlib.metadata import version

        from databricks.feature_engineering import FeatureEngineeringClient  # noqa: F401

        return {"available": True, "version": version("databricks-feature-engineering")}
    except Exception as exc:  # ImportError, or a dependency clash on serverless
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def describe_feature_table(table_name: str) -> dict:
    """What the Feature Engineering client reports for the table (keys, description).

    Attribute names differ slightly between client versions, so they are read defensively.
    """
    from databricks.feature_engineering import FeatureEngineeringClient

    table = FeatureEngineeringClient().get_table(name=table_name)
    info = {"name": getattr(table, "name", table_name)}
    for attr in ("primary_keys", "timeseries_keys", "timestamp_keys", "description"):
        if hasattr(table, attr):
            info[attr] = getattr(table, attr)
    return info
