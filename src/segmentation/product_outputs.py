"""Gold output of product segmentation (Phase 4.3b): `retail.gold.product_segments`.

One row per (stock_code, as_of_date) for **every** product of the snapshot: clustered
products carry their segment_id; Long Tail / New Products are rule-based (segment_id NULL,
`cluster_eligibility` says why). Key features are copied in so analysts and the Phase 8
dashboard can read a product's segment without a join; lineage = `model_run_id` (MLflow run
of the final model) and `scored_at`. Each run replaces only its own snapshot (replaceWhere).
"""

from __future__ import annotations

from datetime import date

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

PRODUCT_SEGMENTS_SCHEMA: list[tuple[str, str, str]] = [
    ("stock_code", "STRING", "Product code (primary key, with as_of_date)"),
    ("as_of_date", "DATE", "Feature snapshot the segment was computed from"),
    ("description", "STRING", "Product description (gold.dim_product)"),
    ("segment_id", "INT", "Cluster id ordered by velocity (0 = fastest); NULL for rule-based"),
    ("segment_key", "STRING", "Stable key: core / fading / slow / long_tail / new"),
    ("segment_name", "STRING", "Business name of the segment"),
    ("cluster_eligibility", "STRING", "clustered / low_volume / new_product"),
    ("action_group", "STRING", "Rule-based group inside the segment that decides the action"),
    ("recommended_action", "STRING", "What the business should do for this group"),
    ("revenue", "DOUBLE", "Sales revenue in GBP"),
    ("revenue_share", "DOUBLE", "Share of total revenue"),
    ("n_invoices", "INT", "Distinct invoices"),
    ("units_per_week", "DOUBLE", "Units sold per week since first sale"),
    ("active_week_share", "DOUBLE", "Share of weeks with a sale"),
    ("median_unit_price", "DOUBLE", "Median line price in GBP"),
    ("unique_customers", "INT", "Distinct known customers"),
    ("non_uk_revenue_share", "DOUBLE", "Share of revenue outside the UK"),
    ("seasonal_revenue_share", "DOUBLE", "Share of revenue in Oct-Dec"),
    ("days_since_last_sale", "INT", "Days since the last sale"),
    ("return_rate", "DOUBLE", "Units returned / sold, capped at 1"),
    ("is_seasonal", "BOOLEAN", "Q4 share of revenue > 0.5"),
    ("is_discontinued", "BOOLEAN", "No sale for more than 180 days"),
    ("is_premium", "BOOLEAN", "Median price in the top 10% of clustered products"),
    ("is_high_return", "BOOLEAN", "Return rate >= 10%"),
    ("model_run_id", "STRING", "MLflow run id of the product segmentation model"),
    ("scored_at", "TIMESTAMP", "When the segments were written"),
]
PRODUCT_SEGMENTS_COLUMNS = [c for c, _, _ in PRODUCT_SEGMENTS_SCHEMA]
PRODUCT_SEGMENTS_COMMENT = (
    "Product segments (bisecting k-means k=3 on velocity/price/seasonality/lifecycle/reach + "
    "rule-based Long Tail / New Products and action groups) per product and snapshot. Built "
    "by notebooks/04_product_segmentation_mba (Phase 4)."
)


def _sql_string(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def product_segments_ddl(table_name: str) -> str:
    lines = []
    for name, dtype, comment in PRODUCT_SEGMENTS_SCHEMA:
        nn = " NOT NULL" if name in ("stock_code", "as_of_date") else ""
        lines.append(f"  {name} {dtype}{nn} COMMENT {_sql_string(comment)}")
    short = table_name.split(".")[-1]
    lines.append(f"  CONSTRAINT {short}_pk PRIMARY KEY (stock_code, as_of_date)")
    body = ",\n".join(lines)
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n{body}\n)\nUSING DELTA\n"
        f"COMMENT {_sql_string(PRODUCT_SEGMENTS_COMMENT)}"
    )


def build_product_segments_output(
    grouped: DataFrame, descriptions: DataFrame, model_run_id: str
) -> DataFrame:
    """grouped = product features + eligibility + segment_id + add_product_groups columns."""
    types = {c: t for c, t, _ in PRODUCT_SEGMENTS_SCHEMA}
    out = (
        grouped.join(descriptions.select("stock_code", "description"), "stock_code", "left")
        .withColumn("model_run_id", F.lit(model_run_id))
        .withColumn("scored_at", F.current_timestamp())
    )
    return out.select([F.col(c).cast(types[c]).alias(c) for c in PRODUCT_SEGMENTS_COLUMNS])


def write_product_segments(df: DataFrame, table_name: str, as_of: date | str) -> None:
    (
        df.select(*PRODUCT_SEGMENTS_COLUMNS)
        .write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"as_of_date = DATE'{as_of}'")
        .saveAsTable(table_name)
    )
