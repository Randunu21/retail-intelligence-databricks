"""Outputs of customer segmentation (Phase 3.5b): the Gold table and the profile document.

`retail.gold.customer_segments`: one row per (customer_id, as_of_date), the segment, action
group and recommended action, the key features (so analysts and the Phase 8 dashboard don't
need a join to read them), and lineage (MLflow run id of the model that scored it, scored_at).
Each run replaces only its own snapshot (replaceWhere), like the feature table.

`docs/segment_profiles.md` is GENERATED from the profile numbers of the run (never typed by
hand), so the document can't drift from the data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from segmentation.naming import ACTION_PLAYBOOK, SegmentDefinition

SEGMENTS_SCHEMA: list[tuple[str, str, str]] = [
    ("customer_id", "INT", "Customer id (primary key, with as_of_date)"),
    ("as_of_date", "DATE", "Feature snapshot the segment was computed from"),
    ("segment_id", "INT", "Cluster id, ordered by value (0 = highest monetary centre)"),
    ("segment_key", "STRING", "Stable machine key: loyal / seasonal_new / hibernating"),
    ("segment_name", "STRING", "Business name of the segment"),
    ("action_group", "STRING", "Rule-based group inside the segment that decides the action"),
    ("recommended_action", "STRING", "What the business should do for this group"),
    ("recency_days", "INT", "Days since last purchase"),
    ("frequency", "INT", "Distinct invoices"),
    ("monetary", "DOUBLE", "Gross sales value in GBP"),
    ("avg_basket_value", "DOUBLE", "monetary / frequency"),
    ("tenure_days", "INT", "Days since first purchase"),
    ("seasonal_revenue_share", "DOUBLE", "Share of spend in Oct-Dec"),
    ("is_new_customer", "BOOLEAN", "First purchase in the last 90 days"),
    ("is_overdue", "BOOLEAN", "Repeat buyer, time since last order > 2x own average gap"),
    ("is_top_value", "BOOLEAN", "Monetary in the top 10% of customers"),
    ("model_run_id", "STRING", "MLflow run id of the segmentation model"),
    ("scored_at", "TIMESTAMP", "When the segments were written"),
]
SEGMENTS_COLUMNS = [c for c, _, _ in SEGMENTS_SCHEMA]
TABLE_COMMENT = (
    "Customer segments (KMeans k=3 on RFM/tenure/breadth/seasonality + rule-based action "
    "groups) per customer and snapshot. Built by notebooks/03_customer_segmentation (Phase 3)."
)


def _sql_string(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def segments_table_ddl(table_name: str) -> str:
    lines = []
    for name, dtype, comment in SEGMENTS_SCHEMA:
        nn = " NOT NULL" if name in ("customer_id", "as_of_date") else ""
        lines.append(f"  {name} {dtype}{nn} COMMENT {_sql_string(comment)}")
    short = table_name.split(".")[-1]
    lines.append(f"  CONSTRAINT {short}_pk PRIMARY KEY (customer_id, as_of_date)")
    body = ",\n".join(lines)
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n{body}\n)\nUSING DELTA\n"
        f"COMMENT {_sql_string(TABLE_COMMENT)}"
    )


def build_segments_output(grouped: DataFrame, model_run_id: str) -> DataFrame:
    """grouped = feature rows + segment_id + add_action_groups columns -> table columns."""
    types = {c: t for c, t, _ in SEGMENTS_SCHEMA}
    out = grouped.withColumn("model_run_id", F.lit(model_run_id)).withColumn(
        "scored_at", F.current_timestamp()
    )
    return out.select([F.col(c).cast(types[c]).alias(c) for c in SEGMENTS_COLUMNS])


def write_segments(df: DataFrame, table_name: str, as_of: date | str) -> None:
    (
        df.select(*SEGMENTS_COLUMNS)
        .write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"as_of_date = DATE'{as_of}'")
        .saveAsTable(table_name)
    )


# ------------------------------------------------------------------------------- document


def _fmt(v, kind="num"):
    if v is None:
        return "–"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "gbp":
        return f"£{v:,.0f}"
    if kind == "int":
        return f"{v:,.0f}"
    return f"{v:,.1f}"


PROFILE_ROWS: list[tuple[str, str, str]] = [
    ("Customers", "customers", "int"),
    ("Share of customers", "customer_share", "pct"),
    ("Share of revenue", "revenue_share", "pct"),
    ("Revenue per customer", "revenue_per_customer", "gbp"),
    ("Median days since last order", "median_recency_days", "int"),
    ("Median orders", "median_frequency", "int"),
    ("Median spend", "median_monetary", "gbp"),
    ("Median basket value", "median_avg_basket_value", "gbp"),
    ("Median distinct products", "median_distinct_products", "int"),
    ("Median tenure (days)", "median_tenure_days", "int"),
    ("Median days between orders (repeat buyers)", "median_mean_days_between_purchases", "num"),
    ("Median Q4 share of spend", "median_seasonal_revenue_share", "pct"),
    ("Bought only once", "pct_one_invoice", "pct"),
    ("Bought in the last 90 days", "pct_bought_last_90d", "pct"),
    ("First bought in the last 90 days", "pct_first_bought_last_90d", "pct"),
    ("Returned something", "pct_with_returns", "pct"),
    ("UK customers", "pct_uk", "pct"),
]


def render_segment_profiles_md(
    *,
    as_of: date | str,
    n_customers: int,
    names: Mapping[int, SegmentDefinition],
    segment_profiles: Sequence[Mapping],
    group_profiles: Sequence[Mapping],
    model_facts: Mapping[str, object],
    images: Mapping[str, str],
) -> str:
    """Markdown for docs/segment_profiles.md. Profiles are rows from segment_profile()
    (keyed by segment_id, or by action_group for group_profiles)."""
    segs = sorted(segment_profiles, key=lambda r: r["segment_id"])
    header = "| | " + " | ".join(names[r["segment_id"]].name for r in segs) + " |"
    lines = [
        "# Customer segment profiles",
        "",
        f"Snapshot **{as_of}** · {n_customers:,} customers with a customer id · generated by "
        "`notebooks/03_customer_segmentation.py` (section 7) from the real run; do not edit by hand.",
        "",
        "## How the segments were built",
        "",
    ]
    for k, v in model_facts.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## The segments", "", header, "|---|" + "---|" * len(segs)]
    for label, col, kind in PROFILE_ROWS:
        lines.append(f"| {label} | " + " | ".join(_fmt(r.get(col), kind) for r in segs) + " |")
    lines.append("")
    for r in segs:
        d = names[r["segment_id"]]
        lines.append(f"- **{d.name}** (segment {r['segment_id']}): {d.description}")
    for title, path in images.items():
        lines += ["", f"![{title}]({path})"]

    lines += [
        "",
        "## Action groups and recommended actions",
        "",
        "Rule-based groups inside each segment (definitions in `src/segmentation/naming.py`): "
        "*overdue* = repeat buyer whose time since the last order is more than twice their own "
        "average gap; *new* = first order in the last 90 days; *top value* = top 10% by spend.",
        "",
        "| Action group | Customers | Share of customers | Share of revenue | Median days since last order | Median spend | Recommended action | KPI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = list(ACTION_PLAYBOOK)
    for g in sorted(group_profiles, key=lambda r: order.index(r["action_group"])):
        play = ACTION_PLAYBOOK[g["action_group"]]
        lines.append(
            f"| {g['action_group']} | {_fmt(g['customers'], 'int')} | "
            f"{_fmt(g['customer_share'], 'pct')} | {_fmt(g['revenue_share'], 'pct')} | "
            f"{_fmt(g['median_recency_days'], 'int')} | {_fmt(g['median_monetary'], 'gbp')} | "
            f"{play['action']} | {play['kpi']} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Seasonal share is confounded with newness: customers acquired in Q4 2011 have only "
        "Q4 history. The *New Customers* / *Seasonal Gifters* split separates them.",
        "- Tenure is censored for customers first seen in Dec 2009 (the first month of data).",
        "- k = 3 was chosen for stability; finer clusterings (k = 4) changed with the random seed, "
        "so finer targeting uses the transparent rules above instead.",
        "",
    ]
    return "\n".join(lines)
