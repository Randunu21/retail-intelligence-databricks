"""Shared test helpers for Gold tests (imported by test files; not a test module itself).

- `sale(...)`: one row shaped like retail.silver.sales.
- `run_pipeline_query(spark, sql_file, tables)`: runs the query part of a pipeline .sql file
  (everything after the line that is just "AS") with each real table name replaced by a
  local temp view, so the SQL used in the pipeline is tested unchanged.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipelines" / "medallion" / "transformations"

SILVER_SCHEMA = (
    "invoice STRING, stock_code STRING, description STRING, quantity INT, "
    "price DECIMAL(10,3), line_revenue DECIMAL(18,3), invoice_ts TIMESTAMP_NTZ, "
    "invoice_date DATE, customer_id INT, is_missing_customer BOOLEAN, country STRING, "
    "_source_file STRING, _ingested_at TIMESTAMP"
)


def sale(
    invoice,
    code,
    desc="PRODUCT",
    qty=1,
    price="1.00",
    ts="2010-12-01 08:00",
    cust=100,
    country="United Kingdom",
):
    """One silver.sales / silver.returns row. line_revenue = qty * price."""
    t = datetime.strptime(ts, "%Y-%m-%d %H:%M")
    p = Decimal(price)
    return (
        invoice,
        code,
        desc,
        qty,
        p,
        (qty * p).quantize(Decimal("0.001")),
        t,
        t.date(),
        cust,
        cust is None,
        country,
        "f.csv",
        datetime(2026, 11, 19),
    )


def run_pipeline_query(spark, sql_file, tables):
    """Run a pipeline SQL file's query against local DataFrames.

    `tables` maps the real table name used in the file (e.g. "retail.gold.fact_sales") to a
    DataFrame. Longer names are replaced first so names never clash.
    """
    text = (PIPELINE_DIR / sql_file).read_text()
    query = re.split(r"(?im)^\s*AS\s*$", text, maxsplit=1)[1]
    for i, (name, df) in enumerate(sorted(tables.items(), key=lambda kv: -len(kv[0]))):
        view = f"test_view_{i}"
        df.createOrReplaceTempView(view)
        query = query.replace(name, view)
    return spark.sql(query)
