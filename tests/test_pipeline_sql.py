"""Tests for optimisation.pipeline_sql: cutting queries out of pipeline SQL files."""

from pathlib import Path

import pytest

from optimisation.pipeline_sql import extract_query, load_query, retarget

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipelines" / "medallion" / "transformations"


def test_extract_query_returns_text_after_as_line():
    text = "-- comment\nCREATE OR REFRESH MATERIALIZED VIEW x\nCOMMENT 'a AS b'\n  as  \nSELECT 1 AS one"
    assert extract_query(text) == "SELECT 1 AS one"


def test_extract_query_without_as_line_fails():
    with pytest.raises(ValueError):
        extract_query("SELECT 1 AS one")


def test_retarget_replaces_whole_names_only():
    q = "FROM retail.gold.fact_sales f JOIN retail.gold.fact_sales_v2 v JOIN x.retail.gold.fact_sales"
    out = retarget(q, {"retail.gold.fact_sales": "retail.bench.p__gold_fact_sales"})
    assert out == (
        "FROM retail.bench.p__gold_fact_sales f JOIN retail.gold.fact_sales_v2 v "
        "JOIN x.retail.gold.fact_sales"
    )


def test_retarget_longer_names_first():
    q = "SELECT * FROM a.b.c JOIN a.b.c_d"
    out = retarget(q, {"a.b.c": "X", "a.b.c_d": "Y"})
    assert out == "SELECT * FROM X JOIN Y"


@pytest.mark.parametrize(
    "file_name",
    [
        "gold_fact_sales.sql",
        "gold_dim_date.sql",
        "gold_daily_sales.sql",
        "gold_monthly_country_sales.sql",
        "gold_product_performance.sql",
    ],
)
def test_real_pipeline_files_are_fully_retargeted(file_name):
    mapping = {
        "retail.silver.sales": "v_sales",
        "retail.silver.returns": "v_returns",
        "retail.gold.fact_sales": "v_fact",
        "retail.gold.dim_date": "v_date",
        "retail.gold.dim_product": "v_product",
    }
    query = load_query(PIPELINE_DIR, file_name, mapping)
    assert "retail." not in query
    assert "CREATE" not in query.upper().split("SELECT", 1)[0]
