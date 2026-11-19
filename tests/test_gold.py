"""Tests for the Gold star schema: DataFrame-API dimensions and the pipeline SQL files.

The SQL files are tested too, via `factories.run_pipeline_query` (query part of the
pipeline .sql file, run against local temp views).
"""

from datetime import date, datetime
from decimal import Decimal

import pytest

from factories import SILVER_SCHEMA, run_pipeline_query, sale
from transformations.gold import (
    DIM_CUSTOMER_COLUMNS,
    DIM_PRODUCT_COLUMNS,
    UNKNOWN_CUSTOMER_KEY,
    build_dim_customer,
    build_dim_product,
)
from transformations.silver import SILVER_OUTPUT_COLUMNS

pytestmark = pytest.mark.spark


@pytest.fixture
def make_sales(spark):
    def _make(rows):
        df = spark.createDataFrame(rows, SILVER_SCHEMA)
        assert df.columns == SILVER_OUTPUT_COLUMNS
        return df

    return _make


@pytest.fixture
def pipeline_select(spark):
    def _run(sql_file, sales_df):
        return run_pipeline_query(spark, sql_file, {"retail.silver.sales": sales_df})

    return _run


# ---------------------------------------------------------------- dim_product
def test_dim_product_most_frequent_description_wins(make_sales):
    sales = make_sales(
        [
            sale("1", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER"),
            sale("2", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER"),
            sale("3", "85123A", "CREAM HANGING HEART T-LIGHT HOLDER", ts="2011-12-01 08:00"),
        ]
    )
    rows = build_dim_product(sales).collect()
    assert len(rows) == 1
    r = rows[0]
    assert r["description"] == "WHITE HANGING HEART T-LIGHT HOLDER"
    assert r["n_description_variants"] == 2
    assert r["first_sold_date"] == date(2010, 12, 1)
    assert r["last_sold_date"] == date(2011, 12, 1)


def test_dim_product_tie_goes_to_most_recent_description(make_sales):
    sales = make_sales(
        [
            sale("1", "22423", "REGENCY CAKESTAND", ts="2010-01-01 09:00"),
            sale("2", "22423", "REGENCY CAKESTAND 3 TIER", ts="2011-06-01 09:00"),
        ]
    )
    assert build_dim_product(sales).collect()[0]["description"] == "REGENCY CAKESTAND 3 TIER"


def test_dim_product_ignores_null_descriptions(make_sales):
    sales = make_sales(
        [
            sale("1", "A1", None),
            sale("2", "A1", None),
            sale("3", "A1", "REAL NAME"),
            sale("4", "B2", None),
        ]
    )
    got = {r["stock_code"]: r for r in build_dim_product(sales).collect()}
    assert got["A1"]["description"] == "REAL NAME"
    assert got["B2"]["description"] is None  # product kept, no name known
    assert got["B2"]["n_description_variants"] == 0


def test_dim_product_one_row_per_code_and_median_price(make_sales):
    sales = make_sales(
        [
            sale("1", "X", price="1.00"),
            sale("2", "X", price="2.00"),
            sale("3", "X", price="9.00"),
            sale("4", "Y", price="5.00"),
        ]
    )
    dim = build_dim_product(sales)
    assert dim.columns == DIM_PRODUCT_COLUMNS
    got = {r["stock_code"]: r["median_unit_price"] for r in dim.collect()}
    assert got == {"X": Decimal("2.000"), "Y": Decimal("5.000")}


# ---------------------------------------------------------------- dim_customer
def test_dim_customer_country_cohort_and_dates(make_sales):
    sales = make_sales(
        [
            sale("1", "A", cust=12346, country="France", ts="2010-03-15 10:00"),
            sale("2", "A", cust=12346, country="France", ts="2010-05-01 10:00"),
            sale("3", "A", cust=12346, country="Germany", ts="2011-01-01 10:00"),
        ]
    )
    dim = build_dim_customer(sales)
    assert dim.columns == DIM_CUSTOMER_COLUMNS
    r = dim.filter("customer_key = 12346").collect()[0]
    assert r["country"] == "France"  # most frequent
    assert r["n_countries"] == 2
    assert r["first_purchase_ts"] == datetime(2010, 3, 15, 10, 0)
    assert r["last_purchase_ts"] == datetime(2011, 1, 1, 10, 0)
    assert r["cohort_month"] == date(2010, 3, 1)
    assert r["is_unknown"] is False


def test_dim_customer_country_tie_goes_to_most_recent(make_sales):
    sales = make_sales(
        [
            sale("1", "A", cust=7, country="EIRE", ts="2010-01-01 10:00"),
            sale("2", "A", cust=7, country="United Kingdom", ts="2010-02-01 10:00"),
        ]
    )
    assert build_dim_customer(sales).filter("customer_key = 7").collect()[0]["country"] == (
        "United Kingdom"
    )


def test_dim_customer_unknown_row_and_unique_keys(make_sales):
    sales = make_sales(
        [
            sale("1", "A", cust=None),
            sale("2", "A", cust=None),
            sale("3", "A", cust=1),
            sale("4", "A", cust=2),
        ]
    )
    rows = build_dim_customer(sales).collect()
    keys = [r["customer_key"] for r in rows]
    assert sorted(keys) == [UNKNOWN_CUSTOMER_KEY, 1, 2]  # one unknown row, no NULL customer
    unknown = next(r for r in rows if r["customer_key"] == UNKNOWN_CUSTOMER_KEY)
    assert unknown["is_unknown"] is True and unknown["customer_id"] is None


# ---------------------------------------------------------------- fact_sales (SQL)
def test_fact_sales_grain_and_keys(make_sales, pipeline_select):
    sales = make_sales(
        [
            sale("536365", "85123A", qty=6, price="2.55", ts="2010-12-01 08:26", cust=17850),
            sale("536365", "71053", qty=6, price="3.39", ts="2010-12-01 08:26", cust=17850),
            sale("536366", "22633", qty=6, price="1.85", ts="2010-12-01 08:28", cust=None),
        ]
    )
    fact = pipeline_select("gold_fact_sales.sql", sales)
    rows = fact.collect()
    assert len(rows) == 3  # one row per Silver line
    assert len({r["line_id"] for r in rows}) == 3  # surrogate key unique
    assert {r["date_key"] for r in rows} == {20101201}
    keys = {r["stock_code"]: r["customer_key"] for r in rows}
    assert keys == {"85123A": 17850, "71053": 17850, "22633": -1}
    assert sum(r["line_revenue"] for r in rows) == Decimal("46.740")


def test_fact_sales_line_id_is_deterministic(make_sales, pipeline_select):
    sales = make_sales([sale("1", "A")])
    first = pipeline_select("gold_fact_sales.sql", sales).collect()[0]["line_id"]
    second = pipeline_select("gold_fact_sales.sql", sales).collect()[0]["line_id"]
    assert first == second


def test_fact_sales_keys_join_to_dimensions(make_sales, pipeline_select):
    sales = make_sales([sale("1", "A", cust=5), sale("2", "B", cust=None)])
    fact = pipeline_select("gold_fact_sales.sql", sales)
    dim_c = build_dim_customer(sales)
    dim_p = build_dim_product(sales)
    assert fact.join(dim_c, "customer_key", "left_anti").count() == 0
    assert fact.join(dim_p, "stock_code", "left_anti").count() == 0


# ---------------------------------------------------------------- dim_date (SQL)
def test_dim_date_covers_every_day_including_no_sales_days(make_sales, pipeline_select):
    sales = make_sales(
        [sale("1", "A", ts="2010-12-23 10:00"), sale("2", "A", ts="2011-01-04 10:00")]
    )
    rows = pipeline_select("gold_dim_date.sql", sales).orderBy("date").collect()
    assert len(rows) == 13  # 23 Dec .. 4 Jan inclusive, incl. Christmas with no sales
    assert rows[0]["date_key"] == 20101223
    assert rows[-1]["date_key"] == 20110104
    assert len({r["date_key"] for r in rows}) == 13


def test_dim_date_attributes(make_sales, pipeline_select):
    sales = make_sales([sale("1", "A", ts="2010-12-04 10:00")])  # a Saturday
    r = pipeline_select("gold_dim_date.sql", sales).collect()[0]
    assert r["date"] == date(2010, 12, 4)
    assert (r["year"], r["quarter"], r["month"], r["day_of_month"]) == (2010, 4, 12, 4)
    assert r["month_name"] == "December"
    assert r["year_month"] == "2010-12"
    assert r["day_name"] == "Saturday"
    assert r["day_of_week"] == 7
    assert r["is_weekend"] is True
