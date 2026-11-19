"""Tests for the Gold aggregates (SQL files) and customer cohort retention (DataFrame API).

Each test builds a tiny silver.sales / silver.returns, derives the real Gold inputs from it
(fact_sales + dim_date via the pipeline SQL, dims via the Python builders) and then runs
the aggregate under test, so the whole Gold chain is exercised.
"""

from datetime import date
from decimal import Decimal

import pytest

from factories import SILVER_SCHEMA, run_pipeline_query, sale
from transformations.gold import (
    COHORT_RETENTION_COLUMNS,
    build_customer_cohort_retention,
    build_dim_customer,
    build_dim_product,
)

pytestmark = pytest.mark.spark


@pytest.fixture
def gold(spark):
    """Build every Gold input from silver rows; returns a dict of DataFrames."""

    def _build(sales_rows, returns_rows=()):
        sales = spark.createDataFrame(sales_rows, SILVER_SCHEMA)
        returns = spark.createDataFrame(list(returns_rows), SILVER_SCHEMA)
        silver = {"retail.silver.sales": sales}
        return {
            "retail.silver.sales": sales,
            "retail.silver.returns": returns,
            "retail.gold.fact_sales": run_pipeline_query(spark, "gold_fact_sales.sql", silver),
            "retail.gold.dim_date": run_pipeline_query(spark, "gold_dim_date.sql", silver),
            "retail.gold.dim_product": build_dim_product(sales),
            "retail.gold.dim_customer": build_dim_customer(sales),
        }

    return _build


@pytest.fixture
def run_sql(spark):
    def _run(sql_file, tables):
        # Materialise inputs first so each temp view is a plain table for the query.
        frozen = {
            name: spark.createDataFrame(df.collect(), df.schema) for name, df in tables.items()
        }
        return run_pipeline_query(spark, sql_file, frozen)

    return _run


# ---------------------------------------------------------------- daily_sales
def test_daily_sales_includes_days_without_sales(gold, run_sql):
    g = gold(
        [
            sale("1", "A", qty=2, price="5.00", ts="2010-12-23 10:00", cust=1),
            sale("1", "B", qty=1, price="3.00", ts="2010-12-23 10:00", cust=1),
            sale("2", "A", qty=1, price="5.00", ts="2010-12-23 11:00", cust=None),
            sale("3", "A", qty=1, price="4.00", ts="2010-12-26 09:00", cust=2),
        ]
    )
    rows = {r["date"]: r for r in run_sql("gold_daily_sales.sql", g).collect()}
    assert len(rows) == 4  # 23..26 Dec, every calendar day
    d23 = rows[date(2010, 12, 23)]
    assert d23["revenue"] == Decimal("18.000")
    assert d23["invoices"] == 2
    assert d23["lines"] == 3
    assert d23["units"] == 4
    assert d23["customers"] == 1  # unknown customer (-1) not counted
    assert d23["avg_basket_value"] == Decimal("9.00")  # 18 / 2 invoices
    d24 = rows[date(2010, 12, 24)]
    assert (d24["revenue"], d24["invoices"], d24["lines"], d24["customers"]) == (0, 0, 0, 0)
    assert d24["avg_basket_value"] is None  # no invoices -> no basket, not a divide error


# ---------------------------------------------------------------- monthly_country_sales
def test_monthly_country_sales_grain_and_share(gold, run_sql):
    g = gold(
        [
            sale("1", "A", qty=3, price="10.00", ts="2011-01-05 10:00", country="France"),
            sale("2", "A", qty=1, price="10.00", ts="2011-01-06 10:00", country="Germany", cust=2),
            sale("3", "A", qty=1, price="10.00", ts="2011-02-01 10:00", country="France"),
        ]
    )
    rows = {
        (r["year_month"], r["country"]): r
        for r in run_sql("gold_monthly_country_sales.sql", g).collect()
    }
    assert set(rows) == {("2011-01", "France"), ("2011-01", "Germany"), ("2011-02", "France")}
    assert rows[("2011-01", "France")]["revenue"] == Decimal("30.000")
    assert rows[("2011-01", "France")]["revenue_share_pct"] == Decimal("75.000")
    assert rows[("2011-01", "Germany")]["revenue_share_pct"] == Decimal("25.000")
    assert rows[("2011-02", "France")]["revenue_share_pct"] == Decimal("100.000")
    assert rows[("2011-01", "France")]["units"] == 3


# ---------------------------------------------------------------- product_performance
def test_product_performance_with_returns_and_rank(gold, run_sql):
    g = gold(
        sales_rows=[
            sale("1", "A", qty=10, price="2.00", cust=1),
            sale("2", "A", qty=10, price="2.00", cust=2, country="France"),
            sale("3", "B", qty=1, price="100.00", cust=1),
            sale("4", "C", qty=5, price="1.00", cust=None),
        ],
        returns_rows=[sale("C1", "A", qty=-4, price="2.00", cust=1)],
    )
    rows = {r["stock_code"]: r for r in run_sql("gold_product_performance.sql", g).collect()}
    a, b, c = rows["A"], rows["B"], rows["C"]
    assert (a["revenue"], a["units_sold"], a["invoices"], a["customers"], a["countries"]) == (
        Decimal("40.000"),
        20,
        2,
        2,
        2,
    )
    assert a["units_returned"] == 4  # negated from silver.returns
    assert a["returned_value"] == Decimal("8.000")
    assert a["return_rate"] == pytest.approx(0.2)
    assert b["units_returned"] == 0 and b["return_rate"] == 0.0
    assert c["customers"] == 0  # only the unknown customer bought it
    assert [b["revenue_rank"], a["revenue_rank"], c["revenue_rank"]] == [1, 2, 3]
    assert sum(r["revenue_share_pct"] for r in rows.values()) == pytest.approx(100, abs=0.001)


# ---------------------------------------------------------------- cohort retention
def test_cohort_retention_complete_grid(gold):
    g = gold(
        [
            # cohort 2011-01: customers 1 and 2
            sale("1", "A", ts="2011-01-05 10:00", cust=1),
            sale("2", "A", ts="2011-01-20 10:00", cust=2),
            sale("3", "A", qty=2, price="5.00", ts="2011-03-02 10:00", cust=1),  # month 2
            # cohort 2011-02: customer 3, never returns
            sale("4", "A", ts="2011-02-10 10:00", cust=3),
            # unknown customer: excluded from cohorts
            sale("5", "A", ts="2011-01-07 10:00", cust=None),
        ]
    )
    ret = build_customer_cohort_retention(
        g["retail.gold.fact_sales"], g["retail.gold.dim_customer"]
    )
    assert ret.columns == COHORT_RETENTION_COLUMNS
    rows = {(r["cohort_month"], r["months_since_first_purchase"]): r for r in ret.collect()}

    jan, feb = date(2011, 1, 1), date(2011, 2, 1)
    # complete grid up to the last data month (2011-03): Jan -> 0,1,2 ; Feb -> 0,1
    assert set(rows) == {(jan, 0), (jan, 1), (jan, 2), (feb, 0), (feb, 1)}

    assert rows[(jan, 0)]["cohort_size"] == 2
    assert rows[(jan, 0)]["active_customers"] == 2
    assert rows[(jan, 0)]["retention_rate"] == 1.0
    assert rows[(jan, 1)]["active_customers"] == 0  # filled gap, not missing
    assert rows[(jan, 1)]["retention_rate"] == 0.0
    assert rows[(jan, 2)]["active_customers"] == 1
    assert rows[(jan, 2)]["retention_rate"] == 0.5
    assert rows[(jan, 2)]["activity_month"] == date(2011, 3, 1)
    assert rows[(jan, 2)]["revenue"] == Decimal("10.000")
    assert rows[(feb, 0)]["cohort_size"] == 1
    assert rows[(feb, 1)]["active_customers"] == 0


def test_cohort_counts_customer_once_per_month(gold):
    g = gold(
        [
            sale("1", "A", ts="2011-01-05 10:00", cust=1),
            sale("2", "A", ts="2011-01-25 10:00", cust=1),  # same customer, same month
        ]
    )
    ret = build_customer_cohort_retention(
        g["retail.gold.fact_sales"], g["retail.gold.dim_customer"]
    )
    r = ret.collect()[0]
    assert (r["cohort_size"], r["active_customers"], r["retention_rate"]) == (1, 1, 1.0)
