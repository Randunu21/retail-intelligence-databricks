"""Tests for features.product_features and the product use of features.feature_store (Phase 4).

Hand-made Silver rows (tests/factories.sale), small enough to work every value out on paper.
as_of = 2011-01-01, so the last data day is Fri 2010-12-31 (week starting Mon 2010-12-27).

| product | sales before as_of | returns |
|---|---|---|
| A1 | 2010-10-04 inv1 10 x 1.00 cust 1 UK; 2010-10-05 inv2 2 x 2.00 cust 2 France; 2010-12-06 inv3 4 x 1.00 no customer UK | 2 units x 1.00 |
| B1 | 2010-10-04 inv1 1 x 5.00 cust 1 UK | 3 units x 5.00 (more than sold -> capped) |
| D1 | 2010-12-20 inv4 3 x 1.00 cust 3 UK (one month only -> CV NULL) | - |
| C1 | - | 1 unit (returns only -> excluded) |
| A1 | 2011-01-05 inv5 100 x 1.00 (after as_of -> ignored) | - |
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from factories import SILVER_SCHEMA, sale
from features.feature_store import constraint_name, feature_table_ddl
from features.product_features import (
    PRODUCT_FEATURE_COLUMNS,
    PRODUCT_FEATURE_SCHEMA,
    PRODUCT_KEY_COLUMNS,
    PRODUCT_TABLE_COMMENT,
    build_product_features,
    long_tail_table,
    pareto_counts,
    reconcile_product_features,
)

AS_OF = date(2011, 1, 1)
DATA_START = date(2009, 12, 1)


@pytest.fixture
def data(spark):
    sales = spark.createDataFrame(
        [
            sale("inv1", "A1", qty=10, price="1.00", ts="2010-10-04 09:00", cust=1),
            sale("inv1", "B1", qty=1, price="5.00", ts="2010-10-04 09:00", cust=1),
            sale(
                "inv2", "A1", qty=2, price="2.00", ts="2010-10-05 09:00", cust=2, country="France"
            ),
            sale("inv3", "A1", qty=4, price="1.00", ts="2010-12-06 09:00", cust=None),
            sale("inv4", "D1", qty=3, price="1.00", ts="2010-12-20 09:00", cust=3),
            sale("inv5", "A1", qty=100, price="1.00", ts="2011-01-05 09:00", cust=1),
        ],
        SILVER_SCHEMA,
    )
    returns = spark.createDataFrame(
        [
            sale("C9001", "A1", qty=-2, price="1.00", ts="2010-12-10 09:00", cust=1),
            sale("C9002", "B1", qty=-3, price="5.00", ts="2010-12-11 09:00", cust=1),
            sale("C9003", "C1", qty=-1, price="1.00", ts="2010-12-12 09:00", cust=1),
        ],
        SILVER_SCHEMA,
    )
    return sales, returns


@pytest.fixture
def features(data):
    sales, returns = data
    rows = build_product_features(sales, returns, AS_OF, data_start=DATA_START).collect()
    return {r["stock_code"]: r for r in rows}


def test_schema_and_one_row_per_product(data):
    sales, returns = data
    df = build_product_features(sales, returns, AS_OF, data_start=DATA_START)
    assert df.columns == PRODUCT_FEATURE_COLUMNS
    assert sorted(r["stock_code"] for r in df.collect()) == ["A1", "B1", "D1"]  # no C1
    types = dict(df.dtypes)
    assert types["units_sold"] == "bigint" and types["n_invoices"] == "int"
    assert types["as_of_date"] == "date"


def test_volume_price_and_reach(features):
    a = features["A1"]
    assert a["n_invoices"] == 3 and a["n_lines"] == 3
    assert a["units_sold"] == 16  # the 100 units after as_of are ignored
    assert a["revenue"] == pytest.approx(18.0)
    assert a["median_unit_price"] == pytest.approx(1.0)
    assert a["avg_unit_price"] == pytest.approx(18 / 16)
    assert a["avg_quantity_per_line"] == pytest.approx(16 / 3)
    assert a["unique_customers"] == 2  # the customer-less line still counts for revenue
    assert a["n_countries"] == 2
    assert a["non_uk_revenue_share"] == pytest.approx(4 / 18)


def test_revenue_share_uses_all_products(features):
    total = 18 + 5 + 3
    assert features["A1"]["revenue_share"] == pytest.approx(18 / total)
    assert features["B1"]["revenue_share"] == pytest.approx(5 / total)
    assert sum(r["revenue_share"] for r in features.values()) == pytest.approx(1.0)


def test_velocity_and_calendar(features):
    a = features["A1"]
    # weeks: Mon 2010-10-04 .. Mon 2010-12-27 = 12 weeks apart -> 13 calendar weeks
    assert a["weeks_on_sale"] == 13
    assert a["months_on_sale"] == 3  # Oct, Nov, Dec
    assert a["units_per_week"] == pytest.approx(16 / 13)
    assert a["active_week_share"] == pytest.approx(2 / 13)  # inv1+inv2 same week, inv3
    assert a["days_on_sale"] == 89
    assert a["days_since_last_sale"] == 26
    assert a["first_sale_date"] == date(2010, 10, 4)
    assert a["last_sale_date"] == date(2010, 12, 6)
    d = features["D1"]
    assert d["weeks_on_sale"] == 2 and d["months_on_sale"] == 1


def test_seasonality(features):
    a = features["A1"]
    assert a["seasonal_revenue_share"] == pytest.approx(1.0)
    # monthly units Oct 12, Nov 0, Dec 4 (zero month included)
    mean = 16 / 3
    var = (12**2 + 0**2 + 4**2) / 3 - mean**2
    assert a["monthly_units_cv"] == pytest.approx(math.sqrt(var) / mean)
    b = features["B1"]  # Oct 1, Nov 0, Dec 0
    assert b["monthly_units_cv"] == pytest.approx(math.sqrt(1 / 3 - 1 / 9) / (1 / 3))
    assert features["D1"]["monthly_units_cv"] is None  # one month < MIN_MONTHS_FOR_CV


def test_returns(features):
    a = features["A1"]
    assert a["units_returned"] == 2 and a["returned_value"] == pytest.approx(2.0)
    assert a["return_rate"] == pytest.approx(2 / 16)
    b = features["B1"]
    assert b["units_returned"] == 3  # raw amount kept
    assert b["return_rate"] == pytest.approx(1.0)  # 3 / 1 capped
    d = features["D1"]
    assert d["units_returned"] == 0 and d["return_rate"] == pytest.approx(0.0)


def test_history_censoring(data):
    sales, returns = data
    rows = {
        r["stock_code"]: r
        for r in build_product_features(
            sales, returns, AS_OF, data_start=date(2010, 10, 1)
        ).collect()
    }
    assert rows["A1"]["is_history_censored"] is True  # first sale within a month of start
    assert rows["D1"]["is_history_censored"] is False


def test_default_data_start_is_first_sale(data):
    sales, returns = data
    rows = {r["stock_code"]: r for r in build_product_features(sales, returns, AS_OF).collect()}
    assert rows["A1"]["is_history_censored"] is True  # data starts 2010-10-04 here


def test_reconcile(data):
    sales, returns = data
    df = build_product_features(sales, returns, AS_OF, data_start=DATA_START)
    checks = reconcile_product_features(df, sales, returns, AS_OF)
    assert checks["products"] == checks["expected_products"] == 3
    assert checks["duplicate_keys"] == 0
    assert checks["revenue_diff"] == pytest.approx(0.0)
    assert checks["units_diff"] == 0
    assert checks["revenue_share_sum"] == pytest.approx(1.0)
    assert checks["short_history_products"] == checks["null_cv"] == 1
    assert checks["returns_only_products"] == 1
    assert checks["return_rate_capped"] == 1
    lo, hi = checks["return_rate_range"]
    assert 0.0 <= lo <= hi <= 1.0


def test_product_feature_table_ddl():
    ddl = feature_table_ddl(
        "retail.ml.product_features",
        schema=PRODUCT_FEATURE_SCHEMA,
        key_columns=PRODUCT_KEY_COLUMNS,
        comment=PRODUCT_TABLE_COMMENT,
    )
    assert constraint_name("retail.ml.product_features") == "product_features_pk"
    assert "CONSTRAINT product_features_pk PRIMARY KEY (stock_code, as_of_date TIMESERIES)" in ddl
    assert "stock_code STRING NOT NULL" in ddl
    assert "as_of_date DATE NOT NULL" in ddl
    assert "revenue DOUBLE COMMENT" in ddl  # non-key columns stay nullable
    assert "computed_at TIMESTAMP" in ddl
    assert "Phase 4" in ddl


def test_customer_ddl_defaults_unchanged():
    ddl = feature_table_ddl("retail.ml.customer_features")
    assert "PRIMARY KEY (customer_id, as_of_date TIMESERIES)" in ddl
    assert "stock_code" not in ddl


def test_long_tail_table(spark):
    df = spark.createDataFrame(
        [("A", 1, 10.0), ("B", 5, 30.0), ("C", 20, 60.0)],
        "stock_code STRING, n_invoices INT, revenue DOUBLE",
    )
    rows = {r["min_invoices"]: r for r in long_tail_table(df, (1, 5, 21))}
    assert rows[1]["products"] == 3 and rows[1]["revenue_share"] == pytest.approx(1.0)
    assert rows[5]["products"] == 2
    assert rows[5]["product_share"] == pytest.approx(2 / 3)
    assert rows[5]["revenue_share"] == pytest.approx(0.9)
    assert rows[21]["products"] == 0 and rows[21]["revenue_share"] == pytest.approx(0.0)


def test_pareto_counts(spark):
    df = spark.createDataFrame(
        [("A", 50.0), ("B", 30.0), ("C", 15.0), ("D", 5.0)], "stock_code STRING, revenue DOUBLE"
    )
    # cumulative: A 0.50, B 0.80, C 0.95, D 1.00
    assert pareto_counts(df, (0.5, 0.8, 0.81, 0.95, 1.0)) == {
        0.5: 1,
        0.8: 2,
        0.81: 3,
        0.95: 3,
        1.0: 4,
    }
