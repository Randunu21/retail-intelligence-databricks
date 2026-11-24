"""Tests for features.customer_features and features.feature_store (Phase 3).

Small hand-made Silver rows (tests/factories.sale), so every expected value can be worked
out on paper. Default as_of in these tests: 2011-01-01.
"""

from __future__ import annotations

from datetime import date

import pytest

from factories import SILVER_SCHEMA, sale
from features.customer_features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    build_customer_features,
    default_as_of,
    describe_features,
    reconcile_features,
)
from features.feature_store import (
    constraint_name,
    feature_table_ddl,
    snapshot_predicate,
)

AS_OF = date(2011, 1, 1)
DATA_START = date(2009, 12, 1)


@pytest.fixture
def frame(spark):
    def _make(rows):
        return spark.createDataFrame(rows, SILVER_SCHEMA)

    return _make


@pytest.fixture
def no_returns(frame):
    return frame([])


def _features(frame, sales_rows, return_rows=(), as_of=AS_OF, **kwargs):
    df = build_customer_features(
        frame(sales_rows), frame(list(return_rows)), as_of, data_start=DATA_START, **kwargs
    )
    return {r["customer_id"]: r.asDict() for r in df.collect()}


def test_rfm_basket_and_tenure(frame):
    rows = [
        # customer 1: 2 invoices on 2010-12-01 and 1 on 2010-12-31; 3 distinct products
        sale("A1", "P1", qty=2, price="5.00", ts="2010-12-01 09:00", cust=1),  # 10
        sale("A1", "P2", qty=1, price="20.00", ts="2010-12-01 09:00", cust=1),  # 20
        sale("A2", "P1", qty=1, price="5.00", ts="2010-12-01 15:00", cust=1),  # 5
        sale("A3", "P3", qty=3, price="5.00", ts="2010-12-31 10:00", cust=1),  # 15
    ]
    f = _features(frame, rows)[1]
    assert f["frequency"] == 3
    assert f["monetary"] == pytest.approx(50.0)
    assert f["avg_basket_value"] == pytest.approx(50.0 / 3)
    assert f["distinct_products"] == 3
    assert f["recency_days"] == 1  # last purchase 2010-12-31, as_of 2011-01-01
    assert f["tenure_days"] == 31
    assert f["n_purchase_days"] == 2
    assert f["first_purchase_date"] == date(2010, 12, 1)
    assert f["last_purchase_date"] == date(2010, 12, 31)
    assert f["as_of_date"] == AS_OF


def test_missing_customer_ids_are_excluded(frame):
    rows = [
        sale("A1", "P1", cust=1),
        sale("A2", "P1", cust=None),
    ]
    feats = _features(frame, rows)
    assert set(feats) == {1}


def test_activity_on_or_after_as_of_is_ignored(frame):
    """No leakage: a sale or return on the snapshot date itself is not visible."""
    sales_rows = [
        sale("A1", "P1", price="10.00", ts="2010-12-15 10:00", cust=1),
        sale("A2", "P1", price="99.00", ts="2011-01-01 08:00", cust=1),  # on as_of
        sale("A3", "P1", price="99.00", ts="2011-02-01 08:00", cust=2),  # only after as_of
    ]
    return_rows = [sale("C9", "P1", qty=-1, price="10.00", ts="2011-01-05 08:00", cust=1)]
    feats = _features(frame, sales_rows, return_rows)
    assert set(feats) == {1}
    assert feats[1]["monetary"] == pytest.approx(10.0)
    assert feats[1]["frequency"] == 1
    assert feats[1]["returned_value"] == 0.0


def test_mean_interval_is_mean_gap_between_purchase_days(frame):
    rows = [
        # days 2010-11-01, 2010-11-11, 2010-12-11 -> gaps 10 and 30 -> mean 20
        sale("A1", "P1", ts="2010-11-01 09:00", cust=1),
        sale("A2", "P1", ts="2010-11-11 09:00", cust=1),
        sale("A3", "P1", ts="2010-11-11 17:00", cust=1),  # same day: not a new gap
        sale("A4", "P1", ts="2010-12-11 09:00", cust=1),
        # one purchase day only -> undefined (NULL)
        sale("B1", "P1", ts="2010-12-01 09:00", cust=2),
        sale("B2", "P2", ts="2010-12-01 18:00", cust=2),
    ]
    feats = _features(frame, rows)
    assert feats[1]["mean_days_between_purchases"] == pytest.approx(20.0)
    assert feats[2]["mean_days_between_purchases"] is None
    assert feats[2]["frequency"] == 2  # two invoices, but one purchase day
    assert feats[2]["n_purchase_days"] == 1


def test_return_rate_value_based_and_capped(frame):
    sales_rows = [
        sale("A1", "P1", qty=4, price="25.00", ts="2010-12-01 09:00", cust=1),  # 100
        sale("B1", "P1", qty=1, price="10.00", ts="2010-12-01 09:00", cust=2),  # 10
        sale("D1", "P1", qty=1, price="10.00", ts="2010-12-01 09:00", cust=3),  # no returns
    ]
    return_rows = [
        sale("C1", "P1", qty=-1, price="25.00", ts="2010-12-05 09:00", cust=1),  # -25
        sale("C2", "P1", qty=-1, price="5.00", ts="2010-12-06 09:00", cust=1),  # -5
        sale("C3", "P9", qty=-3, price="10.00", ts="2010-12-05 09:00", cust=2),  # -30 > sales
        sale("C4", "P1", qty=-1, price="10.00", ts="2010-12-05 09:00", cust=4),  # returns only
    ]
    feats = _features(frame, sales_rows, return_rows)
    assert feats[1]["returned_value"] == pytest.approx(30.0)
    assert feats[1]["n_return_invoices"] == 2
    assert feats[1]["return_rate"] == pytest.approx(0.30)
    assert feats[2]["returned_value"] == pytest.approx(30.0)  # raw amount kept
    assert feats[2]["return_rate"] == pytest.approx(1.0)  # capped
    assert feats[3]["returned_value"] == 0.0
    assert feats[3]["n_return_invoices"] == 0
    assert feats[3]["return_rate"] == 0.0
    assert 4 not in feats  # returns-only customer excluded


def test_preferred_country_by_revenue_then_invoices_then_name(frame):
    rows = [
        # customer 1: more invoices in France, more revenue in Germany -> Germany
        sale("A1", "P1", price="1.00", cust=1, country="France"),
        sale("A2", "P1", price="1.00", cust=1, country="France"),
        sale("A3", "P1", price="5.00", cust=1, country="Germany"),
        # customer 2: equal revenue, more invoices in Spain -> Spain
        sale("B1", "P1", price="2.00", cust=2, country="Spain"),
        sale("B2", "P1", price="2.00", cust=2, country="Spain"),
        sale("B3", "P1", price="4.00", cust=2, country="EIRE"),
        # customer 3: full tie -> alphabetical
        sale("D1", "P1", price="3.00", cust=3, country="Norway"),
        sale("D2", "P1", price="3.00", cust=3, country="Belgium"),
    ]
    feats = _features(frame, rows)
    assert feats[1]["preferred_country"] == "Germany"
    assert feats[1]["n_countries"] == 2
    assert feats[2]["preferred_country"] == "Spain"
    assert feats[3]["preferred_country"] == "Belgium"


def test_seasonal_share_q4_by_default_and_configurable(frame):
    rows = [
        sale("A1", "P1", price="30.00", ts="2010-11-15 09:00", cust=1),  # Q4
        sale("A2", "P1", price="10.00", ts="2010-06-15 09:00", cust=1),  # not Q4
    ]
    assert _features(frame, rows)[1]["seasonal_revenue_share"] == pytest.approx(0.75)
    summer = _features(frame, rows, seasonal_months=(6, 7, 8))
    assert summer[1]["seasonal_revenue_share"] == pytest.approx(0.25)


def test_tenure_censored_flag(frame):
    rows = [
        sale("A1", "P1", ts="2009-12-20 09:00", cust=1),  # first month of data
        sale("B1", "P1", ts="2010-01-01 09:00", cust=2),  # first day after it
    ]
    feats = _features(frame, rows)
    assert feats[1]["is_tenure_censored"] is True
    assert feats[2]["is_tenure_censored"] is False


def test_data_start_defaults_to_earliest_sale(frame, no_returns):
    sales = frame(
        [
            sale("A1", "P1", ts="2010-03-10 09:00", cust=None),  # earliest line, unknown customer
            sale("B1", "P1", ts="2010-04-05 09:00", cust=1),
            sale("D1", "P1", ts="2010-04-20 09:00", cust=2),
        ]
    )
    feats = {
        r["customer_id"]: r["is_tenure_censored"]
        for r in build_customer_features(sales, no_returns, AS_OF).collect()
    }
    assert feats == {1: True, 2: False}  # censored if first purchase < 2010-04-10


def test_schema_types_order_and_one_row_per_customer(frame, no_returns):
    rows = [sale(f"A{i}", "P1", cust=i % 3 + 1) for i in range(9)]
    df = build_customer_features(frame(rows), no_returns, AS_OF, data_start=DATA_START)
    assert df.columns == FEATURE_COLUMNS
    expected = {name: dtype.lower() for name, dtype, _ in FEATURE_SCHEMA}
    actual = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    assert actual == expected
    assert df.count() == df.select("customer_id").distinct().count() == 3


def test_default_as_of_is_day_after_last_sale(frame):
    sales = frame([sale("A1", "P1", ts="2011-12-09 12:50", cust=1)])
    assert default_as_of(sales) == date(2011, 12, 10)


def test_reconcile_features_counts(frame):
    sales_rows = [
        sale("A1", "P1", price="10.00", ts="2010-12-01 09:00", cust=1),
        sale("A2", "P1", price="10.00", ts="2010-12-03 09:00", cust=1),
        sale("B1", "P1", price="5.00", ts="2010-12-01 09:00", cust=2),
        sale("X1", "P1", price="7.00", ts="2010-12-01 09:00", cust=None),
    ]
    return_rows = [
        sale("C1", "P1", qty=-2, price="5.00", ts="2010-12-02 09:00", cust=2),  # 10 > 5 -> capped
        sale("C2", "P1", qty=-1, price="5.00", ts="2010-12-02 09:00", cust=7),  # returns only
    ]
    sales, returns = frame(sales_rows), frame(return_rows)
    feats = build_customer_features(sales, returns, AS_OF, data_start=DATA_START)
    checks = reconcile_features(feats, sales, returns, AS_OF)
    assert checks["customers"] == checks["expected_customers"] == 2
    assert checks["duplicate_keys"] == 0
    assert checks["monetary_diff"] == pytest.approx(0.0)
    assert checks["single_day_customers"] == checks["null_intervals"] == 1
    assert checks["returns_only_customers"] == 1
    assert checks["return_rate_capped"] == 1
    assert checks["return_rate_range"] == (0.0, 1.0)


def test_describe_features_one_row_per_feature(frame, no_returns):
    rows = [sale(f"A{i}", "P1", price=f"{i + 1}.00", cust=i) for i in range(1, 6)]
    feats = build_customer_features(frame(rows), no_returns, AS_OF, data_start=DATA_START)
    desc = {r["feature"]: r for r in describe_features(feats, ["monetary", "frequency"]).collect()}
    assert set(desc) == {"monetary", "frequency"}
    assert desc["monetary"]["min"] == pytest.approx(2.0)
    assert desc["monetary"]["max"] == pytest.approx(6.0)
    assert desc["monetary"]["p50"] == pytest.approx(4.0)
    assert desc["frequency"]["n_null"] == 0


def test_feature_table_ddl_has_keys_comments_and_timeseries_pk():
    ddl = feature_table_ddl("retail.ml.customer_features")
    assert ddl.startswith("CREATE TABLE IF NOT EXISTS retail.ml.customer_features (")
    assert "customer_id INT NOT NULL" in ddl
    assert "as_of_date DATE NOT NULL" in ddl
    assert "recency_days INT COMMENT" in ddl  # non-key columns stay nullable
    assert "CONSTRAINT customer_features_pk PRIMARY KEY (customer_id, as_of_date TIMESERIES)" in ddl
    assert "computed_at TIMESTAMP" in ddl
    for name in FEATURE_COLUMNS:
        assert f"  {name} " in ddl


def test_ddl_escapes_quotes_in_comments(monkeypatch):
    import features.feature_store as fs

    monkeypatch.setattr(fs, "TABLE_COMMENT", "customer's table")
    assert "COMMENT 'customer\\'s table'" in fs.feature_table_ddl("a.b.c")


def test_snapshot_predicate_and_constraint_name():
    assert snapshot_predicate(date(2011, 12, 10)) == "as_of_date = DATE'2011-12-10'"
    assert constraint_name("retail.ml.customer_features") == "customer_features_pk"
