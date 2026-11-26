"""Tests for modeling.build_training_set (Phase 5): split plan, labels, and NO LEAKAGE.

Hand-made Silver rows (tests/factories.sale). Cutoff in most tests: 2011-01-01, horizon 90
days -> label window [2011-01-01, 2011-04-01).
"""

from __future__ import annotations

from datetime import date

import pytest

from factories import SILVER_SCHEMA, sale
from modeling.build_training_set import (
    MODEL_FEATURES,
    LeakageError,
    SplitPlan,
    activity_features,
    build_labels,
    build_snapshot,
    build_training_set,
    check_training_set,
    label_summary,
    output_columns,
    plan_cutoffs,
    seasonal_days,
    snapshot_features,
    stale_cutoffs_sql,
    truncation_check,
    validate_plan,
)
from modeling.environment import package_versions, run_check

CUTOFF = date(2011, 1, 1)
DATA_START = date(2009, 12, 1)


@pytest.fixture
def frame(spark):
    def _make(rows):
        return spark.createDataFrame(rows, SILVER_SCHEMA)

    return _make


def _history():
    """Two customers with history before the cutoff."""
    return [
        sale("A1", "X", qty=2, price="5.00", ts="2010-06-01 10:00", cust=1),
        sale("A2", "Y", qty=1, price="3.00", ts="2010-12-20 10:00", cust=1),
        sale("B1", "X", qty=1, price="10.00", ts="2010-03-15 10:00", cust=2),
    ]


def _future():
    """Events on/after the cutoff: label window, after it, returns, a brand-new customer."""
    return [
        sale("F1", "Z", qty=100, price="99.00", ts="2011-01-01 09:00", cust=1),  # on cutoff day
        sale("F2", "Z", qty=50, price="20.00", ts="2011-02-10 09:00", cust=2),
        sale("F3", "Z", qty=7, price="1.00", ts="2011-06-01 09:00", cust=2),  # after window
        sale("F4", "Z", qty=1, price="1.00", ts="2011-01-05 09:00", cust=3),  # new customer
        sale("F5", "Z", qty=1, price="1.00", ts="2011-01-05 09:00", cust=None),
    ]


def _future_returns():
    return [sale("C9", "X", qty=-2, price="5.00", ts="2011-01-03 10:00", cust=1)]


# ----------------------------------------------------------------------------- split plan


def test_plan_on_real_dataset_dates_fits_the_data():
    plan = plan_cutoffs(DATA_START, date(2011, 12, 10))
    assert plan.test_cutoff == date(2011, 9, 11)  # 2011-12-10 minus 90 days
    assert plan.validation_cutoff == date(2011, 6, 13)
    assert plan.label_end(plan.test_cutoff) == date(2011, 12, 10)
    assert plan.label_end(plan.validation_cutoff) == plan.test_cutoff
    assert plan.train_cutoffs[0] == date(2010, 3, 1)  # first month with 90 days of history
    assert plan.train_cutoffs[-1] == date(2011, 3, 1)  # 2011-03-01 + 90 = 2011-05-30 <= 06-13
    assert len(plan.train_cutoffs) == 13
    assert all(plan.label_end(c) <= plan.validation_cutoff for c in plan.train_cutoffs)
    d = plan.to_dict()
    assert len(d["windows"]) == 15 and d["windows"][-1]["split"] == "test"


def test_validate_plan_rejects_overlapping_or_out_of_data_windows():
    end = date(2011, 12, 10)
    # training label window (2011-05-01 + 90) runs into the validation period
    bad_train = SplitPlan(
        (date(2010, 3, 1), date(2011, 5, 1)), date(2011, 6, 13), date(2011, 9, 11)
    )
    with pytest.raises(LeakageError, match="after the validation cutoff"):
        validate_plan(bad_train, DATA_START, end)
    # test label window ends after the data -> late labels would wrongly be 0
    bad_test = SplitPlan((date(2010, 3, 1),), date(2011, 6, 13), date(2011, 10, 1))
    with pytest.raises(LeakageError, match="after the data"):
        validate_plan(bad_test, DATA_START, end)
    # validation window overlaps the test period
    bad_val = SplitPlan((date(2010, 3, 1),), date(2011, 7, 1), date(2011, 9, 11))
    with pytest.raises(LeakageError, match="validation label window"):
        validate_plan(bad_val, DATA_START, end)
    # too little history
    short = SplitPlan((date(2010, 1, 1),), date(2011, 6, 13), date(2011, 9, 11))
    with pytest.raises(LeakageError, match="history"):
        validate_plan(short, DATA_START, end)


def test_stale_cutoffs_sql_keeps_only_planned_cutoffs():
    plan = SplitPlan((date(2010, 3, 1),), date(2011, 6, 13), date(2011, 9, 11))
    sql = stale_cutoffs_sql("retail.ml.t", plan)
    assert sql.startswith("DELETE FROM retail.ml.t WHERE cutoff_date NOT IN (")
    assert "DATE'2010-03-01'" in sql and "DATE'2011-09-11'" in sql


def test_seasonal_days_counts_q4_days_in_the_window():
    assert seasonal_days(date(2011, 9, 11)) == 70  # Oct 31 + Nov 30 + Dec 1..9
    assert seasonal_days(date(2011, 3, 1)) == 0
    assert seasonal_days(date(2010, 12, 1)) == 31  # only December
    assert seasonal_days(date(2011, 1, 1), horizon_days=10) == 0


# ----------------------------------------------------------------------------- labels


def test_label_window_boundaries(frame):
    rows = [sale("H1", "X", ts="2010-11-01 10:00", cust=c) for c in (1, 2, 3, 4)] + [
        sale("L1", "X", ts="2011-01-01 00:01", cust=1),  # first minute of the window -> 1
        sale("L2", "X", ts="2011-03-31 23:59", cust=2),  # last day (cutoff + 89) -> 1
        sale("L3", "X", ts="2011-04-01 08:00", cust=3),  # cutoff + 90: outside -> 0
        sale("L4", "X", ts="2010-12-31 23:59", cust=4),  # day before cutoff: a feature -> 0
    ]
    sales = frame(rows)
    snap = build_snapshot(sales, frame([]), CUTOFF, "train", DATA_START)
    got = {r["customer_id"]: r["label"] for r in snap.collect()}
    assert got == {1: 1, 2: 1, 3: 0, 4: 0}
    labels = {r["customer_id"]: r["label_orders"] for r in build_labels(sales, CUTOFF).collect()}
    assert labels == {1: 1, 2: 1}


def test_sale_on_cutoff_day_is_label_not_feature(frame):
    sales = frame(_history() + _future())
    snap = {
        r["customer_id"]: r
        for r in build_snapshot(sales, frame([]), CUTOFF, "train", DATA_START).collect()
    }
    c1 = snap[1]
    assert c1["label"] == 1
    assert c1["frequency"] == 2  # A1, A2 only (F1 on the cutoff day is not counted)
    assert c1["monetary"] == pytest.approx(13.0)
    assert c1["last_purchase_date"] == date(2010, 12, 20)
    assert c1["recency_days"] == 12
    assert snap[2]["label"] == 1 and snap[2]["label_orders"] == 1  # F2 only (F3 after window)


def test_population_is_customers_known_before_cutoff(frame):
    sales = frame(_history() + _future())
    snap = build_snapshot(sales, frame([]), CUTOFF, "train", DATA_START)
    assert sorted(r["customer_id"] for r in snap.collect()) == [1, 2]  # 3 is new, None unknown


# ----------------------------------------------------------------------------- NO LEAKAGE


def _feature_rows(df):
    cols = [c for c in df.columns if c not in ("label", "label_orders")]
    return {r["customer_id"]: tuple(r[c] for c in cols) for r in df.select(cols).collect()}


def test_no_feature_uses_events_on_or_after_cutoff(frame):
    """THE leakage test: adding any amount of future activity (big orders, returns, new
    customers, events after the label window) changes the labels but not one feature value."""
    past_only = build_snapshot(frame(_history()), frame([]), CUTOFF, "train", DATA_START)
    with_future = build_snapshot(
        frame(_history() + _future()), frame(_future_returns()), CUTOFF, "train", DATA_START
    )
    assert _feature_rows(past_only) == _feature_rows(with_future)
    # ... while the labels did change (the future really was in the input)
    assert {r["customer_id"]: r["label"] for r in past_only.collect()} == {1: 0, 2: 0}
    assert {r["customer_id"]: r["label"] for r in with_future.collect()} == {1: 1, 2: 1}


def test_every_model_feature_is_covered_by_the_leakage_test(frame):
    snap = build_snapshot(frame(_history() + _future()), frame([]), CUTOFF, "train", DATA_START)
    assert set(MODEL_FEATURES) <= set(snap.columns)
    assert not {"label", "label_orders", "label_end_date", "split"} & set(MODEL_FEATURES)
    assert snap.columns == output_columns()


def test_truncation_check_finds_no_differences(frame):
    sales = frame(_history() + _future())
    returns = frame(_future_returns())
    r = truncation_check(sales, returns, CUTOFF, DATA_START)
    assert r["rows_full"] == r["rows_truncated"] == 2
    assert r["only_in_full"] == 0 and r["only_in_truncated"] == 0


def test_truncation_check_would_catch_a_leaky_feature(frame, monkeypatch):
    """Sanity check of the check itself: a feature that peeks past the cutoff is detected."""
    import modeling.build_training_set as bts

    real = bts.activity_features

    def leaky(sales, as_of, windows=bts.ACTIVITY_WINDOWS):
        from pyspark.sql import functions as F

        total = sales.groupBy("customer_id").agg(F.count("*").alias("all_lines_ever"))
        return real(sales, as_of, windows).join(total, "customer_id", "left")

    monkeypatch.setattr(bts, "activity_features", leaky)
    r = truncation_check(frame(_history() + _future()), frame([]), CUTOFF, DATA_START)
    assert r["only_in_full"] > 0


# ----------------------------------------------------------------------------- features


def test_activity_windows_are_strictly_before_cutoff(frame):
    rows = [
        sale("W1", "X", price="1.00", ts="2010-12-02 10:00", cust=1),  # cutoff - 30 -> in 30d
        sale("W2", "X", price="2.00", ts="2010-12-01 10:00", cust=1),  # cutoff - 31 -> 90d only
        sale("W3", "X", price="4.00", ts="2010-03-01 10:00", cust=1),  # 306 days -> 365d only
        sale("W4", "X", price="8.00", ts="2011-01-01 10:00", cust=1),  # on cutoff -> nowhere
    ]
    r = activity_features(frame(rows), CUTOFF, windows=(30, 90, 365)).first()
    assert (r["orders_last_30d"], r["revenue_last_30d"]) == (1, 1.0)
    assert (r["orders_last_90d"], r["revenue_last_90d"]) == (2, 3.0)
    assert (r["orders_last_365d"], r["revenue_last_365d"]) == (3, 7.0)


def test_snapshot_features_flags_and_calendar(frame):
    rows = _history() + [sale("D1", "X", ts="2010-07-01 10:00", cust=5, country="Germany")]
    f = {
        r["customer_id"]: r
        for r in snapshot_features(frame(rows), frame([]), CUTOFF, DATA_START).collect()
    }
    assert f[1]["is_uk"] == 1 and f[5]["is_uk"] == 0
    assert f[1]["label_window_q4_days"] == 0  # Jan-Mar window
    assert f[1]["cutoff_date"] == CUTOFF


# ----------------------------------------------------------------------------- whole set


def test_training_set_checks_and_summary(frame):
    sales = frame(_history() + _future())
    plan = SplitPlan((date(2010, 9, 1),), date(2011, 1, 1), date(2011, 4, 1))
    ts = build_training_set(sales, frame([]), plan, DATA_START)
    checks = check_training_set(ts)
    assert checks["duplicate_keys"] == 0
    assert checks["null_labels"] == 0 and checks["bad_labels"] == 0
    assert checks["purchases_on_or_after_cutoff"] == 0
    assert checks["min_recency_days"] >= 1
    summary = {str(r["cutoff_date"]): r for r in label_summary(ts).collect()}
    assert summary["2010-09-01"]["split"] == "train"
    assert summary["2011-01-01"]["positives"] == 2  # both customers come back in Jan/Feb
    assert summary["2011-04-01"]["split"] == "test"


def test_build_snapshot_rejects_unknown_split(frame):
    with pytest.raises(ValueError, match="split"):
        build_snapshot(frame(_history()), frame([]), CUTOFF, "holdout", DATA_START)


# ----------------------------------------------------------------------------- probe helpers


def test_package_versions_and_run_check():
    v = package_versions(["pytest", "surely-not-installed-pkg-xyz"])
    assert v["pytest"] and v["surely-not-installed-pkg-xyz"] is None
    assert run_check("ok", lambda: 1) == {"check": "ok", "ok": True, "detail": 1}
    bad = run_check("bad", lambda: 1 / 0)
    assert bad["ok"] is False and "ZeroDivisionError" in bad["detail"]
