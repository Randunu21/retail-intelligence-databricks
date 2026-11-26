"""Tests for segmentation.market_basket (Phase 4.4). Small baskets with hand-countable
supports; FP-Growth runs on classic local Spark (it is the same algorithm on serverless)."""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from factories import SILVER_SCHEMA, sale
from segmentation.market_basket import (
    basket_size_summary,
    build_baskets,
    fpgrowth_summary,
    item_support,
    items_above_support,
    threshold_sweep,
    variant_family,
)


@pytest.fixture
def sales(spark):
    rows = [
        # inv1: A, B, C  (B twice on the invoice -> still one item)
        sale("i1", "A", ts="2011-01-03 10:00", cust=1),
        sale("i1", "B", ts="2011-01-03 10:00", cust=1),
        sale("i1", "B", qty=3, ts="2011-01-03 10:00", cust=1),
        sale("i1", "C", ts="2011-01-03 10:00", cust=1),
        # inv2: A, B   (no customer)
        sale("i2", "A", ts="2011-01-04 10:00", cust=None, country="France"),
        sale("i2", "B", ts="2011-01-04 10:00", cust=None, country="France"),
        # inv3: A, C
        sale("i3", "A", ts="2011-01-05 10:00", cust=2),
        sale("i3", "C", ts="2011-01-05 10:00", cust=2),
        # inv4: D only
        sale("i4", "D", ts="2011-01-06 10:00", cust=3),
        # inv5: after as_of
        sale("i5", "A", ts="2011-02-01 10:00", cust=3),
        sale("i5", "D", ts="2011-02-01 10:00", cust=3),
    ]
    return spark.createDataFrame(rows, SILVER_SCHEMA)


@pytest.fixture
def baskets(sales):
    return build_baskets(sales, as_of="2011-01-31")


def test_build_baskets(baskets):
    got = {r["invoice"]: r for r in baskets.collect()}
    assert set(got) == {"i1", "i2", "i3", "i4"}  # i5 is on/after as_of
    assert got["i1"]["items"] == ["A", "B", "C"] and got["i1"]["n_items"] == 3
    assert got["i2"]["has_customer"] is False and got["i2"]["country"] == "France"
    assert got["i4"]["items"] == ["D"]


def test_build_baskets_without_cutoff(sales):
    assert build_baskets(sales).count() == 5


def test_basket_size_summary(baskets):
    s = basket_size_summary(baskets)
    assert s["baskets"] == 4
    assert s["mean_items"] == pytest.approx((3 + 2 + 2 + 1) / 4)
    assert s["max_items"] == 3
    assert s["share_single_item"] == pytest.approx(0.25)
    assert s["share_known_customer"] == pytest.approx(0.75)
    assert s["item_pairs"] == pytest.approx(3 + 1 + 1 + 0)


def test_item_support(baskets):
    sup = {r["stock_code"]: r for r in item_support(baskets).collect()}
    assert sup["A"]["basket_count"] == 3 and sup["A"]["support"] == pytest.approx(0.75)
    assert sup["D"]["support"] == pytest.approx(0.25)
    assert items_above_support(item_support(baskets), (0.5, 0.25, 0.8)) == {
        0.5: 3,
        0.25: 4,
        0.8: 0,
    }


def test_variant_family(spark):
    df = spark.createDataFrame(
        [("85099B",), ("85099C",), ("79323W",), ("22423",), ("DCGS0003",), ("15056BL",), ("PADS",)],
        "stock_code STRING",
    )
    got = [r[0] for r in df.select(variant_family(F.col("stock_code"))).collect()]
    assert got == ["85099", "85099", "79323", "22423", "DCGS0003", "15056", "PADS"]


def test_fpgrowth_summary_counts(baskets):
    # support >= 0.5 (2 of 4 baskets): items A(3) B(2) C(2); pairs AB(2) AC(2); BC=1 no
    r = fpgrowth_summary(baskets, 0.5, 0.5)
    assert r["error"] is None
    assert r["itemsets_by_size"] == {1: 3, 2: 2}
    assert (r["frequent_items"], r["frequent_pairs"], r["frequent_triples"]) == (3, 2, 0)
    # rules: B->A (1.0), A->B (2/3), C->A (1.0), A->C (2/3); lift = conf / sup(consequent)
    assert r["rules"] == 4
    assert r["max_lift"] == pytest.approx(1.0 / 0.75)
    assert r["rules_lift_gt_1"] == 4
    assert r["same_family_pairs"] == 0


def test_same_family_pairs(spark):
    df = spark.createDataFrame(
        [(i, ["85099B", "85099C"]) for i in range(3)] + [(9, ["22423"])],
        "invoice INT, items ARRAY<STRING>",
    )
    assert fpgrowth_summary(df, 0.5, 0.5)["same_family_pairs"] == 1


def test_threshold_sweep_order_and_stop(baskets):
    seen = []
    res = threshold_sweep(baskets, [0.25, 0.5], 0.1, on_result=seen.append)
    assert [r["min_support"] for r in res] == [0.5, 0.25]  # high -> low
    assert seen == res
    # 0.75 -> 1 itemset (A), 0.5 -> 5 itemsets > 1: stop there, 0.25 is never fitted
    stopped = threshold_sweep(baskets, [0.25, 0.5, 0.75], 0.1, max_itemsets=1)
    assert [r["min_support"] for r in stopped] == [0.75, 0.5]
    assert "itemsets" in stopped[-1]["stopped_sweep"]


def test_fpgrowth_summary_records_errors(baskets):
    r = fpgrowth_summary(baskets.withColumnRenamed("items", "other"), 0.5, 0.5)
    assert r["error"] is not None and "seconds" in r
