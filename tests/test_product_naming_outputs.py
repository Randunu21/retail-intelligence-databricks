"""Tests for segmentation.product_naming and segmentation.product_outputs (Phase 4.3b)."""

from __future__ import annotations

from datetime import date

import pytest
from pyspark.sql import functions as F

from segmentation.naming import NamingError
from segmentation.product_naming import (
    PRODUCT_ACTION_PLAYBOOK,
    PRODUCT_SEGMENT_DEFINITIONS,
    add_product_groups,
    group_profile,
    name_product_segments,
    premium_price_threshold,
)
from segmentation.product_outputs import (
    PRODUCT_SEGMENTS_COLUMNS,
    build_product_segments_output,
    product_segments_ddl,
)

FEATURES = [
    "units_per_week",
    "median_unit_price",
    "seasonal_revenue_share",
    "monthly_units_cv",
    "days_since_last_sale",
    "non_uk_revenue_share",
    "active_week_share",
    "unique_customers",
]
# Real centres from run 47f3364b (4.3a), by segment_id.
REAL_CENTERS = [
    [0.652, -0.160, 0.174, -0.616, -0.620, 0.215, 0.788, 0.648],
    [-0.592, 0.026, -0.455, 1.136, 1.486, -0.144, -1.097, -0.631],
    [-1.014, 0.394, 0.093, 0.251, -0.167, -0.396, -0.758, -0.956],
]


def test_names_from_real_centres():
    names = name_product_segments(REAL_CENTERS, FEATURES)
    assert {sid: d.key for sid, d in names.items()} == {0: "core", 1: "fading", 2: "slow"}


def test_naming_collision_raises():
    # Segment 0 is both fastest and most recently unsold: signatures collide.
    centers = [[1.0, 0, 0, 0, 2.0, 0, 0, 0], [0.0, 0, 0, 0, 0.0, 0, 0, 0], [-1.0] + [0] * 7]
    with pytest.raises(NamingError, match="matches both"):
        name_product_segments(centers, FEATURES)
    with pytest.raises(NamingError, match="names for"):
        name_product_segments(REAL_CENTERS[:2], FEATURES)


def test_playbook_covers_every_group():
    keys = {d.key for d in PRODUCT_SEGMENT_DEFINITIONS} | {"long_tail", "new"}
    assert {a["segment"] for a in PRODUCT_ACTION_PLAYBOOK.values()} == keys


@pytest.fixture
def grouped(spark):
    rows = [
        # stock, eligibility, segment_id, revenue, q4, days_since, price, return, upw
        ("C1", "clustered", 0, 100.0, 0.8, 1, 2.0, 0.0, 50.0),  # core seasonal
        ("C2", "clustered", 0, 80.0, 0.2, 1, 2.0, 0.12, 40.0),  # core staple, high return
        ("F1", "clustered", 1, 10.0, 0.2, 400, 3.0, 0.0, 2.0),  # fading discontinued
        ("F2", "clustered", 1, 10.0, 0.2, 100, 3.0, 0.0, 2.0),  # fading declining
        ("S1", "clustered", 2, 5.0, 0.9, 5, 30.0, 0.0, 1.0),  # slow premium (before seasonal)
        ("S2", "clustered", 2, 5.0, 0.9, 5, 3.0, 0.0, 1.0),  # slow seasonal
        ("S3", "clustered", 2, 5.0, 0.1, 5, 3.0, 0.0, 1.0),  # slow low demand
        ("L1", "low_volume", None, 2.0, 0.1, 300, 3.0, 0.0, 0.1),
        ("N1", "new_product", None, 3.0, 1.0, 1, 3.0, 0.0, 9.0),
    ]
    df = spark.createDataFrame(
        rows,
        "stock_code STRING, cluster_eligibility STRING, segment_id INT, revenue DOUBLE, "
        "seasonal_revenue_share DOUBLE, days_since_last_sale INT, median_unit_price DOUBLE, "
        "return_rate DOUBLE, units_per_week DOUBLE",
    )
    names = name_product_segments(REAL_CENTERS, FEATURES)
    return add_product_groups(df, names, premium_min=20.0)


def test_action_groups(grouped):
    got = {r["stock_code"]: (r["segment_name"], r["action_group"]) for r in grouped.collect()}
    assert got == {
        "C1": ("Core Fast Movers", "Core: Seasonal Best-Sellers"),
        "C2": ("Core Fast Movers", "Core: Year-round Staples"),
        "F1": ("Fading Lines", "Fading: Discontinued"),
        "F2": ("Fading Lines", "Fading: Declining"),
        "S1": ("Slow Movers", "Slow Movers: Premium Niche"),
        "S2": ("Slow Movers", "Slow Movers: Seasonal"),
        "S3": ("Slow Movers", "Slow Movers: Low Demand"),
        "L1": ("Long Tail", "Long Tail"),
        "N1": ("New Products", "New Products"),
    }
    flags = {r["stock_code"]: r for r in grouped.collect()}
    assert flags["C2"]["is_high_return"] is True and flags["C1"]["is_high_return"] is False
    assert flags["L1"]["segment_key"] == "long_tail" and flags["N1"]["segment_key"] == "new"
    assert all(r["recommended_action"] for r in grouped.collect())


def test_group_profile(grouped):
    rows = group_profile(grouped)
    assert [r["action_group"] for r in rows] == list(PRODUCT_ACTION_PLAYBOOK)  # all 9, in order
    by = {r["action_group"]: r for r in rows}
    assert sum(r["products"] for r in rows) == 9
    assert sum(r["revenue_share"] for r in rows) == pytest.approx(1.0)
    assert by["Core: Seasonal Best-Sellers"]["revenue_share"] == pytest.approx(100 / 220)
    assert by["Core: Year-round Staples"]["pct_high_return"] == pytest.approx(1.0)
    order = list(PRODUCT_ACTION_PLAYBOOK)
    assert [order.index(r["action_group"]) for r in rows] == sorted(
        order.index(r["action_group"]) for r in rows
    )


def test_premium_threshold(spark):
    df = spark.createDataFrame([(float(p),) for p in range(1, 11)], "median_unit_price DOUBLE")
    assert premium_price_threshold(df, 0.9) == pytest.approx(9.0)


def test_output_table(grouped, spark):
    desc = spark.createDataFrame([("C1", "STAR LIGHT")], "stock_code STRING, description STRING")
    feats = grouped.withColumn("as_of_date", F.lit(date(2011, 12, 10)))
    for c in ("revenue_share", "active_week_share", "non_uk_revenue_share"):
        feats = feats.withColumn(c, F.lit(0.0))
    for c in ("n_invoices", "unique_customers"):
        feats = feats.withColumn(c, F.lit(1))
    out = build_product_segments_output(feats, desc, "run123")
    assert out.columns == PRODUCT_SEGMENTS_COLUMNS
    rows = {r["stock_code"]: r for r in out.collect()}
    assert rows["C1"]["description"] == "STAR LIGHT" and rows["C2"]["description"] is None
    assert rows["L1"]["segment_id"] is None and rows["L1"]["segment_name"] == "Long Tail"
    assert {r["model_run_id"] for r in rows.values()} == {"run123"}
    ddl = product_segments_ddl("retail.gold.product_segments")
    assert "CONSTRAINT product_segments_pk PRIMARY KEY (stock_code, as_of_date)" in ddl
    assert "stock_code STRING NOT NULL" in ddl and "segment_id INT COMMENT" in ddl
