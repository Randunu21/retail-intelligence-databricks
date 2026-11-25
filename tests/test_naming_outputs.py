"""Tests for segmentation.naming and segmentation.outputs (Phase 3.5b)."""

from __future__ import annotations

from datetime import date

import pytest

from segmentation.naming import (
    ACTION_PLAYBOOK,
    SEGMENT_DEFINITIONS,
    NamingError,
    add_action_groups,
    name_segments,
    top_value_threshold,
)
from segmentation.outputs import (
    SEGMENTS_COLUMNS,
    build_segments_output,
    render_segment_profiles_md,
    segments_table_ddl,
)

FEATURES = [
    "recency_days",
    "monetary",
    "frequency",
    "tenure_days",
    "distinct_products",
    "seasonal_revenue_share",
]
# real Phase 3.5a centres (run b27fc5b7)
CENTERS = [
    [-0.564, 0.917, 0.981, 0.608, 0.821, -0.099],
    [-0.314, -0.491, -0.573, -0.985, -0.34, 1.153],
    [0.874, -0.677, -0.688, 0.042, -0.68, -0.747],
]


def test_name_segments_on_real_centres():
    names = name_segments(CENTERS, FEATURES)
    assert {sid: d.key for sid, d in names.items()} == {
        0: "loyal",
        1: "seasonal_new",
        2: "hibernating",
    }


def test_name_segments_follows_profiles_not_ids():
    swapped = [CENTERS[2], CENTERS[0], CENTERS[1]]
    names = name_segments(swapped, FEATURES)
    assert {sid: d.key for sid, d in names.items()} == {
        0: "hibernating",
        1: "loyal",
        2: "seasonal_new",
    }


def test_name_segments_raises_when_profiles_collide():
    collide = [list(CENTERS[0]), list(CENTERS[1]), list(CENTERS[2])]
    collide[0][5] = 2.0  # loyal segment now also the most seasonal
    with pytest.raises(NamingError, match="review"):
        name_segments(collide, FEATURES)
    with pytest.raises(NamingError):
        name_segments(CENTERS[:2], FEATURES)


def test_every_action_group_has_a_playbook_entry_for_a_known_segment():
    keys = {d.key for d in SEGMENT_DEFINITIONS}
    assert {a["segment"] for a in ACTION_PLAYBOOK.values()} == keys
    assert all(a["action"] and a["kpi"] for a in ACTION_PLAYBOOK.values())


@pytest.fixture
def segmented(spark):
    # (cid, segment_id, recency, frequency, monetary, tenure, n_days, mean_gap)
    rows = [
        (1, 0, 200, 10, 9000.0, 700, 10, 50.0),  # overdue (200 > 100) + top value -> At Risk
        (2, 0, 20, 10, 9000.0, 700, 10, 50.0),  # top value, not overdue -> Champions
        (3, 0, 20, 5, 500.0, 600, 5, 60.0),  # Active
        (4, 1, 30, 1, 100.0, 60, 1, None),  # new
        (5, 1, 40, 3, 300.0, 400, 3, 150.0),  # seasonal gifter (not overdue: 40 < 300)
        (6, 2, 400, 2, 200.0, 600, 2, 100.0),  # former repeat (overdue too)
        (7, 2, 500, 1, 50.0, 500, 1, None),  # one-time
    ]
    schema = (
        "customer_id INT, segment_id INT, recency_days INT, frequency INT, monetary DOUBLE, "
        "tenure_days INT, n_purchase_days INT, mean_days_between_purchases DOUBLE"
    )
    return spark.createDataFrame(rows, schema)


def test_add_action_groups(segmented):
    names = name_segments(CENTERS, FEATURES)
    out = {r["customer_id"]: r for r in add_action_groups(segmented, names, 5000.0).collect()}
    assert out[1]["action_group"] == "Loyal: At Risk"
    assert out[2]["action_group"] == "Loyal: Champions"
    assert out[3]["action_group"] == "Loyal: Active"
    assert out[4]["action_group"] == "Seasonal & New: New Customers"
    assert out[5]["action_group"] == "Seasonal & New: Seasonal Gifters"
    assert out[6]["action_group"] == "Hibernating: Former Repeat Buyers"
    assert out[7]["action_group"] == "Hibernating: One-time Buyers"
    assert out[1]["segment_name"] == "Loyal Customers"
    assert out[4]["is_new_customer"] is True
    assert out[4]["is_overdue"] is False  # NULL gap -> not overdue
    assert out[6]["is_overdue"] is True
    assert out[2]["is_top_value"] is True and out[3]["is_top_value"] is False
    assert out[7]["recommended_action"] == ACTION_PLAYBOOK["Hibernating: One-time Buyers"]["action"]


def test_top_value_threshold(spark):
    df = spark.createDataFrame([(float(i),) for i in range(1, 11)], "monetary DOUBLE")
    assert top_value_threshold(df, 0.9) == 9.0


def test_build_segments_output_columns(segmented):
    from pyspark.sql import functions as F

    names = name_segments(CENTERS, FEATURES)
    grouped = (
        add_action_groups(segmented, names, 5000.0)
        .withColumn("as_of_date", F.lit(date(2011, 12, 10)))
        .withColumn("avg_basket_value", F.col("monetary") / F.col("frequency"))
        .withColumn("seasonal_revenue_share", F.lit(0.5))
    )
    out = build_segments_output(grouped, "run123")
    assert out.columns == SEGMENTS_COLUMNS
    r = out.where("customer_id = 1").first()
    assert r["model_run_id"] == "run123" and r["scored_at"] is not None
    assert out.count() == 7


def test_segments_table_ddl():
    ddl = segments_table_ddl("retail.gold.customer_segments")
    assert ddl.startswith("CREATE TABLE IF NOT EXISTS retail.gold.customer_segments (")
    assert "CONSTRAINT customer_segments_pk PRIMARY KEY (customer_id, as_of_date)" in ddl
    assert "customer_id INT NOT NULL" in ddl
    for c in SEGMENTS_COLUMNS:
        assert f"  {c} " in ddl


def test_render_markdown_contains_real_numbers_and_all_groups():
    names = name_segments(CENTERS, FEATURES)
    seg_rows = [
        {"segment_id": 0, "customers": 2308, "customer_share": 0.394, "revenue_share": 0.86},
        {"segment_id": 1, "customers": 1513, "customer_share": 0.259, "revenue_share": 0.069},
        {"segment_id": 2, "customers": 2031, "customer_share": 0.347, "revenue_share": 0.07},
    ]
    groups = [
        {
            "action_group": g,
            "customers": 10,
            "customer_share": 0.1,
            "revenue_share": 0.1,
            "median_recency_days": 5,
            "median_monetary": 100.0,
        }
        for g in ACTION_PLAYBOOK
    ]
    md = render_segment_profiles_md(
        as_of=date(2011, 12, 10),
        n_customers=5852,
        names=names,
        segment_profiles=seg_rows,
        group_profiles=list(reversed(groups)),
        model_facts={"Algorithm": "KMeans, k = 3"},
        images={"Sizes": "img/phase3/segment_sizes.png"},
    )
    assert "| | Loyal Customers | Seasonal & New | Hibernating |" in md
    assert "| Customers | 2,308 | 1,513 | 2,031 |" in md
    assert "| Share of revenue | 86.0% | 6.9% | 7.0% |" in md
    assert "Median spend | – |" in md  # missing values shown as a dash, never invented
    assert "![Sizes](img/phase3/segment_sizes.png)" in md
    positions = [md.index(f"| {g} |") for g in ACTION_PLAYBOOK]
    assert positions == sorted(positions)  # playbook order, whatever the input order
