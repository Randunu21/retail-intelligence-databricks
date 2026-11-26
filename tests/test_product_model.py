"""Tests for segmentation.product_model (Phase 4.3a).

Three obvious product groups on two features (velocity, price), so the right segments are
known: fast & cheap (F*), medium (M*), slow & expensive (S*).
"""

from __future__ import annotations

import pytest

from segmentation.clustering import run_trial
from segmentation.metrics import silhouette
from segmentation.preprocessing import apply_preprocessor, fit_preprocessor
from segmentation.product_model import (
    SEGMENT_COL,
    SEGMENT_LABEL_COL,
    ProductSegmentModel,
    label_all_products,
    order_leaves,
    product_model_from_trial,
    product_segment_profile,
    top_products,
)
from segmentation.product_segments import CLUSTERED, ELIGIBILITY_COL, LOW_VOLUME

FEATS = ["units_per_week", "median_unit_price"]
GROUPS = {"F": (100.0, 1.0), "M": (10.0, 5.0), "S": (1.0, 20.0)}


@pytest.fixture(scope="module")
def setup(spark):
    rows = []
    for g, (u, p) in GROUPS.items():
        for i in range(6):
            rows.append((f"{g}{i}", u * (1 + 0.02 * i), p * (1 + 0.01 * i)))
    raw = spark.createDataFrame(
        rows, "stock_code STRING, units_per_week DOUBLE, median_unit_price DOUBLE"
    )
    params = fit_preprocessor(raw, FEATS, FEATS, (0.0, 1.0))
    prepared = apply_preprocessor(raw, params, keep=["stock_code"])
    z = [f"z_{f}" for f in FEATS]
    trials = [
        run_trial(a, prepared, 3, z, seed=7).to_dict() for a in ("bisecting_kmeans", "kmeans")
    ]
    return raw, params, prepared, trials


def _groups(df):
    """segment id per group letter (each group must map to exactly one segment)."""
    out = {}
    for r in df.select("stock_code", SEGMENT_COL).collect():
        out.setdefault(r["stock_code"][0], set()).add(r[SEGMENT_COL])
    assert all(len(v) == 1 for v in out.values()), out
    return {g: v.pop() for g, v in out.items()}


@pytest.mark.parametrize("algorithm", ["bisecting_kmeans", "kmeans"])
def test_model_from_trial_orders_by_velocity(setup, algorithm):
    raw, params, prepared, trials = setup
    model = product_model_from_trial(trials, algorithm, 3, params, prepared, seed=7)
    seg = _groups(model.assign_prepared(prepared))
    assert seg == {"F": 0, "M": 1, "S": 2}  # 0 = fastest
    assert model.leaf_sizes == [6, 6, 6]
    assert model.leaf_centers[0][0] > model.leaf_centers[1][0] > model.leaf_centers[2][0]
    if algorithm == "bisecting_kmeans":
        assert len(model.splits) == 2 and model.centers == []
    else:
        assert len(model.centers) == 3 and model.splits == []


def test_reproduces_trial_silhouette(setup):
    raw, params, prepared, trials = setup
    model = product_model_from_trial(trials, "bisecting_kmeans", 3, params, prepared, seed=7)
    sil = silhouette(model.assign_prepared(prepared), "features", SEGMENT_COL)
    trial = next(t for t in trials if t["algorithm"] == "bisecting_kmeans")
    assert sil == pytest.approx(trial["metrics"]["silhouette"], abs=1e-9)


def test_score_raw_features_and_json_roundtrip(setup):
    raw, params, prepared, trials = setup
    model = product_model_from_trial(trials, "bisecting_kmeans", 3, params, prepared, seed=7)
    again = ProductSegmentModel.from_dict(model.to_dict())
    assert again.to_dict() == model.to_dict()
    scored = again.score(raw)
    assert "features" not in scored.columns
    assert _groups(scored) == {"F": 0, "M": 1, "S": 2}


def test_missing_trial_and_empty_id_map(setup):
    raw, params, prepared, trials = setup
    with pytest.raises(ValueError, match="no successful"):
        product_model_from_trial(trials, "gmm", 3, params, prepared, seed=7)
    bare = ProductSegmentModel("kmeans", 1, 0, FEATS, params.to_dict(), centers=[[0.0, 0.0]])
    with pytest.raises(ValueError, match="id_map"):
        bare.assign_prepared(prepared)


def test_order_leaves_ties_and_sizes():
    stats = {
        5: {"n": 2, "centroid": [1.0, 0.0]},
        3: {"n": 4, "centroid": [1.0, 9.0]},
        0: {"n": 1, "centroid": [-2.0, 0.0]},
    }
    id_map, centers, sizes = order_leaves(stats, ["a", "b"], "a")
    assert id_map == {3: 0, 5: 1, 0: 2}  # tie on 'a' -> lower raw id first
    assert sizes == [4, 2, 1]
    assert centers[2] == [-2.0, 0.0]


@pytest.fixture
def labelled(spark):
    feats = spark.createDataFrame(
        [
            ("A", CLUSTERED, 100.0, 10, 50, 5.0, 2.0, 0.2, 0.1, 0.0, 10, True),
            ("B", CLUSTERED, 50.0, 8, 30, 3.0, 12.0, 0.8, 0.5, 0.1, 200, False),
            ("C", CLUSTERED, 30.0, 6, 20, 1.0, 4.0, 0.3, 0.0, 0.0, 400, False),
            ("D", LOW_VOLUME, 20.0, 2, 5, 0.1, 1.0, 0.1, 0.0, 0.0, 5, False),
        ],
        "stock_code STRING, cluster_eligibility STRING, revenue DOUBLE, n_invoices INT, "
        "units_sold INT, units_per_week DOUBLE, median_unit_price DOUBLE, "
        "seasonal_revenue_share DOUBLE, non_uk_revenue_share DOUBLE, return_rate DOUBLE, "
        "days_since_last_sale INT, is_history_censored BOOLEAN",
    )
    seg = spark.createDataFrame([("A", 0), ("B", 0), ("C", 1)], "stock_code STRING, segment_id INT")
    return label_all_products(feats, seg)


def test_label_all_products(labelled):
    got = {r["stock_code"]: (r[SEGMENT_LABEL_COL], r[SEGMENT_COL]) for r in labelled.collect()}
    assert got == {
        "A": ("cluster 0", 0),
        "B": ("cluster 0", 0),
        "C": ("cluster 1", 1),
        "D": (LOW_VOLUME, None),
    }
    assert labelled.where(f"{ELIGIBILITY_COL} = '{CLUSTERED}'").count() == 3


def test_product_segment_profile(labelled):
    feats = ["revenue", "units_per_week", "median_unit_price"]
    rows = {r[SEGMENT_LABEL_COL]: r for r in product_segment_profile(labelled, feats).collect()}
    c0 = rows["cluster 0"]
    assert c0["products"] == 2
    assert c0["product_share"] == pytest.approx(0.5)
    assert c0["revenue_share"] == pytest.approx(150 / 200)
    assert c0["revenue_per_product"] == pytest.approx(75.0)
    assert c0["mean_median_unit_price"] == pytest.approx(7.0)
    assert c0["pct_sold_last_30d"] == pytest.approx(0.5)  # A: 10 days, B: 200
    assert c0["pct_not_sold_180d"] == pytest.approx(0.5)
    assert c0["pct_q4_majority"] == pytest.approx(0.5)  # B 0.8
    assert c0["pct_international"] == pytest.approx(0.5)  # B 0.5
    assert c0["pct_price_10_plus"] == pytest.approx(0.5)  # B 12
    assert c0["pct_return_rate_5_plus"] == pytest.approx(0.5)  # B 0.1
    assert c0["pct_history_censored"] == pytest.approx(0.5)
    assert rows[LOW_VOLUME]["products"] == 1
    assert sum(r["revenue_share"] for r in rows.values()) == pytest.approx(1.0)


def test_top_products(labelled, spark):
    desc = spark.createDataFrame(
        [("A", "MUG"), ("B", "STAR"), ("C", "BOX")], "stock_code STRING, description STRING"
    )
    rows = top_products(labelled, desc, n=1).collect()
    got = {r[SEGMENT_LABEL_COL]: (r["stock_code"], r["description"]) for r in rows}
    assert got == {"cluster 0": ("A", "MUG"), "cluster 1": ("C", "BOX"), LOW_VOLUME: ("D", None)}
