"""Tests for segmentation.profiling (Phase 3.5)."""

from __future__ import annotations

from datetime import date

import pytest

from segmentation.preprocessing import fit_preprocessor
from segmentation.profiling import (
    SEGMENT_COL,
    SegmentModel,
    fit_segment_model,
    order_centers,
    overall_profile,
    segment_model_from_centers,
    segment_model_from_trial,
    segment_profile,
)

FEATURES = ["monetary", "frequency"]


@pytest.fixture(scope="module")
def customers(spark):
    """Three obvious groups: big spenders (ids 1-4), regulars (5-8), one-timers (9-12)."""
    rows = []
    groups = [(5000.0, 40), (500.0, 8), (20.0, 1)]
    for g, (m, f) in enumerate(groups):
        for j in range(4):
            cid = g * 4 + j + 1
            rows.append(
                (
                    cid,
                    m * (1 + 0.01 * j),
                    f + (j % 2 if f > 1 else 0),
                    10 * (g + 1),  # recency
                    100.0,  # avg basket (unused by the model)
                    5,
                    300,
                    f,
                    None if f == 1 else 20.0,
                    0.1 if g == 0 else 0.0,
                    50.0 if g == 0 else 0.0,
                    1 if g == 0 else 0,
                    0.5,
                    "United Kingdom" if cid % 4 else "France",
                    cid == 1,
                    date(2011, 12, 10),
                )
            )
    schema = (
        "customer_id INT, monetary DOUBLE, frequency INT, recency_days INT, avg_basket_value DOUBLE, "
        "distinct_products INT, tenure_days INT, n_purchase_days INT, "
        "mean_days_between_purchases DOUBLE, return_rate DOUBLE, returned_value DOUBLE, "
        "n_return_invoices INT, seasonal_revenue_share DOUBLE, preferred_country STRING, "
        "is_tenure_censored BOOLEAN, as_of_date DATE"
    )
    return spark.createDataFrame(rows, schema)


@pytest.fixture(scope="module")
def model(customers):
    from segmentation.preprocessing import apply_preprocessor

    params = fit_preprocessor(customers, FEATURES, FEATURES, clip_quantiles=(0.0, 1.0))
    prepared = apply_preprocessor(customers, params)
    return fit_segment_model(prepared, params, k=3, seed=1)


def test_order_centers_by_monetary_descending():
    centers = [[0.0, 1.0], [2.0, 0.0], [-1.0, 5.0]]
    ordered, mapping = order_centers(centers, ["monetary", "frequency"])
    assert ordered == [[2.0, 0.0], [0.0, 1.0], [-1.0, 5.0]]
    assert mapping == {1: 0, 0: 1, 2: 2}
    _, by_freq = order_centers(centers, ["monetary", "frequency"], order_by="frequency")
    assert by_freq == {2: 0, 0: 1, 1: 2}


def test_segment_ids_ordered_by_value(model, customers):
    scored = {r["customer_id"]: r[SEGMENT_COL] for r in model.score(customers).collect()}
    assert {scored[i] for i in (1, 2, 3, 4)} == {0}  # big spenders -> segment 0
    assert {scored[i] for i in (5, 6, 7, 8)} == {1}
    assert {scored[i] for i in (9, 10, 11, 12)} == {2}
    assert model.k == 3 and model.algorithm == "kmeans"
    m_idx = model.features.index("monetary")
    assert [c[m_idx] for c in model.centers] == sorted(
        (c[m_idx] for c in model.centers), reverse=True
    )


def test_model_round_trip_and_score_columns(model, customers):
    again = SegmentModel.from_dict(model.to_dict())
    assert again == model
    cols = again.score(customers).columns
    assert cols == ["customer_id", "z_monetary", "z_frequency", SEGMENT_COL]
    assert again.z_cols == ["z_monetary", "z_frequency"]


def test_segment_profile_values(model, customers):
    segmented = customers.join(
        model.score(customers).select("customer_id", SEGMENT_COL), "customer_id"
    )
    prof = {r[SEGMENT_COL]: r for r in segment_profile(segmented).collect()}
    assert sorted(prof) == [0, 1, 2]
    assert sum(p["customers"] for p in prof.values()) == 12
    assert sum(p["customer_share"] for p in prof.values()) == pytest.approx(1.0)
    assert sum(p["revenue_share"] for p in prof.values()) == pytest.approx(1.0)
    top = prof[0]
    assert top["revenue_share"] > 0.9
    assert top["pct_with_returns"] == 1.0
    assert top["pct_uk"] == 0.75  # one France customer per group
    assert top["pct_tenure_censored"] == 0.25  # customer 1
    assert prof[2]["pct_one_invoice"] == 1.0
    assert prof[2]["mean_mean_days_between_purchases"] is None  # all NULL for one-timers
    assert prof[0]["revenue_per_customer"] == pytest.approx(top["revenue"] / 4)


def test_overall_profile(customers):
    o = overall_profile(customers, ["frequency"])
    assert o["mean_frequency"] == pytest.approx(customers.selectExpr("avg(frequency)").first()[0])
    assert set(o) == {"mean_frequency", "median_frequency"}


def test_segment_model_from_centers_orders_by_value(customers):
    params = fit_preprocessor(customers, FEATURES, FEATURES)
    m = segment_model_from_centers([[-1.0, 0.0], [2.0, 1.0], [0.5, -1.0]], params, seed=7)
    assert m.centers == [[2.0, 1.0], [0.5, -1.0], [-1.0, 0.0]]
    assert (m.k, m.seed, m.features) == (3, 7, FEATURES)
    assert m.preprocess == params.to_dict()


def test_segment_model_from_trial_uses_recorded_centres(customers):
    params = fit_preprocessor(customers, FEATURES, FEATURES)
    sweep = [
        {"algorithm": "gmm", "k": 2, "error": None, "model_info": {"means": [[0, 0], [1, 1]]}},
        {"algorithm": "kmeans", "k": 2, "error": "boom", "model_info": {}},
        {
            "algorithm": "kmeans",
            "k": 3,
            "error": None,
            "model_info": {"centers": [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0]]},
        },
    ]
    m = segment_model_from_trial(sweep, 3, params, seed=42)
    assert m.centers == [[3.0, 3.0], [1.0, 1.0], [0.0, 0.0]]
    with pytest.raises(ValueError, match="k=2"):
        segment_model_from_trial(sweep, 2, params, seed=42)  # only a failed KMeans trial


def test_recovered_model_reproduces_the_fitted_assignment(model, customers):
    """Rebuilding from the recorded centres gives the same segments as the fitted model."""
    params = fit_preprocessor(customers, FEATURES, FEATURES, clip_quantiles=(0.0, 1.0))
    again = segment_model_from_centers(list(reversed(model.centers)), params, seed=model.seed)
    a = {r["customer_id"]: r[SEGMENT_COL] for r in model.score(customers).collect()}
    b = {r["customer_id"]: r[SEGMENT_COL] for r in again.score(customers).collect()}
    assert a == b
