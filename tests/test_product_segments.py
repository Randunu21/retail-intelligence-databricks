"""Tests for segmentation.product_segments (Phase 4.2), plus a check that the Phase 3
preprocessing / clustering / stability code works unchanged with `stock_code` as the id."""

from __future__ import annotations

import pytest

from segmentation.clustering import run_trial
from segmentation.preprocessing import apply_preprocessor, fit_preprocessor, scaled_name
from segmentation.product_segments import (
    CLUSTERED,
    ELIGIBILITY_COL,
    LOW_VOLUME,
    NEW_PRODUCT,
    PRODUCT_CANDIDATE_FEATURES,
    PRODUCT_PROFILE_ONLY_FEATURES,
    RULE_SEGMENTS,
    add_eligibility,
    eligibility_summary,
    stability_candidates,
)
from segmentation.stability import stability


@pytest.fixture
def products(spark):
    return spark.createDataFrame(
        [
            ("A", 100, 25, 60.0),  # clustered
            ("B", 19, 25, 10.0),  # low volume (just under 20)
            ("C", 20, 25, 20.0),  # clustered (exactly 20)
            ("D", 500, 2, 10.0),  # new: checked before volume
            ("E", 3, 1, 0.0),  # new as well
        ],
        "stock_code STRING, n_invoices INT, months_on_sale INT, revenue DOUBLE",
    )


def test_eligibility_rules(products):
    got = {r["stock_code"]: r[ELIGIBILITY_COL] for r in add_eligibility(products).collect()}
    assert got == {
        "A": CLUSTERED,
        "B": LOW_VOLUME,
        "C": CLUSTERED,
        "D": NEW_PRODUCT,
        "E": NEW_PRODUCT,
    }


def test_eligibility_thresholds_are_parameters(products):
    got = {
        r["stock_code"]: r[ELIGIBILITY_COL]
        for r in add_eligibility(products, min_invoices=10, min_months=1).collect()
    }
    assert got["B"] == CLUSTERED and got["D"] == CLUSTERED and got["E"] == LOW_VOLUME


def test_eligibility_summary(products):
    rows = eligibility_summary(add_eligibility(products))
    assert [r["eligibility"] for r in rows] == [CLUSTERED, LOW_VOLUME, NEW_PRODUCT]
    clustered = rows[0]
    assert clustered["products"] == 2
    assert clustered["product_share"] == pytest.approx(2 / 5)
    assert clustered["revenue_share"] == pytest.approx(80 / 100)
    assert sum(r["products"] for r in rows) == 5


def test_feature_lists_dont_overlap():
    assert not set(PRODUCT_CANDIDATE_FEATURES) & set(PRODUCT_PROFILE_ONLY_FEATURES)
    assert len(set(PRODUCT_CANDIDATE_FEATURES)) == len(PRODUCT_CANDIDATE_FEATURES)
    assert set(RULE_SEGMENTS) == {NEW_PRODUCT, LOW_VOLUME}


def _trial(algorithm, k, sil, min_share=0.1, error=None):
    return {
        "algorithm": algorithm,
        "k": k,
        "error": error,
        "metrics": {"silhouette": sil, "min_share": min_share},
    }


def test_stability_candidates_rule():
    sweep = [
        _trial("kmeans", 2, 0.90),  # k < 3: never a candidate
        _trial("kmeans", 3, 0.50),
        _trial("kmeans", 4, 0.55),
        _trial("kmeans", 5, 0.60, min_share=0.01),  # segment too small
        _trial("kmeans", 6, 0.45),
        _trial("kmeans", 7, 0.40, error="boom"),
        _trial("gmm", 3, 0.20),
        _trial("gmm", 4, 0.30),
    ]
    got = stability_candidates(sweep, {"kmeans": 2, "gmm": 1})
    assert got == [("kmeans", 4), ("kmeans", 3), ("gmm", 4)]


def test_phase3_machinery_works_with_stock_code(spark):
    """Two obvious product groups: cheap fast sellers vs expensive slow ones."""
    rows = []
    for i in range(6):
        rows.append((f"F{i}", 100.0 + i, 1.0 + 0.01 * i))
        rows.append((f"S{i}", 1.0 + 0.01 * i, 50.0 + i))
    df = spark.createDataFrame(
        rows, "stock_code STRING, units_per_week DOUBLE, median_unit_price DOUBLE"
    )
    feats = ["units_per_week", "median_unit_price"]
    params = fit_preprocessor(df, feats, feats, (0.0, 1.0))
    prepared = apply_preprocessor(df, params, keep=["stock_code"])
    z = [scaled_name(f) for f in feats]

    trial = run_trial("kmeans", prepared, 2, z, seed=1)
    assert trial.error is None
    assert trial.shares == pytest.approx({0: 0.5, 1: 0.5})

    res = stability(
        "kmeans", prepared, 2, z, seeds=(1, 2), bootstrap_seeds=(7,), id_col="stock_code"
    )
    assert res.error is None
    assert res.seed_ari == [pytest.approx(1.0)]
    assert res.bootstrap_ari == [pytest.approx(1.0)]
