"""Tests for segmentation.preprocessing (Phase 3.2)."""

from __future__ import annotations

import math

import pytest

from segmentation.preprocessing import (
    CANDIDATE_FEATURES,
    PROFILE_ONLY_FEATURES,
    PreprocessParams,
    apply_preprocessor,
    choose_log_features,
    clip_counts,
    correlation_matrix,
    fit_preprocessor,
    log_transform,
    prune_correlated,
    scaled_name,
)


@pytest.fixture
def df(spark):
    # x is skewed (one huge value), y = 2x exactly, w moves against x, k constant
    rows = [(i, float(i), 2.0 * i, float(10 - i), 5.0) for i in range(1, 10)] + [
        (10, 1000.0, 2000.0, 0.0, 5.0)
    ]
    return spark.createDataFrame(rows, "customer_id INT, x DOUBLE, y DOUBLE, w DOUBLE, k DOUBLE")


def test_choose_log_features_uses_threshold_and_keeps_order():
    skew = {"a": 0.5, "b": 12.0, "c": 1.0, "d": 3.0, "e": None}
    assert choose_log_features(skew, ["d", "a", "b", "c", "e"]) == ["d", "b"]
    assert choose_log_features(skew, ["d", "a", "b"], threshold=5) == ["b"]


def test_log_transform_is_log1p(df):
    out = log_transform(df, ["x"]).where("customer_id = 1").first()
    assert out["x"] == pytest.approx(math.log(2.0))
    assert out["y"] == 2.0  # other columns untouched


def test_correlation_matrix_symmetric_with_known_values(df):
    corr = correlation_matrix(df, ["x", "y", "w"])
    assert corr[("x", "x")] == 1.0
    assert corr[("x", "y")] == pytest.approx(1.0)
    assert corr[("y", "x")] == corr[("x", "y")]
    assert corr[("x", "w")] < 0


def test_prune_correlated_keeps_higher_priority():
    corr = {
        ("a", "b"): 0.95,
        ("b", "a"): 0.95,
        ("a", "c"): 0.2,
        ("c", "a"): 0.2,
        ("b", "c"): -0.93,
        ("c", "b"): -0.93,
    }
    kept, dropped = prune_correlated(["a", "b", "c"], corr, threshold=0.9)
    # b clashes with a (dropped); c only clashes with b, which was not kept -> c kept
    assert kept == ["a", "c"]
    assert dropped == {"b": ("a", 0.95)}
    kept2, _ = prune_correlated(["b", "a", "c"], corr, threshold=0.9)
    assert kept2 == ["b"]  # priority order decides who survives


def test_candidates_and_profile_only_do_not_overlap():
    assert not set(CANDIDATE_FEATURES) & set(PROFILE_ONLY_FEATURES)
    assert CANDIDATE_FEATURES[:3] == ["recency_days", "monetary", "frequency"]


def test_fit_preprocessor_rejects_log_feature_not_in_features(df):
    with pytest.raises(ValueError, match="not in features"):
        fit_preprocessor(df, ["x"], ["y"])


def test_fit_and_apply_standardise_after_clipping(df):
    params = fit_preprocessor(df, ["x", "w", "k"], ["x"], clip_quantiles=(0.0, 0.9))
    # upper bound for x is the 90th percentile of log1p(x) = log1p(9), so 1000 is clipped
    assert params.upper["x"] == pytest.approx(math.log1p(9.0))
    assert params.std["k"] == 1.0  # constant column: no division by zero

    out = apply_preprocessor(df, params)
    assert out.columns == ["customer_id", "z_x", "z_w", "z_k", "features"]
    stats = out.selectExpr(
        "avg(z_x) AS m", "stddev_samp(z_x) AS s", "max(z_x) AS mx", "max(abs(z_k)) AS k"
    ).first()
    assert stats["m"] == pytest.approx(0.0, abs=1e-9)
    assert stats["s"] == pytest.approx(1.0)
    assert stats["k"] == 0.0
    # the outlier sits exactly at the clip bound, equal to customer 9
    z = {r["customer_id"]: r["z_x"] for r in out.collect()}
    assert z[10] == pytest.approx(z[9])
    assert len(out.first()["features"]) == 3


def test_apply_on_new_data_uses_stored_params(df, spark):
    params = fit_preprocessor(df, ["x"], [], clip_quantiles=(0.0, 1.0))
    new = spark.createDataFrame([(99, 5000.0)], "customer_id INT, x DOUBLE")
    z = apply_preprocessor(new, params).first()["z_x"]
    # clipped to the TRAINING max (1000), not refitted on the new row
    assert z == pytest.approx((1000.0 - params.mean["x"]) / params.std["x"])


def test_params_round_trip_dict(df):
    params = fit_preprocessor(df, ["x", "w"], ["x"])
    again = PreprocessParams.from_dict(params.to_dict())
    assert again == params
    assert isinstance(params.to_dict()["clip_quantiles"], list)


def test_clip_counts(df):
    params = fit_preprocessor(df, ["x"], ["x"], clip_quantiles=(0.0, 0.9))
    assert clip_counts(df, params) == {"x": (0, 1)}


def test_scaled_name():
    assert scaled_name("monetary") == "z_monetary"
