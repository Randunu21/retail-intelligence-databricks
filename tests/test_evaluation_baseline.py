"""Tests for modeling.evaluation, modeling.dataset and modeling.baseline (Phase 5.2).

Small arrays whose gain / calibration values can be worked out by hand. scikit-learn is
needed (metrics and the logistic pipeline); the module is skipped without it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from modeling.baseline import (  # noqa: E402
    LOG_FEATURES,
    best_by,
    coefficients,
    fit_logistic_grid,
    make_logistic_pipeline,
    positive_proba,
    recency_rule_scores,
)
from modeling.build_training_set import MODEL_FEATURES  # noqa: E402
from modeling.dataset import feature_health, prepare_features, split_frames  # noqa: E402
from modeling.evaluation import (  # noqa: E402
    binary_metrics,
    calibration_table,
    capture_at,
    comparison_table,
    evaluate,
    expected_calibration_error,
    gain_table,
    prefixed,
)

# 10 customers, 4 buyers; scores rank 3 buyers in the top 3.
Y = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])


# ----------------------------------------------------------------------------- metrics


def test_binary_metrics_known_values():
    m = binary_metrics(Y, P)
    assert m["n"] == 10 and m["positives"] == 4 and m["base_rate"] == 0.4
    # pairs (buyer, non-buyer) ranked correctly: 3 top buyers beat all 6; the 4th beats 4 of 6
    assert m["roc_auc"] == pytest.approx((3 * 6 + 4) / 24)
    # AP = mean precision at each buyer's rank: (1/1 + 2/2 + 3/3 + 4/6) / 4
    assert m["pr_auc"] == pytest.approx((1 + 1 + 1 + 4 / 6) / 4)
    assert m["pr_auc_lift"] == pytest.approx(m["pr_auc"] / 0.4)
    assert m["brier"] == pytest.approx(np.mean((P - Y) ** 2))


def test_perfect_and_constant_scores():
    perfect = binary_metrics(Y, Y.astype(float))
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
    flat = binary_metrics(Y, np.full(10, 0.4))
    assert flat["roc_auc"] == 0.5
    assert flat["pr_auc"] == pytest.approx(0.4)  # no-skill PR-AUC = base rate


def test_single_class_gives_nan_auc_not_error():
    m = binary_metrics(np.zeros(5), np.linspace(0, 1, 5))
    assert math.isnan(m["roc_auc"]) and math.isnan(m["pr_auc"])
    assert "validation_roc_auc" not in prefixed(m, "validation")  # NaN dropped for MLflow


def test_input_validation():
    with pytest.raises(ValueError, match="length"):
        binary_metrics([0, 1], [0.5])
    with pytest.raises(ValueError, match="0/1"):
        binary_metrics([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        binary_metrics([0, 1], [0.1, np.nan])


# ----------------------------------------------------------------------------- gain / lift


def test_capture_at_top_fractions():
    assert capture_at(Y, P, 0.2) == pytest.approx(2 / 4)  # top 2 customers: 2 buyers
    assert capture_at(Y, P, 0.3) == pytest.approx(3 / 4)
    assert capture_at(Y, P, 0.25) == pytest.approx(3 / 4)  # ceil(2.5) = 3 customers
    assert capture_at(Y, P, 1.0) == 1.0


def test_gain_table_bands():
    g = gain_table(Y, P, n_bins=5)  # 5 bands of 2 customers
    assert g["customers"].tolist() == [2, 2, 2, 2, 2]
    assert g["positives"].tolist() == [2, 1, 1, 0, 0]
    assert g["cum_positive_share"].tolist() == pytest.approx([0.5, 0.75, 1.0, 1.0, 1.0])
    assert g["lift"].iloc[0] == pytest.approx(1.0 / 0.4)  # band rate 100 % vs base 40 %
    assert g["cum_lift"].iloc[-1] == pytest.approx(1.0)
    assert g["max_score"].iloc[0] == 0.9 and g["min_score"].iloc[-1] == 0.05


def test_gain_table_ties_are_deterministic_and_bands_never_empty():
    y = np.array([1, 0, 1, 0, 0, 0])
    p = np.full(6, 0.3)
    g1, g2 = gain_table(y, p, n_bins=4), gain_table(y, p, n_bins=4)
    pd.testing.assert_frame_equal(g1, g2)
    assert (g1["customers"] > 0).all() and g1["customers"].sum() == 6


# ----------------------------------------------------------------------------- calibration


def test_calibration_table_and_ece():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.8])
    cal = calibration_table(y, p, n_bins=2)
    assert cal["mean_predicted"].tolist() == pytest.approx([0.2, 0.7])
    assert cal["observed_rate"].tolist() == [0.0, 1.0]
    assert expected_calibration_error(cal) == pytest.approx(0.5 * 0.2 + 0.5 * 0.3)


def test_evaluate_bundles_everything():
    r = evaluate(Y, P, top_fractions=(0.2,), n_bins=5)
    assert r["metrics"]["capture_top_20pct"] == pytest.approx(0.5)
    assert "ece" in r["metrics"] and len(r["gain"]) == 5 and len(r["calibration"]) == 5
    t = comparison_table({"a": r["metrics"]}, ["roc_auc", "capture_top_20pct"])
    assert t.columns.tolist() == ["model", "roc_auc", "capture_top_20pct"]


# ----------------------------------------------------------------------------- dataset


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.exponential(5, n) for c in MODEL_FEATURES}
    data["is_uk"] = rng.integers(0, 2, n)
    data["seasonal_revenue_share"] = rng.uniform(0, 1, n)
    interval = rng.exponential(30, n)
    interval[rng.uniform(size=n) < 0.3] = np.nan
    data["mean_days_between_purchases"] = interval
    # label driven by recency and recent orders, so a model can learn something
    logit = 1.5 - 0.4 * data["recency_days"] + 0.3 * data["orders_last_90d"]
    label = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(data)
    df.insert(0, "split", np.where(np.arange(n) < 150, "train", "validation"))
    df.insert(0, "cutoff_date", pd.Timestamp("2011-01-01"))
    df.insert(0, "customer_id", np.arange(n))
    df["label"] = label
    return df


def test_prepare_features_order_type_and_missing_columns():
    df = _frame(5)
    shuffled = df[list(reversed(df.columns))]
    X = prepare_features(shuffled)
    assert X.columns.tolist() == list(MODEL_FEATURES)
    assert (X.dtypes == "float64").all()
    with pytest.raises(KeyError, match="recency_days"):
        prepare_features(df.drop(columns="recency_days"))


def test_split_frames_and_health():
    splits = split_frames(_frame())
    assert set(splits) == {"train", "validation"}
    assert splits["train"].n == 150 and splits["validation"].n == 50
    assert splits["train"].keys.columns.tolist() == ["customer_id", "cutoff_date"]
    h = feature_health(splits["train"].X)
    assert h.loc["mean_days_between_purchases", "n_nan"] > 0
    assert h["n_negative"].sum() == 0


# ----------------------------------------------------------------------------- baselines


def test_recency_rule_ranks_recent_buyers_first():
    X = pd.DataFrame({"recency_days": [1.0, 10.0, 100.0]})
    s = recency_rule_scores(X)
    assert s[0] > s[1] > s[2] and s[0] == 0.5


def test_logistic_pipeline_handles_nan_and_learns():
    splits = split_frames(
        _frame(400, seed=1).assign(
            split=lambda d: np.where(d.customer_id < 300, "train", "validation")
        )
    )
    tr, va = splits["train"], splits["validation"]
    model = make_logistic_pipeline().fit(tr.X, tr.y)
    p = positive_proba(model, va.X)
    assert p.shape == (va.n,) and np.all((p > 0) & (p < 1))
    assert binary_metrics(va.y, p)["roc_auc"] > 0.7
    coefs = coefficients(model)
    assert "log__recency_days" in coefs.index
    assert "missing__missingindicator_mean_days_between_purchases" in coefs.index
    assert coefs["log__recency_days"] < 0  # more days since last order -> less likely


def test_grid_and_best_by():
    splits = split_frames(
        _frame(300, seed=2).assign(
            split=lambda d: np.where(d.customer_id < 200, "train", "validation")
        )
    )
    tr, va = splits["train"], splits["validation"]
    results = fit_logistic_grid(tr.X, tr.y, va.X, va.y, Cs=(0.01, 1.0))
    assert [r.C for r in results] == [0.01, 1.0]
    best = best_by(results)
    assert best.validation["pr_auc"] == max(r.validation["pr_auc"] for r in results)


def test_log_features_are_model_features():
    assert set(LOG_FEATURES) <= set(MODEL_FEATURES)


def test_pipeline_pickles_without_project_code():
    """The fitted model must load where `src/` isn't importable (MLflow serving, Phase 6)."""
    import pickle

    splits = split_frames(_frame(100))
    model = make_logistic_pipeline().fit(splits["train"].X, splits["train"].y)
    blob = pickle.dumps(model)
    assert b"modeling" not in blob  # no reference to our own modules inside the pickle
    assert np.allclose(
        positive_proba(pickle.loads(blob), splits["train"].X),
        positive_proba(model, splits["train"].X),
    )
