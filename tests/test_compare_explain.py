"""Tests for modeling.compare (paired bootstrap) and modeling.explain (SHAP), Phase 5.4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")

from modeling.compare import metric_fn, paired_bootstrap  # noqa: E402
from modeling.explain import (  # noqa: E402
    additivity_error,
    dependence_table,
    direction,
    importance,
    lgbm_contributions,
    strongest_interaction,
)
from modeling.lgbm import refit_final  # noqa: E402

PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "reg_lambda": 0.1,
    "reg_alpha": 0.01,
}


def _scores(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.4).astype(int)
    signal = y + rng.normal(0, 1.0, n)
    good = 1 / (1 + np.exp(-signal))
    worse = 1 / (1 + np.exp(-(y + rng.normal(0, 2.0, n))))
    return y, good, worse


# ----------------------------------------------------------------------------- bootstrap


def test_bootstrap_detects_a_clearly_better_model():
    y, good, worse = _scores()
    r = paired_bootstrap(y, worse, good, "pr_auc", n_boot=200)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_b_better"] > 0.95
    assert r["ci_low"] <= r["diff"] <= r["ci_high"]
    assert r["n_boot"] == 200 and r["n"] == 2000


def test_bootstrap_same_model_is_centred_on_zero():
    y, good, _ = _scores()
    r = paired_bootstrap(y, good, good, "roc_auc", n_boot=50)
    assert r["diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["p_b_better"] == 0  # never strictly better


def test_bootstrap_brier_direction_and_reproducibility():
    y, good, worse = _scores()
    r1 = paired_bootstrap(y, worse, good, "brier", n_boot=100, seed=1)
    r2 = paired_bootstrap(y, worse, good, "brier", n_boot=100, seed=1)
    assert r1 == r2
    assert r1["diff"] < 0 and r1["p_b_better"] > 0.95  # lower Brier = better


def test_metric_fn_rejects_unknown():
    with pytest.raises(ValueError, match="unknown metric"):
        metric_fn("accuracy")


# ----------------------------------------------------------------------------- SHAP


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    n = 1500
    X = pd.DataFrame(
        {
            "orders_last_90d": rng.poisson(1.0, n).astype(float),
            "recency_days": rng.exponential(60, n),
            "noise": rng.normal(size=n),
            "interval": np.where(rng.uniform(size=n) < 0.3, np.nan, rng.exponential(40, n)),
        }
    )
    logit = -0.5 + 0.8 * X["orders_last_90d"] - 0.02 * X["recency_days"]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    return refit_final(PARAMS, 80, X, y), X


def test_lgbm_contributions_are_additive(fitted):
    model, X = fitted
    values, base = lgbm_contributions(model, X)
    assert values.shape == X.shape
    assert additivity_error(model, X, values, base) < 1e-6


def test_shap_library_matches_lightgbm_treeshap(fitted):
    pytest.importorskip("shap")
    from modeling.explain import shap_values

    model, X = fitted
    v_shap, b_shap = shap_values(model, X)
    v_lgb, b_lgb = lgbm_contributions(model, X)
    assert np.max(np.abs(v_shap - v_lgb)) < 1e-6
    assert b_shap == pytest.approx(b_lgb, abs=1e-6)


def test_importance_direction_and_dependence(fitted):
    model, X = fitted
    values, _ = lgbm_contributions(model, X)
    imp = importance(values, X.columns)
    assert imp["share"].sum() == pytest.approx(1.0)
    assert imp["feature"].iloc[0] in {"orders_last_90d", "recency_days"}
    share = imp.set_index("feature")["share"]
    assert share["noise"] < min(share["orders_last_90d"], share["recency_days"])
    d = direction(values, X)
    assert d["orders_last_90d"] > 0.5 and d["recency_days"] < -0.5
    t = dependence_table(values, X, "interval", n_bins=4)
    assert t["bin"].iloc[0] == "missing" and t["n"].sum() == len(X)
    t2 = dependence_table(values, X, "orders_last_90d", n_bins=4)
    assert t2["mean_shap"].iloc[-1] > t2["mean_shap"].iloc[0]  # more orders -> higher score


def test_strongest_interaction_returns_another_feature(fitted):
    model, X = fitted
    values, _ = lgbm_contributions(model, X)
    other = strongest_interaction(values, X, "orders_last_90d")
    assert other in set(X.columns) - {"orders_last_90d"}
