"""Tests for modeling.lgbm (Phase 5.3): search space, early stopping, refit, tuning.

Synthetic data with an interaction a linear model can't express, small round limits so the
whole module runs in a few seconds. Skipped without LightGBM / Optuna.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
optuna = pytest.importorskip("optuna")

from modeling.evaluation import binary_metrics  # noqa: E402
from modeling.lgbm import (  # noqa: E402
    BASE_PARAMS,
    feature_importance,
    fit_with_early_stopping,
    positive_proba,
    refit_final,
    suggest_params,
    trials_frame,
    tune,
)

PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "reg_lambda": 0.1,
    "reg_alpha": 0.01,
}


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "recency_days": rng.exponential(60, n),
            "orders_last_90d": rng.poisson(1.0, n).astype(float),
            "noise": rng.normal(size=n),
            "interval": np.where(rng.uniform(size=n) < 0.3, np.nan, rng.exponential(40, n)),
        }
    )
    # interaction: recency only matters for customers with recent orders
    logit = (
        -0.5
        + 1.2 * (X["orders_last_90d"] >= 1) * (X["recency_days"] < 30)
        + 0.4 * X["orders_last_90d"]
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    return X, y


@pytest.fixture(scope="module")
def data():
    X, y = _data()
    return X[:2000], y[:2000], X[2000:].reset_index(drop=True), y[2000:]


def test_search_space_ranges():
    p = suggest_params(optuna.trial.FixedTrial(PARAMS))
    assert p == PARAMS
    assert not set(p) & set(BASE_PARAMS)  # tuned and fixed params never overlap


def test_early_stopping_and_metrics(data):
    Xt, yt, Xv, yv = data
    r = fit_with_early_stopping(PARAMS, Xt, yt, Xv, yv, max_rounds=400, patience=20)
    assert 1 <= r.best_iteration < 400
    assert r.validation["roc_auc"] > 0.6
    assert r.validation["n"] == len(yv)


def test_refit_reproduces_the_early_stopped_model(data):
    """Training ignores the eval set, so best_iteration trees refitted == the stopped model."""
    Xt, yt, Xv, yv = data
    r = fit_with_early_stopping(PARAMS, Xt, yt, Xv, yv, max_rounds=400, patience=20)
    final = refit_final(PARAMS, r.best_iteration, Xt, yt)
    assert final.n_estimators == r.best_iteration
    assert np.allclose(positive_proba(final, Xv), positive_proba(r.model, Xv))
    again = refit_final(PARAMS, r.best_iteration, Xt, yt)  # deterministic
    assert np.array_equal(positive_proba(final, Xv), positive_proba(again, Xv))


def test_handles_nan_and_pickles_without_project_code(data):
    Xt, yt, Xv, _ = data
    assert Xt["interval"].isna().any()
    model = refit_final(PARAMS, 50, Xt, yt)
    blob = pickle.dumps(model)
    assert b"modeling" not in blob
    assert np.allclose(positive_proba(pickle.loads(blob), Xv), positive_proba(model, Xv))


def test_beats_a_linear_model_on_an_interaction(data):
    from sklearn.linear_model import LogisticRegression

    Xt, yt, Xv, yv = data
    lgbm = refit_final(PARAMS, 100, Xt, yt)
    lin = LogisticRegression(max_iter=2000).fit(Xt.fillna(0), yt)
    auc_tree = binary_metrics(yv, positive_proba(lgbm, Xv))["roc_auc"]
    auc_lin = binary_metrics(yv, lin.predict_proba(Xv.fillna(0))[:, 1])["roc_auc"]
    assert auc_tree > auc_lin


def test_tune_is_reproducible_and_reports_trials(data):
    Xt, yt, Xv, yv = data
    seen = []
    kw = dict(n_trials=3, max_rounds=200, patience=20)
    s1 = tune(Xt, yt, Xv, yv, on_trial=lambda n, r: seen.append((n, r.best_iteration)), **kw)
    s2 = tune(Xt, yt, Xv, yv, **kw)
    assert [n for n, _ in seen] == [0, 1, 2]
    assert s1.best_params == s2.best_params and s1.best_value == s2.best_value
    t = trials_frame(s1)
    assert len(t) == 3 and {
        "value",
        "learning_rate",
        "best_iteration",
        "validation_roc_auc",
    } <= set(t)
    assert t["value"].max() == pytest.approx(s1.best_value)


def test_feature_importance_normalised(data):
    Xt, yt, _, _ = data
    imp = feature_importance(refit_final(PARAMS, 50, Xt, yt))
    assert imp.sum() == pytest.approx(1.0)
    assert imp.index[0] in {"orders_last_90d", "recency_days"}
