"""LightGBM propensity model tuned with Optuna (Phase 5.3).

Why gradient-boosted trees after the logistic baseline (interview point):
- they learn **interactions and thresholds** without hand-made features (recency matters
  differently for a one-order customer than for a loyal one; "orders in the last 90 days
  >= 2" is a step, not a line);
- no scaling or log transforms needed (splits only use the order of values);
- **missing values handled natively** (a NULL interval gets its own branch direction);
- at this size (43k rows, 26 features) training takes seconds on one machine.
Costs: less transparent than coefficients (hence SHAP in 5.4) and easier to overfit
(hence regularisation in the search space and early stopping).

Tuning protocol:
- **Optuna TPE** sampler (Bayesian: proposes new trials near the good ones), fixed seed.
- Each trial trains on the training cutoffs with **early stopping on validation log loss**
  (smooth, so the stopping point is stable), and is scored by **validation PR-AUC** (the
  selection metric used for the baseline too).
- Validation therefore takes part in model selection for every model (C grid, trials,
  rounds); the **test cutoff is used once, at the end** (5.4), for the honest estimate.
- The final model is refitted with exactly `best_iteration` trees (no eval set inside the
  saved model). Training never looks at the eval set, so the refit is the same model.
- `deterministic=True` + a fixed thread count: the same parameters give the same model in
  any session (the Phase 3 KMeans lesson: record, don't trust refits blindly; here they are
  checked to match).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from modeling.evaluation import binary_metrics

SEED = 42
N_THREADS = 4  # fixed: LightGBM is only reproducible for a fixed number of threads
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100

# Fixed for every trial.
BASE_PARAMS: dict = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "subsample_freq": 1,  # bagging every iteration (subsample < 1 then takes effect)
    "random_state": SEED,
    "n_jobs": N_THREADS,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


def suggest_params(trial) -> dict:
    """Search space (Optuna define-by-run).

    - learning_rate (log): smaller = more trees, usually a bit better, slower
    - num_leaves / min_child_samples (log): tree complexity vs minimum evidence per leaf
      (min_child_samples up to 500 because the same customer appears in up to 13 snapshots)
    - subsample / colsample_bytree: row / feature bagging (variance reduction)
    - reg_lambda / reg_alpha (log): L2 / L1 penalties on leaf values
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def make_lgbm(params: Mapping, n_estimators: int = MAX_ROUNDS):
    import lightgbm as lgb

    return lgb.LGBMClassifier(n_estimators=n_estimators, **{**BASE_PARAMS, **params})


def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


@dataclass
class FitResult:
    params: dict
    best_iteration: int
    train: dict
    validation: dict
    fit_seconds: float
    model: object


def fit_with_early_stopping(
    params: Mapping,
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    max_rounds: int = MAX_ROUNDS,
    patience: int = EARLY_STOPPING_ROUNDS,
) -> FitResult:
    """Train with early stopping on validation log loss; metrics at the best iteration."""
    import lightgbm as lgb

    t0 = time.perf_counter()
    model = make_lgbm(params, max_rounds)
    model.fit(
        X_train,
        y_train,
        eval_X=(X_val,),  # LightGBM >= 4.7 (eval_set is deprecated there)
        eval_y=(y_val,),
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    best = int(model.best_iteration_ or max_rounds)
    return FitResult(
        params=dict(params),
        best_iteration=best,
        train=binary_metrics(y_train, positive_proba(model, X_train)),
        validation=binary_metrics(y_val, positive_proba(model, X_val)),
        fit_seconds=round(time.perf_counter() - t0, 2),
        model=model,
    )


def refit_final(params: Mapping, best_iteration: int, X_train: pd.DataFrame, y_train):
    """The model to save: exactly `best_iteration` trees, no eval set attached."""
    return make_lgbm(params, best_iteration).fit(X_train, y_train)


TrialCallback = Callable[[int, FitResult], None]


def tune(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    n_trials: int = 40,
    timeout: float | None = None,
    seed: int = SEED,
    on_trial: TrialCallback | None = None,
    max_rounds: int = MAX_ROUNDS,
    patience: int = EARLY_STOPPING_ROUNDS,
):
    """Optuna study maximising validation PR-AUC. `on_trial(number, FitResult)` is called
    after each trial (the notebook logs a nested MLflow run there). Models are not kept."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        r = fit_with_early_stopping(
            suggest_params(trial), X_train, y_train, X_val, y_val, max_rounds, patience
        )
        trial.set_user_attr("best_iteration", r.best_iteration)
        trial.set_user_attr("fit_seconds", r.fit_seconds)
        for k in ("roc_auc", "pr_auc", "brier", "log_loss"):
            trial.set_user_attr(f"validation_{k}", r.validation[k])
            trial.set_user_attr(f"train_{k}", r.train[k])
        if on_trial is not None:
            on_trial(trial.number, r)
        return r.validation["pr_auc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="lgbm_repeat_purchase",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


def trials_frame(study) -> pd.DataFrame:
    """One row per completed trial: number, value (validation PR-AUC), params, user attrs."""
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append({"trial": t.number, "value": t.value, **t.params, **t.user_attrs})
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)


def feature_importance(model, kind: str = "gain") -> pd.Series:
    """Total split gain (or split count) per feature, normalised to sum to 1."""
    booster = model.booster_
    imp = pd.Series(booster.feature_importance(importance_type=kind), index=booster.feature_name())
    return (imp / imp.sum()).sort_values(ascending=False)


def plot_trials(trials: pd.DataFrame, baseline_value: float | None = None):
    """Validation PR-AUC per trial + running best (an Optuna 'optimisation history')."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trials["trial"], trials["value"], "o", alpha=0.6, label="trial")
    ax.plot(trials["trial"], trials["value"].cummax(), "-", label="best so far")
    if baseline_value is not None:
        ax.axhline(baseline_value, ls="--", color="grey", label=f"logreg {baseline_value:.4f}")
    ax.set(xlabel="trial", ylabel="validation PR-AUC", title="Optuna optimisation history")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
