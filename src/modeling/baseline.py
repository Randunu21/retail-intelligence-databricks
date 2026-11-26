"""Baselines for the propensity model (Phase 5): a rule and a logistic regression.

Two baselines answer two different questions:
1. **Recency rule** (score = 1 / (1 + recency_days)): "how good is the obvious thing a
   marketer would do without ML, i.e. contact the most recent buyers first?"
2. **Logistic regression**: "how much do a linear model and all features add?" It's also the
   model LightGBM must beat, and the first champion in the registry.

The logistic pipeline is pure scikit-learn (no custom classes), so the pickled model loads
anywhere MLflow installs scikit-learn: no need to ship `src/` with the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modeling.build_training_set import MODEL_FEATURES
from modeling.evaluation import binary_metrics

# Right-skewed counts and money amounts (Phase 3: monetary skew 25, frequency 12, ...):
# log1p turns multiplicative differences into additive ones, which suits a linear model.
LOG_FEATURES: tuple[str, ...] = (
    "frequency",
    "monetary",
    "avg_basket_value",
    "distinct_products",
    "returned_value",
    "n_return_invoices",
    "n_purchase_days",
    "mean_days_between_purchases",
    "recency_days",
    *[c for c in MODEL_FEATURES if c.startswith(("orders_last_", "revenue_last_"))],
)
# Features with legitimate NULLs (one purchase day -> no interval): get a missing indicator.
NULLABLE_FEATURES: tuple[str, ...] = ("mean_days_between_purchases",)

DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


def recency_rule_scores(X: pd.DataFrame) -> np.ndarray:
    """Most recent buyers first. A score in (0, 1], monotone in recency (not a probability)."""
    return 1.0 / (1.0 + X["recency_days"].to_numpy(dtype=float))


def make_logistic_pipeline(
    features: Sequence[str] = MODEL_FEATURES,
    C: float = 1.0,
    log_features: Sequence[str] = LOG_FEATURES,
    nullable: Sequence[str] = NULLABLE_FEATURES,
    class_weight: str | None = None,
    max_iter: int = 5000,
):
    """impute (median) -> log1p (skewed columns) -> standardise -> L2 logistic regression.

    Missing indicators are added for `nullable` columns, so "no interval yet" is information
    the model can use instead of being hidden by the median.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    features = list(features)
    log_cols = [c for c in features if c in set(log_features)]
    lin_cols = [c for c in features if c not in set(log_features)]
    null_cols = [c for c in features if c in set(nullable)]
    transformers = [
        (
            "log",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("scale", StandardScaler()),
                ]
            ),
            log_cols,
        ),
        (
            "linear",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            lin_cols,
        ),
    ]
    if null_cols:
        transformers.append(("missing", MissingIndicator(features="all"), null_cols))
    return Pipeline(
        [
            ("prep", ColumnTransformer(transformers, verbose_feature_names_out=True)),
            (
                "model",
                LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight),
            ),
        ]
    )


def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


@dataclass
class GridResult:
    C: float
    train: dict
    validation: dict
    model: object = field(repr=False)


def fit_logistic_grid(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    Cs: Sequence[float] = DEFAULT_C_GRID,
    features: Sequence[str] = MODEL_FEATURES,
    class_weight: str | None = None,
) -> list[GridResult]:
    """Fit one pipeline per C on train, score train and validation (metrics only)."""
    results = []
    for C in Cs:
        model = make_logistic_pipeline(features, C=C, class_weight=class_weight)
        model.fit(X_train, y_train)
        results.append(
            GridResult(
                C=C,
                train=binary_metrics(y_train, positive_proba(model, X_train)),
                validation=binary_metrics(y_val, positive_proba(model, X_val)),
                model=model,
            )
        )
    return results


def best_by(results: Sequence[GridResult], metric: str = "pr_auc") -> GridResult:
    """Highest validation `metric`; ties -> the smaller C (stronger regularisation)."""
    return max(results, key=lambda r: (r.validation[metric], -r.C))


def coefficients(pipeline) -> pd.Series:
    """Standardised coefficients by transformed feature name (log__monetary, ...)."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_.ravel()
    return pd.Series(coefs, index=names, name="coefficient")
