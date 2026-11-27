"""Comparing two models on the same customers (Phase 5.4, reused by the promotion rule).

A difference of +0.01 PR-AUC on one test snapshot could be noise. The **paired bootstrap**
answers "how sure are we that model B beats model A on this population?":
resample customers with replacement (the same resample for both models, hence *paired*),
recompute both metrics, and look at the distribution of the difference B - A. Pairing
removes the variation that both models share (an "easy" resample is easy for both), so
the interval is much narrower than comparing two separate confidence intervals.

Read-out: the 95 % interval of the difference and `p_b_better` = share of resamples where B
wins. If the interval contains 0, the data can't tell the models apart at this size.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from modeling.evaluation import _arrays

Metric = Callable[[np.ndarray, np.ndarray], float]


def metric_fn(name: str) -> Metric:
    """'pr_auc' | 'roc_auc' | 'brier' -> f(y, p)."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    fns = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score, "brier": brier_score_loss}
    if name not in fns:
        raise ValueError(f"unknown metric {name!r}; choose from {sorted(fns)}")
    return fns[name]


def paired_bootstrap(
    y,
    p_a,
    p_b,
    metric: str = "pr_auc",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Difference metric(B) - metric(A) with a percentile bootstrap interval.

    For 'brier' (lower is better) the difference is still B - A, so negative = B better;
    `p_b_better` accounts for the direction.
    """
    y, p_a = _arrays(y, p_a)
    _, p_b = _arrays(y, p_b)
    fn = metric_fn(metric)
    lower_is_better = metric == "brier"
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():  # a resample with one class has no AUC; skip it
            continue
        diffs.append(fn(yb, p_b[idx]) - fn(yb, p_a[idx]))
    diffs = np.asarray(diffs)
    better = (diffs < 0) if lower_is_better else (diffs > 0)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric,
        "a": float(fn(y, p_a)),
        "b": float(fn(y, p_b)),
        "diff": float(fn(y, p_b) - fn(y, p_a)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_b_better": float(better.mean()),
        "n_boot": int(len(diffs)),
        "n": int(n),
    }
