"""SHAP explanations for the LightGBM propensity model (Phase 5.4).

SHAP value of feature j for customer i = how much that feature moved this customer's score
away from the average score, in **log-odds** (the model's raw output). For every customer,
base value + sum of SHAP values = the model's raw score exactly ("additivity"), which is
what makes SHAP trustworthy compared with split-count importances.

- **Global importance** = mean |SHAP| per feature (how much a feature moves scores on
  average), not split gain (which favours features used in many splits).
- **Direction** = Spearman correlation between a feature's value and its SHAP value
  (+ = higher value -> higher propensity).
- **Dependence** = SHAP value against feature value (shape of the effect: thresholds,
  saturation), coloured by the feature it interacts with most.

TreeSHAP is computed twice and compared: by the `shap` library and by LightGBM itself
(`predict(pred_contrib=True)`, the same algorithm built in). Matching numbers = the
explanation pipeline is right; and the LightGBM path is the fallback if `shap` ever fails
to install on serverless.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def lgbm_contributions(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """TreeSHAP from LightGBM: (n x f contributions, base value), log-odds."""
    contrib = np.asarray(model.predict(X, pred_contrib=True))
    return contrib[:, :-1], float(contrib[0, -1])


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """TreeSHAP from the `shap` library: (n x f values, base value), log-odds."""
    import warnings

    import shap

    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():  # "output has changed to a list of ndarray": handled below
        warnings.simplefilter("ignore", UserWarning)
        values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, f, classes)
        values = values[:, :, 1]
    base = np.ravel(explainer.expected_value)
    return values, float(base[-1])


def additivity_error(model, X: pd.DataFrame, values: np.ndarray, base: float) -> float:
    """max |base + sum(SHAP) - raw score| over customers (should be ~1e-6 or smaller)."""
    raw = np.asarray(model.predict(X, raw_score=True))
    return float(np.max(np.abs(base + values.sum(axis=1) - raw)))


def importance(values: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """mean |SHAP| (and its share of the total) per feature, largest first."""
    m = np.abs(values).mean(axis=0)
    out = pd.DataFrame({"feature": list(names), "mean_abs_shap": m, "share": m / m.sum()})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def direction(values: np.ndarray, X: pd.DataFrame) -> pd.Series:
    """Spearman correlation between each feature's value and its SHAP value (NaN rows dropped)."""
    out = {}
    for j, c in enumerate(X.columns):
        x = X[c].to_numpy()
        ok = ~np.isnan(x)
        if ok.sum() < 3 or np.unique(x[ok]).size < 2:
            out[c] = np.nan
            continue
        out[c] = pd.Series(x[ok]).rank().corr(pd.Series(values[ok, j]).rank())
    return pd.Series(out, name="spearman_value_vs_shap")


def dependence_table(
    values: np.ndarray, X: pd.DataFrame, feature: str, n_bins: int = 8
) -> pd.DataFrame:
    """Mean SHAP per quantile bin of the feature (NaN gets its own row): the shape of the
    effect as numbers, readable without the plot."""
    j = list(X.columns).index(feature)
    x = X[feature]
    s = pd.Series(values[:, j], index=X.index)
    rows = []
    if x.isna().any():
        rows.append(
            {
                "bin": "missing",
                "x_min": np.nan,
                "x_max": np.nan,
                "n": int(x.isna().sum()),
                "mean_shap": float(s[x.isna()].mean()),
            }
        )
    known = x.dropna()
    bins = pd.qcut(known.rank(method="first"), q=min(n_bins, known.size), labels=False)
    for b in sorted(bins.unique()):
        idx = bins.index[bins == b]
        rows.append(
            {
                "bin": int(b) + 1,
                "x_min": float(known[idx].min()),
                "x_max": float(known[idx].max()),
                "n": int(len(idx)),
                "mean_shap": float(s[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def strongest_interaction(values: np.ndarray, X: pd.DataFrame, feature: str) -> str | None:
    """The feature that best explains the spread of `feature`'s SHAP values (shap's own
    heuristic if available, else the other feature most correlated with the SHAP values)."""
    j = list(X.columns).index(feature)
    try:
        import shap

        order = shap.utils.approximate_interactions(j, values, X)
        return X.columns[order[0]]
    except Exception:
        s = pd.Series(values[:, j])
        corr = {c: abs(s.corr(X[c].reset_index(drop=True))) for c in X.columns if c != feature}
        corr = {k: v for k, v in corr.items() if pd.notna(v)}
        return max(corr, key=corr.get) if corr else None


# ----------------------------------------------------------------------------- plots


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def plot_summary(values: np.ndarray, X: pd.DataFrame, max_display: int = 15):
    """SHAP beeswarm: one dot per customer and feature, colour = feature value."""
    import shap

    plt = _plt()
    plt.figure()
    shap.summary_plot(values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP summary (test snapshot): impact on log-odds of a repeat purchase")
    fig.tight_layout()
    return fig


def plot_dependence(values: np.ndarray, X: pd.DataFrame, feature: str, color_by: str | None = None):
    """SHAP value vs feature value (symlog x for skewed non-negative features)."""
    plt = _plt()
    j = list(X.columns).index(feature)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = X[feature].to_numpy()
    if color_by is not None:
        sc = ax.scatter(
            x, values[:, j], c=X[color_by].rank(pct=True), cmap="coolwarm", s=6, alpha=0.6
        )
        fig.colorbar(sc, ax=ax, label=f"{color_by} (percentile)")
    else:
        ax.scatter(x, values[:, j], s=6, alpha=0.6)
    finite = x[~np.isnan(x)]
    if finite.size and finite.min() >= 0 and finite.max() > 50 * max(np.median(finite), 1):
        ax.set_xscale("symlog", linthresh=1)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel=feature, ylabel="SHAP value (log-odds)", title=f"SHAP dependence: {feature}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
