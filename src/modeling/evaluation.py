"""Evaluation of propensity scores (Phase 5): metrics, gain/lift, calibration, plots.

Everything takes plain arrays: `y` (0/1 labels) and `p` (predicted probability of a repeat
purchase), so the same functions evaluate the logistic baseline, LightGBM, a simple rule
and (Phase 7) a model on newly labelled data.

Which numbers matter here, and why (interview point):
- **Accuracy is not used.** With a 32-45 % base rate, "predict nobody comes back" already
  scores 55-68 % accuracy, and accuracy depends on an arbitrary 0.5 threshold. The business
  doesn't threshold at 0.5: it contacts the top N customers its budget allows.
- **ROC-AUC**: probability that a random repeat buyer is scored above a random non-buyer.
  Threshold-free ranking quality, but insensitive to how good the very top of the list is.
- **PR-AUC (average precision)**: precision averaged over recall levels; its "no-skill"
  value is the base rate, so it is read next to it (`pr_auc_lift` = PR-AUC / base rate).
  It focuses on the positive class, which is what a campaign acts on. **Primary metric.**
- **Gain / lift**: "the top 20 % of scored customers contain X % of all repeat buyers";
  directly answers "how many buyers do we reach if we can only contact N customers".
- **Calibration (Brier score, reliability curve, ECE)**: do customers scored 0.7 come back
  70 % of the time? Needed when scores are used as probabilities (expected revenue =
  p x basket value), not just for ranking.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-15
DEFAULT_TOP_FRACTIONS: tuple[float, ...] = (0.1, 0.2, 0.3)


def _arrays(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y and p differ in length: {y.shape} vs {p.shape}")
    if len(y) == 0:
        raise ValueError("empty input")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must be 0/1")
    if np.isnan(p).any():
        raise ValueError("p contains NaN")
    return y, p


def _rank_desc(p: np.ndarray) -> np.ndarray:
    """Indices that sort scores high -> low; ties keep input order (stable, reproducible)."""
    return np.argsort(-p, kind="mergesort")


# ----------------------------------------------------------------------------- metrics


def binary_metrics(y, p) -> dict:
    """ROC-AUC, PR-AUC (average precision), Brier, log loss, base rate, counts."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y, p = _arrays(y, p)
    base = float(y.mean())
    both_classes = 0.0 < base < 1.0
    pc = np.clip(p, EPS, 1 - EPS)
    pr_auc = float(average_precision_score(y, p)) if both_classes else math.nan
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": base,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else math.nan,
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / base if both_classes else math.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "mean_score": float(p.mean()),
    }


def capture_at(y, p, top_fraction: float) -> float:
    """Share of all positives found in the top `top_fraction` of customers by score."""
    y, p = _arrays(y, p)
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    k = math.ceil(len(y) * top_fraction)
    total = y.sum()
    return float(y[_rank_desc(p)][:k].sum() / total) if total else math.nan


def gain_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Cumulative gain and lift by score band (band 1 = highest scores).

    Bands have (almost) equal numbers of customers, made by splitting the ranked list, so
    ties in the score can't produce empty or uneven bands (unlike quantile cuts).

    Columns: band, customers, positives, rate, lift (band rate / base rate),
    cum_customer_share, cum_positive_share (= gain), cum_lift, min_score, max_score.
    """
    y, p = _arrays(y, p)
    order = _rank_desc(p)
    ys, ps = y[order], p[order]
    base = y.mean()
    rows, seen_n, seen_pos = [], 0, 0.0
    for band, idx in enumerate(np.array_split(np.arange(len(ys)), n_bins), start=1):
        if len(idx) == 0:
            continue
        pos = ys[idx].sum()
        seen_n += len(idx)
        seen_pos += pos
        rate = pos / len(idx)
        rows.append(
            {
                "band": band,
                "customers": int(len(idx)),
                "positives": int(pos),
                "rate": float(rate),
                "lift": float(rate / base) if base else math.nan,
                "cum_customer_share": seen_n / len(ys),
                "cum_positive_share": float(seen_pos / y.sum()) if y.sum() else math.nan,
                "cum_lift": float((seen_pos / seen_n) / base) if base else math.nan,
                "min_score": float(ps[idx].min()),
                "max_score": float(ps[idx].max()),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: equal-count bins by score (low -> high), mean predicted vs observed."""
    y, p = _arrays(y, p)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins), start=1):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": b,
                "count": int(len(idx)),
                "mean_predicted": float(p[idx].mean()),
                "observed_rate": float(y[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(cal: pd.DataFrame) -> float:
    """Count-weighted mean |observed - predicted| over the calibration bins."""
    w = cal["count"] / cal["count"].sum()
    return float((w * (cal["observed_rate"] - cal["mean_predicted"]).abs()).sum())


def evaluate(
    y, p, top_fractions: Sequence[float] = DEFAULT_TOP_FRACTIONS, n_bins: int = 10
) -> dict:
    """Everything for one (labels, scores) pair: metrics + capture rates + tables."""
    metrics = binary_metrics(y, p)
    for f in top_fractions:
        metrics[f"capture_top_{round(f * 100)}pct"] = capture_at(y, p, f)
    cal = calibration_table(y, p, n_bins)
    metrics["ece"] = expected_calibration_error(cal)
    return {"metrics": metrics, "gain": gain_table(y, p, n_bins), "calibration": cal}


def prefixed(metrics: Mapping[str, float], prefix: str) -> dict:
    """{'roc_auc': .8} -> {'validation_roc_auc': .8}; drops NaN (MLflow rejects them)."""
    return {
        f"{prefix}_{k}": float(v)
        for k, v in metrics.items()
        if isinstance(v, int | float) and not math.isnan(float(v))
    }


def comparison_table(
    results: Mapping[str, Mapping[str, float]], keys: Sequence[str]
) -> pd.DataFrame:
    """Rows = models, columns = chosen metric keys (for side-by-side reporting)."""
    return pd.DataFrame(
        [{"model": name, **{k: r.get(k) for k in keys}} for name, r in results.items()]
    )


# ----------------------------------------------------------------------------- plots
# matplotlib is imported lazily so the metric functions work without it.


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


Scores = Mapping[str, tuple[np.ndarray, np.ndarray]]  # model name -> (y, p)


def plot_roc_pr(scores: Scores, title: str = ""):
    """ROC and precision-recall curves side by side, one line per model."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    base = None
    for name, (y, p) in scores.items():
        m = binary_metrics(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, label=f"{name} (AUC {m['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, label=f"{name} (AP {m['pr_auc']:.3f})")
        base = m["base_rate"]
    ax1.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="random")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    if base is not None:
        ax2.axhline(base, ls="--", color="grey", lw=1, label=f"base rate {base:.3f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall", ylim=(0, 1.02))
    for ax in (ax1, ax2):
        ax.legend(fontsize=8, loc="lower right" if ax is ax1 else "upper right")
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_calibration(scores: Scores, n_bins: int = 10, title: str = "Calibration"):
    """Reliability curve (mean predicted vs observed rate per equal-count bin)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in scores.items():
        cal = calibration_table(y, p, n_bins)
        ece = expected_calibration_error(cal)
        ax.plot(cal["mean_predicted"], cal["observed_rate"], "o-", label=f"{name} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed repeat rate", title=title)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gain(scores: Scores, n_bins: int = 10, title: str = "Cumulative gain"):
    """Cumulative share of repeat buyers captured vs share of customers contacted."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    base = None
    for name, (y, p) in scores.items():
        g = gain_table(y, p, n_bins)
        ax.plot([0, *g["cum_customer_share"]], [0, *g["cum_positive_share"]], "o-", label=name)
        base = float(np.mean(y))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="random")
    if base:
        ax.plot([0, base, 1], [0, 1, 1], ":", color="black", lw=1, label="perfect")
    ax.axvline(0.2, color="grey", lw=0.8, alpha=0.6)
    ax.set(
        xlabel="Share of customers contacted (by score)", ylabel="Share of repeat buyers captured"
    )
    ax.set(title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_lift(scores: Scores, n_bins: int = 10, title: str = "Lift by score decile"):
    """Bar chart of per-band lift (band repeat rate / base rate), one bar group per model."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(scores)
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        g = gain_table(*scores[name], n_bins)
        ax.bar(g["band"] + (i - (len(names) - 1) / 2) * width, g["lift"], width, label=name)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set(xlabel="Score band (1 = highest scores)", ylabel="Lift vs base rate", title=title)
    ax.set_xticks(range(1, n_bins + 1))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bars(values: pd.Series, title: str, xlabel: str):
    """Horizontal bar chart of a named series (e.g. coefficients, importances), largest |x| on top."""
    plt = _plt()
    v = values.reindex(values.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(v))))
    ax.barh(v.index, v.values, color=["#3b7dd8" if x >= 0 else "#d8573b" for x in v.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
