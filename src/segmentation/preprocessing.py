"""Preprocessing customer features for clustering (Phase 3.2).

Steps, in order (each one is a pure function, so the same steps can be re-applied at scoring
time from saved parameters):

1. **Choose the clustering features** from a priority-ordered candidate list, dropping any
   candidate that is almost a duplicate (|Pearson r| > threshold on the transformed scale)
   of a higher-priority one that is already kept.
2. **log1p** the right-skewed features (skewness > 1 on the raw scale). Money and counts have
   long right tails (monetary: median 856, max 580,987); without the log, distance-based
   clustering is dominated by a handful of very large customers.
3. **Clip (winsorise)** every feature to its [1st, 99th] percentile *after* the log, so the
   few remaining extremes can't drag centroids. Customers are not removed: an extreme
   customer still gets a segment, just computed from the clipped value.
4. **Standardise** (mean 0, sd 1) so each feature has equal weight in the distance.

Steps 3-4 are *fitted* (bounds, means, sds come from the data). They are stored in a
`PreprocessParams` object that serialises to a dict, which is logged to MLflow with every
run and can be re-applied later (Phase 6 scoring) without refitting.

Why plain column expressions instead of Spark ML's `StandardScaler` in a Pipeline: the
parameters are visible JSON (easy to log, diff and test), nothing is added to the serverless
Spark Connect model cache, and clipping isn't a Spark ML stage anyway.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Candidates in PRIORITY order: when two are near-duplicates, the earlier one is kept.
# recency/monetary/frequency first (RFM, the language the business uses), then tenure
# (separates New customers from established ones), breadth, seasonality.
CANDIDATE_FEATURES: list[str] = [
    "recency_days",
    "monetary",
    "frequency",
    "tenure_days",
    "distinct_products",
    "seasonal_revenue_share",
    "n_purchase_days",  # near-copy of frequency: expected to be pruned (shows the rule works)
]

# Kept in the feature table and used to PROFILE segments, but not clustered on:
PROFILE_ONLY_FEATURES: dict[str, str] = {
    "avg_basket_value": "log(basket) = log(monetary) - log(frequency): no new direction",
    "mean_days_between_purchases": "derived from tenure/recency/days; NULL for 29% (one day)",
    "return_rate": "zero-inflated (most customers 0): would carve a 'returners' split",
    "returned_value": "mostly 0 and scales with monetary",
    "n_countries": "almost constant (p99 = 1)",
}

SKEW_THRESHOLD = 1.0
CORRELATION_THRESHOLD = 0.9
CLIP_QUANTILES = (0.01, 0.99)


def choose_log_features(
    skewness: Mapping[str, float], features: Sequence[str], threshold: float = SKEW_THRESHOLD
) -> list[str]:
    """Features (in the given order) whose raw skewness is above `threshold`."""
    return [f for f in features if skewness.get(f) is not None and skewness[f] > threshold]


def log_transform(df: DataFrame, columns: Sequence[str]) -> DataFrame:
    """Replace each column by log1p(column) (log(1 + x): defined at 0, ~log(x) for big x)."""
    for c in columns:
        df = df.withColumn(c, F.log1p(F.col(c).cast("double")))
    return df


def correlation_matrix(df: DataFrame, columns: Sequence[str]) -> dict[tuple[str, str], float]:
    """Pearson correlation for every pair, computed in ONE aggregation.

    Returns a symmetric dict {(a, b): r} including the diagonal (1.0).
    """
    pairs = [(a, b) for i, a in enumerate(columns) for b in columns[i + 1 :]]
    row = df.agg(
        *[
            F.corr(F.col(a).cast("double"), F.col(b).cast("double")).alias(f"{i}")
            for i, (a, b) in enumerate(pairs)
        ]
    ).first()
    out = {(c, c): 1.0 for c in columns}
    for i, (a, b) in enumerate(pairs):
        r = row[f"{i}"]
        out[(a, b)] = out[(b, a)] = float("nan") if r is None else float(r)
    return out


def prune_correlated(
    candidates: Sequence[str],
    corr: Mapping[tuple[str, str], float],
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[list[str], dict[str, tuple[str, float]]]:
    """Greedy pruning in priority order.

    Walk the candidates in order; keep one unless |r| > threshold with an already-kept
    feature. Returns (kept, dropped) where dropped maps feature -> (kept feature, r).
    """
    kept: list[str] = []
    dropped: dict[str, tuple[str, float]] = {}
    for c in candidates:
        clash = next(
            ((k, corr[(c, k)]) for k in kept if abs(corr[(c, k)]) > threshold),
            None,
        )
        if clash is None:
            kept.append(c)
        else:
            dropped[c] = clash
    return kept, dropped


@dataclass
class PreprocessParams:
    """Everything needed to re-apply the preprocessing (JSON-friendly)."""

    features: list[str]
    log_features: list[str]
    clip_quantiles: tuple[float, float]
    lower: dict[str, float] = field(default_factory=dict)
    upper: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["clip_quantiles"] = list(self.clip_quantiles)
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> PreprocessParams:
        return cls(
            features=list(d["features"]),
            log_features=list(d["log_features"]),
            clip_quantiles=tuple(d["clip_quantiles"]),
            lower=dict(d["lower"]),
            upper=dict(d["upper"]),
            mean=dict(d["mean"]),
            std=dict(d["std"]),
        )


def _clip(col, lo: float, hi: float):
    return F.least(F.greatest(col, F.lit(lo)), F.lit(hi))


def fit_preprocessor(
    df: DataFrame,
    features: Sequence[str],
    log_features: Sequence[str],
    clip_quantiles: tuple[float, float] = CLIP_QUANTILES,
    accuracy: int = 100_000,
) -> PreprocessParams:
    """Learn clip bounds (quantiles after log) and mean/sd (after clipping). Two small jobs."""
    unknown = set(log_features) - set(features)
    if unknown:
        raise ValueError(f"log_features not in features: {sorted(unknown)}")
    logged = log_transform(df.select(*features), log_features)
    q_lo, q_hi = clip_quantiles
    q = logged.agg(
        *[F.percentile_approx(F.col(c), [q_lo, q_hi], accuracy).alias(c) for c in features]
    ).first()
    lower = {c: float(q[c][0]) for c in features}
    upper = {c: float(q[c][1]) for c in features}

    clipped = logged.select(*[_clip(F.col(c), lower[c], upper[c]).alias(c) for c in features])
    m = clipped.agg(
        *[F.mean(c).alias(f"m_{c}") for c in features],
        *[F.stddev_samp(c).alias(f"s_{c}") for c in features],
    ).first()
    mean = {c: float(m[f"m_{c}"]) for c in features}
    std = {}
    for c in features:
        s = m[f"s_{c}"]
        # A constant column would divide by 0: keep it at 0 after centring instead.
        std[c] = 1.0 if s is None or s == 0 or math.isnan(s) else float(s)
    return PreprocessParams(
        features=list(features),
        log_features=list(log_features),
        clip_quantiles=(q_lo, q_hi),
        lower=lower,
        upper=upper,
        mean=mean,
        std=std,
    )


def scaled_name(feature: str) -> str:
    return f"z_{feature}"


def apply_preprocessor(
    df: DataFrame,
    params: PreprocessParams,
    keep: Sequence[str] = ("customer_id",),
    vector_col: str = "features",
) -> DataFrame:
    """keep columns + one standardised column `z_<feature>` per feature + a vector column."""
    logged = log_transform(df, params.log_features)
    z_cols = [
        (
            (
                _clip(F.col(c).cast("double"), params.lower[c], params.upper[c])
                - F.lit(params.mean[c])
            )
            / F.lit(params.std[c])
        ).alias(scaled_name(c))
        for c in params.features
    ]
    out = logged.select(*keep, *z_cols)
    assembler = VectorAssembler(
        inputCols=[scaled_name(c) for c in params.features], outputCol=vector_col
    )
    return assembler.transform(out)


def clip_counts(df: DataFrame, params: PreprocessParams) -> dict[str, tuple[int, int]]:
    """How many customers were clipped at the lower / upper bound, per feature."""
    logged = log_transform(df.select(*params.features), params.log_features)
    row = logged.agg(
        *[
            F.sum((F.col(c) < params.lower[c]).cast("int")).alias(f"lo_{c}")
            for c in params.features
        ],
        *[
            F.sum((F.col(c) > params.upper[c]).cast("int")).alias(f"hi_{c}")
            for c in params.features
        ],
    ).first()
    return {c: (int(row[f"lo_{c}"] or 0), int(row[f"hi_{c}"] or 0)) for c in params.features}
