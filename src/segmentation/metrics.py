"""Cluster quality metrics and k-selection rules (Phase 3.3).

All metrics are computed on the SAME preprocessed space for every algorithm, so KMeans,
bisecting k-means and GMM can be compared directly.

- **WCSS** (within-cluster sum of squares, "inertia"): sum of squared distances from each
  customer to its cluster's centroid. Always falls as k grows; the *elbow* is where adding a
  cluster stops buying much. Computed from sums in one aggregation:
  SSE = sum(||x||^2) - n * ||mean||^2 (no second pass, no join back to centroids).
- **Silhouette** (Spark `ClusteringEvaluator`): for each point, how much closer it is to its
  own cluster than to the nearest other one, in [-1, 1], higher = better separated. Spark's
  default uses *squared* Euclidean distance, so values are not identical to scikit-learn's
  (Euclidean) silhouette; they're still comparable across our runs.
- **Size shares**: a segment holding 0.5% of customers is rarely actionable; the k rule below
  requires a minimum share.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def cluster_stats(df: DataFrame, label_col: str, z_cols: Sequence[str]) -> dict[int, dict]:
    """Per cluster: size n, centroid (list, in z_cols order) and SSE. One aggregation."""
    sq_norm = sum((F.col(c) * F.col(c) for c in z_cols), F.lit(0.0))
    rows = (
        df.groupBy(label_col)
        .agg(
            F.count(F.lit(1)).alias("_n"),
            *[F.sum(c).alias(f"_s{i}") for i, c in enumerate(z_cols)],
            F.sum(sq_norm).alias("_sq"),
        )
        .collect()
    )
    out = {}
    for r in rows:
        n = r["_n"]
        centroid = [r[f"_s{i}"] / n for i in range(len(z_cols))]
        sse = r["_sq"] - n * sum(m * m for m in centroid)
        out[int(r[label_col])] = {"n": n, "centroid": centroid, "sse": max(sse, 0.0)}
    return out


def wcss(stats: Mapping[int, Mapping]) -> float:
    return float(sum(s["sse"] for s in stats.values()))


def size_shares(stats: Mapping[int, Mapping], k: int) -> dict[int, float]:
    """Share of customers per cluster id 0..k-1 (0.0 for an empty cluster)."""
    total = sum(s["n"] for s in stats.values())
    return {i: (stats[i]["n"] / total if i in stats else 0.0) for i in range(k)}


def silhouette(df: DataFrame, features_col: str, label_col: str) -> float | None:
    """Spark ClusteringEvaluator silhouette (squared Euclidean). None if < 2 clusters."""
    from pyspark.ml.evaluation import ClusteringEvaluator

    if df.select(label_col).distinct().limit(2).count() < 2:
        return None
    evaluator = ClusteringEvaluator(
        featuresCol=features_col, predictionCol=label_col, metricName="silhouette"
    )
    return float(evaluator.evaluate(df))


def elbow_k(ks: Sequence[int], values: Sequence[float]) -> int:
    """Elbow of a decreasing curve: the k farthest from the straight line joining the first
    and last points (after scaling both axes to [0, 1]); the 'kneedle' idea, no tuning."""
    if len(ks) < 3:
        return ks[0]
    k0, k1 = ks[0], ks[-1]
    v0, v1 = values[0], values[-1]
    span_v = (v0 - v1) or 1.0
    best_k, best_d = ks[0], -1.0
    for k, v in zip(ks, values, strict=True):
        x = (k - k0) / (k1 - k0)
        y = (v0 - v) / span_v  # 0 at the first point, 1 at the last
        d = y - x  # how far the curve has dropped ahead of the straight line
        if d > best_d:
            best_k, best_d = k, d
    return best_k


def suggest_k(
    results: Sequence[Mapping],
    algorithm: str,
    min_share: float = 0.03,
    k_min: int = 3,
) -> int | None:
    """Best silhouette for one algorithm among k >= k_min whose smallest segment holds at
    least `min_share` of customers. k=2 is excluded by default: 'high value vs the rest' is
    usually the best-separated split but too coarse to act on."""
    ok = [
        r
        for r in results
        if r["algorithm"] == algorithm
        and r.get("error") is None
        and r["k"] >= k_min
        and r["metrics"].get("silhouette") is not None
        and r["metrics"]["min_share"] >= min_share
    ]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["metrics"]["silhouette"], -r["k"]))["k"]
