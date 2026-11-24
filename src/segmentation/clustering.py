"""Fit and evaluate the three clustering algorithms over a range of k (Phase 3.3).

| Algorithm | How it assigns | Strengths | Weaknesses |
|---|---|---|---|
| KMeans | hard, nearest centroid | fast, simple, centroids = segment "prototypes" | assumes round, similar-size clusters; depends on the random start |
| Bisecting k-means | hard, top-down splits | deterministic hierarchy (k=4 refines k=3), less start-dependent | greedy: an early bad split is never undone |
| Gaussian mixture (GMM) | soft, probability per cluster | elliptical clusters of different sizes, gives assignment confidence | more parameters (a covariance per cluster), can collapse on near-duplicate points, slower |

Serverless notes (Spark Connect ML):
- A fitted model lives in a server-side cache with a per-session size limit, and a DataFrame
  returned by `model.transform` still references it. So each trial computes everything it
  needs while the model is alive, extracts the parameters as plain Python (centres, weights,
  means) and then deletes the model. Nothing returned here holds a model reference.
- `BisectingKMeans` fails on Free Edition (DBFS_DISABLED), so `bisecting_impl="kmeans_splits"`
  uses `segmentation.bisecting` (built on KMeans) instead.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from segmentation.bisecting import Split, assign, fit_bisecting
from segmentation.metrics import cluster_stats, silhouette, size_shares, wcss

ALGORITHMS: tuple[str, ...] = ("kmeans", "bisecting_kmeans", "gmm")
BISECTING_IMPLS = ("spark_ml", "kmeans_splits")
LABEL_COL = "cluster"


@dataclass
class TrialResult:
    algorithm: str
    k: int
    seed: int
    implementation: str
    metrics: dict = field(default_factory=dict)
    shares: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shares"] = {str(k): v for k, v in self.shares.items()}
        return d


def _evaluate(
    pred: DataFrame, k: int, z_cols: Sequence[str], features_col: str
) -> tuple[dict, dict]:
    stats = cluster_stats(pred, LABEL_COL, z_cols)
    shares = size_shares(stats, k)
    metrics = {
        "silhouette": silhouette(pred, features_col, LABEL_COL),
        "wcss": wcss(stats),
        "min_share": min(shares.values()),
        "max_share": max(shares.values()),
        "n_empty_clusters": sum(1 for v in shares.values() if v == 0.0),
    }
    return metrics, shares


def run_trial(
    algorithm: str,
    df: DataFrame,
    k: int,
    z_cols: Sequence[str],
    seed: int = 42,
    features_col: str = "features",
    bisecting_impl: str = "kmeans_splits",
    bisecting_splits: Sequence[Split] | None = None,
) -> TrialResult:
    """Fit one (algorithm, k), evaluate it, return plain results. Never raises: a failure
    is recorded in `error` so one bad fit doesn't stop a sweep."""
    from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
    from pyspark.ml.functions import vector_to_array

    impl = bisecting_impl if algorithm == "bisecting_kmeans" else "spark_ml"
    result = TrialResult(algorithm=algorithm, k=k, seed=seed, implementation=impl)
    model = None
    t0 = time.perf_counter()
    try:
        if algorithm == "kmeans":
            model = KMeans(
                k=k, seed=seed, maxIter=100, featuresCol=features_col, predictionCol=LABEL_COL
            ).fit(df)
            pred = model.transform(df)
            result.model_info = {"centers": [[float(x) for x in c] for c in model.clusterCenters()]}
        elif algorithm == "gmm":
            model = GaussianMixture(
                k=k,
                seed=seed,
                maxIter=200,
                featuresCol=features_col,
                predictionCol=LABEL_COL,
                probabilityCol="probability",
            ).fit(df)
            pred = model.transform(df)
            means = [
                [float(x) for x in r["mean"]] for r in model.gaussiansDF.select("mean").collect()
            ]
            result.model_info = {"weights": [float(w) for w in model.weights], "means": means}
        elif algorithm == "bisecting_kmeans" and impl == "spark_ml":
            model = BisectingKMeans(
                k=k, seed=seed, featuresCol=features_col, predictionCol=LABEL_COL
            ).fit(df)
            pred = model.transform(df)
            result.model_info = {"centers": [[float(x) for x in c] for c in model.clusterCenters()]}
        elif algorithm == "bisecting_kmeans":
            splits = bisecting_splits or fit_bisecting(df, k, z_cols, seed, features_col)
            pred = assign(df, splits, z_cols, k=k, label_col=LABEL_COL)
            result.model_info = {"splits": [s.to_dict() for s in list(splits)[: k - 1]]}
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")

        result.metrics, result.shares = _evaluate(pred, k, z_cols, features_col)
        if algorithm == "gmm":
            # Soft assignment confidence: mean of each customer's highest cluster probability.
            result.metrics["mean_max_probability"] = float(
                pred.agg(F.mean(F.array_max(vector_to_array("probability")))).first()[0]
            )
    except Exception as exc:  # recorded, the sweep continues
        result.error = f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        result.metrics["fit_eval_seconds"] = round(time.perf_counter() - t0, 3)
        if model is not None:
            del model  # release the server-side model cache entry
    return result


def run_sweep(
    df: DataFrame,
    z_cols: Sequence[str],
    ks: Sequence[int],
    algorithms: Sequence[str] = ALGORITHMS,
    seed: int = 42,
    features_col: str = "features",
    bisecting_impl: str = "kmeans_splits",
    on_result: Callable[[TrialResult], None] | None = None,
) -> list[TrialResult]:
    """Every (algorithm, k). With the KMeans-splits bisecting, the tree is fitted once up to
    max(ks) and truncated per k (identical to fitting each k separately, k-1 fits saved)."""
    if bisecting_impl not in BISECTING_IMPLS:
        raise ValueError(f"bisecting_impl must be one of {BISECTING_IMPLS}")
    splits = None
    if "bisecting_kmeans" in algorithms and bisecting_impl == "kmeans_splits":
        splits = fit_bisecting(df, max(ks), z_cols, seed, features_col)
    results = []
    for algorithm in algorithms:
        for k in ks:
            r = run_trial(algorithm, df, k, z_cols, seed, features_col, bisecting_impl, splits)
            results.append(r)
            if on_result is not None:
                on_result(r)
    return results
