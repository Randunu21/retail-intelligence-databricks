"""Stability of a segmentation (Phase 3.4).

A segmentation is only usable for campaigns if it doesn't change when nothing important
changes. Two checks, both measured with the **adjusted Rand index (ARI)**:

1. **Seed stability**: refit on the same data with different random seeds; compare every pair
   of labelings. Catches dependence on KMeans' random start.
2. **Bootstrap stability**: refit on bootstrap samples (n customers drawn with replacement),
   then label ALL customers with each refitted model and compare with the reference model
   (fitted on the full data with the reference seed). Catches segments that depend on a
   handful of specific customers.

ARI compares two labelings of the same customers, ignoring what the clusters are called
(cluster "2" in one run may be cluster "0" in another). 1 = identical partition, ~0 = no
better than random, can be slightly negative. Rule of thumb used here: mean ARI >= 0.8 is
stable.

Labels are collected to the driver (5,852 small rows) and ARI is computed in plain Python:
cheap, exact, and no dependency on scikit-learn.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import comb
from statistics import mean

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from segmentation.bisecting import assign, fit_bisecting

LABEL_COL = "cluster"


def adjusted_rand_index(a: Sequence, b: Sequence) -> float:
    """ARI of two labelings of the same items (same order). Hubert & Arabie (1985)."""
    if len(a) != len(b):
        raise ValueError("labelings must have the same length")
    n = len(a)
    if n < 2:
        return 1.0
    pairs = sum(comb(c, 2) for c in Counter(zip(a, b, strict=True)).values())
    sum_a = sum(comb(c, 2) for c in Counter(a).values())
    sum_b = sum(comb(c, 2) for c in Counter(b).values())
    expected = sum_a * sum_b / comb(n, 2)
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:  # both labelings trivial (all one cluster / all singletons)
        return 1.0
    return (pairs - expected) / (max_index - expected)


def assign_nearest(
    df: DataFrame, centers: Sequence[Sequence[float]], z_cols: Sequence[str], label_col=LABEL_COL
) -> DataFrame:
    """KMeans assignment without the model: index of the nearest centre (ties -> lowest index,
    as Spark KMeans does)."""
    dists = []
    for center in centers:
        d = None
        for c, v in zip(z_cols, center, strict=True):
            term = (F.col(c) - F.lit(float(v))) ** 2
            d = term if d is None else d + term
        dists.append(d)
    names = [f"_d{i}" for i in range(len(centers))]
    out = df.select("*", *[d.alias(n) for d, n in zip(dists, names, strict=True)])
    nearest = F.least(*[F.col(n) for n in names]) if len(names) > 1 else F.col(names[0])
    label = F.lit(len(names) - 1)
    for i in reversed(range(len(names) - 1)):
        label = F.when(F.col(names[i]) == nearest, F.lit(i)).otherwise(label)
    return out.withColumn(label_col, label.cast("int")).drop(*names)


def _collect_labels(df: DataFrame, id_col: str) -> dict:
    return {r[0]: r[1] for r in df.select(id_col, LABEL_COL).collect()}


def fit_and_label(
    algorithm: str,
    fit_df: DataFrame,
    score_df: DataFrame,
    k: int,
    seed: int,
    z_cols: Sequence[str],
    features_col: str = "features",
    id_col: str = "customer_id",
) -> dict:
    """Fit on `fit_df`, label every row of `score_df`; returns {id: label}. Models are
    deleted before returning (serverless model cache)."""
    from pyspark.ml.clustering import GaussianMixture, KMeans

    if algorithm == "kmeans":
        model = KMeans(k=k, seed=seed, maxIter=100, featuresCol=features_col).fit(fit_df)
        centers = [[float(x) for x in c] for c in model.clusterCenters()]
        del model
        return _collect_labels(assign_nearest(score_df, centers, z_cols), id_col)
    if algorithm == "bisecting_kmeans":
        splits = fit_bisecting(fit_df, k, z_cols, seed, features_col)
        return _collect_labels(assign(score_df, splits, z_cols, k=k, label_col=LABEL_COL), id_col)
    if algorithm == "gmm":
        model = GaussianMixture(
            k=k, seed=seed, maxIter=200, featuresCol=features_col, predictionCol=LABEL_COL
        ).fit(fit_df)
        try:
            return _collect_labels(model.transform(score_df), id_col)
        finally:
            del model
    raise ValueError(f"unknown algorithm {algorithm!r}")


def ari_between(x: Mapping, y: Mapping) -> float:
    ids = sorted(x)
    if set(ids) != set(y):
        raise ValueError("labelings cover different customers")
    return adjusted_rand_index([x[i] for i in ids], [y[i] for i in ids])


@dataclass
class StabilityResult:
    algorithm: str
    k: int
    reference_seed: int
    seeds: list[int]
    bootstrap_seeds: list[int]
    seed_ari: list[float] = field(default_factory=list)  # every pair of seeds
    bootstrap_ari: list[float] = field(default_factory=list)  # each bootstrap vs reference
    error: str | None = None

    def summary(self) -> dict:
        def stats(xs):
            return (round(mean(xs), 4), round(min(xs), 4)) if xs else (None, None)

        s_mean, s_min = stats(self.seed_ari)
        b_mean, b_min = stats(self.bootstrap_ari)
        return {
            "seed_ari_mean": s_mean,
            "seed_ari_min": s_min,
            "bootstrap_ari_mean": b_mean,
            "bootstrap_ari_min": b_min,
        }

    def to_dict(self) -> dict:
        return {**asdict(self), **self.summary()}


def stability(
    algorithm: str,
    df: DataFrame,
    k: int,
    z_cols: Sequence[str],
    seeds: Sequence[int] = (42, 1, 2, 3, 4),
    bootstrap_seeds: Sequence[int] = (101, 102, 103, 104, 105),
    features_col: str = "features",
    id_col: str = "customer_id",
) -> StabilityResult:
    """Seed + bootstrap stability for one (algorithm, k). The first seed is the reference."""
    result = StabilityResult(algorithm, k, seeds[0], list(seeds), list(bootstrap_seeds))
    try:
        by_seed = [
            fit_and_label(algorithm, df, df, k, s, z_cols, features_col, id_col) for s in seeds
        ]
        result.seed_ari = [ari_between(a, b) for a, b in combinations(by_seed, 2)]
        reference = by_seed[0]
        for b in bootstrap_seeds:
            sample = df.sample(withReplacement=True, fraction=1.0, seed=b)
            labels = fit_and_label(algorithm, sample, df, k, seeds[0], z_cols, features_col, id_col)
            result.bootstrap_ari.append(ari_between(reference, labels))
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {str(exc)[:500]}"
    return result


def choose_final(
    sweep_rows: Sequence[Mapping],
    stability_rows: Sequence[Mapping],
    min_share: float = 0.03,
    min_ari: float = 0.8,
    silhouette_tolerance: float = 0.01,
    min_bootstrap_ari: float | None = None,
) -> dict | None:
    """Final (algorithm, k) rule:

    1. eligible = has a stability result, no error, smallest segment >= min_share, mean seed
       ARI >= min_ari and mean bootstrap ARI >= min_bootstrap_ari (default: = min_ari, the
       Phase 3 rule; Phase 4 products use a separate, documented bootstrap threshold);
    2. best = highest silhouette among eligible;
    3. among eligible within `silhouette_tolerance` of best (a practical tie), take the
       LARGEST k: more segments = more targeted actions, at no real cost in separation.
    """
    boot_min = min_ari if min_bootstrap_ari is None else min_bootstrap_ari
    sweep = {(r["algorithm"], r["k"]): r for r in sweep_rows if not r.get("error")}
    eligible = []
    for s in stability_rows:
        key = (s["algorithm"], s["k"])
        if s.get("error") or key not in sweep:
            continue
        if sweep[key]["min_share"] < min_share:
            continue
        if (s["seed_ari_mean"] or 0) < min_ari or (s["bootstrap_ari_mean"] or 0) < boot_min:
            continue
        eligible.append({**s, "silhouette": sweep[key]["silhouette"]})
    if not eligible:
        return None
    best = max(r["silhouette"] for r in eligible)
    ties = [r for r in eligible if r["silhouette"] >= best - silhouette_tolerance]
    pick = max(ties, key=lambda r: (r["k"], r["silhouette"]))
    return {
        "algorithm": pick["algorithm"],
        "k": pick["k"],
        "silhouette": pick["silhouette"],
        "best_silhouette": best,
        "seed_ari_mean": pick["seed_ari_mean"],
        "bootstrap_ari_mean": pick["bootstrap_ari_mean"],
        "eligible": sorted((r["algorithm"], r["k"]) for r in eligible),
        "min_seed_ari": min_ari,
        "min_bootstrap_ari": boot_min,
    }
