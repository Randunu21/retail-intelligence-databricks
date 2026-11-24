"""Bisecting k-means built on Spark ML KMeans (fallback for Databricks Free Edition).

Why this exists: Spark ML's `BisectingKMeans` fails on Free Edition serverless with
`DBFS_DISABLED` (the Spark Connect model cache tries to write the fitted model's metadata to
a blocked path; KMeans and GaussianMixture don't). The algorithm itself is simple, so it is
implemented here with the parts that do work.

Algorithm (divisive / top-down hierarchical clustering):
1. Start with every customer in one cluster (id 0).
2. Pick the cluster with the largest SSE (the "worst" cluster) and split it in two with
   KMeans(k=2) fitted on that cluster's customers only.
3. The left half keeps the parent's id, the right half gets the next new id.
4. Repeat until there are k clusters.

Spark's own BisectingKMeans splits every divisible cluster level by level; choosing the
largest-SSE cluster first is the "biggest inertia" strategy (scikit-learn's default). Both
are standard.

The fitted model is just the ordered list of splits (two centres each), stored as JSON.
Assigning a customer = replay the splits: if it's in the parent cluster, move it to the
right child when it is strictly closer to the right centre (ties go left, like KMeans picks
the first centre). Because it's greedy, **the first k-1 splits of a k_max tree are exactly the
k-cluster model**, so one fit gives the models for every k in a sweep.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from segmentation.metrics import cluster_stats

LEAF_COL = "_leaf"


@dataclass(frozen=True)
class Split:
    parent: int
    child: int
    left_center: tuple[float, ...]
    right_center: tuple[float, ...]
    parent_size: int
    parent_sse: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> Split:
        return cls(
            parent=int(d["parent"]),
            child=int(d["child"]),
            left_center=tuple(d["left_center"]),
            right_center=tuple(d["right_center"]),
            parent_size=int(d["parent_size"]),
            parent_sse=float(d["parent_sse"]),
        )


def _sq_dist(z_cols: Sequence[str], center: Sequence[float]) -> Column:
    terms = [(F.col(c) - F.lit(float(v))) ** 2 for c, v in zip(z_cols, center, strict=True)]
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def apply_split(label: Column, split: Split, z_cols: Sequence[str]) -> Column:
    go_right = _sq_dist(z_cols, split.right_center) < _sq_dist(z_cols, split.left_center)
    return F.when((label == split.parent) & go_right, F.lit(split.child)).otherwise(label)


def assign(
    df: DataFrame,
    splits: Sequence[Split],
    z_cols: Sequence[str],
    k: int | None = None,
    label_col: str = "cluster",
) -> DataFrame:
    """Add `label_col` (ids 0..k-1) by replaying the first k-1 splits.

    One withColumn per split (not one nested expression): each step references the previous
    column instead of copying it, so the plan stays linear in k instead of doubling.
    """
    use = list(splits) if k is None else list(splits)[: k - 1]
    out = df.withColumn(label_col, F.lit(0))
    for s in use:
        out = out.withColumn(label_col, apply_split(F.col(label_col), s, z_cols))
    return out.withColumn(label_col, F.col(label_col).cast("int"))


def fit_bisecting(
    df: DataFrame,
    k_max: int,
    z_cols: Sequence[str],
    seed: int = 42,
    features_col: str = "features",
    max_iter: int = 50,
    min_split_size: int = 2,
) -> list[Split]:
    """Fit the split sequence up to k_max clusters (k_max - 1 KMeans(k=2) fits)."""
    from pyspark.ml.clustering import KMeans

    splits: list[Split] = []
    unsplittable: set[int] = set()
    work = df.withColumn(LEAF_COL, F.lit(0))
    while len(splits) < k_max - 1:
        stats = cluster_stats(work, LEAF_COL, z_cols)
        candidates = [
            (s["sse"], leaf)
            for leaf, s in stats.items()
            if s["n"] >= min_split_size and s["sse"] > 0 and leaf not in unsplittable
        ]
        if not candidates:
            break
        sse, leaf = max(candidates)
        subset = work.where(F.col(LEAF_COL) == leaf)
        model = KMeans(
            k=2, seed=seed, maxIter=max_iter, featuresCol=features_col, predictionCol="_km"
        ).fit(subset)
        centers = [tuple(float(x) for x in c) for c in model.clusterCenters()]
        del model  # free the Spark Connect model cache (serverless keeps fitted models)
        if len(centers) < 2:  # all points identical: can't split this one
            unsplittable.add(leaf)
            continue
        split = Split(
            parent=leaf,
            child=len(splits) + 1,
            left_center=centers[0],
            right_center=centers[1],
            parent_size=stats[leaf]["n"],
            parent_sse=sse,
        )
        splits.append(split)
        work = work.withColumn(LEAF_COL, apply_split(F.col(LEAF_COL), split, z_cols))
    return splits
