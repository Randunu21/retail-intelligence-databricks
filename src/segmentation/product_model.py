"""Final product segmentation model and product segment profiles (Phase 4.3).

**The model is the evaluated sweep trial, not a refit** (same rule as Phase 3): on serverless
a refit isn't bit-reproducible across sessions, so the final model is rebuilt from what the
sweep RECORDED. The 4.2 choice was bisecting k-means, k = 3, whose recorded model is the
sequence of splits (two centres each). Scoring replays the splits; it is **not** "nearest leaf
centre" (a hierarchical assignment can differ from nearest-centroid near the boundaries).
KMeans trials (recorded centres, nearest-centre assignment) are supported too, so the same
class works whichever algorithm a later snapshot picks.

**Stable segment ids**: leaves are re-numbered by their centre on `order_by` (default
`units_per_week`, descending): segment 0 = the fastest-selling group. Names are attached in
4.3b by profile signature, never by raw cluster number.

Everything is plain JSON (preprocessing params + splits or centres + id map), logged to
MLflow and re-applied at scoring time (Phase 6) without refitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from segmentation.bisecting import Split, assign
from segmentation.metrics import cluster_stats
from segmentation.preprocessing import PreprocessParams, apply_preprocessor, scaled_name
from segmentation.product_segments import CLUSTERED, ELIGIBILITY_COL
from segmentation.stability import assign_nearest

SEGMENT_COL = "segment_id"
SEGMENT_LABEL_COL = "segment_label"
_RAW = "_raw_cluster"


@dataclass
class ProductSegmentModel:
    algorithm: str  # "bisecting_kmeans" (splits) or "kmeans" (centres)
    k: int
    seed: int
    features: list[str]
    preprocess: dict  # PreprocessParams.to_dict()
    splits: list[dict] = field(default_factory=list)  # bisecting: Split.to_dict(), in order
    centers: list[list[float]] = field(default_factory=list)  # kmeans: raw cluster order
    id_map: dict[int, int] = field(default_factory=dict)  # raw cluster id -> segment_id
    leaf_centers: list[list[float]] = field(default_factory=list)  # by segment_id (z space)
    leaf_sizes: list[int] = field(default_factory=list)  # by segment_id, at fit time
    order_by: str = "units_per_week"

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "k": self.k,
            "seed": self.seed,
            "features": list(self.features),
            "preprocess": dict(self.preprocess),
            "splits": [dict(s) for s in self.splits],
            "centers": [list(c) for c in self.centers],
            "id_map": {str(k): v for k, v in self.id_map.items()},
            "leaf_centers": [list(c) for c in self.leaf_centers],
            "leaf_sizes": list(self.leaf_sizes),
            "order_by": self.order_by,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> ProductSegmentModel:
        return cls(
            algorithm=d["algorithm"],
            k=int(d["k"]),
            seed=int(d["seed"]),
            features=list(d["features"]),
            preprocess=dict(d["preprocess"]),
            splits=[dict(s) for s in d.get("splits", [])],
            centers=[list(map(float, c)) for c in d.get("centers", [])],
            id_map={int(k): int(v) for k, v in d["id_map"].items()},
            leaf_centers=[list(map(float, c)) for c in d.get("leaf_centers", [])],
            leaf_sizes=[int(n) for n in d.get("leaf_sizes", [])],
            order_by=d.get("order_by", "units_per_week"),
        )

    @property
    def z_cols(self) -> list[str]:
        return [scaled_name(f) for f in self.features]

    def assign_raw(self, prepared: DataFrame, label_col: str = _RAW) -> DataFrame:
        """Raw cluster id from the recorded model (prepared = has the z_ columns)."""
        if self.algorithm == "bisecting_kmeans":
            splits = [Split.from_dict(s) for s in self.splits]
            return assign(prepared, splits, self.z_cols, k=self.k, label_col=label_col)
        if self.algorithm == "kmeans":
            return assign_nearest(prepared, self.centers, self.z_cols, label_col=label_col)
        raise ValueError(f"unsupported algorithm {self.algorithm!r}")

    def assign_prepared(self, prepared: DataFrame) -> DataFrame:
        """Add segment_id (stable, ordered ids) to an already preprocessed frame."""
        if not self.id_map:
            raise ValueError("id_map is empty: build the model with product_model_from_trial")
        mapping = F.create_map(
            *[x for raw, sid in sorted(self.id_map.items()) for x in (F.lit(raw), F.lit(sid))]
        )
        return (
            self.assign_raw(prepared)
            .withColumn(SEGMENT_COL, mapping[F.col(_RAW)].cast("int"))
            .drop(_RAW)
        )

    def score(self, features: DataFrame, keep: Sequence[str] = ("stock_code",)) -> DataFrame:
        """Raw feature rows -> keep columns + z columns + segment_id (no refit)."""
        params = PreprocessParams.from_dict(self.preprocess)
        prepared = apply_preprocessor(features, params, keep=list(keep))
        return self.assign_prepared(prepared).drop("features")


def order_leaves(stats: Mapping[int, Mapping], features: Sequence[str], order_by: str):
    """Sort raw clusters by their centre on `order_by`, descending (ties: lower raw id).

    Returns ({raw id: segment_id}, centres by segment_id, sizes by segment_id)."""
    j = list(features).index(order_by)
    raws = sorted(stats, key=lambda r: (-stats[r]["centroid"][j], r))
    id_map = {raw: sid for sid, raw in enumerate(raws)}
    return (
        id_map,
        [list(map(float, stats[r]["centroid"])) for r in raws],
        [int(stats[r]["n"]) for r in raws],
    )


def product_model_from_trial(
    sweep_results: Sequence[Mapping],
    algorithm: str,
    k: int,
    params: PreprocessParams,
    prepared: DataFrame,
    seed: int,
    order_by: str = "units_per_week",
) -> ProductSegmentModel:
    """Rebuild the evaluated model from the recorded sweep trial and order its segments.

    `prepared` (the preprocessed training products) is only used to compute each leaf's
    centre and size, which fix the segment order and are kept for profiling/naming.
    """
    trial = next(
        (
            r
            for r in sweep_results
            if r["algorithm"] == algorithm and r["k"] == k and not r.get("error")
        ),
        None,
    )
    if trial is None:
        raise ValueError(f"no successful {algorithm} trial with k={k} in the sweep results")
    info = trial["model_info"]
    model = ProductSegmentModel(
        algorithm=algorithm,
        k=k,
        seed=seed,
        features=list(params.features),
        preprocess=params.to_dict(),
        splits=[dict(s) for s in info.get("splits", [])][: k - 1],
        centers=[list(map(float, c)) for c in info.get("centers", [])],
        order_by=order_by,
    )
    stats = cluster_stats(model.assign_raw(prepared), _RAW, model.z_cols)
    if len(stats) != k:
        raise ValueError(f"expected {k} non-empty clusters, got {len(stats)}")
    model.id_map, model.leaf_centers, model.leaf_sizes = order_leaves(
        stats, model.features, order_by
    )
    return model


# --------------------------------------------------------------------------- profiles

PRODUCT_PROFILE_FEATURES: list[str] = [
    "revenue",
    "n_invoices",
    "units_sold",
    "units_per_week",
    "active_week_share",
    "median_unit_price",
    "avg_quantity_per_line",
    "unique_customers",
    "n_countries",
    "non_uk_revenue_share",
    "seasonal_revenue_share",
    "monthly_units_cv",
    "return_rate",
    "days_on_sale",
    "days_since_last_sale",
]


def label_all_products(with_eligibility: DataFrame, segmented: DataFrame) -> DataFrame:
    """Every product with a `segment_label`: 'cluster <id>' for clustered products,
    the eligibility class (low_volume / new_product) for the rest. Also keeps segment_id
    (NULL for rule-based products)."""
    seg = segmented.select("stock_code", SEGMENT_COL)
    return with_eligibility.join(seg, "stock_code", "left").withColumn(
        SEGMENT_LABEL_COL,
        F.when(
            F.col(ELIGIBILITY_COL) == CLUSTERED,
            F.concat(F.lit("cluster "), F.col(SEGMENT_COL).cast("string")),
        ).otherwise(F.col(ELIGIBILITY_COL)),
    )


def product_segment_profile(
    labelled: DataFrame,
    features: Sequence[str] = PRODUCT_PROFILE_FEATURES,
    label_col: str = SEGMENT_LABEL_COL,
) -> DataFrame:
    """One row per segment label: size, revenue, shares of the whole catalogue, mean and
    median of each profile feature, and shares of products with notable properties."""
    from pyspark.sql import Window

    everything = Window.partitionBy()
    aggs = [
        F.count(F.lit(1)).alias("products"),
        F.sum("revenue").alias("total_revenue"),
        F.sum("units_sold").alias("total_units"),
        F.avg((F.col("days_since_last_sale") <= 30).cast("double")).alias("pct_sold_last_30d"),
        F.avg((F.col("days_since_last_sale") > 180).cast("double")).alias("pct_not_sold_180d"),
        F.avg((F.col("seasonal_revenue_share") > 0.5).cast("double")).alias("pct_q4_majority"),
        F.avg((F.col("non_uk_revenue_share") > 0.25).cast("double")).alias("pct_international"),
        F.avg((F.col("median_unit_price") >= 10).cast("double")).alias("pct_price_10_plus"),
        F.avg((F.col("return_rate") > 0.05).cast("double")).alias("pct_return_rate_5_plus"),
        F.avg(F.col("is_history_censored").cast("double")).alias("pct_history_censored"),
    ]
    for c in features:
        aggs.append(F.avg(c).alias(f"mean_{c}"))
        aggs.append(F.percentile_approx(F.col(c), 0.5, 10000).alias(f"median_{c}"))
    out = labelled.groupBy(label_col).agg(*aggs)
    return (
        out.withColumn("product_share", F.col("products") / F.sum("products").over(everything))
        .withColumn(
            "revenue_share", F.col("total_revenue") / F.sum("total_revenue").over(everything)
        )
        .withColumn("revenue_per_product", F.col("total_revenue") / F.col("products"))
        .orderBy(label_col)
    )


def top_products(
    labelled: DataFrame,
    descriptions: DataFrame,
    n: int = 8,
    label_col: str = SEGMENT_LABEL_COL,
) -> DataFrame:
    """Top-n products by revenue per segment label, with their description (examples make
    the profiles readable: a 'seasonal' segment should show Christmas items)."""
    from pyspark.sql import Window

    w = Window.partitionBy(label_col).orderBy(F.col("revenue").desc(), F.col("stock_code"))
    return (
        labelled.join(descriptions.select("stock_code", "description"), "stock_code", "left")
        .withColumn("rank", F.row_number().over(w))
        .where(F.col("rank") <= n)
        .select(
            label_col,
            "rank",
            "stock_code",
            "description",
            "revenue",
            "n_invoices",
            "units_per_week",
            "median_unit_price",
            "seasonal_revenue_share",
            "days_since_last_sale",
        )
        .orderBy(label_col, "rank")
    )
