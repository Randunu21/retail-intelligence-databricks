"""Final segmentation model and segment profiles (Phase 3.5).

`SegmentModel` is everything needed to put a customer in a segment, as plain JSON:
the preprocessing parameters (log, clip bounds, mean/sd) and the cluster centres. Scoring a
new feature snapshot (Phase 6) = `apply_preprocessor` + nearest centre; no Spark ML model has
to be saved or loaded (which on Free Edition serverless is exactly what fails for some
models, see BisectingKMeans).

**Stable segment ids.** KMeans numbers clusters arbitrarily, and a refit can swap them. Ids
are therefore re-ordered by a business quantity: segment 0 has the highest mean `monetary`
(on the scaled axis), then descending. Names are attached later by profile, never by raw
cluster number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from segmentation.preprocessing import PreprocessParams, apply_preprocessor, scaled_name
from segmentation.stability import assign_nearest

SEGMENT_COL = "segment_id"


@dataclass
class SegmentModel:
    algorithm: str
    k: int
    seed: int
    features: list[str]
    centers: list[list[float]]  # in scaled space, ordered by segment_id
    preprocess: dict = field(default_factory=dict)  # PreprocessParams.to_dict()
    order_by: str = "monetary"

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "k": self.k,
            "seed": self.seed,
            "features": list(self.features),
            "centers": [list(c) for c in self.centers],
            "preprocess": dict(self.preprocess),
            "order_by": self.order_by,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> SegmentModel:
        return cls(
            algorithm=d["algorithm"],
            k=int(d["k"]),
            seed=int(d["seed"]),
            features=list(d["features"]),
            centers=[list(map(float, c)) for c in d["centers"]],
            preprocess=dict(d["preprocess"]),
            order_by=d.get("order_by", "monetary"),
        )

    @property
    def z_cols(self) -> list[str]:
        return [scaled_name(f) for f in self.features]

    def assign_prepared(self, prepared: DataFrame) -> DataFrame:
        """Add segment_id to an already preprocessed frame (has the z_ columns)."""
        return assign_nearest(prepared, self.centers, self.z_cols, label_col=SEGMENT_COL)

    def score(self, features: DataFrame, keep: Sequence[str] = ("customer_id",)) -> DataFrame:
        """Raw feature rows -> keep columns + z columns + segment_id (no refit)."""
        params = PreprocessParams.from_dict(self.preprocess)
        prepared = apply_preprocessor(features, params, keep=list(keep))
        return self.assign_prepared(prepared).drop("features")


def order_centers(
    centers: Sequence[Sequence[float]], features: Sequence[str], order_by: str = "monetary"
) -> tuple[list[list[float]], dict[int, int]]:
    """Sort centres by the `order_by` coordinate, descending.

    Returns (sorted centres, {original cluster index: new segment_id}).
    """
    idx = list(features).index(order_by)
    order = sorted(range(len(centers)), key=lambda i: (-centers[i][idx], i))
    return [list(map(float, centers[i])) for i in order], {
        old: new for new, old in enumerate(order)
    }


def fit_segment_model(
    prepared: DataFrame,
    params: PreprocessParams,
    k: int,
    seed: int,
    features_col: str = "features",
    order_by: str = "monetary",
) -> SegmentModel:
    """Fit KMeans on the preprocessed customers and return a model-free SegmentModel."""
    from pyspark.ml.clustering import KMeans

    model = KMeans(k=k, seed=seed, maxIter=100, featuresCol=features_col).fit(prepared)
    centers = [[float(x) for x in c] for c in model.clusterCenters()]
    del model  # free the serverless model cache; the centres are all we need
    ordered, _ = order_centers(centers, params.features, order_by)
    return SegmentModel(
        algorithm="kmeans",
        k=k,
        seed=seed,
        features=list(params.features),
        centers=ordered,
        preprocess=params.to_dict(),
        order_by=order_by,
    )


# Profile columns: numeric features summarised by mean and median per segment.
PROFILE_FEATURES: list[str] = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_basket_value",
    "distinct_products",
    "tenure_days",
    "n_purchase_days",
    "mean_days_between_purchases",
    "return_rate",
    "returned_value",
    "seasonal_revenue_share",
]


def segment_profile(
    segmented: DataFrame,
    features: Sequence[str] = PROFILE_FEATURES,
    segment_col: str = SEGMENT_COL,
) -> DataFrame:
    """One row per segment: size, value, mean + median of each feature, and shares of
    customers with notable properties. `segmented` = feature rows + segment_id."""
    total_customers = F.sum(F.col("customers")).over(_all())
    total_revenue = F.sum(F.col("revenue")).over(_all())
    aggs = [
        F.count(F.lit(1)).alias("customers"),
        F.sum("monetary").alias("revenue"),
        F.sum("returned_value").alias("returned"),
        F.avg((F.col("frequency") == 1).cast("double")).alias("pct_one_invoice"),
        F.avg((F.col("n_return_invoices") > 0).cast("double")).alias("pct_with_returns"),
        F.avg((F.col("preferred_country") == "United Kingdom").cast("double")).alias("pct_uk"),
        F.avg(F.col("is_tenure_censored").cast("double")).alias("pct_tenure_censored"),
        F.avg((F.col("tenure_days") <= 90).cast("double")).alias("pct_first_bought_last_90d"),
        F.avg((F.col("recency_days") <= 90).cast("double")).alias("pct_bought_last_90d"),
    ]
    for c in features:
        aggs.append(F.avg(c).alias(f"mean_{c}"))
        aggs.append(F.percentile_approx(F.col(c), 0.5, 10000).alias(f"median_{c}"))
    out = segmented.groupBy(segment_col).agg(*aggs)
    return (
        out.withColumn("customer_share", F.col("customers") / total_customers)
        .withColumn("revenue_share", F.col("revenue") / total_revenue)
        .withColumn("revenue_per_customer", F.col("revenue") / F.col("customers"))
        .orderBy(segment_col)
    )


def _all():
    from pyspark.sql import Window

    return Window.partitionBy()


def overall_profile(features_df: DataFrame, features: Sequence[str] = PROFILE_FEATURES) -> dict:
    """All-customer means and medians, the baseline each segment is compared with."""
    row = features_df.agg(
        *[F.avg(c).alias(f"mean_{c}") for c in features],
        *[F.percentile_approx(F.col(c), 0.5, 10000).alias(f"median_{c}") for c in features],
    ).first()
    return row.asDict()
