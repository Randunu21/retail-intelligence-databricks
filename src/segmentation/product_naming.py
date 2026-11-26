"""Business names, action groups and recommended actions for products (Phase 4.3b).

**Names from the real profiles** (run 47f3364b, bisecting k-means k = 3, as of 2011-12-10;
shares of the whole catalogue of 4,725 products):

| Segment | Profile (medians) | Products / revenue |
|---|---|---|
| Core Fast Movers | 34 units/week, sells in 91 % of weeks, 151 customers, last sale 2 days ago; 96 % sold in the last 30 days | 43 % / 87 % |
| Fading Lines | last sale 331 days ago (75 % not sold for 180+ days), sells in 26 % of weeks, lumpiest months (CV 2.1); 90 % on sale since the data started | 20 % / 7 % |
| Slow Movers | 2.5 units/week, 28 customers, still selling (85 % in the last 30 days), the highest prices (median 2.95, mean 5.55) and most UK-centric | 16 % / 4 % |

Plus the two rule-based groups from the long-tail rule (`product_segments`): Long Tail
(< 20 invoices; 20 % / 0.8 %) and New Products (< 3 months on sale; 0.7 % / 0.9 %).

**No "seasonal" segment.** At k = 3 seasonality is not what separates products: the Q4 share
centre is close to average in every cluster, and Q4-heavy products exist in all of them
(25 % of Core, 23 % of Slow Movers). So seasonality is a *flag inside segments*, like the
Phase 3 action groups, instead of a cluster. (Expected categories like "seasonal" or
"premium" were hypotheses; the data supports velocity and lifecycle as the main axes.)

Names are attached by **signature** on the cluster centres (z scale), never by cluster id,
and the signatures must pick three different segments, otherwise `NamingError`:
- Core Fast Movers: highest `units_per_week`
- Fading Lines: highest `days_since_last_sale`
- Slow Movers: lowest `units_per_week`
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from segmentation.naming import NamingError
from segmentation.product_segments import ELIGIBILITY_COL, LOW_VOLUME, NEW_PRODUCT, RULE_SEGMENTS

SEASONAL_SHARE_MIN = 0.5  # Q4 share above half (a flat product is ~0.25 in this data)
DISCONTINUED_DAYS = 180  # no sale for ~6 months
PREMIUM_QUANTILE = 0.9  # top 10 % of median price among clustered products
HIGH_RETURN_RATE = 0.10  # 10 %+ of units returned: quality / description check


@dataclass(frozen=True)
class ProductSegmentDefinition:
    key: str
    name: str
    signature_feature: str
    direction: str  # "max" or "min": the segment with the highest / lowest centre
    description: str


PRODUCT_SEGMENT_DEFINITIONS: tuple[ProductSegmentDefinition, ...] = (
    ProductSegmentDefinition(
        "core",
        "Core Fast Movers",
        "units_per_week",
        "max",
        "Sell almost every week to many customers: the range that makes most of the revenue.",
    ),
    ProductSegmentDefinition(
        "fading",
        "Fading Lines",
        "days_since_last_sale",
        "max",
        "Mostly not sold for months: discontinued or dying lines, sold in bursts while active.",
    ),
    ProductSegmentDefinition(
        "slow",
        "Slow Movers",
        "units_per_week",
        "min",
        "Still on sale but low demand: a few units a week, few customers, higher-priced and "
        "UK-centric.",
    ),
)

RULE_SEGMENT_KEYS = {LOW_VOLUME: "long_tail", NEW_PRODUCT: "new"}


def name_product_segments(
    centers: Sequence[Sequence[float]],
    features: Sequence[str],
    definitions: Sequence[ProductSegmentDefinition] = PRODUCT_SEGMENT_DEFINITIONS,
) -> dict[int, ProductSegmentDefinition]:
    """{segment_id: definition} by signature; every segment named exactly once."""
    if len(definitions) != len(centers):
        raise NamingError(f"{len(definitions)} names for {len(centers)} segments")
    named: dict[int, ProductSegmentDefinition] = {}
    for d in definitions:
        j = list(features).index(d.signature_feature)
        pick = max if d.direction == "max" else min
        sid = pick(range(len(centers)), key=lambda i: centers[i][j])
        if sid in named:
            raise NamingError(
                f"segment {sid} matches both {named[sid].name!r} and {d.name!r}: review the names"
            )
        named[sid] = d
    return named


# action group -> segment key, recommended action, KPI to track
PRODUCT_ACTION_PLAYBOOK: dict[str, dict[str, str]] = {
    "Core: Seasonal Best-Sellers": {
        "segment": "core",
        "action": "Stock up before Q4 (order by August), feature in the Christmas catalogue and "
        "gift guides; anchor seasonal cross-sell bundles.",
        "kpi": "Q4 stock-outs; Q4 revenue vs last year; sell-through by mid-December.",
    },
    "Core: Year-round Staples": {
        "segment": "core",
        "action": "Never out of stock: safety stock and automatic replenishment; volume-tier "
        "pricing for wholesale buyers; use as cross-sell anchors (Phase 4 rules).",
        "kpi": "Availability (in-stock rate); units per week; attach rate of cross-sells.",
    },
    "Fading: Discontinued": {
        "segment": "fading",
        "action": "Confirm delisting, clear remaining stock (clearance bundles), remove from "
        "the website and catalogue; suggest the closest active alternative.",
        "kpi": "Stock cleared; holding cost released.",
    },
    "Fading: Declining": {
        "segment": "fading",
        "action": "Decide keep or delist: check margin and whether sales are a one-off burst; "
        "if kept, promote with a Core product; otherwise run down stock.",
        "kpi": "Units per week over the next quarter; margin per unit.",
    },
    "Slow Movers: Premium Niche": {
        "segment": "slow",
        "action": "Keep as a curated premium range with low stock depth (or made to order); "
        "show next to related best-sellers; targeted to high-value customers.",
        "kpi": "Revenue per product; stock days of cover; margin.",
    },
    "Slow Movers: Seasonal": {
        "segment": "slow",
        "action": "Buy shallow for Q4 only and promote within the season; no year-round stock.",
        "kpi": "Q4 sell-through; leftover stock in January.",
    },
    "Slow Movers: Low Demand": {
        "segment": "slow",
        "action": "Review the range: reduce stock depth, bundle with Core products, or replace "
        "with a better-selling variant.",
        "kpi": "Units per week; stock days of cover.",
    },
    "Long Tail": {
        "segment": "long_tail",
        "action": RULE_SEGMENTS[LOW_VOLUME]["recommended_action"],
        "kpi": "Revenue per listed product; delisted count.",
    },
    "New Products": {
        "segment": "new",
        "action": RULE_SEGMENTS[NEW_PRODUCT]["recommended_action"],
        "kpi": "Units per week in the first 3 months; repeat orders.",
    },
}


def premium_price_threshold(clustered: DataFrame, quantile: float = PREMIUM_QUANTILE) -> float:
    """p90 of median_unit_price among clustered products (computed, not hand-picked)."""
    return float(
        clustered.agg(F.percentile_approx("median_unit_price", quantile, 100000)).first()[0]
    )


def add_product_groups(
    labelled: DataFrame,
    names: Mapping[int, ProductSegmentDefinition],
    premium_min: float,
    segment_col: str = "segment_id",
) -> DataFrame:
    """labelled = all products with cluster_eligibility and segment_id (NULL for rule-based)
    -> + segment_key, segment_name, flags, action_group, recommended_action."""
    key_map = F.create_map(*[x for sid, d in names.items() for x in (F.lit(sid), F.lit(d.key))])
    name_map = F.create_map(*[x for sid, d in names.items() for x in (F.lit(sid), F.lit(d.name))])
    elig = F.col(ELIGIBILITY_COL)
    segment_key = (
        F.when(elig == LOW_VOLUME, F.lit(RULE_SEGMENT_KEYS[LOW_VOLUME]))
        .when(elig == NEW_PRODUCT, F.lit(RULE_SEGMENT_KEYS[NEW_PRODUCT]))
        .otherwise(key_map[F.col(segment_col)])
    )
    segment_name = (
        F.when(elig == LOW_VOLUME, F.lit(RULE_SEGMENTS[LOW_VOLUME]["segment_name"]))
        .when(elig == NEW_PRODUCT, F.lit(RULE_SEGMENTS[NEW_PRODUCT]["segment_name"]))
        .otherwise(name_map[F.col(segment_col)])
    )
    out = (
        labelled.withColumn("segment_key", segment_key)
        .withColumn("segment_name", segment_name)
        .withColumn("is_seasonal", F.col("seasonal_revenue_share") > F.lit(SEASONAL_SHARE_MIN))
        .withColumn("is_discontinued", F.col("days_since_last_sale") > F.lit(DISCONTINUED_DAYS))
        .withColumn("is_premium", F.col("median_unit_price") >= F.lit(premium_min))
        .withColumn("is_high_return", F.col("return_rate") >= F.lit(HIGH_RETURN_RATE))
    )
    key = F.col("segment_key")
    group = (
        F.when((key == "core") & F.col("is_seasonal"), "Core: Seasonal Best-Sellers")
        .when(key == "core", "Core: Year-round Staples")
        .when((key == "fading") & F.col("is_discontinued"), "Fading: Discontinued")
        .when(key == "fading", "Fading: Declining")
        .when((key == "slow") & F.col("is_premium"), "Slow Movers: Premium Niche")
        .when((key == "slow") & F.col("is_seasonal"), "Slow Movers: Seasonal")
        .when(key == "slow", "Slow Movers: Low Demand")
        .when(key == "long_tail", "Long Tail")
        .when(key == "new", "New Products")
    )
    action_map = F.create_map(
        *[x for g, a in PRODUCT_ACTION_PLAYBOOK.items() for x in (F.lit(g), F.lit(a["action"]))]
    )
    return out.withColumn("action_group", group).withColumn(
        "recommended_action", action_map[F.col("action_group")]
    )


def group_profile(grouped: DataFrame) -> list[dict]:
    """Per action group: products, shares of products and revenue, medians, flag shares.
    Ordered like PRODUCT_ACTION_PLAYBOOK."""
    total = grouped.agg(F.count(F.lit(1)).alias("n"), F.sum("revenue").alias("rev")).first()
    rows = (
        grouped.groupBy("segment_name", "action_group")
        .agg(
            F.count(F.lit(1)).alias("products"),
            F.sum("revenue").alias("revenue"),
            F.percentile_approx("units_per_week", 0.5, 10000).alias("median_units_per_week"),
            F.percentile_approx("median_unit_price", 0.5, 10000).alias("median_price"),
            F.percentile_approx("days_since_last_sale", 0.5, 10000).alias(
                "median_days_since_last_sale"
            ),
            F.percentile_approx("seasonal_revenue_share", 0.5, 10000).alias("median_q4_share"),
            F.avg(F.col("is_high_return").cast("double")).alias("pct_high_return"),
        )
        .collect()
    )
    order = list(PRODUCT_ACTION_PLAYBOOK)
    out = []
    for r in rows:
        d = r.asDict()
        d["product_share"] = d["products"] / total["n"]
        d["revenue_share"] = d["revenue"] / total["rev"]
        out.append(d)
    return sorted(out, key=lambda d: order.index(d["action_group"]))
