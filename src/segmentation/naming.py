"""Business names, action groups and recommended actions (Phase 3.5b).

Two layers:

1. **Segments** (from the clustering): 3 stable groups, named from their real profiles in
   Phase 3.5a (run b27fc5b7, as of 2011-12-10):

   | Segment | Profile (medians) | Share of customers / revenue |
   |---|---|---|
   | Loyal Customers | 37 days since last order, 8 orders, 118 products, 654 days tenure, nobody bought only once | 39% / 86% |
   | Seasonal & New | Q4 share of spend 1.0, 281 days tenure, 35% first bought in the last 90 days | 26% / 7% |
   | Hibernating | 396 days since last order, 1 order (50% bought once), 7.5% active in the last 90 days | 35% / 7% |

   A name is attached by a **signature** (e.g. "highest recency centre"), never by cluster
   number, and the signatures must pick three *different* segments. If a future refit changes
   the profiles so that they don't, `name_segments` raises: the names have to be reviewed,
   not silently reused.

2. **Action groups** (rules inside each segment): the clusters are stable at k = 3 but too
   coarse to act on (k = 4 was unstable: seed ARI 0.62). Transparent rules split each
   segment into groups that need different actions:
   - `is_overdue`: a repeat buyer (2+ purchase days) whose time since the last order is more
     than twice their own average gap between orders. This is personal: a monthly buyer is
     overdue after ~2 months, a twice-a-year buyer after ~a year.
   - `is_new_customer`: first purchase in the last 90 days.
   - `is_top_value`: monetary in the top 10% of all customers.
   Precedence inside Loyal Customers: At Risk before Champions (a top customer drifting away is
   the most urgent case).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

NEW_CUSTOMER_DAYS = 90
OVERDUE_FACTOR = 2.0
TOP_VALUE_QUANTILE = 0.9


@dataclass(frozen=True)
class SegmentDefinition:
    key: str
    name: str
    signature_feature: str  # the segment whose centre is highest on this feature
    description: str


SEGMENT_DEFINITIONS: tuple[SegmentDefinition, ...] = (
    SegmentDefinition(
        "loyal",
        "Loyal Customers",
        "monetary",
        "Recent, frequent, broad buyers with long tenure: the revenue base.",
    ),
    SegmentDefinition(
        "seasonal_new",
        "Seasonal & New",
        "seasonal_revenue_share",
        "Spend concentrated in Oct-Dec: new customers acquired in Q4 2011 and gift-season-only buyers.",
    ),
    SegmentDefinition(
        "hibernating",
        "Hibernating",
        "recency_days",
        "Last order about a year ago or more, mostly one or two orders.",
    ),
)


class NamingError(ValueError):
    """The profiles no longer match the agreed names (review before reusing them)."""


def name_segments(
    centers: Sequence[Sequence[float]],
    features: Sequence[str],
    definitions: Sequence[SegmentDefinition] = SEGMENT_DEFINITIONS,
) -> dict[int, SegmentDefinition]:
    """{segment_id: definition}: each definition takes the segment with the highest centre on
    its signature feature; all segments must be named exactly once."""
    if len(definitions) != len(centers):
        raise NamingError(f"{len(definitions)} names for {len(centers)} segments")
    named: dict[int, SegmentDefinition] = {}
    for d in definitions:
        j = list(features).index(d.signature_feature)
        sid = max(range(len(centers)), key=lambda i: centers[i][j])
        if sid in named:
            raise NamingError(
                f"segment {sid} matches both {named[sid].name!r} and {d.name!r}: review the names"
            )
        named[sid] = d
    return named


# action group -> (segment key, recommended action, KPI to track)
ACTION_PLAYBOOK: dict[str, dict[str, str]] = {
    "Loyal: At Risk": {
        "segment": "loyal",
        "action": "Personal win-back before they lapse: account call/email from a named contact, "
        "reminder of their usual products, time-limited loyalty discount.",
        "kpi": "Share placing an order within 60 days; revenue recovered.",
    },
    "Loyal: Champions": {
        "segment": "loyal",
        "action": "Retain and grow: VIP tier, early access to new ranges and the Christmas "
        "catalogue, referral programme, no blanket discounts (they buy anyway).",
        "kpi": "Retention rate; revenue per customer; referrals.",
    },
    "Loyal: Active": {
        "segment": "loyal",
        "action": "Grow basket and frequency: cross-sell from Phase 4 association rules, "
        "volume-tier pricing, replenishment reminders at their usual interval.",
        "kpi": "Orders per customer; average basket value; distinct products.",
    },
    "Seasonal & New: New Customers": {
        "segment": "seasonal_new",
        "action": "Onboarding journey: welcome series, best-sellers, second-order incentive "
        "within 30-45 days so they become repeat buyers outside Q4.",
        "kpi": "Second-order rate within 90 days (Phase 5 propensity target).",
    },
    "Seasonal & New: Seasonal Gifters": {
        "segment": "seasonal_new",
        "action": "Seasonal calendar: pre-season reminders in September/October, gift guides, "
        "plus off-season occasions (Valentine's, Easter, Mother's Day) to spread demand.",
        "kpi": "Share buying outside Q4; Q4 repeat rate.",
    },
    "Hibernating: Former Repeat Buyers": {
        "segment": "hibernating",
        "action": "Low-cost reactivation: 'we miss you' email with an offer on previously bought "
        "categories; ask why they left; stop if no response.",
        "kpi": "Reactivation rate; cost per reactivated customer.",
    },
    "Hibernating: One-time Buyers": {
        "segment": "hibernating",
        "action": "Lowest priority: include in seasonal newsletters only, then suppress to cut "
        "marketing cost and keep the list clean.",
        "kpi": "Reactivation rate vs mailing cost; unsubscribe rate.",
    },
}


def top_value_threshold(features: DataFrame, quantile: float = TOP_VALUE_QUANTILE) -> float:
    return float(features.agg(F.percentile_approx("monetary", quantile, 100000)).first()[0])


def add_action_groups(
    segmented: DataFrame,
    names: Mapping[int, SegmentDefinition],
    top_value_min: float,
    segment_col: str = "segment_id",
) -> DataFrame:
    """Add segment_key, segment_name, the three flags, action_group and recommended_action."""
    key_map = F.create_map(*[x for sid, d in names.items() for x in (F.lit(sid), F.lit(d.key))])
    name_map = F.create_map(*[x for sid, d in names.items() for x in (F.lit(sid), F.lit(d.name))])
    is_overdue = (F.col("n_purchase_days") >= 2) & (
        F.col("recency_days") > F.lit(OVERDUE_FACTOR) * F.col("mean_days_between_purchases")
    )
    out = (
        segmented.withColumn("segment_key", key_map[F.col(segment_col)])
        .withColumn("segment_name", name_map[F.col(segment_col)])
        .withColumn("is_new_customer", F.col("tenure_days") <= NEW_CUSTOMER_DAYS)
        .withColumn("is_overdue", F.coalesce(is_overdue, F.lit(False)))
        .withColumn("is_top_value", F.col("monetary") >= F.lit(top_value_min))
    )
    key = F.col("segment_key")
    group = (
        F.when((key == "loyal") & F.col("is_overdue"), "Loyal: At Risk")
        .when((key == "loyal") & F.col("is_top_value"), "Loyal: Champions")
        .when(key == "loyal", "Loyal: Active")
        .when((key == "seasonal_new") & F.col("is_new_customer"), "Seasonal & New: New Customers")
        .when(key == "seasonal_new", "Seasonal & New: Seasonal Gifters")
        .when(
            (key == "hibernating") & (F.col("frequency") >= 2), "Hibernating: Former Repeat Buyers"
        )
        .when(key == "hibernating", "Hibernating: One-time Buyers")
    )
    action_map = F.create_map(
        *[x for g, a in ACTION_PLAYBOOK.items() for x in (F.lit(g), F.lit(a["action"]))]
    )
    return out.withColumn("action_group", group).withColumn(
        "recommended_action", action_map[F.col("action_group")]
    )
