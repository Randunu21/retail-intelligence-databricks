"""Leakage-safe training set for the repeat-purchase propensity model (Phase 5).

Question the model answers: **"Will this customer buy again in the next 90 days?"**, asked on
a cutoff date for every customer who has bought at least once before that date.

For one cutoff date `c` (a "snapshot"):

    history (features)                       label window
    [data start ........................ c)  [c ............ c + 90 days)
     everything strictly before c             label = 1 if >= 1 order in here

- **Features** come from the Phase 3 function `build_customer_features(..., as_of=c)`, which
  only reads lines with `invoice_date < c`, plus a few recent-activity counts computed here
  with the same "strictly before c" filter.
- **Label** = 1 if the customer has at least one sales invoice dated in `[c, c + horizon)`.
  The end is exclusive, so a 90-day window is exactly 90 calendar days.
- A sale **on** the cutoff day belongs to the label, never to the features.
- **Population** = customers with a sale before `c` (customers first seen inside the label
  window can't be scored at time `c`: we don't know them yet).

Rolling snapshots (see `plan_cutoffs`): many training cutoffs (one per month), then a later
validation cutoff and a final test cutoff. Every label window must end before the next
split starts, so no training label overlaps the validation period and no validation label
overlaps the test period. The data ends on 2011-12-09, so the test cutoff is the last date
whose 90-day window still fits (2011-12-10 - 90 days = 2011-09-11).

Everything here is a pure function (DataFrames in, DataFrame out) except the two writers at
the bottom; `truncation_check` is the runtime proof of "no leakage" on the real data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from features.customer_features import (
    FEATURE_COLUMNS as PHASE3_FEATURE_COLUMNS,
)
from features.customer_features import (
    SEASONAL_MONTHS,
    build_customer_features,
    known_customer_lines,
)

HORIZON_DAYS = 90
MIN_HISTORY_DAYS = 90  # a training cutoff needs at least this much history before it
ACTIVITY_WINDOWS: tuple[int, ...] = (30, 90, 180, 365)  # days before the cutoff
SPLITS = ("train", "validation", "test")
KEY_COLUMNS = ("customer_id", "cutoff_date")
LABEL_COLUMN = "label"

# ----------------------------------------------------------------------------- features used
# Phase 3 features that go into the model. Left out on purpose:
# - is_tenure_censored: an artefact of where the data starts (almost every customer is
#   "censored" at an early cutoff), not customer behaviour -> would be unstable over time.
# - first/last_purchase_date: raw dates; tenure_days / recency_days carry the same information
#   in a form that means the same thing at every cutoff.
# - preferred_country: 41 values, 91 % UK -> replaced by the is_uk flag.
PHASE3_MODEL_FEATURES: tuple[str, ...] = (
    "recency_days",
    "frequency",
    "monetary",
    "avg_basket_value",
    "distinct_products",
    "returned_value",
    "n_return_invoices",
    "return_rate",
    "tenure_days",
    "n_purchase_days",
    "mean_days_between_purchases",  # NULL for one-day customers (LightGBM handles NULLs)
    "n_countries",
    "seasonal_revenue_share",
)


def activity_feature_names(windows: Sequence[int] = ACTIVITY_WINDOWS) -> list[str]:
    names = []
    for w in windows:
        names += [f"orders_last_{w}d", f"revenue_last_{w}d"]
    return names


MODEL_FEATURES: tuple[str, ...] = (
    *PHASE3_MODEL_FEATURES,
    "is_uk",
    *activity_feature_names(),
    "label_window_q4_days",
)


# ----------------------------------------------------------------------------- split plan


@dataclass(frozen=True)
class SplitPlan:
    """Which cutoff dates are used for which split, and how long the label window is."""

    train_cutoffs: tuple[date, ...]
    validation_cutoff: date
    test_cutoff: date
    horizon_days: int = HORIZON_DAYS
    notes: dict = field(default_factory=dict, compare=False)

    def label_end(self, cutoff: date) -> date:
        """Exclusive end of the label window."""
        return cutoff + timedelta(days=self.horizon_days)

    def cutoffs(self) -> list[tuple[date, str]]:
        """[(cutoff, split)] in date order."""
        return [
            *[(c, "train") for c in self.train_cutoffs],
            (self.validation_cutoff, "validation"),
            (self.test_cutoff, "test"),
        ]

    def to_dict(self) -> dict:
        return {
            "horizon_days": self.horizon_days,
            "train_cutoffs": [str(c) for c in self.train_cutoffs],
            "validation_cutoff": str(self.validation_cutoff),
            "test_cutoff": str(self.test_cutoff),
            "windows": [
                {"cutoff": str(c), "split": s, "label_end_exclusive": str(self.label_end(c))}
                for c, s in self.cutoffs()
            ],
        }


def _first_of_month_on_or_after(d: date) -> date:
    if d.day == 1:
        return d
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


def plan_cutoffs(
    data_start: date,
    data_end_exclusive: date,
    horizon_days: int = HORIZON_DAYS,
    min_history_days: int = MIN_HISTORY_DAYS,
    step_months: int = 1,
) -> SplitPlan:
    """Rolling-snapshot plan, working backwards from the end of the data.

    - test cutoff = the last date whose label window still fits: data_end_exclusive - horizon
    - validation cutoff = test cutoff - horizon (its label window ends exactly at the test cutoff)
    - training cutoffs = the 1st of every `step_months` month, from the first month with
      `min_history_days` of history, up to the last one whose label window ends on or before
      the validation cutoff.
    """
    test = data_end_exclusive - timedelta(days=horizon_days)
    validation = test - timedelta(days=horizon_days)
    c = _first_of_month_on_or_after(data_start + timedelta(days=min_history_days))
    train = []
    while c + timedelta(days=horizon_days) <= validation:
        train.append(c)
        c = _add_months(c, step_months)
    plan = SplitPlan(
        tuple(train),
        validation,
        test,
        horizon_days,
        notes={"data_start": str(data_start), "data_end_exclusive": str(data_end_exclusive)},
    )
    validate_plan(plan, data_start, data_end_exclusive, min_history_days)
    return plan


class LeakageError(ValueError):
    """A split plan or a snapshot would let information from the future into training."""


def validate_plan(
    plan: SplitPlan,
    data_start: date,
    data_end_exclusive: date,
    min_history_days: int = MIN_HISTORY_DAYS,
) -> None:
    """Raise LeakageError if the plan breaks any time-split rule:

    1. cutoffs strictly increasing, at least one training cutoff
    2. every label window lies inside the data (otherwise late labels are wrongly 0)
    3. every training label window ends on/before the validation cutoff
    4. the validation label window ends on/before the test cutoff
    5. every cutoff has at least `min_history_days` of history
    """
    cutoffs = [c for c, _ in plan.cutoffs()]
    problems = []
    if not plan.train_cutoffs:
        problems.append("no training cutoffs")
    if any(b <= a for a, b in zip(cutoffs, cutoffs[1:], strict=False)):
        problems.append(f"cutoffs not strictly increasing: {cutoffs}")
    for c, split in plan.cutoffs():
        if plan.label_end(c) > data_end_exclusive:
            problems.append(
                f"{split} cutoff {c}: label window ends {plan.label_end(c)} after the data "
                f"({data_end_exclusive})"
            )
        if (c - data_start).days < min_history_days:
            problems.append(f"{split} cutoff {c}: less than {min_history_days} days of history")
    for c in plan.train_cutoffs:
        if plan.label_end(c) > plan.validation_cutoff:
            problems.append(
                f"train cutoff {c}: label window ends {plan.label_end(c)} after the "
                f"validation cutoff {plan.validation_cutoff}"
            )
    if plan.label_end(plan.validation_cutoff) > plan.test_cutoff:
        problems.append(
            f"validation label window ends {plan.label_end(plan.validation_cutoff)} after the "
            f"test cutoff {plan.test_cutoff}"
        )
    if problems:
        raise LeakageError("; ".join(problems))


# ----------------------------------------------------------------------------- pieces


def _lit_date(d: date | str):
    return F.to_date(F.lit(str(d)))


def seasonal_days(
    cutoff: date, horizon_days: int = HORIZON_DAYS, months: Sequence[int] = SEASONAL_MONTHS
) -> int:
    """How many days of the label window [cutoff, cutoff + horizon) fall in Q4.

    Known on the cutoff date (it's the calendar, not customer data), so it is a legitimate
    feature: it lets the model learn that a window covering the Christmas peak has a higher
    base rate of repeat orders.
    """
    return sum((cutoff + timedelta(days=i)).month in months for i in range(horizon_days))


def activity_features(
    sales: DataFrame, as_of: date | str, windows: Sequence[int] = ACTIVITY_WINDOWS
) -> DataFrame:
    """Orders and revenue in the last w days before `as_of`, for every w, in ONE aggregation.

    Window for w = [as_of - w days, as_of): strictly before the cutoff, like every feature.
    """
    as_of_col = _lit_date(as_of)
    revenue = F.col("line_revenue").cast("double")
    aggs = []
    for w in windows:
        in_window = F.col("invoice_date") >= F.date_sub(as_of_col, w)
        aggs += [
            F.countDistinct(F.when(in_window, F.col("invoice")))
            .cast("int")
            .alias(f"orders_last_{w}d"),
            F.sum(F.when(in_window, revenue).otherwise(F.lit(0.0))).alias(f"revenue_last_{w}d"),
        ]
    return known_customer_lines(sales, as_of).groupBy("customer_id").agg(*aggs)


def build_labels(
    sales: DataFrame, cutoff: date | str, horizon_days: int = HORIZON_DAYS
) -> DataFrame:
    """customer_id, label_orders: sales invoices per customer in [cutoff, cutoff + horizon).

    Only customers WITH an order in the window appear; `build_snapshot` left-joins this onto
    the population, so everyone else gets label 0.
    """
    start = _lit_date(cutoff)
    end = F.date_add(start, horizon_days)
    return (
        sales.where(
            F.col("customer_id").isNotNull()
            & (F.col("invoice_date") >= start)
            & (F.col("invoice_date") < end)
        )
        .groupBy("customer_id")
        .agg(F.countDistinct("invoice").cast("int").alias("label_orders"))
    )


def snapshot_features(
    sales: DataFrame,
    returns: DataFrame,
    cutoff: date | str,
    data_start: date | str,
    horizon_days: int = HORIZON_DAYS,
    windows: Sequence[int] = ACTIVITY_WINDOWS,
) -> DataFrame:
    """All features for one cutoff (no label). Reads nothing on or after `cutoff`."""
    cutoff_d = date.fromisoformat(str(cutoff))
    base = build_customer_features(sales, returns, cutoff, data_start=data_start)
    return (
        base.join(activity_features(sales, cutoff, windows), "customer_id", "left")
        .withColumn("cutoff_date", _lit_date(cutoff))
        .withColumn("is_uk", (F.col("preferred_country") == F.lit("United Kingdom")).cast("int"))
        .withColumn(
            "label_window_q4_days", F.lit(seasonal_days(cutoff_d, horizon_days)).cast("int")
        )
    )


def build_snapshot(
    sales: DataFrame,
    returns: DataFrame,
    cutoff: date | str,
    split: str,
    data_start: date | str,
    horizon_days: int = HORIZON_DAYS,
    windows: Sequence[int] = ACTIVITY_WINDOWS,
) -> DataFrame:
    """One snapshot: features (before cutoff) + label (90 days from cutoff) for one split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    feats = snapshot_features(sales, returns, cutoff, data_start, horizon_days, windows)
    labels = build_labels(sales, cutoff, horizon_days)
    out = (
        feats.join(labels, "customer_id", "left")
        .fillna({"label_orders": 0})
        .withColumn(LABEL_COLUMN, (F.col("label_orders") > 0).cast("int"))
        .withColumn("split", F.lit(split))
        .withColumn("label_end_date", F.date_add(F.col("cutoff_date"), horizon_days))
    )
    return out.select(*output_columns(windows))


def output_columns(windows: Sequence[int] = ACTIVITY_WINDOWS) -> list[str]:
    """Column order of the training-set table."""
    extra = [c for c in PHASE3_FEATURE_COLUMNS if c not in ("customer_id", "as_of_date")]
    return [
        "customer_id",
        "cutoff_date",
        "split",
        *extra,
        "is_uk",
        *activity_feature_names(windows),
        "label_window_q4_days",
        "label_end_date",
        "label_orders",
        LABEL_COLUMN,
    ]


def build_training_set(
    sales: DataFrame,
    returns: DataFrame,
    plan: SplitPlan,
    data_start: date | str,
    windows: Sequence[int] = ACTIVITY_WINDOWS,
) -> DataFrame:
    """All snapshots of the plan, unioned (handy for tests; the notebook writes per cutoff)."""
    frames = [
        build_snapshot(sales, returns, c, s, data_start, plan.horizon_days, windows)
        for c, s in plan.cutoffs()
    ]
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    return out


# ----------------------------------------------------------------------------- checks


def _rounded(df: DataFrame, columns: Iterable[str], digits: int = 6) -> DataFrame:
    """Round doubles so that sums done in a different order still compare equal."""
    types = dict(df.dtypes)
    return df.select(
        *[F.round(F.col(c), digits).alias(c) if types[c] == "double" else F.col(c) for c in columns]
    )


def truncation_check(
    sales: DataFrame,
    returns: DataFrame,
    cutoff: date | str,
    data_start: date | str,
    horizon_days: int = HORIZON_DAYS,
    windows: Sequence[int] = ACTIVITY_WINDOWS,
) -> dict:
    """Runtime proof that features don't look at the future.

    Builds the features twice: from the full tables, and from tables physically cut at the
    cutoff (`invoice_date < cutoff`). If any feature read an event on/after the cutoff, the
    two results would differ. Returns row counts and the number of rows that differ in each
    direction (both must be 0).
    """
    full = snapshot_features(sales, returns, cutoff, data_start, horizon_days, windows)
    cut = F.col("invoice_date") < _lit_date(cutoff)
    truncated = snapshot_features(
        sales.where(cut), returns.where(cut), cutoff, data_start, horizon_days, windows
    )
    columns = [c for c in full.columns]
    a, b = _rounded(full, columns), _rounded(truncated, columns)
    return {
        "cutoff": str(cutoff),
        "rows_full": a.count(),
        "rows_truncated": b.count(),
        "only_in_full": a.exceptAll(b).count(),
        "only_in_truncated": b.exceptAll(a).count(),
    }


def label_summary(training_set: DataFrame) -> DataFrame:
    """Per cutoff: split, customers, positives, base rate (sorted by date)."""
    return (
        training_set.groupBy("cutoff_date", "split")
        .agg(
            F.count(F.lit(1)).alias("customers"),
            F.sum(LABEL_COLUMN).alias("positives"),
            F.round(F.avg(LABEL_COLUMN), 4).alias("positive_rate"),
            F.first("label_window_q4_days").alias("label_window_q4_days"),
        )
        .orderBy("cutoff_date")
    )


def check_training_set(training_set: DataFrame) -> dict:
    """Structural checks the notebook asserts on: unique keys, label in {0, 1}, no NULL
    labels, and every customer had a sale before its cutoff (recency >= 1)."""
    r = training_set.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct(*KEY_COLUMNS).alias("keys"),
        F.sum(F.when(F.col(LABEL_COLUMN).isNull(), 1).otherwise(0)).alias("null_labels"),
        F.sum(F.when(~F.col(LABEL_COLUMN).isin(0, 1), 1).otherwise(0)).alias("bad_labels"),
        F.min("recency_days").alias("min_recency"),
        F.sum(F.when(F.col("last_purchase_date") >= F.col("cutoff_date"), 1).otherwise(0)).alias(
            "purchases_on_or_after_cutoff"
        ),
    ).first()
    return {
        "rows": r["rows"],
        "duplicate_keys": r["rows"] - r["keys"],
        "null_labels": r["null_labels"],
        "bad_labels": r["bad_labels"],
        "min_recency_days": r["min_recency"],
        "purchases_on_or_after_cutoff": r["purchases_on_or_after_cutoff"],
    }


# ----------------------------------------------------------------------------- writers


def cutoff_predicate(cutoff: date | str) -> str:
    return f"cutoff_date = DATE'{cutoff}'"


def write_training_snapshot(df: DataFrame, table_name: str, cutoff: date | str) -> None:
    """Replace one cutoff's rows (idempotent re-runs; other cutoffs untouched).

    The first write creates the table; later writes use `replaceWhere` on the cutoff.
    """
    out = df.withColumn("built_at", F.current_timestamp())
    writer = out.write.format("delta").mode("overwrite")
    if df.sparkSession.catalog.tableExists(table_name):
        writer = writer.option("replaceWhere", cutoff_predicate(cutoff))
    writer.saveAsTable(table_name)


def stale_cutoffs_sql(table_name: str, plan: SplitPlan) -> str:
    """DELETE rows whose cutoff is no longer in the plan (after a plan change)."""
    keep = ", ".join(f"DATE'{c}'" for c, _ in plan.cutoffs())
    return f"DELETE FROM {table_name} WHERE cutoff_date NOT IN ({keep})"
