"""Phase 2.4: compute-side optimisations — shuffles, joins, and an alternative to caching.

Two experiments, both measured with the same harness as before:

A. **Silver: prepare Bronze once, not twice.** The pipeline's `silver_staged` is a temporary
   VIEW read by both `sales` and `returns`, so the typing + window de-dup over every Bronze
   row runs twice (60 % of the x10 build is Silver). Variants of one combined "silver" step
   (writes silver_sales AND silver_returns, timed together):

   | experiment     | how staging is shared                                           |
   |----------------|-----------------------------------------------------------------|
   | baseline       | computed twice, like the pipeline                               |
   | staged_table   | written ONCE to a Delta table, both outputs read it             |
   | cached         | `df.cache()` once, both outputs read it (serverless accepted     |
   |                | the call in the probe; this measures whether it actually helps) |
   | agg_dedup      | computed twice, but de-dup by GROUP BY instead of a window      |

   Why `agg_dedup`: `row_number() OVER (PARTITION BY 8 columns ORDER BY _source_file)`
   shuffles every row with ALL its columns and then sorts each partition. A GROUP BY on the
   same 8 columns can aggregate partially before the shuffle (duplicates collapse early) and
   needs no sort. The kept row is the same one: the copy from the earliest `_source_file`.

B. **Gold joins.** Which join strategy does Spark pick, and does forcing one help?

   | experiment    | what changes                                                       |
   |---------------|--------------------------------------------------------------------|
   | baseline      | optimiser decides (AQE on serverless can't be switched off)         |
   | broadcast     | `BROADCAST` hints on the small side of each join                    |
   | no_broadcast  | `MERGE` hints: forces sort-merge joins = what auto-broadcast saves  |
   | rewrite       | fewer wide operations: daily_sales aggregates the fact BEFORE the   |
   |               | join (10M rows -> 739); monthly sales takes year/month from         |
   |               | invoice_ts instead of joining dim_date                              |

   `autoBroadcastJoinThreshold` can't be set on serverless (probe), so hints are the only
   lever. A broadcast hint on the LEFT side of a LEFT JOIN is ignored by Spark (the
   preserved side can't be the broadcast side); daily_sales shows that on purpose.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Mapping
from contextlib import redirect_stdout

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from common.tables import bench_table
from optimisation.benchmark import Variant, WriteSpec, write_table
from optimisation.builds import BuildContext, BuildStep
from transformations.gold import build_customer_cohort_retention
from transformations.silver import (
    BUSINESS_COLUMNS,
    RETURNS_DROP_RULES,
    SALES_DROP_RULES,
    add_derived_columns,
    apply_drop_rules,
    stage_silver,
    standardise,
    to_silver_output,
)

# ----------------------------------------------------------------------------------------
# A. Silver
# ----------------------------------------------------------------------------------------
SILVER_STEP = "silver"  # combined step; its "table" for DESCRIBE DETAIL is silver_sales


def stage_silver_agg(bronze: DataFrame) -> DataFrame:
    """Same rows/columns as `stage_silver` after keeping `_dup_rank = 1`, de-duplicated by
    GROUP BY instead of a window. `_dup_rank` is 1 for every row (duplicates are gone).

    Derived columns depend only on business columns, so they're added after the GROUP BY:
    fewer columns go through the shuffle. Of the copies, the one from the earliest
    `_source_file` is kept (`min`), with that copy's `_ingested_at` (`min_by`).
    """
    deduped = (
        standardise(bronze)
        .groupBy(*BUSINESS_COLUMNS)
        .agg(
            F.min("_source_file").alias("_source_file"),
            F.min_by("_ingested_at", "_source_file").alias("_ingested_at"),
        )
    )
    return add_derived_columns(deduped).withColumn("_dup_rank", F.lit(1))


def _sales(staged: DataFrame) -> DataFrame:
    return to_silver_output(apply_drop_rules(staged, SALES_DROP_RULES))


def _returns(staged: DataFrame) -> DataFrame:
    return to_silver_output(apply_drop_rules(staged.filter("is_cancellation"), RETURNS_DROP_RULES))


def _write(ctx: BuildContext, logical: str, df: DataFrame) -> None:
    write_table(df, ctx.names[logical], WriteSpec())


def silver_separate(ctx: BuildContext) -> None:
    """Baseline: staging computed separately for each output (what the pipeline does)."""
    _write(ctx, "silver_sales", _sales(stage_silver(ctx.read("bronze"))))
    _write(ctx, "silver_returns", _returns(stage_silver(ctx.read("bronze"))))


def silver_staged_table(ctx: BuildContext) -> None:
    """Staging written once to Delta; both outputs read the stored result."""
    _write(ctx, "silver_staged", stage_silver(ctx.read("bronze")))
    staged = ctx.read("silver_staged")
    _write(ctx, "silver_sales", _sales(staged))
    _write(ctx, "silver_returns", _returns(staged))


def silver_cached(ctx: BuildContext) -> None:
    """Staging cached in memory once; both outputs read the cached DataFrame."""
    staged = stage_silver(ctx.read("bronze")).cache()
    try:
        _write(ctx, "silver_sales", _sales(staged))
        _write(ctx, "silver_returns", _returns(staged))
    finally:
        staged.unpersist()


def silver_agg_dedup(ctx: BuildContext) -> None:
    """Like the baseline (computed per output), but de-dup by GROUP BY instead of a window."""
    _write(ctx, "silver_sales", _sales(stage_silver_agg(ctx.read("bronze"))))
    _write(ctx, "silver_returns", _returns(stage_silver_agg(ctx.read("bronze"))))


SILVER_COMBINED_STEPS: tuple[BuildStep, ...] = (
    BuildStep(SILVER_STEP, "silver", silver_separate, ("bronze",)),
)
SILVER_BUILDS = {
    "baseline": silver_separate,
    "staged_table": silver_staged_table,
    "cached": silver_cached,
    "agg_dedup": silver_agg_dedup,
}


def silver_variants(bronze_table: str, scale: str) -> list[Variant]:
    """One variant per SILVER_BUILDS entry, each writing to its own bench tables."""
    variants = []
    for experiment, build in SILVER_BUILDS.items():
        prefix = f"{scale}_silver_{experiment}"
        names = {
            "bronze": bronze_table,
            "silver_sales": bench_table(prefix, "silver_sales"),
            "silver_returns": bench_table(prefix, "silver_returns"),
            "silver_staged": bench_table(prefix, "silver_staged"),
        }
        names[SILVER_STEP] = names["silver_sales"]
        variants.append(Variant(experiment, names, step_overrides={SILVER_STEP: build}))
    return variants


# ----------------------------------------------------------------------------------------
# B. Gold joins
# ----------------------------------------------------------------------------------------
GOLD_JOIN_STEPS: tuple[str, ...] = (
    "gold_daily_sales",
    "gold_monthly_country_sales",
    "gold_product_performance",
    "gold_customer_cohort_retention",
)
# step -> (pipeline SQL file, which SELECT gets the hint (1-based), aliases of small sides)
_HINT_TARGETS: dict[str, tuple[str, int, str]] = {
    "gold_daily_sales": ("gold_daily_sales.sql", 1, "d"),  # d is the LEFT side: ignored
    "gold_monthly_country_sales": ("gold_monthly_country_sales.sql", 1, "d"),
    "gold_product_performance": ("gold_product_performance.sql", 3, "s, r"),
}


def add_hint(query: str, hint: str, occurrence: int = 1) -> str:
    """Insert `/*+ hint */` after the n-th SELECT keyword of `query`."""
    matches = list(re.finditer(r"\bSELECT\b", query, flags=re.IGNORECASE))
    if not 1 <= occurrence <= len(matches):
        raise ValueError(f"query has {len(matches)} SELECTs, can't hint number {occurrence}")
    pos = matches[occurrence - 1].end()
    return f"{query[:pos]} /*+ {hint} */{query[pos:]}"


def _hinted(step: str, strategy: str):
    file_name, occurrence, aliases = _HINT_TARGETS[step]

    def _build(ctx: BuildContext) -> DataFrame:
        return ctx.spark.sql(add_hint(ctx.query(file_name), f"{strategy}({aliases})", occurrence))

    return _build


def _cohort(hint: str):
    def _build(ctx: BuildContext) -> DataFrame:
        return build_customer_cohort_retention(
            ctx.read("gold_fact_sales"), ctx.read("gold_dim_customer"), join_hint=hint
        )

    return _build


# daily_sales, rewritten: aggregate 10M fact rows down to one row per date_key FIRST, then
# left-join the 739-row calendar. Same columns, types and values as the pipeline query
# (an invoice belongs to exactly one day, so per-day DISTINCT counts are unchanged).
DAILY_SALES_PREAGG_SQL = """
WITH f AS (
  SELECT
    date_key,
    SUM(line_revenue)                                                    AS revenue,
    COUNT(DISTINCT invoice)                                              AS invoices,
    SUM(quantity)                                                        AS units,
    COUNT(line_id)                                                       AS lines,
    COUNT(DISTINCT CASE WHEN customer_key <> -1 THEN customer_key END)   AS customers
  FROM retail.gold.fact_sales
  GROUP BY date_key
)
SELECT
  d.date_key,
  d.date,
  d.year_month,
  d.day_name,
  d.is_weekend,
  COALESCE(f.revenue, 0)                          AS revenue,
  COALESCE(f.invoices, 0)                         AS invoices,
  COALESCE(f.units, 0)                            AS units,
  COALESCE(f.lines, 0)                            AS lines,
  COALESCE(f.customers, 0)                        AS customers,
  ROUND(try_divide(f.revenue, f.invoices), 2)     AS avg_basket_value
FROM retail.gold.dim_date AS d
LEFT JOIN f
  ON f.date_key = d.date_key
"""

# monthly_country_sales, rewritten: year / month / year_month come from the fact's own
# invoice_ts (dim_date derives them from the same date), so the join disappears.
MONTHLY_COUNTRY_NOJOIN_SQL = """
WITH monthly AS (
  SELECT
    date_format(f.invoice_ts, 'yyyy-MM')                                  AS year_month,
    year(f.invoice_ts)                                                    AS year,
    month(f.invoice_ts)                                                   AS month,
    f.country,
    SUM(f.line_revenue)                                                   AS revenue,
    COUNT(DISTINCT f.invoice)                                             AS invoices,
    SUM(f.quantity)                                                       AS units,
    COUNT(DISTINCT CASE WHEN f.customer_key <> -1 THEN f.customer_key END) AS customers
  FROM retail.gold.fact_sales AS f
  GROUP BY 1, 2, 3, f.country
)
SELECT
  *,
  ROUND(revenue * 100 / SUM(revenue) OVER (PARTITION BY year_month), 3) AS revenue_share_pct
FROM monthly
"""

GOLD_OVERRIDES: dict[str, dict[str, object]] = {
    "baseline": {},
    "broadcast": {
        **{s: _hinted(s, "BROADCAST") for s in _HINT_TARGETS},
        "gold_customer_cohort_retention": _cohort("broadcast"),
    },
    "no_broadcast": {
        **{s: _hinted(s, "MERGE") for s in _HINT_TARGETS},
        "gold_customer_cohort_retention": _cohort("merge"),
    },
    "rewrite": {
        "gold_daily_sales": lambda ctx: ctx.sql_text(DAILY_SALES_PREAGG_SQL),
        "gold_monthly_country_sales": lambda ctx: ctx.sql_text(MONTHLY_COUNTRY_NOJOIN_SQL),
    },
}


def gold_variants(base_names: Mapping[str, str], scale: str) -> list[Variant]:
    """Gold join variants: upstream tables = the baseline's; the 4 join steps get own tables."""
    variants = []
    for experiment, overrides in GOLD_OVERRIDES.items():
        names = dict(base_names)
        for step in GOLD_JOIN_STEPS:
            names[step] = bench_table(f"{scale}_gold_{experiment}", step)
        variants.append(Variant(experiment, names, step_overrides=overrides))
    return variants


# ----------------------------------------------------------------------------------------
# Plan inspection (evidence for "which join / how many shuffles")
# ----------------------------------------------------------------------------------------
# One block per numbered node in the formatted plan's details: "(7) Name ...body...".
_NODE_BLOCK = re.compile(r"^\((\d+)\) (\w+)(.*?)(?=^\(\d+\) |\Z)", flags=re.MULTILINE | re.DOTALL)
_JOIN_TYPES = ("BroadcastHashJoin", "SortMergeJoin", "ShuffledHashJoin",
               "BroadcastNestedLoopJoin", "CartesianProduct")  # fmt: skip


def plan_text(df: DataFrame) -> str:
    """The formatted physical plan as a string (`explain` prints; we capture it)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        df.explain(mode="formatted")
    return buf.getvalue()


def plan_summary(plan: str) -> dict[str, object]:
    """Count join strategies, shuffles and broadcasts in a formatted physical plan.

    Each numbered node's details block is read once. Data movement is named differently by
    the two engines:
    - classic Spark: `Exchange` (a shuffle) and `BroadcastExchange` (a broadcast);
    - Photon (serverless): every movement is a Sink -> `PhotonShuffleMapStage` -> Source
      triple, and the MapStage's arguments say what it is: `EXECUTOR_BROADCAST` = a
      broadcast, anything else (e.g. `ENSURE_REQUIREMENTS`) = a shuffle. Counting the
      MapStage counts each movement once.
    `runtime_filter`: a broadcast build side also prunes the big side's scan at run time
    (`hashedrelationcontains(...)` in OptionalDataFilters), a free bonus of broadcasting.
    """
    joins: Counter[str] = Counter()
    shuffles = broadcasts = 0
    for _, name, body in _NODE_BLOCK.findall(plan):
        joins.update(j for j in _JOIN_TYPES if name.endswith(j))
        if name == "PhotonShuffleMapStage":
            if "EXECUTOR_BROADCAST" in body:
                broadcasts += 1
            else:
                shuffles += 1
        elif name == "BroadcastExchange":
            broadcasts += 1
        elif name in ("Exchange", "ShuffleExchange"):
            shuffles += 1
    return {
        "joins": ", ".join(f"{k}×{v}" for k, v in sorted(joins.items())) or "-",
        "shuffles": shuffles,
        "broadcasts": broadcasts,
        "runtime_filter": "hashedrelationcontains" in plan,
        "adaptive": "AdaptiveSparkPlan" in plan,
    }
