"""The Silver + Gold builds as a list of steps the benchmark harness can time one by one.

Why not time the Lakeflow pipeline itself?
- A pipeline update has a fixed start-up cost (planning, compute, event logging) that is
  large next to ~1M rows and varies from run to run; per-table durations in the UI are
  rounded and include waiting for upstream tables.
- One update = one sample. Getting 5 samples per variant means 5 full updates per variant.
- We can't change a single thing (e.g. one join hint) without editing and re-deploying it.

So the harness rebuilds the SAME logic in a notebook, step by step: the Silver functions
from `transformations.silver`, the Gold DataFrame functions from `transformations.gold` and
the Gold SQL read straight from the pipeline's `.sql` files (see `pipeline_sql`). Outputs go
to `retail.bench.<prefix>__<logical name>`, so production tables are never touched.
Section 3 of the notebook proves the baseline copies match production row for row.

Every table is referred to by a LOGICAL name ("silver_sales", "gold_fact_sales", ...).
A `BuildContext.names` mapping turns logical names into real tables (or temp views in
tests), so the same step runs against production, x1 or x10 benchmark copies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from common.tables import (
    BRONZE_TRANSACTIONS,
    GOLD_CUSTOMER_COHORT_RETENTION,
    GOLD_DAILY_SALES,
    GOLD_DIM_CUSTOMER,
    GOLD_DIM_DATE,
    GOLD_DIM_PRODUCT,
    GOLD_FACT_SALES,
    GOLD_MONTHLY_COUNTRY_SALES,
    GOLD_PRODUCT_PERFORMANCE,
    SILVER_RETURNS,
    SILVER_SALES,
    bench_table,
)
from optimisation.pipeline_sql import load_query, retarget
from transformations.gold import (
    build_customer_cohort_retention,
    build_dim_customer,
    build_dim_product,
)
from transformations.silver import (
    RETURNS_DROP_RULES,
    SALES_DROP_RULES,
    apply_drop_rules,
    stage_silver,
    to_silver_output,
)

# Logical name -> production table. The SQL files contain the production names, so this is
# also the "from" side when SQL is retargeted to benchmark tables.
PROD_NAMES: dict[str, str] = {
    "bronze": BRONZE_TRANSACTIONS,
    "silver_sales": SILVER_SALES,
    "silver_returns": SILVER_RETURNS,
    "gold_dim_date": GOLD_DIM_DATE,
    "gold_fact_sales": GOLD_FACT_SALES,
    "gold_dim_product": GOLD_DIM_PRODUCT,
    "gold_dim_customer": GOLD_DIM_CUSTOMER,
    "gold_daily_sales": GOLD_DAILY_SALES,
    "gold_monthly_country_sales": GOLD_MONTHLY_COUNTRY_SALES,
    "gold_product_performance": GOLD_PRODUCT_PERFORMANCE,
    "gold_customer_cohort_retention": GOLD_CUSTOMER_COHORT_RETENTION,
}


@dataclass(frozen=True)
class BuildContext:
    """Everything a step needs: a session, where each logical table lives, where SQL is."""

    spark: SparkSession
    names: Mapping[str, str]
    pipeline_dir: Path

    def read(self, logical: str) -> DataFrame:
        return self.spark.read.table(self.names[logical])

    @property
    def mapping(self) -> dict[str, str]:
        """Production table name -> this context's table (for retargeting SQL)."""
        return {PROD_NAMES[k]: v for k, v in self.names.items() if k in PROD_NAMES}

    def query(self, file_name: str) -> str:
        """A pipeline SQL file's query text, retargeted to this context's tables."""
        return load_query(self.pipeline_dir, file_name, self.mapping)

    def sql_file(self, file_name: str) -> DataFrame:
        """Run a pipeline SQL file's query with production names swapped for `names`."""
        return self.spark.sql(self.query(file_name))

    def sql_text(self, sql: str) -> DataFrame:
        """Run SQL written with PRODUCTION table names (e.g. an optimised variant of a
        pipeline query), retargeted to this context's tables."""
        return self.spark.sql(retarget(sql, self.mapping))


@dataclass(frozen=True)
class BuildStep:
    """One table build. `build(ctx)` returns the DataFrame; the harness writes and times it.

    A build may also write its own output(s) and return None (e.g. one step that produces
    silver_sales AND silver_returns); the harness then only times it.
    """

    name: str  # logical name of the OUTPUT table, also the step's name in benchmark_runs
    layer: str  # "silver" | "gold"
    build: Callable[[BuildContext], DataFrame]
    inputs: tuple[str, ...] = field(default=())  # logical names read (documentation + checks)


def bench_names(prefix: str, bronze_table: str = BRONZE_TRANSACTIONS) -> dict[str, str]:
    """Logical name -> benchmark table for one variant. Bronze is read, never rebuilt."""
    names = {logical: bench_table(prefix, logical) for logical in PROD_NAMES if logical != "bronze"}
    names["bronze"] = bronze_table
    return names


# ----------------------------------------------------------------------------------------
# Baseline steps: an exact notebook copy of what the pipeline does today (Phase 1).
# Note: in the pipeline `silver_staged` is a TEMPORARY VIEW read by both sales_validated and
# returns_validated, i.e. the expensive staging (typing + window de-dup over all Bronze rows)
# is computed twice. The baseline keeps that on purpose; step 2.4 measures fixing it.
# ----------------------------------------------------------------------------------------
def _silver_sales(ctx: BuildContext) -> DataFrame:
    return to_silver_output(apply_drop_rules(stage_silver(ctx.read("bronze")), SALES_DROP_RULES))


def _silver_returns(ctx: BuildContext) -> DataFrame:
    staged = stage_silver(ctx.read("bronze")).filter("is_cancellation")
    return to_silver_output(apply_drop_rules(staged, RETURNS_DROP_RULES))


# Gold queries rewritten in the pipeline in step 2.6. The baseline keeps measuring their
# Phase 1 versions, frozen in src/optimisation/phase1_sql/ (see the README there).
PHASE1_SQL_DIR = Path(__file__).resolve().parent / "phase1_sql"
FROZEN_PHASE1_FILES = frozenset({"gold_daily_sales.sql", "gold_monthly_country_sales.sql"})


def phase1_query(ctx: BuildContext, file_name: str) -> str:
    """A Gold query as it was in Phase 1 (frozen copy if rewritten since), retargeted."""
    if file_name in FROZEN_PHASE1_FILES:
        return load_query(PHASE1_SQL_DIR, file_name, ctx.mapping)
    return ctx.query(file_name)


def _sql_step(file_name: str) -> Callable[[BuildContext], DataFrame]:
    def _build(ctx: BuildContext) -> DataFrame:
        return ctx.spark.sql(phase1_query(ctx, file_name))

    _build.__name__ = f"sql:{file_name}"
    return _build


BASELINE_STEPS: tuple[BuildStep, ...] = (
    BuildStep("silver_sales", "silver", _silver_sales, ("bronze",)),
    BuildStep("silver_returns", "silver", _silver_returns, ("bronze",)),
    BuildStep("gold_dim_date", "gold", _sql_step("gold_dim_date.sql"), ("silver_sales",)),
    BuildStep("gold_fact_sales", "gold", _sql_step("gold_fact_sales.sql"), ("silver_sales",)),
    BuildStep(
        "gold_dim_product",
        "gold",
        lambda ctx: build_dim_product(ctx.read("silver_sales")),
        ("silver_sales",),
    ),
    BuildStep(
        "gold_dim_customer",
        "gold",
        lambda ctx: build_dim_customer(ctx.read("silver_sales")),
        ("silver_sales",),
    ),
    BuildStep(
        "gold_daily_sales",
        "gold",
        _sql_step("gold_daily_sales.sql"),
        ("gold_dim_date", "gold_fact_sales"),
    ),
    BuildStep(
        "gold_monthly_country_sales",
        "gold",
        _sql_step("gold_monthly_country_sales.sql"),
        ("gold_fact_sales", "gold_dim_date"),
    ),
    BuildStep(
        "gold_product_performance",
        "gold",
        _sql_step("gold_product_performance.sql"),
        ("gold_dim_product", "gold_fact_sales", "silver_returns"),
    ),
    BuildStep(
        "gold_customer_cohort_retention",
        "gold",
        lambda ctx: build_customer_cohort_retention(
            ctx.read("gold_fact_sales"), ctx.read("gold_dim_customer")
        ),
        ("gold_fact_sales", "gold_dim_customer"),
    ),
)


def check_step_order(steps: Sequence[BuildStep]) -> None:
    """Fail fast if a step reads a table that no earlier step (or Bronze) produces."""
    available = {"bronze"}
    for step in steps:
        missing = [i for i in step.inputs if i not in available]
        if missing:
            raise ValueError(f"Step {step.name!r} needs {missing} before it runs")
        available.add(step.name)
