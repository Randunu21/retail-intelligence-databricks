"""Phase 2.3: data layout of gold.fact_sales — partitioning vs Z-order vs liquid clustering.

The question isn't "which layout builds fastest" but "which layout makes the queries people
actually run read less data, and what does it cost to maintain?". So each variant is
measured twice:

1. **Build cost**: write fact_sales with that layout (+ its maintenance, OPTIMIZE), timed
   by `run_benchmark` (write and OPTIMIZE recorded separately).
2. **Read cost**: three typical queries (`LAYOUT_QUERIES`), timed by `run_read_benchmark`,
   plus files read / pruned and bytes read from `system.query.history`.

Variants (all read the same baseline Silver, so only fact_sales differs):

| experiment   | layout                                     | why it's in the test                      |
|--------------|--------------------------------------------|-------------------------------------------|
| baseline     | Delta defaults                             | reference                                 |
| part_date    | PARTITIONED BY (date_key)                  | the classic choice, and the classic       |
|              |                                            | over-partitioning mistake: ~600 days      |
| zorder_32mb  | OPTIMIZE ... ZORDER BY (date_key,          | older multi-column layout; needs explicit |
|              | customer_key), 32 MB target files          | OPTIMIZE and rewrites everything each time|
| lc_auto      | CLUSTER BY (date_key, customer_key)        | liquid clustering with default file sizes |
|              | + OPTIMIZE                                 |                                           |
| lc_32mb      | as lc_auto, 32 MB target files             | same keys, files small enough to skip     |

Why the 32 MB variants: Databricks sizes files automatically from the table size. A
~137 MB table can end up as ONE file (the x10 baseline did), and a single file can never
be skipped. `delta.targetFileSize` is an ordinary table property (no Spark config needed,
so it works on serverless). Comparing lc_auto with lc_32mb separates the effect of
*sorting* from the effect of *file size*.

Clustering keys: date_key (date-range reports, monthly marts, incremental loads) and
customer_key (customer lookups, segmentation features in Phase 3). Order matters less for
liquid clustering than for Z-order, but low-cardinality-first is the usual guidance.
"""

from __future__ import annotations

from collections.abc import Mapping

from pyspark.sql import SparkSession

from common.tables import bench_table
from optimisation.benchmark import DEFAULT_TABLE_PROPERTIES, Variant, WriteSpec

FACT_STEP = "gold_fact_sales"
CLUSTER_KEYS: tuple[str, ...] = ("date_key", "customer_key")
TARGET_FILE_SIZE = "32mb"

# Typical queries against the fact table. `{table}` is filled in per variant.
LAYOUT_QUERIES: dict[str, str] = {
    # A monthly report: one month of a two-year table (date-range filter).
    "month_by_country": (
        "SELECT country, SUM(line_revenue) AS revenue, COUNT(*) AS lines FROM {table} "
        "WHERE date_key BETWEEN 20111101 AND 20111130 GROUP BY country"
    ),
    # A customer lookup (customer service screen, feature lookup): one customer's lines.
    # 13085 is a real customer from the first rows of the source file.
    "one_customer": (
        "SELECT invoice, invoice_ts, stock_code, quantity, line_revenue FROM {table} "
        "WHERE customer_key = 13085"
    ),
    # Control: no filter, reads everything. Layout can't help; it should NOT get faster.
    "all_by_country": "SELECT country, SUM(line_revenue) AS revenue FROM {table} GROUP BY country",
}


def optimize(spark: SparkSession, table: str) -> None:
    """Liquid clustering: OPTIMIZE clusters the data by the table's clustering keys."""
    spark.sql(f"OPTIMIZE {table}").collect()


def optimize_zorder(spark: SparkSession, table: str) -> None:
    """Z-order: OPTIMIZE ... ZORDER BY sorts along a space-filling curve of the columns."""
    spark.sql(f"OPTIMIZE {table} ZORDER BY ({', '.join(CLUSTER_KEYS)})").collect()


def _props(**extra: str) -> dict[str, str]:
    return {**DEFAULT_TABLE_PROPERTIES, **extra}


def layout_variants(base_names: Mapping[str, str], scale: str) -> list[Variant]:
    """The five fact_sales layouts. Every variant reads the same (baseline) Silver tables;
    only its fact table is separate: retail.bench.<scale>_<experiment>__gold_fact_sales."""

    def names(experiment: str) -> dict[str, str]:
        return {**base_names, FACT_STEP: bench_table(f"{scale}_{experiment}", FACT_STEP)}

    small = _props(**{"delta.targetFileSize": TARGET_FILE_SIZE})
    return [
        Variant("baseline", names("layout_baseline"), notes="Delta defaults"),
        Variant(
            "part_date",
            names("part_date"),
            {FACT_STEP: WriteSpec(partition_by=("date_key",))},
            notes="partitioned by day",
        ),
        Variant(
            "zorder_32mb",
            names("zorder_32mb"),
            {FACT_STEP: WriteSpec(properties=small)},
            post_write={FACT_STEP: optimize_zorder},
            notes=f"OPTIMIZE ZORDER BY {CLUSTER_KEYS}, targetFileSize {TARGET_FILE_SIZE}",
        ),
        Variant(
            "lc_auto",
            names("lc_auto"),
            {FACT_STEP: WriteSpec(cluster_by=CLUSTER_KEYS)},
            post_write={FACT_STEP: optimize},
            notes="liquid clustering + OPTIMIZE, default file size",
        ),
        Variant(
            "lc_32mb",
            names("lc_32mb"),
            {FACT_STEP: WriteSpec(cluster_by=CLUSTER_KEYS, properties=small)},
            post_write={FACT_STEP: optimize},
            notes=f"liquid clustering + OPTIMIZE, targetFileSize {TARGET_FILE_SIZE}",
        ),
    ]
