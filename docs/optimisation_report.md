# Pipeline optimisation report (Phase 2)

> Every number in this report comes from `retail.bench.benchmark_runs` (real runs on
> Databricks Free Edition, serverless). `<<PLACEHOLDER>>` = not measured yet.
> Where a change made no measurable difference, the table says so.

## 1. Setup

| Item | Value |
|---|---|
| Platform | Databricks Free Edition, serverless notebooks (Spark `<<SPARK_VERSION>>`) |
| Data (x1, real) | UCI Online Retail II: Bronze 1,067,371 rows → Silver sales 1,003,357 / returns 17,915 |
| Data (x10, synthetic) | `<<X10_BRONZE_ROWS>>` Bronze rows: the real data replicated 10× with shifted customer/invoice keys (see §3). Labelled synthetic everywhere it's used. |
| What is timed | Rebuild of each Silver/Gold table with the pipeline's own code (`CREATE OR REPLACE`), wall clock on the driver |
| Runs per measurement | 1 warm-up (discarded) + 5 measured at x1, 3 at x10; median reported |
| Noise rule | A change counts only if the min–max ranges of before and after don't overlap |
| Cache state | Warm (serverless can't clear its disk cache, so all variants are compared warm) |

### Free Edition feature check (probe, notebook §1)

| Feature | Result | Consequence |
|---|---|---|
| `df.cache()` | Call runs without error (returned 10) | Contrary to the docs, the call is accepted. Whether data is actually cached on serverless is unverified; tested by timing in step 2.4 |
| `CACHE TABLE ... AS SELECT` | Error `NOT_SUPPORTED_WITH_SERVERLESS` | SQL caching unavailable |
| `spark.sql.shuffle.partitions` | Settable (default `auto`) | Only shuffle knob available |
| `spark.sql.autoBroadcastJoinThreshold` | Error `CONFIG_NOT_AVAILABLE` | Broadcast is controlled with join hints instead |
| `spark.sql.adaptive.enabled` (AQE) | Error `CONFIG_NOT_AVAILABLE` | AQE is managed by the platform (can't be turned off or tuned) |
| Liquid clustering + `OPTIMIZE` | Works (`clusteringColumns = k`) | Usable |
| Partitioned tables | Works (3 files for 3 partition values) | Usable for the comparison |
| Change Data Feed | Works (`insert` 2, `update_preimage` 1, `update_postimage` 1) | Incremental MERGE + CDF possible |
| `noop` write format | Works | Read queries can be benchmarked without writing |
| `system.query.history` | Readable (286 statements in the last day) | Files read / pruned per query available for layout evidence |
| Predictive optimisation | `ENABLE` (inherited from the metastore) | Platform may run `OPTIMIZE`/`VACUUM` on managed tables in the background; benchmark tables are rebuilt each run, and file counts are recorded per run |

## 2. Baseline (x1, Phase 1 logic unchanged)

Benchmark copies equal production: **all 10 tables match** (row count + order-independent checksum of every value).
Run `1dea80f3-2ee2-4db2-a73e-be81433918ab`, 2026-11-19, 1 warm-up + 5 measured runs per step.

| Step | Median s | Min–max s | Stdev s | Rows | Files | Size MB |
|---|---:|---:|---:|---:|---:|---:|
| silver_sales | 6.664 | 6.590–7.051 | 0.191 | 1,003,357 | 1 | 6.22 |
| silver_returns | 3.231 | 3.136–4.373 | 0.525 | 17,915 | 1 | 0.25 |
| gold_dim_date | 2.616 | 2.486–2.737 | 0.113 | 739 | 1 | 0.00 |
| gold_fact_sales | 2.774 | 2.651–2.836 | 0.076 | 1,003,357 | 1 | 12.37 |
| gold_dim_product | 2.907 | 2.773–3.185 | 0.156 | 4,725 | 1 | 0.07 |
| gold_dim_customer | 2.768 | 2.711–2.958 | 0.098 | 5,853 | 2 | 0.09 |
| gold_daily_sales | 2.861 | 2.700–3.025 | 0.128 | 739 | 1 | 0.02 |
| gold_monthly_country_sales | 3.069 | 2.834–4.338 | 0.596 | 562 | 1 | 0.01 |
| gold_product_performance | 3.605 | 3.398–4.107 | 0.284 | 4,725 | 1 | 0.14 |
| gold_customer_cohort_retention | 3.421 | 3.275–3.604 | 0.137 | 325 | 1 | 0.01 |
| **Total (sum of medians)** | **33.916** | 32.554–38.214 | | | 11 | 19.18 |

**Reading:** every Gold step costs 2.5–3.6 s regardless of size (739-row `dim_date` 2.616 s vs
1M-row `fact_sales` 2.774 s), so at x1 the build is dominated by per-write overhead, not data
processing. Every table is 1 file (dim_customer 2), so layout changes have nothing to skip at
this size. `silver_returns` (17,915 rows, 3.231 s) re-runs the full Bronze staging that
`silver_sales` already did.

## 3. Why also a scaled-up copy

- At x1 there is one file per table and a fixed ~2.5 s floor per write (§2), so most layout
  optimisations can't show an effect. Reporting "no change" there is honest but says nothing
  about how the design behaves on bigger data.
- So the real data is also replicated 10× (`optimisation.scaling.replicate_bronze`): copy 0 is
  the original; copies 1–9 get customer IDs +100,000 per copy and prefixed invoice numbers
  (`R03-489434`, cancellations `CR03-489449`), so they are new customers and invoices, not
  duplicates. Products, prices, dates, countries and data-quality issues are unchanged.
- Check: Silver/fact rows exactly 10× x1, customers 10 × 5,852 + Unknown, products/dates/
  monthly/cohort tables unchanged, revenue exactly 10×: `<<X10_SCALE_CHECK>>`.
- Limits: no new behaviour (same patterns with relabelled keys). Every x10 figure is labelled
  "synthetic x10 replica" and shown next to the x1 figure.

| Step | x1 median s | x10 median s | Time ratio | x10 files | x10 MB |
|---|---:|---:|---:|---:|---:|
| `<<filled from notebook §4c>>` | | | | | |

## 4. Changes, one at a time

| # | Change | Why | Metric | Before | After | % change | Beyond noise? |
|---|---|---|---|---:|---:|---:|---|
| 1 | Liquid clustering on `<<keys>>` + `OPTIMIZE` | | | | | | |
| 2 | Partitioning by `<<key>>` (compared, not adopted?) | | | | | | |
| 3 | Broadcast hints for small dimensions | | | | | | |
| 4 | Early projection / filtering, fewer shuffles | | | | | | |
| 5 | Stage Silver once (Delta table instead of `cache()`) | | | | | | |
| 6 | Incremental `MERGE` + CDF instead of full rebuild | | | | | | |

## 5. Evidence

Screenshots in `docs/img/` (EXPLAIN plans and query profiles), referenced per change.

## 6. What didn't help, and limits of these results

`<<honest notes>>`

## 7. What I'd do differently on classic compute

`<<filled at the end>>`
