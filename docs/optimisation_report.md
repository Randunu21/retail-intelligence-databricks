# Pipeline optimisation report (Phase 2)

> Every number in this report comes from `retail.bench.benchmark_runs` (real runs on
> Databricks Free Edition, serverless). `<<PLACEHOLDER>>` = not measured yet.
> Where a change made no measurable difference, the table says so.

## 1. Setup

| Item | Value |
|---|---|
| Platform | Databricks Free Edition, serverless notebooks (Spark `<<SPARK_VERSION>>`, Photon) |
| Data (x1, real) | UCI Online Retail II: Bronze 1,067,371 rows → Silver sales 1,003,357 / returns 17,915 |
| Data (x10, synthetic) | `<<X10_BRONZE_ROWS>>` rows: the real data replicated 10× with shifted customer/invoice keys (see §3). Labelled synthetic everywhere it's used. |
| What is timed | Rebuild of each Silver/Gold table with the pipeline's own code (`CREATE OR REPLACE`), wall clock on the driver |
| Runs per measurement | 1 warm-up (discarded) + `<<N_X1>>` measured at x1, `<<N_X10>>` at x10; median reported |
| Noise rule | A change counts only if the min–max ranges of before and after don't overlap |
| Cache state | Warm (serverless can't clear its disk cache, so all variants are compared warm) |

### Free Edition feature check (probe, notebook §1)

| Feature | Result | Consequence |
|---|---|---|
| `df.cache()` / `CACHE TABLE` | `<<PROBE_CACHE>>` | `<<...>>` |
| `spark.sql.shuffle.partitions` | `<<PROBE_SHUFFLE>>` | |
| `spark.sql.autoBroadcastJoinThreshold` | `<<PROBE_BROADCAST_CONF>>` | Broadcast is controlled with hints instead |
| `spark.sql.adaptive.enabled` (AQE) | `<<PROBE_AQE>>` | |
| Liquid clustering + `OPTIMIZE` | `<<PROBE_LC>>` | |
| Partitioned tables | `<<PROBE_PART>>` | |
| Change Data Feed | `<<PROBE_CDF>>` | |
| `system.query.history` | `<<PROBE_QH>>` | |
| Predictive optimisation | `<<PROBE_PO>>` | |

## 2. Baseline (x1, Phase 1 logic unchanged)

Benchmark copies equal production: `<<FINGERPRINT_MATCH>>` (row count + checksum per table).

| Step | Median s | Min–max s | Rows | Files | Size MB |
|---|---:|---:|---:|---:|---:|
| silver_sales | `<<>>` | | 1,003,357 | | |
| silver_returns | | | 17,915 | | |
| gold_dim_date | | | 739 | | |
| gold_fact_sales | | | 1,003,357 | | |
| gold_dim_product | | | 4,725 | | |
| gold_dim_customer | | | 5,853 | | |
| gold_daily_sales | | | 739 | | |
| gold_monthly_country_sales | | | 562 | | |
| gold_product_performance | | | 4,725 | | |
| gold_customer_cohort_retention | | | 325 | | |
| **Total (sum of medians)** | `<<BASELINE_X1_TOTAL_S>>` | | | | |

## 3. Why also a scaled-up copy

`<<filled in step 2.2>>`

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
