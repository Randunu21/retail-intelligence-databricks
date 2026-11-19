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
  monthly/cohort tables unchanged, revenue exactly 10× (196,438,616.370 vs 19,643,861.637): **all 11 checks passed**.
- Limits: no new behaviour (same patterns with relabelled keys). Every x10 figure is labelled
  "synthetic x10 replica" and shown next to the x1 figure.

Baseline at x10: run `39753a7e-fba8-4f8c-8117-72605897c2b9`, 2026-11-20, 1 warm-up + 3 measured runs.

| Step | x1 median s | x10 median s (min–max) | Time ratio | Rows ratio | x10 files | x10 MB |
|---|---:|---:|---:|---:|---:|---:|
| silver_sales | 6.664 | 37.557 (37.159–37.955) | 5.64 | 10.0 | 6 | 78.47 |
| silver_returns | 3.231 | 8.337 (8.143–8.364) | 2.58 | 10.0 | 1 | 0.83 |
| gold_dim_date | 2.616 | 2.557 (2.551–2.812) | 0.98 | 1.0 | 1 | 0.00 |
| gold_fact_sales | 2.774 | 5.866 (5.610–5.870) | 2.11 | 10.0 | 1 | 136.84 |
| gold_dim_product | 2.907 | 3.141 (2.983–3.436) | 1.08 | 1.0 | 1 | 0.07 |
| gold_dim_customer | 2.768 | 2.779 (2.567–2.817) | 1.00 | 10.0 | 2 | 0.54 |
| gold_daily_sales | 2.861 | 4.027 (3.971–4.546) | 1.41 | 1.0 | 1 | 0.02 |
| gold_monthly_country_sales | 3.069 | 4.660 (4.296–5.689) | 1.52 | 1.0 | 1 | 0.01 |
| gold_product_performance | 3.605 | 4.396 (4.386–4.420) | 1.22 | 1.0 | 1 | 0.14 |
| gold_customer_cohort_retention | 3.421 | 3.708 (3.466–3.846) | 1.08 | 1.0 | 1 | 0.01 |
| **Total (sum of medians)** | **33.916** | **77.028** (75.132–79.755) | 2.27 | | 16 | 216.93 |

**Reading:** 10× the rows costs 2.27× the time, because most x1 time was fixed overhead.
Silver is the bottleneck at scale: `silver_sales` alone is 49 % of the x10 build and the only
step whose time grows close to the data (×5.64); with `silver_returns` (which re-runs the
same staging for 179,150 rows) Silver is 60 %. `fact_sales` is a single 136.84 MB file at
x10, so no query against it can skip files with the default layout.

## 4. Changes, one at a time

### 4.1 Data layout of `gold.fact_sales` (step 2.3)

Variants (`optimisation.layout`): `baseline` (defaults), `part_date` (PARTITIONED BY date_key),
`zorder_32mb` (OPTIMIZE ZORDER BY date_key, customer_key; 32 MB target files), `lc_auto`
(CLUSTER BY date_key, customer_key + OPTIMIZE), `lc_32mb` (same, 32 MB target files).
Same data in every variant: `<<LAYOUT_FINGERPRINT>>`.

**Build cost** (write + OPTIMIZE):

| Scale | Variant | Write s | OPTIMIZE s | Total s | Files | MB |
|---|---|---:|---:|---:|---:|---:|
| `<<from notebook §5a>>` | | | | | | |

**Read cost** (median of 5; files/bytes from `system.query.history`):

| Scale | Query | Variant | Client s | Server ms | Files read | Files pruned | MB read |
|---|---|---|---:|---:|---:|---:|---:|
| `<<from notebook §5b–5c>>` | | | | | | | |

**Conclusion:** `<<>>`

### 4.2 Summary of all changes

| # | Change | Why | Metric | Before | After | % change | Beyond noise? |
|---|---|---|---|---:|---:|---:|---|
| 1 | Liquid clustering on `(date_key, customer_key)` + `OPTIMIZE` | | | | | | |
| 2 | Partitioning by `date_key` (compared) | | | | | | |
| 3 | Broadcast hints for small dimensions | | | | | | |
| 4 | Early projection / filtering, fewer shuffles | | | | | | |
| 5 | Stage Silver once (instead of computing it twice) | | | | | | |
| 6 | Incremental `MERGE` + CDF instead of full rebuild | | | | | | |

## 5. Evidence

Screenshots in `docs/img/` (EXPLAIN plans and query profiles), referenced per change.

## 6. What didn't help, and limits of these results

`<<honest notes>>`

## 7. What I'd do differently on classic compute

`<<filled at the end>>`
