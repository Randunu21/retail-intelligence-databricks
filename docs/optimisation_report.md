# Pipeline optimisation report (Phase 2)

> Every number in this report comes from `retail.bench.benchmark_runs` (real runs on
> Databricks Free Edition, serverless). `<<PLACEHOLDER>>` = not measured yet.
> Where a change made no measurable difference, the table says so.

## 1. Setup

| Item | Value |
|---|---|
| Platform | Databricks Free Edition, serverless notebooks (Spark `<<SPARK_VERSION>>`) |
| Data (x1, real) | UCI Online Retail II: Bronze 1,067,371 rows → Silver sales 1,003,357 / returns 17,915 |
| Data (x10, synthetic) | 10,673,710 Bronze rows: the real data replicated 10× with shifted customer/invoice keys (see §3). Labelled synthetic everywhere it's used. |
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
- Limits: no new behaviour (same patterns with relabelled keys). Bronze x10 is 1 file of
  7.71 MB vs 5.61 MB at x1: repeated values compress extremely well in Parquet, so the replica
  understates Bronze scan cost. Downstream tables grow normally (Silver sales 6.22 → 78.47 MB). Every x10 figure is labelled
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
Same data in every variant: **yes, all 10 (variant × scale) fingerprints match their baseline**.

**Build cost** (write + OPTIMIZE; median of 3 after 1 warm-up):

| Scale | Variant | Write s (min–max) | OPTIMIZE s | Total s | Files | MB |
|---|---|---:|---:|---:|---:|---:|
| x1 | baseline | 2.723 (2.603–3.340) | – | 2.723 | 1 | 12.37 |
| x1 | part_date | 16.273 (15.689–18.338) | – | 16.273 | 604 | 16.03 |
| x1 | zorder_32mb | 2.599 (2.517–2.760) | 1.214 | 3.813 | 1 | 12.37 |
| x1 | lc_auto | 3.674 (3.670–4.365) | 4.116 | 7.790 | 1 | 12.56 |
| x1 | lc_32mb | 3.700 (3.427–3.866) | 3.878 | 7.578 | 1 | 12.56 |
| x10 | baseline | 5.551 (5.526–5.586) | – | 5.551 | 1 | 136.84 |
| x10 | part_date | 15.832 (14.627–16.573) | – | 15.832 | 604 | 130.60 |
| x10 | zorder_32mb | 5.603 (5.576–5.649) | 7.621 | 13.224 | 4 | 136.85 |
| x10 | lc_auto | 5.300 (5.196–5.434) | 3.990 | 9.290 | 4 | 141.97 |
| x10 | lc_32mb | 5.143 (5.132–5.493) | 3.867 | 9.010 | 4 | 141.89 |

**Read time on the client** (median of 5 after 1 warm-up, s; min–max in brackets):

| Scale | Query | baseline | part_date | zorder_32mb | lc_auto | lc_32mb |
|---|---|---:|---:|---:|---:|---:|
| x1 | month_by_country | 0.591 (0.552–0.694) | 0.644 (0.593–0.779) | 0.615 (0.599–0.702) | 0.604 (0.565–0.678) | 0.610 (0.583–0.610) |
| x1 | one_customer | 0.599 (0.503–0.650) | 0.682 (0.608–0.790) | 0.554 (0.520–0.622) | 0.581 (0.547–0.609) | 0.554 (0.541–0.621) |
| x1 | all_by_country | 0.581 (0.572–0.659) | **0.734** (0.704–0.746) | 0.588 (0.572–0.645) | 0.617 (0.567–0.666) | 0.602 (0.565–0.622) |
| x10 | month_by_country | 0.658 (0.636–0.699) | 0.638 (0.596–0.841) | 0.666 (0.632–0.730) | 0.674 (0.658–0.680) | 0.694 (0.645–0.718) |
| x10 | one_customer | 0.624 (0.583–0.678) | 0.638 (0.602–0.664) | **0.532** (0.510–0.579) | **0.532** (0.508–0.560) | 0.568 (0.528–0.599) |
| x10 | all_by_country | 0.743 (0.728–0.811) | 0.697 (0.692–0.888) | 0.663 (0.639–0.712) | **0.657** (0.625–0.701) | 0.651 (0.638–0.734) |

Bold = min–max range doesn't overlap the baseline's. Read runs: x1 `ae73c091-f77b-4509-b1bb-bb5a8f415236`,
x10 `ccb0aacb-3122-45c1-a9a5-89b115dc5bba`. Result rows identical across variants (22 / 84 / 43).

**Files and bytes read, x10** (one isolated execution each, from `system.query.history`, all
30 matched by query tag; probe `71e0e4b4-6d5f-4abc-b066-57a432f9b994`, 2026-11-20; no result cache):

| Query | Variant | Files read | Files pruned | MB read | vs baseline | Server ms |
|---|---|---:|---:|---:|---:|---:|
| month_by_country | baseline | 1 | 0 | 16.58 | – | 411 |
| month_by_country | part_date | 26 | 578 | 6.48 | −61 % | 403 |
| month_by_country | zorder_32mb | 2 | 2 | 7.36 | −56 % | 445 |
| month_by_country | lc_auto | 2 | 2 | 10.31 | −38 % | 409 |
| month_by_country | lc_32mb | 2 | 2 | 9.92 | −40 % | 404 |
| one_customer | baseline | 1 | 0 | 52.82 | – | 350 |
| one_customer | part_date | 602 | 2 | 115.42 | **+119 %** | 416 |
| one_customer | zorder_32mb | 2 | 2 | 24.71 | −53 % | 287 |
| one_customer | lc_auto | 2 | 2 | 23.04 | −56 % | 297 |
| one_customer | lc_32mb | 2 | 2 | 29.21 | −45 % | 281 |
| all_by_country (control) | baseline | 1 | 0 | 16.44 | – | 512 |
| all_by_country (control) | part_date | 604 | 0 | 115.56 | **+603 %** | 493 |
| all_by_country (control) | zorder_32mb | 4 | 0 | 15.83 | −4 % | 433 |
| all_by_country (control) | lc_auto | 4 | 0 | 15.81 | −4 % | 494 |
| all_by_country (control) | lc_32mb | 4 | 0 | 15.81 | −4 % | 428 |

x1 I/O: `<<x1 re-probe pending>>`. The first x1 probe shared its tags with x10 (bug, fixed:
tags now include the scale, and duplicate tags raise an error), so its x1 rows were copies of
x10 and were discarded.

`rows_read` is not reported: the history counts rows *after* the scan's filter (84 for the
customer lookup in every layout), so it says nothing about skipping. `one_customer` reads more
MB than the full scan because it needs 6 columns (incl. text columns) instead of 2. Server ms is a
single sample per cell: indicative only.

**Findings:**
- **Data skipping works with clustering and Z-order.** At x10 the fact table has 4 files after
  OPTIMIZE; the month filter and the customer filter each skip 2 of them. Bytes read drop
  38–40 % (month) and 45–56 % (customer) with liquid clustering, 56 % / 53 % with Z-order.
- **Partitioning by day wins only on its own key**: the month report reads 26 of 604 files
  (−61 % bytes), but every other query pays: the customer lookup reads 602 files and +119 % bytes,
  the full scan +603 % bytes (tiny files: per-file overhead, nearly whole files read). Plus a
  ×2.9 slower build at x10 and ×6 at x1. Rejected.
- **The control query (no filter) skips nothing in any layout**, as it should. Bytes are the same
  ±4 %, so the faster client time seen for `lc_auto` there is not less I/O; more files read in
  parallel is consistent with it but not proven by single server samples.
- **Bytes fell, time barely moved:** server time is 280–450 ms in every layout (−15–20 % for
  the customer lookup with clustering). At ~140 MB the query is dominated by fixed costs; the
  byte savings are what matter as the table grows.
- **Z-order vs liquid clustering:** Z-order read a bit less for the month query (7.36 vs ~10 MB)
  but its OPTIMIZE took 7.6 s vs 4.0 s, it rewrites the table every time, and its keys aren't
  stored with the table. Liquid clustering is incremental and the keys can change later.

**Conclusion:** cluster `fact_sales` by `(date_key, customer_key)` with liquid clustering
(default file size; 32 MB made no difference). It halves bytes for customer lookups and cuts
~40 % for date-range reports at x10, for ~4 s of OPTIMIZE. Don't partition: ~600 daily
partitions on a 10–140 MB table cost far more than they save. At x1 (1 file) no gain is possible;
in the pipeline the cost is small enough to keep for when the table grows (decided in step 2.6).

### 4.2 Summary of all changes

| # | Change | Why | Metric | Before | After | % change | Beyond noise? |
|---|---|---|---|---:|---:|---:|---|
| 1 | Liquid clustering on `(date_key, customer_key)` + `OPTIMIZE` (x10) | Data skipping for date and customer filters | MB read: month / customer query | 16.58 / 52.82 | 10.31 / 23.04 | −38 % / −56 % | Deterministic I/O counts (single execution) |
| 2 | Partitioning by `date_key` (compared, rejected) (x10) | Classic layout; tests over-partitioning | Build s; MB read customer / full scan | 5.551; 52.82 / 16.44 | 15.832; 115.42 / 115.56 | +185 %; +119 % / +603 % | Build: ranges don't overlap; I/O deterministic |
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
