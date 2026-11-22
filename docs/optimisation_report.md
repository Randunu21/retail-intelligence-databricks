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

**Files and bytes read** (one isolated execution per cell, from `system.query.history`, all
matched by query tag, no result cache). x10 was probed twice (probe 1: `71e0e4b4…`, probe 2 with
scale-unique tags); file counts were identical in both, MB read varied up to ~20 %, so x10 MB
is shown as probe 1 / probe 2. x1 is from probe 2 only (probe 1's x1 rows were x10 copies due to
the tag bug, discarded).

| Query | Variant | x1 files read / pruned | x1 MB | x10 files read / pruned | x10 MB (p1 / p2) | x10 vs baseline |
|---|---|---:|---:|---:|---:|---:|
| month_by_country | baseline | 1 / 0 | 1.77 | 1 / 0 | 16.58 / 16.58 | – |
| month_by_country | part_date | 26 / 578 | 1.19 | 26 / 578 | 6.48 / 6.48 | −61 % |
| month_by_country | zorder_32mb | 1 / 0 | 1.77 | 2 / 2 | 7.36 / 8.63 | −48 to −56 % |
| month_by_country | lc_auto | 1 / 0 | 1.80 | 2 / 2 | 10.31 / 9.94 | −38 to −40 % |
| month_by_country | lc_32mb | 1 / 0 | 1.80 | 2 / 2 | 9.92 / 9.87 | −40 % |
| one_customer | baseline | 1 / 0 | 4.45 | 1 / 0 | 52.82 / 52.82 | – |
| one_customer | part_date | 602 / 2 | 16.00 | 602 / 2 | 115.42 / 115.42 | **+119 %** |
| one_customer | zorder_32mb | 1 / 0 | 4.45 | 2 / 2 | 24.71 / 29.19 | −45 to −53 % |
| one_customer | lc_auto | 1 / 0 | 4.58 | 2 / 2 | 23.04 / 26.08 | −51 to −56 % |
| one_customer | lc_32mb | 1 / 0 | 4.58 | 2 / 2 | 29.21 / 27.34 | −45 to −48 % |
| all_by_country (control) | baseline | 1 / 0 | 1.77 | 1 / 0 | 16.44 / 16.44 | – |
| all_by_country (control) | part_date | 604 / 0 | 16.03 | 604 / 0 | 115.56 / 115.56 | **+603 %** |
| all_by_country (control) | zorder_32mb | 1 / 0 | 1.77 | 4 / 0 | 15.83 / 15.83 | −4 % |
| all_by_country (control) | lc_auto | 1 / 0 | 1.77 | 4 / 0 | 15.81 / 15.82 | −4 % |
| all_by_country (control) | lc_32mb | 1 / 0 | 1.77 | 4 / 0 | 15.81 / 15.81 | −4 % |

Server time (single samples, indicative): x1 270–400 ms, x10 280–510 ms in every layout; the
customer lookup with clustering/Z-order was 281–297 ms vs 322–350 ms baseline in both probes.

`rows_read` is not reported: the history counts rows *after* the scan's filter (84 for the
customer lookup in every layout), so it says nothing about skipping. `one_customer` reads more
MB than the full scan because it needs 6 columns (incl. text columns) instead of 2.

**Findings:**
- **x1: nothing to skip.** Every non-partitioned layout is 1 file and reads the same bytes;
  partitioning reads 604 files and 9× the bytes for a full scan (16.03 vs 1.77 MB).
- **x10: data skipping works with clustering and Z-order.** The fact table has 4 files after
  OPTIMIZE; the month filter and the customer filter each skip 2 of them. Bytes read drop
  38–40 % (month) and 45–56 % (customer) with liquid clustering, 48–56 % / 45–53 % with Z-order.
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
- **Z-order vs liquid clustering:** Z-order read a bit less for the month query (7.4–8.6 vs ~10 MB)
  but its OPTIMIZE took 7.6 s vs 4.0 s, it rewrites the table every time, and its keys aren't
  stored with the table. Liquid clustering is incremental and the keys can change later.

**Conclusion:** cluster `fact_sales` by `(date_key, customer_key)` with liquid clustering
(default file size; 32 MB made no difference). It halves bytes for customer lookups and cuts
~40 % for date-range reports at x10, for ~4 s of OPTIMIZE. Don't partition: ~600 daily
partitions on a 10–140 MB table cost far more than they save. At x1 (1 file) no gain is possible;
in the pipeline the cost is small enough to keep for when the table grows (decided in step 2.6).

### 4.2 Compute: shuffles, joins and an alternative to caching (step 2.4)

All variants produce the same data as the baseline (fingerprint match on every table, both
scales). 1 warm-up + 3 measured runs, interleaved; median (min–max).

**Caching on serverless:** `CACHE TABLE` is rejected, but `df.cache()` works:
`is_cached: True | storageLevel: Disk Serialized 1x Replicated`, i.e. serverless caches to local
disk, not memory. A cache lives only in the notebook session; the pipeline can't share it.

**Silver: sales + returns built together** (one combined step):

| Variant | What changes | x1 s | x1 vs base | x10 s | x10 vs base |
|---|---|---:|---:|---:|---:|
| baseline | Bronze staging computed twice (as in the pipeline) | 8.581 (8.554–8.928) | – | 46.776 (46.431–46.923) | – |
| staged_table | staging written once to Delta, both outputs read it | 9.459 (9.439–9.607) | **+10.2 %** | 37.245 (37.005–37.256) | **−20.4 %** |
| cached | staging `cache()`d once | 9.573 (9.425–9.579) | **+11.6 %** | 36.501 (36.291–36.602) | **−22.0 %** |
| agg_dedup | de-dup by GROUP BY instead of a window (still twice) | 8.010 (7.984–8.218) | **−6.7 %** | 43.726 (43.632–43.796) | **−6.5 %** |

Bold = beyond noise (ranges don't overlap). Runs: 2026-11-21/22.

- Computing the staging once saves ~10 s of 47 at x10, the largest single gain so far. At x1 the
  extra write / cache costs more than the duplicated work it saves (+10–12 %).
- `cached` beat `staged_table` by ~0.7 s at x10, but only exists inside a notebook session; the
  pipeline equivalent is a stored intermediate (applied in step 2.6).
- GROUP BY de-dup is a steady −6.5 to −6.7 % at both scales (no sort; partial aggregation before
  the shuffle). Its combination with staging-once was not measured separately.

**Gold join plans** (x10, `plan_summary` of each query's physical plan before execution):
`<<re-run §6b with the fixed counter>>`. Join types from the first run (reliable):
baseline = auto-broadcast for monthly (dim_date), product_performance and two cohort joins;
daily_sales is a `ShuffledHashJoin` because dim_date is the preserved side of its LEFT JOIN.
`broadcast` hints produced plans identical to the baseline; `MERGE` hints switched joins to
`SortMergeJoin`. (The first run's shuffle/broadcast counts were wrong: the counter didn't know
Photon's node names; fixed and tested against a real plan.)

**Gold join timing** (median s; % vs baseline; bold = beyond noise):

| Step | Scale | baseline | broadcast | no_broadcast | rewrite |
|---|---|---:|---:|---:|---:|
| daily_sales | x1 | 2.287 | 2.410 (+5.4 %) | 2.571 (**+12.4 %**) | 2.437 (**+6.6 %**) |
| monthly_country_sales | x1 | 2.401 | 2.371 (−1.2 %) | 2.880 (**+20.0 %**) | 2.557 (+6.5 %) |
| product_performance | x1 | 2.917 | 2.759 (−5.4 %) | 2.953 (+1.2 %) | 2.820 (−3.3 %)¹ |
| cohort_retention | x1 | 2.614 | 2.711 (+3.7 %) | 3.028 (+15.8 %) | 2.753 (+5.3 %)¹ |
| daily_sales | x10 | 3.889 | 3.739 (−3.9 %) | 5.355 (**+37.7 %**) | 2.868 (**−26.3 %**) |
| monthly_country_sales | x10 | 3.958 | 3.946 (−0.3 %) | 5.959 (**+50.6 %**) | 3.035 (**−23.3 %**) |
| product_performance | x10 | 3.960 | 3.940 (−0.5 %) | 4.272 (**+7.9 %**) | 3.875 (−2.1 %)¹ |
| cohort_retention | x10 | 3.077 | 2.954 (−4.0 %) | 3.612 (**+17.4 %**) | 3.039 (−1.2 %)¹ |

¹ no rewrite for this step: the baseline query re-run (a noise reference).

- **Broadcast hints: no effect**, because the optimiser already broadcasts the small sides. The
  daily_sales hint is ignored (the preserved side of a LEFT JOIN can't be broadcast).
- **Forcing sort-merge joins costs +8 to +51 % at x10**: that is what automatic broadcasting
  already saves.
- **Rewrites are the Gold win at x10**: aggregating 10M fact rows to 739 days before the join
  (−26.3 %), and taking year/month from `invoice_ts` instead of joining dim_date (−23.3 %). At x1
  the daily rewrite is 6.6 % slower: the extra aggregation stage costs more than it saves.
- Note: the monthly rewrite's `year_month` is nullable where the pipeline's (from dim_date) is
  NOT NULL; values are identical.

### 4.3 Summary of all changes

| # | Change | Why | Metric | Before | After | % change | Beyond noise? |
|---|---|---|---|---:|---:|---:|---|
| 1 | Liquid clustering on `(date_key, customer_key)` + `OPTIMIZE` (x10) | Data skipping for date and customer filters | MB read: month / customer query | 16.58 / 52.82 | 10.31 / 23.04 | −38 % / −56 % | Deterministic I/O counts (single execution) |
| 2 | Partitioning by `date_key` (compared, rejected) (x10) | Classic layout; tests over-partitioning | Build s; MB read customer / full scan | 5.551; 52.82 / 16.44 | 15.832; 115.42 / 115.56 | +185 %; +119 % / +603 % | Build: ranges don't overlap; I/O deterministic |
| 3 | Broadcast hints for small dimensions (and MERGE hints to price auto-broadcast) (x10) | Avoid shuffling the big side | Gold join step s | 3.08–3.96 | 2.95–3.95 | −0.3 to −4.0 % (no change); MERGE: +7.9 to +50.6 % | Hints: no; MERGE: yes |
| 4 | Fewer wide operations (x10): pre-aggregate before join (daily), drop a join (monthly), GROUP BY de-dup (Silver) | Less data through shuffles/joins | s | 3.889 / 3.958 / 46.776 | 2.868 / 3.035 / 43.726 | −26.3 % / −23.3 % / −6.5 % | Yes |
| 5 | Stage Silver once: Delta table vs `cache()` (x10; x1 in brackets) | Typing + window de-dup ran twice | Silver (sales + returns) s | 46.776 (8.581) | 37.245 / 36.501 (9.459 / 9.573) | −20.4 / −22.0 % (+10.2 / +11.6 %) | Yes |
| 6 | Incremental `MERGE` + CDF instead of full rebuild | | | | | | |

## 5. Evidence

Screenshots in `docs/img/` (EXPLAIN plans and query profiles), referenced per change.

## 6. What didn't help, and limits of these results

`<<honest notes>>`

## 7. What I'd do differently on classic compute

`<<filled at the end>>`
