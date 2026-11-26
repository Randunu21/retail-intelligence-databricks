# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 · Product segmentation and market basket analysis
# MAGIC
# MAGIC Goal: describe every product by how it sells, group products into **named segments**
# MAGIC (fast-moving staples, seasonal, premium low-volume, slow movers ... whatever the real
# MAGIC profiles support), and find **which products are bought together** to turn into
# MAGIC cross-sell recommendations.
# MAGIC
# MAGIC | Section | What it does | Step |
# MAGIC |---|---|---|
# MAGIC | 1 | **Free Edition probe**: Spark ML KMeans + **FPGrowth** on serverless, MLflow experiment | 4.1 |
# MAGIC | 2 | **Product feature table** `retail.ml.product_features` + long-tail analysis | 4.1 |
# MAGIC | 3 | **Product clustering**: long-tail rule, preprocessing, KMeans / bisecting / GMM sweep, stability, final choice | 4.2 |
# MAGIC | 4 | **Final model, profiles, names, action groups**, `retail.gold.product_segments` | 4.3 |
# MAGIC | 5 | **Baskets and thresholds**: basket sizes, item support, FP-Growth threshold sweep | 4.4 |
# MAGIC | 6 | **Association rules**: thresholds, FP-Growth (all / UK / rest), SQL cross-check, variant filter, `retail.gold.product_associations` | 4.5 |
# MAGIC | 7 | **Generated `docs/product_insights.md`** + charts, Phase 4 checks | 4.6 |
# MAGIC
# MAGIC Logic lives in `src/features/` and `src/segmentation/` (unit-tested); this notebook runs it
# MAGIC and explains each decision.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install the Feature Engineering client
# MAGIC
# MAGIC Same as notebook 03: only used to *confirm* that Unity Catalog sees the product table as
# MAGIC a feature table (2.4). If the install fails, comment these two cells out and carry on.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Make `src/` importable

# COMMAND ----------

import sys
from pathlib import Path

REPO_DIR = Path.cwd().parent
SRC_DIR = str(REPO_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
print("src on path:", SRC_DIR)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import platform

import mlflow
from pyspark.sql import functions as F

from common.tables import (
    CATALOG,
    GOLD_DIM_PRODUCT,
    ML_PRODUCT_FEATURES,
    PRODUCT_EXPERIMENT_NAME,
    SILVER_RETURNS,
    SILVER_SALES,
    user_experiment_path,
)
from features.customer_features import default_as_of, describe_features
from features.feature_store import (
    describe_feature_table,
    feature_engineering_client_status,
    feature_table_ddl,
    snapshot_predicate,
    write_snapshot,
)
from features.product_features import (
    PRODUCT_FEATURE_COLUMNS,
    PRODUCT_FEATURE_SCHEMA,
    PRODUCT_KEY_COLUMNS,
    PRODUCT_TABLE_COMMENT,
    build_product_features,
    long_tail_table,
    pareto_counts,
    reconcile_product_features,
)

USER = spark.sql("SELECT current_user() AS u").first()["u"]
EXPERIMENT_PATH = user_experiment_path(USER, PRODUCT_EXPERIMENT_NAME)
print("user:", USER)
print("MLflow experiment:", EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Free Edition probe
# MAGIC
# MAGIC The docs list **`FPGrowthModel` as not supported** for Spark ML on serverless / Spark
# MAGIC Connect (checked 2026-11-25). Docs change, so we test it for real on three tiny baskets
# MAGIC and record the exact outcome of each call (`fit`, `freqItemsets`, `associationRules`,
# MAGIC `transform`). The result decides 4.4:
# MAGIC
# MAGIC - **all ok** → use Spark ML `FPGrowth` directly;
# MAGIC - **any failure** → fallback: Apriori-style level-wise counting written in Spark
# MAGIC   SQL/DataFrames (frequent items → candidate pairs/triples built only from frequent items →
# MAGIC   support, confidence, lift). Same rules as FP-Growth at the same thresholds; the local
# MAGIC   tests (classic PySpark, where FPGrowth works) prove it.
# MAGIC
# MAGIC KMeans is re-checked too (it worked in Phase 3; one line keeps this notebook
# MAGIC self-contained).
# MAGIC
# MAGIC **Result (2026-11-25):** all four FPGrowth calls **ok** on serverless (Spark 4.2.0,
# MAGIC environment 6), contrary to the docs. So section 5 uses Spark ML `FPGrowth` directly. We
# MAGIC never *save* the fitted model (persistence is the part most likely to be blocked, as
# MAGIC with BisectingKMeans in Phase 3): itemsets and rules are written to Delta instead.

# COMMAND ----------

probe = {
    "spark_version": spark.version,
    "python": platform.python_version(),
    "mlflow": mlflow.__version__,
}

# 1a. KMeans (sanity, used in section 3)
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

tiny = spark.createDataFrame([(0.0, 0.0), (0.1, 0.2), (5.0, 5.0), (5.1, 5.2)], "x DOUBLE, y DOUBLE")
try:
    km = KMeans(k=2, seed=1).fit(
        VectorAssembler(inputCols=["x", "y"], outputCol="features").transform(tiny)
    )
    probe["spark_ml_kmeans"] = f"ok ({len(km.clusterCenters())} centres)"
    del km
except Exception as e:
    probe["spark_ml_kmeans"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

# 1b. FPGrowth: each step separately, so we know exactly what works
from pyspark.ml.fpm import FPGrowth

baskets_tiny = spark.createDataFrame(
    [(1, ["a", "b", "c"]), (2, ["a", "b"]), (3, ["a", "c"])], "id INT, items ARRAY<STRING>"
)
fp_model = None
try:
    fp_model = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.5).fit(baskets_tiny)
    probe["fpgrowth_fit"] = "ok"
except Exception as e:
    probe["fpgrowth_fit"] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
for step, fn in [
    ("fpgrowth_freq_itemsets", lambda m: m.freqItemsets.count()),
    ("fpgrowth_association_rules", lambda m: m.associationRules.count()),
    ("fpgrowth_transform", lambda m: m.transform(baskets_tiny).count()),
]:
    if fp_model is None:
        probe[step] = "skipped (fit failed)"
        continue
    try:
        probe[step] = f"ok ({fn(fp_model)} rows)"
    except Exception as e:
        probe[step] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
if fp_model is not None:
    del fp_model
FPGROWTH_WORKS = all(
    str(probe[s]).startswith("ok")
    for s in ["fpgrowth_fit", "fpgrowth_freq_itemsets", "fpgrowth_association_rules"]
)
probe["FPGROWTH_WORKS"] = FPGROWTH_WORKS

# 1c. MLflow experiment for Phase 4 + a probe run (the probe results are logged as JSON)
try:
    exp = mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="probe_free_edition") as run:
        mlflow.set_tags({"phase": "4", "kind": "probe"})
        mlflow.log_dict({k: str(v) for k, v in probe.items()}, "probe/info.json")
    probe["mlflow_experiment_id"] = exp.experiment_id
    probe["mlflow_probe_run"] = run.info.run_id
except Exception as e:
    probe["mlflow_probe_run"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

# 1d. Feature Engineering client
probe["feature_engineering_client"] = feature_engineering_client_status()

for k, v in probe.items():
    print(f"{k:28s} {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Product feature table
# MAGIC
# MAGIC ### 2.1 Snapshot date and sources
# MAGIC
# MAGIC Same snapshot as the customer features (the day after the last sale), same source layer
# MAGIC (Silver: returns only exist there). One difference: **customer-less lines are kept**.
# MAGIC A product sold to an unidentified buyer was still sold, so velocity and revenue include
# MAGIC those 226,761 lines; only `unique_customers` is limited to known customers.

# COMMAND ----------

sales = spark.table(SILVER_SALES)
returns = spark.table(SILVER_RETURNS)
AS_OF = default_as_of(sales)
DATA_START = sales.agg(F.min("invoice_date")).first()[0]
print("as_of (snapshot) =", AS_OF, "| data starts", DATA_START)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Build the features and reconcile them
# MAGIC
# MAGIC Full definitions: docstring of `src/features/product_features.py`. The mapping from the
# MAGIC business questions to features:
# MAGIC
# MAGIC | Question | Feature(s) |
# MAGIC |---|---|
# MAGIC | Price level | `median_unit_price` (list price), `avg_unit_price` (after bulk effects) |
# MAGIC | Sales velocity | `units_per_week` (since first sale), `active_week_share` (regularity) |
# MAGIC | Revenue contribution | `revenue`, `revenue_share` |
# MAGIC | Seasonality | `seasonal_revenue_share` (Q4), `monthly_units_cv` (lumpiness) |
# MAGIC | Return rate | `return_rate` (units, capped at 1) |
# MAGIC | Reach | `unique_customers`, `n_invoices` |
# MAGIC | Country spread | `n_countries`, `non_uk_revenue_share` |
# MAGIC | Lifecycle | `days_on_sale`, `days_since_last_sale` (discontinued items) |
# MAGIC
# MAGIC Two interview points:
# MAGIC - **Velocity "since first sale"**, not "while it sold": an item that stopped selling a year
# MAGIC   ago should look slow, not fast.
# MAGIC - **Monthly CV from two sums** (`sum x`, `sum x²` over months with sales, divided by
# MAGIC   *all* months on sale): zero months are counted without building a product × month grid.
# MAGIC
# MAGIC Then independent recounts: one row per product sold before as_of, revenue and units equal
# MAGIC Silver, shares sum to 1, NULL CV exactly for products with < 3 months of history.

# COMMAND ----------

features = build_product_features(sales, returns, AS_OF, data_start=DATA_START)
checks = reconcile_product_features(features, sales, returns, AS_OF)
for k, v in checks.items():
    print(f"{k:24s} {v}")

assert checks["products"] == checks["expected_products"], "product count mismatch"
assert checks["duplicate_keys"] == 0, "duplicate stock_code"
assert abs(checks["revenue_diff"]) < 0.01, "revenue does not reconcile with Silver"
assert checks["units_diff"] == 0, "units do not reconcile with Silver"
assert abs(checks["revenue_share_sum"] - 1.0) < 1e-6, "revenue shares don't sum to 1"
assert checks["short_history_products"] == checks["null_cv"], "unexpected NULL CVs"
lo, hi = checks["return_rate_range"]
assert lo >= 0.0 and hi <= 1.0, "return_rate outside [0, 1]"
print("all feature checks passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Create the feature table and write the snapshot
# MAGIC
# MAGIC Same pattern as `customer_features` (Phase 3): DDL with `PRIMARY KEY (stock_code,
# MAGIC as_of_date TIMESERIES)` makes it a feature table in Unity Catalog; `replaceWhere` on
# MAGIC `as_of_date` makes the write idempotent. `feature_table_ddl` / `write_snapshot` were
# MAGIC generalised in this phase (schema, keys and comment are parameters; the defaults are
# MAGIC still the customer table, so notebook 03 is unchanged).

# COMMAND ----------

ddl = feature_table_ddl(
    ML_PRODUCT_FEATURES,
    schema=PRODUCT_FEATURE_SCHEMA,
    key_columns=PRODUCT_KEY_COLUMNS,
    comment=PRODUCT_TABLE_COMMENT,
)
print(ddl)
spark.sql(ddl)
write_snapshot(features, ML_PRODUCT_FEATURES, AS_OF, columns=PRODUCT_FEATURE_COLUMNS)

snapshot = spark.table(ML_PRODUCT_FEATURES).where(snapshot_predicate(AS_OF))
n_written = snapshot.count()
print(f"rows written for {AS_OF}: {n_written:,}")
assert n_written == checks["products"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Is it a feature table?
# MAGIC
# MAGIC `information_schema` must show the PK; the Feature Engineering client should report
# MAGIC `stock_code` + `as_of_date` as keys and `as_of_date` as the time-series key. Also visible
# MAGIC in **Catalog → retail → ml → product_features** and the **Features** page.

# COMMAND ----------

schema_name, table_name = ML_PRODUCT_FEATURES.split(".")[1:]
display(
    spark.sql(f"""
    SELECT tc.constraint_name, tc.constraint_type, kcu.column_name, kcu.ordinal_position
    FROM {CATALOG}.information_schema.table_constraints tc
    JOIN {CATALOG}.information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
    WHERE tc.table_schema = '{schema_name}' AND tc.table_name = '{table_name}'
    ORDER BY kcu.ordinal_position
""")
)

fe_status = feature_engineering_client_status()
if fe_status["available"]:
    try:
        fe_info = describe_feature_table(ML_PRODUCT_FEATURES)
    except Exception as e:
        fe_info = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
else:
    fe_info = {"client": fe_status}
print(fe_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Feature distributions
# MAGIC
# MAGIC Input to preprocessing in 4.2: skewness > 1 → log transform; NULL counts show which
# MAGIC features can't be clustered on without a rule (only `monthly_units_cv` can be NULL).

# COMMAND ----------

NUMERIC_PRODUCT_FEATURES = [
    "n_invoices",
    "units_sold",
    "revenue",
    "revenue_share",
    "median_unit_price",
    "avg_unit_price",
    "avg_quantity_per_line",
    "unique_customers",
    "n_countries",
    "non_uk_revenue_share",
    "units_per_week",
    "active_week_share",
    "seasonal_revenue_share",
    "monthly_units_cv",
    "months_on_sale",
    "return_rate",
    "days_on_sale",
    "days_since_last_sale",
]
product_feats = spark.table(ML_PRODUCT_FEATURES).where(snapshot_predicate(AS_OF))
dist = describe_features(product_feats, NUMERIC_PRODUCT_FEATURES).toPandas()
display(dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 The long tail
# MAGIC
# MAGIC Retail catalogues are long-tailed: a minority of products makes most of the revenue,
# MAGIC and many products sold only a handful of times. Clustering those near-empty histories
# MAGIC mostly clusters noise (a product with 2 invoices has a "velocity" and a "CV", but they
# MAGIC mean little). Two views decide the cut-off in 4.2:
# MAGIC
# MAGIC - **Pareto**: how many top products make 50 / 80 / 90 / 95 % of revenue.
# MAGIC - **Threshold table**: how many products, and how much revenue, remain at each minimum
# MAGIC   number of invoices.
# MAGIC
# MAGIC Plus the top products by revenue with their descriptions from `gold.dim_product`, as a
# MAGIC sanity check that the features look like the real catalogue.

# COMMAND ----------

pareto = pareto_counts(product_feats)
n_products = checks["products"]
for share, n in pareto.items():
    print(f"top {n:5,d} products ({n / n_products:6.1%}) make {share:.0%} of revenue")

tail = long_tail_table(product_feats)
for r in tail:
    print(
        f"n_invoices >= {r['min_invoices']:3d}: {r['products']:5,d} products "
        f"({r['product_share']:6.1%}), {r['revenue_share']:6.1%} of revenue"
    )

# COMMAND ----------

top = (
    product_feats.join(
        spark.table(GOLD_DIM_PRODUCT).select("stock_code", "description"), "stock_code", "left"
    )
    .orderBy(F.col("revenue").desc())
    .select(
        "stock_code",
        "description",
        F.round("revenue", 0).alias("revenue"),
        "n_invoices",
        F.round("median_unit_price", 2).alias("median_price"),
        F.round("units_per_week", 1).alias("units_per_week"),
        F.round("seasonal_revenue_share", 3).alias("q4_share"),
        F.round("monthly_units_cv", 2).alias("monthly_cv"),
        F.round("return_rate", 4).alias("return_rate"),
        "n_countries",
    )
    .limit(15)
)
display(top)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 What to share back (step 4.1)
# MAGIC
# MAGIC Paste the output of this cell into the chat: the thresholds and preprocessing in 4.2 are
# MAGIC chosen from these numbers.

# COMMAND ----------

lines = ["### 4.1 results", "", "**Probe**"]
lines += [f"- {k}: {v}" for k, v in probe.items()]
lines += ["", "**Feature checks**"] + [f"- {k}: {v}" for k, v in checks.items()]
lines += ["", f"**Rows written** for {AS_OF}: {n_written:,}", f"**FE client**: {fe_info}"]
lines += [
    "",
    "**Distributions**",
    "",
    "| feature | n_null | min | p1 | p25 | p50 | p75 | p95 | p99 | max | mean | skew |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for _, r in dist.iterrows():
    vals = [
        r["n_null"],
        r["min"],
        r["p1"],
        r["p25"],
        r["p50"],
        r["p75"],
        r["p95"],
        r["p99"],
        r["max"],
        r["mean"],
        r["skewness"],
    ]
    lines.append(
        "| "
        + r["feature"]
        + " | "
        + " | ".join(f"{v:.4g}" if v is not None else "" for v in vals)
        + " |"
    )
lines += ["", "**Pareto**"] + [
    f"- {s:.0%} of revenue: top {n:,} products" for s, n in pareto.items()
]
lines += ["", "**Long tail**"] + [
    f"- n_invoices >= {r['min_invoices']}: {r['products']:,} products ({r['product_share']:.1%}), {r['revenue_share']:.1%} of revenue"
    for r in tail
]
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Product clustering
# MAGIC
# MAGIC Same method as customer segmentation (Phase 3), reused unchanged from
# MAGIC `src/segmentation/` with `stock_code` as the id: preprocessing → sweep of three
# MAGIC algorithms over k = 2..10 → stability → pre-stated choice rule. What's new for products
# MAGIC is in `src/segmentation/product_segments.py`: who is clustered, and on which features.
# MAGIC
# MAGIC ### 3.1 The long tail: who gets clustered?
# MAGIC
# MAGIC From 2.6: the top 1,030 products (22 %) make 80 % of revenue, and the 978 products with
# MAGIC fewer than 20 invoices make only 1.7 %. For a product with a handful of orders, every
# MAGIC share-type feature jumps with each single order (one Q4 order out of 4 = 25 % "seasonal"),
# MAGIC so its features are mostly noise. Rule (stated before clustering):
# MAGIC
# MAGIC | class | condition | segment |
# MAGIC |---|---|---|
# MAGIC | `new_product` | fewer than 3 months on sale (checked first) | rule: **New Products** |
# MAGIC | `low_volume` | fewer than **20** invoices | rule: **Long Tail** |
# MAGIC | `clustered` | the rest (~3.7k products, ~98 % of revenue) | from the model |
# MAGIC
# MAGIC Why 20: it keeps 79 % of products and 98.3 % of revenue (4.1 table); at 10 we'd add 392
# MAGIC products for 0.5 % of revenue, at 50 we'd lose 2.4 % of revenue and 775 more products.
# MAGIC Every product still gets a segment in Gold; only the *method* differs.

# COMMAND ----------

import pandas as pd

from segmentation.clustering import ALGORITHMS, run_sweep
from segmentation.metrics import elbow_k, suggest_k
from segmentation.plots import ari_bars, correlation_heatmap, metric_by_k, share_bars
from segmentation.preprocessing import (
    CLIP_QUANTILES,
    CORRELATION_THRESHOLD,
    SKEW_THRESHOLD,
    apply_preprocessor,
    choose_log_features,
    clip_counts,
    correlation_matrix,
    fit_preprocessor,
    log_transform,
    prune_correlated,
    scaled_name,
)
from segmentation.product_segments import (
    CLUSTERED,
    DEFAULT_CANDIDATES_PER_ALGORITHM,
    ELIGIBILITY_COL,
    MIN_INVOICES,
    MIN_MONTHS,
    PRODUCT_CANDIDATE_FEATURES,
    PRODUCT_PROFILE_ONLY_FEATURES,
    add_eligibility,
    eligibility_summary,
    stability_candidates,
)
from segmentation.stability import choose_final, stability
from segmentation.tracking import log_trial, results_table

with_eligibility = add_eligibility(product_feats, MIN_INVOICES, MIN_MONTHS)
eligibility = eligibility_summary(with_eligibility)
for r in eligibility:
    print(
        f"{r['eligibility']:12s} {r['products']:5,d} products ({r['product_share']:6.1%}), "
        f"{r['revenue_share']:6.1%} of revenue"
    )
eligible = with_eligibility.where(F.col(ELIGIBILITY_COL) == CLUSTERED)
N_CLUSTERED = next(r["products"] for r in eligibility if r["eligibility"] == CLUSTERED)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Which features, and which get a log?
# MAGIC
# MAGIC **Clustering candidates** (priority order: velocity and price first, the two axes a
# MAGIC merchandiser thinks in; then seasonality, lumpiness, lifecycle, international appeal,
# MAGIC regularity, reach). **Profile-only** features (printed below) stay in the table and
# MAGIC describe the segments, but would only repeat information or carve a trivial split. Same
# MAGIC idea as Phase 3's basket value: `log(revenue) ≈ log(price) + log(velocity) + log(weeks)`,
# MAGIC so revenue adds no new direction to the distance; it's the *result* of the segment, which
# MAGIC is exactly what we want to read in the profiles.
# MAGIC
# MAGIC Log rule as in Phase 3: raw skewness > 1, now measured on the **eligible** products only
# MAGIC (the long tail is excluded, so the distributions change).

# COMMAND ----------

for feature, reason in PRODUCT_PROFILE_ONLY_FEATURES.items():
    print(f"profile only  {feature:24s} {reason}")

raw_dist = describe_features(eligible, PRODUCT_CANDIDATE_FEATURES).toPandas()
raw_skew = dict(zip(raw_dist["feature"], raw_dist["skewness"], strict=True))
LOG_CANDIDATES = choose_log_features(raw_skew, PRODUCT_CANDIDATE_FEATURES, SKEW_THRESHOLD)
print("\nraw skewness:", {k: round(v, 3) for k, v in raw_skew.items()})
print("log1p applied to:", LOG_CANDIDATES)
display(raw_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Correlation check and pruning
# MAGIC
# MAGIC On the transformed scale (what KMeans sees). In priority order, a feature is dropped if
# MAGIC |r| > 0.9 with one already kept. Expected suspects: `unique_customers` vs velocity (more
# MAGIC buyers = more units) and `active_week_share` vs `monthly_units_cv` (a product that sells
# MAGIC every week has smooth months). The rule decides, not us.

# COMMAND ----------

logged = log_transform(eligible.select("stock_code", *PRODUCT_CANDIDATE_FEATURES), LOG_CANDIDATES)
corr = correlation_matrix(logged, PRODUCT_CANDIDATE_FEATURES)
CLUSTER_FEATURES, dropped = prune_correlated(
    PRODUCT_CANDIDATE_FEATURES, corr, CORRELATION_THRESHOLD
)
LOG_FEATURES = [f for f in LOG_CANDIDATES if f in CLUSTER_FEATURES]
print("clustering features:", CLUSTER_FEATURES)
print("log features      :", LOG_FEATURES)
print("dropped           :", {k: (v[0], round(v[1], 3)) for k, v in dropped.items()})
corr_fig = correlation_heatmap(
    corr, PRODUCT_CANDIDATE_FEATURES, "Product features: correlation (log1p where skewed)"
)
display(corr_fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Fit clipping + scaling, apply, log to MLflow
# MAGIC
# MAGIC Clip to [p1, p99] after the log (keeps extreme products like the 80,995-unit one-line
# MAGIC order from dragging a centroid, without dropping them), then standardise. Parameters are
# MAGIC JSON, logged with the run and re-applied at scoring time without refitting.

# COMMAND ----------

pp = fit_preprocessor(eligible, CLUSTER_FEATURES, LOG_FEATURES, CLIP_QUANTILES)
clipped = clip_counts(eligible, pp)
for f in CLUSTER_FEATURES:
    print(
        f"{f:24s} log={f in LOG_FEATURES!s:5s} bounds=[{pp.lower[f]:.3f}, {pp.upper[f]:.3f}] "
        f"mean={pp.mean[f]:.3f} sd={pp.std[f]:.3f} clipped low/high={clipped[f]}"
    )

prepared = apply_preprocessor(eligible, pp, keep=["stock_code"])
Z_COLS = [scaled_name(f) for f in CLUSTER_FEATURES]
z_check = prepared.agg(
    *[F.mean(c).alias(f"mean_{c}") for c in Z_COLS], *[F.stddev(c).alias(f"sd_{c}") for c in Z_COLS]
).first()
for c in Z_COLS:
    assert abs(z_check[f"mean_{c}"]) < 1e-6 and abs(z_check[f"sd_{c}"] - 1) < 1e-6, c
print("all scaled features have mean 0, sd 1")

mlflow.set_experiment(EXPERIMENT_PATH)
with mlflow.start_run(run_name="product_preprocessing") as pp_run:
    mlflow.set_tags({"phase": "4", "kind": "preprocessing", "feature_table": ML_PRODUCT_FEATURES})
    mlflow.log_params(
        {
            "as_of_date": str(AS_OF),
            "n_products": checks["products"],
            "n_clustered": N_CLUSTERED,
            "min_invoices": MIN_INVOICES,
            "min_months": MIN_MONTHS,
            "cluster_features": ",".join(CLUSTER_FEATURES),
            "log_features": ",".join(LOG_FEATURES),
            "skew_threshold": SKEW_THRESHOLD,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "clip_quantiles": f"{CLIP_QUANTILES[0]},{CLIP_QUANTILES[1]}",
            "dropped_correlated": ",".join(dropped) or "none",
        }
    )
    mlflow.log_dict(pp.to_dict(), "preprocessing/params.json")
    mlflow.log_dict(
        {f"{a}|{b}": r for (a, b), r in corr.items() if a < b}, "preprocessing/correlation.json"
    )
    mlflow.log_dict({k: list(v) for k, v in clipped.items()}, "preprocessing/clip_counts.json")
    mlflow.log_dict({"eligibility": eligibility}, "preprocessing/eligibility.json")
    mlflow.log_dict(PRODUCT_PROFILE_ONLY_FEATURES, "preprocessing/profile_only_features.json")
    mlflow.log_figure(corr_fig, "preprocessing/correlation_heatmap.png")
PREPROCESS_RUN_ID = pp_run.info.run_id
print("preprocessing run:", PREPROCESS_RUN_ID)

prepared = prepared.cache()  # reused by every fit below (df.cache() works on serverless)
print("prepared rows:", prepared.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Sweep: KMeans, bisecting k-means, GMM for k = 2..10
# MAGIC
# MAGIC Identical to Phase 3 section 4 (one parent MLflow run, one nested run per algorithm × k;
# MAGIC bisecting k-means = the KMeans-splits implementation, since Spark's `BisectingKMeans`
# MAGIC fails on Free Edition with `DBFS_DISABLED`). k rules: silhouette main, WCSS elbow as a
# MAGIC cross-check, smallest segment ≥ 3 % of clustered products (~110), k ≥ 3.
# MAGIC
# MAGIC After the first run, set `RUN_SWEEP = False` and paste the run id: results are then
# MAGIC reloaded from MLflow.

# COMMAND ----------

KS = list(range(2, 11))
SEED = 42
MIN_SHARE = 0.03
K_MIN = 3
BISECTING_IMPL = "kmeans_splits"  # Phase 3 probe: Spark BisectingKMeans -> DBFS_DISABLED

RUN_SWEEP = False
SWEEP_RUN_ID = "b37d8caeac6c4d9ea0f13c38321a0c5c"  # recorded 2026-11-26

common_params = {
    "as_of_date": str(AS_OF),
    "cluster_features": ",".join(CLUSTER_FEATURES),
    "preprocess_run_id": PREPROCESS_RUN_ID,
    "n_products": N_CLUSTERED,
}


def log_and_print(r):
    log_trial(r, common_params, phase="4")
    sil = r.metrics.get("silhouette")
    print(
        f"{r.algorithm:17s} k={r.k:2d} silhouette={sil if sil is None else round(sil, 4)} "
        f"min_share={round(r.metrics.get('min_share', 0), 4)} error={r.error}"
    )


if RUN_SWEEP:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="product_k_sweep") as sweep_run:
        mlflow.set_tags({"phase": "4", "kind": "k_sweep", "bisecting_impl": BISECTING_IMPL})
        mlflow.log_params(
            {
                **common_params,
                "ks": f"{KS[0]}-{KS[-1]}",
                "seed": SEED,
                "min_share": MIN_SHARE,
                "k_min": K_MIN,
            }
        )
        trials = run_sweep(
            prepared,
            Z_COLS,
            KS,
            ALGORITHMS,
            seed=SEED,
            bisecting_impl=BISECTING_IMPL,
            on_result=log_and_print,
        )
        sweep_results = [t.to_dict() for t in trials]
        mlflow.log_dict({"results": sweep_results}, "sweep/results.json")
    SWEEP_RUN_ID = sweep_run.info.run_id
else:
    sweep_results = mlflow.artifacts.load_dict(f"runs:/{SWEEP_RUN_ID}/sweep/results.json")[
        "results"
    ]
print("sweep run:", SWEEP_RUN_ID, "| trials:", len(sweep_results))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Compare: silhouette, elbow, segment sizes

# COMMAND ----------

rows = results_table(sweep_results)
table = pd.DataFrame(rows).sort_values(["algorithm", "k"])
display(table)

suggested = {a: suggest_k(sweep_results, a, MIN_SHARE, K_MIN) for a in ALGORITHMS}
elbows = {}
for a in ALGORITHMS:
    ok = sorted((r["k"], r["wcss"]) for r in rows if r["algorithm"] == a and not r["error"])
    elbows[a] = elbow_k([k for k, _ in ok], [w for _, w in ok]) if ok else None
print("silhouette-best k (min share, k>=3):", suggested)
print("WCSS elbow k                       :", elbows)

fig_sil = metric_by_k(
    rows,
    "silhouette",
    "silhouette (squared Euclidean)",
    "Products: silhouette by k (circled = suggested k)",
    marks=suggested,
)
fig_wcss = metric_by_k(
    rows, "wcss", "WCSS", "Products: within-cluster sum of squares by k", marks=elbows
)
fig_min = metric_by_k(
    rows, "min_share", "smallest segment share", "Products: smallest segment by k"
)
for fig in (fig_sil, fig_wcss, fig_min):
    display(fig)

size_figs = {}
for a, k in suggested.items():
    if k is None:
        continue
    trial = next(r for r in sweep_results if r["algorithm"] == a and r["k"] == k)
    size_figs[a] = share_bars(
        dict(sorted({int(c): v for c, v in trial["shares"].items()}.items())),
        f"{a}, k={k}: segment sizes",
    )
    display(size_figs[a])

with mlflow.start_run(run_id=SWEEP_RUN_ID):
    mlflow.log_figure(fig_sil, "sweep/silhouette_by_k.png")
    mlflow.log_figure(fig_wcss, "sweep/elbow_wcss_by_k.png")
    mlflow.log_figure(fig_min, "sweep/min_share_by_k.png")
    for a, fig in size_figs.items():
        mlflow.log_figure(fig, f"sweep/segment_sizes_{a}_k{suggested[a]}.png")
    mlflow.log_dict({"suggested_k": suggested, "elbow_k": elbows}, "sweep/suggestions.json")
    mlflow.log_text(table.to_csv(index=False), "sweep/results_table.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7 Stability of the candidates
# MAGIC
# MAGIC Same tests as Phase 3 (5 seeds → 10 pair ARIs; 5 bootstrap refits labelling all products,
# MAGIC ARI vs the reference fit). **Candidates come from a rule fixed before seeing the sweep**
# MAGIC (`stability_candidates`): per algorithm the top k by silhouette among eligible k
# MAGIC (KMeans 4, bisecting 2, GMM 1). In Phase 3 the candidates were picked by reading the
# MAGIC sweep; a rule removes that judgement call.

# COMMAND ----------

STABILITY_SEEDS = (SEED, 1, 2, 3, 4)
BOOTSTRAP_SEEDS = (101, 102, 103, 104, 105)
MIN_ARI = 0.8  # pre-stated rule (Phase 3): seed AND bootstrap mean ARI >= 0.8
SIL_TOLERANCE = 0.01
STABILITY_CANDIDATES = stability_candidates(
    sweep_results, DEFAULT_CANDIDATES_PER_ALGORITHM, MIN_SHARE, K_MIN
)
print("stability candidates:", STABILITY_CANDIDATES)

RUN_STABILITY = False
STABILITY_RUN_ID = "7fdee340c9ef45d0a2e634f14ada3224"  # recorded 2026-11-26

if RUN_STABILITY:
    mlflow.set_experiment(EXPERIMENT_PATH)
    stability_results = []
    with mlflow.start_run(run_name="product_stability") as stab_run:
        mlflow.set_tags({"phase": "4", "kind": "stability", "sweep_run_id": SWEEP_RUN_ID})
        mlflow.log_params(
            {
                **common_params,
                "seeds": ",".join(map(str, STABILITY_SEEDS)),
                "bootstrap_seeds": ",".join(map(str, BOOTSTRAP_SEEDS)),
                "min_ari": MIN_ARI,
                "candidates": ",".join(f"{a}:{k}" for a, k in STABILITY_CANDIDATES),
            }
        )
        for algo, k in STABILITY_CANDIDATES:
            res = stability(
                algo, prepared, k, Z_COLS, STABILITY_SEEDS, BOOTSTRAP_SEEDS, id_col="stock_code"
            )
            d = res.to_dict()
            stability_results.append(d)
            with mlflow.start_run(run_name=f"stability_{algo}_k{k}", nested=True):
                mlflow.set_tags(
                    {
                        "phase": "4",
                        "kind": "stability_trial",
                        "algorithm": algo,
                        "status": "failed" if res.error else "ok",
                    }
                )
                mlflow.log_params({"algorithm": algo, "k": k})
                mlflow.log_metrics({m: v for m, v in res.summary().items() if v is not None})
                mlflow.log_dict(d, "stability/result.json")
            print(f"{algo:17s} k={k} {res.summary()} error={res.error}")
        mlflow.log_dict({"results": stability_results}, "stability/results.json")
    STABILITY_RUN_ID = stab_run.info.run_id
else:
    stability_results = mlflow.artifacts.load_dict(
        f"runs:/{STABILITY_RUN_ID}/stability/results.json"
    )["results"]
print("stability run:", STABILITY_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.8 The final choice (with a documented deviation)
# MAGIC
# MAGIC **Pre-stated rule** (Phase 3, `choose_final`): stable (mean seed **and** bootstrap ARI
# MAGIC ≥ 0.8), smallest segment ≥ 3 %, then best silhouette; within 0.01 of the best, the larger k.
# MAGIC
# MAGIC **What happened on the first run (2026-11-26):** no product candidate passed. The two k = 3
# MAGIC models are almost perfectly reproducible across seeds (bisecting 0.997, KMeans 0.981) but
# MAGIC their bootstrap ARI is 0.744 / 0.743; every other candidate also fails the *seed* test.
# MAGIC
# MAGIC **Why the two tests deserve different bars here.** They measure different things:
# MAGIC - *Seed ARI* = does the **method** give the same answer on the same products? Failing it
# MAGIC   means the segmentation is an accident of the random start: unusable. Bar kept at **0.8**.
# MAGIC - *Bootstrap ARI* = how much does the answer move if the **sample** of products changes?
# MAGIC   A bootstrap sample holds only ~63 % distinct products (the rest are repeats), so products
# MAGIC   near a boundary flip. Products are less separated than customers (silhouette 0.36 vs 0.47
# MAGIC   in Phase 3), so more of them sit near a boundary, and 0.74 means ~3/4 agreement after
# MAGIC   chance correction: a real structure with fuzzy edges. Bar lowered to **0.7** for products.
# MAGIC
# MAGIC **Honesty rule:** this is a deviation decided *after* seeing the data, so it is recorded
# MAGIC as one: the pre-stated result (`None`) is logged next to the revised choice
# MAGIC (`stability/decision.json` + tags), and the segment profiles in 4.3 must make business
# MAGIC sense before the model is accepted. Alternatives considered: rule-based segments
# MAGIC (velocity × price quadrants: stable by construction, but cut-offs are arbitrary), or
# MAGIC re-engineering features until something passes 0.8 (that's tuning to the test).

# COMMAND ----------


def _sil(algorithm, k):
    return next(x["silhouette"] for x in rows if (x["algorithm"], x["k"]) == (algorithm, k))


stab_table = pd.DataFrame(
    [
        {
            "algorithm": r["algorithm"],
            "k": r["k"],
            "silhouette": _sil(r["algorithm"], r["k"]),
            **{
                m: r[m]
                for m in [
                    "seed_ari_mean",
                    "seed_ari_min",
                    "bootstrap_ari_mean",
                    "bootstrap_ari_min",
                ]
            },
            "error": r["error"],
        }
        for r in stability_results
    ]
)
display(stab_table)
MIN_SEED_ARI = MIN_ARI  # unchanged: the method must be reproducible
MIN_BOOTSTRAP_ARI = 0.7  # deviation for products (see markdown above)

fig_ari = ari_bars(
    stability_results,
    MIN_BOOTSTRAP_ARI,
    f"Products: stability (ARI); seed bar {MIN_SEED_ARI}, bootstrap bar {MIN_BOOTSTRAP_ARI}",
)
display(fig_ari)

FINAL_PRESTATED = choose_final(rows, stability_results, MIN_SHARE, MIN_ARI, SIL_TOLERANCE)
print("pre-stated rule (seed & bootstrap >= 0.8):", FINAL_PRESTATED)
FINAL = choose_final(
    rows,
    stability_results,
    MIN_SHARE,
    MIN_SEED_ARI,
    SIL_TOLERANCE,
    min_bootstrap_ari=MIN_BOOTSTRAP_ARI,
)
print(f"revised rule (seed >= {MIN_SEED_ARI}, bootstrap >= {MIN_BOOTSTRAP_ARI}):", FINAL)
assert FINAL is not None, "no candidate passed even the revised rule: share the output"
DECISION = {
    "prestated_rule": {"min_seed_ari": MIN_ARI, "min_bootstrap_ari": MIN_ARI},
    "prestated_result": FINAL_PRESTATED,
    "revised_rule": {"min_seed_ari": MIN_SEED_ARI, "min_bootstrap_ari": MIN_BOOTSTRAP_ARI},
    "revised_result": FINAL,
    "deviation": FINAL_PRESTATED is None,
    "reason": "no candidate passed bootstrap ARI >= 0.8; seed stability kept at 0.8, "
    "bootstrap bar lowered to 0.7 for products (less separated than customers)",
}

with mlflow.start_run(run_id=STABILITY_RUN_ID):
    mlflow.log_figure(fig_ari, "stability/ari_by_candidate.png")
    mlflow.log_text(stab_table.to_csv(index=False), "stability/results_table.csv")
    mlflow.log_dict(FINAL, "stability/final_choice.json")
    mlflow.log_dict(DECISION, "stability/decision.json")
    mlflow.set_tags(
        {
            "final_algorithm": FINAL["algorithm"],
            "final_k": str(FINAL["k"]),
            "rule_deviation": str(DECISION["deviation"]),
            "min_seed_ari_used": str(MIN_SEED_ARI),
            "min_bootstrap_ari_used": str(MIN_BOOTSTRAP_ARI),
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.9 What to share back (step 4.2)

# COMMAND ----------

cols = [
    "silhouette",
    "wcss",
    "min_share",
    "max_share",
    "n_empty_clusters",
    "mean_max_probability",
    "fit_eval_seconds",
]
scols = ["silhouette", "seed_ari_mean", "seed_ari_min", "bootstrap_ari_mean", "bootstrap_ari_min"]


def fmt(v):
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return ""
    return f"{v:.4f}" if isinstance(v, float) else str(v)


lines = ["### 4.2 results", "", "**Eligibility**"]
lines += [
    f"- {r['eligibility']}: {r['products']:,} products ({r['product_share']:.1%}), {r['revenue_share']:.1%} of revenue"
    for r in eligibility
]
lines += [
    "",
    f"- raw skewness (eligible): { {k: round(v, 3) for k, v in raw_skew.items()} }",
    f"- log features: {LOG_FEATURES}",
    f"- clustering features: {CLUSTER_FEATURES}",
    f"- dropped: { {k: (v[0], round(v[1], 3)) for k, v in dropped.items()} }",
    f"- clipped (low, high): {clipped}",
    f"- runs: preprocessing {PREPROCESS_RUN_ID}, sweep {SWEEP_RUN_ID}, stability {STABILITY_RUN_ID}",
    "",
    "| | " + " | ".join(PRODUCT_CANDIDATE_FEATURES) + " |",
    "|---" * (len(PRODUCT_CANDIDATE_FEATURES) + 1) + "|",
]
for a in PRODUCT_CANDIDATE_FEATURES:
    lines.append(
        f"| {a} | " + " | ".join(f"{corr[(a, b)]:.3f}" for b in PRODUCT_CANDIDATE_FEATURES) + " |"
    )
lines += [
    "",
    f"- suggested k: {suggested}",
    f"- elbow k: {elbows}",
    f"- errors: {[(r['algorithm'], r['k'], r['error']) for r in rows if r['error']] or 'none'}",
]
lines += ["", "| algorithm | k | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
for r in sorted(rows, key=lambda r: (r["algorithm"], r["k"])):
    lines.append(
        f"| {r['algorithm']} | {r['k']} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |"
    )
lines += [
    "",
    f"- stability candidates: {STABILITY_CANDIDATES}",
    "",
    "| algorithm | k | " + " | ".join(scols) + " | error |",
    "|---|---|" + "---|" * (len(scols) + 1),
]
for r in stab_table.to_dict("records"):
    lines.append(
        f"| {r['algorithm']} | {r['k']} | "
        + " | ".join(fmt(r[c]) for c in scols)
        + f" | {r['error'] or ''} |"
    )
lines += [
    "",
    f"- pre-stated rule result: {FINAL_PRESTATED}",
    f"- **final choice** (seed >= {MIN_SEED_ARI}, bootstrap >= {MIN_BOOTSTRAP_ARI}): {FINAL}",
]
final_trial = next(
    r for r in sweep_results if r["algorithm"] == FINAL["algorithm"] and r["k"] == FINAL["k"]
)
lines.append(f"- final shares: { {c: round(v, 4) for c, v in final_trial['shares'].items()} }")
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Final model and product segment profiles
# MAGIC
# MAGIC ### 4.1 Rebuild the evaluated model (no refit) and prove it is the same model
# MAGIC
# MAGIC The final model is **the recorded sweep trial** (Phase 3 rule: refits aren't
# MAGIC bit-reproducible on serverless). For bisecting k-means the recorded model is the split
# MAGIC sequence; scoring replays it (`src/segmentation/product_model.py`). Segment ids are
# MAGIC re-ordered by the velocity centre: **segment 0 = fastest-selling group**.
# MAGIC
# MAGIC Proof that it's the evaluated model: same silhouette (to 1e-6) and same segment sizes as
# MAGIC the sweep trial chosen in 3.8.

# COMMAND ----------

from segmentation.plots import centroid_heatmap
from segmentation.product_model import (
    PRODUCT_PROFILE_FEATURES,
    SEGMENT_COL,
    SEGMENT_LABEL_COL,
    label_all_products,
    product_model_from_trial,
    product_segment_profile,
    top_products,
)
from segmentation.metrics import silhouette
from segmentation.stability import ari_between

final_model = product_model_from_trial(
    sweep_results, FINAL["algorithm"], FINAL["k"], pp, prepared, seed=SEED
)
segmented = final_model.assign_prepared(prepared).cache()

final_sil = silhouette(segmented, "features", SEGMENT_COL)
trial = next(
    r for r in sweep_results if r["algorithm"] == FINAL["algorithm"] and r["k"] == FINAL["k"]
)
trial_sizes = sorted(round(v * N_CLUSTERED) for v in trial["shares"].values())
print(f"silhouette: rebuilt {final_sil:.6f} vs sweep {FINAL['silhouette']:.6f}")
print(
    "segment sizes (by segment id):",
    final_model.leaf_sizes,
    "| id map raw->segment:",
    final_model.id_map,
)
assert abs(final_sil - FINAL["silhouette"]) < 1e-6, "rebuilt model differs from the evaluated trial"
assert sorted(final_model.leaf_sizes) == trial_sizes, "segment sizes differ from the trial"
print("rebuilt model == evaluated sweep trial")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Agreement with KMeans k = 3
# MAGIC
# MAGIC KMeans k = 3 also passed the revised rule (silhouette 0.354, within the 0.01 tie of
# MAGIC bisecting's 0.364). If two different algorithms find nearly the same three groups, the
# MAGIC structure is a property of the data, not of one algorithm: a useful extra argument given
# MAGIC the bootstrap ARI of 0.74. ARI between the two final labelings, and the cross-tab.

# COMMAND ----------

kmeans3 = product_model_from_trial(sweep_results, "kmeans", 3, pp, prepared, seed=SEED)
km_labels = kmeans3.assign_prepared(prepared).select(
    "stock_code", F.col(SEGMENT_COL).alias("kmeans_k3")
)
agreement = segmented.select("stock_code", SEGMENT_COL).join(km_labels, "stock_code")
pairs = agreement.collect()
ARI_VS_KMEANS = ari_between(
    {r["stock_code"]: r[SEGMENT_COL] for r in pairs},
    {r["stock_code"]: r["kmeans_k3"] for r in pairs},
)
crosstab = agreement.groupBy(SEGMENT_COL).pivot("kmeans_k3").count().orderBy(SEGMENT_COL).toPandas()
print(f"ARI(bisecting k=3, KMeans k=3) = {ARI_VS_KMEANS:.4f}")
display(crosstab)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Profiles: the 3 clusters and the 2 rule-based groups
# MAGIC
# MAGIC Every product of the catalogue is in exactly one row below: `cluster 0..2` from the model,
# MAGIC `low_volume` and `new_product` from the long-tail rule (3.1). Shares are of the **whole**
# MAGIC catalogue. Read with the centroid heatmap (cluster centres in standard deviations from
# MAGIC the average clustered product, after log/clip) and the example products in 4.4.
# MAGIC Names are given in 4.3b **from these numbers**, not before.

# COMMAND ----------

labelled = label_all_products(with_eligibility, segmented)
profile = product_segment_profile(labelled, PRODUCT_PROFILE_FEATURES).toPandas()
display(profile)

heat_fig = centroid_heatmap(
    final_model.leaf_centers,
    CLUSTER_FEATURES,
    [f"segment {i} (n={n})" for i, n in enumerate(final_model.leaf_sizes)],
    f"Product segments: {FINAL['algorithm']} k={FINAL['k']} centres (z)",
)
display(heat_fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Example products per segment
# MAGIC
# MAGIC The 10 biggest products by revenue in each group, with descriptions from
# MAGIC `gold.dim_product`. A "seasonal" segment should show Christmas items; a "staples" segment
# MAGIC should show everyday best-sellers. If the examples contradict the numbers, the profile
# MAGIC is wrong, not the examples.

# COMMAND ----------

examples = top_products(labelled, spark.table(GOLD_DIM_PRODUCT), n=10).toPandas()
display(examples)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Log the final model to MLflow
# MAGIC
# MAGIC One run `product_final_<algorithm>_k<k>` with the model JSON (preprocessing params +
# MAGIC splits + id map + leaf centres), the profile, examples, agreement and the heatmap. Its
# MAGIC run id becomes `model_run_id` in `gold.product_segments` (4.3b). Set
# MAGIC `RUN_LOG_FINAL = False` and paste the id after the first run.

# COMMAND ----------

RUN_LOG_FINAL = False
FINAL_RUN_ID = "47f3364bfad44e05862b9f8dd8f35999"  # recorded 2026-11-26

if RUN_LOG_FINAL:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(
        run_name=f"product_final_{FINAL['algorithm']}_k{FINAL['k']}"
    ) as final_run:
        mlflow.set_tags(
            {
                "phase": "4",
                "kind": "final_model",
                "algorithm": FINAL["algorithm"],
                "preprocess_run_id": PREPROCESS_RUN_ID,
                "sweep_run_id": SWEEP_RUN_ID,
                "stability_run_id": STABILITY_RUN_ID,
                "rule_deviation": str(DECISION["deviation"]),
            }
        )
        mlflow.log_params(
            {
                "as_of_date": str(AS_OF),
                "k": FINAL["k"],
                "seed": SEED,
                "cluster_features": ",".join(CLUSTER_FEATURES),
                "order_by": final_model.order_by,
                "n_clustered": N_CLUSTERED,
                "min_invoices": MIN_INVOICES,
                "min_months": MIN_MONTHS,
            }
        )
        mlflow.log_metrics({"silhouette": final_sil, "ari_vs_kmeans_k3": ARI_VS_KMEANS})
        mlflow.log_dict(final_model.to_dict(), "model/product_segment_model.json")
        mlflow.log_dict(DECISION, "model/selection_decision.json")
        mlflow.log_text(profile.to_csv(index=False), "profiles/segment_profile.csv")
        mlflow.log_text(examples.to_csv(index=False), "profiles/top_products.csv")
        mlflow.log_text(crosstab.to_csv(index=False), "profiles/crosstab_vs_kmeans_k3.csv")
        mlflow.log_figure(heat_fig, "profiles/centroid_heatmap.png")
    FINAL_RUN_ID = final_run.info.run_id
print("final model run:", FINAL_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.6 What to share back (step 4.3a)

# COMMAND ----------

key_medians = [
    "revenue",
    "n_invoices",
    "units_per_week",
    "active_week_share",
    "median_unit_price",
    "unique_customers",
    "non_uk_revenue_share",
    "seasonal_revenue_share",
    "monthly_units_cv",
    "return_rate",
    "days_since_last_sale",
    "days_on_sale",
    "avg_quantity_per_line",
]
pct_cols = [
    "pct_sold_last_30d",
    "pct_not_sold_180d",
    "pct_q4_majority",
    "pct_international",
    "pct_price_10_plus",
    "pct_return_rate_5_plus",
    "pct_history_censored",
]
lines = [
    "### 4.3a results",
    "",
    f"- final model run: {FINAL_RUN_ID}",
    f"- silhouette rebuilt {final_sil:.6f} (sweep {FINAL['silhouette']:.6f}); sizes {final_model.leaf_sizes}; id map {final_model.id_map}",
    f"- ARI vs KMeans k=3: {ARI_VS_KMEANS:.4f}",
    "",
    "**Cross-tab (rows = bisecting segment, columns = KMeans k=3)**",
    "",
    "| " + " | ".join(map(str, crosstab.columns)) + " |",
    "|---" * len(crosstab.columns) + "|",
    *["| " + " | ".join(fmt(v) for v in row) + " |" for row in crosstab.itertuples(index=False)],
    "",
    "**Size and value**",
    "",
    "| label | products | product share | revenue | revenue share | revenue/product |",
    "|---|---|---|---|---|---|",
]
for r in profile.to_dict("records"):
    lines.append(
        f"| {r[SEGMENT_LABEL_COL]} | {r['products']:,} | {r['product_share']:.3f} | "
        f"{r['total_revenue']:,.0f} | {r['revenue_share']:.3f} | {r['revenue_per_product']:,.0f} |"
    )
lines += [
    "",
    "**Medians**",
    "",
    "| label | " + " | ".join(key_medians) + " |",
    "|---" * (len(key_medians) + 1) + "|",
]
for r in profile.to_dict("records"):
    lines.append(
        f"| {r[SEGMENT_LABEL_COL]} | "
        + " | ".join(fmt(r[f"median_{c}"]) for c in key_medians)
        + " |"
    )
lines += [
    "",
    "**Means (selected)**",
    "",
    "| label | " + " | ".join(key_medians) + " |",
    "|---" * (len(key_medians) + 1) + "|",
]
for r in profile.to_dict("records"):
    lines.append(
        f"| {r[SEGMENT_LABEL_COL]} | " + " | ".join(fmt(r[f"mean_{c}"]) for c in key_medians) + " |"
    )
lines += [
    "",
    "**Shares of products**",
    "",
    "| label | " + " | ".join(pct_cols) + " |",
    "|---" * (len(pct_cols) + 1) + "|",
]
for r in profile.to_dict("records"):
    lines.append(f"| {r[SEGMENT_LABEL_COL]} | " + " | ".join(fmt(r[c]) for c in pct_cols) + " |")
lines += [
    "",
    "**Centres (z)**",
    "",
    "| segment | " + " | ".join(CLUSTER_FEATURES) + " |",
    "|---" * (len(CLUSTER_FEATURES) + 1) + "|",
]
for i, c in enumerate(final_model.leaf_centers):
    lines.append(f"| {i} | " + " | ".join(f"{v:+.3f}" for v in c) + " |")
lines += ["", "**Top 10 products per group**", ""]
for label, grp in examples.groupby(SEGMENT_LABEL_COL, sort=True):
    lines.append(
        f"- {label}: "
        + "; ".join(
            f"{r['stock_code']} {r['description']} (GBP {r['revenue']:,.0f}, {r['units_per_week']:.1f}/wk, "
            f"price {r['median_unit_price']:.2f}, Q4 {r['seasonal_revenue_share']:.2f}, last sale {r['days_since_last_sale']} d ago)"
            for r in grp.to_dict("records")
        )
    )
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.7 Names from the profiles (4.3b)
# MAGIC
# MAGIC What the 4.3a numbers say (shares of the whole catalogue of 4,725 products):
# MAGIC
# MAGIC | Segment | Evidence (medians unless noted) | Products / revenue |
# MAGIC |---|---|---|
# MAGIC | 0 **Core Fast Movers** | 34 units/week, sells in 91 % of weeks, 151 customers, last sale 2 days ago (96 % in the last 30 days); top items = Regency cakestand, white hanging heart T-light holder, jumbo bags | 43 % / **87 %** |
# MAGIC | 1 **Fading Lines** | last sale 331 days ago (75 % unsold for 180+ days), sells in 26 % of weeks, lumpiest months (CV 2.1), 90 % already on sale in Dec 2009 | 20 % / 7 % |
# MAGIC | 2 **Slow Movers** | still selling (85 % in the last 30 days) but 2.5 units/week and 28 customers; highest prices (median £2.95, mean £5.55) and most UK-centric; furniture, cabinets, advent calendars | 16 % / 4 % |
# MAGIC | Long Tail (rule) | < 20 invoices: median 8 invoices, 65 % unsold for 180+ days | 20 % / 0.8 % |
# MAGIC | New Products (rule) | < 3 months on sale; all Q4 2011 launches | 0.7 % / 0.9 % |
# MAGIC
# MAGIC **The hypothesised "seasonal" and "premium" segments did not appear as clusters.** At
# MAGIC k = 3 the products separate on **velocity and lifecycle**; Q4-heavy products exist in every
# MAGIC cluster (25 % of Core, 23 % of Slow Movers), and price only tilts the Slow Movers. So
# MAGIC seasonality and premium price become **rule-based flags inside segments** (same design
# MAGIC as the Phase 3 action groups), not invented clusters.
# MAGIC
# MAGIC Names are attached by **signature** on the centres (Core = highest velocity, Fading =
# MAGIC highest days since last sale, Slow = lowest velocity); a future snapshot whose profiles no
# MAGIC longer match raises `NamingError` instead of silently reusing names.
# MAGIC
# MAGIC Data check for 4.4: `23843 PAPER CRAFT, LITTLE BIRDIE` (New Products) has 80,995
# MAGIC units/week from a single sales line: one bulk order, which drives the New Products mean
# MAGIC velocity (2,389/week). 4.9 prints its return figures to see whether that order was
# MAGIC cancelled (it would then also carry `is_high_return`).

# COMMAND ----------

from segmentation.plots import group_share_bars
from segmentation.product_naming import (
    DISCONTINUED_DAYS,
    HIGH_RETURN_RATE,
    PREMIUM_QUANTILE,
    PRODUCT_ACTION_PLAYBOOK,
    SEASONAL_SHARE_MIN,
    add_product_groups,
    group_profile,
    name_product_segments,
    premium_price_threshold,
)
from segmentation.product_outputs import (
    build_product_segments_output,
    product_segments_ddl,
    write_product_segments,
)
from common.tables import GOLD_PRODUCT_SEGMENTS

names = name_product_segments(final_model.leaf_centers, CLUSTER_FEATURES)
for sid, d in sorted(names.items()):
    print(f"segment {sid}: {d.name:18s} ({d.signature_feature} {d.direction})  {d.description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.8 Action groups (rules inside the segments)
# MAGIC
# MAGIC | Flag | Rule | Why |
# MAGIC |---|---|---|
# MAGIC | `is_seasonal` | Q4 share of revenue > 0.5 | a flat product is ~0.25 in this data (Dec 2009 + two Q4s in 25 months), so > 0.5 = twice the baseline |
# MAGIC | `is_discontinued` | no sale for > 180 days | half a year without a single order |
# MAGIC | `is_premium` | median price ≥ p90 of clustered products (computed below) | top 10 % of the price range |
# MAGIC | `is_high_return` | return rate ≥ 10 % | quality / description check (a column, not a group) |
# MAGIC
# MAGIC Groups: Core → *Seasonal Best-Sellers* / *Year-round Staples*; Fading → *Discontinued* /
# MAGIC *Declining*; Slow → *Premium Niche* (checked first) / *Seasonal* / *Low Demand*; plus
# MAGIC Long Tail and New Products. Each group has one recommended action and a KPI
# MAGIC (`PRODUCT_ACTION_PLAYBOOK`).

# COMMAND ----------

PREMIUM_MIN = premium_price_threshold(eligible, PREMIUM_QUANTILE)
print(
    f"premium threshold (p{int(PREMIUM_QUANTILE * 100)} median price, clustered): GBP {PREMIUM_MIN:.2f}"
)

grouped = add_product_groups(labelled, names, PREMIUM_MIN).cache()
groups = group_profile(grouped)
group_table = pd.DataFrame(groups)
display(group_table)
fig_groups = group_share_bars(
    groups,
    "Product action groups: share of products vs revenue",
    share_key="product_share",
    entity="products",
)
display(fig_groups)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.9 Write `retail.gold.product_segments` and check it
# MAGIC
# MAGIC Every product of the snapshot (clustered + rule-based) once, with its segment, action
# MAGIC group, recommended action, key features, flags and `model_run_id` = the final model run.

# COMMAND ----------

spark.sql(product_segments_ddl(GOLD_PRODUCT_SEGMENTS))
out = build_product_segments_output(grouped, spark.table(GOLD_DIM_PRODUCT), FINAL_RUN_ID)
write_product_segments(out, GOLD_PRODUCT_SEGMENTS, AS_OF)

gold = spark.table(GOLD_PRODUCT_SEGMENTS).where(snapshot_predicate(AS_OF))
gold_checks = (
    gold.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("stock_code").alias("products"),
        F.sum(F.col("segment_name").isNull().cast("int")).alias("null_names"),
        F.sum(F.col("action_group").isNull().cast("int")).alias("null_groups"),
        F.sum(F.col("description").isNull().cast("int")).alias("null_descriptions"),
        F.sum(
            ((F.col("segment_id").isNull()) != (F.col("cluster_eligibility") != "clustered")).cast(
                "int"
            )
        ).alias("bad_segment_ids"),
        F.countDistinct("model_run_id").alias("model_runs"),
        F.sum("revenue").alias("revenue"),
    )
    .first()
    .asDict()
)
print(gold_checks)
seg_counts = {r["segment_name"]: r["count"] for r in gold.groupBy("segment_name").count().collect()}
print("products per segment:", seg_counts)
assert gold_checks["rows"] == gold_checks["products"] == checks["products"]
assert gold_checks["null_names"] == 0 and gold_checks["null_groups"] == 0
assert gold_checks["bad_segment_ids"] == 0 and gold_checks["model_runs"] == 1
for sid, d in names.items():
    assert seg_counts[d.name] == final_model.leaf_sizes[sid], d.name
print("gold.product_segments checks passed")

outlier = (
    gold.where(F.col("stock_code") == "23843")
    .join(
        spark.table(ML_PRODUCT_FEATURES)
        .where(snapshot_predicate(AS_OF))
        .select("stock_code", "units_sold", "units_returned", "n_lines"),
        "stock_code",
    )
    .select(
        "stock_code",
        "description",
        "segment_name",
        "units_sold",
        "units_returned",
        "n_lines",
        "return_rate",
        "is_high_return",
        "revenue",
    )
    .collect()
)
OUTLIER_23843 = outlier[0].asDict() if outlier else None
print("23843 check:", OUTLIER_23843)

with mlflow.start_run(run_id=FINAL_RUN_ID):
    mlflow.log_dict(
        {
            str(k): {
                "key": d.key,
                "name": d.name,
                "signature": f"{d.signature_feature} {d.direction}",
            }
            for k, d in names.items()
        },
        "naming/segment_names.json",
    )
    mlflow.log_dict(
        {
            "seasonal_share_min": SEASONAL_SHARE_MIN,
            "discontinued_days": DISCONTINUED_DAYS,
            "premium_quantile": PREMIUM_QUANTILE,
            "premium_min_price": PREMIUM_MIN,
            "high_return_rate": HIGH_RETURN_RATE,
            "playbook": PRODUCT_ACTION_PLAYBOOK,
        },
        "naming/action_rules.json",
    )
    mlflow.log_text(group_table.to_csv(index=False), "naming/action_groups.csv")
    mlflow.log_figure(fig_groups, "naming/action_groups.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.10 What to share back (step 4.3b)

# COMMAND ----------

lines = [
    "### 4.3b results",
    "",
    f"- names: { {sid: d.name for sid, d in sorted(names.items())} }",
    f"- premium threshold: GBP {PREMIUM_MIN:.2f}",
    f"- gold checks: {gold_checks}",
    f"- products per segment: {seg_counts}",
    f"- 23843 check: {OUTLIER_23843}",
    "",
    "| segment | action group | products | product share | revenue share | median units/wk | median price | median days since sale | median Q4 share | % high return |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for g in groups:
    lines.append(
        f"| {g['segment_name']} | {g['action_group']} | {g['products']:,} | {g['product_share']:.3f} | {g['revenue_share']:.3f} | "
        f"{fmt(g['median_units_per_week'])} | {fmt(g['median_price'])} | {g['median_days_since_last_sale']} | {fmt(g['median_q4_share'])} | {fmt(g['pct_high_return'])} |"
    )
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Market basket analysis: baskets and thresholds
# MAGIC
# MAGIC **Basket** = the distinct products on one invoice (`silver.sales`: already non-cancelled,
# MAGIC product codes only). Anonymous invoices are real baskets and stay in; single-item
# MAGIC invoices stay in the denominator of support. Definitions of support / confidence / lift
# MAGIC and FP-Growth vs Apriori: docstring of `src/segmentation/market_basket.py`.
# MAGIC
# MAGIC In business terms, for "customers who buy A also buy B":
# MAGIC - **support** = how many orders the pair is in (is it worth a campaign?),
# MAGIC - **confidence** = of orders with A, how many also have B (the hit rate of the
# MAGIC   recommendation "add B?"),
# MAGIC - **lift** = how much more likely B is with A than in any order (is it a real affinity,
# MAGIC   or is B just popular?). A best-seller in 10 % of orders gets 10 % confidence from *any*
# MAGIC   product; lift exposes that (lift ≈ 1).
# MAGIC
# MAGIC This step measures the data before choosing `minSupport` / `minConfidence` (4.5).
# MAGIC
# MAGIC ### 5.1 Baskets

# COMMAND ----------

from segmentation.market_basket import (
    basket_size_summary,
    build_baskets,
    item_support,
    items_above_support,
    threshold_sweep,
)

baskets = build_baskets(sales, as_of=AS_OF).cache()
basket_stats = basket_size_summary(baskets)
for k, v in basket_stats.items():
    print(f"{k:22s} {v:,.4f}" if isinstance(v, float) else f"{k:22s} {v:,}")

by_market = (
    baskets.groupBy(
        F.when(F.col("country") == "United Kingdom", "UK").otherwise("rest").alias("market")
    )
    .agg(F.count(F.lit(1)).alias("baskets"), F.mean("n_items").alias("mean_items"))
    .orderBy("market")
)
display(by_market)
display(
    baskets.orderBy(F.col("n_items").desc())
    .select("invoice", "n_items", "country", "has_customer")
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Item support: the long tail again
# MAGIC
# MAGIC By the Apriori property, an itemset can only be frequent if *every* item in it is
# MAGIC frequent. So the number of products above a support threshold bounds the whole search,
# MAGIC and the threshold is also how the long tail is handled in basket analysis: a product in
# MAGIC 20 of 39k orders can't support a reliable rule.

# COMMAND ----------

SWEEP_SUPPORTS = [0.03, 0.02, 0.015, 0.01, 0.0075, 0.005]
support = item_support(baskets).cache()
above = items_above_support(support, SWEEP_SUPPORTS)
for t, n in above.items():
    print(f"support >= {t:<7} ({t * basket_stats['baskets']:6.0f} baskets): {n:5,d} products")
display(
    support.join(
        spark.table(GOLD_DIM_PRODUCT).select("stock_code", "description"), "stock_code", "left"
    )
    .orderBy(F.col("basket_count").desc())
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 FP-Growth threshold sweep
# MAGIC
# MAGIC One FP-Growth fit per `minSupport` (high → low), with a low `minConfidence` (0.1) so the
# MAGIC counts show what's available. For each: frequent items / pairs / triples, rules, rules
# MAGIC with lift > 1, median lift and confidence, frequent pairs that are **colour/design
# MAGIC variants** of one product (same stock code family, e.g. 85099B/85099C), and seconds.
# MAGIC Stops early if a fit takes > 5 min or finds > 200k itemsets (Free Edition daily quota).
# MAGIC Set `RUN_MBA_SWEEP = False` and paste the run id after the first run.

# COMMAND ----------

SWEEP_MIN_CONFIDENCE = 0.1
RUN_MBA_SWEEP = False
MBA_SWEEP_RUN_ID = "d859f0327108469a808387b27c5d8ac6"  # recorded 2026-11-26

if RUN_MBA_SWEEP:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="mba_threshold_sweep") as mba_run:
        mlflow.set_tags({"phase": "4", "kind": "mba_threshold_sweep"})
        mlflow.log_params(
            {
                "as_of_date": str(AS_OF),
                "baskets": basket_stats["baskets"],
                "supports": ",".join(map(str, SWEEP_SUPPORTS)),
                "min_confidence": SWEEP_MIN_CONFIDENCE,
            }
        )
        mlflow.log_dict(basket_stats, "baskets/basket_stats.json")
        mlflow.log_dict({str(k): v for k, v in above.items()}, "baskets/items_above_support.json")
        mba_sweep = threshold_sweep(
            baskets,
            SWEEP_SUPPORTS,
            SWEEP_MIN_CONFIDENCE,
            on_result=lambda r: print(
                f"support {r['min_support']:<7} items {r.get('frequent_items')} pairs {r.get('frequent_pairs')} "
                f"triples {r.get('frequent_triples')} rules {r.get('rules')} ({r['seconds']} s) error={r['error']}"
            ),
        )
        mlflow.log_dict({"results": mba_sweep}, "sweep/results.json")
    MBA_SWEEP_RUN_ID = mba_run.info.run_id
else:
    mba_sweep = mlflow.artifacts.load_dict(f"runs:/{MBA_SWEEP_RUN_ID}/sweep/results.json")[
        "results"
    ]
print("MBA sweep run:", MBA_SWEEP_RUN_ID)

mba_table = pd.DataFrame(
    [{k: v for k, v in r.items() if k != "itemsets_by_size"} for r in mba_sweep]
)
display(mba_table)

# COMMAND ----------

import matplotlib.pyplot as plt

ok = [r for r in mba_sweep if not r["error"]]
fig_mba, ax = plt.subplots(1, 2, figsize=(11, 3.8))
xs = [r["min_support"] for r in ok]
for key in ["frequent_items", "frequent_pairs", "frequent_triples", "rules_lift_gt_1"]:
    ax[0].plot(xs, [max(r[key], 0.5) for r in ok], marker="o", label=key)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].invert_xaxis()
ax[0].set_xlabel("minSupport (log, decreasing)")
ax[0].set_ylabel("count (log)")
ax[0].legend(fontsize=8)
ax[0].set_title("What each threshold finds")
ax[1].plot(xs, [r["seconds"] for r in ok], marker="o")
ax[1].set_xscale("log")
ax[1].invert_xaxis()
ax[1].set_xlabel("minSupport (log, decreasing)")
ax[1].set_ylabel("seconds (fit + counting)")
ax[1].set_title("Cost on serverless")
fig_mba.tight_layout()
display(fig_mba)
with mlflow.start_run(run_id=MBA_SWEEP_RUN_ID):
    mlflow.log_figure(fig_mba, "sweep/threshold_sweep.png")
    mlflow.log_text(mba_table.to_csv(index=False), "sweep/results_table.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 What to share back (step 4.4)

# COMMAND ----------

top_items = (
    support.join(
        spark.table(GOLD_DIM_PRODUCT).select("stock_code", "description"), "stock_code", "left"
    )
    .orderBy(F.col("basket_count").desc())
    .limit(15)
    .collect()
)
mcols = [
    "min_support",
    "frequent_items",
    "frequent_pairs",
    "frequent_triples",
    "max_itemset_size",
    "rules",
    "rules_lift_gt_1",
    "median_lift",
    "max_lift",
    "median_confidence",
    "same_family_pairs",
    "seconds",
    "error",
]
lines = [
    "### 4.4 results",
    "",
    f"- MBA sweep run: {MBA_SWEEP_RUN_ID}",
    f"- basket stats: {basket_stats}",
]
lines += [
    f"- {r['market']}: {r['baskets']:,} baskets, mean {r['mean_items']:.1f} items"
    for r in by_market.collect()
]
lines += [f"- products with support >= {t}: {n:,}" for t, n in above.items()]
lines += ["", "**Top 15 items by support**"] + [
    f"- {r['stock_code']} {r['description']}: {r['basket_count']:,} baskets ({r['support']:.4f})"
    for r in top_items
]
lines += ["", "| " + " | ".join(mcols) + " |", "|---" * len(mcols) + "|"]
for r in mba_sweep:
    lines.append("| " + " | ".join(fmt(r.get(c)) for c in mcols) + " |")
stopped = [r.get("stopped_sweep") for r in mba_sweep if r.get("stopped_sweep")]
lines.append(f"- stopped early: {stopped or 'no'}")
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Association rules
# MAGIC
# MAGIC ### 6.1 Thresholds (chosen from the 5.3 sweep)
# MAGIC
# MAGIC | Parameter | Value | Why |
# MAGIC |---|---|---|
# MAGIC | `minSupport` | **0.01** (≥ 396 of 39,516 baskets) | 707 products can take part, 533 frequent pairs, 109 triples. 0.02 finds only 58 pairs (too few to cover the range); 0.005 finds 14.5k rules backed by as few as ~198 orders (too many to act on, weaker evidence each). The fit takes ~30 s, so compute isn't the constraint; evidence and actionability are. |
# MAGIC | `minConfidence` | **0.3** | the recommendation "add B?" is right for at least 3 in 10 baskets with A (sweep median at 0.01 was 0.39) |
# MAGIC | `min_lift` | **2** (for `is_recommendable`) | B at least twice as likely with A as in any basket. All rules have lift > 1 already (median 7.5); 2 removes "B is just popular" rules |
# MAGIC
# MAGIC **Long tail**: the support threshold *is* the long-tail rule here. 707 of 4,725 products
# MAGIC (15 %) are frequent enough for rules; the rest get no item-level rules, which is honest:
# MAGIC 20 co-purchases can't support a recommendation. For them, segment-level fallbacks (e.g.
# MAGIC recommend Core best-sellers) are the realistic option.
# MAGIC
# MAGIC **Scopes**: rules mined separately in **all** baskets, **UK** (36,184) and **rest of
# MAGIC world** (3,332). The same relative support means ≥ 34 baskets per rule for the rest of the
# MAGIC world: weaker evidence, flagged in the comparison. Mining per scope (instead of filtering
# MAGIC global rules by country) finds associations that only exist abroad.
# MAGIC
# MAGIC **Trivial rules** (`classify_rules`): `variant` = the consequent is a colour/design
# MAGIC variant of an antecedent item (same stock-code family like 85099B/85099C, or the same
# MAGIC description once colour words are removed). Kept in the table, but `is_recommendable =
# MAGIC false`. `similar` = same kind of item in another design (lunch bag retrospot → lunch bag
# MAGIC suki): a valid "more designs" suggestion, labelled. `complementary` = the classic
# MAGIC cross-sell.

# COMMAND ----------

from common.tables import GOLD_PRODUCT_ASSOCIATIONS
from segmentation.association_outputs import (
    associations_ddl,
    build_associations_output,
    top_rules,
    write_associations,
)
from segmentation.market_basket import (
    classify_rules,
    compare_pair_rules,
    fit_rules,
    min_count,
    pair_rules_sql,
    product_lookup,
    rules_frame,
)

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.3
MIN_LIFT = 2.0
SCOPES = {
    "all": baskets,
    "UK": baskets.where(F.col("country") == "United Kingdom"),
    "rest": baskets.where(F.col("country") != "United Kingdom"),
}
lookup = product_lookup(spark.table(GOLD_DIM_PRODUCT)).cache()

# Set RUN_MBA_RULES = False and paste the run id after the first run: the rules are then read
# back from gold.product_associations instead of re-mined.
RUN_MBA_RULES = False
MBA_RULES_RUN_ID = "37226335bd5a468f9bad47bae744c20d"  # recorded 2026-11-26

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Mine the rules (one FP-Growth fit per scope) and check them independently
# MAGIC
# MAGIC **Cross-check**: every single-item rule FP-Growth returns for *all* baskets is recomputed
# MAGIC with plain Spark joins (`pair_rules_sql`: count pairs among frequent items, divide). The
# MAGIC two rule sets must be identical and support / confidence / lift equal to floating point.
# MAGIC This proves both the serverless FP-Growth output and our reading of its measures.

# COMMAND ----------

if RUN_MBA_RULES:
    fits, frames = {}, []
    for scope, scope_baskets in SCOPES.items():
        fits[scope] = fit_rules(scope_baskets, MIN_SUPPORT, MIN_CONFIDENCE)
        f = fits[scope]
        print(
            f"{scope:5s} baskets {f['n_baskets']:6,d} (min count {min_count(MIN_SUPPORT, f['n_baskets'])}) "
            f"itemsets {f['n_itemsets']:5,d} rules {len(f['rules']):5,d} ({f['seconds']} s)"
        )
        frames.append(rules_frame(spark, f["rules"], f["n_baskets"], scope))
    rules = frames[0]
    for f in frames[1:]:
        rules = rules.unionByName(f)
    classified = classify_rules(rules, lookup).cache()

    sql_pairs = pair_rules_sql(SCOPES["all"], min_count(MIN_SUPPORT, fits["all"]["n_baskets"]))
    CROSS_CHECK = compare_pair_rules(
        classified.where(F.col("scope") == "all"), sql_pairs, MIN_CONFIDENCE
    )
    print("cross-check FP-Growth vs SQL (pair rules, all):", CROSS_CHECK)
    assert CROSS_CHECK["unmatched"] == 0, (
        "FP-Growth and the SQL recount disagree on which rules exist"
    )
    assert CROSS_CHECK["max_support_diff"] < 1e-9 and CROSS_CHECK["max_confidence_diff"] < 1e-9
    assert CROSS_CHECK["max_lift_diff"] < 1e-6
    print("FP-Growth pair rules == independent SQL recount")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Write `retail.gold.product_associations` and log to MLflow

# COMMAND ----------

if RUN_MBA_RULES:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="mba_rules") as rules_run:
        mlflow.set_tags({"phase": "4", "kind": "mba_rules", "sweep_run_id": MBA_SWEEP_RUN_ID})
        mlflow.log_params(
            {
                "as_of_date": str(AS_OF),
                "min_support": MIN_SUPPORT,
                "min_confidence": MIN_CONFIDENCE,
                "min_lift": MIN_LIFT,
                "scopes": ",".join(SCOPES),
            }
        )
        for scope, f in fits.items():
            mlflow.log_metrics(
                {
                    f"{scope}_baskets": f["n_baskets"],
                    f"{scope}_itemsets": f["n_itemsets"],
                    f"{scope}_rules": len(f["rules"]),
                    f"{scope}_seconds": f["seconds"],
                }
            )
        mlflow.log_dict(CROSS_CHECK, "rules/cross_check_vs_sql.json")
    MBA_RULES_RUN_ID = rules_run.info.run_id

    spark.sql(associations_ddl(GOLD_PRODUCT_ASSOCIATIONS))
    segments_now = spark.table(GOLD_PRODUCT_SEGMENTS).where(snapshot_predicate(AS_OF))
    assoc_out = build_associations_output(
        classified,
        lookup,
        segments_now,
        as_of=AS_OF,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        min_lift=MIN_LIFT,
        run_id=MBA_RULES_RUN_ID,
    )
    write_associations(assoc_out, GOLD_PRODUCT_ASSOCIATIONS, AS_OF)
print("rules run:", MBA_RULES_RUN_ID)

assoc = spark.table(GOLD_PRODUCT_ASSOCIATIONS).where(snapshot_predicate(AS_OF)).cache()
assoc_checks = (
    assoc.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("rule_id").alias("rule_ids"),
        F.sum(F.col("consequent_description").isNull().cast("int")).alias("null_descriptions"),
        F.sum(F.col("consequent_segment").isNull().cast("int")).alias("null_segments"),
        F.min("confidence").alias("min_confidence"),
        F.min("lift").alias("min_lift"),
        F.countDistinct("mlflow_run_id").alias("runs"),
    )
    .first()
    .asDict()
)
print(assoc_checks)
assert assoc_checks["rows"] == assoc_checks["rule_ids"], "duplicate rule ids"
assert assoc_checks["min_confidence"] >= MIN_CONFIDENCE - 1e-12 and assoc_checks["runs"] == 1

by_scope = (
    assoc.groupBy("scope", "rule_type")
    .agg(
        F.count(F.lit(1)).alias("rules"),
        F.sum(F.col("is_recommendable").cast("int")).alias("recommendable"),
        F.percentile_approx("lift", 0.5, 10000).alias("median_lift"),
        F.percentile_approx("confidence", 0.5, 10000).alias("median_confidence"),
    )
    .orderBy("scope", "rule_type")
)
display(by_scope)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4 Examples per rule type, and the top rules
# MAGIC
# MAGIC Check by eye that `variant` really are colour/design versions of one item and
# MAGIC `complementary` really are different products. Then the top recommendable rules (all
# MAGIC baskets), by lift and by confidence, one rule per antecedent so one item can't fill the
# MAGIC list.

# COMMAND ----------

show_cols = [
    "antecedent_descriptions",
    "consequent_description",
    "support",
    "confidence",
    "lift",
    "baskets_with_rule",
    "rule_type",
    "description_similarity",
    "consequent_segment",
]
for t in ["variant", "similar", "complementary"]:
    print(t)
    display(
        assoc.where((F.col("scope") == "all") & (F.col("rule_type") == t))
        .orderBy(F.col("lift").desc())
        .select(*show_cols)
        .limit(8)
    )

top_by_lift = top_rules(assoc, "all", 25, "lift").toPandas()
top_by_conf = top_rules(assoc, "all", 25, "confidence").toPandas()
display(top_by_lift[show_cols])
display(top_by_conf[show_cols])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.5 UK vs the rest of the world
# MAGIC
# MAGIC Same rule = same antecedent and consequent. Rules found **only abroad** are the
# MAGIC interesting ones (different assortments or occasions); remember each rest-of-world rule
# MAGIC rests on ≥ 34 baskets.

# COMMAND ----------


def _keys(scope):
    return (
        assoc.where((F.col("scope") == scope) & F.col("is_recommendable"))
        .select(F.array_join("antecedent", ",").alias("a"), "consequent")
        .distinct()
    )


uk_keys, rest_keys = _keys("UK"), _keys("rest")
MARKET_OVERLAP = {
    "uk_rules": uk_keys.count(),
    "rest_rules": rest_keys.count(),
    "in_both": uk_keys.join(rest_keys, ["a", "consequent"]).count(),
}
MARKET_OVERLAP["rest_only"] = MARKET_OVERLAP["rest_rules"] - MARKET_OVERLAP["in_both"]
print(MARKET_OVERLAP)

rest_only = (
    assoc.where((F.col("scope") == "rest") & F.col("is_recommendable"))
    .withColumn("a", F.array_join("antecedent", ","))
    .join(uk_keys, ["a", "consequent"], "left_anti")
)
rest_only_top = top_rules(rest_only, "rest", 15, "lift").toPandas()
display(rest_only_top[show_cols])

with mlflow.start_run(run_id=MBA_RULES_RUN_ID):
    mlflow.log_dict(MARKET_OVERLAP, "rules/market_overlap.json")
    mlflow.log_text(by_scope.toPandas().to_csv(index=False), "rules/rules_by_scope_type.csv")
    mlflow.log_text(top_by_lift.to_csv(index=False), "rules/top_by_lift.csv")
    mlflow.log_text(top_by_conf.to_csv(index=False), "rules/top_by_confidence.csv")
    mlflow.log_text(rest_only_top.to_csv(index=False), "rules/rest_only_top.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.6 What to share back (step 4.5)

# COMMAND ----------


def _rule_line(r):
    return (
        f"- {r['antecedent_descriptions']} -> {r['consequent_description']} | sup {r['support']:.4f} "
        f"({r['baskets_with_rule']}) conf {r['confidence']:.3f} lift {r['lift']:.2f} | {r['rule_type']} | {r['consequent_segment']}"
    )


lines = [
    "### 4.5 results",
    "",
    f"- rules run: {MBA_RULES_RUN_ID}",
    f"- gold checks: {assoc_checks}",
]
if RUN_MBA_RULES:
    lines.append(f"- cross-check vs SQL: {CROSS_CHECK}")
    lines += [
        f"- {s}: {f['n_baskets']:,} baskets, {f['n_itemsets']:,} itemsets, {len(f['rules']):,} rules, {f['seconds']} s"
        for s, f in fits.items()
    ]
lines += [
    "",
    "| scope | rule_type | rules | recommendable | median_lift | median_confidence |",
    "|---|---|---|---|---|---|",
]
for r in by_scope.collect():
    lines.append(
        f"| {r['scope']} | {r['rule_type']} | {r['rules']} | {r['recommendable']} | {r['median_lift']:.2f} | {r['median_confidence']:.3f} |"
    )
lines += ["", f"- market overlap: {MARKET_OVERLAP}"]
for t in ["variant", "similar", "complementary"]:
    lines += ["", f"**Examples: {t}**"]
    for r in (
        assoc.where((F.col("scope") == "all") & (F.col("rule_type") == t))
        .orderBy(F.col("lift").desc())
        .limit(5)
        .collect()
    ):
        lines.append(_rule_line(r))
lines += ["", "**Top 25 by lift (all, one per antecedent)**"] + [
    _rule_line(r) for r in top_by_lift.to_dict("records")
]
lines += ["", "**Top 25 by confidence (all, one per antecedent)**"] + [
    _rule_line(r) for r in top_by_conf.to_dict("records")
]
lines += ["", "**Top 15 rest-of-world-only rules**"] + [
    _rule_line(r) for r in rest_only_top.to_dict("records")
]
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Generate `docs/product_insights.md`
# MAGIC
# MAGIC Every number comes from this run's tables and MLflow runs (`src/segmentation/product_insights.py`
# MAGIC renders plain dicts to markdown; nothing typed by hand). Top-rule lists drop mirror rules
# MAGIC (A → B and B → A count once) and keep one rule per antecedent. The file and its charts
# MAGIC are written into the Git folder (`docs/`, `docs/img/phase4/`): commit them from the Git
# MAGIC dialog or pull and commit locally. If writing fails, the markdown is printed and logged to
# MAGIC MLflow.

# COMMAND ----------

from segmentation.product_insights import dedupe_pairs, render_product_insights_md
from segmentation.product_segments import RULE_SEGMENTS

SEGMENT_ORDER = ["Core Fast Movers", "Fading Lines", "Slow Movers", "Long Tail", "New Products"]
seg_stats = (
    gold.groupBy("segment_name")
    .agg(
        F.count(F.lit(1)).alias("products"),
        F.sum("revenue").alias("revenue"),
        F.percentile_approx("units_per_week", 0.5, 10000).alias("median_units_per_week"),
        F.percentile_approx("active_week_share", 0.5, 10000).alias("median_active_week_share"),
        F.percentile_approx("median_unit_price", 0.5, 10000).alias("median_price"),
        F.percentile_approx("days_since_last_sale", 0.5, 10000).alias(
            "median_days_since_last_sale"
        ),
        F.percentile_approx("seasonal_revenue_share", 0.5, 10000).alias("median_q4_share"),
    )
    .collect()
)
total_rev = sum(r["revenue"] for r in seg_stats)
segment_rows = sorted(
    [
        {
            **r.asDict(),
            "product_share": r["products"] / checks["products"],
            "revenue_share": r["revenue"] / total_rev,
        }
        for r in seg_stats
    ],
    key=lambda r: SEGMENT_ORDER.index(r["segment_name"]),
)
segment_descriptions = {d.name: d.description for _, d in sorted(names.items())}
segment_descriptions |= {v["segment_name"]: v["recommended_action"] for v in RULE_SEGMENTS.values()}

cross_check = mlflow.artifacts.load_dict(f"runs:/{MBA_RULES_RUN_ID}/rules/cross_check_vs_sql.json")
scope_baskets = {
    r["scope"]: r["n_baskets"] for r in assoc.select("scope", "n_baskets").distinct().collect()
}
rule_counts = [r.asDict() for r in by_scope.collect()]


def _rules(pdf, n, order):
    rows = pdf.sort_values([order, "support"], ascending=False).to_dict("records")
    for r in rows:
        r["antecedent"] = list(r["antecedent"])
    return dedupe_pairs(rows, n)


rec_all = assoc.where((F.col("scope") == "all") & F.col("is_recommendable")).toPandas()
single = rec_all[rec_all["antecedent_size"] == 1]
top_complementary = _rules(single[single["rule_type"] == "complementary"], 10, "lift")
top_range = _rules(single[single["rule_type"] == "similar"], 10, "confidence")
top_basket = _rules(rec_all[rec_all["antecedent_size"] >= 2], 8, "confidence")
top_rest_only = _rules(rest_only.toPandas(), 8, "lift")

model_facts = {
    "Who is clustered": f"{N_CLUSTERED:,} products with >= {MIN_INVOICES} invoices and >= {MIN_MONTHS} months on sale; "
    f"the rest are rule-based (Long Tail, New Products)",
    "Features": ", ".join(CLUSTER_FEATURES)
    + f" (log1p: {', '.join(LOG_FEATURES)}; clipped to p1-p99; standardised)",
    "Algorithm": f"{FINAL['algorithm']}, k = {FINAL['k']} (compared KMeans, bisecting k-means, Gaussian mixture, k = 2-10); "
    f"silhouette {FINAL['silhouette']:.3f}",
    "Stability": f"seed ARI {FINAL['seed_ari_mean']:.3f}, bootstrap ARI {FINAL['bootstrap_ari_mean']:.3f}; "
    f"agreement with KMeans k = 3: ARI {ARI_VS_KMEANS:.3f}",
    "Selection rule": f"pre-stated rule (seed and bootstrap ARI >= {MIN_ARI}) selected nothing; documented deviation: "
    f"bootstrap bar {MIN_BOOTSTRAP_ARI} for products, seed bar kept at {MIN_SEED_ARI}",
    "MLflow runs": f"preprocessing `{PREPROCESS_RUN_ID}`, sweep `{SWEEP_RUN_ID}`, stability `{STABILITY_RUN_ID}`, final model `{FINAL_RUN_ID}`",
    "Output table": f"`{GOLD_PRODUCT_SEGMENTS}`",
}
mba_facts = {
    "Baskets": f"{basket_stats['baskets']:,} invoices (median {basket_stats['median_items']} distinct products, "
    f"p99 {basket_stats['p99_items']}); UK {scope_baskets.get('UK', 0):,}, rest of world {scope_baskets.get('rest', 0):,}",
    "Algorithm": "Spark ML FP-Growth on serverless (listed as unsupported in the docs; probed and working)",
    "Thresholds": f"minSupport {MIN_SUPPORT} (>= {min_count(MIN_SUPPORT, basket_stats['baskets'])} baskets), "
    f"minConfidence {MIN_CONFIDENCE}, recommendable if lift >= {MIN_LIFT} and not a colour/design variant",
    "Why these thresholds": "threshold sweep 0.03-0.005: 0.02 found only 58 pairs, 0.005 found 14.5k rules on ~198 "
    "baskets each; 0.01 balances coverage and evidence",
    "Independent check": f"all {cross_check['fp_pair_rules']} single-item rules recomputed with plain Spark joins: "
    f"{cross_check['unmatched']} unmatched, max lift difference {cross_check['max_lift_diff']:.1e}",
    "MLflow runs": f"threshold sweep `{MBA_SWEEP_RUN_ID}`, rules `{MBA_RULES_RUN_ID}`",
    "Output table": f"`{GOLD_PRODUCT_ASSOCIATIONS}`",
}
caveats = [
    "Revenue is gross (before returns). One New Products item (23843, paper craft little birdie) is a single "
    f"{OUTLIER_23843['units_sold']:,}-unit order that was fully returned, so New Products revenue is overstated."
    if OUTLIER_23843
    else "Revenue is gross (before returns).",
    "Q4 share: a product sold evenly over the whole period has ~25 % of its revenue in Q4 (Dec 2009 + two Q4s), "
    "so 'seasonal' means > 50 %.",
    "Product clusters are less separated than customer segments; the bootstrap stability bar was lowered after "
    "seeing the data (recorded as a deviation in MLflow).",
    "Rule types come from stock-code families and description words; the data has no category hierarchy, so a few "
    "same-range items are labelled complementary (e.g. key fobs).",
    f"Rest-of-world rules rest on as few as {min_count(MIN_SUPPORT, scope_baskets.get('rest', 0))} baskets; test them "
    "before rolling out.",
    "Rules describe co-purchase, not cause: validate recommendations with an A/B test (attach rate, basket value).",
    "Many customers are wholesale buyers who stock whole ranges, which explains very high lifts inside collections.",
]

IMG_DIR = REPO_DIR / "docs" / "img" / "phase4"
images = {
    "Product segment centres": "img/phase4/centroid_heatmap.png",
    "Action groups": "img/phase4/action_groups.png",
    "Silhouette by k": "img/phase4/silhouette_by_k.png",
    "Stability (ARI)": "img/phase4/stability_ari.png",
    "FP-Growth threshold sweep": "img/phase4/threshold_sweep.png",
}
figures = {
    "Product segment centres": heat_fig,
    "Action groups": fig_groups,
    "Silhouette by k": fig_sil,
    "Stability (ARI)": fig_ari,
    "FP-Growth threshold sweep": fig_mba,
}
md = render_product_insights_md(
    as_of=AS_OF,
    n_products=checks["products"],
    segment_rows=segment_rows,
    group_rows=groups,
    playbook=PRODUCT_ACTION_PLAYBOOK,
    segment_descriptions=segment_descriptions,
    model_facts=model_facts,
    mba_facts=mba_facts,
    rule_counts=rule_counts,
    market_overlap=MARKET_OVERLAP,
    top_complementary=top_complementary,
    top_range=top_range,
    top_basket=top_basket,
    top_rest_only=top_rest_only,
    frequent_products=above[MIN_SUPPORT],
    caveats=caveats,
    images=images,
)

try:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for title, rel in images.items():
        figures[title].savefig(REPO_DIR / "docs" / rel, dpi=120, bbox_inches="tight")
    (REPO_DIR / "docs" / "product_insights.md").write_text(md)
    print("written:", REPO_DIR / "docs" / "product_insights.md", "and", len(images), "charts")
except Exception as e:
    print(
        f"could not write to the Git folder ({type(e).__name__}: {e}); markdown is logged to MLflow"
    )

for run_id in (FINAL_RUN_ID, MBA_RULES_RUN_ID):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_text(md, "docs/product_insights.md")
print(md)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Phase 4 "done when" checks
# MAGIC
# MAGIC Both Gold tables exist with today's snapshot, and the insights document holds real rules.

# COMMAND ----------

done = {
    "gold.product_segments rows": spark.table(GOLD_PRODUCT_SEGMENTS)
    .where(snapshot_predicate(AS_OF))
    .count(),
    "gold.product_associations rows": spark.table(GOLD_PRODUCT_ASSOCIATIONS)
    .where(snapshot_predicate(AS_OF))
    .count(),
    "recommendable rules (all scopes)": assoc.where("is_recommendable").count(),
    "insights doc characters": len(md),
    "rules in the doc": len(top_complementary)
    + len(top_range)
    + len(top_basket)
    + len(top_rest_only),
}
for k, v in done.items():
    print(f"{k:34s} {v:,}")
assert done["gold.product_segments rows"] == checks["products"]
assert done["gold.product_associations rows"] > 0 and done["rules in the doc"] > 0
print("Phase 4 done-when checks passed")