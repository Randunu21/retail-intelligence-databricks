# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Customer segmentation
# MAGIC
# MAGIC Goal: describe every known customer with a small set of behavioural features, cluster
# MAGIC them, and turn the clusters into **named business segments with actions**.
# MAGIC
# MAGIC | Section | What it does |
# MAGIC |---|---|
# MAGIC | 1 | **Free Edition probe**: Spark ML on serverless, MLflow experiment, Feature Engineering client |
# MAGIC | 2 | **Customer feature table** `retail.ml.customer_features` (feature table in Unity Catalog) |
# MAGIC | 3 | **Preprocessing**: feature choice, log transforms, correlation pruning, clipping, scaling |
# MAGIC | 4 | **Model selection**: KMeans / bisecting k-means / GMM for k = 2..10, every run in MLflow |
# MAGIC | 5+ | Stability, interpretation, outputs (added in later steps) |
# MAGIC
# MAGIC Logic lives in `src/features/` and `src/segmentation/` (unit-tested); this notebook runs it
# MAGIC and explains each decision.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install the Feature Engineering client
# MAGIC
# MAGIC Serverless notebooks don't ship `databricks-feature-engineering` (it's pre-installed only
# MAGIC on ML runtimes), so it's a notebook-scoped `%pip install`. `restartPython()` then reloads
# MAGIC Python so the new package is importable; it wipes variables, which is why this is the
# MAGIC first cell.
# MAGIC
# MAGIC **If this cell fails** (e.g. PyPI not reachable), comment it out and carry on: the
# MAGIC feature table doesn't depend on the client (see section 2), only the confirmation in 2.4.

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
    ML_CUSTOMER_FEATURES,
    SILVER_RETURNS,
    SILVER_SALES,
    user_experiment_path,
)
from features.customer_features import (
    FEATURE_COLUMNS,
    build_customer_features,
    default_as_of,
    describe_features,
    reconcile_features,
)
from features.feature_store import (
    describe_feature_table,
    feature_engineering_client_status,
    feature_table_ddl,
    snapshot_predicate,
    write_snapshot,
)

USER = spark.sql("SELECT current_user() AS u").first()["u"]
EXPERIMENT_PATH = user_experiment_path(USER)
print("user:", USER)
print("MLflow experiment:", EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Free Edition probe
# MAGIC
# MAGIC Never assume a feature works on Free Edition: each check below runs a tiny real call and
# MAGIC records `ok` or the exact error. What the docs say (checked 2026-11-24):
# MAGIC
# MAGIC - **Spark ML** (`pyspark.ml`) is supported on serverless from environment version 4 (this
# MAGIC   notebook pins 6). Limits: one model max 100 MB, 1 GB of in-memory models per session.
# MAGIC   Our models are a few centroids, i.e. bytes. (Also noted: `FPGrowthModel` is *not*
# MAGIC   supported on serverless, which matters for Phase 4.)
# MAGIC - **Feature Engineering in UC**: any Delta table with a PRIMARY KEY in Unity Catalog *is* a
# MAGIC   feature table. Online tables/feature serving are not on Free Edition (not needed here).
# MAGIC - **MLflow**: experiments live in the workspace; tracking URI `databricks` by default.

# COMMAND ----------

probe = {
    "spark_version": spark.version,
    "python": platform.python_version(),
    "mlflow": mlflow.__version__,
}

# 1a. Spark ML: fit the three algorithms on 6 points and score a silhouette.
from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

tiny = spark.createDataFrame(
    [(0.0, 0.0), (0.1, 0.2), (0.2, 0.1), (5.0, 5.0), (5.1, 5.2), (5.2, 5.1)], "x DOUBLE, y DOUBLE"
)
tiny_vec = VectorAssembler(inputCols=["x", "y"], outputCol="features").transform(tiny)
for name, algo in [
    ("spark_ml_kmeans", KMeans(k=2, seed=1)),
    ("spark_ml_bisecting_kmeans", BisectingKMeans(k=2, seed=1)),
    ("spark_ml_gmm", GaussianMixture(k=2, seed=1)),
]:
    try:
        pred = algo.fit(tiny_vec).transform(tiny_vec)
        sil = ClusteringEvaluator().evaluate(pred)
        probe[name] = f"ok (silhouette {sil:.3f})"
    except Exception as e:
        probe[name] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

# 1b. MLflow: create/select the experiment and log one run with a param, metric, figure, dict.
try:
    import matplotlib.pyplot as plt

    exp = mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="probe_free_edition") as run:
        mlflow.set_tag("phase", "3")
        mlflow.set_tag("kind", "probe")
        mlflow.log_param("probe_param", 1)
        mlflow.log_metric("probe_metric", 0.5)
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([1, 2, 3], [3, 1, 2])
        mlflow.log_figure(fig, "probe/figure.png")
        plt.close(fig)
        mlflow.log_dict({"hello": "free edition"}, "probe/info.json")
    logged = mlflow.get_run(run.info.run_id)
    probe["mlflow_experiment_id"] = exp.experiment_id
    probe["mlflow_run"] = (
        f"ok (run {run.info.run_id[:8]}, metric {logged.data.metrics.get('probe_metric')}, "
        f"artifacts {[a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id, artifact_path='probe')]})"
    )
except Exception as e:
    probe["mlflow_run"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

# 1c. Feature Engineering client importable?
probe["feature_engineering_client"] = feature_engineering_client_status()

for k, v in probe.items():
    print(f"{k:28s} {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Customer feature table
# MAGIC
# MAGIC ### 2.1 Snapshot date and source tables
# MAGIC
# MAGIC Features are computed **as of** a snapshot date: only sales/returns dated *strictly
# MAGIC before* it are used. Here it's "today" in this dataset, the day after the last sale. In
# MAGIC Phase 5 the same function runs with an earlier cutoff, so the propensity model's features
# MAGIC never see its label window (that's the leakage guard).
# MAGIC
# MAGIC Source = **Silver** (`silver.sales` + `silver.returns`), not Gold: returns only exist in
# MAGIC Silver, and reading both from the same layer keeps them consistent (same customer ids,
# MAGIC same cleaning). Lines without a customer id are excluded (they can't be profiled).

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
# MAGIC Definitions are in the docstring of `src/features/customer_features.py`. Three worth
# MAGIC knowing for an interview:
# MAGIC
# MAGIC - **Mean days between purchases** needs no window function: the mean of consecutive gaps
# MAGIC   between sorted purchase days telescopes to `(last - first) / (n_days - 1)`, so it comes out
# MAGIC   of the same single `groupBy` as RFM. NULL for customers with one purchase day (handled in
# MAGIC   preprocessing, 3.x).
# MAGIC - **Return rate** is value-based (`returned_value / monetary`) and capped at 1: some
# MAGIC   customers return goods bought before the data starts.
# MAGIC - **Seasonal share** = share of revenue in Oct-Dec (gift retailer, Christmas peak). Caveat:
# MAGIC   a customer whose whole history is in Q4 2011 gets 1.0, so read it with tenure.
# MAGIC
# MAGIC Then independent recounts (`reconcile_features`): one row per customer, the right number of
# MAGIC customers, monetary totals equal Silver, NULL intervals exactly for single-day customers.

# COMMAND ----------

features = build_customer_features(sales, returns, AS_OF, data_start=DATA_START)
checks = reconcile_features(features, sales, returns, AS_OF)
for k, v in checks.items():
    print(f"{k:24s} {v}")

assert checks["customers"] == checks["expected_customers"], "customer count mismatch"
assert checks["duplicate_keys"] == 0, "duplicate customer_id"
assert abs(checks["monetary_diff"]) < 0.01, "monetary does not reconcile with Silver"
assert checks["single_day_customers"] == checks["null_intervals"], "interval NULLs unexpected"
lo, hi = checks["return_rate_range"]
assert lo >= 0.0 and hi <= 1.0, "return_rate outside [0, 1]"
print("all feature checks passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Create the feature table and write the snapshot
# MAGIC
# MAGIC **Feature table in Unity Catalog = Delta table + PRIMARY KEY.** The DDL below declares
# MAGIC `PRIMARY KEY (customer_id, as_of_date TIMESERIES)`:
# MAGIC
# MAGIC - `customer_id` identifies the entity; `as_of_date` is a **time-series key**, so many
# MAGIC   snapshots can live in one table and Phase 5 can do *point-in-time* lookups.
# MAGIC - PK constraints in Databricks are **informational** (not enforced on write), which is why
# MAGIC   2.2 checks uniqueness itself. `NOT NULL` on the key columns *is* enforced.
# MAGIC - Writing uses `replaceWhere as_of_date = ...`: a re-run replaces only this snapshot and
# MAGIC   keeps the others (idempotent).
# MAGIC
# MAGIC Alternatives (interview point): a plain Delta table (no discovery/lineage, joins written by
# MAGIC hand), `FeatureEngineeringClient.create_table` (same result, but needs the client
# MAGIC everywhere), or an online store for real-time serving (not available on Free Edition and
# MAGIC not needed for batch segmentation).

# COMMAND ----------

ddl = feature_table_ddl(ML_CUSTOMER_FEATURES)
print(ddl)
spark.sql(ddl)
write_snapshot(features, ML_CUSTOMER_FEATURES, AS_OF)

snapshot = spark.table(ML_CUSTOMER_FEATURES).where(snapshot_predicate(AS_OF))
n_written = snapshot.count()
print(f"rows written for {AS_OF}: {n_written:,}")
assert n_written == checks["customers"]
display(
    spark.table(ML_CUSTOMER_FEATURES)
    .groupBy("as_of_date")
    .agg(F.count(F.lit(1)).alias("rows"), F.max("computed_at").alias("computed_at"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Is it a feature table? (two independent confirmations)
# MAGIC
# MAGIC 1. Unity Catalog's `information_schema` shows the PRIMARY KEY constraint.
# MAGIC 2. The Feature Engineering client (if installed) loads it with its keys.
# MAGIC
# MAGIC Also look in the UI: **Catalog → retail → ml → customer_features**, and the **Features**
# MAGIC page in the left sidebar under *Machine Learning*.

# COMMAND ----------

schema_name, table_name = ML_CUSTOMER_FEATURES.split(".")[1:]
constraints = spark.sql(f"""
    SELECT tc.constraint_name, tc.constraint_type, kcu.column_name, kcu.ordinal_position
    FROM {CATALOG}.information_schema.table_constraints tc
    JOIN {CATALOG}.information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
    WHERE tc.table_schema = '{schema_name}' AND tc.table_name = '{table_name}'
    ORDER BY kcu.ordinal_position
""")
display(constraints)

fe_status = feature_engineering_client_status()
if fe_status["available"]:
    try:
        fe_info = describe_feature_table(ML_CUSTOMER_FEATURES)
    except Exception as e:
        fe_info = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
else:
    fe_info = {"client": fe_status}
print(fe_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Feature distributions (input to preprocessing)
# MAGIC
# MAGIC Skewness tells us which features need a log transform before clustering: distance-based
# MAGIC algorithms (KMeans) would otherwise be dominated by a handful of very large customers.
# MAGIC Rule of thumb: |skewness| > 1 is strongly skewed.

# COMMAND ----------

NUMERIC_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_basket_value",
    "distinct_products",
    "returned_value",
    "return_rate",
    "tenure_days",
    "n_purchase_days",
    "mean_days_between_purchases",
    "n_countries",
    "seasonal_revenue_share",
]
assert set(NUMERIC_FEATURES) <= set(FEATURE_COLUMNS)
desc = describe_features(snapshot, NUMERIC_FEATURES)
display(desc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Summary to paste back into the chat

# COMMAND ----------


def _fmt(v):
    return f"{v:,.3f}" if isinstance(v, float) else str(v)


extra = snapshot.agg(
    F.sum(F.col("is_tenure_censored").cast("int")).alias("tenure_censored"),
    F.sum((F.col("frequency") == 1).cast("int")).alias("one_invoice_customers"),
    F.sum((F.col("n_return_invoices") > 0).cast("int")).alias("customers_with_returns"),
    F.countDistinct("preferred_country").alias("preferred_countries"),
    F.sum((F.col("preferred_country") == "United Kingdom").cast("int")).alias("uk_customers"),
).first()

lines = ["### Phase 3.1 results", "", f"- as_of: {AS_OF}, data start: {DATA_START}"]
lines += [f"- probe `{k}`: {v}" for k, v in probe.items()]
lines += [f"- check `{k}`: {v}" for k, v in checks.items()]
lines += [f"- {k}: {v}" for k, v in extra.asDict().items()]
lines += [f"- constraints: {[tuple(r) for r in constraints.collect()]}", f"- FE client: {fe_info}"]
lines += [
    "",
    "| feature | n_null | min | p50 | p95 | p99 | max | mean | skewness |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in desc.collect():
    lines.append(
        f"| {r['feature']} | {r['n_null']} | {_fmt(r['min'])} | {_fmt(r['p50'])} | {_fmt(r['p95'])} | "
        f"{_fmt(r['p99'])} | {_fmt(r['max'])} | {_fmt(r['mean'])} | {_fmt(r['skewness'])} |"
    )
print("\n".join(lines))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Preprocessing
# MAGIC
# MAGIC Clustering groups customers by **distance**, so what goes into the distance matters more
# MAGIC than which algorithm we pick. Four decisions, each a pure function in
# MAGIC `src/segmentation/preprocessing.py`:
# MAGIC
# MAGIC | Step | Choice | Why (and the alternative) |
# MAGIC |---|---|---|
# MAGIC | Feature choice | 7 behavioural candidates in priority order, near-duplicates pruned | Fewer, independent, explainable dimensions. Alternative: PCA, which removes redundancy but makes segments hard to explain to the business |
# MAGIC | Skew | `log1p` where raw skewness > 1 | Money/counts have long tails (monetary skew 25). Alternative: rank/quantile transform (RFM-score style): robust but throws away *how much* bigger |
# MAGIC | Outliers | clip to [p1, p99] after the log, **keep** the customer | The top customers are real and valuable; removing them would leave them unsegmented. Alternative: drop/separate them, or a robust scaler |
# MAGIC | Scale | standardise (mean 0, sd 1) | Equal weight per feature. Alternative: min-max (one outlier squeezes everyone else) |
# MAGIC
# MAGIC Kept in the feature table but **not** clustered on (used to *profile* segments in step 5):

# COMMAND ----------

from segmentation.plots import correlation_heatmap
from segmentation.preprocessing import (
    CANDIDATE_FEATURES,
    CLIP_QUANTILES,
    CORRELATION_THRESHOLD,
    PROFILE_ONLY_FEATURES,
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

for feature, reason in PROFILE_ONLY_FEATURES.items():
    print(f"profile only  {feature:30s} {reason}")

customers = spark.table(ML_CUSTOMER_FEATURES).where(snapshot_predicate(AS_OF))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Log transform: which features?
# MAGIC
# MAGIC Rule: skewness > 1 on the raw scale. Recency (0.89), tenure (-0.62) and seasonal share
# MAGIC (0.50) stay linear: their ranges are bounded (days in the data, a share in [0, 1]).

# COMMAND ----------

raw_skew = {
    r["feature"]: r["skewness"] for r in describe_features(customers, CANDIDATE_FEATURES).collect()
}
LOG_CANDIDATES = choose_log_features(raw_skew, CANDIDATE_FEATURES, SKEW_THRESHOLD)
print("log1p applied to:", LOG_CANDIDATES)

logged = log_transform(customers.select("customer_id", *CANDIDATE_FEATURES), LOG_CANDIDATES)
display(describe_features(logged, CANDIDATE_FEATURES))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Correlation check and pruning
# MAGIC
# MAGIC Correlation is measured on the **transformed** scale (what the algorithm will see).
# MAGIC Walking the candidates in priority order, a feature is dropped if |r| > 0.9 with one
# MAGIC already kept. Two highly correlated features would count the same behaviour twice in the
# MAGIC distance. `n_purchase_days` is in the list on purpose, as a near-copy of `frequency`.
# MAGIC
# MAGIC Pairwise correlation can't see *combinations*: `log(avg_basket) = log(monetary) -
# MAGIC log(frequency)` is an exact identity, so basket value is excluded by design (above),
# MAGIC not by this check.

# COMMAND ----------

corr = correlation_matrix(logged, CANDIDATE_FEATURES)
CLUSTER_FEATURES, dropped = prune_correlated(CANDIDATE_FEATURES, corr, CORRELATION_THRESHOLD)
LOG_FEATURES = [f for f in LOG_CANDIDATES if f in CLUSTER_FEATURES]
print("clustering features:", CLUSTER_FEATURES)
print("log features      :", LOG_FEATURES)
print("dropped           :", {k: (v[0], round(v[1], 3)) for k, v in dropped.items()})
corr_fig = correlation_heatmap(corr, CANDIDATE_FEATURES, "Correlation (log1p where skewed)")
display(corr_fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Fit clipping + scaling, apply, verify
# MAGIC
# MAGIC The fitted parameters (bounds, means, sds) are a small JSON dict. The same dict is logged
# MAGIC to MLflow with every model run, and Phase 6 scoring re-applies it **without refitting**,
# MAGIC so new customers are scaled exactly like the training customers.

# COMMAND ----------

pp = fit_preprocessor(customers, CLUSTER_FEATURES, LOG_FEATURES, CLIP_QUANTILES)
clipped = clip_counts(customers, pp)
for f in CLUSTER_FEATURES:
    print(
        f"{f:24s} log={f in LOG_FEATURES!s:5s} bounds=[{pp.lower[f]:.3f}, {pp.upper[f]:.3f}] "
        f"mean={pp.mean[f]:.3f} sd={pp.std[f]:.3f} clipped low/high={clipped[f]}"
    )

prepared = apply_preprocessor(customers, pp, keep=["customer_id"])
Z_COLS = [scaled_name(f) for f in CLUSTER_FEATURES]
z_check = prepared.agg(
    *[F.mean(c).alias(f"mean_{c}") for c in Z_COLS], *[F.stddev(c).alias(f"sd_{c}") for c in Z_COLS]
).first()
for c in Z_COLS:
    assert abs(z_check[f"mean_{c}"]) < 1e-6 and abs(z_check[f"sd_{c}"] - 1) < 1e-6, c
print("all scaled features have mean 0, sd 1")
display(describe_features(prepared, Z_COLS))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Log the preprocessing to MLflow, cache the prepared data
# MAGIC
# MAGIC `prepared` (5,852 rows) is reused by every model fit in sections 4-5, so it's cached
# MAGIC (`df.cache()` works on serverless, to local disk, for this session; Phase 2 probe).

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_PATH)
with mlflow.start_run(run_name="preprocessing") as pp_run:
    mlflow.set_tags({"phase": "3", "kind": "preprocessing", "feature_table": ML_CUSTOMER_FEATURES})
    mlflow.log_params(
        {
            "as_of_date": str(AS_OF),
            "n_customers": checks["customers"],
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
    mlflow.log_dict(PROFILE_ONLY_FEATURES, "preprocessing/profile_only_features.json")
    mlflow.log_figure(corr_fig, "preprocessing/correlation_heatmap.png")
PREPROCESS_RUN_ID = pp_run.info.run_id
print("preprocessing run:", PREPROCESS_RUN_ID)

prepared = prepared.cache()
print("prepared rows:", prepared.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Summary to paste back into the chat

# COMMAND ----------

lines = ["### Phase 3.2 results", ""]
lines.append(f"- raw skewness: { {k: round(v, 3) for k, v in raw_skew.items()} }")
lines.append(f"- log features: {LOG_FEATURES}")
lines.append(f"- clustering features: {CLUSTER_FEATURES}")
lines.append(f"- dropped: { {k: (v[0], round(v[1], 3)) for k, v in dropped.items()} }")
lines.append(f"- clipped (low, high): {clipped}")
lines.append(f"- preprocessing run: {PREPROCESS_RUN_ID}")
lines += [
    "",
    "| | " + " | ".join(CANDIDATE_FEATURES) + " |",
    "|---" * (len(CANDIDATE_FEATURES) + 1) + "|",
]
for a in CANDIDATE_FEATURES:
    lines.append(f"| {a} | " + " | ".join(f"{corr[(a, b)]:.3f}" for b in CANDIDATE_FEATURES) + " |")
lines += ["", "| feature | lower | upper | mean | sd |", "|---|---|---|---|---|"]
for f in CLUSTER_FEATURES:
    lines.append(
        f"| {f} | {pp.lower[f]:.4f} | {pp.upper[f]:.4f} | {pp.mean[f]:.4f} | {pp.std[f]:.4f} |"
    )
print("\n".join(lines))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model selection: three algorithms, k = 2..10
# MAGIC
# MAGIC **Why cluster at all, instead of RFM scoring rules?** Classic RFM gives each customer a
# MAGIC 1-5 score per dimension (quintiles) and maps score combinations to names with
# MAGIC hand-written rules (e.g. R>=4 and F>=4 -> Champions). It's transparent and needs no
# MAGIC model, but the cut-offs are arbitrary, it uses only 3 dimensions, and 125 score cells must
# MAGIC be grouped by hand. Clustering finds the groups from the data across all 6 features (incl.
# MAGIC tenure and seasonality); we still give the clusters RFM-style business names (section 6).
# MAGIC Interview answer: rules when the business wants fixed, explainable definitions; clustering
# MAGIC to *discover* structure, then name it.
# MAGIC
# MAGIC | Algorithm | Assigns | Good at | Watch out for |
# MAGIC |---|---|---|---|
# MAGIC | KMeans | hard, nearest centroid | fast; centroids are easy-to-read segment "prototypes" | assumes round, similar-size clusters; result depends on the random start |
# MAGIC | Bisecting k-means | hard, top-down splits | a hierarchy (k=4 refines k=3); less start-dependent | greedy: an early bad split is never undone |
# MAGIC | Gaussian mixture | soft (probability per segment) | elliptical, different-size clusters; assignment confidence | many more parameters; can collapse onto piles of identical points (we have them: 1,618 one-invoice customers) |
# MAGIC
# MAGIC **Free Edition:** Spark ML `BisectingKMeans` fails here (`DBFS_DISABLED`, probe 1a), so
# MAGIC bisecting k-means is implemented with Spark `KMeans(k=2)` splits in
# MAGIC `src/segmentation/bisecting.py` (same algorithm; the split sequence is the model).
# MAGIC
# MAGIC **How k is chosen** (same rules for every algorithm, all on the same preprocessed space):
# MAGIC 1. **Silhouette** (separation, higher is better) as the main score;
# MAGIC 2. the **elbow** of WCSS (within-cluster sum of squares) as a cross-check;
# MAGIC 3. every segment must hold at least **3%** of customers (~175 people) to be worth a campaign;
# MAGIC 4. **k >= 3**: k = 2 ("high value vs the rest") usually separates best but is too coarse.
# MAGIC
# MAGIC The final choice also needs **stability** (section 5): a segmentation that changes with the
# MAGIC random seed can't be used for campaigns.

# COMMAND ----------

from segmentation.clustering import ALGORITHMS, run_sweep
from segmentation.metrics import elbow_k, suggest_k
from segmentation.plots import metric_by_k, share_bars
from segmentation.tracking import log_trial, results_table

KS = list(range(2, 11))
SEED = 42
MIN_SHARE = 0.03
K_MIN = 3
# From probe 1a; Free Edition fails Spark's BisectingKMeans -> KMeans-split implementation.
_probe_bkm = globals().get("probe", {}).get("spark_ml_bisecting_kmeans", "FAILED")
BISECTING_IMPL = "spark_ml" if str(_probe_bkm).startswith("ok") else "kmeans_splits"

# Set RUN_SWEEP = False once a sweep is recorded and paste its run id below: the results are
# then reloaded from MLflow instead of refitting 27 models.
RUN_SWEEP = True
SWEEP_RUN_ID = None
print("bisecting implementation:", BISECTING_IMPL)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Run the sweep (one parent MLflow run, one nested run per algorithm x k)
# MAGIC
# MAGIC 3 algorithms x 9 values of k = 27 child runs, each with params (algorithm, k, seed,
# MAGIC implementation, features, preprocessing run id), metrics (silhouette, WCSS, min/max
# MAGIC share, empty clusters, GMM confidence, seconds) and the cluster shares + model
# MAGIC parameters as JSON. Expect several minutes on serverless.

# COMMAND ----------

common_params = {
    "as_of_date": str(AS_OF),
    "cluster_features": ",".join(CLUSTER_FEATURES),
    "preprocess_run_id": PREPROCESS_RUN_ID,
    "n_customers": checks["customers"],
}


def log_and_print(r):
    """Called after every trial: nested MLflow run + one progress line."""
    log_trial(r, common_params)
    sil = r.metrics.get("silhouette")
    print(
        f"{r.algorithm:17s} k={r.k:2d} silhouette={sil if sil is None else round(sil, 4)} "
        f"min_share={round(r.metrics.get('min_share', 0), 4)} error={r.error}"
    )


if RUN_SWEEP:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="k_sweep") as sweep_run:
        mlflow.set_tags({"phase": "3", "kind": "k_sweep", "bisecting_impl": BISECTING_IMPL})
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
# MAGIC ### 4.2 Compare: silhouette, elbow, segment sizes

# COMMAND ----------

import pandas as pd

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
    "Silhouette by k (circled = suggested k)",
    marks=suggested,
)
fig_wcss = metric_by_k(
    rows, "wcss", "WCSS", "Elbow: within-cluster sum of squares by k", marks=elbows
)
fig_min = metric_by_k(rows, "min_share", "smallest segment share", "Smallest segment by k")
display(fig_sil)
display(fig_wcss)
display(fig_min)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Segment sizes at each algorithm's suggested k, and logging the comparison

# COMMAND ----------

size_figs = {}
for a, k in suggested.items():
    if k is None:
        continue
    trial = next(r for r in sweep_results if r["algorithm"] == a and r["k"] == k)
    shares = {int(c): v for c, v in trial["shares"].items()}
    size_figs[a] = share_bars(dict(sorted(shares.items())), f"{a}, k={k}: segment sizes")
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
# MAGIC ### 4.4 Summary to paste back into the chat

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
lines = [
    "### Phase 3.3 results",
    "",
    f"- sweep run: {SWEEP_RUN_ID}",
    f"- bisecting implementation: {BISECTING_IMPL}",
    f"- suggested k (silhouette, min share {MIN_SHARE}, k>={K_MIN}): {suggested}",
    f"- elbow k: {elbows}",
]
errors = [(r["algorithm"], r["k"], r["error"]) for r in rows if r["error"]]
lines.append(f"- errors: {errors or 'none'}")
lines += ["", "| algorithm | k | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
for r in sorted(rows, key=lambda r: (r["algorithm"], r["k"])):
    vals = [r.get(c) for c in cols]
    lines.append(
        f"| {r['algorithm']} | {r['k']} | "
        + " | ".join(
            "" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals
        )
        + " |"
    )
for a, k in suggested.items():
    if k is not None:
        t = next(r for r in sweep_results if r["algorithm"] == a and r["k"] == k)
        lines.append(f"- shares {a} k={k}: { {c: round(v, 4) for c, v in t['shares'].items()} }")
print("\n".join(lines))
