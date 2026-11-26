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
# MAGIC | 3 | Preprocessing, clustering sweep, stability (coming in 4.2) | 4.2 |
# MAGIC | 4 | Final model, profiles, names, `retail.gold.product_segments` (4.3) | 4.3 |
# MAGIC | 5 | Baskets and frequent itemsets (4.4) | 4.4 |
# MAGIC | 6 | Association rules, `retail.gold.product_associations` (4.5) | 4.5 |
# MAGIC | 7 | Generated `docs/product_insights.md` (4.6) | 4.6 |
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
