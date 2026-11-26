# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05 · Repeat-purchase propensity model
# MAGIC
# MAGIC Goal: for every known customer on a given date, predict the probability that they **buy
# MAGIC again in the next 90 days**, without any leakage from the future, and register the best
# MAGIC model in the **Unity Catalog model registry** as `retail.ml.repeat_purchase_model@champion`.
# MAGIC
# MAGIC | Section | What it does | Step |
# MAGIC |---|---|---|
# MAGIC | 1 | **Free Edition probe**: UC model registry + aliases, Optuna, LightGBM, SHAP on serverless | 5.1 |
# MAGIC | 2 | **Leakage-safe training set** `retail.ml.propensity_training_set`: rolling cutoffs, 90-day labels, leakage proofs, feature-store point-in-time check | 5.1 |
# MAGIC | 3 | Evaluation helpers + logistic regression baseline | 5.2 |
# MAGIC | 4 | LightGBM tuned with Optuna | 5.3 |
# MAGIC | 5 | SHAP explanations + held-out test evaluation | 5.4 |
# MAGIC | 6 | Registry: champion / challenger + promotion rule | 5.5 |
# MAGIC | 7 | Phase 5 checks | 5.6 |
# MAGIC
# MAGIC Logic lives in `src/modeling/` (unit-tested); this notebook runs it and explains each
# MAGIC decision.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install the ML libraries
# MAGIC
# MAGIC Serverless environment 6 ships scikit-learn and MLflow, but not (necessarily) LightGBM,
# MAGIC Optuna or SHAP, and not the Feature Engineering client. `%pip install` makes them
# MAGIC notebook-scoped (only this notebook's Python environment changes); `restartPython()`
# MAGIC reloads Python so the packages are importable, which wipes variables, so this comes first.
# MAGIC
# MAGIC The probe (section 1) prints the exact versions; they go into `PROGRESS_LOG.md`, and
# MAGIC MLflow records them again with each logged model (`requirements.txt`), so the model can
# MAGIC be loaded later with the same libraries.

# COMMAND ----------

# MAGIC %pip install --quiet lightgbm optuna shap databricks-feature-engineering

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

import json
import platform
import time

import mlflow
from pyspark.sql import functions as F

from common.tables import (
    ML_CUSTOMER_FEATURES,
    ML_PROPENSITY_PROBE_MODEL,
    ML_PROPENSITY_TRAINING_SET,
    PROPENSITY_EXPERIMENT_NAME,
    SILVER_RETURNS,
    SILVER_SALES,
    user_experiment_path,
)

USER = spark.sql("SELECT current_user() AS u").first()["u"]
EXPERIMENT_PATH = user_experiment_path(USER, PROPENSITY_EXPERIMENT_NAME)
mlflow.set_registry_uri("databricks-uc")  # register models in Unity Catalog (not the old workspace registry)
print("user:", USER)
print("experiment:", EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Free Edition probe (step 5.1)
# MAGIC
# MAGIC Before building anything, check on **this** workspace that every piece Phase 5 needs
# MAGIC actually works on serverless (docs and reality have differed before: BisectingKMeans
# MAGIC failed in Phase 3, FP-Growth worked in Phase 4 although the docs said it wouldn't).
# MAGIC
# MAGIC | Check | Why |
# MAGIC |---|---|
# MAGIC | library versions | LightGBM / Optuna / SHAP installed by `%pip` above |
# MAGIC | **UC model registry round trip** | log a tiny scikit-learn model with a signature, register it in UC, set an alias, load it **by alias**, predict, then delete it |
# MAGIC | **Optuna** | a 5-trial study (single-node, in the notebook's Python process) |
# MAGIC | **LightGBM** | fit + `predict_proba` |
# MAGIC | **SHAP** | `TreeExplainer` on the LightGBM model, output shape |
# MAGIC
# MAGIC **Tuning library choice: Optuna, not Hyperopt.** Databricks has deprecated Hyperopt
# MAGIC (no longer maintained, removed from recent ML runtimes) and recommends Optuna for
# MAGIC single-node tuning on serverless. Our training set is small (tens of thousands of rows),
# MAGIC so each trial trains in seconds on one machine; Hyperopt's main extra (`SparkTrials`,
# MAGIC trials spread over a cluster) needs classic clusters anyway. Optuna also has pruning
# MAGIC and a define-by-run search space (plain Python `if`s for conditional parameters).
# MAGIC *Interview point:* "distributed tuning only pays off when one trial is expensive; here
# MAGIC it isn't, so the simpler, maintained tool wins."
# MAGIC
# MAGIC The probe model goes to `retail.ml.propensity_probe_model` and is deleted at the end,
# MAGIC so the real model name stays clean. Set `RUN_PROBE = False` and paste the run id into
# MAGIC `PROBE_RUN_ID` once it has run.

# COMMAND ----------

RUN_PROBE = True
PROBE_RUN_ID = None  # e.g. "abc123..." after the first run

# COMMAND ----------

import numpy as np
import pandas as pd

from modeling.environment import package_versions, run_check


def _toy_data(n=400, seed=7):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.exponential(size=n), "c": rng.integers(0, 5, n)})
    p = 1 / (1 + np.exp(-(1.5 * x["a"] - x["b"] + 0.3 * x["c"])))
    y = (rng.uniform(size=n) < p).astype(int)
    return x, pd.Series(y, name="label")


def probe_uc_registry():
    """Log -> register in UC -> alias -> load by alias -> predict -> delete."""
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from sklearn.linear_model import LogisticRegression

    x, y = _toy_data()
    model = LogisticRegression(max_iter=200).fit(x, y)
    proba = model.predict_proba(x)
    signature = infer_signature(x, proba)
    with mlflow.start_run(run_name="probe_uc_model", nested=True):
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=x.head(3),
            pyfunc_predict_fn="predict_proba",
        )
    mv = mlflow.register_model(info.model_uri, ML_PROPENSITY_PROBE_MODEL)
    client = MlflowClient()
    client.set_registered_model_alias(ML_PROPENSITY_PROBE_MODEL, "probe", mv.version)
    loaded = mlflow.pyfunc.load_model(f"models:/{ML_PROPENSITY_PROBE_MODEL}@probe")
    pred = np.asarray(loaded.predict(x.head(5)))
    by_alias = client.get_model_version_by_alias(ML_PROPENSITY_PROBE_MODEL, "probe")
    client.delete_registered_model(ML_PROPENSITY_PROBE_MODEL)  # clean up (all versions)
    return {
        "registered_version": mv.version,
        "alias_resolves_to": by_alias.version,
        "prediction_shape": list(pred.shape),
        "probabilities_match": bool(np.allclose(pred, proba[:5])),
        "signature_inputs": [c["name"] for c in signature.inputs.to_dict()],
        "deleted": True,
    }


def probe_optuna():
    import optuna
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    x, y = _toy_data()
    xt, xv, yt, yv = x[:300], x[300:], y[:300], y[300:]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        c = trial.suggest_float("C", 1e-3, 10, log=True)
        m = LogisticRegression(C=c, max_iter=200).fit(xt, yt)
        return roc_auc_score(yv, m.predict_proba(xv)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(objective, n_trials=5)
    return {"trials": len(study.trials), "best_value": round(study.best_value, 4)}


def probe_lightgbm_and_shap():
    import lightgbm as lgb
    import shap

    x, y = _toy_data()
    m = lgb.LGBMClassifier(n_estimators=50, verbose=-1).fit(x, y)
    p = m.predict_proba(x)[:, 1]
    values = shap.TreeExplainer(m).shap_values(x.head(50))
    values = values[1] if isinstance(values, list) else values  # older SHAP: one array per class
    return {"proba_range": [round(float(p.min()), 3), round(float(p.max()), 3)], "shap_shape": list(np.shape(values))}


if RUN_PROBE:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="probe") as run:
        mlflow.set_tags({"phase": "5", "kind": "probe"})
        env = {
            "spark": spark.version,
            "python": platform.python_version(),
            **package_versions(),
        }
        checks = [
            run_check("uc_model_registry", probe_uc_registry),
            run_check("optuna", probe_optuna),
            run_check("lightgbm_shap", probe_lightgbm_and_shap),
        ]
        mlflow.log_dict({"environment": env, "checks": checks}, "probe/probe.json")
        mlflow.log_params({f"v_{k}": v for k, v in env.items()})
        PROBE_RUN_ID = run.info.run_id
else:
    probe = mlflow.artifacts.load_dict(f"runs:/{PROBE_RUN_ID}/probe/probe.json")
    env, checks = probe["environment"], probe["checks"]

print("probe run:", PROBE_RUN_ID)
for k, v in env.items():
    print(f"  {k:32s} {v}")
for c in checks:
    print(f"  {'OK ' if c['ok'] else 'FAIL'} {c['check']:20s} {c['detail']}")
PROBE_OK = {c["check"]: c["ok"] for c in checks}

# COMMAND ----------

# MAGIC %md
# MAGIC **If a check failed:** `uc_model_registry` → fallback is logging models to MLflow only and
# MAGIC tracking "champion" as a tag on the run (we'd still use the same promotion rule);
# MAGIC `optuna` → plain random search over the same space; `lightgbm_shap` → install a pinned
# MAGIC SHAP version, or use LightGBM's built-in `pred_contrib=True` (same TreeSHAP values,
# MAGIC no extra library). Paste the output above into the chat before going on.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Leakage-safe training set (step 5.1)
# MAGIC
# MAGIC ### 2.1 How a snapshot is built
# MAGIC
# MAGIC ```
# MAGIC    features: everything strictly before c          label window
# MAGIC  |------------------------------------------------|c=============c+90d|
# MAGIC  data start                                    cutoff            label = 1 if any order here
# MAGIC ```
# MAGIC
# MAGIC - **Features** = the Phase 3 customer features (`build_customer_features(..., as_of=c)`,
# MAGIC   unchanged code) + recent-activity counts (orders / revenue in the last 30, 90, 180,
# MAGIC   365 days) + `is_uk` + `label_window_q4_days` (how many days of the coming window fall
# MAGIC   in Oct–Dec: known on day `c` from the calendar, so not leakage, and it lets the model
# MAGIC   learn that a window over Christmas has a higher base rate).
# MAGIC - **Label** = 1 if the customer places at least one order in `[c, c + 90 days)`. A sale
# MAGIC   **on** the cutoff day is in the label, never in the features.
# MAGIC - **Population** = customers with at least one purchase before `c`. A customer first seen
# MAGIC   inside the window isn't known on day `c`, so they can't be scored then.
# MAGIC - Features left out on purpose: `is_tenure_censored` (depends on where the *data* starts,
# MAGIC   not on the customer: at an early cutoff nearly everyone is "censored"), raw dates
# MAGIC   (recency/tenure carry them), `preferred_country` (→ `is_uk`).
# MAGIC
# MAGIC ### 2.2 Rolling snapshots and the time-based split
# MAGIC
# MAGIC A single cutoff would give one customer table (~5k rows) and only one season. Instead,
# MAGIC one snapshot per month for training, then two later cutoffs for validation and test:
# MAGIC
# MAGIC - **test cutoff** = last date whose 90-day window still fits in the data:
# MAGIC   2011-12-10 (day after the last sale) − 90 days = **2011-09-11**
# MAGIC - **validation cutoff** = test − 90 days = **2011-06-13** (its window ends exactly at the test cutoff)
# MAGIC - **training cutoffs** = the 1st of each month from 2010-03-01 (first month with 90 days
# MAGIC   of history) to the last one whose window ends before the validation cutoff
# MAGIC
# MAGIC `validate_plan` enforces the rules (window inside the data, no training label window
# MAGIC overlapping validation, no validation window overlapping test, minimum history) and
# MAGIC raises `LeakageError` otherwise.
# MAGIC
# MAGIC **Why time-based, not a random split** *(interview point)*: a random split puts the same
# MAGIC customer's March and April snapshots on both sides, and lets the model learn from
# MAGIC windows that lie *after* the rows it is tested on; the score is optimistic and doesn't
# MAGIC measure what we need: "train on the past, predict the future". Alternatives:
# MAGIC one cutoff only (simple, but few rows and one season); expanding-window backtests
# MAGIC (several train/validation folds in time, better variance estimate, more compute: a good
# MAGIC Phase 7 extension). The same customer appears in several *training* snapshots; that's
# MAGIC fine (each row is a different moment), but it's why we never split rows randomly.
# MAGIC
# MAGIC Note that the test window (Sep–Dec 2011) covers the Christmas peak while validation
# MAGIC (Jun–Sep) doesn't: the base rates will differ. That's real-world seasonality, so we
# MAGIC report it rather than hide it, and the training cutoffs include Q4 2010.

# COMMAND ----------

from features.customer_features import build_customer_features, default_as_of
from features.feature_store import write_snapshot
from modeling.build_training_set import (
    MODEL_FEATURES,
    build_snapshot,
    check_training_set,
    label_summary,
    plan_cutoffs,
    stale_cutoffs_sql,
    truncation_check,
    write_training_snapshot,
)

sales = spark.table(SILVER_SALES)
returns = spark.table(SILVER_RETURNS)

DATA_START = sales.agg(F.min("invoice_date").alias("d")).first()["d"]
DATA_END_EXCLUSIVE = default_as_of(sales)  # day after the last sale
PLAN = plan_cutoffs(DATA_START, DATA_END_EXCLUSIVE)  # raises LeakageError if a rule breaks

print(f"data: {DATA_START} .. {DATA_END_EXCLUSIVE} (exclusive); horizon {PLAN.horizon_days} days")
for c, split in PLAN.cutoffs():
    print(f"  {split:10s} cutoff {c}  label window [{c}, {PLAN.label_end(c)})")
print(f"{len(PLAN.train_cutoffs)} training cutoffs; model features ({len(MODEL_FEATURES)}):")
print(" ", ", ".join(MODEL_FEATURES))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Proof of no leakage on the real data
# MAGIC
# MAGIC The unit tests prove it on hand-made rows (`tests/test_build_training_set.py`: future
# MAGIC orders, returns and new customers change the labels but not one feature value; and a
# MAGIC deliberately leaky feature is caught). Here the same idea runs on the real tables:
# MAGIC `truncation_check` builds the features twice, from the full Silver tables and from
# MAGIC tables **physically cut** at the cutoff (`invoice_date < cutoff`). Any feature that
# MAGIC reads an event on/after the cutoff would make the two differ. Checked for the first
# MAGIC training cutoff, the validation cutoff and the test cutoff.

# COMMAND ----------

RUN_LEAKAGE_CHECK = True

if RUN_LEAKAGE_CHECK:
    leakage_checks = [
        truncation_check(sales, returns, c, DATA_START)
        for c in (PLAN.train_cutoffs[0], PLAN.validation_cutoff, PLAN.test_cutoff)
    ]
    for r in leakage_checks:
        print(r)
    assert all(r["only_in_full"] == 0 and r["only_in_truncated"] == 0 for r in leakage_checks)
    assert all(r["rows_full"] == r["rows_truncated"] for r in leakage_checks)
    print("no leakage: features identical with and without the future")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Build and write the training set
# MAGIC
# MAGIC One snapshot per cutoff, each written with `replaceWhere cutoff_date = ...` (idempotent
# MAGIC re-runs; other cutoffs untouched; cutoffs no longer in the plan are deleted). One write
# MAGIC per cutoff instead of one giant union: smaller plans, visible progress, and a failure
# MAGIC only costs one snapshot.
# MAGIC
# MAGIC The same loop also writes each cutoff's **Phase 3 feature snapshot** into the feature
# MAGIC table `retail.ml.customer_features` (its `as_of_date` TIMESERIES key was designed for
# MAGIC exactly this). The Phase 3 snapshot (2011-12-10) is not touched: it isn't a cutoff here,
# MAGIC and every Phase 3 read filters on its own `as_of_date`. Those snapshots are used for the
# MAGIC point-in-time check in 2.6.

# COMMAND ----------

RUN_BUILD_TRAINING_SET = True

if RUN_BUILD_TRAINING_SET:
    build_times = {}
    for c, split in PLAN.cutoffs():
        t0 = time.perf_counter()
        snap = build_snapshot(sales, returns, c, split, DATA_START, PLAN.horizon_days)
        write_training_snapshot(snap, ML_PROPENSITY_TRAINING_SET, c)
        write_snapshot(
            build_customer_features(sales, returns, c, data_start=DATA_START),
            ML_CUSTOMER_FEATURES,
            c,
        )
        build_times[str(c)] = round(time.perf_counter() - t0, 1)
        print(f"{split:10s} {c}: written in {build_times[str(c)]} s")
    spark.sql(stale_cutoffs_sql(ML_PROPENSITY_TRAINING_SET, PLAN))

training_set = spark.table(ML_PROPENSITY_TRAINING_SET)  # no .cache(): this notebook rewrites it

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Checks and base rates
# MAGIC
# MAGIC `check_training_set` asserts unique `(customer_id, cutoff_date)`, labels in {0, 1}, and
# MAGIC that no row's last purchase is on/after its cutoff (a second, cheap leakage guard on the
# MAGIC written table). The summary shows how many customers each snapshot has and how many
# MAGIC came back: that base rate decides whether class imbalance needs handling (step 5.3).

# COMMAND ----------

ts_checks = check_training_set(training_set)
print(ts_checks)
assert ts_checks["duplicate_keys"] == 0
assert ts_checks["null_labels"] == 0 and ts_checks["bad_labels"] == 0
assert ts_checks["purchases_on_or_after_cutoff"] == 0
assert ts_checks["min_recency_days"] >= 1

planned = {str(c) for c, _ in PLAN.cutoffs()}
written = {str(r["cutoff_date"]) for r in training_set.select("cutoff_date").distinct().collect()}
assert written == planned, (written ^ planned)

summary = label_summary(training_set).toPandas()
summary["cutoff_date"] = summary["cutoff_date"].astype(str)
display(summary)

by_split = (
    training_set.groupBy("split")
    .agg(F.count(F.lit(1)).alias("rows"), F.sum("label").alias("positives"), F.round(F.avg("label"), 4).alias("positive_rate"))
    .toPandas()
    .set_index("split")
    .loc[["train", "validation", "test"]]
)
display(by_split.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Feature store point-in-time check (optional, needs the FE client)
# MAGIC
# MAGIC In a feature-store workflow, the training set is a **label table**
# MAGIC `(customer_id, cutoff_date, label)` and the features are looked up with a
# MAGIC **point-in-time join**: for each row, the latest feature snapshot whose `as_of_date` is
# MAGIC on/before the row's `cutoff_date` (`FeatureLookup(..., timestamp_lookup_key=...)`).
# MAGIC That is exactly the rule that prevents leakage when features are refreshed over time.
# MAGIC Here it's a cross-check: the Phase 3 columns fetched by the Feature Engineering client
# MAGIC must equal the ones our builder computed.

# COMMAND ----------

RUN_FE_PIT_CHECK = True
PIT_COLUMNS = ["recency_days", "frequency", "monetary", "tenure_days", "return_rate", "seasonal_revenue_share"]

if RUN_FE_PIT_CHECK:
    from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

    fe = FeatureEngineeringClient()
    labels_df = training_set.select("customer_id", "cutoff_date", "label")
    fe_set = fe.create_training_set(
        df=labels_df,
        feature_lookups=[
            FeatureLookup(
                table_name=ML_CUSTOMER_FEATURES,
                lookup_key="customer_id",
                timestamp_lookup_key="cutoff_date",
                feature_names=PIT_COLUMNS,
            )
        ],
        label="label",
    )
    pit = fe_set.load_df()

    def _r(df):
        return df.select("customer_id", "cutoff_date", *[F.round(F.col(c).cast("double"), 6).alias(c) for c in PIT_COLUMNS])

    ours, theirs = _r(training_set), _r(pit)
    pit_result = {
        "rows_builder": ours.count(),
        "rows_feature_store": theirs.count(),
        "only_in_builder": ours.exceptAll(theirs).count(),
        "only_in_feature_store": theirs.exceptAll(ours).count(),
    }
    print(pit_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 Record the training set in MLflow
# MAGIC
# MAGIC One run `training_set` with the split plan, checks and base rates as artefacts, so every
# MAGIC later model run can point to the exact data definition it was trained on (its run id is
# MAGIC logged as a tag on the model runs). Paste the printed block into the chat.

# COMMAND ----------

RUN_LOG_TRAINING_SET = True
TRAINING_SET_RUN_ID = None  # paste after the first run

if RUN_LOG_TRAINING_SET:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="training_set") as run:
        mlflow.set_tags({"phase": "5", "kind": "training_set", "table": ML_PROPENSITY_TRAINING_SET})
        mlflow.log_params(
            {
                "horizon_days": PLAN.horizon_days,
                "n_train_cutoffs": len(PLAN.train_cutoffs),
                "validation_cutoff": str(PLAN.validation_cutoff),
                "test_cutoff": str(PLAN.test_cutoff),
                "n_model_features": len(MODEL_FEATURES),
            }
        )
        mlflow.log_dict(PLAN.to_dict(), "training_set/plan.json")
        mlflow.log_dict({"features": list(MODEL_FEATURES)}, "training_set/features.json")
        mlflow.log_dict(
            {
                "checks": ts_checks,
                "leakage_checks": leakage_checks if RUN_LEAKAGE_CHECK else None,
                "pit_check": pit_result if RUN_FE_PIT_CHECK else None,
                "build_times_s": build_times if RUN_BUILD_TRAINING_SET else None,
                "by_cutoff": json.loads(summary.to_json(orient="records")),
                "by_split": json.loads(by_split.reset_index().to_json(orient="records")),
            },
            "training_set/summary.json",
        )
        for split, row in by_split.iterrows():
            mlflow.log_metric(f"{split}_rows", int(row["rows"]))
            mlflow.log_metric(f"{split}_positive_rate", float(row["positive_rate"]))
        TRAINING_SET_RUN_ID = run.info.run_id

ts_summary = mlflow.artifacts.load_dict(f"runs:/{TRAINING_SET_RUN_ID}/training_set/summary.json")

lines = ["### Phase 5.1 results", "", f"- probe run: {PROBE_RUN_ID}, training_set run: {TRAINING_SET_RUN_ID}"]
lines += ["- env: " + ", ".join(f"{k} {v}" for k, v in env.items())]
lines += ["- probe: " + ", ".join(f"{c['check']} {'ok' if c['ok'] else 'FAILED'}: {c['detail']}" for c in checks)]
lines += [f"- data {DATA_START} .. {DATA_END_EXCLUSIVE} (excl.); train cutoffs {PLAN.train_cutoffs[0]} .. {PLAN.train_cutoffs[-1]} ({len(PLAN.train_cutoffs)}), validation {PLAN.validation_cutoff}, test {PLAN.test_cutoff}"]
lines += [f"- checks: {ts_summary['checks']}"]
lines += [f"- leakage: {ts_summary['leakage_checks']}"]
lines += [f"- PIT check: {ts_summary['pit_check']}"]
lines += [f"- build times (s): {ts_summary['build_times_s']}"]
lines += ["", "| cutoff | split | customers | positives | rate | Q4 days |", "|---|---|---:|---:|---:|---:|"]
lines += [
    f"| {r['cutoff_date']} | {r['split']} | {r['customers']:,} | {r['positives']:,} | {r['positive_rate']} | {r['label_window_q4_days']} |"
    for r in ts_summary["by_cutoff"]
]
lines += ["", "| split | rows | positives | rate |", "|---|---:|---:|---:|"]
lines += [f"| {r['split']} | {r['rows']:,} | {r['positives']:,} | {r['positive_rate']} |" for r in ts_summary["by_split"]]
print("\n".join(lines))
