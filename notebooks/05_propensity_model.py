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

RUN_PROBE = False
PROBE_RUN_ID = "7748bb7d8e3e4b088f0129bca7d91528"  # recorded 2026-11-26

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

RUN_LEAKAGE_CHECK = False  # recorded in the training_set run (all 0 differences)

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

RUN_BUILD_TRAINING_SET = False  # table built 2026-11-26 (53,440 rows)

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

RUN_FE_PIT_CHECK = False  # recorded: 53,440 = 53,440, 0 differences
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

RUN_LOG_TRAINING_SET = False
TRAINING_SET_RUN_ID = "84a37971498d4e5487cb6535665c852a"  # recorded 2026-11-26

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluation helpers and the logistic regression baseline (step 5.2)
# MAGIC
# MAGIC ### 3.0 What "good" means here *(interview point)*
# MAGIC
# MAGIC The model's job is to **rank** customers for a campaign with a limited budget ("contact
# MAGIC the 20 % most likely to come back"), and sometimes to give **probabilities** (expected
# MAGIC revenue = p × basket value). So:
# MAGIC
# MAGIC | Metric | Reads as | Why it's here |
# MAGIC |---|---|---|
# MAGIC | **PR-AUC** (average precision) | precision averaged over all recall levels; no-skill = base rate | **primary metric**: focuses on the buyers a campaign acts on; read next to the base rate (`pr_auc_lift`) |
# MAGIC | ROC-AUC | chance a random buyer is scored above a random non-buyer | threshold-free ranking quality; stable, but blind to *where* in the list the errors are |
# MAGIC | capture top 10/20/30 % | share of all repeat buyers in the top x % by score | the business sentence: "top 20 % captures X % of repeat buyers" |
# MAGIC | Brier, ECE, reliability curve | are scores honest probabilities? | needed before p is multiplied by money |
# MAGIC
# MAGIC **Accuracy is not reported**: at a 32 % base rate, "nobody comes back" is already 68 %
# MAGIC accurate, and accuracy depends on a 0.5 threshold nobody uses.
# MAGIC
# MAGIC **Class imbalance: no reweighting.** Positives are 45 % of training rows and 32–44 % in
# MAGIC validation / test: not rare. Reweighting (`class_weight="balanced"`, `scale_pos_weight`)
# MAGIC barely changes the *ranking* but shifts every probability upwards, which breaks
# MAGIC calibration. With rare positives (say < 5 %) we'd revisit: weights or undersampling
# MAGIC for training, then recalibrate.
# MAGIC
# MAGIC **Model selection uses validation only; the test cutoff (2011-09-11) is not loaded until
# MAGIC step 5.4**, so the final numbers are an honest estimate on data no decision has seen.

# COMMAND ----------

from modeling.baseline import (
    DEFAULT_C_GRID,
    LOG_FEATURES,
    best_by,
    coefficients,
    fit_logistic_grid,
    positive_proba,
    recency_rule_scores,
)
from modeling.dataset import feature_health, split_frames, to_pandas
from modeling.evaluation import (
    evaluate,
    plot_bars,
    plot_calibration,
    plot_gain,
    plot_lift,
    plot_roc_pr,
    prefixed,
)

print("mlflow module version:", mlflow.__version__)

pdf = to_pandas(training_set, splits=("train", "validation"))  # test stays unloaded until 5.4
SPLITS = split_frames(pdf)
train, val = SPLITS["train"], SPLITS["validation"]
print(f"train {train.n:,} rows ({train.y.mean():.4f} positive), validation {val.n:,} ({val.y.mean():.4f})")

health = feature_health(train.X)
assert health.loc[list(LOG_FEATURES), "n_negative"].sum() == 0, "log1p needs non-negative inputs"
display(health.reset_index(names="feature"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Rule baseline: "most recent buyers first"
# MAGIC
# MAGIC Before any model: what does the obvious marketing rule achieve? Score = 1 / (1 + recency
# MAGIC days). It's not a probability (so no calibration), only a ranking. Every model has to
# MAGIC beat this to be worth running.

# COMMAND ----------

rule_val = evaluate(val.y, recency_rule_scores(val.X))
print({k: round(v, 4) for k, v in rule_val["metrics"].items() if k not in ("n", "positives")})

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Logistic regression
# MAGIC
# MAGIC Pipeline (`modeling.baseline.make_logistic_pipeline`, pure scikit-learn so the saved
# MAGIC model needs none of our code to load):
# MAGIC
# MAGIC 1. median imputation (only `mean_days_between_purchases` has NULLs: one-day customers),
# MAGIC    plus a **missing indicator** for it, so "no second purchase day yet" stays visible;
# MAGIC 2. `log1p` on skewed counts and money (Phase 3: monetary skew 25) so a linear model sees
# MAGIC    "10× more spend" as a constant step, not a huge outlier;
# MAGIC 3. standardise; 4. L2-regularised logistic regression.
# MAGIC
# MAGIC Regularisation strength `C` from a small grid (0.01, 0.1, 1, 10), chosen on **validation
# MAGIC PR-AUC** (ties → smaller C). One parent MLflow run, one nested run per C; the chosen
# MAGIC model is logged with a **signature** (input columns and types, output shape) and an
# MAGIC **input example**, both required by the Unity Catalog registry in step 5.5.
# MAGIC `pyfunc_predict_fn="predict_proba"` makes the generic MLflow loader return
# MAGIC probabilities (column 1 = P(repeat)) rather than 0/1 classes.
# MAGIC
# MAGIC Why logistic regression as the baseline: fast, well understood, calibrated by
# MAGIC construction (it minimises log loss), and its coefficients are explainable. Its
# MAGIC limitation is the point of step 5.3: it can't model interactions (e.g. recency
# MAGIC matters differently for one-order vs loyal customers) unless we hand-craft them.

# COMMAND ----------

RUN_LOGREG = False
LOGREG_RUN_ID = "61fa2f8d58d14004982813d1c2a45264"  # recorded 2026-11-26 (C = 0.01)

if RUN_LOGREG:
    from mlflow.models import infer_signature

    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="logreg_baseline") as parent:
        mlflow.set_tags(
            {
                "phase": "5",
                "kind": "baseline",
                "model_type": "logistic_regression",
                "training_set_run_id": TRAINING_SET_RUN_ID,
                "selection_metric": "validation_pr_auc",
            }
        )
        t0 = time.perf_counter()
        grid = fit_logistic_grid(train.X, train.y, val.X, val.y, Cs=DEFAULT_C_GRID)
        fit_seconds = round(time.perf_counter() - t0, 1)
        for r in grid:
            with mlflow.start_run(run_name=f"logreg_C{r.C}", nested=True):
                mlflow.set_tags({"phase": "5", "kind": "baseline_trial"})
                mlflow.log_param("C", r.C)
                mlflow.log_metrics({**prefixed(r.train, "train"), **prefixed(r.validation, "validation")})
        best = best_by(grid, "pr_auc")
        model = best.model

        p_train, p_val = positive_proba(model, train.X), positive_proba(model, val.X)
        ev_train, ev_val = evaluate(train.y, p_train), evaluate(val.y, p_val)

        example = train.X.dropna().head(5)
        signature = infer_signature(example, model.predict_proba(example))
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=example,
            pyfunc_predict_fn="predict_proba",
        )

        mlflow.log_params({"C": best.C, "C_grid": list(DEFAULT_C_GRID), "n_features": train.X.shape[1], "class_weight": "none"})
        mlflow.log_metrics(
            {
                **prefixed(ev_train["metrics"], "train"),
                **prefixed(ev_val["metrics"], "validation"),
                **prefixed(rule_val["metrics"], "rule_validation"),
                "fit_grid_seconds": fit_seconds,
            }
        )
        coefs = coefficients(model)
        mlflow.log_dict(
            {
                "model_uri": info.model_uri,
                "validation": ev_val["metrics"],
                "train": ev_train["metrics"],
                "rule_validation": rule_val["metrics"],
                "grid": [{"C": r.C, "validation": r.validation, "train": r.train} for r in grid],
                "coefficients": coefs.round(6).to_dict(),
                "gain_validation": json.loads(ev_val["gain"].to_json(orient="records")),
                "calibration_validation": json.loads(ev_val["calibration"].to_json(orient="records")),
            },
            "evaluation/logreg.json",
        )
        scores = {"logreg": (val.y, p_val), "recency rule": (val.y, recency_rule_scores(val.X))}
        figs = {
            "roc_pr_validation": plot_roc_pr(scores, "Validation (cutoff 2011-06-13)"),
            "gain_validation": plot_gain(scores, title="Cumulative gain (validation)"),
            "lift_validation": plot_lift(scores, title="Lift by decile (validation)"),
            "calibration_validation": plot_calibration({"logreg": (val.y, p_val)}, title="Calibration (validation)"),
            "coefficients": plot_bars(coefs, "Logistic regression: standardised coefficients", "coefficient"),
        }
        for name, fig in figs.items():
            mlflow.log_figure(fig, f"plots/{name}.png")
        LOGREG_RUN_ID = parent.info.run_id

logreg = mlflow.artifacts.load_dict(f"runs:/{LOGREG_RUN_ID}/evaluation/logreg.json")
LOGREG_MODEL_URI = logreg["model_uri"]
print("logreg run:", LOGREG_RUN_ID, "| model:", LOGREG_MODEL_URI)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Results to paste into the chat
# MAGIC
# MAGIC Read the coefficients as "effect of +1 standard deviation of the (logged) feature on the
# MAGIC log-odds, all else equal": correlated features (frequency, n_purchase_days, monetary)
# MAGIC share credit, so single signs can look odd. SHAP on LightGBM (5.4) is the better
# MAGIC explanation; the coefficients are a sanity check.

# COMMAND ----------

KEYS = ["roc_auc", "pr_auc", "base_rate", "pr_auc_lift", "capture_top_10pct", "capture_top_20pct", "capture_top_30pct", "brier", "log_loss", "ece"]


def _fmt(m):
    return " | ".join(f"{m[k]:.4f}" if isinstance(m.get(k), float) else str(m.get(k)) for k in KEYS)


lines = ["### Phase 5.2 results", "", f"- logreg run: {LOGREG_RUN_ID}, model: {LOGREG_MODEL_URI}"]
lines += ["", "| model / split | " + " | ".join(KEYS) + " |", "|---|" + "---:|" * len(KEYS)]
lines += [f"| logreg train | {_fmt(logreg['train'])} |", f"| logreg validation | {_fmt(logreg['validation'])} |"]
lines += [f"| recency rule validation | {_fmt(logreg['rule_validation'])} |"]
lines += ["", "| C | train PR-AUC | validation PR-AUC | validation ROC-AUC | validation Brier |", "|---:|---:|---:|---:|---:|"]
lines += [
    f"| {g['C']} | {g['train']['pr_auc']:.4f} | {g['validation']['pr_auc']:.4f} | {g['validation']['roc_auc']:.4f} | {g['validation']['brier']:.4f} |"
    for g in logreg["grid"]
]
top = sorted(logreg["coefficients"].items(), key=lambda kv: -abs(kv[1]))[:8]
lines += ["", "- top coefficients: " + ", ".join(f"{k} {v:+.3f}" for k, v in top)]
lines += ["", "| band | customers | positives | rate | lift | cum. share of buyers |", "|---:|---:|---:|---:|---:|---:|"]
lines += [
    f"| {g['band']} | {g['customers']:,} | {g['positives']:,} | {g['rate']:.3f} | {g['lift']:.2f} | {g['cum_positive_share']:.3f} |"
    for g in logreg["gain_validation"]
]
lines += ["", "| calib. bin | mean predicted | observed |", "|---:|---:|---:|"]
lines += [f"| {c['bin']} | {c['mean_predicted']:.3f} | {c['observed_rate']:.3f} |" for c in logreg["calibration_validation"]]
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. LightGBM tuned with Optuna (step 5.3)
# MAGIC
# MAGIC ### 4.0 Why trees next, and how they're tuned *(interview point)*
# MAGIC
# MAGIC Logistic regression adds one straight-line effect per (logged) feature. Repeat buying has
# MAGIC **thresholds and interactions**: "≥ 2 orders in the last 90 days" is a step, and
# MAGIC recency means something different for a one-order customer than for a loyal one.
# MAGIC Gradient-boosted trees learn those without hand-made features, need no scaling or
# MAGIC logs, and handle NULLs natively (the one-day customers' missing interval).
# MAGIC
# MAGIC Protocol (`modeling.lgbm`):
# MAGIC
# MAGIC | Piece | Choice | Why |
# MAGIC |---|---|---|
# MAGIC | sampler | Optuna **TPE**, seed 42, 40 trials (20 min cap) | Bayesian: new trials are proposed near good ones; fixed seed = repeatable study |
# MAGIC | search space | learning rate, leaves, min samples per leaf, row/feature bagging, L1/L2 | complexity vs evidence; `min_child_samples` up to 500 because one customer appears in up to 13 snapshots |
# MAGIC | trees | early stopping on **validation log loss** (patience 100, max 3,000) | log loss is smooth, so the stopping point is stable |
# MAGIC | objective | **validation PR-AUC** | same selection metric as the baseline |
# MAGIC | imbalance | none (`is_unbalance` off) | 45 % positives; reweighting would distort probabilities (3.0) |
# MAGIC | final model | refit with exactly `best_iteration` trees, checked to reproduce the trial | the saved model has no eval set attached; `deterministic=True` + 4 fixed threads |
# MAGIC
# MAGIC **Validation is used for selection by both models** (C grid, trials, number of trees),
# MAGIC so validation scores are slightly optimistic for both; the **test cutoff, used once in
# MAGIC 5.4, is the honest comparison**. Alternatives: an inner time split inside the training
# MAGIC cutoffs for early stopping (cleaner, but drops the most recent training months), or
# MAGIC time-series cross-validation over several cutoffs (more robust, ~k× the compute).
# MAGIC Pruning of bad trials (`LightGBMPruningCallback`) isn't used: it needs the separate
# MAGIC `optuna-integration` package and each trial only takes seconds.

# COMMAND ----------

from modeling.lgbm import (
    EARLY_STOPPING_ROUNDS,
    MAX_ROUNDS,
    N_THREADS,
    SEED,
    feature_importance,
    plot_trials,
    refit_final,
    trials_frame,
    tune,
)
from modeling.lgbm import positive_proba as lgbm_proba

N_TRIALS = 40
TUNING_TIMEOUT_S = 20 * 60

RUN_LGBM = False
LGBM_RUN_ID = "96712fb7b1b64cefa6a03389fbb805c8"  # recorded 2026-11-27 (40 trials, best trial 31)

if RUN_LGBM:
    import lightgbm as lgb
    from mlflow.models import infer_signature

    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="lgbm_optuna") as parent:
        mlflow.set_tags(
            {
                "phase": "5",
                "kind": "model",
                "model_type": "lightgbm",
                "tuner": "optuna_tpe",
                "training_set_run_id": TRAINING_SET_RUN_ID,
                "selection_metric": "validation_pr_auc",
            }
        )

        def log_trial(number, r):
            with mlflow.start_run(run_name=f"lgbm_trial_{number:03d}", nested=True):
                mlflow.set_tags({"phase": "5", "kind": "lgbm_trial"})
                mlflow.log_params({**r.params, "best_iteration": r.best_iteration})
                mlflow.log_metrics(
                    {**prefixed(r.train, "train"), **prefixed(r.validation, "validation"), "fit_seconds": r.fit_seconds}
                )
            print(f"trial {number:3d}: val PR-AUC {r.validation['pr_auc']:.4f}, {r.best_iteration} trees, {r.fit_seconds} s")

        t0 = time.perf_counter()
        study = tune(train.X, train.y, val.X, val.y, n_trials=N_TRIALS, timeout=TUNING_TIMEOUT_S, seed=SEED, on_trial=log_trial)
        tuning_seconds = round(time.perf_counter() - t0, 1)
        trials = trials_frame(study)
        best_params = dict(study.best_params)
        best_iteration = int(study.best_trial.user_attrs["best_iteration"])

        # Final model = best trial's parameters + its number of trees, refitted without an eval set.
        model = refit_final(best_params, best_iteration, train.X, train.y)
        p_train, p_val = lgbm_proba(model, train.X), lgbm_proba(model, val.X)
        ev_train, ev_val = evaluate(train.y, p_train), evaluate(val.y, p_val)
        refit_gap = abs(ev_val["metrics"]["pr_auc"] - study.best_value)
        assert refit_gap < 1e-6, f"refit doesn't reproduce the best trial (diff {refit_gap})"

        example = train.X.dropna().head(5)
        signature = infer_signature(example, model.predict_proba(example))
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=example,
            pyfunc_predict_fn="predict_proba",
            extra_pip_requirements=[f"lightgbm=={lgb.__version__}"],
        )

        mlflow.log_params(
            {
                **best_params,
                "n_estimators": best_iteration,
                "n_trials": len(trials),
                "max_rounds": MAX_ROUNDS,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "seed": SEED,
                "n_threads": N_THREADS,
                "class_weight": "none",
            }
        )
        mlflow.log_metrics(
            {
                **prefixed(ev_train["metrics"], "train"),
                **prefixed(ev_val["metrics"], "validation"),
                "tuning_seconds": tuning_seconds,
                "refit_gap": refit_gap,
            }
        )

        logreg_model = mlflow.sklearn.load_model(LOGREG_MODEL_URI)
        p_val_lr = positive_proba(logreg_model, val.X)
        importance = feature_importance(model, "gain")
        mlflow.log_dict(
            {
                "model_uri": info.model_uri,
                "best_params": best_params,
                "best_iteration": best_iteration,
                "train": ev_train["metrics"],
                "validation": ev_val["metrics"],
                "trials": json.loads(trials.to_json(orient="records")),
                "importance_gain": importance.round(6).to_dict(),
                "gain_validation": json.loads(ev_val["gain"].to_json(orient="records")),
                "calibration_validation": json.loads(ev_val["calibration"].to_json(orient="records")),
                "tuning_seconds": tuning_seconds,
            },
            "evaluation/lgbm.json",
        )
        scores = {
            "lightgbm": (val.y, p_val),
            "logreg": (val.y, p_val_lr),
            "recency rule": (val.y, recency_rule_scores(val.X)),
        }
        figs = {
            "optuna_history": plot_trials(trials, baseline_value=logreg["validation"]["pr_auc"]),
            "roc_pr_validation": plot_roc_pr(scores, "Validation (cutoff 2011-06-13)"),
            "gain_validation": plot_gain(scores, title="Cumulative gain (validation)"),
            "lift_validation": plot_lift(scores, title="Lift by decile (validation)"),
            "calibration_validation": plot_calibration(
                {"lightgbm": (val.y, p_val), "logreg": (val.y, p_val_lr)}, title="Calibration (validation)"
            ),
            "importance_gain": plot_bars(importance, "LightGBM: share of total split gain", "share of gain"),
        }
        for name, fig in figs.items():
            mlflow.log_figure(fig, f"plots/{name}.png")
        LGBM_RUN_ID = parent.info.run_id

lgbm = mlflow.artifacts.load_dict(f"runs:/{LGBM_RUN_ID}/evaluation/lgbm.json")
LGBM_MODEL_URI = lgbm["model_uri"]
print("lgbm run:", LGBM_RUN_ID, "| model:", LGBM_MODEL_URI)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Results to paste into the chat
# MAGIC
# MAGIC What to look at: the train–validation gap (overfitting; but remember train pools 13
# MAGIC cutoffs with different base rates, so it isn't a like-for-like comparison), the gain over
# MAGIC the baseline on PR-AUC and top-20 % capture, calibration (trees are often less
# MAGIC calibrated than logistic regression), and whether the best trial sits at the edge of
# MAGIC the search space (then the range should widen).

# COMMAND ----------

lines = ["### Phase 5.3 results", "", f"- lgbm run: {LGBM_RUN_ID}, model: {LGBM_MODEL_URI}"]
lines += [f"- trials: {len(lgbm['trials'])}, tuning {lgbm['tuning_seconds']} s; best trees: {lgbm['best_iteration']}"]
lines += ["- best params: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in lgbm["best_params"].items())]
lines += ["", "| model / split | " + " | ".join(KEYS) + " |", "|---|" + "---:|" * len(KEYS)]
lines += [f"| lgbm train | {_fmt(lgbm['train'])} |", f"| lgbm validation | {_fmt(lgbm['validation'])} |"]
lines += [f"| logreg validation | {_fmt(logreg['validation'])} |", f"| recency rule validation | {_fmt(logreg['rule_validation'])} |"]
d = {k: lgbm["validation"][k] - logreg["validation"][k] for k in ("pr_auc", "roc_auc", "brier", "capture_top_20pct", "ece")}
lines += ["- lgbm − logreg (validation): " + ", ".join(f"{k} {v:+.4f}" for k, v in d.items())]
tr = sorted(lgbm["trials"], key=lambda t: -t["value"])
lines += ["", "| trial | val PR-AUC | val ROC-AUC | trees | lr | leaves | min_child | subsample | colsample | lambda | alpha |", "|---:|" + "---:|" * 10]
lines += [
    f"| {t['trial']} | {t['value']:.4f} | {t['validation_roc_auc']:.4f} | {t['best_iteration']} | {t['learning_rate']:.4f} | {t['num_leaves']} | {t['min_child_samples']} | {t['subsample']:.2f} | {t['colsample_bytree']:.2f} | {t['reg_lambda']:.3g} | {t['reg_alpha']:.3g} |"
    for t in tr[:5]
]
vals = [t["value"] for t in lgbm["trials"]]
lines += [f"- trial PR-AUC range: {min(vals):.4f} .. {max(vals):.4f} (median {sorted(vals)[len(vals) // 2]:.4f})"]
top = list(lgbm["importance_gain"].items())[:10]
lines += ["- importance (gain share): " + ", ".join(f"{k} {v:.3f}" for k, v in top)]
lines += ["", "| band | customers | positives | rate | lift | cum. share of buyers |", "|---:|---:|---:|---:|---:|---:|"]
lines += [
    f"| {g['band']} | {g['customers']:,} | {g['positives']:,} | {g['rate']:.3f} | {g['lift']:.2f} | {g['cum_positive_share']:.3f} |"
    for g in lgbm["gain_validation"]
]
lines += ["", "| calib. bin | mean predicted | observed |", "|---:|---:|---:|"]
lines += [f"| {c['bin']} | {c['mean_predicted']:.3f} | {c['observed_rate']:.3f} |" for c in lgbm["calibration_validation"]]
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Held-out test evaluation and SHAP (step 5.4)
# MAGIC
# MAGIC ### 5.0 The test cutoff, used once
# MAGIC
# MAGIC Cutoff **2011-09-11**, label window to 2011-12-09: no decision so far has seen it (C,
# MAGIC trials and tree counts were all chosen on validation). It's also the most realistic
# MAGIC check: the latest population (5,256 customers), and a window over the **Christmas
# MAGIC peak** (base rate 43.6 % vs 32.1 % in validation), so it tests whether the models carry
# MAGIC over to a different season.
# MAGIC
# MAGIC Both models are **reloaded from their MLflow model URIs** (not the Python objects
# MAGIC above), which also proves the logged artefacts work before they're registered in 5.5.
# MAGIC
# MAGIC **Is LightGBM really better?** On validation it led by ~0.01 PR-AUC. One snapshot of ~5k
# MAGIC customers has sampling noise of the same order, so the comparison uses a **paired
# MAGIC bootstrap** (`modeling.compare.paired_bootstrap`, 1,000 resamples of customers; both
# MAGIC models scored on the same resample): the 95 % interval of the difference, and how often
# MAGIC LightGBM wins. If the interval contains 0, the honest statement is "no detectable
# MAGIC difference at this sample size". This also tells us whether the 0.005 promotion margin
# MAGIC (5.5) is above the noise level.

# COMMAND ----------

from modeling.compare import paired_bootstrap
from modeling.explain import (
    additivity_error,
    dependence_table,
    direction,
    importance,
    lgbm_contributions,
    plot_dependence,
    plot_summary,
    shap_values,
    strongest_interaction,
)

test = split_frames(to_pandas(training_set, splits=("test",)))["test"]
print(f"test: {test.n:,} customers, base rate {test.y.mean():.4f}")

logreg_model = mlflow.sklearn.load_model(LOGREG_MODEL_URI)
lgbm_model = mlflow.sklearn.load_model(LGBM_MODEL_URI)
p_test = {
    "lightgbm": lgbm_proba(lgbm_model, test.X),
    "logreg": positive_proba(logreg_model, test.X),
    "recency rule": recency_rule_scores(test.X),
}
p_val_reload = {"lightgbm": lgbm_proba(lgbm_model, val.X), "logreg": positive_proba(logreg_model, val.X)}
# the reloaded models must reproduce the logged validation metrics
for name, logged in (("lightgbm", lgbm), ("logreg", logreg)):
    gap = abs(evaluate(val.y, p_val_reload[name])["metrics"]["pr_auc"] - logged["validation"]["pr_auc"])
    assert gap < 1e-9, f"{name}: reloaded model differs from the logged one ({gap})"

# COMMAND ----------

RUN_TEST_EVAL = False  # recorded 2026-11-27 in the lgbm and logreg runs

if RUN_TEST_EVAL:
    ev_test = {name: evaluate(test.y, p) for name, p in p_test.items()}
    boot = {
        f"{split}_{m}": paired_bootstrap(y, pa, pb, m, n_boot=1000, seed=SEED)
        for split, y, pa, pb in (
            ("test", test.y, p_test["logreg"], p_test["lightgbm"]),
            ("validation", val.y, p_val_reload["logreg"], p_val_reload["lightgbm"]),
        )
        for m in ("pr_auc", "roc_auc", "brier")
    }
    test_scores = {name: (test.y, p) for name, p in p_test.items()}
    test_figs = {
        "roc_pr_test": plot_roc_pr(test_scores, "Test (cutoff 2011-09-11)"),
        "gain_test": plot_gain(test_scores, title="Cumulative gain (test)"),
        "lift_test": plot_lift(test_scores, title="Lift by decile (test)"),
        "calibration_test": plot_calibration(
            {k: test_scores[k] for k in ("lightgbm", "logreg")}, title="Calibration (test)"
        ),
    }
    test_payload = {
        "metrics": {name: ev["metrics"] for name, ev in ev_test.items()},
        "gain": {name: json.loads(ev["gain"].to_json(orient="records")) for name, ev in ev_test.items()},
        "calibration": {name: json.loads(ev["calibration"].to_json(orient="records")) for name, ev in ev_test.items()},
        "bootstrap_lgbm_minus_logreg": boot,
    }
    with mlflow.start_run(run_id=LGBM_RUN_ID):
        mlflow.log_metrics(prefixed(ev_test["lightgbm"]["metrics"], "test"))
        mlflow.log_metrics(prefixed(ev_test["recency rule"]["metrics"], "rule_test"))
        mlflow.log_metrics({f"boot_{k}_diff_ci_low": v["ci_low"] for k, v in boot.items()})
        mlflow.log_dict(test_payload, "evaluation/test.json")
        for name, fig in test_figs.items():
            mlflow.log_figure(fig, f"plots/{name}.png")
    with mlflow.start_run(run_id=LOGREG_RUN_ID):
        mlflow.log_metrics(prefixed(ev_test["logreg"]["metrics"], "test"))
        mlflow.log_dict({"metrics": ev_test["logreg"]["metrics"]}, "evaluation/test.json")
        for name, fig in test_figs.items():
            mlflow.log_figure(fig, f"plots/{name}.png")

test_eval = mlflow.artifacts.load_dict(f"runs:/{LGBM_RUN_ID}/evaluation/test.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 SHAP: what drives a repeat purchase?
# MAGIC
# MAGIC Computed on the **test snapshot** with the LightGBM model (the population we'd score
# MAGIC next). Units are **log-odds**: +0.5 moves a 30 % customer to about 41 %.
# MAGIC
# MAGIC - **Checks first**: SHAP from the `shap` library vs LightGBM's built-in TreeSHAP
# MAGIC   (`pred_contrib=True`) must agree, and base value + Σ SHAP must equal the raw score
# MAGIC   for every customer (additivity).
# MAGIC - **Summary (beeswarm)**: each dot a customer; x = effect on the score; colour = feature
# MAGIC   value (red high). Ranked by mean |SHAP|, which differs from the split-gain importance
# MAGIC   in 4.x: gain counts how much splits improved training loss, SHAP measures how much a
# MAGIC   feature actually moves predictions.
# MAGIC - **Dependence plots** for the top 4 features: the *shape* of each effect (thresholds,
# MAGIC   saturation), coloured by its strongest interaction partner.
# MAGIC
# MAGIC Caveat *(interview point)*: SHAP explains the **model**, not causality. "More orders in
# MAGIC the last 180 days → higher score" doesn't mean a promotion that buys an order causes
# MAGIC loyalty; correlated features (orders vs revenue windows) also share credit.

# COMMAND ----------

RUN_SHAP = False  # recorded 2026-11-27 (lgbm run, shap/)
N_DEPENDENCE = 4

if RUN_SHAP:
    sv, base_value = shap_values(lgbm_model, test.X)
    contrib, contrib_base = lgbm_contributions(lgbm_model, test.X)
    shap_checks = {
        "max_abs_diff_shap_vs_lightgbm": float(np.max(np.abs(sv - contrib))),
        "base_value_diff": abs(base_value - contrib_base),
        "additivity_error": additivity_error(lgbm_model, test.X, sv, base_value),
        "base_value_log_odds": base_value,
        "base_value_probability": float(1 / (1 + np.exp(-base_value))),
    }
    assert shap_checks["max_abs_diff_shap_vs_lightgbm"] < 1e-6 and shap_checks["additivity_error"] < 1e-6

    imp = importance(sv, test.X.columns)
    dirs = direction(sv, test.X)
    imp["direction"] = imp["feature"].map(dirs)
    top_features = imp["feature"].head(N_DEPENDENCE).tolist()
    interactions = {f: strongest_interaction(sv, test.X, f) for f in top_features}
    dep_tables = {f: dependence_table(sv, test.X, f) for f in top_features}

    shap_figs = {
        "shap_summary": plot_summary(sv, test.X),
        "shap_importance": plot_bars(imp.set_index("feature")["mean_abs_shap"], "Mean |SHAP| (test)", "mean |SHAP| (log-odds)"),
        **{f"shap_dependence_{f}": plot_dependence(sv, test.X, f, interactions[f]) for f in top_features},
    }
    with mlflow.start_run(run_id=LGBM_RUN_ID):
        mlflow.log_dict(
            {
                "checks": shap_checks,
                "importance": json.loads(imp.to_json(orient="records")),
                "interactions": interactions,
                "dependence": {f: json.loads(t.to_json(orient="records")) for f, t in dep_tables.items()},
            },
            "shap/summary.json",
        )
        for name, fig in shap_figs.items():
            mlflow.log_figure(fig, f"shap/{name}.png")
    display(imp)

shap_summary = mlflow.artifacts.load_dict(f"runs:/{LGBM_RUN_ID}/shap/summary.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Results to paste into the chat

# COMMAND ----------

m = test_eval["metrics"]
lines = ["### Phase 5.4 results", "", f"- test: {test.n:,} customers, base rate {test.y.mean():.4f}"]
lines += ["", "| model (test) | " + " | ".join(KEYS) + " |", "|---|" + "---:|" * len(KEYS)]
lines += [f"| {name} | {_fmt(m[name])} |" for name in ("lightgbm", "logreg", "recency rule")]
lines += ["", "| LightGBM − logreg | diff | 95 % CI | P(LightGBM better) |", "|---|---:|---|---:|"]
lines += [
    f"| {k} | {b['diff']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | {b['p_b_better']:.3f} |"
    for k, b in test_eval["bootstrap_lgbm_minus_logreg"].items()
]
lines += ["", "| band (test, lightgbm) | customers | positives | rate | lift | cum. share of buyers |", "|---:|---:|---:|---:|---:|---:|"]
lines += [
    f"| {g['band']} | {g['customers']:,} | {g['positives']:,} | {g['rate']:.3f} | {g['lift']:.2f} | {g['cum_positive_share']:.3f} |"
    for g in test_eval["gain"]["lightgbm"]
]
lines += ["", "| calib. bin | lgbm predicted | lgbm observed | logreg predicted | logreg observed |", "|---:|---:|---:|---:|---:|"]
lines += [
    f"| {a['bin']} | {a['mean_predicted']:.3f} | {a['observed_rate']:.3f} | {b['mean_predicted']:.3f} | {b['observed_rate']:.3f} |"
    for a, b in zip(test_eval["calibration"]["lightgbm"], test_eval["calibration"]["logreg"], strict=True)
]
lines += ["", f"- SHAP checks: {shap_summary['checks']}"]
lines += ["", "| feature | mean abs SHAP | share | direction (Spearman) |", "|---|---:|---:|---:|"]
lines += [
    f"| {r['feature']} | {r['mean_abs_shap']:.4f} | {r['share']:.3f} | {r['direction']:+.2f} |" if r["direction"] is not None
    else f"| {r['feature']} | {r['mean_abs_shap']:.4f} | {r['share']:.3f} | n/a |"
    for r in shap_summary["importance"][:12]
]


def _bin_text(r):
    x_range = "NULL" if r["x_min"] is None else f"{r['x_min']:.4g}-{r['x_max']:.4g}"
    return f"{x_range} -> {r['mean_shap']:+.3f} (n={r['n']})"


for f, rows in shap_summary["dependence"].items():
    lines += ["", f"- dependence `{f}` (interaction: {shap_summary['interactions'][f]}): " + "; ".join(_bin_text(r) for r in rows)]
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Unity Catalog model registry: champion / challenger (step 5.5)
# MAGIC
# MAGIC ### 6.0 Concepts and the promotion rule *(interview point)*
# MAGIC
# MAGIC - **Registered model** `retail.ml.repeat_purchase_model`: a Unity Catalog object (same
# MAGIC   three-level name as a table) with grants, lineage to its training run, and versions.
# MAGIC - **Version**: an immutable copy of one logged model (with its signature, input example
# MAGIC   and pinned requirements). Registering needs the signature: UC rejects models without one.
# MAGIC - **Alias**: a movable pointer, e.g. `@champion`. The Phase 6 scoring job will load
# MAGIC   `models:/retail.ml.repeat_purchase_model@champion`; promotion or rollback = moving
# MAGIC   the alias, with no change to the job. (Aliases replace the old Staging/Production stages.)
# MAGIC
# MAGIC **Promotion rule** (`modeling.promote.PromotionRule`), decided on the **validation
# MAGIC cutoff**, with both models scored again on the same rows at decision time:
# MAGIC
# MAGIC | Check | Threshold | Why |
# MAGIC |---|---|---|
# MAGIC | PR-AUC gain | ≥ +0.005 | primary metric; a minimum practical gain |
# MAGIC | paired-bootstrap 95 % CI of the gain | lower bound > 0 | 5.4 showed ±~0.007 noise on one snapshot: the margin alone would sit inside it |
# MAGIC | ROC-AUC | not worse by > 0.01 | guardrail: overall ranking |
# MAGIC | Brier | not worse by > 0.005 | guardrail: probabilities must stay usable |
# MAGIC
# MAGIC Flow: register the **logistic regression → version 1**; with no champion yet it becomes
# MAGIC `@champion`. Register **LightGBM → version 2** as `@challenger`; the rule decides. If it
# MAGIC passes, `@champion` moves to version 2 and `@challenger` is removed; if not, version 2
# MAGIC keeps `@challenger` for inspection. Every decision is stored on the version (tags) and in
# MAGIC an MLflow run. Registration is idempotent (a version is found again by its source run),
# MAGIC and the rule only runs for newly registered versions, so a re-run can't demote anything.
# MAGIC
# MAGIC Alternatives: promote on test (then test is no longer untouched for the next model),
# MAGIC shadow / A-B deployment (compare live outcomes before switching: better, but needs 90
# MAGIC days of labels per comparison), or a manual approval step after the rule (common in
# MAGIC regulated settings; easy to add in Phase 6 as a job task waiting on a tag).

# COMMAND ----------

from mlflow import MlflowClient

from common.tables import CHAMPION_ALIAS, ML_PROPENSITY_MODEL
from modeling.promote import (
    PromotionRule,
    aliases_table,
    positive_scores,
    promote_challenger,
    register,
)

client = MlflowClient()
RULE = PromotionRule()
MODEL_DESCRIPTION = (
    "Repeat-purchase propensity: P(customer places an order in the 90 days after the scoring "
    "date). Input: modeling.dataset.prepare_features(training-set columns) as float64; output: "
    "predict_proba, column 1 = P(repeat). Trained on retail.ml.propensity_training_set "
    "(rolling monthly cutoffs, time-based split). Load with @champion."
)
CANDIDATES = [
    {
        "run_id": LOGREG_RUN_ID,
        "model_uri": LOGREG_MODEL_URI,
        "model_type": "logistic_regression",
        "description": "Baseline: impute + log1p + standardise + L2 logistic regression (C = 0.01).",
        "metrics": {"validation": logreg["validation"], "test": test_eval["metrics"]["logreg"]},
    },
    {
        "run_id": LGBM_RUN_ID,
        "model_uri": LGBM_MODEL_URI,
        "model_type": "lightgbm",
        "description": f"LightGBM tuned with Optuna (40 trials), {lgbm['best_iteration']} trees.",
        "metrics": {"validation": lgbm["validation"], "test": test_eval["metrics"]["lightgbm"]},
    },
]

RUN_REGISTRY = True
REGISTRY_RUN_ID = None  # paste after the first run

if RUN_REGISTRY:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name="registry_promotion") as run:
        mlflow.set_tags({"phase": "5", "kind": "registry", "model_name": ML_PROPENSITY_MODEL})
        mlflow.log_params({f"rule_{k}": v for k, v in RULE.to_dict().items()})
        steps = []
        for c in CANDIDATES:
            tags = {
                "model_type": c["model_type"],
                "training_set_run_id": TRAINING_SET_RUN_ID,
                "validation_pr_auc": f"{c['metrics']['validation']['pr_auc']:.4f}",
                "test_pr_auc": f"{c['metrics']['test']['pr_auc']:.4f}",
                "n_features": str(len(MODEL_FEATURES)),
            }
            version, created = register(client, c["model_uri"], ML_PROPENSITY_MODEL, c["run_id"], tags, c["description"])
            if version == "1" or created:
                client.update_registered_model(ML_PROPENSITY_MODEL, description=MODEL_DESCRIPTION)
            step = {"model_type": c["model_type"], "version": version, "created": created}
            if created:
                d = promote_challenger(client, ML_PROPENSITY_MODEL, version, val.X, val.y, RULE, evaluated_on="validation 2011-06-13")
                step["decision"] = d.to_dict()
                print(f"v{version} ({c['model_type']}): promote={d.promote} -- {d.reason}")
                for chk in d.checks:
                    print(f"    {'PASS' if chk['passed'] else 'FAIL'} {chk['check']}: {chk['value']:+.4f}")
            else:
                print(f"v{version} ({c['model_type']}) already registered: rule not re-run")
            steps.append(step)
        mlflow.log_dict({"rule": RULE.to_dict(), "steps": steps}, "registry/promotion.json")
        REGISTRY_RUN_ID = run.info.run_id

promotion = mlflow.artifacts.load_dict(f"runs:/{REGISTRY_RUN_ID}/registry/promotion.json")
versions = aliases_table(client, ML_PROPENSITY_MODEL)
display(versions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Load the champion the way Phase 6 will
# MAGIC
# MAGIC Generic MLflow loader + alias, nothing model-specific: the scoring job doesn't need to
# MAGIC know whether the champion is LightGBM or a logistic regression. Checks: the signature is
# MAGIC stored, the champion's test scores reproduce the 5.4 metrics of the model it points to.

# COMMAND ----------

champion_uri = f"models:/{ML_PROPENSITY_MODEL}@{CHAMPION_ALIAS}"
champion_mv = client.get_model_version_by_alias(ML_PROPENSITY_MODEL, CHAMPION_ALIAS)
champion_info = mlflow.models.get_model_info(champion_uri)
assert champion_info.signature is not None, "registered model has no signature"
signature_inputs = [c["name"] for c in champion_info.signature.inputs.to_dict()]
assert signature_inputs == list(MODEL_FEATURES), "signature columns differ from MODEL_FEATURES"

champion = mlflow.pyfunc.load_model(champion_uri)
p_champion = positive_scores(champion.predict(test.X))
champion_type = client.get_model_version(ML_PROPENSITY_MODEL, champion_mv.version).tags.get("model_type")
expected = test_eval["metrics"]["lightgbm" if champion_type == "lightgbm" else "logreg"]
got = evaluate(test.y, p_champion)["metrics"]
assert abs(got["pr_auc"] - expected["pr_auc"]) < 1e-9, (got["pr_auc"], expected["pr_auc"])
print(f"@champion = version {champion_mv.version} ({champion_type}); test PR-AUC {got['pr_auc']:.4f} reproduced")

# COMMAND ----------

lines = ["### Phase 5.5 results", "", f"- registry run: {REGISTRY_RUN_ID}; model `{ML_PROPENSITY_MODEL}`"]
lines += [f"- rule: {promotion['rule']}"]
for st in promotion["steps"]:
    d = st.get("decision")
    head = f"- v{st['version']} {st['model_type']} (created={st['created']})"
    if d is None:
        lines += [head + ": already registered"]
        continue
    lines += [head + f": promote={d['promote']}, reason: {d['reason']}, vs champion v{d['champion_version']}"]
    lines += [f"    - {'PASS' if c['passed'] else 'FAIL'} {c['check']}: {c['value']:+.4f}" for c in d["checks"]]
    if d.get("bootstrap"):
        b = d["bootstrap"]
        lines += [f"    - bootstrap {b['metric']}: diff {b['diff']:+.4f}, 95 % CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}], P(better) {b['p_b_better']:.3f}"]
lines += ["", "| version | aliases | model_type | source run |", "|---:|---|---|---|"]
lines += [f"| {r.version} | {r.aliases} | {r.model_type} | {r.source_run_id} |" for r in versions.itertuples()]
lines += [f"- @champion = v{champion_mv.version} ({champion_type}), signature: {len(signature_inputs)} inputs, test PR-AUC reproduced {got['pr_auc']:.4f}"]
print("\n".join(lines))
