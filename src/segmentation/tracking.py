"""MLflow logging for segmentation runs (Phase 3).

Layout in the experiment:
- one parent run per sweep (`k_sweep`), holding the comparison plots and all results as JSON,
- one nested child run per (algorithm, k): params, metrics, cluster shares, model parameters.

mlflow is imported inside the functions: it's on Databricks, not a local test dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from segmentation.clustering import TrialResult


def flat_row(r: TrialResult | Mapping) -> dict:
    """One flat dict per trial (for a table / pandas)."""
    d = r.to_dict() if isinstance(r, TrialResult) else dict(r)
    row = {
        "algorithm": d["algorithm"],
        "k": d["k"],
        "implementation": d["implementation"],
        "error": d.get("error"),
    }
    row.update(d.get("metrics", {}))
    return row


def results_table(results: Sequence[TrialResult | Mapping]) -> list[dict]:
    return [flat_row(r) for r in results]


def trial_run_name(r: TrialResult) -> str:
    return f"{r.algorithm}_k{r.k}"


def log_trial(r: TrialResult, common_params: Mapping[str, object]) -> str:
    """Log one trial as a nested run under the active parent run. Returns its run id."""
    import mlflow

    with mlflow.start_run(run_name=trial_run_name(r), nested=True) as run:
        mlflow.set_tags(
            {
                "phase": "3",
                "kind": "sweep_trial",
                "algorithm": r.algorithm,
                "implementation": r.implementation,
                "status": "failed" if r.error else "ok",
            }
        )
        if r.error:
            mlflow.set_tag("error", r.error[:250])
        mlflow.log_params(
            {
                **common_params,
                "algorithm": r.algorithm,
                "k": r.k,
                "seed": r.seed,
                "implementation": r.implementation,
            }
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in r.metrics.items() if isinstance(v, int | float)}
        )
        mlflow.log_dict(r.to_dict(), "trial/result.json")
    return run.info.run_id
