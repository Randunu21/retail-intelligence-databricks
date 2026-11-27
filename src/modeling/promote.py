"""Champion / challenger promotion in the Unity Catalog model registry (Phase 5.5, reused in
Phase 6 retraining jobs).

Registry vocabulary:
- **Registered model** `retail.ml.repeat_purchase_model`: a UC object (catalog.schema.name)
  with grants and lineage, like a table.
- **Version** (1, 2, ...): an immutable snapshot of one logged model.
- **Alias** (`@champion`, `@challenger`): a movable pointer to one version. Scoring jobs load
  `models:/retail.ml.repeat_purchase_model@champion`, so promoting or rolling back is moving an
  alias: no job or code change. (Aliases replace the old Staging/Production "stages".)

The promotion rule (`PromotionRule`, defaults agreed in Phase 5):
1. no champion yet -> the challenger becomes champion;
2. otherwise score **both on the same rows now** (never compare metrics stored from different
   data) and promote only if
   - PR-AUC(challenger) - PR-AUC(champion) >= `min_improvement` (0.005), **and**
   - the paired-bootstrap 95 % interval of that difference is above 0
     (`require_significant`; on ~5k customers the noise is about +-0.007, so the margin
     alone would sit inside it), **and** the guardrails hold:
   - ROC-AUC not worse by more than `max_roc_auc_drop` (0.01),
   - Brier score not worse by more than `max_brier_increase` (0.005).
3. promoted: `@champion` moves to the challenger and `@challenger` is removed; rejected:
   `@challenger` stays on it for inspection. The decision is written to the version's tags.

`decide` is pure (arrays in, Decision out). `promote_challenger` does the registry calls
through a client object (MlflowClient in production, a fake in tests).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from modeling.compare import paired_bootstrap
from modeling.evaluation import binary_metrics

CHAMPION = "champion"
CHALLENGER = "challenger"
SOURCE_RUN_TAG = "source_run_id"


@dataclass(frozen=True)
class PromotionRule:
    metric: str = "pr_auc"
    min_improvement: float = 0.005
    require_significant: bool = True
    alpha: float = 0.05
    n_boot: int = 1000
    max_roc_auc_drop: float = 0.01
    max_brier_increase: float = 0.005
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    promote: bool
    reason: str
    champion_version: str | None = None
    challenger_version: str | None = None
    checks: list = field(default_factory=list)
    champion_metrics: dict | None = None
    challenger_metrics: dict | None = None
    bootstrap: dict | None = None
    rule: dict | None = None
    evaluated_on: str | None = None
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


def decide(y, p_champion, p_challenger, rule: PromotionRule = PromotionRule()) -> Decision:
    """Apply the rule to two score vectors on the same customers (champion present)."""
    champ = binary_metrics(y, p_champion)
    chall = binary_metrics(y, p_challenger)
    boot = paired_bootstrap(
        y, p_champion, p_challenger, rule.metric, rule.n_boot, rule.seed, rule.alpha
    )
    gain = chall[rule.metric] - champ[rule.metric]
    checks = [
        {
            "check": f"{rule.metric} improvement >= {rule.min_improvement}",
            "value": gain,
            "passed": gain >= rule.min_improvement,
        },
        {
            "check": "bootstrap lower bound > 0",
            "value": boot["ci_low"],
            "passed": (boot["ci_low"] > 0) or not rule.require_significant,
        },
        {
            "check": f"roc_auc drop <= {rule.max_roc_auc_drop}",
            "value": champ["roc_auc"] - chall["roc_auc"],
            "passed": champ["roc_auc"] - chall["roc_auc"] <= rule.max_roc_auc_drop,
        },
        {
            "check": f"brier increase <= {rule.max_brier_increase}",
            "value": chall["brier"] - champ["brier"],
            "passed": chall["brier"] - champ["brier"] <= rule.max_brier_increase,
        },
    ]
    failed = [c["check"] for c in checks if not c["passed"]]
    return Decision(
        promote=not failed,
        reason="all checks passed" if not failed else "failed: " + "; ".join(failed),
        checks=checks,
        champion_metrics=champ,
        challenger_metrics=chall,
        bootstrap=boot,
        rule=rule.to_dict(),
    )


# ----------------------------------------------------------------------------- registry


def alias_version(client, name: str, alias: str) -> str | None:
    """Version number an alias points to, or None if the alias isn't set."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except Exception:  # RESOURCE_DOES_NOT_EXIST (also when the model doesn't exist yet)
        return None


def _full_versions(client, name: str) -> list:
    """All versions with tags and aliases. Unity Catalog's search_model_versions may leave
    tags/aliases out of its results, so each version is fetched with get_model_version."""
    try:
        found = client.search_model_versions(f"name='{name}'")
    except Exception:  # model doesn't exist yet
        return []
    return [client.get_model_version(name, str(v.version)) for v in found]


def find_version(client, name: str, source_run_id: str) -> str | None:
    """Version registered earlier from this run (tag `source_run_id`), for idempotent re-runs."""
    versions = _full_versions(client, name)
    hits = [
        v for v in versions if (getattr(v, "tags", None) or {}).get(SOURCE_RUN_TAG) == source_run_id
    ]
    return str(max(int(v.version) for v in hits)) if hits else None


def register(
    client,
    model_uri: str,
    name: str,
    source_run_id: str,
    tags: Mapping[str, str] | None = None,
    description: str | None = None,
    register_fn: Callable | None = None,
) -> tuple[str, bool]:
    """Register `model_uri` as a new version unless this run is already registered.

    Returns (version, created). `register_fn` defaults to `mlflow.register_model`.
    """
    existing = find_version(client, name, source_run_id)
    if existing is not None:
        return existing, False
    if register_fn is None:
        import mlflow

        register_fn = mlflow.register_model
    mv = register_fn(model_uri, name)
    version = str(mv.version)
    for k, v in {SOURCE_RUN_TAG: source_run_id, **(tags or {})}.items():
        client.set_model_version_tag(name, version, k, str(v))
    if description:
        client.update_model_version(name, version, description=description)
    return version, True


def positive_scores(predictions) -> np.ndarray:
    """pyfunc output -> P(repeat): 2-column predict_proba output, or a single column."""
    a = np.asarray(
        predictions.to_numpy() if isinstance(predictions, pd.DataFrame) else predictions,
        dtype=float,
    )
    return a[:, -1] if a.ndim == 2 else a.ravel()


def load_scorer(name: str, version: str) -> Callable[[pd.DataFrame], np.ndarray]:
    """Score function for one registered version (generic MLflow pyfunc loader)."""
    import mlflow

    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return lambda X: positive_scores(model.predict(X))


def promote_challenger(
    client,
    name: str,
    challenger_version: str,
    X: pd.DataFrame,
    y,
    rule: PromotionRule = PromotionRule(),
    evaluated_on: str = "",
    loader: Callable[[str, str], Callable] = load_scorer,
) -> Decision:
    """Point @challenger at the version, decide against @champion, move aliases, tag the result."""
    challenger_version = str(challenger_version)
    champion = alias_version(client, name, CHAMPION)

    if champion == challenger_version:
        decision = Decision(
            False,
            "challenger is already the champion",
            champion,
            challenger_version,
            rule=rule.to_dict(),
        )
    else:
        client.set_registered_model_alias(name, CHALLENGER, challenger_version)
        if champion is None:
            p = loader(name, challenger_version)(X)
            decision = Decision(
                True,
                "no champion yet",
                None,
                challenger_version,
                challenger_metrics=binary_metrics(y, p),
                rule=rule.to_dict(),
            )
        else:
            decision = decide(
                y, loader(name, champion)(X), loader(name, challenger_version)(X), rule
            )
            decision.champion_version, decision.challenger_version = champion, challenger_version
    decision.evaluated_on = evaluated_on

    if decision.promote:
        client.set_registered_model_alias(name, CHAMPION, challenger_version)
        client.delete_registered_model_alias(name, CHALLENGER)
    if champion != challenger_version:
        summary = {"promote": decision.promote, "reason": decision.reason, "vs_champion": champion}
        client.set_model_version_tag(
            name, challenger_version, "promotion_decision", json.dumps(summary)
        )
        client.set_model_version_tag(
            name, challenger_version, "promotion_decided_at", decision.decided_at
        )
        if evaluated_on:
            client.set_model_version_tag(
                name, challenger_version, "promotion_evaluated_on", evaluated_on
            )
    return decision


def aliases_table(client, name: str) -> pd.DataFrame:
    """One row per version: version, aliases, selected tags (for the notebook / docs)."""
    rows = []
    for v in _full_versions(client, name):
        tags = getattr(v, "tags", None) or {}
        rows.append(
            {
                "version": int(v.version),
                "aliases": ", ".join(getattr(v, "aliases", None) or []),
                "model_type": tags.get("model_type"),
                SOURCE_RUN_TAG: tags.get(SOURCE_RUN_TAG),
                "promotion_decision": tags.get("promotion_decision"),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("version").reset_index(drop=True)
        if rows
        else pd.DataFrame(rows)
    )
