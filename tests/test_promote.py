"""Tests for modeling.promote (Phase 5.5): the rule, and the alias moves on a fake registry."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from modeling.promote import (  # noqa: E402
    CHALLENGER,
    CHAMPION,
    PromotionRule,
    alias_version,
    aliases_table,
    decide,
    find_version,
    positive_scores,
    promote_challenger,
    register,
)

NAME = "retail.ml.repeat_purchase_model"


class FakeRegistry:
    """Just enough of MlflowClient for the promotion code."""

    def __init__(self):
        self.versions: dict[str, dict] = {}
        self.aliases: dict[str, str] = {}
        self.calls: list[tuple] = []

    def create(self, uri, name):
        v = str(len(self.versions) + 1)
        self.versions[v] = {"uri": uri, "tags": {}, "description": None}
        return SimpleNamespace(version=v)

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
        return SimpleNamespace(version=self.aliases[alias])

    def set_registered_model_alias(self, name, alias, version):
        self.calls.append(("set_alias", alias, version))
        self.aliases[alias] = str(version)

    def delete_registered_model_alias(self, name, alias):
        self.calls.append(("delete_alias", alias))
        self.aliases.pop(alias, None)

    def set_model_version_tag(self, name, version, key, value):
        self.versions[str(version)]["tags"][key] = value

    def update_model_version(self, name, version, description=None):
        self.versions[str(version)]["description"] = description

    def search_model_versions(self, filter_string):
        # like Unity Catalog: search results carry no tags / aliases
        return [SimpleNamespace(version=v) for v in self.versions]

    def get_model_version(self, name, version):
        d = self.versions[str(version)]
        aliases = [a for a, ver in self.aliases.items() if ver == str(version)]
        return SimpleNamespace(version=str(version), tags=d["tags"], aliases=aliases)


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.4).astype(int)
    weak = 1 / (1 + np.exp(-(0.8 * y + rng.normal(0, 1.0, n))))
    strong = 1 / (1 + np.exp(-(1.6 * y + rng.normal(0, 1.0, n) - 0.6)))
    X = pd.DataFrame({"row": np.arange(n)})
    return X, y, {"weak": weak, "strong": strong}


def _loader(scores_by_version):
    return lambda name, version: lambda X: scores_by_version[version][X["row"].to_numpy()]


# ----------------------------------------------------------------------------- rule


def test_rule_defaults_are_the_agreed_ones():
    r = PromotionRule()
    assert (r.metric, r.min_improvement, r.max_roc_auc_drop, r.max_brier_increase) == (
        "pr_auc",
        0.005,
        0.01,
        0.005,
    )
    assert r.require_significant and r.alpha == 0.05


def test_decide_promotes_a_clearly_better_model():
    _, y, s = _data()
    d = decide(y, s["weak"], s["strong"], PromotionRule(n_boot=200))
    assert d.promote, d.reason
    assert all(c["passed"] for c in d.checks)
    assert d.bootstrap["ci_low"] > 0


def test_decide_rejects_an_equal_model():
    _, y, s = _data()
    d = decide(y, s["strong"], s["strong"], PromotionRule(n_boot=50))
    assert not d.promote and "improvement" in d.reason


def test_decide_rejects_small_noisy_gain_but_margin_only_rule_would_accept():
    rng = np.random.default_rng(5)
    n = 400
    y = (rng.uniform(size=n) < 0.4).astype(int)
    base = 1 / (1 + np.exp(-(y + rng.normal(0, 1.2, n))))
    noisy = np.clip(base + rng.normal(0, 0.03, n), 0.001, 0.999)
    strict = decide(y, base, noisy, PromotionRule(n_boot=300, min_improvement=-1.0))
    loose = decide(
        y, base, noisy, PromotionRule(n_boot=300, min_improvement=-1.0, require_significant=False)
    )
    assert not strict.promote and "bootstrap" in strict.reason
    assert loose.promote


def test_decide_guardrail_on_calibration():
    """Better ranking but badly calibrated (all scores squashed upwards) -> Brier guardrail."""
    _, y, s = _data()
    squashed = 0.6 + 0.4 * s["strong"]  # same ranking as strong, probabilities too high
    d = decide(y, s["weak"], squashed, PromotionRule(n_boot=100))
    assert not d.promote and "brier" in d.reason


# ----------------------------------------------------------------------------- registry


def test_register_is_idempotent_per_source_run():
    reg = FakeRegistry()
    v1, created1 = register(
        reg, "models:/m-1", NAME, "run-a", {"model_type": "lr"}, "LR", register_fn=reg.create
    )
    v1b, created1b = register(reg, "models:/m-1", NAME, "run-a", register_fn=reg.create)
    v2, created2 = register(reg, "models:/m-2", NAME, "run-b", register_fn=reg.create)
    assert (v1, created1, v1b, created1b, v2, created2) == ("1", True, "1", False, "2", True)
    assert reg.versions["1"]["tags"] == {"source_run_id": "run-a", "model_type": "lr"}
    assert reg.versions["1"]["description"] == "LR"
    assert find_version(reg, NAME, "run-b") == "2" and find_version(reg, NAME, "nope") is None


def test_full_champion_challenger_flow():
    reg = FakeRegistry()
    X, y, s = _data()
    v1, _ = register(reg, "u1", NAME, "run-lr", register_fn=reg.create)
    v2, _ = register(reg, "u2", NAME, "run-lgbm", register_fn=reg.create)
    v3, _ = register(reg, "u3", NAME, "run-bad", register_fn=reg.create)
    loader = _loader({"1": s["weak"], "2": s["strong"], "3": s["weak"]})
    rule = PromotionRule(n_boot=100)

    d1 = promote_challenger(reg, NAME, v1, X, y, rule, "validation", loader)
    assert d1.promote and d1.reason == "no champion yet"
    assert (
        alias_version(reg, NAME, CHAMPION) == "1" and alias_version(reg, NAME, CHALLENGER) is None
    )

    d2 = promote_challenger(reg, NAME, v2, X, y, rule, "validation", loader)
    assert d2.promote and d2.champion_version == "1" and d2.challenger_version == "2"
    assert (
        alias_version(reg, NAME, CHAMPION) == "2" and alias_version(reg, NAME, CHALLENGER) is None
    )
    assert json.loads(reg.versions["2"]["tags"]["promotion_decision"])["vs_champion"] == "1"

    d3 = promote_challenger(reg, NAME, v3, X, y, rule, "validation", loader)
    assert not d3.promote
    assert alias_version(reg, NAME, CHAMPION) == "2"  # champion unchanged
    assert alias_version(reg, NAME, CHALLENGER) == "3"  # rejected challenger kept for inspection

    d4 = promote_challenger(reg, NAME, v2, X, y, rule, "validation", loader)  # re-run on champion
    assert not d4.promote and "already the champion" in d4.reason
    assert alias_version(reg, NAME, CHAMPION) == "2"

    table = aliases_table(reg, NAME)
    assert table["version"].tolist() == [1, 2, 3]
    assert table.set_index("version").loc[2, "aliases"] == "champion"


def test_alias_version_missing_model_is_none():
    assert alias_version(FakeRegistry(), NAME, CHAMPION) is None


def test_positive_scores_shapes():
    assert positive_scores(np.array([[0.7, 0.3], [0.2, 0.8]])).tolist() == [0.3, 0.8]
    assert positive_scores(pd.DataFrame({"0": [0.6], "1": [0.4]})).tolist() == [0.4]
    assert positive_scores([0.1, 0.2]).tolist() == [0.1, 0.2]
