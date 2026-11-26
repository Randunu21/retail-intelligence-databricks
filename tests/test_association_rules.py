"""Tests for the rule part of segmentation.market_basket and segmentation.association_outputs
(Phase 4.5). The key test: FP-Growth's single-item rules equal an independent recount with
plain joins (support, confidence, lift)."""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from segmentation.association_outputs import (
    ASSOCIATIONS_COLUMNS,
    associations_ddl,
    build_associations_output,
    top_rules,
)
from segmentation.market_basket import (
    classify_rules,
    compare_pair_rules,
    description_tokens,
    fit_rules,
    min_count,
    pair_rules_sql,
    product_lookup,
    rules_frame,
)

DESCRIPTIONS = [
    ("85099B", "JUMBO BAG RED RETROSPOT"),
    ("85099C", "JUMBO  BAG BAROQUE BLACK WHITE"),
    ("20725", "LUNCH BAG RED RETROSPOT"),
    ("22383", "LUNCH BAG SUKI DESIGN"),
    ("22423", "REGENCY CAKESTAND 3 TIER"),
    ("22697", "GREEN REGENCY TEACUP AND SAUCER"),
    ("22699", "ROSES REGENCY TEACUP AND SAUCER"),
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER"),
    ("20676", "RED RETROSPOT CUP"),
    ("20677", "PINK POLKADOT CUP"),
    ("21218", "RED RETROSPOT SUGAR JAM BOWL"),
    ("21219", "RED RETROSPOT SMALL MILK JUG"),
    ("23175", "KEY FOB , BACK DOOR"),
    ("23176", "KEY FOB , GARAGE DESIGN"),
    ("21733", "RED HANGING HEART T-LIGHT HOLDER"),
]


@pytest.fixture
def lookup(spark):
    return product_lookup(
        spark.createDataFrame(DESCRIPTIONS, "stock_code STRING, description STRING")
    )


@pytest.fixture
def baskets(spark):
    b = [
        ["85099B", "85099C", "22423", "22697"],
        ["85099B", "85099C", "22423"],
        ["85099B", "85099C"],
        ["20725", "22383", "22423"],
        ["20725", "22383"],
        ["22423", "22697", "22699"],
        ["22423", "22697", "22699"],
        ["85123A", "21733"],
        ["85123A", "21733", "22423"],
        ["85123A"],
    ]
    return spark.createDataFrame(
        [(f"i{i}", sorted(x)) for i, x in enumerate(b)], "invoice STRING, items ARRAY<STRING>"
    )


def test_description_tokens(spark):
    df = spark.createDataFrame(
        [("LUNCH BAG RED RETROSPOT",), ("SET OF 3 CAKE TINS PANTRY DESIGN",), (None,)],
        "d STRING",
    )
    got = [r[0] for r in df.select(description_tokens(F.col("d"))).collect()]
    assert got[0] == ["BAG", "LUNCH", "RETROSPOT"]
    assert got[1] == ["CAKE", "PANTRY", "TINS"]
    assert got[2] is None or got[2] == []


def test_min_count_matches_spark():
    assert min_count(0.01, 39516) == 396
    assert min_count(0.2, 10) == 2


def test_fpgrowth_pair_rules_equal_sql_recount(spark, baskets):
    fit = fit_rules(baskets, 0.2, 0.3)
    assert fit["n_baskets"] == 10 and fit["n_itemsets"] > 0
    rules = rules_frame(spark, fit["rules"], fit["n_baskets"], "all")
    sql = pair_rules_sql(baskets, min_count(0.2, 10))
    cmp = compare_pair_rules(rules, sql, 0.3)
    assert cmp["unmatched"] == 0
    assert cmp["fp_pair_rules"] == cmp["sql_pair_rules"] > 0
    assert cmp["max_support_diff"] < 1e-12
    assert cmp["max_confidence_diff"] < 1e-12
    assert cmp["max_lift_diff"] < 1e-9


def test_rules_frame_columns(spark, baskets):
    fit = fit_rules(baskets, 0.2, 0.3)
    rules = rules_frame(spark, fit["rules"], 10, "UK")
    r = rules.where(
        (F.col("antecedent") == F.array(F.lit("22697"))) & (F.col("consequent_item") == "22699")
    ).first()
    # 22697 in 3 baskets, 22697 & 22699 in 2, 22699 in 2: conf 2/3, lift (2/3)/(2/10)
    assert r["confidence"] == pytest.approx(2 / 3)
    assert r["lift"] == pytest.approx((2 / 3) / 0.2)
    assert r["baskets_with_rule"] == 2 and r["scope"] == "UK" and r["antecedent_size"] == 1
    assert rules.select("rule_id").distinct().count() == rules.count()


def _rule(spark, lookup, antecedent, consequent):
    df = rules_frame(
        spark,
        [
            {
                "antecedent": antecedent,
                "consequent": [consequent],
                "confidence": 0.5,
                "lift": 3.0,
                "support": 0.1,
            }
        ],
        10,
        "all",
    )
    return classify_rules(df, lookup).first()


def test_classify_rules(spark, lookup):
    # same stock-code family -> variant
    assert _rule(spark, lookup, ["85099B"], "85099C")["rule_type"] == "variant"
    # different codes, same description without the colour -> variant
    assert _rule(spark, lookup, ["85123A"], "21733")["rule_type"] == "variant"
    # same item type, other design -> similar
    r = _rule(spark, lookup, ["20725"], "22383")
    assert r["rule_type"] == "similar" and r["description_similarity"] == pytest.approx(0.5)
    # teacups in two designs: {REGENCY, TEACUP, SAUCER} vs {ROSES, REGENCY, TEACUP, SAUCER}
    assert _rule(spark, lookup, ["22697"], "22699")["rule_type"] == "similar"
    # short descriptions sharing 1 of 3 / 2 of 5 words (Phase 4 recalibration to 1/3)
    r = _rule(spark, lookup, ["20677"], "20676")  # {POLKADOT, CUP} vs {RETROSPOT, CUP}
    assert r["rule_type"] == "similar" and r["description_similarity"] == pytest.approx(1 / 3)
    assert _rule(spark, lookup, ["23175"], "23176")["rule_type"] == "similar"  # 2/5
    # same range but different product types stays complementary: 1/6
    assert _rule(spark, lookup, ["21218"], "21219")["rule_type"] == "complementary"
    # cakestand -> teacup: complementary
    assert _rule(spark, lookup, ["22423"], "22697")["rule_type"] == "complementary"
    # multi-item antecedent: the closest item decides (85099B is the same family)
    assert _rule(spark, lookup, ["22423", "85099B"], "85099C")["rule_type"] == "variant"


@pytest.fixture
def output(spark, baskets, lookup):
    fit = fit_rules(baskets, 0.2, 0.3)
    classified = classify_rules(rules_frame(spark, fit["rules"], 10, "all"), lookup)
    segments = spark.createDataFrame(
        [("22699", "Core Fast Movers")], "stock_code STRING, segment_name STRING"
    )
    return build_associations_output(
        classified,
        lookup,
        segments,
        as_of="2011-12-10",
        min_support=0.2,
        min_confidence=0.3,
        min_lift=2.0,
        run_id="run1",
    )


def test_build_associations_output(output):
    assert output.columns == ASSOCIATIONS_COLUMNS
    rows = output.collect()
    by = {(tuple(r["antecedent"]), r["consequent"]): r for r in rows}
    r = by[(("22697",), "22699")]
    assert r["antecedent_descriptions"] == "GREEN REGENCY TEACUP AND SAUCER"
    assert r["consequent_description"] == "ROSES REGENCY TEACUP AND SAUCER"
    assert r["consequent_segment"] == "Core Fast Movers"
    assert r["is_recommendable"] is True and r["mlflow_run_id"] == "run1"
    v = by[(("85099B",), "85099C")]
    assert v["rule_type"] == "variant" and v["is_recommendable"] is False
    multi = [r for r in rows if r["antecedent_size"] == 2]
    assert all(" + " in r["antecedent_descriptions"] for r in multi)
    assert all(r["lift"] >= 2.0 or not r["is_recommendable"] for r in rows)


def test_top_rules_one_per_antecedent(output):
    top = top_rules(output, "all", n=50).collect()
    keys = [tuple(r["antecedent"]) for r in top]
    assert len(keys) == len(set(keys))
    assert all(r["is_recommendable"] for r in top)
    lifts = [r["lift"] for r in top]
    assert lifts == sorted(lifts, reverse=True)


def test_associations_ddl():
    ddl = associations_ddl("retail.gold.product_associations")
    assert "CONSTRAINT product_associations_pk PRIMARY KEY (rule_id, as_of_date)" in ddl
    assert "antecedent ARRAY<STRING>" in ddl and "rule_id BIGINT NOT NULL" in ddl


def test_fit_rules_refuses_huge_results(baskets):
    with pytest.raises(ValueError, match="max_rules"):
        fit_rules(baskets, 0.2, 0.3, max_rules=1)
