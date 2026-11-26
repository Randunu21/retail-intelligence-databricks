"""Tests for segmentation.product_insights (Phase 4.6): plain Python, no Spark."""

from __future__ import annotations

from segmentation.product_insights import (
    dedupe_pairs,
    recommendation,
    render_product_insights_md,
)


def _r(a, c, conf=0.5, lift=10.0, sup=0.01, n=400, desc=None):
    return {
        "antecedent": a,
        "consequent": c,
        "antecedent_descriptions": desc or " + ".join(a),
        "consequent_description": c,
        "confidence": conf,
        "lift": lift,
        "support": sup,
        "baskets_with_rule": n,
    }


def test_dedupe_pairs_drops_mirror_rules_and_repeated_antecedents():
    rules = [
        _r(["A"], "B"),
        _r(["B"], "A"),  # mirror of A -> B
        _r(["A"], "C"),  # same antecedent as the first
        _r(["C"], "D"),
        _r(["D", "E"], "F"),
    ]
    got = dedupe_pairs(rules, 10)
    assert [(tuple(r["antecedent"]), r["consequent"]) for r in got] == [
        (("A",), "B"),
        (("C",), "D"),
        (("D", "E"), "F"),
    ]
    assert len(dedupe_pairs(rules, 2)) == 2


def test_recommendation_sentence():
    s = recommendation(_r(["A"], "B", conf=0.734, lift=51.45, n=409, desc="CUP"))
    assert "**CUP** → suggest **B**" in s
    assert "73.4%" in s and "51.5×" in s and "409 orders" in s


def test_render_contains_real_numbers_and_sections():
    seg = [
        {
            "segment_name": "Core Fast Movers",
            "products": 2034,
            "product_share": 0.43,
            "revenue_share": 0.869,
            "median_units_per_week": 33.8,
            "median_active_week_share": 0.9057,
            "median_price": 1.65,
            "median_days_since_last_sale": 2,
            "median_q4_share": 0.34,
        },
        {
            "segment_name": "Long Tail",
            "products": 959,
            "product_share": 0.203,
            "revenue_share": 0.008,
            "median_units_per_week": 0.27,
            "median_active_week_share": 0.07,
            "median_price": 2.9,
            "median_days_since_last_sale": 410,
            "median_q4_share": 0.35,
        },
    ]
    groups = [
        {
            "segment_name": "Core Fast Movers",
            "action_group": "Core: Year-round Staples",
            "products": 1519,
            "revenue_share": 0.697,
        }
    ]
    playbook = {"Core: Year-round Staples": {"action": "Never out of stock", "kpi": "In-stock"}}
    counts = [
        {"scope": "all", "rule_type": "similar", "rules": 458, "recommendable": 458, "median_lift": 10.0, "median_confidence": 0.53},
        {"scope": "all", "rule_type": "complementary", "rules": 319, "recommendable": 319, "median_lift": 8.3, "median_confidence": 0.41},
        {"scope": "all", "rule_type": "variant", "rules": 100, "recommendable": 0, "median_lift": 19.0, "median_confidence": 0.58},
    ]  # fmt: skip
    md = render_product_insights_md(
        as_of="2011-12-10",
        n_products=4725,
        segment_rows=seg,
        group_rows=groups,
        playbook=playbook,
        segment_descriptions={"Core Fast Movers": "sells every week"},
        model_facts={"Algorithm": "bisecting k-means"},
        mba_facts={"Thresholds": "0.01 / 0.3"},
        rule_counts=counts,
        market_overlap={"in_both": 228, "rest_only": 1870},
        top_complementary=[_r(["X"], "Y")],
        top_range=[],
        top_basket=[_r(["P", "Q"], "R", desc="P + Q")],
        top_rest_only=[_r(["K"], "L")],
        frequent_products=707,
        caveats=["gross revenue"],
        images={"Centres": "img/phase4/centroid_heatmap.png"},
    )
    assert md.startswith("# Product segments and cross-sell insights")
    assert "Core Fast Movers are 43.0% of products but 86.9% of revenue" in md
    assert "Of 777 recommendable rules" in md and "458 link items of the same range" in md
    assert "variants (100 rules)" in md
    assert "Only 707 products (15.0%)" in md
    assert "228 rules hold in both" in md and "1,870 appear only abroad" in md
    assert (
        "| Core Fast Movers | Core: Year-round Staples | 1,519 | 69.7% | Never out of stock | In-stock |"
        in md
    )
    assert "_No rules in this group._" in md  # empty top_range
    assert "Basket has **P + Q** → suggest **R**" in md
    assert "![Centres](img/phase4/centroid_heatmap.png)" in md
    assert "- gross revenue" in md
