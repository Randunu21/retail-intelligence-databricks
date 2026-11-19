"""Tests for optimisation.layout: the variants differ only in the fact table's layout."""

from optimisation.benchmark import POST_SUFFIX
from optimisation.builds import bench_names
from optimisation.layout import (
    CLUSTER_KEYS,
    FACT_STEP,
    LAYOUT_QUERIES,
    layout_variants,
    optimize,
    optimize_zorder,
)


def test_variants_share_silver_and_have_own_fact_table():
    base = bench_names("x10_baseline", bronze_table="retail.bench.bronze_x10")
    variants = layout_variants(base, "x10")
    assert [v.experiment for v in variants] == [
        "baseline", "part_date", "zorder_32mb", "lc_auto", "lc_32mb"]  # fmt: skip
    facts = {v.names[FACT_STEP] for v in variants}
    assert len(facts) == len(variants)  # one fact table per variant
    assert base[FACT_STEP] not in facts  # never overwrites the recorded baseline
    for v in variants:
        assert v.names["silver_sales"] == base["silver_sales"]
    assert "retail.bench.x10_lc_32mb__gold_fact_sales" in facts


def test_variant_layouts():
    v = {x.experiment: x for x in layout_variants(bench_names("x1_baseline"), "x1")}
    assert v["baseline"].spec_for(FACT_STEP).describe() == "default layout"
    assert v["part_date"].spec_for(FACT_STEP).partition_by == ("date_key",)
    assert v["lc_32mb"].spec_for(FACT_STEP).cluster_by == CLUSTER_KEYS
    assert v["lc_32mb"].spec_for(FACT_STEP).properties["delta.targetFileSize"] == "32mb"
    assert "delta.targetFileSize" not in v["lc_auto"].spec_for(FACT_STEP).properties
    assert v["lc_auto"].post_write[FACT_STEP] is optimize
    assert v["zorder_32mb"].post_write[FACT_STEP] is optimize_zorder
    assert not v["baseline"].post_write and not v["part_date"].post_write
    assert POST_SUFFIX == "__post"


def test_queries_have_table_placeholder():
    for sql in LAYOUT_QUERIES.values():
        assert "{table}" in sql
        assert "retail." not in sql
