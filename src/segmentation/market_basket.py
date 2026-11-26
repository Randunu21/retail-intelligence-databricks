"""Market basket analysis with Spark ML FP-Growth (Phase 4.4-4.5).

**Baskets** = the set of distinct products on one invoice, from retail.silver.sales (already
non-cancelled, product codes only: Phase 1 removed cancellations to silver.returns and
non-product codes like POST, M, BANK CHARGES). Invoices without a customer id are kept: an
anonymous basket is still a real basket. Single-item invoices are kept too: they count in the
denominator of support (they are transactions in which no pair was bought).

**The three measures** (for a rule A -> B over N baskets):
- support(A -> B) = baskets with A and B / N: how common the combination is (reach);
- confidence(A -> B) = baskets with A and B / baskets with A = P(B | A): if a customer puts
  A in the basket, how often do they also take B (the cross-sell hit rate);
- lift(A -> B) = confidence / support(B) = P(B | A) / P(B): how much more likely B is with
  A than in general. 1 = independent, > 1 = bought together more than chance. Lift is
  symmetric; confidence is not (cakestand -> plates can be strong while plates -> cakestand
  is weak).

**FP-Growth vs Apriori.** Apriori generates candidate itemsets level by level and rescans
the data for each level; FP-Growth compresses the baskets into a prefix tree (FP-tree) once
and mines frequent itemsets from it without candidate generation. Same itemsets at the same
minimum support; FP-Growth is faster on many, long baskets (here ~25 items on average).
Spark's FPGrowth is the parallel version (PFP: items partitioned across workers).

**Free Edition:** the docs list `FPGrowthModel` as unsupported on serverless, but the 4.1
probe showed fit / freqItemsets / associationRules / transform all work (Spark 4.2.0,
environment 6). The model is never saved; results are extracted and the model deleted
(serverless model cache), as in Phase 3.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import date

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

ITEMS_COL = "items"


def build_baskets(
    sales: DataFrame, as_of: date | str | None = None, country_col: str = "country"
) -> DataFrame:
    """One row per invoice: invoice, items (sorted distinct stock codes), n_items,
    country, has_customer. Optionally only invoices dated strictly before `as_of`."""
    lines = sales
    if as_of is not None:
        lines = lines.where(F.col("invoice_date") < F.to_date(F.lit(str(as_of))))
    return (
        lines.groupBy("invoice")
        .agg(
            F.array_sort(F.collect_set("stock_code")).alias(ITEMS_COL),
            F.first(country_col, ignorenulls=True).alias("country"),
            F.max(F.col("customer_id").isNotNull()).alias("has_customer"),
        )
        .withColumn("n_items", F.size(ITEMS_COL))
    )


def basket_size_summary(baskets: DataFrame) -> dict:
    """Number of baskets, size quantiles, single-item share, share with a known customer,
    and how many item pairs the baskets contain (sum of n(n-1)/2: the work FP-Growth faces)."""
    n = F.col("n_items")
    row = baskets.agg(
        F.count(F.lit(1)).alias("baskets"),
        F.mean(n).alias("mean_items"),
        F.percentile_approx(n, [0.25, 0.5, 0.75, 0.9, 0.99], 10000).alias("q"),
        F.max(n).alias("max_items"),
        F.avg((n == 1).cast("double")).alias("share_single_item"),
        F.avg(F.col("has_customer").cast("double")).alias("share_known_customer"),
        F.sum((n * (n - 1) / 2).cast("double")).alias("item_pairs"),
    ).first()
    q = row["q"]
    return {
        "baskets": int(row["baskets"]),
        "mean_items": float(row["mean_items"]),
        "p25_items": int(q[0]),
        "median_items": int(q[1]),
        "p75_items": int(q[2]),
        "p90_items": int(q[3]),
        "p99_items": int(q[4]),
        "max_items": int(row["max_items"]),
        "share_single_item": float(row["share_single_item"]),
        "share_known_customer": float(row["share_known_customer"]),
        "item_pairs": float(row["item_pairs"]),
    }


def item_support(baskets: DataFrame) -> DataFrame:
    """stock_code, baskets containing it, support (= share of all baskets)."""
    n = baskets.count()
    return (
        baskets.select(F.explode(ITEMS_COL).alias("stock_code"))
        .groupBy("stock_code")
        .agg(F.count(F.lit(1)).alias("basket_count"))
        .withColumn("support", F.col("basket_count") / F.lit(n))
    )


def items_above_support(support: DataFrame, thresholds: Sequence[float]) -> dict[float, int]:
    """How many products reach each minimum support (the Apriori property: only these can
    appear in any frequent itemset, so this bounds the search)."""
    row = support.agg(
        *[
            F.sum((F.col("support") >= F.lit(t)).cast("int")).alias(f"t{i}")
            for i, t in enumerate(thresholds)
        ]
    ).first()
    return {t: int(row[f"t{i}"] or 0) for i, t in enumerate(thresholds)}


def variant_family(stock_code: Column) -> Column:
    """Product family = stock code without its trailing letters: 85099B / 85099C / 85099F
    (jumbo bag in different designs) -> 85099; 79323W / 79323P (cherry lights white / pink)
    -> 79323. Rules inside one family are colour/design variants of the same item.
    Letters are only stripped after a digit, so all-letter codes (PADS) stay themselves."""
    return F.regexp_replace(stock_code, r"(?<=[0-9])[A-Z]+$", "")


def _same_family_count(itemsets: DataFrame) -> int:
    """Frequent PAIRS whose two items belong to the same variant family."""
    pairs = itemsets.where(F.size(ITEMS_COL) == 2)
    a = variant_family(F.element_at(ITEMS_COL, 1))
    b = variant_family(F.element_at(ITEMS_COL, 2))
    return int(pairs.where(a == b).count())


def fpgrowth_summary(
    baskets: DataFrame,
    min_support: float,
    min_confidence: float,
    num_partitions: int | None = None,
) -> dict:
    """Fit FP-Growth once and return plain counts (never raises; the model is deleted).

    Used by the threshold sweep: how many items/pairs/triples are frequent, how many rules,
    how many with lift > 1, how many pairs are colour variants, and how long it took.
    """
    from pyspark.ml.fpm import FPGrowth

    out: dict = {"min_support": min_support, "min_confidence": min_confidence, "error": None}
    model = None
    t0 = time.perf_counter()
    try:
        fp = FPGrowth(itemsCol=ITEMS_COL, minSupport=min_support, minConfidence=min_confidence)
        if num_partitions:
            fp = fp.setNumPartitions(num_partitions)
        model = fp.fit(baskets.select(ITEMS_COL))
        itemsets = model.freqItemsets
        by_size = {
            int(r["size"]): int(r["n"])
            for r in itemsets.groupBy(F.size(ITEMS_COL).alias("size"))
            .agg(F.count(F.lit(1)).alias("n"))
            .collect()
        }
        rules = model.associationRules
        r = rules.agg(
            F.count(F.lit(1)).alias("rules"),
            F.sum((F.col("lift") > 1).cast("int")).alias("rules_lift_gt_1"),
            F.max("lift").alias("max_lift"),
            F.percentile_approx("lift", 0.5, 10000).alias("median_lift"),
            F.percentile_approx("confidence", 0.5, 10000).alias("median_confidence"),
        ).first()
        out.update(
            {
                "itemsets_by_size": by_size,
                "frequent_items": by_size.get(1, 0),
                "frequent_pairs": by_size.get(2, 0),
                "frequent_triples": by_size.get(3, 0),
                "max_itemset_size": max(by_size) if by_size else 0,
                "rules": int(r["rules"] or 0),
                "rules_lift_gt_1": int(r["rules_lift_gt_1"] or 0),
                "max_lift": r["max_lift"],
                "median_lift": r["median_lift"],
                "median_confidence": r["median_confidence"],
                "same_family_pairs": _same_family_count(itemsets),
            }
        )
    except Exception as exc:  # recorded, the sweep continues
        out["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    finally:
        out["seconds"] = round(time.perf_counter() - t0, 2)
        if model is not None:
            del model
    return out


def threshold_sweep(
    baskets: DataFrame,
    supports: Sequence[float],
    min_confidence: float,
    max_seconds: float = 300.0,
    max_itemsets: int = 200_000,
    on_result=None,
) -> list[dict]:
    """fpgrowth_summary for each support, from HIGH to LOW. Stops early (and says why) once
    a fit takes longer than `max_seconds` or finds more than `max_itemsets` itemsets: lower
    supports only get bigger, and Free Edition has daily compute quotas."""
    results = []
    for s in sorted(supports, reverse=True):
        r = fpgrowth_summary(baskets, s, min_confidence)
        results.append(r)
        if on_result is not None:
            on_result(r)
        n_itemsets = sum((r.get("itemsets_by_size") or {}).values())
        if r["error"] or r["seconds"] > max_seconds or n_itemsets > max_itemsets:
            r["stopped_sweep"] = (
                "error" if r["error"] else f"{r['seconds']} s, {n_itemsets:,} itemsets"
            )
            break
    return results


# =============================================================================== rules (4.5)

# Words that only name a colour (or finish): two products whose descriptions differ only by
# these are colour variants (e.g. 85123A WHITE / 21733 RED HANGING HEART T-LIGHT HOLDER).
COLOUR_WORDS: tuple[str, ...] = (
    "RED",
    "PINK",
    "BLUE",
    "WHITE",
    "BLACK",
    "GREEN",
    "IVORY",
    "CREAM",
    "YELLOW",
    "PURPLE",
    "ORANGE",
    "GREY",
    "GRAY",
    "BROWN",
    "GOLD",
    "SILVER",
    "NATURAL",
    "LILAC",
    "TURQUOISE",
    "AQUA",
    "ROSE",
    "MULTICOLOUR",
    "COLOURED",
    "ASSORTED",
    "DARK",
    "LIGHT",
    "PASTEL",
)
# Filler words that don't identify a product type.
STOP_WORDS: tuple[str, ...] = (
    "SET", "OF", "PACK", "THE", "AND", "WITH", "IN", "A", "DESIGN", "SMALL", "LARGE", "MEDIUM",
    "MINI", "BIG", "TO", "FOR",
)  # fmt: skip
SIMILARITY_THRESHOLD = 0.5  # token Jaccard >= 0.5 = the same kind of item in another design


def description_tokens(description: Column) -> Column:
    """Distinct, sorted words of a description without colours, filler words, numbers and
    punctuation: 'LUNCH BAG RED RETROSPOT' -> [BAG, LUNCH, RETROSPOT]."""
    words = F.split(F.trim(F.regexp_replace(F.upper(description), r"[^A-Z ]", " ")), r"\s+")
    drop = F.array(*[F.lit(w) for w in (*COLOUR_WORDS, *STOP_WORDS)])
    kept = F.filter(words, lambda w: (F.length(w) >= 2) & ~F.array_contains(drop, w))
    return F.array_sort(F.array_distinct(kept))


def product_lookup(descriptions: DataFrame) -> DataFrame:
    """stock_code -> description, variant family, description tokens."""
    return descriptions.select(
        "stock_code",
        "description",
        variant_family(F.col("stock_code")).alias("family"),
        description_tokens(F.col("description")).alias("tokens"),
    )


def min_count(min_support: float, n_baskets: int) -> int:
    """Minimum number of baskets for an itemset, as Spark's FPGrowth computes it."""
    import math

    return int(math.ceil(min_support * n_baskets))


def fit_rules(
    baskets: DataFrame, min_support: float, min_confidence: float, max_rules: int = 200_000
) -> dict:
    """Fit FP-Growth and return the rules as plain rows (the fitted model is deleted).

    Rules are collected to the driver (a few thousand rows at most at these thresholds) and
    rebuilt with `rules_frame`, so nothing downstream references the server-side model.
    """
    from pyspark.ml.fpm import FPGrowth

    n = baskets.count()
    t0 = time.perf_counter()
    model = FPGrowth(itemsCol=ITEMS_COL, minSupport=min_support, minConfidence=min_confidence).fit(
        baskets.select(ITEMS_COL)
    )
    try:
        n_itemsets = model.freqItemsets.count()
        n_rules = model.associationRules.count()
        if n_rules > max_rules:  # don't pull a huge result to the driver
            raise ValueError(
                f"{n_rules:,} rules > max_rules={max_rules:,}: raise min_support/min_confidence"
            )
        rules = [r.asDict() for r in model.associationRules.collect()]
    finally:
        del model
    return {
        "n_baskets": n,
        "n_itemsets": int(n_itemsets),
        "rules": rules,
        "seconds": round(time.perf_counter() - t0, 2),
    }


RULE_SCHEMA = (
    "antecedent ARRAY<STRING>, consequent ARRAY<STRING>, confidence DOUBLE, lift DOUBLE, "
    "support DOUBLE"
)


def rules_frame(spark, rules: Sequence[dict], n_baskets: int, scope: str) -> DataFrame:
    """Collected FP-Growth rules -> DataFrame with scope, sizes, basket counts and a rule id."""
    rows = [
        (list(r["antecedent"]), list(r["consequent"]), r["confidence"], r["lift"], r["support"])
        for r in rules
    ]
    df = spark.createDataFrame(rows, RULE_SCHEMA)
    return (
        df.withColumn("scope", F.lit(scope))
        .withColumn("n_baskets", F.lit(n_baskets))
        .withColumn("antecedent_size", F.size("antecedent"))
        .withColumn("consequent_item", F.element_at("consequent", 1))
        .withColumn("baskets_with_rule", F.round(F.col("support") * F.lit(n_baskets)).cast("int"))
        .withColumn(
            "rule_id",
            F.xxhash64(
                F.col("scope"),
                F.array_join("antecedent", ","),
                F.array_join("consequent", ","),
            ),
        )
    )


def classify_rules(
    rules: DataFrame, lookup: DataFrame, similarity_threshold: float = SIMILARITY_THRESHOLD
) -> DataFrame:
    """Add rule_type and description_similarity:

    - `variant`: the consequent is a colour/design variant of an antecedent item (same stock
      code family, or the same description once colour words are removed). Trivial: "people
      who buy the red one also buy the white one" is not a cross-sell.
    - `similar`: same kind of item in another design (token Jaccard >= threshold, e.g. LUNCH
      BAG RETROSPOT -> LUNCH BAG SUKI). A "more designs" recommendation, kept but labelled.
    - `complementary`: different products bought together (REGENCY CAKESTAND -> REGENCY TEA
      PLATE): the classic cross-sell.
    For multi-item antecedents the closest antecedent item decides.
    """
    lk = lookup.select("stock_code", "family", "tokens")
    pairs = (
        rules.select("rule_id", F.explode("antecedent").alias("a_item"), "consequent_item")
        .join(
            lk.select(
                F.col("stock_code").alias("a_item"),
                F.col("family").alias("a_family"),
                F.col("tokens").alias("a_tokens"),
            ),
            "a_item",
            "left",
        )
        .join(
            lk.select(
                F.col("stock_code").alias("consequent_item"),
                F.col("family").alias("c_family"),
                F.col("tokens").alias("c_tokens"),
            ),
            "consequent_item",
            "left",
        )
    )
    inter = F.size(F.array_intersect("a_tokens", "c_tokens"))
    union = F.size(F.array_union("a_tokens", "c_tokens"))
    jaccard = F.when(union > 0, inter / union).otherwise(F.lit(0.0))
    is_variant = (F.col("a_family") == F.col("c_family")) | (
        (F.size("a_tokens") > 0) & (F.col("a_tokens") == F.col("c_tokens"))
    )
    per_rule = pairs.groupBy("rule_id").agg(
        F.max(F.coalesce(is_variant, F.lit(False))).alias("_variant"),
        F.max(F.coalesce(jaccard, F.lit(0.0))).alias("description_similarity"),
    )
    rule_type = (
        F.when(F.col("_variant"), F.lit("variant"))
        .when(F.col("description_similarity") >= F.lit(similarity_threshold), F.lit("similar"))
        .otherwise(F.lit("complementary"))
    )
    return (
        rules.join(per_rule, "rule_id", "left").withColumn("rule_type", rule_type).drop("_variant")
    )


def pair_rules_sql(baskets: DataFrame, min_basket_count: int) -> DataFrame:
    """Independent check: single-item rules a -> b computed with plain joins (no FP-Growth).

    Only items in >= min_basket_count baskets can be in a frequent pair (Apriori property),
    so baskets are reduced to those items before the self-join.
    """
    n = baskets.count()
    exploded = baskets.select("invoice", F.explode(ITEMS_COL).alias("item"))
    counts = exploded.groupBy("item").agg(F.count(F.lit(1)).alias("n_item"))
    frequent = counts.where(F.col("n_item") >= min_basket_count)
    kept = exploded.join(frequent.select("item"), "item")
    a = kept.select("invoice", F.col("item").alias("a"))
    b = kept.select("invoice", F.col("item").alias("b"))
    pair = (
        a.join(b, "invoice")
        .where(F.col("a") < F.col("b"))
        .groupBy("a", "b")
        .agg(F.count(F.lit(1)).alias("n_ab"))
        .where(F.col("n_ab") >= min_basket_count)
    )
    both_ways = pair.select("a", "b", "n_ab").unionByName(
        pair.select(F.col("b").alias("a"), F.col("a").alias("b"), "n_ab")
    )
    ca = frequent.select(F.col("item").alias("a"), F.col("n_item").alias("n_a"))
    cb = frequent.select(F.col("item").alias("b"), F.col("n_item").alias("n_b"))
    return (
        both_ways.join(ca, "a")
        .join(cb, "b")
        .select(
            F.col("a").alias("antecedent_item"),
            F.col("b").alias("consequent_item"),
            (F.col("n_ab") / F.lit(n)).alias("support"),
            (F.col("n_ab") / F.col("n_a")).alias("confidence"),
            (F.col("n_ab") * F.lit(n) / (F.col("n_a") * F.col("n_b"))).alias("lift"),
        )
    )


def compare_pair_rules(fp_rules: DataFrame, sql_rules: DataFrame, min_confidence: float) -> dict:
    """FP-Growth single-item rules vs the SQL recount (same min confidence): both sets must be
    identical and the measures equal up to floating point."""
    fp = fp_rules.where(F.col("antecedent_size") == 1).select(
        F.element_at("antecedent", 1).alias("antecedent_item"),
        "consequent_item",
        F.col("support").alias("fp_support"),
        F.col("confidence").alias("fp_confidence"),
        F.col("lift").alias("fp_lift"),
    )
    sq = sql_rules.where(F.col("confidence") >= F.lit(min_confidence))
    joined = fp.join(sq, ["antecedent_item", "consequent_item"], "full_outer")
    row = joined.agg(
        F.sum(F.col("fp_support").isNotNull().cast("int")).alias("fp_pair_rules"),
        F.sum(F.col("support").isNotNull().cast("int")).alias("sql_pair_rules"),
        F.sum((F.col("fp_support").isNull() | F.col("support").isNull()).cast("int")).alias(
            "unmatched"
        ),
        F.max(F.abs(F.col("fp_support") - F.col("support"))).alias("max_support_diff"),
        F.max(F.abs(F.col("fp_confidence") - F.col("confidence"))).alias("max_confidence_diff"),
        F.max(F.abs(F.col("fp_lift") - F.col("lift"))).alias("max_lift_diff"),
    ).first()
    return {k: row[k] for k in row.asDict()}
