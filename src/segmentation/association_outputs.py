"""Gold output of market basket analysis (Phase 4.5): `retail.gold.product_associations`.

One row per (rule_id, as_of_date): a rule `antecedent -> consequent` found by FP-Growth in a
`scope` of baskets (all / UK / rest of world), with support, confidence, lift, the number of
baskets behind it, readable descriptions, the rule type (variant / similar / complementary),
the consequent's product segment, and whether it is **recommendable** (not a variant and lift
>= the minimum). Variant rules are kept but flagged, so the filtering is visible, not hidden.
Each run replaces only its own snapshot (replaceWhere).
"""

from __future__ import annotations

from datetime import date

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

ASSOCIATIONS_SCHEMA: list[tuple[str, str, str]] = [
    ("rule_id", "BIGINT", "xxhash64(scope, antecedent, consequent) (primary key, with as_of_date)"),
    ("as_of_date", "DATE", "Snapshot: baskets dated strictly before this date"),
    ("scope", "STRING", "Baskets the rule was mined in: all / UK / rest"),
    ("antecedent", "ARRAY<STRING>", "Stock codes already in the basket (if ...)"),
    ("consequent", "STRING", "Stock code to recommend (... then)"),
    ("antecedent_descriptions", "STRING", "Descriptions of the antecedent items, ' + '-joined"),
    ("consequent_description", "STRING", "Description of the consequent"),
    ("antecedent_size", "INT", "Number of items in the antecedent"),
    ("support", "DOUBLE", "Share of baskets containing antecedent and consequent"),
    ("confidence", "DOUBLE", "P(consequent | antecedent)"),
    ("lift", "DOUBLE", "confidence / support(consequent); 1 = independent"),
    ("baskets_with_rule", "INT", "Baskets containing antecedent and consequent"),
    ("n_baskets", "INT", "Baskets in the scope"),
    ("rule_type", "STRING", "variant / similar / complementary"),
    ("description_similarity", "DOUBLE", "Max token Jaccard between consequent and antecedent"),
    ("is_recommendable", "BOOLEAN", "Not a variant and lift >= min_lift"),
    ("consequent_segment", "STRING", "Product segment of the consequent (gold.product_segments)"),
    ("min_support", "DOUBLE", "FP-Growth minSupport used"),
    ("min_confidence", "DOUBLE", "FP-Growth minConfidence used"),
    ("min_lift", "DOUBLE", "Minimum lift for is_recommendable"),
    ("mlflow_run_id", "STRING", "MLflow run of the rule mining"),
    ("created_at", "TIMESTAMP", "When the rules were written"),
]
ASSOCIATIONS_COLUMNS = [c for c, _, _ in ASSOCIATIONS_SCHEMA]
ASSOCIATIONS_COMMENT = (
    "Product association rules (Spark ML FP-Growth on invoice baskets; all / UK / rest) with "
    "support, confidence, lift, rule type and recommendability. Built by "
    "notebooks/04_product_segmentation_mba (Phase 4)."
)


def _sql_string(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def associations_ddl(table_name: str) -> str:
    lines = []
    for name, dtype, comment in ASSOCIATIONS_SCHEMA:
        nn = " NOT NULL" if name in ("rule_id", "as_of_date") else ""
        lines.append(f"  {name} {dtype}{nn} COMMENT {_sql_string(comment)}")
    short = table_name.split(".")[-1]
    lines.append(f"  CONSTRAINT {short}_pk PRIMARY KEY (rule_id, as_of_date)")
    body = ",\n".join(lines)
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n{body}\n)\nUSING DELTA\n"
        f"COMMENT {_sql_string(ASSOCIATIONS_COMMENT)}"
    )


def antecedent_descriptions(rules: DataFrame, lookup: DataFrame) -> DataFrame:
    """rule_id -> antecedent descriptions in antecedent order, ' + '-joined."""
    exploded = rules.select("rule_id", F.posexplode("antecedent").alias("pos", "stock_code"))
    named = exploded.join(lookup.select("stock_code", "description"), "stock_code", "left")
    return named.groupBy("rule_id").agg(
        F.array_join(
            F.transform(
                F.array_sort(
                    F.collect_list(
                        F.struct("pos", F.coalesce("description", "stock_code").alias("d"))
                    )
                ),
                lambda s: s["d"],
            ),
            " + ",
        ).alias("antecedent_descriptions")
    )


def build_associations_output(
    classified: DataFrame,
    lookup: DataFrame,
    segments: DataFrame,
    *,
    as_of: date | str,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    run_id: str,
) -> DataFrame:
    """classified = rules_frame(...) + classify_rules(...) (any number of scopes, unioned)."""
    types = {c: t for c, t, _ in ASSOCIATIONS_SCHEMA}
    out = (
        classified.join(antecedent_descriptions(classified, lookup), "rule_id", "left")
        .join(
            lookup.select(
                F.col("stock_code").alias("consequent_item"),
                F.col("description").alias("consequent_description"),
            ),
            "consequent_item",
            "left",
        )
        .join(
            segments.select(
                F.col("stock_code").alias("consequent_item"),
                F.col("segment_name").alias("consequent_segment"),
            ),
            "consequent_item",
            "left",
        )
        .withColumn("consequent", F.col("consequent_item"))
        .withColumn("as_of_date", F.to_date(F.lit(str(as_of))))
        .withColumn(
            "is_recommendable",
            (F.col("rule_type") != "variant") & (F.col("lift") >= F.lit(min_lift)),
        )
        .withColumn("min_support", F.lit(min_support))
        .withColumn("min_confidence", F.lit(min_confidence))
        .withColumn("min_lift", F.lit(min_lift))
        .withColumn("mlflow_run_id", F.lit(run_id))
        .withColumn("created_at", F.current_timestamp())
    )
    return out.select([F.col(c).cast(types[c]).alias(c) for c in ASSOCIATIONS_COLUMNS])


def write_associations(df: DataFrame, table_name: str, as_of: date | str) -> None:
    (
        df.select(*ASSOCIATIONS_COLUMNS)
        .write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"as_of_date = DATE'{as_of}'")
        .saveAsTable(table_name)
    )


def top_rules(
    associations: DataFrame,
    scope: str = "all",
    n: int = 25,
    order: str = "lift",
    one_per_antecedent: bool = True,
) -> DataFrame:
    """Recommendable rules of one scope, best first by `order` (lift or confidence).

    With `one_per_antecedent`, only the best consequent per antecedent is kept, so the list
    isn't 10 rules about the same item (a readable top list for the insights document).
    """
    from pyspark.sql import Window

    df = associations.where((F.col("scope") == scope) & F.col("is_recommendable"))
    if one_per_antecedent:
        w = Window.partitionBy(F.array_join("antecedent", ",")).orderBy(
            F.col(order).desc(), F.col("confidence").desc(), F.col("consequent")
        )
        df = df.withColumn("_r", F.row_number().over(w)).where(F.col("_r") == 1).drop("_r")
    return df.orderBy(F.col(order).desc(), F.col("support").desc(), F.col("rule_id")).limit(n)
