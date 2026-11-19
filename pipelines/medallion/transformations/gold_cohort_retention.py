"""Lakeflow pipeline source: Gold customer cohort retention (DataFrame API).

Built with the DataFrame API because it needs a complete cohort x month grid (explode a
month sequence, then left join), which is clearer and unit-tested as a Python function in
`src/transformations/gold.py`. The other Gold aggregates are SQL files next to this one.
"""

import sys

from pyspark import pipelines as dp

_SRC = spark.conf.get("retail.src_path")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from common.tables import GOLD_CUSTOMER_COHORT_RETENTION, GOLD_DIM_CUSTOMER, GOLD_FACT_SALES
from transformations.gold import build_customer_cohort_retention


@dp.materialized_view(
    name=GOLD_CUSTOMER_COHORT_RETENTION,
    comment="Cohort (first-purchase month) x months since first purchase: active customers, "
    "retention rate and revenue. Complete grid (0 where nobody returned).",
)
def customer_cohort_retention():
    return build_customer_cohort_retention(
        spark.read.table(GOLD_FACT_SALES), spark.read.table(GOLD_DIM_CUSTOMER)
    )
