"""Lakeflow pipeline source: Gold dimensions built with the DataFrame API.

dim_product and dim_customer need ranked-window logic (which description / which country
wins), so their logic lives in `src/transformations/gold.py` with unit tests. The SQL
files next to this one build fact_sales and dim_date.
"""

import sys

from pyspark import pipelines as dp

_SRC = spark.conf.get("retail.src_path")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from common.tables import GOLD_DIM_CUSTOMER, GOLD_DIM_PRODUCT, SILVER_SALES
from transformations.gold import GOLD_TABLE_PROPERTIES, build_dim_customer, build_dim_product


@dp.materialized_view(
    name=GOLD_DIM_PRODUCT,
    comment="One row per stock_code. description = most frequent variant (ties: most recent).",
)
def dim_product():
    return build_dim_product(spark.read.table(SILVER_SALES))


@dp.materialized_view(
    name=GOLD_DIM_CUSTOMER,
    comment="One row per customer + 'Unknown' row (customer_key = -1). country = most frequent.",
    table_properties=GOLD_TABLE_PROPERTIES,
)
def dim_customer():
    return build_dim_customer(spark.read.table(SILVER_SALES))
