"""Lakeflow pipeline source: Bronze layer.

Only pipeline wiring lives here (decorators, Auto Loader). The logic is in `src/` and
is unit-tested with pytest, because the `pyspark.pipelines` module only exists inside a
running pipeline.
"""

import sys

from pyspark import pipelines as dp

# Pipeline setting `retail.src_path` = <Git folder>/src, so this file imports the same
# tested modules as pytest and the notebooks.
_SRC = spark.conf.get("retail.src_path")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from common.tables import BRONZE_TRANSACTIONS, RAW_CSV_DIR
from ingestion.bronze import AUTOLOADER_OPTIONS, BRONZE_TABLE_PROPERTIES, add_ingestion_metadata
from ingestion.raw_reader import RAW_SCHEMA


@dp.table(
    name=BRONZE_TRANSACTIONS,
    comment="Raw Online Retail II rows exactly as delivered (all strings) + _source_file, _ingested_at.",
    table_properties=BRONZE_TABLE_PROPERTIES,
)
def online_retail_transactions():
    # Auto Loader: a streaming read that remembers which files it has already loaded
    # (checkpoint managed by the pipeline). Re-running the pipeline only picks up NEW files.
    raw = (
        spark.readStream.format("cloudFiles")
        .options(**AUTOLOADER_OPTIONS)
        .schema(RAW_SCHEMA)
        .load(RAW_CSV_DIR)
    )
    return add_ingestion_metadata(raw)
