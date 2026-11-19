"""Bronze layer helpers: raw rows unchanged, plus ingestion metadata.

Bronze rules:
- Keep every row and every value exactly as delivered (all strings, original column names).
  No filtering, casting or de-duplication, so Silver can always be rebuilt from Bronze.
- Add lineage columns, prefixed with `_` so they never clash with source columns:
  `_source_file` (which file/sheet the row came from) and `_ingested_at` (when it was loaded).

The pipeline file `pipelines/medallion/transformations/bronze.py` reads the files with Auto
Loader and calls `add_ingestion_metadata`. Tests call the same function on a batch read.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ingestion.raw_reader import CSV_OPTIONS, with_source_file

# The source column "Customer ID" contains a space. Delta only allows spaces (and ,;{}()=)
# in column names when column mapping is enabled, so Bronze turns it on to keep the
# original names. Silver renames everything to snake_case and doesn't need it.
BRONZE_TABLE_PROPERTIES: dict[str, str] = {"delta.columnMapping.mode": "name"}

# Auto Loader (`cloudFiles`) = the same CSV parsing options as the batch reader.
AUTOLOADER_OPTIONS: dict[str, str] = {"cloudFiles.format": "csv", **CSV_OPTIONS}


def add_ingestion_metadata(df: DataFrame, ingested_at: datetime | None = None) -> DataFrame:
    """Add `_source_file` and `_ingested_at` to a DataFrame read straight from files.

    - `ingested_at` given: every row gets that value (used by tests for exact assertions).
    - Omitted: `current_timestamp()`. In a batch query it is one value for the whole query;
      in a streaming query it is evaluated per micro-batch, i.e. when that file was loaded.
    """
    ts = F.lit(ingested_at).cast("timestamp") if ingested_at else F.current_timestamp()
    return with_source_file(df).withColumn("_ingested_at", ts)
