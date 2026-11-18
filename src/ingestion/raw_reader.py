"""Read the raw Online Retail II CSV files exactly as they were exported.

The raw files are produced by `scripts/convert_online_retail.py`: one CSV per source sheet,
original column headers, every value written as text, missing values as empty fields.

Why an explicit all-string schema instead of `inferSchema=True`:
- Inference needs an extra full pass over the data (slower) and guesses types, e.g. it would
  turn the `Invoice` column into an integer and then fail or null out "C489449" cancellations.
- Bronze should mirror the source. Casting to proper types is done explicitly in Silver.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

# Column names exactly as in the UCI workbook (note the space in "Customer ID").
RAW_COLUMNS: list[str] = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

RAW_SCHEMA = StructType([StructField(c, StringType(), nullable=True) for c in RAW_COLUMNS])

# pandas writes embedded quotes as "" (RFC 4180). Spark's default escape is a backslash,
# so we set escape='"' or descriptions like 'HEART, "T-LIGHT" HOLDER' would be mis-parsed.
CSV_OPTIONS: dict[str, str] = {
    "header": "true",
    "escape": '"',
    "mode": "FAILFAST",  # raise on malformed rows instead of silently nulling them
}


def read_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    """Read raw CSV file(s) at `path` (file, folder or glob) with the fixed string schema."""
    return spark.read.options(**CSV_OPTIONS).schema(RAW_SCHEMA).csv(path)


def with_source_file(df: DataFrame, col_name: str = "_source_file") -> DataFrame:
    """Add the name of the file each row came from (uses Spark's hidden `_metadata` column).

    Must be applied directly to a DataFrame returned by a file reader such as `read_raw_csv`.
    """
    return df.withColumn(col_name, F.col("_metadata.file_name"))
