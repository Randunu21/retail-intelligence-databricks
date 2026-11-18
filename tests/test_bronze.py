"""Tests for ingestion.bronze: rows/values unchanged, lineage columns added."""

from datetime import datetime

import pytest

from ingestion.bronze import AUTOLOADER_OPTIONS, add_ingestion_metadata
from ingestion.raw_reader import RAW_COLUMNS, read_raw_csv

HEADER = "Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country\n"
LINE_1 = "489434,85048,15CM CHRISTMAS GLASS BALL 20 LIGHTS,12,2009-12-01 07:45:00,6.95,13085,United Kingdom\n"
CANCEL = "C489449,22087,PAPER BUNTING WHITE LACE,-12,2009-12-01 10:33:00,2.95,16321,Australia\n"
POSTAGE = "536365,POST,POSTAGE,1,2010-12-01 08:26:00,18,,France\n"

# LINE_1 is in both files: the real sheets overlap, and Bronze must NOT de-duplicate.
FILE_A = HEADER + LINE_1 + CANCEL
FILE_B = HEADER + LINE_1 + POSTAGE

RUN_TS = datetime(2026, 11, 18, 12, 0, 0)


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "online_retail_II_2009_2010.csv").write_text(FILE_A)
    (tmp_path / "online_retail_II_2010_2011.csv").write_text(FILE_B)
    return tmp_path


@pytest.fixture
def bronze_df(spark, raw_dir):
    return add_ingestion_metadata(read_raw_csv(spark, str(raw_dir)), ingested_at=RUN_TS)


@pytest.mark.spark
def test_columns_and_types(bronze_df):
    assert bronze_df.columns == [*RAW_COLUMNS, "_source_file", "_ingested_at"]
    types = dict(bronze_df.dtypes)
    assert {types[c] for c in RAW_COLUMNS} == {"string"}
    assert types["_ingested_at"] == "timestamp"


@pytest.mark.spark
def test_keeps_every_row_including_duplicates_and_junk(bronze_df):
    rows = bronze_df.collect()
    assert len(rows) == 4  # nothing filtered, nothing de-duplicated
    assert sum(r["Invoice"] == "489434" for r in rows) == 2
    assert any(r["StockCode"] == "POST" for r in rows)
    assert any(r["Invoice"].startswith("C") for r in rows)


@pytest.mark.spark
def test_lineage_columns(bronze_df):
    rows = bronze_df.collect()
    assert {r["_source_file"] for r in rows} == {
        "online_retail_II_2009_2010.csv",
        "online_retail_II_2010_2011.csv",
    }
    assert {r["_ingested_at"] for r in rows} == {RUN_TS}


@pytest.mark.spark
def test_default_timestamp_is_single_value_in_batch(spark, raw_dir):
    df = add_ingestion_metadata(read_raw_csv(spark, str(raw_dir)))
    assert df.select("_ingested_at").distinct().count() == 1


def test_autoloader_uses_same_csv_options_as_batch_reader():
    assert AUTOLOADER_OPTIONS["cloudFiles.format"] == "csv"
    assert AUTOLOADER_OPTIONS["escape"] == '"'
    assert AUTOLOADER_OPTIONS["header"] == "true"
