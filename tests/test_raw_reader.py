"""Tests for ingestion.raw_reader on a tiny CSV that mimics the converted Online Retail II files."""

import pytest

from ingestion.raw_reader import RAW_COLUMNS, read_raw_csv, with_source_file

SAMPLE_CSV = (
    "Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country\n"
    "489434,85048,15CM CHRISTMAS GLASS BALL 20 LIGHTS,12,2009-12-01 07:45:00,6.95,13085,"
    "United Kingdom\n"
    'C489449,85123A,"WHITE HANGING HEART, ""T-LIGHT"" HOLDER",-12,2009-12-01 10:33:00,'
    "2.55,,Australia\n"
)


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "online_retail_II_2009_2010.csv").write_text(SAMPLE_CSV)
    return tmp_path


@pytest.mark.spark
def test_schema_is_all_strings_with_original_names(spark, raw_dir):
    df = read_raw_csv(spark, str(raw_dir / "*.csv"))
    assert df.columns == RAW_COLUMNS
    assert {t for _, t in df.dtypes} == {"string"}


@pytest.mark.spark
def test_values_kept_as_is(spark, raw_dir):
    rows = read_raw_csv(spark, str(raw_dir)).collect()
    assert len(rows) == 2
    cancel = rows[1]
    assert cancel["Invoice"] == "C489449"  # not lost to integer inference
    assert cancel["StockCode"] == "85123A"
    assert cancel["Description"] == 'WHITE HANGING HEART, "T-LIGHT" HOLDER'  # comma + quotes
    assert cancel["Customer ID"] is None  # empty field -> NULL


@pytest.mark.spark
def test_with_source_file(spark, raw_dir):
    df = with_source_file(read_raw_csv(spark, str(raw_dir)))
    files = {r["_source_file"] for r in df.select("_source_file").collect()}
    assert files == {"online_retail_II_2009_2010.csv"}
