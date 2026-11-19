"""Tests for transformations.silver on small hand-made Bronze-shaped DataFrames.

Edge cases covered: cancellations, missing customer, zero/negative price, negative quantity,
non-product codes (incl. lowercase and patterns), real products with odd codes, duplicates
across source files, whitespace in codes/descriptions, unparseable values, line_revenue.
"""

from datetime import date, datetime
from decimal import Decimal

import pytest
from pyspark.sql import functions as F

from transformations.silver import (
    RETURNS_DROP_RULES,
    SALES_DROP_RULES,
    SILVER_OUTPUT_COLUMNS,
    apply_drop_rules,
    count_rule_failures,
    stage_silver,
    stock_code_profile,
    to_silver_output,
)

pytestmark = pytest.mark.spark

BRONZE_COLS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "_source_file",
]
F1, F2 = "online_retail_II_2009_2010.csv", "online_retail_II_2010_2011.csv"
TS = "2010-12-01 08:26:00"


def row(invoice, code, qty="1", price="2.50", cust="13085", desc="SOME PRODUCT", src=F1, ts=TS):
    return (invoice, code, desc, qty, ts, price, cust, "United Kingdom", src)


@pytest.fixture
def make_bronze(spark):
    """Build a Bronze-shaped DataFrame: all-string columns, like the real Bronze table."""
    schema = ", ".join(f"`{c}` STRING" for c in BRONZE_COLS)

    def _make(rows):
        df = spark.createDataFrame(rows, schema)
        return df.withColumn("_ingested_at", F.lit(datetime(2026, 11, 18, 20, 0)))

    return _make


def sales_of(df):
    return to_silver_output(apply_drop_rules(stage_silver(df), SALES_DROP_RULES))


def returns_of(df):
    staged = stage_silver(df).filter("is_cancellation")
    return to_silver_output(apply_drop_rules(staged, RETURNS_DROP_RULES))


# ---------------------------------------------------------------- standardise / types
def test_columns_types_and_cleaning(make_bronze):
    bronze = make_bronze(
        [row("489434", " 85048a ", qty="12", price="6.95", desc="  GLASS   BALL  LIGHTS ")]
    )
    out = sales_of(bronze)
    assert out.columns == SILVER_OUTPUT_COLUMNS
    types = dict(out.dtypes)
    assert types["quantity"] == "int"
    assert types["price"] == "decimal(10,3)"
    assert types["line_revenue"] == "decimal(18,3)"
    assert types["invoice_ts"] == "timestamp_ntz"
    assert types["invoice_date"] == "date"
    assert types["customer_id"] == "int"

    r = out.collect()[0]
    assert r["stock_code"] == "85048A"  # trimmed + upper-cased
    assert r["description"] == "GLASS BALL LIGHTS"  # trimmed + whitespace collapsed
    assert r["invoice_ts"] == datetime(2010, 12, 1, 8, 26)
    assert r["invoice_date"] == date(2010, 12, 1)
    assert r["customer_id"] == 13085
    assert r["line_revenue"] == Decimal("83.400")  # 12 * 6.95, exact decimal


def test_empty_description_becomes_null(make_bronze):
    r = sales_of(make_bronze([row("1", "85048", desc="   ")])).collect()[0]
    assert r["description"] is None


# ---------------------------------------------------------------- cancellations
def test_cancellation_goes_to_returns_not_sales(make_bronze):
    bronze = make_bronze([row("489434", "85048"), row("C489449", "22087", qty="-12")])
    assert [r["invoice"] for r in sales_of(bronze).collect()] == ["489434"]
    returns = returns_of(bronze).collect()
    assert [r["invoice"] for r in returns] == ["C489449"]
    assert returns[0]["quantity"] == -12
    assert returns[0]["line_revenue"] == Decimal("-30.000")


def test_non_product_cancellation_not_in_returns(make_bronze):
    bronze = make_bronze([row("C500000", "POST", qty="-1"), row("C500001", "D", qty="-1")])
    assert returns_of(bronze).count() == 0


# ---------------------------------------------------------------- missing customer
def test_missing_customer_kept_and_flagged(make_bronze):
    bronze = make_bronze([row("1", "85048", cust=None), row("2", "85048", cust="12346")])
    flags = {r["invoice"]: r["is_missing_customer"] for r in sales_of(bronze).collect()}
    assert flags == {"1": True, "2": False}


def test_empty_string_customer_is_missing(make_bronze):
    r = sales_of(make_bronze([row("1", "85048", cust="")])).collect()[0]
    assert r["customer_id"] is None and r["is_missing_customer"] is True


# ---------------------------------------------------------------- invalid price / quantity
@pytest.mark.parametrize(
    "qty, price",
    [("1", "0"), ("1", "0.000"), ("1", "-53594.36"), ("0", "2.50"), ("-5", "2.50"), (None, "1")],
)
def test_invalid_price_or_quantity_dropped_from_sales(make_bronze, qty, price):
    assert sales_of(make_bronze([row("1", "85048", qty=qty, price=price)])).count() == 0


@pytest.mark.parametrize("field", ["qty", "price", "ts"])
def test_unparseable_values_become_null_and_are_dropped(make_bronze, field):
    kwargs = {"qty": "abc"} if field == "qty" else {"price": "x"} if field == "price" else {}
    if field == "ts":
        kwargs["ts"] = "not a date"
    bronze = make_bronze([row("1", "85048", **kwargs)])
    assert sales_of(bronze).count() == 0  # no exception, just dropped


# ---------------------------------------------------------------- non-product codes
@pytest.mark.parametrize(
    "code",
    [
        "POST",
        "DOT",
        "C2",
        "C3",
        "M",
        "m",
        "D",
        "S",
        "B",
        "BANK CHARGES",
        "AMAZONFEE",
        "CRUK",
        "GIFT",
        "gift_0001_20",
        "TEST001",
        "TEST002",
        "ADJUST",
        "ADJUST2",
        " POST ",
    ],
)
def test_non_product_codes_removed(make_bronze, code):
    assert sales_of(make_bronze([row("1", code)])).count() == 0


@pytest.mark.parametrize(
    "code", ["85123A", "85123a", "15056BL", "DCGS0058", "DCGSSGIRL", "PADS", "SP1002", "47503J "]
)
def test_real_products_with_any_code_kept(make_bronze, code):
    assert sales_of(make_bronze([row("1", code)])).count() == 1


# ---------------------------------------------------------------- duplicates
def test_duplicates_across_files_kept_once_from_first_file(make_bronze):
    bronze = make_bronze([row("536365", "85123A", src=F2), row("536365", "85123A", src=F1)])
    rows = sales_of(bronze).collect()
    assert len(rows) == 1
    assert rows[0]["_source_file"] == F1


def test_duplicates_after_cleaning_are_detected(make_bronze):
    # Same line except whitespace/case in code and description -> duplicate after cleaning.
    bronze = make_bronze(
        [row("1", "85123A", desc="HEART HOLDER"), row("1", " 85123a", desc="HEART  HOLDER ")]
    )
    assert sales_of(bronze).count() == 1


def test_same_product_different_quantity_is_not_duplicate(make_bronze):
    bronze = make_bronze([row("1", "85123A", qty="1"), row("1", "85123A", qty="2")])
    assert sales_of(bronze).count() == 2


def test_duplicate_cancellations_kept_once_in_returns(make_bronze):
    bronze = make_bronze(
        [row("C1", "22087", qty="-1", src=F1), row("C1", "22087", qty="-1", src=F2)]
    )
    assert returns_of(bronze).count() == 1


# ---------------------------------------------------------------- rule counting
def test_count_rule_failures_counts_each_rule_independently(make_bronze):
    bronze = make_bronze(
        [
            row("1", "85123A"),  # good
            row("2", "POST", price="0"),  # fails is_product AND positive_price
            row("C3", "22087", qty="-1"),  # fails not_cancellation (qty rule too: -1)
            row("1", "85123A", src=F2),  # duplicate of the first row
        ]
    )
    counts = count_rule_failures(stage_silver(bronze), SALES_DROP_RULES).collect()[0]
    assert counts["input_rows"] == 4
    assert counts["not_duplicate"] == 1
    assert counts["not_cancellation"] == 1
    assert counts["is_product"] == 1
    assert counts["positive_price"] == 1
    assert counts["positive_quantity"] == 1
    assert sales_of(bronze).count() == 1


# ---------------------------------------------------------------- profiling helper
def test_stock_code_profile_lists_only_non_standard_codes(make_bronze):
    bronze = make_bronze(
        [row("1", "85123A"), row("2", "POST"), row("3", "POST"), row("4", "DCGS0058")]
    )
    prof = {r["stock_code"]: r for r in stock_code_profile(bronze).collect()}
    assert set(prof) == {"POST", "DCGS0058"}
    assert prof["POST"]["rows"] == 2
    assert prof["POST"]["treated_as_non_product"] is True
    assert prof["DCGS0058"]["treated_as_non_product"] is False
