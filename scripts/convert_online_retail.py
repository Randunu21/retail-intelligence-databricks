"""Convert the UCI Online Retail II workbook (.xlsx, 2 sheets) into one CSV per sheet.

Why: Spark on Databricks serverless has no built-in Excel reader (spark-excel is a JVM/Maven
library that serverless can't install), so we convert once, locally, to a format Spark
reads natively and in parallel.

Design choices (Bronze = "raw, as-is"):
- Every column is read as TEXT (dtype=str): no type guessing, so invoice codes like
  "C489449", stock codes like "85123A" and IDs keep their original form. Typing happens
  in Silver (Phase 1), where it is explicit and tested.
- Column headers are kept exactly as in the source (e.g. "Customer ID"); renaming to
  snake_case is a Silver concern.
- No extra columns are added. The sheet/year is encoded in the file name, and Spark can
  read it back from the `_metadata.file_name` column.
- Missing values are written as empty fields, which Spark reads as NULL.

Usage (from the repo root, with the venv active):
    python scripts/convert_online_retail.py data/online_retail_II.xlsx --out data/raw_csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def sheet_to_filename(sheet_name: str) -> str:
    """'Year 2009-2010' -> 'online_retail_II_2009_2010.csv'."""
    years = re.findall(r"\d{4}", sheet_name)
    suffix = "_".join(years) if years else re.sub(r"\W+", "_", sheet_name).strip("_").lower()
    return f"online_retail_II_{suffix}.csv"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(xlsx_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {xlsx_path} (all sheets, as text). This can take a few minutes...")
    sheets: dict[str, pd.DataFrame] = pd.read_excel(
        xlsx_path, sheet_name=None, dtype=str, engine="openpyxl"
    )

    manifest = {
        "source_file": xlsx_path.name,
        "source_sha256": sha256_of(xlsx_path),
        "sheets": [],
    }

    for sheet_name, df in sheets.items():
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Sheet {sheet_name!r} is missing columns: {missing}")

        out_file = out_dir / sheet_to_filename(sheet_name)
        df.to_csv(out_file, index=False, encoding="utf-8")  # NaN -> empty field

        info = {
            "sheet": sheet_name,
            "csv_file": out_file.name,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "null_customer_id": int(df["Customer ID"].isna().sum()),
        }
        manifest["sheets"].append(info)
        print(f"  {sheet_name!r:>20} -> {out_file.name}: {info['rows']:,} rows")

    manifest["total_rows"] = sum(s["rows"] for s in manifest["sheets"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Total rows: {manifest['total_rows']:,}. Manifest: {out_dir / 'manifest.json'}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xlsx", type=Path, help="Path to online_retail_II.xlsx")
    parser.add_argument("--out", type=Path, default=Path("data/raw_csv"), help="Output folder")
    args = parser.parse_args(argv)

    if not args.xlsx.exists():
        print(f"File not found: {args.xlsx}", file=sys.stderr)
        return 1
    convert(args.xlsx, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
