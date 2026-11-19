"""Reuse the Lakeflow pipeline's SQL files outside the pipeline.

The Gold SQL lives in `pipelines/medallion/transformations/*.sql` as
`CREATE OR REFRESH MATERIALIZED VIEW <name> ... AS <query>`. The benchmark harness needs the
same `<query>`, but reading from benchmark copies of the tables instead of production ones.
So instead of copying the SQL (and letting the two versions drift apart), we:

1. cut the query out of the file (everything after the line that is just `AS`, the same
   convention `tests/factories.run_pipeline_query` relies on), and
2. swap each production table name for another name (a benchmark table or a temp view).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

# A line that contains only "AS" (any case, surrounding spaces allowed).
_AS_LINE = re.compile(r"(?im)^\s*AS\s*$")


def extract_query(sql_text: str) -> str:
    """Return the query part of a pipeline SQL file (the text after the `AS` line)."""
    parts = _AS_LINE.split(sql_text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("No line containing only 'AS' found: is this a pipeline SQL file?")
    return parts[1].strip()


def retarget(query: str, mapping: Mapping[str, str]) -> str:
    """Replace whole table names in `query` (e.g. retail.silver.sales -> retail.bench.x__y).

    Matching is on the full dotted name with boundaries on both sides, so
    `retail.gold.fact_sales` never matches inside `retail.gold.fact_sales_v2`. Longer names
    are replaced first, so one name being a prefix of another can't cause a clash.
    """
    for old in sorted(mapping, key=len, reverse=True):
        pattern = rf"(?<![\w.`]){re.escape(old)}(?![\w`])"
        query = re.sub(pattern, mapping[old], query)
    return query


def load_query(pipeline_dir: str | Path, file_name: str, mapping: Mapping[str, str]) -> str:
    """Read a pipeline SQL file, extract its query and retarget the table names."""
    text = (Path(pipeline_dir) / file_name).read_text()
    return retarget(extract_query(text), mapping)
