-- Lakeflow pipeline source (SQL): Gold daily sales.
--
-- One row per calendar day (from dim_date), INCLUDING days without sales (revenue 0),
-- so trend charts show closures instead of silently joining neighbouring days.
-- customers = distinct KNOWN customers (the Unknown member -1 is not a person).
--
-- Phase 2.6 rewrite: the fact table is aggregated to one row per date_key FIRST (10M rows ->
-- ~600 on the x10 replica), then left-joined to the calendar. Same columns, types and values
-- as the Phase 1 query (an invoice belongs to exactly one day, so per-day DISTINCT counts are
-- unchanged); x10: −26.3 % build time. Phase 1 version: src/optimisation/phase1_sql/.
-- Convention used by the tests: the query starts after a line containing only "AS".

CREATE OR REFRESH MATERIALIZED VIEW retail.gold.daily_sales
COMMENT 'Daily revenue, invoices, units, lines, known customers and average basket value (all calendar days).'
AS
WITH f AS (
  SELECT
    date_key,
    SUM(line_revenue)                                                    AS revenue,
    COUNT(DISTINCT invoice)                                              AS invoices,
    SUM(quantity)                                                        AS units,
    COUNT(line_id)                                                       AS lines,
    COUNT(DISTINCT CASE WHEN customer_key <> -1 THEN customer_key END)   AS customers
  FROM retail.gold.fact_sales
  GROUP BY date_key
)
SELECT
  d.date_key,
  d.date,
  d.year_month,
  d.day_name,
  d.is_weekend,
  COALESCE(f.revenue, 0)                          AS revenue,
  COALESCE(f.invoices, 0)                         AS invoices,
  COALESCE(f.units, 0)                            AS units,
  COALESCE(f.lines, 0)                            AS lines,
  COALESCE(f.customers, 0)                        AS customers,
  ROUND(try_divide(f.revenue, f.invoices), 2)     AS avg_basket_value
FROM retail.gold.dim_date AS d
LEFT JOIN f
  ON f.date_key = d.date_key
