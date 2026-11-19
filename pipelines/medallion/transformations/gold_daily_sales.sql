-- Lakeflow pipeline source (SQL): Gold daily sales.
--
-- One row per calendar day (from dim_date), INCLUDING days without sales (revenue 0),
-- so trend charts show closures instead of silently joining neighbouring days.
-- customers = distinct KNOWN customers (the Unknown member -1 is not a person).
-- Convention used by the tests: the query starts after a line containing only "AS".

CREATE OR REFRESH MATERIALIZED VIEW retail.gold.daily_sales
COMMENT 'Daily revenue, invoices, units, lines, known customers and average basket value (all calendar days).'
AS
SELECT
  d.date_key,
  d.date,
  d.year_month,
  d.day_name,
  d.is_weekend,
  COALESCE(SUM(f.line_revenue), 0)                                     AS revenue,
  COUNT(DISTINCT f.invoice)                                            AS invoices,
  COALESCE(SUM(f.quantity), 0)                                         AS units,
  COUNT(f.line_id)                                                     AS lines,
  COUNT(DISTINCT CASE WHEN f.customer_key <> -1 THEN f.customer_key END) AS customers,
  ROUND(try_divide(SUM(f.line_revenue), COUNT(DISTINCT f.invoice)), 2)  AS avg_basket_value
FROM retail.gold.dim_date AS d
LEFT JOIN retail.gold.fact_sales AS f
  ON f.date_key = d.date_key
GROUP BY d.date_key, d.date, d.year_month, d.day_name, d.is_weekend
