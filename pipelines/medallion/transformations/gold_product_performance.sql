-- Lakeflow pipeline source (SQL): Gold product performance.
--
-- One row per product in dim_product: sales KPIs from fact_sales plus returns from
-- retail.silver.returns (quantities/values there are negative, so they are negated here).
-- return_rate = units returned / units sold. It can exceed 1 for a product whose returns
-- relate to sales from before Dec 2009 (outside the dataset); kept as-is, not capped.
-- Convention used by the tests: the query starts after a line containing only "AS".

CREATE OR REFRESH MATERIALIZED VIEW retail.gold.product_performance
COMMENT 'Per-product revenue, units, invoices, customers, countries, returns, return rate and revenue rank.'
AS
WITH sold AS (
  SELECT
    stock_code,
    SUM(line_revenue)                                                   AS revenue,
    SUM(quantity)                                                       AS units_sold,
    COUNT(DISTINCT invoice)                                             AS invoices,
    COUNT(DISTINCT CASE WHEN customer_key <> -1 THEN customer_key END)   AS customers,
    COUNT(DISTINCT country)                                             AS countries
  FROM retail.gold.fact_sales
  GROUP BY stock_code
),
returned AS (
  SELECT
    stock_code,
    -SUM(quantity)     AS units_returned,
    -SUM(line_revenue) AS returned_value
  FROM retail.silver.returns
  GROUP BY stock_code
)
SELECT
  p.stock_code,
  p.description,
  s.revenue,
  s.units_sold,
  s.invoices,
  s.customers,
  s.countries,
  COALESCE(r.units_returned, 0)                                    AS units_returned,
  COALESCE(r.returned_value, 0)                                    AS returned_value,
  ROUND(try_divide(COALESCE(r.units_returned, 0), s.units_sold), 4) AS return_rate,
  ROUND(s.revenue * 100 / SUM(s.revenue) OVER (), 4)               AS revenue_share_pct,
  DENSE_RANK() OVER (ORDER BY s.revenue DESC)                      AS revenue_rank,
  p.first_sold_date,
  p.last_sold_date
FROM retail.gold.dim_product AS p
JOIN sold AS s
  ON s.stock_code = p.stock_code
LEFT JOIN returned AS r
  ON r.stock_code = p.stock_code
