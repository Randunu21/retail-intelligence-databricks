-- Lakeflow pipeline source (SQL): Gold monthly sales per country.
--
-- Grain: one row per (year_month, country) with sales. Country = the sale's own country
-- (fact_sales.country), not the customer's main country.
-- revenue_share_pct = the country's share of that month's total revenue (window function).
-- Convention used by the tests: the query starts after a line containing only "AS".

CREATE OR REFRESH MATERIALIZED VIEW retail.gold.monthly_country_sales
COMMENT 'Monthly revenue, invoices, units and known customers per country, with share of monthly revenue.'
AS
WITH monthly AS (
  SELECT
    d.year_month,
    d.year,
    d.month,
    f.country,
    SUM(f.line_revenue)                                                   AS revenue,
    COUNT(DISTINCT f.invoice)                                             AS invoices,
    SUM(f.quantity)                                                       AS units,
    COUNT(DISTINCT CASE WHEN f.customer_key <> -1 THEN f.customer_key END) AS customers
  FROM retail.gold.fact_sales AS f
  JOIN retail.gold.dim_date AS d
    ON f.date_key = d.date_key
  GROUP BY d.year_month, d.year, d.month, f.country
)
SELECT
  *,
  ROUND(revenue * 100 / SUM(revenue) OVER (PARTITION BY year_month), 3) AS revenue_share_pct
FROM monthly
