-- Lakeflow pipeline source (SQL): Gold monthly sales per country.
--
-- Grain: one row per (year_month, country) with sales. Country = the sale's own country
-- (fact_sales.country), not the customer's main country.
-- revenue_share_pct = the country's share of that month's total revenue (window function).
--
-- Phase 2.6 rewrite: year / month / year_month come from the fact's own invoice_ts (dim_date
-- derives them from the same date), so the join to dim_date is gone. Same values as the
-- Phase 1 query; year_month is now nullable in the schema (Spark can't prove it non-null),
-- the values never are. x10: −23.3 % build time. Phase 1 version: src/optimisation/phase1_sql/.
-- Convention used by the tests: the query starts after a line containing only "AS".

CREATE OR REFRESH MATERIALIZED VIEW retail.gold.monthly_country_sales
COMMENT 'Monthly revenue, invoices, units and known customers per country, with share of monthly revenue.'
AS
WITH monthly AS (
  SELECT
    date_format(f.invoice_ts, 'yyyy-MM')                                  AS year_month,
    year(f.invoice_ts)                                                    AS year,
    month(f.invoice_ts)                                                   AS month,
    f.country,
    SUM(f.line_revenue)                                                   AS revenue,
    COUNT(DISTINCT f.invoice)                                             AS invoices,
    SUM(f.quantity)                                                       AS units,
    COUNT(DISTINCT CASE WHEN f.customer_key <> -1 THEN f.customer_key END) AS customers
  FROM retail.gold.fact_sales AS f
  GROUP BY 1, 2, 3, f.country
)
SELECT
  *,
  ROUND(revenue * 100 / SUM(revenue) OVER (PARTITION BY year_month), 3) AS revenue_share_pct
FROM monthly
