-- Lakeflow pipeline source (SQL): Gold fact table.
--
-- Grain: ONE ROW PER DE-DUPLICATED INVOICE LINE of retail.silver.sales (product sales only;
-- cancellations live in retail.silver.returns).
-- Keys:
--   line_id       surrogate key = hash of the business columns (unique after Silver de-dup)
--   customer_key  -> gold.dim_customer  (-1 = unknown customer instead of a NULL key)
--   stock_code    -> gold.dim_product
--   date_key      -> gold.dim_date      (yyyyMMdd as INT)
-- Layout (Phase 2.6): liquid clustering on (date_key, customer_key). Date-range reports and
-- customer lookups skip files: on the x10 replica they read 38–56 % fewer bytes (Phase 2.3).
-- Chosen over partitioning, which made ~600 tiny daily files and slowed everything else.
-- Convention used by the tests: the query starts after a line containing only "AS".

CREATE OR REFRESH MATERIALIZED VIEW retail.gold.fact_sales
COMMENT 'Sales fact, one row per de-duplicated invoice line. FKs: customer_key, stock_code, date_key.'
TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
CLUSTER BY (date_key, customer_key)
AS
SELECT
  xxhash64(
    s.invoice, s.stock_code, s.description, s.quantity,
    s.invoice_ts, s.price, s.customer_id, s.country
  )                                                   AS line_id,
  s.invoice,
  CAST(date_format(s.invoice_date, 'yyyyMMdd') AS INT) AS date_key,
  COALESCE(s.customer_id, -1)                         AS customer_key,
  s.stock_code,
  s.country,
  s.invoice_ts,
  s.quantity,
  s.price                                             AS unit_price,
  s.line_revenue,
  s.is_missing_customer
FROM retail.silver.sales AS s
