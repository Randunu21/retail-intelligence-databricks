-- Lakeflow pipeline source (SQL): Gold calendar dimension.
--
-- One row per calendar day from the first to the last sales date, INCLUDING days without
-- sales (e.g. shop closed over Christmas). That's the point of a date dimension: reports can
-- show zero-sales days and group by week/month/quarter without date functions in every query.
-- Convention used by the tests: the query starts after a line containing only "AS".

CREATE OR REFRESH MATERIALIZED VIEW retail.gold.dim_date
COMMENT 'Calendar: one row per day between the first and last sales date (incl. days without sales).'
AS
WITH bounds AS (
  SELECT MIN(invoice_date) AS first_day, MAX(invoice_date) AS last_day
  FROM retail.silver.sales
),
days AS (
  SELECT explode(sequence(first_day, last_day, INTERVAL 1 DAY)) AS date
  FROM bounds
)
SELECT
  CAST(date_format(date, 'yyyyMMdd') AS INT) AS date_key,
  date,
  year(date)                                 AS year,
  quarter(date)                              AS quarter,
  month(date)                                AS month,
  date_format(date, 'MMMM')                  AS month_name,
  date_format(date, 'yyyy-MM')               AS year_month,
  weekofyear(date)                           AS week_of_year,
  dayofmonth(date)                           AS day_of_month,
  dayofweek(date)                            AS day_of_week,  -- 1 = Sunday ... 7 = Saturday
  date_format(date, 'EEEE')                  AS day_name,
  dayofweek(date) IN (1, 7)                  AS is_weekend
FROM days
