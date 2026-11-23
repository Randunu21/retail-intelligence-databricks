# Phase 1 versions of rewritten pipeline queries

Phase 2.6 rewrote two Gold queries in `pipelines/medallion/transformations/`
(daily_sales: aggregate before the join; monthly_country_sales: no join to dim_date).

The benchmark "baseline" must keep meaning "the Phase 1 pipeline", so these are frozen,
unchanged copies of the Phase 1 files. `optimisation.builds` reads them instead of the
current pipeline files for these two steps. Don't edit them.
