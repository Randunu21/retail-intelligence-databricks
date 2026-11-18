-- =====================================================================
-- Phase 0: Unity Catalog setup for the Retail Intelligence project
-- Run once in the Databricks SQL editor (or a notebook with %sql).
-- Every statement is idempotent (IF NOT EXISTS), so it is safe to re-run.
-- =====================================================================

-- 1. Catalog: the top level of the three-level namespace (catalog.schema.object).
--    On Free Edition no MANAGED LOCATION is given: the catalog uses the
--    workspace's default managed storage.
CREATE CATALOG IF NOT EXISTS retail
  COMMENT 'Retail Customer & Product Intelligence portfolio project (Online Retail II)';

USE CATALOG retail;

-- 2. Schemas: one per medallion layer, plus one for ML assets.
CREATE SCHEMA IF NOT EXISTS bronze COMMENT 'Raw data as ingested + ingestion metadata; raw files Volume';
CREATE SCHEMA IF NOT EXISTS silver COMMENT 'Cleaned, typed, conformed tables';
CREATE SCHEMA IF NOT EXISTS gold   COMMENT 'Business-level star schema, aggregates, segments, scores';
CREATE SCHEMA IF NOT EXISTS ml     COMMENT 'Feature tables, registered models, monitoring/drift tables';

-- 3. Managed Volume for raw files (governed file storage, not a table).
--    Path: /Volumes/retail/bronze/raw/
CREATE VOLUME IF NOT EXISTS retail.bronze.raw
  COMMENT 'Raw Online Retail II files uploaded from local machine (xlsx + per-sheet CSV)';

-- 4. Verify
SHOW SCHEMAS IN retail;
DESCRIBE VOLUME retail.bronze.raw;
