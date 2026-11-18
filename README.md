# Retail Customer & Product Intelligence Platform on Databricks

End-to-end retail analytics and ML platform built on **Databricks** with **PySpark, Spark SQL, Delta Lake, Unity Catalog and MLflow**, using the UCI **Online Retail II** dataset (~1M UK e-commerce transactions, Dec 2009 – Dec 2011).

> 🚧 Work in progress. Results below are filled in only from real runs.

## What it does (planned)
1. **Medallion pipeline**: Bronze → Silver → Gold Delta tables with data-quality checks
2. **Pipeline optimisation**: measured before/after (liquid clustering, broadcast joins, incremental MERGE)
3. **Customer segmentation**: RFM + behavioural features, KMeans / BisectingKMeans / GMM
4. **Product segmentation & market basket analysis**: product clustering, FP-Growth
5. **Repeat-purchase propensity model**: LightGBM, MLflow, UC model registry (`@champion`/`@challenger`)
6. **MLOps**: Databricks Jobs, batch scoring, drift monitoring, CI/CD with GitHub Actions + Asset Bundles
7. **Analytics**: Databricks AI/BI dashboard with business insights

## Repository layout
```
src/            reusable PySpark modules (ingestion, transformations, features, ...)
notebooks/      one explanatory notebook per phase (imports from src/)
sql/            Gold views and dashboard queries
tests/          pytest unit tests on small local DataFrames
scripts/        local helper scripts (e.g. dataset conversion)
docs/           architecture, optimisation report, segment profiles
resources/      job/pipeline definitions for the Asset Bundle
```

## Local setup (macOS)
```bash
brew install python@3.12 openjdk@17
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
ruff check . && pytest
```

## Results
_To be added with real numbers after each phase._

## Dataset
Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
