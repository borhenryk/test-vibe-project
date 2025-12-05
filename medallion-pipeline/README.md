# 🏗️ Medallion Architecture Pipeline

A complete Data Engineering pipeline using the **Medallion Architecture** pattern, deployed to Databricks using **Databricks Asset Bundles (DABs)**.

## 📊 Architecture Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   BRONZE    │───▶│   SILVER    │───▶│    GOLD     │
│  (Raw Data) │    │ (Cleaned)   │    │(Aggregated) │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
  bronze_iris      silver_iris      gold_species_summary
                                   gold_iris_features
```

## 🗂️ Project Structure

```
medallion-pipeline/
├── databricks.yml              # DABs bundle configuration
├── resources/
│   └── medallion_pipeline_job.yml  # Job definitions
├── config/
│   ├── dev.yaml               # Development config
│   ├── staging.yaml           # Staging config
│   └── prod.yaml              # Production config
├── src/medallion_pipeline/
│   ├── bronze/
│   │   └── 01_ingest_bronze.py    # Raw data ingestion
│   ├── silver/
│   │   └── 02_transform_silver.py # Data cleaning & standardization
│   └── gold/
│       └── 03_aggregate_gold.py   # Aggregations & ML features
├── tests/                     # Unit tests
└── .github/workflows/
    └── ci.yml                 # CI/CD pipeline
```

## 🥉 Bronze Layer

**Table**: `{catalog}.{schema}.bronze_iris`

- Raw Iris dataset from sklearn
- Original measurements (sepal/petal length & width)
- Metadata columns for lineage tracking
- No transformations applied

## 🥈 Silver Layer

**Table**: `{catalog}.{schema}.silver_iris`

Transformations applied:
- Data type validation
- Null filtering
- Deduplication
- Z-score normalization
- Derived features (areas, ratios)
- Data quality scoring

## 🥇 Gold Layer

**Tables**:

1. **`gold_species_summary`** - Species-level aggregated statistics
   - Mean, std, min, max for all measurements
   - Perfect for dashboards and reporting

2. **`gold_iris_features`** - ML-ready feature table
   - Standardized features
   - Feature interactions
   - Percentile ranks
   - One-hot encoded species

## 🚀 Deployment

### Prerequisites

1. Databricks workspace access
2. Unity Catalog enabled
3. GitHub repository

### GitHub Secrets Required

Add these secrets to your repository (`Settings > Secrets and variables > Actions`):

| Secret | Description |
|--------|-------------|
| `DATABRICKS_HOST` | Databricks workspace URL (e.g., `https://dbc-xxxxx.cloud.databricks.com`) |
| `DATABRICKS_TOKEN` | Personal Access Token |

### Deploy with GitHub Actions

The CI/CD pipeline automatically:
- **Validates** on every push
- **Deploys to dev** on `develop` branch
- **Deploys to staging** on `main` branch
- **Deploys to prod** after staging (on `main`)

### Manual Deployment

```bash
# Install Databricks CLI
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# Validate bundle
databricks bundle validate -t dev

# Deploy to dev
databricks bundle deploy -t dev

# Run the pipeline job
databricks bundle run medallion_pipeline_job -t dev
```

## 📋 Tables Created

| Layer | Table Name | Description |
|-------|------------|-------------|
| Bronze | `bronze_iris` | Raw Iris dataset |
| Silver | `silver_iris` | Cleaned and standardized data |
| Gold | `gold_species_summary` | Species statistics |
| Gold | `gold_iris_features` | ML-ready features |

## 🔧 Configuration

Edit variables in `databricks.yml`:

```yaml
variables:
  catalog:
    default: "your_catalog"
  schema:
    default: "your_schema"
```

## 📚 References

- [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/index.html)
- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)
- [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
