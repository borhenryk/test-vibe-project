# Databricks notebook source
# MAGIC %md
# MAGIC # 🥉 Bronze Layer - Raw Data Ingestion
# MAGIC 
# MAGIC This notebook ingests the Iris dataset into the Bronze layer.
# MAGIC The Bronze layer stores raw, unprocessed data as-is from the source.
# MAGIC 
# MAGIC **Pipeline Stage**: Bronze (Raw Data)
# MAGIC **Source**: Iris Dataset (sklearn)
# MAGIC **Target**: `{catalog}.{schema}.bronze_iris`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Parameters

# COMMAND ----------

# Get parameters from widgets or use defaults
try:
    dbutils.widgets.text("catalog", "dbdemos_henryk", "Catalog Name")
    dbutils.widgets.text("schema", "medallion_pipeline", "Schema Name")
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
except Exception as e:
    # Fallback for serverless or when widgets aren't available
    catalog = "dbdemos_henryk"
    schema = "medallion_pipeline"

print(f"📦 Catalog: {catalog}")
print(f"📁 Schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install scikit-learn pandas -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-fetch parameters after Python restart
try:
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
except:
    catalog = "dbdemos_henryk"
    schema = "medallion_pipeline"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema if Not Exists

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"✅ Schema {catalog}.{schema} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and Load Iris Dataset

# COMMAND ----------

from sklearn.datasets import load_iris
import pandas as pd
from datetime import datetime

# Load iris dataset
iris = load_iris()

# Create pandas DataFrame with feature names and target
iris_df = pd.DataFrame(
    data=iris.data,
    columns=[
        "sepal_length_cm",
        "sepal_width_cm", 
        "petal_length_cm",
        "petal_width_cm"
    ]
)

# Add target column with species names
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
iris_df["target"] = iris.target
iris_df["species"] = iris_df["target"].map(species_map)

# Add metadata columns for Bronze layer
iris_df["_ingestion_timestamp"] = datetime.utcnow()
iris_df["_source"] = "sklearn_iris_dataset"
iris_df["_source_version"] = "1.0"

print(f"📊 Loaded {len(iris_df)} records from Iris dataset")
iris_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Bronze Table

# COMMAND ----------

# Convert to Spark DataFrame
bronze_df = spark.createDataFrame(iris_df)

# Define table name
bronze_table = f"{catalog}.{schema}.bronze_iris"

# Write to Bronze table with schema evolution enabled
bronze_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(bronze_table)

print(f"✅ Bronze table created: {bronze_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Table Metadata

# COMMAND ----------

# Add table comment
spark.sql(f"""
    ALTER TABLE {bronze_table}
    SET TBLPROPERTIES (
        'delta.minReaderVersion' = '1',
        'delta.minWriterVersion' = '2'
    )
""")

spark.sql(f"""
    COMMENT ON TABLE {bronze_table} IS 
    'Bronze layer: Raw Iris dataset ingested from sklearn. Contains unprocessed flower measurements for setosa, versicolor, and virginica species.'
""")

# Add column comments
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN sepal_length_cm COMMENT 'Sepal length in centimeters (raw measurement)'")
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN sepal_width_cm COMMENT 'Sepal width in centimeters (raw measurement)'")
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN petal_length_cm COMMENT 'Petal length in centimeters (raw measurement)'")
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN petal_width_cm COMMENT 'Petal width in centimeters (raw measurement)'")
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN target COMMENT 'Numeric species identifier (0=setosa, 1=versicolor, 2=virginica)'")
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN species COMMENT 'Species name: setosa, versicolor, or virginica'")
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN _ingestion_timestamp COMMENT 'Timestamp when data was ingested into Bronze layer'")
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN _source COMMENT 'Data source identifier'")
spark.sql(f"ALTER TABLE {bronze_table} ALTER COLUMN _source_version COMMENT 'Version of the source data'")

print("✅ Table metadata added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Bronze Table

# COMMAND ----------

# Show table info
print(f"📋 Table: {bronze_table}")
print(f"📊 Record count: {spark.table(bronze_table).count()}")
print("\n📝 Schema:")
spark.table(bronze_table).printSchema()

# Display sample data
display(spark.table(bronze_table).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Status

# COMMAND ----------

# Return success status
dbutils.notebook.exit("SUCCESS: Bronze layer ingestion complete")
