# Databricks notebook source
# MAGIC %md
# MAGIC # 🥇 Gold Layer - Aggregated & Analytics-Ready Data
# MAGIC 
# MAGIC This notebook creates Gold layer tables from Silver data.
# MAGIC The Gold layer contains business-level aggregations and analytics-ready datasets.
# MAGIC 
# MAGIC **Pipeline Stage**: Gold (Analytics Layer)
# MAGIC **Source**: `{catalog}.{schema}.silver_iris`
# MAGIC **Targets**: 
# MAGIC - `{catalog}.{schema}.gold_species_summary`
# MAGIC - `{catalog}.{schema}.gold_iris_features`
# MAGIC 
# MAGIC **Aggregations Applied**:
# MAGIC - Species-level statistics
# MAGIC - Feature correlations
# MAGIC - ML-ready feature table

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
    catalog = "dbdemos_henryk"
    schema = "medallion_pipeline"

print(f"📦 Catalog: {catalog}")
print(f"📁 Schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver Data

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# Source and target tables
silver_table = f"{catalog}.{schema}.silver_iris"
gold_species_summary = f"{catalog}.{schema}.gold_species_summary"
gold_features_table = f"{catalog}.{schema}.gold_iris_features"

# Load Silver data
silver_df = spark.table(silver_table)
print(f"📥 Loaded {silver_df.count()} records from Silver layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Table 1: Species Summary Statistics
# MAGIC 
# MAGIC Aggregated statistics per species - useful for reporting and dashboards.

# COMMAND ----------

# Aggregate statistics by species
species_summary_df = silver_df.groupBy("species", "target").agg(
    # Count
    F.count("*").alias("record_count"),
    
    # Sepal statistics
    F.round(F.mean("sepal_length_cm"), 3).alias("avg_sepal_length_cm"),
    F.round(F.stddev("sepal_length_cm"), 3).alias("std_sepal_length_cm"),
    F.round(F.min("sepal_length_cm"), 3).alias("min_sepal_length_cm"),
    F.round(F.max("sepal_length_cm"), 3).alias("max_sepal_length_cm"),
    
    F.round(F.mean("sepal_width_cm"), 3).alias("avg_sepal_width_cm"),
    F.round(F.stddev("sepal_width_cm"), 3).alias("std_sepal_width_cm"),
    F.round(F.min("sepal_width_cm"), 3).alias("min_sepal_width_cm"),
    F.round(F.max("sepal_width_cm"), 3).alias("max_sepal_width_cm"),
    
    # Petal statistics
    F.round(F.mean("petal_length_cm"), 3).alias("avg_petal_length_cm"),
    F.round(F.stddev("petal_length_cm"), 3).alias("std_petal_length_cm"),
    F.round(F.min("petal_length_cm"), 3).alias("min_petal_length_cm"),
    F.round(F.max("petal_length_cm"), 3).alias("max_petal_length_cm"),
    
    F.round(F.mean("petal_width_cm"), 3).alias("avg_petal_width_cm"),
    F.round(F.stddev("petal_width_cm"), 3).alias("std_petal_width_cm"),
    F.round(F.min("petal_width_cm"), 3).alias("min_petal_width_cm"),
    F.round(F.max("petal_width_cm"), 3).alias("max_petal_width_cm"),
    
    # Area statistics
    F.round(F.mean("sepal_area_cm2"), 3).alias("avg_sepal_area_cm2"),
    F.round(F.mean("petal_area_cm2"), 3).alias("avg_petal_area_cm2"),
    
    # Ratio statistics
    F.round(F.mean("sepal_petal_length_ratio"), 3).alias("avg_sepal_petal_length_ratio"),
    F.round(F.mean("sepal_petal_width_ratio"), 3).alias("avg_sepal_petal_width_ratio")
)

# Add metadata
species_summary_df = species_summary_df.withColumn(
    "_gold_processed_timestamp",
    F.lit(datetime.utcnow())
).withColumn(
    "_aggregation_version",
    F.lit("1.0.0")
)

# Order by species
species_summary_df = species_summary_df.orderBy("target")

print("📊 Species Summary Statistics:")
display(species_summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold Species Summary Table

# COMMAND ----------

# Write to Gold table
species_summary_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(gold_species_summary)

print(f"✅ Gold table created: {gold_species_summary}")

# Add table comment
spark.sql(f"""
    COMMENT ON TABLE {gold_species_summary} IS 
    'Gold layer: Species-level aggregated statistics for Iris dataset. Contains mean, std, min, max for all measurements per species. Ideal for dashboards and reporting.'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Table 2: ML-Ready Features Table
# MAGIC 
# MAGIC Feature-engineered table optimized for machine learning workflows.

# COMMAND ----------

# Create a unique identifier for each record
window_spec = Window.orderBy(F.monotonically_increasing_id())

features_df = silver_df.withColumn(
    "record_id",
    F.row_number().over(window_spec)
)

# Calculate percentile ranks within species (useful for ML)
species_window = Window.partitionBy("species")

features_df = features_df.withColumn(
    "sepal_length_percentile",
    F.round(F.percent_rank().over(species_window.orderBy("sepal_length_cm")), 3)
).withColumn(
    "sepal_width_percentile",
    F.round(F.percent_rank().over(species_window.orderBy("sepal_width_cm")), 3)
).withColumn(
    "petal_length_percentile",
    F.round(F.percent_rank().over(species_window.orderBy("petal_length_cm")), 3)
).withColumn(
    "petal_width_percentile",
    F.round(F.percent_rank().over(species_window.orderBy("petal_width_cm")), 3)
)

# Add species as one-hot encoded columns
features_df = features_df.withColumn(
    "is_setosa",
    F.when(F.col("species") == "setosa", 1).otherwise(0)
).withColumn(
    "is_versicolor",
    F.when(F.col("species") == "versicolor", 1).otherwise(0)
).withColumn(
    "is_virginica",
    F.when(F.col("species") == "virginica", 1).otherwise(0)
)

# Calculate feature interactions
features_df = features_df.withColumn(
    "sepal_length_x_petal_length",
    F.round(F.col("sepal_length_cm") * F.col("petal_length_cm"), 3)
).withColumn(
    "sepal_width_x_petal_width",
    F.round(F.col("sepal_width_cm") * F.col("petal_width_cm"), 3)
).withColumn(
    "total_area_cm2",
    F.round(F.col("sepal_area_cm2") + F.col("petal_area_cm2"), 3)
)

# Add Gold layer metadata
features_df = features_df.withColumn(
    "_gold_processed_timestamp",
    F.lit(datetime.utcnow())
).withColumn(
    "_feature_engineering_version",
    F.lit("1.0.0")
).withColumn(
    "_is_ml_ready",
    F.lit(True)
)

# COMMAND ----------

# Select final columns for ML features table
ml_features_df = features_df.select(
    # Identifiers
    "record_id",
    "species",
    "target",
    
    # Original measurements
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
    
    # Standardized features (z-score)
    "sepal_length_cm_standardized",
    "sepal_width_cm_standardized",
    "petal_length_cm_standardized",
    "petal_width_cm_standardized",
    
    # Derived features
    "sepal_area_cm2",
    "petal_area_cm2",
    "total_area_cm2",
    "sepal_petal_length_ratio",
    "sepal_petal_width_ratio",
    
    # Feature interactions
    "sepal_length_x_petal_length",
    "sepal_width_x_petal_width",
    
    # Percentile ranks
    "sepal_length_percentile",
    "sepal_width_percentile",
    "petal_length_percentile",
    "petal_width_percentile",
    
    # One-hot encoded species
    "is_setosa",
    "is_versicolor",
    "is_virginica",
    
    # Metadata
    "_gold_processed_timestamp",
    "_feature_engineering_version",
    "_is_ml_ready"
)

print(f"📊 ML Features Table - {ml_features_df.count()} records with {len(ml_features_df.columns)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold ML Features Table

# COMMAND ----------

# Write to Gold table
ml_features_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(gold_features_table)

print(f"✅ Gold table created: {gold_features_table}")

# Add table comment
spark.sql(f"""
    COMMENT ON TABLE {gold_features_table} IS 
    'Gold layer: ML-ready feature table for Iris dataset. Contains original measurements, standardized features, derived features (areas, ratios), percentile ranks, one-hot encoded species, and feature interactions. Optimized for machine learning workflows.'
""")

# Add column comments for key features
feature_comments = {
    "record_id": "Unique record identifier (primary key)",
    "total_area_cm2": "Combined sepal and petal area in cm²",
    "sepal_length_x_petal_length": "Feature interaction: sepal_length × petal_length",
    "sepal_width_x_petal_width": "Feature interaction: sepal_width × petal_width",
    "sepal_length_percentile": "Percentile rank of sepal length within species",
    "is_setosa": "One-hot encoded: 1 if species is setosa, 0 otherwise",
    "is_versicolor": "One-hot encoded: 1 if species is versicolor, 0 otherwise",
    "is_virginica": "One-hot encoded: 1 if species is virginica, 0 otherwise",
    "_is_ml_ready": "Flag indicating data is ready for ML consumption"
}

for col_name, comment in feature_comments.items():
    try:
        spark.sql(f"ALTER TABLE {gold_features_table} ALTER COLUMN {col_name} COMMENT '{comment}'")
    except Exception as e:
        print(f"Warning: Could not add comment for {col_name}: {e}")

print("✅ Table metadata added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Gold Tables

# COMMAND ----------

# Summary of all Gold tables
print("=" * 60)
print("🥇 GOLD LAYER SUMMARY")
print("=" * 60)

print(f"\n📋 Table 1: {gold_species_summary}")
print(f"   Records: {spark.table(gold_species_summary).count()}")
print(f"   Columns: {len(spark.table(gold_species_summary).columns)}")

print(f"\n📋 Table 2: {gold_features_table}")
print(f"   Records: {spark.table(gold_features_table).count()}")
print(f"   Columns: {len(spark.table(gold_features_table).columns)}")

print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Final Gold Tables

# COMMAND ----------

print("📊 Gold Species Summary:")
display(spark.table(gold_species_summary))

# COMMAND ----------

print("📊 Gold ML Features (sample):")
display(spark.table(gold_features_table).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Status

# COMMAND ----------

# Return success status
dbutils.notebook.exit("SUCCESS: Gold layer aggregation complete - 2 tables created")
