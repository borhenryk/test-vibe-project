# Databricks notebook source
# MAGIC %md
# MAGIC # 🥈 Silver Layer - Data Cleaning & Standardization
# MAGIC 
# MAGIC This notebook transforms Bronze data into the Silver layer.
# MAGIC The Silver layer contains cleaned, validated, and standardized data.
# MAGIC 
# MAGIC **Pipeline Stage**: Silver (Cleaned Data)
# MAGIC **Source**: `{catalog}.{schema}.bronze_iris`
# MAGIC **Target**: `{catalog}.{schema}.silver_iris`
# MAGIC 
# MAGIC **Transformations Applied**:
# MAGIC - Data type validation
# MAGIC - Null handling
# MAGIC - Standardization (z-score normalization)
# MAGIC - Data quality checks
# MAGIC - Deduplication

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
# MAGIC ## Load Bronze Data

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, IntegerType, TimestampType
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# Source and target tables
bronze_table = f"{catalog}.{schema}.bronze_iris"
silver_table = f"{catalog}.{schema}.silver_iris"

# Load Bronze data
bronze_df = spark.table(bronze_table)
print(f"📥 Loaded {bronze_df.count()} records from Bronze layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check for nulls in critical columns
critical_columns = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "species"]

print("🔍 Data Quality Report - Bronze Layer:")
print("-" * 50)

for col in critical_columns:
    null_count = bronze_df.filter(F.col(col).isNull()).count()
    print(f"  {col}: {null_count} null values")

total_records = bronze_df.count()
print(f"\n📊 Total records: {total_records}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Transformations

# COMMAND ----------

# Step 1: Remove duplicates based on all measurement columns
silver_df = bronze_df.dropDuplicates([
    "sepal_length_cm", 
    "sepal_width_cm", 
    "petal_length_cm", 
    "petal_width_cm", 
    "species"
])

print(f"🧹 After deduplication: {silver_df.count()} records")

# COMMAND ----------

# Step 2: Filter out invalid records (nulls in critical columns)
silver_df = silver_df.filter(
    F.col("sepal_length_cm").isNotNull() &
    F.col("sepal_width_cm").isNotNull() &
    F.col("petal_length_cm").isNotNull() &
    F.col("petal_width_cm").isNotNull() &
    F.col("species").isNotNull()
)

print(f"✅ After null filtering: {silver_df.count()} records")

# COMMAND ----------

# Step 3: Calculate statistics for z-score normalization
measurement_cols = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]

# Calculate mean and stddev for each column
stats = {}
for col_name in measurement_cols:
    mean_val = silver_df.select(F.mean(col_name)).collect()[0][0]
    std_val = silver_df.select(F.stddev(col_name)).collect()[0][0]
    stats[col_name] = {"mean": mean_val, "std": std_val}
    print(f"📈 {col_name}: mean={mean_val:.3f}, std={std_val:.3f}")

# COMMAND ----------

# Step 4: Add standardized (z-score) columns
for col_name in measurement_cols:
    standardized_col = f"{col_name}_standardized"
    silver_df = silver_df.withColumn(
        standardized_col,
        (F.col(col_name) - F.lit(stats[col_name]["mean"])) / F.lit(stats[col_name]["std"])
    )

print("✅ Added standardized (z-score) columns")

# COMMAND ----------

# Step 5: Add derived features
silver_df = silver_df.withColumn(
    "sepal_area_cm2",
    F.col("sepal_length_cm") * F.col("sepal_width_cm")
).withColumn(
    "petal_area_cm2", 
    F.col("petal_length_cm") * F.col("petal_width_cm")
).withColumn(
    "sepal_petal_length_ratio",
    F.col("sepal_length_cm") / F.col("petal_length_cm")
).withColumn(
    "sepal_petal_width_ratio",
    F.col("sepal_width_cm") / F.col("petal_width_cm")
)

print("✅ Added derived features (areas and ratios)")

# COMMAND ----------

# Step 6: Add Silver layer metadata
silver_df = silver_df.withColumn(
    "_silver_processed_timestamp",
    F.lit(datetime.utcnow())
).withColumn(
    "_data_quality_score",
    F.lit(1.0)  # All records passed quality checks
).withColumn(
    "_transformation_version",
    F.lit("1.0.0")
)

# COMMAND ----------

# Step 7: Select and reorder columns for Silver table
silver_df = silver_df.select(
    # Original measurements
    F.col("sepal_length_cm").cast(DoubleType()),
    F.col("sepal_width_cm").cast(DoubleType()),
    F.col("petal_length_cm").cast(DoubleType()),
    F.col("petal_width_cm").cast(DoubleType()),
    
    # Standardized measurements
    F.col("sepal_length_cm_standardized").cast(DoubleType()),
    F.col("sepal_width_cm_standardized").cast(DoubleType()),
    F.col("petal_length_cm_standardized").cast(DoubleType()),
    F.col("petal_width_cm_standardized").cast(DoubleType()),
    
    # Derived features
    F.col("sepal_area_cm2").cast(DoubleType()),
    F.col("petal_area_cm2").cast(DoubleType()),
    F.col("sepal_petal_length_ratio").cast(DoubleType()),
    F.col("sepal_petal_width_ratio").cast(DoubleType()),
    
    # Species info
    F.col("target").cast(IntegerType()),
    F.col("species").cast(StringType()),
    
    # Metadata
    F.col("_ingestion_timestamp").cast(TimestampType()),
    F.col("_silver_processed_timestamp").cast(TimestampType()),
    F.col("_data_quality_score").cast(DoubleType()),
    F.col("_transformation_version").cast(StringType()),
    F.col("_source").cast(StringType())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Silver Table

# COMMAND ----------

# Write to Silver table
silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(silver_table)

print(f"✅ Silver table created: {silver_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Table Metadata

# COMMAND ----------

# Add table comment
spark.sql(f"""
    COMMENT ON TABLE {silver_table} IS 
    'Silver layer: Cleaned and standardized Iris dataset. Contains validated measurements, z-score normalized features, and derived calculations (areas, ratios). Data quality verified.'
""")

# Add column comments
comments = {
    "sepal_length_cm": "Sepal length in centimeters (validated)",
    "sepal_width_cm": "Sepal width in centimeters (validated)",
    "petal_length_cm": "Petal length in centimeters (validated)",
    "petal_width_cm": "Petal width in centimeters (validated)",
    "sepal_length_cm_standardized": "Z-score normalized sepal length",
    "sepal_width_cm_standardized": "Z-score normalized sepal width",
    "petal_length_cm_standardized": "Z-score normalized petal length",
    "petal_width_cm_standardized": "Z-score normalized petal width",
    "sepal_area_cm2": "Calculated sepal area (length × width) in cm²",
    "petal_area_cm2": "Calculated petal area (length × width) in cm²",
    "sepal_petal_length_ratio": "Ratio of sepal length to petal length",
    "sepal_petal_width_ratio": "Ratio of sepal width to petal width",
    "target": "Numeric species identifier (0=setosa, 1=versicolor, 2=virginica)",
    "species": "Species name: setosa, versicolor, or virginica",
    "_ingestion_timestamp": "Original ingestion timestamp from Bronze layer",
    "_silver_processed_timestamp": "Timestamp when data was processed into Silver layer",
    "_data_quality_score": "Data quality score (1.0 = passed all checks)",
    "_transformation_version": "Version of the transformation logic applied",
    "_source": "Original data source identifier"
}

for col_name, comment in comments.items():
    try:
        spark.sql(f"ALTER TABLE {silver_table} ALTER COLUMN {col_name} COMMENT '{comment}'")
    except Exception as e:
        print(f"Warning: Could not add comment for {col_name}: {e}")

print("✅ Table metadata added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Silver Table

# COMMAND ----------

# Show table info
print(f"📋 Table: {silver_table}")
print(f"📊 Record count: {spark.table(silver_table).count()}")
print("\n📝 Schema:")
spark.table(silver_table).printSchema()

# Display sample data
display(spark.table(silver_table).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Status

# COMMAND ----------

# Return success status
dbutils.notebook.exit("SUCCESS: Silver layer transformation complete")
