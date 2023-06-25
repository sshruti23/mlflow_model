# Databricks notebook source
# MAGIC %md #Helper Notebook - Feature Store

# COMMAND ----------

# MAGIC %md ##Imports

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
import uuid

# COMMAND ----------

# MAGIC %md ##Create Feature Store Client

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md ##Create Feature store database and store df as feature table

# COMMAND ----------

def create_feature_store_database():
    spark.sql(f"CREATE DATABASE IF NOT EXISTS stockpred_db")

    # Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
    global table_name
    unique_table_name = f"stockpred_db" + str(uuid.uuid4())[:6]
    table_name=f"hive_metastore.stockpred_db.{unique_table_name}"
    print(table_name)

def create_feature_store(feature_store_df):
    display(feature_store_df.limit(20))
    create_feature_store_database()
    fs.create_table(
        name=f"{table_name}",
        primary_keys=["row_id"],
        df=feature_store_df,
        schema=feature_store_df.schema,
        description="stockpred features",
    )
    dbutils.jobs.taskValues.set(key="fs_table_name", value=table_name)
