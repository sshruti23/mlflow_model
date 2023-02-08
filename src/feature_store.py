# Databricks notebook source
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

# COMMAND ----------

 fs = feature_store.FeatureStoreClient()

# COMMAND ----------

def add_primary_key(training_data , id_column_name):
    """Add id column to dataframe"""
    columns = training_data.columns
    new_df = training_data.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

# COMMAND ----------

def create_df_with_label():
    spark_df_with_label=spark.createDataFrame(complete_df_with_label)
    df_with_primary_key=add_primary_key(spark_df_with_label ,'row_id')
    return df_with_primary_key

# COMMAND ----------

def create_feature_store_database():
    spark.sql(f"CREATE DATABASE IF NOT EXISTS stockpred_db")

    # Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
    global table_name
    table_name = f"stockpred_db" + str(uuid.uuid4())[:6]
    print(table_name)

# COMMAND ----------

def create_feature_store(feature_store_df):
    display(feature_store_df)
    create_feature_store_database()
    fs.create_table(
    name=table_name,
    primary_keys=["row_id"],
    df=feature_store_df,
    schema=feature_store_df.schema,
    description="stockpred features"
    )
    dbutils.jobs.taskValues.set(key = 'fs_table_name', value = table_name)
