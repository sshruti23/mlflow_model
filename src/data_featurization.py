# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split

# COMMAND ----------

def create_feature_table(fs, table_name, features_df):
    fs.create_table(
        name=table_name,
        primary_keys=[""],
        df=features_df,
        schema=features_df.schema,
        description="wine features"
    )
    pass

# COMMAND ----------


def create_feature_store():
    fs = feature_store.FeatureStoreClient()
    table_name = create_database()
    create_feature_table(fs)
    pass

# COMMAND ----------

def create_database():
    spark.sql(f"CREATE DATABASE IF NOT EXISTS stockpred_db")
    # Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
    table_name = f"stockpred_db" + str(uuid.uuid4())[:6]
    print(table_name)
    return table_name
