# Databricks notebook source
# MAGIC %md #Featurization Notebook
# MAGIC ###Short Description:
# MAGIC ###- read raw data stored with databricks file store system(dbfs)
# MAGIC ###- preprocess raw data
# MAGIC ###- creates features table
# MAGIC ###- stores feature in databricks feature store

# COMMAND ----------

# MAGIC %md ##Imports

# COMMAND ----------

# MAGIC %md ###Supporting/helper notebooks

# COMMAND ----------

# MAGIC %run ../src/data_preparation
# MAGIC %run ../src/feature_store

# COMMAND ----------

# MAGIC %md ###python libraries

# COMMAND ----------

import numpy as np

from pyspark.sql.functions import monotonically_increasing_id

from pyspark.sql.types import DoubleType, DateType, LongType
import uuid
from mlflow.tracking.client import MlflowClient

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow.sklearn

# COMMAND ----------

# MAGIC %md ##Declaring constants

# COMMAND ----------

WINDOW_SIZE = 14

# COMMAND ----------

# MAGIC %md ##Prepare raw data for preprocessing

# COMMAND ----------

def read_raw_delta_table():
    src_delta_table = DeltaTable.forPath(spark, "dbfs:/stockpred_delta_lake/")
    return src_delta_table

def add_primary_key(training_data, id_column_name):
    """Add id column to dataframe"""
    columns = training_data.columns
    new_df = training_data.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def change_column_datatype(df):
    output_df = (
        df.withColumn("Close", df["Close"].cast(DoubleType()))
        .withColumn("Open", df["Open"].cast(DoubleType()))
        .withColumn("Date", df["Date"].cast(DateType()))
        .withColumn("High", df["High"].cast(DoubleType()))
        .withColumn("Low", df["Low"].cast(DoubleType()))
        .withColumn("Adj_Close", df["Adj_Close"].cast(LongType()))
        .withColumn("Volume", df["Volume"].cast(DoubleType()))
    )
    return output_df

# COMMAND ----------

training_data = read_raw_delta_table()  # source delta_table
transformed_df = change_column_datatype(training_data.toDF())
raw_df = prepare_training_data(transformed_df.toPandas())
btc_mat = raw_df.to_numpy()
display(btc_mat)

# COMMAND ----------

# MAGIC %md ##Transform raw df as per window duration defined

# COMMAND ----------

def rolling_window(a, window):
    """
    Takes np.array 'a' and size 'window' as parameters
    Outputs an np.array with all the ordered sequences of values of 'a' of size 'window'
    e.g. Input: ( np.array([1, 2, 3, 4, 5, 6]), 4 )
         Output:
                 array([[1, 2, 3, 4],
                       [2, 3, 4, 5],
                       [3, 4, 5, 6]])
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# COMMAND ----------

X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
Y = raw_df["to_predict"].to_numpy()[WINDOW_SIZE:]
complete_df_with_label = prepare_data(X, Y)
display(complete_df_with_label)

# COMMAND ----------

# MAGIC %md ##Feature Store

# COMMAND ----------

# MAGIC %md ###Create df for feature store

# COMMAND ----------

def create_df_with_label(df):
    spark_df_with_label = spark.createDataFrame(df)
    df_with_primary_key = add_primary_key(spark_df_with_label, "row_id")
    return df_with_primary_key
  
df_with_label = create_df_with_label(complete_df_with_label)


# COMMAND ----------

display(training_data.toDF())

# COMMAND ----------

display(complete_df_with_label)

# COMMAND ----------

# MAGIC %md ###Calling feature_store notebook to create feature table and store df in the feature store

# COMMAND ----------

create_feature_store(df_with_label.drop("to_predict"))

# COMMAND ----------

# MAGIC %md ###Create inference df , refernced by feature lookup to create training set df
# MAGIC inference df is stored as delta for easy refernce by training block

# COMMAND ----------

# write this df as delta , getting used by train
inference_data_df = df_with_label.select("row_id", "to_predict")
inference_data_df.write.format("delta").mode("overwrite").save(
    "dbfs:/inference_data_df"
)
display(inference_data_df)
