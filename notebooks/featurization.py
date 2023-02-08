# Databricks notebook source
# MAGIC %run ../src/data_preparation

# COMMAND ----------

# MAGIC %run ../src/feature_store

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


def read_raw_delta_table():
    src_delta_table = DeltaTable.forPath(spark, "dbfs:/stockpred_delta_lake/")
    return src_delta_table


# COMMAND ----------


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


def add_primary_key(training_data, id_column_name):
    """Add id column to dataframe"""
    columns = training_data.columns
    new_df = training_data.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]


# COMMAND ----------


def create_df_with_label(df):
    spark_df_with_label = spark.createDataFrame(df)
    df_with_primary_key = add_primary_key(spark_df_with_label, "row_id")
    return df_with_primary_key


# COMMAND ----------

training_data = read_raw_delta_table()  # source delta_table
transformed_df = change_column_datatype(training_data.toDF())
raw_df = prepare_training_data(transformed_df.toPandas())
btc_mat = raw_df.to_numpy()
WINDOW_SIZE = 14
display(btc_mat)
X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
Y = raw_df["to_predict"].to_numpy()[WINDOW_SIZE:]

complete_df_with_label = prepare_data(X, Y)

# COMMAND ----------

df_with_label = create_df_with_label(complete_df_with_label)

# COMMAND ----------

# creating feature df
display(training_data.toDF())

# COMMAND ----------

display(complete_df_with_label)

# COMMAND ----------

# Calls feature_store notebook
create_feature_store(df_with_label.drop("to_predict"))

# COMMAND ----------

# write this df as delta , getting used by train
inference_data_df = df_with_label.select("row_id", "to_predict")
inference_data_df.write.format("delta").mode("overwrite").save(
    "dbfs:/inference_data_df"
)
display(inference_data_df)
