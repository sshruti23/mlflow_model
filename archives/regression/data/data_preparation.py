# Databricks notebook source
# MAGIC %md #Helper Notebook - Data Preparation
# MAGIC ###- Adds the label `to_predict` to the raw dataframe

# COMMAND ----------

# MAGIC %md ##Imports

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md ##Prepare `to_predict` column based on delta data

# COMMAND ----------

def digitize(n):
    if n > 0:
        return 1
    return 0

def prepare_training_data(data):
    """
    Return a prepared numpy dataframe
    input : Dataframe with expected schema

    """
    print("prepare_training_data")
    data["delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["delta"].apply(lambda d: digitize(d))
    return data

# COMMAND ----------

# MAGIC %md ##Transforms the raw df into 14 days data along with to_predict column

# COMMAND ----------

def prepare_data(X, Y):
    X = pd.DataFrame(X)
    X.columns = ["day_" + str(i) for i in range(14)]
    Y = pd.DataFrame(Y)
    Y.columns = ["to_predict"]
    print(X.size)
    print(Y.size)
    df = pd.concat([X, Y], axis=1)
    return df
