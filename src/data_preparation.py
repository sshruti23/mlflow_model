# Databricks notebook source
import pandas as pd

# COMMAND ----------

def digitize(n):
    if n > 0:
        return 1
    return 0

# COMMAND ----------

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

def prepare_data(X, Y):
    X = pd.DataFrame(X)
    X.columns = ["day_" + str(i) for i in range(14)]
    Y = pd.DataFrame(Y)
    Y.columns = ["to_predict"]
    print(X.size)
    print(Y.size)
    df = pd.concat([X, Y], axis=1)
    return df
