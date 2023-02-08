# Databricks notebook source
# MAGIC %run ./data_featurization

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from delta.tables import *


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
    df = pd.concat([X, Y] , axis=1)
    return df