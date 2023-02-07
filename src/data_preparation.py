# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split
from delta.tables import *


# COMMAND ----------

def digitize(n):
    if n > 0:
        return 1
    return 0


# COMMAND ----------

def acquire_training_data():
    df = pd.read_csv('/dbfs/data/raw.csv')
    print("acquire_training_data")
    return df


def read_raw_delta_table():
    src_delta_table = DeltaTable.forPath(spark, "dbfs:/stockpred_raw_delta_lake/")
    return src_delta_table


# COMMAND ----------

def prepare_training_data(data):
    """
    Return a prepared numpy dataframe
    input : Dataframe with expected schema

    """
    print("prepare_training_data")
    data["Delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["Delta"].apply(lambda d: digitize(d))
    return data


# COMMAND ----------


def prepare_data(X, Y):
    print("prepare_data")
    X = pd.DataFrame(X)
    X.columns = ["day_" + str(i) for i in range(14)]
    Y = pd.DataFrame(Y)
    Y.columns = ["to_predict"]
    df = pd.concat([X, Y], axis=1)
    display(df)

    train_data, test_data = train_test_split(df, test_size=0.25, random_state=4284)
    return train_data, test_data
