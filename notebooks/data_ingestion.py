# Databricks notebook source
# MAGIC %pip install pandas_datareader
# MAGIC %pip install yfinance

# COMMAND ----------

import datetime

from pandas_datareader import data as pdr
import yfinance as yf


# COMMAND ----------


def download_yfinance_data():
    yf.pdr_override()
    y_symbols = ["BTC-USD"]

    startdate = datetime.datetime(2022, 1, 1)
    enddate = datetime.datetime(2022, 12, 31)
    df = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)
    df.to_csv("/dbfs/data/raw.csv", mode="w", header=True)
    return df


# COMMAND ----------


def create_raw_delta_lake():
    raw_df = spark.read.option("header", True).csv("dbfs:/data/raw.csv")
    transformed_df = raw_df.withColumnRenamed("Adj Close", "Adj_Close")
    transformed_df.write.format("delta").mode("overwrite").save(
        "dbfs:/stockpred_delta_lake/"
    )


# COMMAND ----------

print("Running from feature store branch")
source_df = download_yfinance_data()
create_raw_delta_lake()
