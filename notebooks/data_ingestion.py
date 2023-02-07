# Databricks notebook source
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
    return df


# COMMAND ----------

def create_raw_delta_lake(raw_df):
    raw_df.write.format("delta").mode("overwrite").save("dbfs:/stockpred_raw_delta_lake/")


# COMMAND ----------


source_df = download_yfinance_data()
create_raw_delta_lake(raw_df=source_df)
