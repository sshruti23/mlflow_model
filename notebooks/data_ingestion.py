# Databricks notebook source
# MAGIC %md #Raw Data Ingestion Notebook
# MAGIC 
# MAGIC ###Notebook Description :
# MAGIC ###- Download yahoo finance data from web
# MAGIC ###- Store df as delta table

# COMMAND ----------

# MAGIC %md #Import

# COMMAND ----------

# MAGIC %md ###Pypi Packages

# COMMAND ----------

# MAGIC %pip install pandas_datareader
# MAGIC %pip install yfinance

# COMMAND ----------

# MAGIC %md ###Python Libraries

# COMMAND ----------

import datetime

from pandas_datareader import data as pdr
import yfinance as yf


# COMMAND ----------

# MAGIC %md #Declare Constants

# COMMAND ----------

raw_ingestion_data_path="dbfs:/data/raw.csv"
raw_delta_lake_path="dbfs:/stockpred_delta_lake/"

# COMMAND ----------

# MAGIC %md #Create Delta Lake

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
    raw_df = spark.read.option("header", True).csv(raw_ingestion_data_path)
    # columns needs to be renamed because delta table cannot store columns with special character
    transformed_df = raw_df.withColumnRenamed("Adj Close", "Adj_Close")
    transformed_df.write.format("delta").mode("overwrite").save(raw_delta_lake_path)


# COMMAND ----------

print("Running from feature store branch")
source_df = download_yfinance_data()
create_raw_delta_lake()
