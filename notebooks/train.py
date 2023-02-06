# Databricks notebook source
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow.sklearn
from delta.tables import *

# COMMAND ----------

def create_and_log_experiment(X_train, y_train, n_estimators, dlt_table_name, dlt_table_version):
    mlflow.set_experiment(experiment_id="196699694392376")
    for n_est in n_estimators:
        with mlflow.start_run(run_name=f"stock_estimator_{n_est}") as run:
            mlflow.sklearn.autolog()
            clf = RandomForestClassifier(
                bootstrap=True,
                criterion='gini',
                min_samples_split=2,
                min_weight_fraction_leaf=0.1,
                n_estimators=n_est,
                random_state=4284,
                verbose=0)

            clf.fit(X_train, y_train)
            mlflow.log_param("train_table", dlt_table_name)
            mlflow.log_param("train_table_version", dlt_table_version)

# COMMAND ----------

dlt_table_name = "default.train"
dlt_table = DeltaTable.forName(spark, dlt_table_name)
dlt_table_version = dlt_table.history().head(1)[0].version
train_data = dlt_table.toDF().toPandas()
y_train = train_data['to_predict']
X_train = train_data.drop('to_predict', axis=1)
create_and_log_experiment(X_train, y_train, [50, 100, 200, 500, 1000], dlt_table_name, dlt_table_version)