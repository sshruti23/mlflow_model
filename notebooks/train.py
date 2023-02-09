# Databricks notebook source
# MAGIC %md #Training Model
# MAGIC ###- Model uses random forest classifier for training
# MAGIC ###- Read featurerized data , inference data and log model using fs.log_model
# MAGIC ###- Write training set as delta table to be used by evaluate

# COMMAND ----------

# MAGIC %md ##Imports

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow.sklearn
from delta.tables import *
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from mlflow.tracking.client import MlflowClient
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md ##Delta Lake Functions

# COMMAND ----------

def create_delta_table_for_df(table_name, df):
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    
def read_inference_delta_table():
    inference_df = DeltaTable.forPath(spark, "dbfs:/inference_data_df")
    return inference_df.toDF()

# COMMAND ----------

# MAGIC %md ##Training Functions

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
    inference_data_df = read_inference_delta_table()
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="to_predict", exclude_columns="row_id")
    training_pd = training_set.load_df().toPandas()
    X = training_pd.drop("to_predict", axis=1)
    y = training_pd["to_predict"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

# COMMAND ----------

def train_model(X_train, X_test, y_train, y_test, training_set, fs):
    ## fit and log model
    n_estimators=[50, 100, 200, 500, 1000]
    mlflow.set_experiment(experiment_id="1335986235541854")
    for n_est in n_estimators:
        with mlflow.start_run(run_name=f"stock_estimator_{n_est}") as run:
            rf = RandomForestClassifier(bootstrap=True,
                    criterion='gini',
                    min_samples_split=2,
                    min_weight_fraction_leaf=0.1,
                    n_estimators=n_est,
                    random_state=42,
                    verbose=0)
            
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            fs.log_model(
                model=rf,
                artifact_path="binary_classification_stock_prediction",
                flavor=mlflow.sklearn,
                training_set=training_set,
                registered_model_name="stockpred_model",
            )

# COMMAND ----------

# MAGIC %md ##Mlflow Client 

# COMMAND ----------

client = MlflowClient()

try:
    client.delete_registered_model("stockpred_model") # Delete the model if already created
except:
    None

    
# Disable MLflow autologging and instead log the model using Feature Store
mlflow.sklearn.autolog(log_models=False)

# COMMAND ----------

# MAGIC %md ##Training Model

# COMMAND ----------

table_name= dbutils.jobs.taskValues.get(taskKey = "Featurization", key = "fs_table_name", default = "stockpred_dbebfee8", debugValue = 0)
fs = feature_store.FeatureStoreClient()
print(table_name)
X_train, X_test, y_train, y_test, training_set = load_data(table_name, "row_id")
train_model(X_train, X_test, y_train, y_test, training_set, fs)

# COMMAND ----------

display(training_set.load_df())
create_delta_table_for_df("default.training_set",training_set.load_df())
