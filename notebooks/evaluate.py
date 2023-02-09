# Databricks notebook source
# MAGIC %pip install databricks-feature-store

# COMMAND ----------

from sklearn.metrics import classification_report
import pandas as pd
from delta.tables import *

import mlflow.sklearn
import mlflow
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup


# COMMAND ----------

client = mlflow.MlflowClient()


# COMMAND ----------


def find_best_run(metric: str = "training_f1_score"):
    experiment_runs = client.search_runs(experiment_ids=["1335986235541854"])
    best_run_id = None
    best_artifact_uri = None
    best_metric_score = None
    for run in experiment_runs:
        if not metric in run.data.metrics:
            raise Exception("Bad metric passed for evaluation.")
        else:
            metrics = run.data.metrics
            if best_run_id is None:               
                best_run_id = run.info.run_id
                best_artifact_uri = run.info.artifact_uri
                best_metric_score = metrics[metric]
            else:
                if metrics[metric] >= best_metric_score:
                    best_run_id = run.info.run_id
                    best_artifact_uri = run.info.artifact_uri
                    best_metric_score = metrics[metric]
    return best_run_id, best_artifact_uri, best_metric_score 


# COMMAND ----------

# default.training_set
dlt_table_name = "default.training_set"
dlt_table = DeltaTable.forName(spark, dlt_table_name)
test_data = dlt_table.toDF().toPandas()
y_test = test_data["to_predict"]
X_test = test_data.drop("to_predict", axis=1)

best_run_id, best_artifact_uri, best_metric_score = find_best_run()
print(best_artifact_uri)
print(best_run_id)



# COMMAND ----------


run_id = best_run_id    
filter_string = "run_id='{}'".format(run_id)
print(filter_string)
results = client.search_model_versions(filter_string)
version=results[0].version
print(version)

# COMMAND ----------

def read_inference_delta_table():
    inference_df = DeltaTable.forPath(spark, "dbfs:/inference_data_df")
    return inference_df.toDF()

# COMMAND ----------

inference_data_df = read_inference_delta_table()
fs = feature_store.FeatureStoreClient()


# COMMAND ----------

## For simplicity, this example uses inference_data_df as input data for prediction
batch_input_df = inference_data_df.drop("to_predict") # Drop the label column
display(batch_input_df)
predictions_df = fs.score_batch(f"models:/stockpred_model/{version}", batch_input_df)
display(predictions_df["row_id","prediction"])

# COMMAND ----------

y_predict = predictions_df.select("prediction").toPandas()

# COMMAND ----------

display(y_predict)

# COMMAND ----------

print(classification_report(y_test, y_predict))

# COMMAND ----------

dbutils.jobs.taskValues.set(key="best_model_version", value=version)
