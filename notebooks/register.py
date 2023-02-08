# Databricks notebook source
import mlflow
import mlflow.sklearn

# COMMAND ----------


def register_model(model_name, model_uri):
    model_uri = model_uri + "/" + model_name
    new_model_version = mlflow.register_model(model_uri, model_name)
    return new_model_version


# COMMAND ----------

best_artifact_uri_met_acc = dbutils.jobs.taskValues.get(
    taskKey="Evaluate", key="best_artifact_uri", default=None, debugValue=0
)
print(best_artifact_uri_met_acc)

register_model("stockpred_model", best_artifact_uri_met_acc)
