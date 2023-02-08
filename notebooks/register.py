# Databricks notebook source
def register_model(model_name, model_uri):

    model_uri = model_uri + "/" + model_name
    new_model_version = mlflow.register_model(model_uri, model_name)
    return new_model_version


# COMMAND ----------

best_artifact_uri_met_acc= table_name= dbutils.jobs.taskValues.get(taskKey = "Evaluate", key = "best_artifact_uri", default = None, debugValue = 0)

register_model(
        model_name="stockpred_model", model_uri=best_artifact_uri_met_acc
    )
