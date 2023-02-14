# Databricks notebook source
import mlflow
import mlflow.sklearn

# COMMAND ----------

version= dbutils.jobs.taskValues.get(taskKey = "Evaluate", key = "best_model_version", default = None, debugValue = 0)
print(f"best model version {version}")


# COMMAND ----------

client = mlflow.MlflowClient()

# COMMAND ----------

print("promote best model version to Staging")

# COMMAND ----------

registered_stages=client.get_latest_versions("stockpred_model",["Staging"])
registered_version=int(registered_stages[0].version)
if registered_version is not version:
    client.transition_model_version_stage(
        name="stockpred_model", version=version, stage="Staging"
    )
    print(f"version {version} moved to STAGING")

print("Model is already registered as Staging.")
print("Model Promotion Completed!!")
