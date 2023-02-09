# Databricks notebook source
import mlflow
import mlflow.sklearn

# COMMAND ----------

version= dbutils.jobs.taskValues.get(taskKey = "Evaluate", key = "version", default = None, debugValue = 0)
print(f"best model version {version}")


# COMMAND ----------

client = mlflow.MlflowClient()

# COMMAND ----------

print("promote best model version to Staging")

# COMMAND ----------

client.transition_model_version_stage(
    name="stockpred_model", version=version, stage="STAGING"
)
