# Databricks notebook source
import mlflow
import mlflow.sklearn

# COMMAND ----------

version= dbutils.jobs.taskValues.get(taskKey = "Evaluate", key = "best_model_version", default = None, debugValue = 0)
registered_model_name=dbutils.jobs.taskValues.get(taskKey = "Evaluate", key = "registred_model_name", default = None, debugValue = 0)
best_model_version=int(version)
print(best_model_version)
print(f"registered_model_name : {registered_model_name}")
print(type(best_model_version))
print(f"best model version {best_model_version}")

# COMMAND ----------

client = mlflow.MlflowClient()

# COMMAND ----------

print("promote best model version to Staging")

# COMMAND ----------

registered_stages=client.get_latest_versions(f"{registered_model_name}",["Staging"])
registered_version= int(registered_stages[0].version) if registered_stages != [] else -1
print(f"best model version {best_model_version}")
print(f"registered model version {registered_version}")

if  registered_version==-1 or registered_version != best_model_version:
    client.transition_model_version_stage(
        name=f"{registered_model_name}", version=best_model_version, stage="Staging"
    )
    print(f"version {best_model_version} moved to STAGING")     
else:
    print("Model is already registered as Staging.")
    print("Model Promotion Completed!!")   
