from sklearn.metrics import classification_report
from delta.tables import *

import mlflow.sklearn


def find_best_run(metric: str = "training_f1_score"):
    client = mlflow.MlflowClient()
    experiment_runs = client.search_runs(experiment_ids=["<mlflow_experiment_id>"])
    best_run_id = None
    best_artifact_uri = None
    best_metric_score = None
    for run in experiment_runs:
        print(run.info)
        print("Run data metrics :" + str(run.data.metrics))
        if not metric in run.data.metrics:
            print(metric)
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


if __name__ == "__main__":
    dlt_table_name = "default.test"
    dlt_table = DeltaTable.forName(spark, dlt_table_name)
    test_data = dlt_table.toDF().toPandas()
    y_test = test_data['to_predict']
    X_test = test_data.drop('to_predict', axis=1)

    best_run_id, best_artifact_uri, best_metric_score = find_best_run()
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    y_predict = best_model.predict(X_test)

    print(classification_report(y_test, y_predict))
