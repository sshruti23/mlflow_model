import warnings
import numpy as np
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
from platform import python_version
import mlflow.sklearn
from pyspark.sql import SparkSession



EXPERIMENT_ID = "1663633462792152"


def rolling_window(a, window):
    """
    Takes np.array 'a' and size 'window' as parameters
    Outputs an np.array with all the ordered sequences of values of 'a' of size 'window'
    e.g. Input: ( np.array([1, 2, 3, 4, 5, 6]), 4 )
         Output:
                 array([[1, 2, 3, 4],
                       [2, 3, 4, 5],
                       [3, 4, 5, 6]])
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def digitize(n):
    if n > 0:
        return 1
    return 0


def prepare_training_data(data):
    """
    Return a prepared numpy dataframe
    input : Dataframe with expected schema

    """
    data["Delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["Delta"].apply(lambda d: digitize(d))
    return data


def pull_data():
    spark = SparkSession.builder.appName("read_csv_using_spark").enableHiveSupport().getOrCreate()
    import os
    from os import listdir
    from os.path import isfile, join
    print("--------")
    cwd = os.getcwd()
    onlyfiles = [os.path.join(cwd, f) for f in os.listdir(cwd) if
                 os.path.isfile(os.path.join(cwd, f))]
    print(onlyfiles)
    print("--------")
    input_df = (
        spark.read.option("header", True)
        .option("inferschema", True)
        .csv(f"./data/inference.csv")
    )
    btc_mat = input_df.toPandas().to_numpy()

    WINDOW_SIZE = 14

    X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
    Y = input_df.toPandas()["to_predict"].to_numpy()[WINDOW_SIZE:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=4284, stratify=Y
    )
    return X_test, y_test


def acquire_training_data():
    yf.pdr_override()
    y_symbols = ["BTC-USD"]

    startdate = datetime.datetime(2022, 1, 1)
    enddate = datetime.datetime(2022, 12, 31)
    df = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)
    return df


def inference(model_name, model_version):
    X_test, y_test = pull_data()
    rf_model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    return rf_model.predict(X_test)


def find_best_run(metric: str = "training_f1_score"):
    client = mlflow.MlflowClient()
    experiment_runs = client.search_runs(experiment_ids=EXPERIMENT_ID)
    best_run_id = None
    best_artifact_uri = None
    best_metric_score = None
    for run in experiment_runs:
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


def register_model(model_name, rf_uri):

    model_uri = rf_uri + "/" + model_name
    new_model_version = mlflow.register_model(model_uri, model_name)
    return new_model_version


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    n_estimators = [50]
    criterion = ["gini"]
    min_weight_fraction_leaf = [0.0]

    for n_est in n_estimators:
        for crit in criterion:
            for mwfl in min_weight_fraction_leaf:
                print("*" * 100)
                model_name = "random_forest_model_{}_{}_{}".format(n_est, crit, mwfl)
                print(
                    "Triggering training with params :: estimators : {}, criterion : {} and min_weight_fraction : {}".format(
                        n_est, crit, mwfl
                    )
                )
                with mlflow.start_run(
                    run_name=f"stock_estimator_{n_est}_{crit}_{mwfl}"
                ) as run:
                    training_data = acquire_training_data()

                    mlflow.sklearn.autolog()

                    prepared_training_data_df = prepare_training_data(training_data)

                    btc_mat = prepared_training_data_df.to_numpy()

                    WINDOW_SIZE = 14

                    X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
                    Y = prepared_training_data_df["to_predict"].to_numpy()[WINDOW_SIZE:]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, Y, test_size=0.25, random_state=4284, stratify=Y
                    )

                    # clf is my Model
                    clf = RandomForestClassifier(
                        bootstrap=True,
                        criterion=crit,
                        min_samples_split=2,
                        min_weight_fraction_leaf=mwfl,
                        n_estimators=n_est,
                        random_state=4284,
                        verbose=0,
                    )

                    # training
                    clf.fit(X_train, y_train)

                    print(" --- Model Predict ---- ")
                    # inference
                    predicted = clf.predict(X_test)
                    print(classification_report(y_test, predicted))
                    rf_uri = run.info.artifact_uri
    # TODO : Experiment Name needs to be passed to Best Run function
    best_run_id, best_artifact_uri, best_metric_score = find_best_run()
    model_name = "model"
    model_version = register_model(model_name=model_name, rf_uri=best_artifact_uri)
    print(f"model registeration complete model:{model_version.version}")

    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=model_version.version, stage="STAGING"
    )
    client.transition_model_version_stage(
        name=model_name, version=model_version.version, stage="PRODUCTION"
    )
    print(" --- Model Inference ---- ")
    print(f"\ntriggering inference with {model_name}:{model_version.version}")

    predictions = inference(model_name, model_version.version)
    print(predictions, "\n")
    (
        best_run_id_met_acc,
        best_artifact_uri_met_acc,
        best_metric_score_met_acc,
    ) = find_best_run(metric="training_roc_auc")
    model_version = register_model(
        model_name=model_name, rf_uri=best_artifact_uri_met_acc
    )
    print(f"model registeration complete {model_name}:{model_version.version}")
    print(" --- Model Inference ---- ")
    print(f"\ntriggering inference with {model_name}:{model_version.version}")
    X_test, y_test = pull_data()
    loaded_model = mlflow.pyfunc.load_model(
        f"models:/{model_name}/{model_version.version}"
    )
    print("Predict X_test")
    print(X_test)
    print("Predict Y_test")
    print(y_test)
    loaded_model.predict(X_test)
    print(predictions, "\n")


