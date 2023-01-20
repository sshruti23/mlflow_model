import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
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


class AutogluonModel(mlflow.pyfunc.PythonModel):
    """
        Creates a TabularPredictor Autogluon Model
        input : python model
    """
    def load_context(self, context):
        self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))

    def predict(self, context, model_input):
        return self.predictor.predict(model_input)


def log_model():
    model = AutogluonModel()
    predictor_path = predictor.path + "models/" + predictor.get_model_best()
    artifacts = {"predictor_path": predictor_path}
    conda_env = {
        "channels": ["conda-forge"],
        "dependencies": [f"python={python_version()}", "pip"],
        "pip": [f"mlflow=={mlflow.__version__}", f'cloudpickle=="2.2.0"'],
        "name": "mlflow-env",
    }
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        artifacts=artifacts,
        conda_env=conda_env,
    )


def log_experiments(predictor):
    for i, model_name in enumerate(list(predictor.leaderboard(silent=True)["model"])):
        with mlflow.start_run(run_name=model_name):
            if i == 0:
                log_model()
            info = predictor.info()["model_info"][model_name]
            score = info["val_score"]
            model_type = info["model_type"]
            hyper_params = info["hyperparameters"]
            hyper_params["model_type"] = model_type
            mlflow.log_params(hyper_params)
            mlflow.log_metric("acc", score)


def create_autogluon_experiment(train_df):
    predictor = TabularPredictor(label="to_predict", eval_metric="accuracy").fit(
        train_data=train_df, verbosity=2, presets="medium_quality"
    )
    return predictor


def prepare_data(X, Y):
    X = pd.DataFrame(X)
    X.columns = ["day_" + str(i) for i in range(14)]
    Y = pd.DataFrame(Y)
    Y.columns = ["to_predict"]
    df = pd.concat([X, Y], axis=1)

    train_data, test_data = train_test_split(df, test_size=0.25, random_state=4284)
    return train_data, test_data


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


def acquire_training_data():
    yf.pdr_override()
    y_symbols = ["BTC-USD"]

    startdate = datetime.datetime(2022, 1, 1)
    enddate = datetime.datetime(2022, 12, 31)
    df = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)
    return df


if __name__ == "__main__":
    # data preparation
    training_data = acquire_training_data()
    prepared_training_data_df = prepare_training_data(training_data)
    btc_mat = prepared_training_data_df.to_numpy()
    WINDOW_SIZE = 14
    X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
    Y = prepared_training_data_df["to_predict"].to_numpy()[WINDOW_SIZE:]
    train_data, test_data = prepare_data(X, Y)

    # AutoML model selection
    predictor = create_autogluon_experiment(train_data)

    # Logging of AutoML experiments
    log_experiments(predictor)

    predictor.leaderboard()


