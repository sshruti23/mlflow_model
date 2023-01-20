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

def acquire_training_data():
    df=pd.read_csv('/dbfs/data/raw.csv')

    return df


def digitize(n):
    if n > 0:
        return 1
    return 0


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


def prepare_training_data(data):
    """
    Return a prepared numpy dataframe
    input : Dataframe with expected schema

    """
    data["Delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["Delta"].apply(lambda d: digitize(d))
    return data



def prepare_data(X, Y):
    X = pd.DataFrame(X)
    X.columns = ["day_" + str(i) for i in range(14)]
    Y = pd.DataFrame(Y)
    Y.columns = ["to_predict"]
    df = pd.concat([X, Y], axis=1)

    train_data, test_data = train_test_split(df, test_size=0.25, random_state=4284)
    return train_data, test_data

if __name__ == "__main__":

    training_data = acquire_training_data()
    prepared_training_data_df = prepare_training_data(training_data)
    btc_mat = prepared_training_data_df.to_numpy()
    WINDOW_SIZE = 14
    X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
    Y = prepared_training_data_df["to_predict"].to_numpy()[WINDOW_SIZE:]
    train_data, test_data = prepare_data(X, Y)
    train_data.to_csv('/dbfs/data/train_data.csv',index=False)
    test_data.to_csv('/dbfs/data/test_data.csv',index=False)
