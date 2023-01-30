import pandas as pd
from sklearn.model_selection import train_test_split


def digitize(n):
    if n > 0:
        return 1
    return 0


def acquire_training_data():
    df = pd.read_csv('/dbfs/data/raw.csv')

    return df


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