import pandas as pd
from sklearn.model_selection import train_test_split


def digitize(n):
    if n > 0:
        return 1
    return 0


def acquire_training_data():
    df = pd.read_csv('/dbfs/request_body/raw.csv')
    print("acquire_training_data")
    return df


def prepare_training_data(data):
    """
    Return a prepared numpy dataframe
    input : Dataframe with expected schema

    """
    print("prepare_training_data")
    data["Delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["Delta"].apply(lambda d: digitize(d))
    return data


def prepare_data(X, Y):
    print("prepare_data")
    X = pd.DataFrame(X)
    X.columns = ["day_" + str(i) for i in range(14)]
    Y = pd.DataFrame(Y)
    Y.columns = ["to_predict"]
    df = pd.concat([X, Y], axis=1)
    # df* convert to spark df ---> add primary key --> create fs


    # remove this line
    train_data, test_data = train_test_split(df, test_size=0.25, random_state=4284)
    return train_data, test_data


# fs --> day 1 - day 14 | -- to_predict | ++ primary key
# raw_df --> yahoo data
#     created features
#     dumped periodiaclly
#     cadence
#
# inference_df -> delta
#
# re train in 30 days
# - fs features
# - lables
# - pull fs lookup  & pull delta inference df & merge and create a training df



