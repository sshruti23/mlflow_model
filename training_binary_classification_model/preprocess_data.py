import numpy as np

from training_binary_classification_model.data_preparation import acquire_training_data, prepare_training_data, prepare_data


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


if __name__ == "__main__":
    training_data = acquire_training_data()
    prepared_training_data_df = prepare_training_data(training_data)
    btc_mat = prepared_training_data_df.to_numpy()
    WINDOW_SIZE = 14
    X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
    Y = prepared_training_data_df["to_predict"].to_numpy()[WINDOW_SIZE:]
    train_data, test_data = prepare_data(X, Y)

    spark.createDataFrame(train_data).write.format("delta").mode("overwrite").saveAsTable("default.train")
    spark.createDataFrame(test_data).write.format("delta").mode("overwrite").saveAsTable("default.test")
