from pandas import DataFrame, Series


def create_dataset_filter(dataset: DataFrame) -> Series(bool):
    return (
                   (dataset["fare_amount"] > 0)
                   & (dataset["trip_distance"] < 400)
                   & (dataset["trip_distance"] > 0)
                   & (dataset["fare_amount"] < 1000)
           ) | (~dataset.isna().any(axis=1))
