import logging

from pandas import DataFrame


def load_file_as_dataframe(file_path: str, file_format: str) -> DataFrame:
    if file_format == "csv":
        import pandas
        return pandas.read_csv(file_path, index_col=0)
    else:
        raise NotImplementedError
