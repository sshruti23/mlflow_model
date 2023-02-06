import logging

import pandas
from pandas import DataFrame

_logger = logging.getLogger(__name__)


def load_file_as_dataframe(file_path: str, file_format: str) -> DataFrame:
    if file_format == "csv":
        return pandas.read_csv(file_path, sep=",")
    else:
        raise NotImplementedError
