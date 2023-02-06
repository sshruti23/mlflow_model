import logging

import pandas
from pandas import DataFrame

_logger = logging.getLogger(__name__)


def load_file_as_dataframe(file_path: str, file_format: str) -> DataFrame:
    if file_format == "csv":
        csv = pandas.read_csv(file_path, sep=",")
        print(csv["to_predict"])
        print(type(csv["to_predict"][0]))
        return csv
    else:
        raise NotImplementedError
