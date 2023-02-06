from typing import Dict
from pandas import DataFrame
from sklearn.metrics import mean_squared_error


def weighted_mean_squared_error(
        eval_df: DataFrame,
        builtin_metrics: Dict[str, float],  # pylint: disable=unused-argument
) -> float:
    return mean_squared_error(
        eval_df["prediction"],
        eval_df["target"],
        sample_weight=1 / eval_df["prediction"].values,
    )
