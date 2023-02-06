from typing import Dict, Any


def estimator_fn(estimator_params: Dict[str, Any] = None):
    if estimator_params is None:
        estimator_params = {}
    from sklearn.linear_model import SGDRegressor

    return SGDRegressor(random_state=42, **estimator_params)
