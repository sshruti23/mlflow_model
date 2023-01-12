from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier


def estimator_fn(estimator_params: Dict[str, Any] = None):
    if estimator_params is None:
        estimator_params = {}
    return RandomForestClassifier(
        bootstrap=True,
        criterion="gini",
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=50,
        random_state=4284,
        verbose=0
    )
