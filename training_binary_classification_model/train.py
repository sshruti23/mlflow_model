
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow.sklearn


def acquire_training_data(data_path):
    df = pd.read_csv(data_path)
    return df


def create_and_log_experiment(X_train, y_train, n_estimators):
    mlflow.set_experiment(experiment_id="196699694392376")
    for n_est in n_estimators:
        with mlflow.start_run(run_name=f"stock_estimator_{n_est}") as run:
            mlflow.sklearn.autolog()
            clf = RandomForestClassifier(
                bootstrap=True,
                criterion='gini',
                min_samples_split=2,
                min_weight_fraction_leaf=0.1,
                n_estimators=n_est,
                random_state=4284,
                verbose=0)

            clf.fit(X_train, y_train)


if __name__ == "__main__":
    train_data_path = '/dbfs/data/train_data.csv'
    train_data = acquire_training_data(train_data_path)
    y_train = train_data['to_predict']
    X_train = train_data.drop('to_predict', axis=1)
    create_and_log_experiment(X_train, y_train, n_estimators=[50, 100, 200, 500, 1000])



