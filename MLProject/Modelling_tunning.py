import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn

try:
    import dagshub
except ImportError:
    dagshub = None


def plot_confusion(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()


def main(args):
    # initialize dagshub 
    if args.use_dagshub and dagshub:
        dagshub.init(
            repo_owner=os.environ.get("DAGSHUB_REPO_OWNER"),
            repo_name=os.environ.get("DAGSHUB_REPO_NAME"),
            mlflow=True
        )

    # set mlflow tracking uri
    mlflow.set_tracking_uri(
        args.mlflow_tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )

    # set experiment
    mlflow.set_experiment("Titanic-Advanced-Tuning")

    # autolog
    mlflow.sklearn.autolog(log_models=True)

    # load data
    X_train = pd.read_csv("processed_data/X_train.csv")
    X_test = pd.read_csv("processed_data/X_test.csv")
    y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
    y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train = X_train[num_cols]
    X_test = X_test[num_cols]

    # model + hyperparameter tuning
    model = RandomForestClassifier(random_state=42)

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        random_state=42,
        n_jobs=1
    )

    # mlflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("final_accuracy", acc)

        plot_confusion(y_test, y_pred)
        mlflow.log_artifact("confusion_matrix.png")

        fi = pd.DataFrame({
            "feature": X_train.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)

        fi.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        with open("run_info.txt", "w") as f:
            f.write(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--use-dagshub", action="store_true")
    args = parser.parse_args()
    main(args)